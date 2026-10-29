"""Nonlinear enthalpy-conservative NERVA-derived fuel/H2 thermal transport.

Extends the existing Stage54 2D cut-cell / axial finite-volume discretization
without changing OpenMC geometry, materials, criticality or reactor controls.
Integrates tabulated c_p(T) exactly as piecewise-linear *specific enthalpy*,
and recomputes thermal resistances at every nonlinear iterate. Hydrogen
advection transports h(T), not an artificial fixed cp * delta T.

Prescribed density and H2 zone mass are held constant within a timestep.
Pressure work, variable mass inventory, fluid momentum, radiation and
neutron-power normalization are deliberately NOT modeled. The tables must
be supplied by the caller: no asserted NASA hot-material properties exist
in this module. Temperatures outside their measured/qualified intervals
are rejected, not silently extrapolated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

from .config import NervaConfig
from .fuel_cutcell_axial_transport import AxialCutCellModel


@dataclass(frozen=True)
class TemperatureTable:
    """Positive (or nonnegative) piecewise-linear property against kelvin.

    integral(T) integrates its piecewise-linear values exactly from the
    first knot, so a heat capacity table gives specific enthalpy [J/kg].
    No extrapolation is supported.
    """
    temperature_k: object
    values: object
    allow_zero: bool = False

    def __post_init__(self) -> None:
        t=np.asarray(self.temperature_k,dtype=float)
        v=np.asarray(self.values,dtype=float)
        if (t.ndim!=1 or v.ndim!=1 or len(t)<2 or
                t.shape!=v.shape or not np.all(np.isfinite(t)) or
                not np.all(np.isfinite(v)) or np.any(np.diff(t)<=0) or
                t[0]<=0 or (np.any(v<0) if self.allow_zero else np.any(v<=0))):
            raise ValueError("property table requires increasing positive K and finite valid values")
        object.__setattr__(self,"temperature_k",t.copy())
        object.__setattr__(self,"values",v.copy())

    def _indices(self, temperatures: object) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        a=np.asarray(temperatures,dtype=float)
        if not np.all(np.isfinite(a)) or np.any(a<self.temperature_k[0]) or (
            np.any(a>self.temperature_k[-1])
        ):
            raise ValueError(
                f"temperature outside qualified table range "
                f"[{self.temperature_k[0]}, {self.temperature_k[-1]}] K"
            )
        index=np.clip(np.searchsorted(self.temperature_k,a,side="right")-1,
                      0,len(self.temperature_k)-2)
        delta=a-self.temperature_k[index]
        slopes=np.diff(self.values)/np.diff(self.temperature_k)
        return index,delta,slopes

    def at(self, temperatures: object) -> np.ndarray:
        idx,dt,slopes=self._indices(temperatures)
        return self.values[idx]+slopes[idx]*dt

    def integral(self, temperatures: object) -> np.ndarray:
        """Exact integral of the piecewise-linear property dT from Tmin."""
        idx,dt,slopes=self._indices(temperatures)
        tdiff=np.diff(self.temperature_k)
        increments=.5*(self.values[:-1]+self.values[1:])*tdiff
        prefix=np.concatenate(([0.],np.cumsum(increments)))
        return prefix[idx]+self.values[idx]*dt+.5*slopes[idx]*dt**2


@dataclass(frozen=True)
class NonlinearThermalProperties:
    fuel_cp_j_kg_k: TemperatureTable
    fuel_k_w_m_k: TemperatureTable
    coating_k_w_m_k: TemperatureTable
    hydrogen_cp_j_kg_k: TemperatureTable
    hydrogen_film_w_m2_k: TemperatureTable


@dataclass(frozen=True)
class NonlinearAxialStep:
    fuel_temperature_k: np.ndarray
    hydrogen_temperature_k: np.ndarray
    iterations: int
    final_max_residual_w: float
    fuel_energy_change_j: float
    hydrogen_energy_change_j: float
    hydrogen_net_export_j: float
    deposited_energy_j: float
    first_law_residual_j: float
    relative_first_law_residual: float
    fuel_to_hydrogen_transfer_j: float


def _typed(name: str, value: object, shape: tuple[int,...],
           *, positive: bool=True) -> np.ndarray:
    a=np.asarray(value,dtype=float)
    if a.ndim==0:
        a=np.full(shape,float(a))
    if (a.shape!=shape or not np.all(np.isfinite(a)) or
            (np.any(a<=0) if positive else np.any(a<0))):
        raise ValueError(f"{name} requires finite {'positive' if positive else 'nonnegative'} values with shape {shape}")
    return a


def _forcing(name: str,value: object,shape: tuple[int,...]) -> np.ndarray:
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite watt array with shape {shape}")
    return a


class _NonlinearSystem:
    """Cached exact geometry; iteration-dependent sparse coefficients."""

    def __init__(
        self, model: AxialCutCellModel, config: NervaConfig,
        properties: NonlinearThermalProperties,
        solid_density_kg_m3: np.ndarray, hydrogen_mass_kg: np.ndarray,
        old_f: np.ndarray, old_h: np.ndarray, inlet: np.ndarray,
        qf: np.ndarray, qh: np.ndarray, dt_s: float,
    ) -> None:
        self.model=model
        self.p=properties
        self.nf=model.transverse.n_fuel
        self.nz=model.axial_zones
        self.nsolid=model.fuel_unknowns
        self.nh2=model.hydrogen_unknowns
        self.mf=solid_density_kg_m3*model.solid_volume_m3
        self.mh=hydrogen_mass_kg
        self.old_f=old_f
        self.old_h=old_h
        self.inlet=inlet
        self.qf=qf
        self.qh=qh
        self.dt=dt_s
        self.flow=model.hydrogen_mass_flow_kg_s
        self.dz=np.diff(model.axial_edges_m)
        self.cross_area=model.transverse.fuel_areas_cm2*1e-4
        base=model.transverse
        self.i=base.lateral_face_pairs[:,0]
        self.j=base.lateral_face_pairs[:,1]
        kr=base.fuel_conductivity_w_m_k
        kh=2*kr[self.i]*kr[self.j]/(kr[self.i]+kr[self.j])
        self.lateral_geometry=base.lateral_face_conductance_w_k/kh
        coo=base.fuel_zrc_contact_m2.tocoo()
        self.wall_cell=coo.row.astype(int)
        self.wall_channel=coo.col.astype(int)
        self.wall_area_m2=coo.data
        self.cell_center_wall_distance_m=0.5*min(
            float(np.min(np.diff(base.x_edges_cm))),
            float(np.min(np.diff(base.y_edges_cm))),
        )/100
        self.rcoat=config.coated_channel_radius_cm/100
        self.rbore=config.channel_radius_cm/100
        if (self.rbore<=0 or self.rcoat<=self.rbore or
                abs(config.active_length_cm/100-base.axial_length_m)>1e-12):
            raise ValueError("nonlinear coating geometry inconsistent with resolved axial fuel")
        self.log_coat=math.log(self.rcoat/self.rbore)
        self.h_inlet=properties.hydrogen_cp_j_kg_k.integral(inlet)
        self.h_oldf=properties.fuel_cp_j_kg_k.integral(old_f)
        self.h_oldh=properties.hydrogen_cp_j_kg_k.integral(old_h)

    def _operators(self, tf: np.ndarray, th: np.ndarray
                   ) -> tuple[csr_matrix,csr_matrix]:
        n=self.nsolid
        nz=self.nz
        nf=self.nf
        k=self.p.fuel_k_w_m_k.at(tf)
        rows=[]
        cols=[]
        data=[]
        diagonal=np.zeros(n)
        # Transverse two-point heat flux: harmonic conductance evaluated
        # at current fuel temperatures and true Stage53 cut-face geometry.
        for iz in range(nz):
            ke=2*k[iz,self.i]*k[iz,self.j]/(
                k[iz,self.i]+k[iz,self.j])
            g=self.lateral_geometry*ke*(
                self.dz[iz]/self.model.transverse.axial_length_m)
            a=iz*nf+self.i
            b=iz*nf+self.j
            np.add.at(diagonal,a,g)
            np.add.at(diagonal,b,g)
            rows.extend(a.tolist()+b.tolist())
            cols.extend(b.tolist()+a.tolist())
            data.extend((-g).tolist()+(-g).tolist())
        # Axial conduction through each exact fuel-only cut area.
        for iz in range(nz-1):
            g=self.cross_area/(
                self.dz[iz]/(2*k[iz,:])+
                self.dz[iz+1]/(2*k[iz+1,:]))
            idx=np.arange(nf,dtype=int)
            a=iz*nf+idx
            b=(iz+1)*nf+idx
            np.add.at(diagonal,a,g)
            np.add.at(diagonal,b,g)
            rows.extend(a.tolist()+b.tolist())
            cols.extend(b.tolist()+a.tolist())
            data.extend((-g).tolist()+(-g).tolist())
        diag_idx=np.arange(n)
        rows.extend(diag_idx.tolist())
        cols.extend(diag_idx.tolist())
        data.extend(diagonal.tolist())
        lap=coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr()
        lap.sum_duplicates()
        # Stage53 analytical fuel/ZrC arc area, slab-scaled. At each
        # arc, resistances in series: half-cell fuel, cylindrical ZrC,
        # and actual H2 bore film. Conductivity/film may depend on T.
        rows=[]
        cols=[]
        vals=[]
        cell=self.wall_cell
        channel=self.wall_channel
        for iz in range(nz):
            tfwall=tf[iz,cell]
            thwall=th[iz,channel]
            kz=self.p.coating_k_w_m_k.at(.5*(tfwall+thwall))
            film=self.p.hydrogen_film_w_m2_k.at(thwall)
            denom=(self.cell_center_wall_distance_m/k[iz,cell]+
                   self.rcoat*self.log_coat/kz)
            conductance=np.zeros(len(denom))
            valid=film>0
            if np.any(valid):
                denom[valid]+=self.rcoat/(self.rbore*film[valid])
                arc_area=self.wall_area_m2[valid]*(
                    self.dz[iz]/self.model.transverse.axial_length_m)
                conductance[valid]=arc_area/denom[valid]
            rows.extend((iz*nf+cell).tolist())
            cols.extend((iz*19+channel).tolist())
            vals.extend(conductance.tolist())
        wall=coo_matrix((vals,(rows,cols)),
                        shape=(n,self.nh2)).tocsr()
        wall.sum_duplicates()
        return lap,wall

    def evaluate(self, x: np.ndarray, *, jacobian: bool
                 ) -> tuple[np.ndarray,csr_matrix|None,csr_matrix]:
        nz=self.nz
        nf=self.nf
        tf=x[:self.nsolid].reshape((nz,nf))
        th=x[self.nsolid:].reshape((nz,19))
        solid_cp=self.p.fuel_cp_j_kg_k.at(tf)
        h2_cp=self.p.hydrogen_cp_j_kg_k.at(th)
        ent_f=self.p.fuel_cp_j_kg_k.integral(tf)
        ent_h=self.p.hydrogen_cp_j_kg_k.integral(th)
        lap,wall=self._operators(tf,th)
        f=tf.ravel()
        h=th.ravel()
        wf=np.asarray(wall.sum(axis=1)).ravel()
        wh=np.asarray(wall.sum(axis=0)).ravel()
        rf=(self.mf*(ent_f-self.h_oldf)/self.dt).ravel()
        rf+=lap@f+wf*f-wall@h-self.qf.ravel()
        transport=np.zeros((nz,19))
        transport[0,:]=self.flow*(ent_h[0,:]-self.h_inlet)
        if nz>1:
            transport[1:,:]=self.flow[None,:]*(
                ent_h[1:,:]-ent_h[:-1,:])
        rh=(self.mh*(ent_h-self.h_oldh)/self.dt+transport).ravel()
        rh+=wh*h-wall.T@f-self.qh.ravel()
        residual=np.concatenate((rf,rh))
        if not np.all(np.isfinite(residual)):
            raise ArithmeticError("nonlinear heat equation residual nonfinite")
        if not jacobian:
            return residual,None,wall

        # Inexact Newton: all local enthalpy derivatives are analytic and
        # exact. Temperature derivatives of conductivity/film are lagged;
        # line search enforces strict nonlinear residual reduction.
        js=(lap+diags(
            (self.mf*solid_cp/self.dt).ravel()+wf,
            format="csr")).tocsr()
        jhdiag=(self.mh*h2_cp/self.dt+
                self.flow[None,:]*h2_cp).ravel()+wh
        hrows=np.arange(nz*19,dtype=int)
        hcols=hrows.copy()
        hvals=jhdiag.tolist()
        rows=hrows.tolist()
        cols=hcols.tolist()
        for iz in range(1,nz):
            upstream=np.arange(19,dtype=int)+(iz-1)*19
            current=np.arange(19,dtype=int)+iz*19
            negative=-self.flow*h2_cp[iz-1,:]
            rows.extend(current.tolist())
            cols.extend(upstream.tolist())
            hvals.extend(negative.tolist())
        jh=coo_matrix((hvals,(rows,cols)),
                       shape=(nz*19,nz*19)).tocsr()
        jac=bmat([[js,-wall],[-wall.T,jh]],format="csr")
        return residual,jac,wall


def advance_nonlinear_axial_cutcell(
    model: AxialCutCellModel,
    config: NervaConfig,
    properties: NonlinearThermalProperties,
    old_fuel_temperature_k: object,
    old_hydrogen_temperature_k: object,
    *,
    dt_s: float,
    fuel_source_w: object,
    hydrogen_source_w: object,
    solid_density_kg_m3: object,
    hydrogen_mass_kg: object,
    inlet_hydrogen_temperature_k: object,
    max_iterations: int = 24,
    residual_tolerance_w: float = 2e-8,
) -> NonlinearAxialStep:
    """Implicit enthalpy-conservative nonlinear 2D+axial fuel/H2 step.

    All source arrays are prescribed W (not independently normalized OpenMC
    heating scores). Newton iteration uses analytic enthalpy derivatives
    and lagged material/wall conductance derivatives with backtracking.
    """
    if not isinstance(max_iterations,int) or not 1<=max_iterations<=100:
        raise ValueError("max_iterations must be 1..100")
    if (not math.isfinite(dt_s) or dt_s<=0 or
            not math.isfinite(residual_tolerance_w) or
            not 0<residual_tolerance_w<=0.01):
        raise ValueError("dt and residual tolerance must be finite positive")
    nz=model.axial_zones
    nf=model.transverse.n_fuel
    tf=_typed("initial fuel T",old_fuel_temperature_k,(nz,nf))
    th=_typed("initial hydrogen T",old_hydrogen_temperature_k,(nz,19))
    rho=_typed("solid density",solid_density_kg_m3,(nz,nf))
    mass=_typed("H2 zone mass",hydrogen_mass_kg,(nz,19))
    inlet=_typed("hydrogen inlet T",inlet_hydrogen_temperature_k,(19,))
    qf=_forcing("fuel input W",fuel_source_w,(nz,nf))
    qh=_forcing("hydrogen input W",hydrogen_source_w,(nz,19))
    system=_NonlinearSystem(model,config,properties,rho,mass,tf,th,inlet,qf,qh,dt_s)
    x=np.concatenate((tf.ravel(),th.ravel()))
    residual,jac,_=system.evaluate(x,jacobian=True)
    residual_size=float(np.max(np.abs(residual)))
    iters=0
    for iteration in range(max_iterations):
        if residual_size<=residual_tolerance_w:
            break
        delta=np.asarray(spsolve(jac.tocsc(),-residual),dtype=float)
        if delta.shape!=x.shape or not np.all(np.isfinite(delta)):
            raise ArithmeticError("nonlinear thermal Jacobian solve failed")
        accepted=False
        step=1.
        for _ in range(25):
            trial=x+step*delta
            if np.any(trial<=0):
                step*=0.5
                continue
            try:
                rtry,_,_=system.evaluate(trial,jacobian=False)
            except ValueError:
                step*=0.5
                continue
            score=float(np.max(np.abs(rtry)))
            if score < residual_size*(1-1e-5*step) or score<=residual_tolerance_w:
                x=trial
                residual_size=score
                accepted=True
                break
            step*=.5
        if not accepted:
            raise ArithmeticError(
                f"nonlinear fuel/H2 iteration line search failed, "
                f"max residual={residual_size:.6g} W"
            )
        iters=iteration+1
        if residual_size<=residual_tolerance_w:
            break
        residual,jac,_=system.evaluate(x,jacobian=True)
        residual_size=float(np.max(np.abs(residual)))
    if residual_size>residual_tolerance_w:
        raise ArithmeticError(
            f"nonlinear fuel/H2 system not converged after "
            f"{max_iterations} iterations: {residual_size:.6g} W"
        )

    f=x[:system.nsolid].reshape((nz,nf))
    h=x[system.nsolid:].reshape((nz,19))
    ef=properties.fuel_cp_j_kg_k.integral(f)
    eh=properties.hydrogen_cp_j_kg_k.integral(h)
    duf=float(np.sum(system.mf*(ef-system.h_oldf)))
    duh=float(np.sum(system.mh*(eh-system.h_oldh)))
    export=float(dt_s*np.dot(
        system.flow,eh[-1,:]-system.h_inlet))
    qtotal=float(dt_s*(qf.sum()+qh.sum()))
    residual_energy=duf+duh+export-qtotal
    scale=max(1.,abs(duf),abs(duh),abs(export),abs(qtotal))
    # Explicit state-energy round-off bound, as in Stage54. This does not
    # adjust any reported energy or invent deposited reactor watts.
    roundoff=(
        float(np.sum(system.mf*np.maximum(
            np.abs(ef),np.abs(system.h_oldf))))
        +float(np.sum(system.mh*np.maximum(
            np.abs(eh),np.abs(system.h_oldh))))
        +dt_s*float(np.dot(system.flow,
            np.maximum(np.abs(eh[-1,:]),np.abs(system.h_inlet)))))
    allowed=max(2e-9*scale,
                2e-11*roundoff,
                2*residual_tolerance_w*dt_s*(system.nsolid+system.nh2))
    if abs(residual_energy)>allowed:
        raise ArithmeticError(
            f"nonlinear enthalpy conservation failed: "
            f"residual={residual_energy:.9g} J, allowed={allowed:.9g} J"
        )
    _,_,wfinal=system.evaluate(x,jacobian=False)
    wf=np.asarray(wfinal.sum(axis=1)).ravel()
    wh=np.asarray(wfinal.sum(axis=0)).ravel()
    wall_energy=dt_s*(
        float(np.dot(wf,f.ravel()))-float(np.dot(wh,h.ravel())))
    return NonlinearAxialStep(
        fuel_temperature_k=f,
        hydrogen_temperature_k=h,
        iterations=iters,
        final_max_residual_w=residual_size,
        fuel_energy_change_j=duf,
        hydrogen_energy_change_j=duh,
        hydrogen_net_export_j=export,
        deposited_energy_j=qtotal,
        first_law_residual_j=residual_energy,
        relative_first_law_residual=abs(residual_energy)/scale,
        fuel_to_hydrogen_transfer_j=wall_energy,
    )
