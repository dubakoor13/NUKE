"""Conservative axial extension of the resolved 2D NERVA fuel cut-cell model.

Extrudes the exact 19-channel Stage53 transverse fuel volumes into arbitrary
axial slabs, adds symmetric axial solid conduction and *directed* upstream-
to-downstream enthalpy transport in all 19 coolant passages. Fuel/H2 heat
exchange is implicitly coupled in one sparse linear solve.

Inputs are user-specified constant-within-step thermophysical properties.
This is a numerically testable engineering surrogate, NOT historical NERVA
power, source-normalized OpenMC heat deposition, a real-fluid equation of
state, or experimentally qualified fuel/hydrogen temperature prediction.
No neutronic design parameters are changed here.

Index conventions: fuel[z, transverse_cut_cell]; hydrogen[z, channel].
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags, kron
from scipy.sparse.linalg import spsolve

from .fuel_cutcell_conduction_2d import CutCellFuel2D


@dataclass(frozen=True)
class AxialCutCellModel:
    transverse: CutCellFuel2D
    axial_edges_m: np.ndarray
    solid_volume_m3: np.ndarray
    solid_storage_j_k: np.ndarray
    hydrogen_storage_j_k: np.ndarray
    axial_solid_conductivity_w_m_k: np.ndarray
    hydrogen_mass_flow_kg_s: np.ndarray
    hydrogen_flow_cp_j_kg_k: np.ndarray
    axial_solid_interface_w_k: np.ndarray
    solid_diffusion_w_k: csr_matrix
    fuel_hydrogen_exchange_w_k: csr_matrix
    hydrogen_upwind_w_k: csr_matrix

    @property
    def axial_zones(self) -> int:
        return len(self.axial_edges_m)-1

    @property
    def fuel_unknowns(self) -> int:
        return self.axial_zones*self.transverse.n_fuel

    @property
    def hydrogen_unknowns(self) -> int:
        return self.axial_zones*19


@dataclass(frozen=True)
class AxialCutCellResult:
    fuel_temperature_k: np.ndarray
    hydrogen_temperature_k: np.ndarray
    fuel_stored_energy_change_j: float
    hydrogen_stored_energy_change_j: float
    coolant_outlet_net_enthalpy_j: float
    deposited_energy_j: float
    net_fuel_to_hydrogen_heat_j: float
    conservation_residual_j: float
    relative_conservation_residual: float
    axial_solid_interface_heat_w: np.ndarray


def _array(name: str, value: object, shape: tuple[int, ...],
           *, allow_zero: bool=False) -> np.ndarray:
    a=np.asarray(value,dtype=float)
    if a.ndim==0:
        a=np.full(shape,float(a))
    if a.shape!=shape or not np.all(np.isfinite(a)) or (
        np.any(a<0) if allow_zero else np.any(a<=0)
    ):
        raise ValueError(
            f"{name} must be finite {'nonnegative' if allow_zero else 'positive'} "
            f"scalar or shape {shape}"
        )
    return a


def _power(name: str,value: object,shape: tuple[int, ...]) -> np.ndarray:
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite array of shape {shape}")
    return a


def build_axial_cutcell_model(
    transverse: CutCellFuel2D,
    *,
    axial_edges_m: object,
    solid_axial_conductivity_w_m_k: object,
    hydrogen_mass_kg: object,
    hydrogen_cp_j_kg_k: object,
    hydrogen_mass_flow_kg_s: object,
    hydrogen_flow_cp_j_kg_k: object,
) -> AxialCutCellModel:
    """Construct the linear conservative cross-section/axial operators.

    Transverse conductances and wall resistances from Stage53 are scaled
    with local axial slab height. Axial fluxes pass through the *actual*
    fuel-only cut-cell cross-sectional areas, with half-cell series
    resistances for unequal axial thicknesses and conductivities.

    H2 advection is first-order upwind (positive flow toward increasing z);
    no fluid momentum, pressure drop, or temperature dependence is inferred.
    """
    z=np.asarray(axial_edges_m,dtype=float)
    if z.ndim!=1 or len(z)<2 or len(z)>257 or not np.all(np.isfinite(z)):
        raise ValueError("axial_edges_m must be a finite 1D array with 2..257 edges")
    dz=np.diff(z)
    if np.any(dz<=0) or abs(float(z[0]))>1e-12 or (
        abs(float(z[-1])-transverse.axial_length_m)>1e-11
    ):
        raise ValueError("axial edges must strictly partition the original fuel length")
    nz=len(dz)
    nf=transverse.n_fuel
    axial_k=_array("axial fuel conductivity",
                   solid_axial_conductivity_w_m_k,(nz,nf))
    h2mass=_array("hydrogen zone mass",hydrogen_mass_kg,(nz,19))
    h2cp=_array("hydrogen heat capacity",hydrogen_cp_j_kg_k,(nz,19))
    flow=_array("hydrogen mass flow",hydrogen_mass_flow_kg_s,(19,),
                allow_zero=True)
    adv_cp=_array("hydrogen transport heat capacity",
                  hydrogen_flow_cp_j_kg_k,(19,))
    heatflow=flow*adv_cp

    cross_area=transverse.fuel_areas_cm2*1e-4
    volumes=dz[:,None]*cross_area[None,:]
    # Re-use exact Stage53 heat capacity per unit length; all solid
    # transverse cells preserve material-specific rho*c within a slab.
    storage=(transverse.fuel_heat_capacity_j_k[None,:]
             *dz[:,None]/transverse.axial_length_m)
    h2storage=h2mass*h2cp
    if not np.isclose(float(np.sum(volumes)),
                      float(np.sum(transverse.fuel_volumes_m3)),
                      rtol=0.,atol=1e-15):
        raise ArithmeticError("axial extrusion lost physical fuel material volume")

    scale=dz/transverse.axial_length_m
    slab_scaling=diags(scale,format="csr")
    lateral=kron(slab_scaling,transverse.lateral_laplacian_w_k,format="csr")
    wall=kron(slab_scaling,transverse.fuel_h2_conductance_w_k,format="csr")

    # Axial face coupling of identical physical cut-cell footprints. The
    # cross-section is not homogenized into a full hex or coated-bore disk.
    axial_face=np.zeros((max(nz-1,0),nf))
    rows=[]
    cols=[]
    values=[]
    diagonal=np.zeros(nz*nf)
    for zz in range(nz-1):
        res=dz[zz]/(2*axial_k[zz,:])+dz[zz+1]/(2*axial_k[zz+1,:])
        conductance=cross_area/res
        axial_face[zz,:]=conductance
        idx=np.arange(nf,dtype=int)
        a=zz*nf+idx
        b=(zz+1)*nf+idx
        diagonal[a]+=conductance
        diagonal[b]+=conductance
        rows.extend(a.tolist())
        cols.extend(b.tolist())
        values.extend((-conductance).tolist())
        rows.extend(b.tolist())
        cols.extend(a.tolist())
        values.extend((-conductance).tolist())
    for i,g in enumerate(diagonal):
        if g!=0:
            rows.append(i)
            cols.append(i)
            values.append(float(g))
    axial=coo_matrix((values,(rows,cols)),shape=(nz*nf,nz*nf)).tocsr()
    solid=(lateral+axial).tocsr()
    solid.sum_duplicates()
    if not np.all(np.isfinite(solid.data)) or not np.all(np.isfinite(wall.data)):
        raise ArithmeticError("invalid nonfinite thermal conductances")
    if np.max(np.abs(np.asarray(solid.sum(axis=1))))>3e-9:
        raise ArithmeticError("solid diffusion does not conserve constant temperature")

    # The coolant matrix has +m_dot*cp on every slab and -m_dot*cp to its
    # immediately upstream neighbor. Inlet is supplied on the first slab
    # RHS. Interior face transport cancels *exactly* in the global budget.
    ar=[]
    ac=[]
    av=[]
    for zz in range(nz):
        for ch in range(19):
            g=float(heatflow[ch])
            if g==0.:
                continue
            current=zz*19+ch
            ar.append(current)
            ac.append(current)
            av.append(g)
            if zz>0:
                ar.append(current)
                ac.append((zz-1)*19+ch)
                av.append(-g)
    upstream=coo_matrix((av,(ar,ac)),shape=(nz*19,nz*19)).tocsr()
    return AxialCutCellModel(
        transverse=transverse,
        axial_edges_m=z.copy(),
        solid_volume_m3=volumes,
        solid_storage_j_k=storage,
        hydrogen_storage_j_k=h2storage,
        axial_solid_conductivity_w_m_k=axial_k,
        hydrogen_mass_flow_kg_s=flow,
        hydrogen_flow_cp_j_kg_k=adv_cp,
        axial_solid_interface_w_k=axial_face,
        solid_diffusion_w_k=solid,
        fuel_hydrogen_exchange_w_k=wall,
        hydrogen_upwind_w_k=upstream,
    )


def advance_axial_cutcell(
    model: AxialCutCellModel,
    old_fuel_temperature_k: object,
    old_hydrogen_temperature_k: object,
    *,
    dt_s: float,
    fuel_source_w: object,
    hydrogen_source_w: object,
    inlet_hydrogen_temperature_k: object,
) -> AxialCutCellResult:
    """Single sparse backward-Euler 3D fuel / 1D per-channel hydrogen solve.

    For each fuel control volume:
       C_f*(T-T_old)/dt + diffusion(T) + wall(T-T_h) = Q_f
    For each H2 control volume:
       C_h*(T-T_old)/dt + wall(T_h-T_f)
         + m_dot cp*(T_h-T_upstream) = Q_h

    Upstream boundary is specified temperature at the *physical inlet*.
    Hydrogen advection is directional; solid diffusion remains symmetric.
    """
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("dt_s must be finite positive seconds")
    nz=model.axial_zones
    nf=model.transverse.n_fuel
    fshape=(nz,nf)
    hshape=(nz,19)
    oldf=_array("old solid temperature",old_fuel_temperature_k,fshape)
    oldh=_array("old hydrogen temperature",old_hydrogen_temperature_k,hshape)
    qf=_power("fuel source",fuel_source_w,fshape)
    qh=_power("hydrogen source",hydrogen_source_w,hshape)
    tin=_array("hydrogen inlet temperature",inlet_hydrogen_temperature_k,(19,))
    cf=model.solid_storage_j_k.ravel()
    ch=model.hydrogen_storage_j_k.ravel()
    W=model.fuel_hydrogen_exchange_w_k
    wf=np.asarray(W.sum(axis=1)).ravel()
    wh=np.asarray(W.sum(axis=0)).ravel()
    heatflow=model.hydrogen_mass_flow_kg_s*model.hydrogen_flow_cp_j_kg_k
    a=bmat([
        [model.solid_diffusion_w_k+diags(cf/dt_s+wf), -W],
        [-W.T,diags(ch/dt_s+wh)+model.hydrogen_upwind_w_k],
    ],format="csc")
    inlet_rhs=np.zeros(hshape,dtype=float)
    inlet_rhs[0,:]=heatflow*tin
    rhs=np.concatenate((
        cf*oldf.ravel()/dt_s+qf.ravel(),
        ch*oldh.ravel()/dt_s+qh.ravel()+inlet_rhs.ravel(),
    ))
    solved=np.asarray(spsolve(a,rhs),dtype=float)
    if (solved.shape!=(nz*nf+nz*19,) or
            not np.all(np.isfinite(solved)) or np.any(solved<=0)):
        raise ArithmeticError("invalid axial fuel/hydrogen implicit solution")
    newf=solved[:nz*nf].reshape(fshape)
    newh=solved[nz*nf:].reshape(hshape)
    duf=float(np.dot(cf,newf.ravel()-oldf.ravel()))
    duh=float(np.dot(ch,newh.ravel()-oldh.ravel()))
    exported=float(dt_s*np.dot(heatflow,newh[-1,:]-tin))
    deposited=float(dt_s*(qf.sum()+qh.sum()))
    exchange=float(dt_s*(
        np.dot(wf,newf.ravel())-np.dot(wh,newh.ravel())))
    residual=duf+duh+exported-deposited
    scale=max(1.,abs(duf),abs(duh),abs(exported),abs(deposited))
    relative=abs(residual)/scale
    # An exactly steady state or mutually cancelling axial fluxes may have
    # *zero* net deposition/export while the solved system stores 1e5 J.
    # Forming C*(T_new - T_old) then loses ~eps*C*T absolute precision.
    # Preserve the original stringent source-relative tolerance AND allow
    # a separately reported floating-point roundoff floor proportional to
    # the state-energy terms. This is not an energy-balance adjustment.
    u_roundoff=(float(np.dot(cf,np.maximum(np.abs(newf.ravel()),
                                         np.abs(oldf.ravel()))))
                +float(np.dot(ch,np.maximum(np.abs(newh.ravel()),
                                         np.abs(oldh.ravel()))))
                +float(dt_s*np.dot(heatflow,
                                 np.maximum(np.abs(newh[-1,:]),np.abs(tin)))))
    allowed=max(5e-9*scale,1e-11*u_roundoff)
    if abs(residual)>allowed:
        raise ArithmeticError(
            f"axial fuel/H2 first law failed: residual={residual:.12g} J, "
            f"allowed={allowed:.12g} J, scale={scale:.12g} J, "
            f"dUf={duf:.12g}, dUh={duh:.12g}, "
            f"outlet={exported:.12g}, deposited={deposited:.12g}")
    interface=model.axial_solid_interface_w_k*(
        newf[:-1,:]-newf[1:,:])
    return AxialCutCellResult(
        fuel_temperature_k=newf,
        hydrogen_temperature_k=newh,
        fuel_stored_energy_change_j=duf,
        hydrogen_stored_energy_change_j=duh,
        coolant_outlet_net_enthalpy_j=exported,
        deposited_energy_j=deposited,
        net_fuel_to_hydrogen_heat_j=exchange,
        conservation_residual_j=residual,
        relative_conservation_residual=relative,
        axial_solid_interface_heat_w=interface,
    )
