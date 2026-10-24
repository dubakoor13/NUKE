"""Implicit radial-axial fuel-sector conduction coupled to axial hydrogen enthalpy.

General-purpose finite-volume reactor *thermal* physics; reactor power is a
caller-supplied input (never inferred from k-effective). The 19 irregular
Voronoi fuel sectors are represented by equal-area annuli, with explicit ZrC
coating and fuel-matrix radial cells. This is NOT the true 3-D hexagonal
geometry, does not calculate neutron heating, and has no reactor controls.

Unknowns in each solve:
    T_s[z, r]  : fuel/coating finite-volume temperatures [K]
    T_g[z]     : mixed channel H2 control-volume temperatures [K]

Physics:
    rho * d(u)/dt = div(k grad(T)) + nuclear_volumetric_source
    m_g * dh/dt + mdot * (h_out - h_in)
        = Q_conv_from_solid + Q_direct_H2
with implicit upwind gas advection, cylindrical-face radial conduction,
axial solid conduction, conservative center-to-center interface resistances,
and adiabatic fuel outer radius/axial ends. Convection couples the coated
inner wall to its *actual* gas control volume as a shared flux.

Transient backward Euler uses integral heat capacities/enthalpy; steady
state is solved by dt_s=None. Picard updates nonlinear conductivities and
enthalpy secants while assembling a sparse, monolithic solid+gas matrix.
The result explicitly audits the combined first law, including stored
solid and fluid energy, prescribed nuclear inputs and inlet/outlet enthalpy.

Properties, heat-transfer coefficients, coolant density and power must be
provided independently and are NOT NERVA historical material calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .implicit_hydrogen import hydrogen_specific_enthalpy
from .nasa_hydrogen import T_MIN, T_MAX


@dataclass(frozen=True)
class CoreSolidLaw:
    """Linearized-in-temperature thermal-property law about reference K.

    k(T)=k0*(1+k_slope*(T-T_ref)).
    cp(T)=cp0*(1+cp_slope*(T-T_ref)).
    rho is constant. Slopes are assumptions, not material data.
    """
    density_kg_m3: float
    cp_ref_j_kg_k: float
    conductivity_ref_w_m_k: float
    reference_k: float = 500.0
    cp_slope_per_k: float = 0.0
    conductivity_slope_per_k: float = 0.0

    def validate(self) -> None:
        values=(self.density_kg_m3,self.cp_ref_j_kg_k,
                self.conductivity_ref_w_m_k,self.reference_k)
        if any(not math.isfinite(v) or v<=0 for v in values):
            raise ValueError("solid law requires positive finite properties")
        if not all(math.isfinite(v) for v in
                   (self.cp_slope_per_k,self.conductivity_slope_per_k)):
            raise ValueError("solid property slope must be finite")

    def conductivity(self, t: float) -> float:
        v=self.conductivity_ref_w_m_k*(
            1.0+self.conductivity_slope_per_k*(t-self.reference_k))
        if not math.isfinite(v) or v<=0:
            raise ValueError("solid conductivity outside physical domain")
        return v

    def secant_cp(self, a: float, b: float) -> float:
        v=self.cp_ref_j_kg_k*(
            1.0+0.5*self.cp_slope_per_k*(a+b-2*self.reference_k))
        if not math.isfinite(v) or v<=0:
            raise ValueError("solid heat capacity outside physical domain")
        return v

    def enthalpy_change(self, a: float, b: float) -> float:
        return self.secant_cp(a,b)*(b-a)


@dataclass(frozen=True)
class CoreSectorGrid:
    """Cylindrical FV mesh [axial bin, radial bin], dimensions in metres."""
    z_edges_m: np.ndarray
    r_edges_m: np.ndarray
    n_coating_cells: int

    @classmethod
    def equivalent_annulus(cls, z_edges_m: np.ndarray, *,
                           bore_radius_m: float,
                           coating_thickness_m: float,
                           fuel_sector_area_m2: float,
                           coating_cells: int = 2,
                           fuel_cells: int = 8) -> "CoreSectorGrid":
        if (coating_cells<1 or fuel_cells<1
                or any(not math.isfinite(x) or x<=0 for x in (
                    bore_radius_m,coating_thickness_m,fuel_sector_area_m2))):
            raise ValueError("invalid equivalent-annulus geometry")
        coat=bore_radius_m+coating_thickness_m
        outer=math.sqrt(coat*coat+fuel_sector_area_m2/math.pi)
        r=np.r_[np.linspace(bore_radius_m,coat,coating_cells+1),
                np.linspace(coat,outer,fuel_cells+1)[1:]]
        grid=cls(np.asarray(z_edges_m,dtype=float),r,coating_cells)
        grid.validate()
        return grid

    def validate(self) -> None:
        z=np.asarray(self.z_edges_m,dtype=float)
        r=np.asarray(self.r_edges_m,dtype=float)
        if (z.ndim!=1 or r.ndim!=1 or z.size<2 or r.size<3
                or not np.isfinite(z).all() or not np.isfinite(r).all()
                or np.any(np.diff(z)<=0) or np.any(np.diff(r)<=0)
                or r[0]<=0 or not 1<=self.n_coating_cells<r.size-1):
            raise ValueError("nonphysical radial-axial sector mesh")

    @property
    def shape(self)->tuple[int,int]:
        return (len(self.z_edges_m)-1,len(self.r_edges_m)-1)

    @property
    def volumes_m3(self)->np.ndarray:
        return (math.pi*np.diff(self.r_edges_m**2)[None,:] *
                np.diff(self.z_edges_m)[:,None])


@dataclass(frozen=True)
class CoreSectorBoundary:
    """Advection and convection per axial gas control volume."""
    mass_flow_kg_s: float
    inlet_temperature_k: float
    convection_w_m2_k: np.ndarray
    direct_hydrogen_heating_w: np.ndarray
    gas_mass_kg: np.ndarray
    gas_enthalpy_model: str = "nasa7-h2"
    constant_gas_cp_j_kg_k: float = 14300.0

    def validate(self,nz:int) -> None:
        if (not math.isfinite(self.mass_flow_kg_s)
                or self.mass_flow_kg_s<=0
                or not math.isfinite(self.inlet_temperature_k)
                or self.inlet_temperature_k<=0
                or not math.isfinite(self.constant_gas_cp_j_kg_k)
                or self.constant_gas_cp_j_kg_k<=0):
            raise ValueError("invalid coolant flow/inlet/heat capacity")
        if self.gas_enthalpy_model not in ("nasa7-h2","constant"):
            raise ValueError("invalid hydrogen enthalpy model")
        for label,value,nonnegative in (
            ("convection",self.convection_w_m2_k,False),
            ("direct hydrogen heating",self.direct_hydrogen_heating_w,True),
            ("hydrogen mass",self.gas_mass_kg,True),
        ):
            a=np.asarray(value,dtype=float)
            if a.shape!=(nz,) or not np.isfinite(a).all():
                raise ValueError(label+" must have finite axial-bin values")
            if np.any(a<0) if nonnegative else np.any(a<=0):
                raise ValueError("invalid "+label)
        _gas_enthalpy(self.inlet_temperature_k,self)

@dataclass(frozen=True)
class CoreSectorState:
    solid_temperature_k: np.ndarray
    hydrogen_temperature_k: np.ndarray


@dataclass(frozen=True)
class CoreSectorResult:
    state: CoreSectorState
    coating_inner_wall_temperature_k: np.ndarray
    convected_wall_heating_w: np.ndarray
    solid_nuclear_heating_w: float
    hydrogen_direct_heating_w: float
    solid_energy_change_j: float
    hydrogen_energy_change_j: float
    inlet_enthalpy_w: float
    outlet_enthalpy_w: float
    energy_balance_residual_j: float
    iterations: int
    transient: bool


def _gas_enthalpy(t:float,b:CoreSectorBoundary)->float:
    if not math.isfinite(t) or t<=0:
        raise ValueError("nonphysical coolant temperature")
    if b.gas_enthalpy_model=="nasa7-h2" and not T_MIN<=t<=T_MAX:
        raise ValueError("gas outside NASA-7 H2 temperature domain")
    v=hydrogen_specific_enthalpy(
        float(t),b.constant_gas_cp_j_kg_k,b.gas_enthalpy_model)
    if not math.isfinite(v):
        raise ValueError("nonfinite hydrogen enthalpy")
    return float(v)


def _gas_secant(a:float,c:float,b:CoreSectorBoundary)->float:
    if abs(c-a)>1e-7*max(1.0,abs(a),abs(c)):
        return (_gas_enthalpy(c,b)-_gas_enthalpy(a,b))/(c-a)
    # Stable centered numerical derivative within the valid enthalpy domain.
    delta=max(1e-3,abs(a)*1e-5)
    if b.gas_enthalpy_model=="nasa7-h2":
        lo=max(T_MIN,a-delta);hi=min(T_MAX,a+delta)
    else:
        lo=max(1e-9,a-delta);hi=a+delta
    if hi<=lo:
        raise ValueError("invalid gas property derivative interval")
    return (_gas_enthalpy(hi,b)-_gas_enthalpy(lo,b))/(hi-lo)


def _nuclear_cells(grid:CoreSectorGrid,power:np.ndarray)->np.ndarray:
    """Allocate axial fuel power by fuel volume, or accept exact cell powers."""
    a=np.asarray(power,dtype=float)
    nz,nr=grid.shape
    if a.shape==(nz,nr):
        out=a.copy()
    elif a.shape==(nz,):
        v=grid.volumes_m3.copy()
        v[:,:grid.n_coating_cells]=0.0
        out=a[:,None]*v/np.sum(v,axis=1)[:,None]
    else:
        raise ValueError("solid nuclear power must be [z] or [z,r] W")
    if not np.isfinite(out).all():
        raise ValueError("nonfinite nuclear heat input")
    return out


def solve_core_sector_conjugate(
    grid: CoreSectorGrid,
    state: CoreSectorState,
    boundary: CoreSectorBoundary,
    fuel: CoreSolidLaw,
    coating: CoreSolidLaw,
    nuclear_solid_power_w: np.ndarray,
    *,
    dt_s: float|None = None,
    max_iterations: int = 80,
    tolerance_k: float = 1e-7,
    relaxation: float = 0.75,
    energy_tolerance: float = 1e-7,
    fuel_coating_contact_resistance_m2_k_w: float = 0.0,
)->CoreSectorResult:
    """Implicit 2-D solid / 1-D gas finite-volume conjugate step.

    dt_s=None gives the steady-state solution; otherwise backward Euler.
    Boundary flow and h(z) are prescribed; no momentum/pressure equations,
    chemistry, fuel damage, radiation loss or neutronic power feedback.
    The optional fuel/coating contact resistance is a prescribed thermal
    interface resistance [m2 K/W], not a calibrated NERVA fuel gap model.
    """
    grid.validate()
    fuel.validate()
    coating.validate()
    nz,nr=grid.shape
    boundary.validate(nz)
    if (dt_s is not None and (not math.isfinite(dt_s) or dt_s<=0)
            or max_iterations<2 or not math.isfinite(tolerance_k)
            or tolerance_k<=0 or not 0<relaxation<=1
            or not math.isfinite(energy_tolerance) or energy_tolerance<=0
            or not math.isfinite(fuel_coating_contact_resistance_m2_k_w)
            or fuel_coating_contact_resistance_m2_k_w<0):
        raise ValueError("invalid implicit solver controls")
    s0=np.asarray(state.solid_temperature_k,dtype=float)
    g0=np.asarray(state.hydrogen_temperature_k,dtype=float)
    if (s0.shape!=(nz,nr) or g0.shape!=(nz,)
            or not np.isfinite(s0).all() or not np.isfinite(g0).all()
            or np.min(s0)<=0 or np.min(g0)<=0):
        raise ValueError("initial thermal state does not match mesh")
    for t in g0:
        _gas_enthalpy(float(t),boundary)
    qsolid=_nuclear_cells(grid,nuclear_solid_power_w)
    qgas=np.asarray(boundary.direct_hydrogen_heating_w,dtype=float)
    mass=np.asarray(boundary.gas_mass_kg,dtype=float)
    hcoef=np.asarray(boundary.convection_w_m2_k,dtype=float)
    v=grid.volumes_m3
    re=np.asarray(grid.r_edges_m)
    ze=np.asarray(grid.z_edges_m)
    dz=np.diff(ze)
    # Volume-bisector centers preserve cylindrical FV control-volume moments.
    rc=np.sqrt(0.5*(re[1:]**2+re[:-1]**2))
    nsolid=nz*nr
    ngas=nz
    total=nsolid+ngas
    sigma=np.empty((nz,nr))
    cp_ref=np.empty((nz,nr))
    laws=[coating if j<grid.n_coating_cells else fuel for j in range(nr)]
    for j,law in enumerate(laws):
        sigma[:,j]=law.density_kg_m3
        cp_ref[:,j]=law.cp_ref_j_kg_k

    sg=s0.copy()
    gg=g0.copy()
    converged=False
    lasterr=math.inf
    k=np.empty((nz,nr))
    gconv=np.zeros(nz)
    for iteration in range(1,max_iterations+1):
        for j,law in enumerate(laws):
            k[:,j]=[law.conductivity(float(t)) for t in sg[:,j]]
        rows:list[int]=[]
        cols:list[int]=[]
        vals:list[float]=[]
        rhs=np.zeros(total,dtype=float)
        def add(i:int,j:int,g:float)->None:
            if not math.isfinite(g):
                raise ValueError("nonfinite coupled matrix coefficient")
            rows.append(i);cols.append(j);vals.append(g)
        def index(i:int,j:int)->int:
            return i*nr+j
        # 2-D cylindrical solid conduction, conservative interior faces.
        for i in range(nz):
            for j in range(nr):
                a=index(i,j)
                law=laws[j]
                store=(0.0 if dt_s is None else
                       v[i,j]*law.density_kg_m3 *
                       law.secant_cp(float(s0[i,j]),float(sg[i,j]))/dt_s)
                diagonal=store
                rhs[a]=store*s0[i,j]+qsolid[i,j]
                for neighbor in (j-1,j+1):
                    if not 0<=neighbor<nr:continue
                    rface=re[max(j,neighbor)]
                    resistance=(abs(math.log(rface/rc[j]))/k[i,j] +
                                abs(math.log(rc[neighbor]/rface))/k[i,neighbor])
                    if min(j,neighbor)==grid.n_coating_cells-1:
                        resistance+=fuel_coating_contact_resistance_m2_k_w/rface
                    conductance=2*math.pi*dz[i]/resistance
                    diagonal+=conductance
                    add(a,index(i,neighbor),-conductance)
                face_area=math.pi*(re[j+1]**2-re[j]**2)
                for neighbor in (i-1,i+1):
                    if not 0<=neighbor<nz:continue
                    resistance=(0.5*dz[i]/k[i,j]+
                                0.5*dz[neighbor]/k[neighbor,j])
                    conductance=face_area/resistance
                    diagonal+=conductance
                    add(a,index(neighbor,j),-conductance)
                if j==0:
                    resistance=(math.log(rc[j]/re[0])/
                                (2*math.pi*k[i,j]*dz[i]) +
                                1/(hcoef[i]*2*math.pi*re[0]*dz[i]))
                    gconv[i]=1.0/resistance
                    diagonal+=gconv[i]
                    add(a,nsolid+i,-gconv[i])
                add(a,a,diagonal)
        # Implicit upwind enthalpy conservation in the H2 fluid.
        for i in range(nz):
            a=nsolid+i
            oldgas=float(g0[i])
            cpstorage=(0.0 if dt_s is None else
                       mass[i]*_gas_secant(oldgas,float(gg[i]),boundary)/dt_s)
            upguess=(boundary.inlet_temperature_k if i==0 else float(gg[i-1]))
            cpflow=_gas_secant(upguess,float(gg[i]),boundary)
            if not math.isfinite(cpflow) or cpflow<=0:
                raise ValueError("invalid positive hydrogen heat capacity")
            adv=boundary.mass_flow_kg_s*cpflow
            add(a,a,cpstorage+adv+gconv[i])
            add(a,index(i,0),-gconv[i])
            rhs[a]=cpstorage*oldgas+qgas[i]
            if i:
                add(a,nsolid+i-1,-adv)
            else:
                rhs[a]+=adv*boundary.inlet_temperature_k
        matrix=coo_matrix((vals,(rows,cols)),shape=(total,total)).tocsr()
        trial=np.asarray(spsolve(matrix,rhs),dtype=float)
        if not np.isfinite(trial).all() or np.min(trial)<=0:
            raise RuntimeError("implicit core thermal solve returned nonphysical temperatures")
        sn=trial[:nsolid].reshape((nz,nr))
        gn=trial[nsolid:]
        for t in gn:
            _gas_enthalpy(float(t),boundary)
        lasterr=float(max(np.max(np.abs(sn-sg)),np.max(np.abs(gn-gg))))
        sg=relaxation*sn+(1-relaxation)*sg
        gg=relaxation*gn+(1-relaxation)*gg
        if lasterr<=tolerance_k:
            converged=True
            break
    if not converged:
        raise RuntimeError("core conjugate FV Picard nonconvergence: "
                           f"{lasterr:.6g} K after {max_iterations} iterations")
    # Re-evaluate wall conductance with final material coefficients.
    gconv=np.array([
        1.0/(math.log(rc[0]/re[0])/
             (2*math.pi*laws[0].conductivity(float(sg[i,0]))*dz[i])+
             1.0/(hcoef[i]*2*math.pi*re[0]*dz[i]))
        for i in range(nz)],dtype=float)
    wall=gconv*(sg[:,0]-gg)
    innerwall=gg+wall/(hcoef*2*math.pi*re[0]*dz)
    stored_s=0.0
    stored_g=0.0
    if dt_s is not None:
        stored_s=math.fsum(
            laws[j].density_kg_m3*v[i,j]*
            laws[j].enthalpy_change(float(s0[i,j]),float(sg[i,j]))
            for i in range(nz) for j in range(nr))
        stored_g=math.fsum(
            mass[i]*(_gas_enthalpy(float(gg[i]),boundary)-
                     _gas_enthalpy(float(g0[i]),boundary))
            for i in range(nz))
    enthalpy_in=boundary.mass_flow_kg_s*_gas_enthalpy(
        boundary.inlet_temperature_k,boundary)
    enthalpy_out=boundary.mass_flow_kg_s*_gas_enthalpy(
        float(gg[-1]),boundary)
    # Combined domain first law: interior solid/gas wall exchange cancels.
    power=float(np.sum(qsolid)+np.sum(qgas)+enthalpy_in-enthalpy_out)
    residual=stored_s+stored_g-(power*(1.0 if dt_s is None else dt_s))
    # Steady [W] residual, transient [J] residual.
    scale=max(1.0,abs(enthalpy_in),abs(enthalpy_out),
              float(np.sum(np.abs(qsolid))),float(np.sum(np.abs(qgas))))
    scale*=1.0 if dt_s is None else dt_s
    if abs(residual)>energy_tolerance*scale+1e-7:
        raise RuntimeError("coupled core FV first-law residual exceeds tolerance: "
                           f"{residual:.6g} {'W' if dt_s is None else 'J'}")
    return CoreSectorResult(
        state=CoreSectorState(sg.copy(),gg.copy()),
        coating_inner_wall_temperature_k=innerwall,
        convected_wall_heating_w=wall,
        solid_nuclear_heating_w=float(qsolid.sum()),
        hydrogen_direct_heating_w=float(qgas.sum()),
        solid_energy_change_j=stored_s,
        hydrogen_energy_change_j=stored_g,
        inlet_enthalpy_w=enthalpy_in,
        outlet_enthalpy_w=enthalpy_out,
        energy_balance_residual_j=residual,
        iterations=iteration,
        transient=dt_s is not None,
    )
