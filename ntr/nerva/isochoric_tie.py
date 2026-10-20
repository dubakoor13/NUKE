"""Fixed-volume real-fluid NERVA tie counterflow: mass and INTERNAL energy.

This separate CoolProp H2 finite-volume model corrects an important
thermodynamic inconsistency in the earlier *isobaric* cryogenic tie
screening model, which stores fixed-mass enthalpy despite changing
density. Here each supply/return control volume has an explicit rigid
volume V. With equal prescribed inlet/outlet mass fluxes (m_dot) its
hydrogen mass is constant and rho = m/V at every step. Pressure is NOT
fixed: CoolProp solves p,T,h,Q from (rho,u). Fluid energy uses m*u
(internal energy), while advective energy uses m_dot*h.

The shared solid and two counterflow fluids are advanced simultaneously
using backward Euler, including OpenMC nuclear deposition in distinct
solid, supply-H2 and return-H2 partitions. No boundary pressure, flow
distribution, axial momentum, thermal strain or fluid acceleration is
solved. Equal inlet/outlet cell flux is a *prescribed* special case of
the mass equation, not a hydraulic closure; EOS pressures may conflict
with the network's independently solved pressure loss. Refuse to claim
coupled thermal-hydraulic or NASA-validated NERVA behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import ResolvedHeating


@dataclass(frozen=True)
class IsochoricTieParameters:
    supply_cell_volume_m3: float
    return_cell_volume_m3: float
    solid_capacity_j_k: float
    supply_ua_w_k: float
    return_ua_w_k: float
    min_temperature_k: float = 17.0
    max_temperature_k: float = 850.0
    min_pressure_pa: float = 100.0
    max_pressure_pa: float = 50e6
    coupling_tolerance_k: float = 1e-7
    max_iterations: int = 80

    def validate(self) -> None:
        for name in ("supply_cell_volume_m3", "return_cell_volume_m3",
                     "solid_capacity_j_k"):
            x=getattr(self,name)
            if not math.isfinite(x) or x<=0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("supply_ua_w_k","return_ua_w_k"):
            x=getattr(self,name)
            if not math.isfinite(x) or x<0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not (math.isfinite(self.min_temperature_k)
                and math.isfinite(self.max_temperature_k)
                and 14.1<self.min_temperature_k<self.max_temperature_k<=900.):
            raise ValueError("invalid CoolProp temperature bracket")
        if not (math.isfinite(self.min_pressure_pa)
                and math.isfinite(self.max_pressure_pa)
                and 0<self.min_pressure_pa<self.max_pressure_pa):
            raise ValueError("invalid thermodynamic pressure bounds")
        if (not isinstance(self.max_iterations,int)
                or not 2<=self.max_iterations<=500
                or not math.isfinite(self.coupling_tolerance_k)
                or self.coupling_tolerance_k<=0):
            raise ValueError("invalid shared-solid iteration controls")


@dataclass(frozen=True)
class IsochoricTieState:
    solid_temperature_k: np.ndarray
    supply_mass_kg: np.ndarray
    return_mass_kg: np.ndarray
    supply_internal_energy_j_kg: np.ndarray
    return_internal_energy_j_kg: np.ndarray

    def validate(self, heating: ResolvedHeating) -> None:
        shape=heating.tie_solid_w.shape
        for label in ("solid_temperature_k","supply_mass_kg",
                      "return_mass_kg","supply_internal_energy_j_kg",
                      "return_internal_energy_j_kg"):
            x=np.asarray(getattr(self,label))
            if x.shape!=shape or not np.all(np.isfinite(x)):
                raise ValueError(f"invalid isochoric tie field {label}")
        if (np.any(self.solid_temperature_k<=0)
                or np.any(self.supply_mass_kg<=0)
                or np.any(self.return_mass_kg<=0)):
            raise ValueError("isochoric tie solid temperatures and masses must be positive")


@dataclass(frozen=True)
class IsochoricTieStep:
    state: IsochoricTieState
    supply_pressure_pa: np.ndarray
    return_pressure_pa: np.ndarray
    supply_temperature_k: np.ndarray
    return_temperature_k: np.ndarray
    supply_density_kg_m3: np.ndarray
    return_density_kg_m3: np.ndarray
    supply_enthalpy_j_kg: np.ndarray
    return_enthalpy_j_kg: np.ndarray
    supply_quality: np.ndarray
    return_quality: np.ndarray
    supply_wall_heat_w: np.ndarray
    return_wall_heat_w: np.ndarray
    solid_storage_rate_w: float
    fluid_internal_energy_storage_rate_w: float
    nuclear_deposition_w: float
    inlet_enthalpy_power_w: float
    outlet_enthalpy_power_w: float
    energy_residual_j: float
    max_local_mass_balance_residual_kg_s: float
    max_local_energy_balance_residual_w: float
    max_picard_iterations: int


def _properties(eos: HydrogenEOS, density: float, u: float) -> tuple[float,float,float]:
    p=eos.get("P","Dmass",density,"Umass",u)
    t=eos.get("T","Dmass",density,"Umass",u)
    if not (math.isfinite(p) and math.isfinite(t) and p>0 and t>0):
        raise ValueError("nonphysical isochoric H2 EOS state")
    return p,t,u+p/density


def _root(
    eos: HydrogenEOS, rho: float, initial_u: float, h_upstream: float,
    mass: float, mass_flow: float, wall_solid_t: float, ua: float,
    direct_power_w: float, dt_s: float, u_min: float, u_max: float,
) -> float:
    storage=mass/dt_s
    def residual(u: float) -> float:
        _,t,h=_properties(eos,rho,u)
        return (storage*(u-initial_u)+mass_flow*(h-h_upstream)
                -ua*(wall_solid_t-t)-direct_power_w)
    lo,hi=u_min,u_max
    if residual(lo)>1e-5 or residual(hi)<-1e-5:
        raise ValueError("isochoric hydrogen energy root outside bounded EOS domain")
    for _ in range(52):
        mid=(lo+hi)*0.5
        if residual(mid)<=0:
            lo=mid
        else:
            hi=mid
    return 0.5*(lo+hi)


def initial_isochoric_tie_state(
    heating: ResolvedHeating, params: IsochoricTieParameters,
    pressure_pa: float, *, quality: float | None = None,
    fluid_temperature_k: float | None = None,
    solid_temperature_k: float, eos: HydrogenEOS | None = None,
) -> IsochoricTieState:
    """Initialize masses from actual CoolProp density TIMES fixed volumes."""
    heating.validate()
    params.validate()
    if (quality is None)==(fluid_temperature_k is None):
        raise ValueError("specify exactly one of initial temperature or quality")
    if not math.isfinite(pressure_pa) or pressure_pa<=0:
        raise ValueError("invalid initial H2 pressure")
    if not math.isfinite(solid_temperature_k) or solid_temperature_k<=0:
        raise ValueError("invalid initial tie solid temperature")
    eos=eos or HydrogenEOS()
    if quality is not None:
        if not math.isfinite(quality) or not 0<=quality<=1:
            raise ValueError("initial vapor quality must be in [0,1]")
        rho=eos.get("Dmass","P",pressure_pa,"Q",quality)
        u=eos.get("Umass","P",pressure_pa,"Q",quality)
    else:
        rho=eos.tp("Dmass",float(fluid_temperature_k),pressure_pa)
        u=eos.tp("Umass",float(fluid_temperature_k),pressure_pa)
    if not math.isfinite(rho) or rho<=0 or not math.isfinite(u):
        raise ValueError("invalid initial CoolProp mass density/energy")
    shape=heating.tie_solid_w.shape
    state=IsochoricTieState(
        solid_temperature_k=np.full(shape,solid_temperature_k),
        supply_mass_kg=np.full(shape,rho*params.supply_cell_volume_m3),
        return_mass_kg=np.full(shape,rho*params.return_cell_volume_m3),
        supply_internal_energy_j_kg=np.full(shape,u),
        return_internal_energy_j_kg=np.full(shape,u),
    )
    state.validate(heating)
    return state


def advance_isochoric_tie(
    heating: ResolvedHeating, old: IsochoricTieState,
    params: IsochoricTieParameters, inlet_enthalpy_j_kg: float, dt_s: float,
    *, eos: HydrogenEOS | None = None,
) -> IsochoricTieStep:
    """Implicit CV energy with m*u storage and constant-volume EOS p(rho,u).

    Equal prescribed cell-in/cell-out flow makes d(m)/dt=0 for each
    rigid CV. The actual pressure profile is determined by its current
    EOS state, NOT by an independently imposed hydraulic pressure
    boundary. Report (never hide) pressure mismatches downstream.
    """
    heating.validate()
    params.validate()
    old.validate(heating)
    if not math.isfinite(inlet_enthalpy_j_kg):
        raise ValueError("invalid inlet H2 enthalpy")
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("positive implicit timestep required")
    eos=eos or HydrogenEOS()
    shape=heating.tie_solid_w.shape
    us=np.empty(shape)
    ur=np.empty(shape)
    solids=np.empty(shape)
    ps=np.empty(shape)
    pr=np.empty(shape)
    ts=np.empty(shape)
    tr=np.empty(shape)
    hs=np.empty(shape)
    hr=np.empty(shape)
    qs=np.empty(shape)
    qr=np.empty(shape)
    rs=old.supply_mass_kg/params.supply_cell_volume_m3
    rr=old.return_mass_kg/params.return_cell_volume_m3
    max_iters=0
    cache={}

    def energy_limits(rho: float)->tuple[float,float]:
        key=float(rho)
        if key not in cache:
            low=eos.get("Umass","T",params.min_temperature_k,"Dmass",rho)
            high=eos.get("Umass","T",params.max_temperature_k,"Dmass",rho)
            if not math.isfinite(low) or not math.isfinite(high) or low>=high:
                raise ValueError("cannot bracket constant-volume CoolProp internal energy")
            cache[key]=(low,high)
        return cache[key]

    for j in range(shape[0]):
        flow=float(heating.tie_flow_kg_s[j])
        guess=np.asarray(old.solid_temperature_k[j]).copy()
        def fluids(t_solid: np.ndarray)->tuple[np.ndarray,np.ndarray]:
            a=np.empty(shape[1])
            b=np.empty(shape[1])
            for i in range(shape[1]):
                up=inlet_enthalpy_j_kg if i==0 else _properties(
                    eos,float(rs[j,i-1]),float(a[i-1])
                )[2]
                a[i]=_root(
                    eos,float(rs[j,i]),float(old.supply_internal_energy_j_kg[j,i]),
                    float(up),float(old.supply_mass_kg[j,i]),
                    flow,float(t_solid[i]),params.supply_ua_w_k,
                    float(heating.tie_supply_h2_w[j,i]),dt_s,
                    *energy_limits(float(rs[j,i])),
                )
            for i in range(shape[1]-1,-1,-1):
                up=_properties(
                    eos,float(rs[j,-1]),float(a[-1])
                )[2] if i==shape[1]-1 else _properties(
                    eos,float(rr[j,i+1]),float(b[i+1])
                )[2]
                b[i]=_root(
                    eos,float(rr[j,i]),float(old.return_internal_energy_j_kg[j,i]),
                    float(up),float(old.return_mass_kg[j,i]),
                    flow,float(t_solid[i]),params.return_ua_w_k,
                    float(heating.tie_return_h2_w[j,i]),dt_s,
                    *energy_limits(float(rr[j,i])),
                )
            return a,b
        for k in range(1,params.max_iterations+1):
            a,b=fluids(guess)
            at=np.array([_properties(eos,float(rs[j,i]),float(a[i]))[1]
                         for i in range(shape[1])])
            bt=np.array([_properties(eos,float(rr[j,i]),float(b[i]))[1]
                         for i in range(shape[1])])
            rate=params.solid_capacity_j_k/dt_s
            target=(rate*old.solid_temperature_k[j]+
                    heating.tie_solid_w[j]+params.supply_ua_w_k*at+
                    params.return_ua_w_k*bt
                    )/(rate+params.supply_ua_w_k+params.return_ua_w_k)
            if float(np.max(abs(target-guess)))<=params.coupling_tolerance_k:
                guess=target
                break
            guess=0.5*(guess+target)
        else:
            raise RuntimeError(f"isochoric tie instance {j} failed solid/fluid convergence")
        a,b=fluids(guess)
        us[j]=a
        ur[j]=b
        solids[j]=guess
        max_iters=max(max_iters,k)
    for j,i in np.ndindex(shape):
        ps[j,i],ts[j,i],hs[j,i]=_properties(eos,float(rs[j,i]),float(us[j,i]))
        pr[j,i],tr[j,i],hr[j,i]=_properties(eos,float(rr[j,i]),float(ur[j,i]))
        if any((v<params.min_pressure_pa or v>params.max_pressure_pa)
               for v in (ps[j,i],pr[j,i])):
            raise ValueError("EOS pressure outside declared isochoric tie bounds")
        if not (params.min_temperature_k<=ts[j,i]<=params.max_temperature_k
                and params.min_temperature_k<=tr[j,i]<=params.max_temperature_k):
            raise ValueError("isochoric H2 EOS temperature outside declared bounds")
    qsw=params.supply_ua_w_k*(solids-ts)
    qrw=params.return_ua_w_k*(solids-tr)
    solid_rate=(params.solid_capacity_j_k/dt_s)*(solids-old.solid_temperature_k)
    local_solid=solid_rate-heating.tie_solid_w+qsw+qrw
    mass_residual=0.
    local_h_residual=0.
    for j,i in np.ndindex(shape):
        flow=float(heating.tie_flow_kg_s[j])
        sup_up=float(inlet_enthalpy_j_kg if i==0 else hs[j,i-1])
        ret_up=float(hs[j,-1] if i==shape[1]-1 else hr[j,i+1])
        sw=(old.supply_mass_kg[j,i]*(us[j,i]-old.supply_internal_energy_j_kg[j,i])/dt_s
            -flow*(sup_up-hs[j,i])-qsw[j,i]-heating.tie_supply_h2_w[j,i])
        rw=(old.return_mass_kg[j,i]*(ur[j,i]-old.return_internal_energy_j_kg[j,i])/dt_s
            -flow*(ret_up-hr[j,i])-qrw[j,i]-heating.tie_return_h2_w[j,i])
        local_h_residual=max(local_h_residual,abs(float(sw)),abs(float(rw)))
    max_local=max(float(np.max(abs(local_solid))),local_h_residual)
    storage_s=float(np.sum(solid_rate))
    storage_h=float(np.sum(
        old.supply_mass_kg*(us-old.supply_internal_energy_j_kg)+
        old.return_mass_kg*(ur-old.return_internal_energy_j_kg)
    )/dt_s)
    nuclear=float(sum(float(np.sum(x)) for x in (
        heating.tie_solid_w,heating.tie_supply_h2_w,
        heating.tie_return_h2_w,
    )))
    incoming=float(np.sum(heating.tie_flow_kg_s)*inlet_enthalpy_j_kg)
    outgoing=float(np.dot(heating.tie_flow_kg_s,hr[:,0]))
    residual_j=(storage_s+storage_h-nuclear-incoming+outgoing)*dt_s
    power_scale=max(1.,abs(storage_s),abs(storage_h),abs(nuclear),
                    abs(incoming),abs(outgoing))
    if max_local>1e-8*power_scale+1e-5:
        raise RuntimeError("isochoric tie local solid/fluid first-law failure")
    if abs(residual_j)>1e-8*power_scale*dt_s+1e-5:
        raise RuntimeError("isochoric tie global first-law closure failure")
    return IsochoricTieStep(
        state=IsochoricTieState(
            solid_temperature_k=solids,
            supply_mass_kg=np.array(old.supply_mass_kg,copy=True),
            return_mass_kg=np.array(old.return_mass_kg,copy=True),
            supply_internal_energy_j_kg=us,
            return_internal_energy_j_kg=ur,
        ),
        supply_pressure_pa=ps,return_pressure_pa=pr,
        supply_temperature_k=ts,return_temperature_k=tr,
        supply_density_kg_m3=rs,return_density_kg_m3=rr,
        supply_enthalpy_j_kg=hs,return_enthalpy_j_kg=hr,
        supply_quality=np.array([
            eos.get("Q","Dmass",float(rs[j,i]),"Umass",float(us[j,i]))
            for j,i in np.ndindex(shape)
        ]).reshape(shape),
        return_quality=np.array([
            eos.get("Q","Dmass",float(rr[j,i]),"Umass",float(ur[j,i]))
            for j,i in np.ndindex(shape)
        ]).reshape(shape),
        supply_wall_heat_w=qsw,return_wall_heat_w=qrw,
        solid_storage_rate_w=storage_s,
        fluid_internal_energy_storage_rate_w=storage_h,
        nuclear_deposition_w=nuclear,
        inlet_enthalpy_power_w=incoming,outlet_enthalpy_power_w=outgoing,
        energy_residual_j=residual_j,
        max_local_mass_balance_residual_kg_s=mass_residual,
        max_local_energy_balance_residual_w=max_local,
        max_picard_iterations=max_iters,
    )
