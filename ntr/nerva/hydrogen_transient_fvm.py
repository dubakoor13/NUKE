"""Stage57 finite-volume transient H2: conservative mass, momentum and energy.

19 actual coolant bores; ideal-H2 gas with bounded tabulated cp(T) and
viscosity. Variable cp uses an exact cv(T)=cp(T)-R integral and inverse.
SSP-RK2 and Rusanov intercell flux, Darcy friction (total energy retained),
externally supplied wall watts, and exact boundary-flux budget accounting.

This is a first-order spatial, quasi-1D ideal-gas numerical model, not a
validated hot NERVA fluid EOS, choked-flow, chemistry, wall-compliance,
real-gas or implicit fuel/H2 conjugate solver. Never infer reactor power
or material properties from numerical demonstrations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .config import NervaConfig
from .hydrogen_channel_hydraulics import (
    IdealHydrogenProperties, darcy_friction_factor,
)


def _field(name: str, value: object, shape: tuple[int, ...]) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = np.full(shape, float(a))
    if a.shape != shape or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f"{name}: expected positive finite scalar or {shape}")
    return a


@dataclass(frozen=True)
class TransientHydrogenModel:
    eos: IdealHydrogenProperties
    axial_edges_m: np.ndarray
    diameter_m: float
    area_m2: float
    relative_roughness: float
    maximum_mach: float = .75
    maximum_cfl: float = .45

    @property
    def nzone(self) -> int:
        return len(self.axial_edges_m)-1

    @property
    def dz_m(self) -> np.ndarray:
        return np.diff(self.axial_edges_m)

    @property
    def volume_m3(self) -> np.ndarray:
        return self.dz_m[:,None]*self.area_m2*np.ones((1,19))


@dataclass(frozen=True)
class ConservativeHydrogenState:
    """Conserved density [rho, rho*u, rho*(e+u^2/2)] [axial,channel,3]."""
    conserved: np.ndarray


@dataclass(frozen=True)
class HydrogenPrimitive:
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    density_kg_m3: np.ndarray
    velocity_m_s: np.ndarray
    mach: np.ndarray
    specific_internal_energy_j_kg: np.ndarray
    sound_speed_m_s: np.ndarray


@dataclass(frozen=True)
class HydrogenBoundary:
    """'wall', 'transmissive', or 'reservoir' (two specified ghost states).

    Reservoir ghost states are NOT characteristic nonreflecting boundaries.
    """
    mode: str = "wall"
    inlet_pressure_pa: object | None = None
    inlet_temperature_k: object | None = None
    inlet_velocity_m_s: object | None = None
    outlet_pressure_pa: object | None = None
    outlet_temperature_k: object | None = None
    outlet_velocity_m_s: object | None = None


@dataclass(frozen=True)
class TransientHydrogenStep:
    state: ConservativeHydrogenState
    time_step_s: float
    maximum_cfl_used: float
    inlet_mass_flux_kg_s: np.ndarray
    outlet_mass_flux_kg_s: np.ndarray
    inlet_energy_flux_w: np.ndarray
    outlet_energy_flux_w: np.ndarray
    inlet_momentum_flux_n: np.ndarray
    outlet_momentum_flux_n: np.ndarray
    wall_heat_j: float
    momentum_wall_friction_impulse_n_s: float
    mass_change_kg: float
    total_energy_change_j: float
    momentum_change_kg_m_s: float
    mass_balance_residual_kg: float
    energy_balance_residual_j: float
    momentum_balance_residual_kg_m_s: float


def build_transient_hydrogen_model(
    config: NervaConfig, eos: IdealHydrogenProperties,
    *, axial_edges_m: object, absolute_roughness_m: float = 0.,
    maximum_mach: float = .75, maximum_cfl: float = .45,
) -> TransientHydrogenModel:
    config.validate()
    z=np.asarray(axial_edges_m,dtype=float)
    if (z.ndim!=1 or not 2<=len(z)<=513 or not np.all(np.isfinite(z))
            or np.any(np.diff(z)<=0.) or abs(float(z[0]))>1e-12
            or abs(float(z[-1])-config.active_length_cm/100)>1e-10):
        raise ValueError("axial faces must partition actual active fuel length")
    d=config.coolant_bore_diameter_cm/100
    if (not math.isfinite(d) or d<=0
            or not math.isfinite(absolute_roughness_m)
            or not 0<=absolute_roughness_m<.05*d):
        raise ValueError("unqualified channel diameter or relative roughness")
    if (not math.isfinite(maximum_mach) or not 0<maximum_mach<=.75
            or not math.isfinite(maximum_cfl) or not 0<maximum_cfl<=.5):
        raise ValueError("subsonic Mach <=.75 and SSP CFL <=.5 required")
    return TransientHydrogenModel(
        eos,z.copy(),d,math.pi*d*d/4,absolute_roughness_m/d,
        maximum_mach,maximum_cfl)


def _e_internal(eos: IdealHydrogenProperties, temperatures: object) -> np.ndarray:
    t=np.asarray(temperatures,dtype=float)
    return (eos.cp_j_kg_k.integral(t)
            -eos.gas_constant*(t-float(eos.cp_j_kg_k.temperature_k[0])))


def _T_from_e(eos: IdealHydrogenProperties, energy: object) -> np.ndarray:
    """Exact inversion of piecewise linear cv, no extrapolation."""
    e=np.asarray(energy,dtype=float)
    tab=eos.cp_j_kg_k
    t=tab.temperature_k
    cv=tab.values-eos.gas_constant
    knot_e=_e_internal(eos,t)
    if (not np.all(np.isfinite(e)) or np.any(e < -1e-7)
            or np.any(e>knot_e[-1]+1e-7)):
        raise ValueError("internal energy outside qualified cp(T) interval")
    clipped=np.clip(e,0.,float(knot_e[-1]))
    idx=np.clip(np.searchsorted(knot_e,clipped,side="right")-1,0,len(t)-2)
    de=clipped-knot_e[idx]
    slope=(cv[idx+1]-cv[idx])/(t[idx+1]-t[idx])
    dt=2*de/(cv[idx]+np.sqrt(np.maximum(0.,cv[idx]**2+2*slope*de)))
    out=t[idx]+dt
    tab.at(out)
    return out


def primitive(
    model: TransientHydrogenModel, state: ConservativeHydrogenState
) -> HydrogenPrimitive:
    u=np.asarray(state.conserved,dtype=float)
    if u.shape!=(model.nzone,19,3) or not np.all(np.isfinite(u)):
        raise ValueError("conserved state requires finite (nz,19,3)")
    rho=u[...,0]
    if np.any(rho<=0):
        raise ValueError("nonpositive hydrogen density")
    velocity=u[...,1]/rho
    internal=u[...,2]/rho-.5*velocity**2
    temp=_T_from_e(model.eos,internal)
    p=rho*model.eos.gas_constant*temp
    sound=model.eos.sound_speed_m_s(temp)
    mach=np.abs(velocity)/sound
    if np.any(mach>=model.maximum_mach):
        raise ValueError("subsonic flow limit exceeded")
    return HydrogenPrimitive(p,temp,rho,velocity,mach,internal,sound)


def make_state(
    model: TransientHydrogenModel, *, pressure_pa: object,
    temperature_k: object, velocity_m_s: object=0.,
) -> ConservativeHydrogenState:
    shape=(model.nzone,19)
    p=_field("absolute pressure",pressure_pa,shape)
    t=_field("temperature K",temperature_k,shape)
    v=np.asarray(velocity_m_s,dtype=float)
    if v.ndim==0:
        v=np.full(shape,float(v))
    if v.shape!=shape or not np.all(np.isfinite(v)):
        raise ValueError("velocity must be finite scalar or (nz,19)")
    rho=model.eos.density_kg_m3(p,t)
    e=_e_internal(model.eos,t)
    out=ConservativeHydrogenState(
        np.stack((rho,rho*v,rho*(e+.5*v*v)),axis=-1))
    primitive(model,out)
    return out


def _flux(u: np.ndarray, rho: np.ndarray,
          velocity: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.stack((rho*velocity,rho*velocity**2+p,
                     velocity*(u[...,2]+p)),axis=-1)


def _reservoir(
    model: TransientHydrogenModel, p: object,t: object,v: object
) -> np.ndarray:
    p=_field("reservoir p",p,(19,))
    t=_field("reservoir T",t,(19,))
    v=np.asarray(v,dtype=float)
    if v.ndim==0:
        v=np.full(19,float(v))
    if v.shape!=(19,) or not np.all(np.isfinite(v)):
        raise ValueError("reservoir velocity must be finite scalar or 19 vector")
    rho=model.eos.density_kg_m3(p,t)
    if np.any(np.abs(v)/model.eos.sound_speed_m_s(t)>=model.maximum_mach):
        raise ValueError("reservoir outside subsonic qualification")
    e=_e_internal(model.eos,t)
    return np.stack((rho,rho*v,rho*(e+.5*v*v)),axis=-1)


def _rusanov(
    model: TransientHydrogenModel, left: np.ndarray, right: np.ndarray,
) -> np.ndarray:
    rho_l=left[...,0]
    rho_r=right[...,0]
    vl=left[...,1]/rho_l
    vr=right[...,1]/rho_r
    tl=_T_from_e(model.eos,left[...,2]/rho_l-.5*vl**2)
    tr=_T_from_e(model.eos,right[...,2]/rho_r-.5*vr**2)
    pl=rho_l*model.eos.gas_constant*tl
    pr=rho_r*model.eos.gas_constant*tr
    speed=np.maximum(
        np.abs(vl)+model.eos.sound_speed_m_s(tl),
        np.abs(vr)+model.eos.sound_speed_m_s(tr))
    return (.5*(_flux(left,rho_l,vl,pl)+_flux(right,rho_r,vr,pr))
            -.5*speed[...,None]*(right-left))


def _face_fluxes(
    model: TransientHydrogenModel, u: np.ndarray,
    prim: HydrogenPrimitive, boundary: HydrogenBoundary,
) -> np.ndarray:
    nz=model.nzone
    faces=np.zeros((nz+1,19,3))
    if nz>1:
        faces[1:-1]=_rusanov(model,u[:-1],u[1:])
    if boundary.mode=="wall":
        faces[0,:,1]=prim.pressure_pa[0]
        faces[-1,:,1]=prim.pressure_pa[-1]
    elif boundary.mode=="transmissive":
        faces[0]=_rusanov(model,u[0],u[0])
        faces[-1]=_rusanov(model,u[-1],u[-1])
    elif boundary.mode=="reservoir":
        args=(boundary.inlet_pressure_pa,boundary.inlet_temperature_k,
              boundary.inlet_velocity_m_s,boundary.outlet_pressure_pa,
              boundary.outlet_temperature_k,boundary.outlet_velocity_m_s)
        if any(v is None for v in args):
            raise ValueError("reservoir mode requires all six primitive BC data")
        left=_reservoir(model,*args[:3])
        right=_reservoir(model,*args[3:])
        faces[0]=_rusanov(model,left,u[0])
        faces[-1]=_rusanov(model,u[-1],right)
    else:
        raise ValueError("unknown boundary mode: wall/transmissive/reservoir")
    return faces


def _rhs(
    model: TransientHydrogenModel, u: np.ndarray,
    boundary: HydrogenBoundary, heat_w: np.ndarray,
    friction_enabled: bool,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    prim=primitive(model,ConservativeHydrogenState(u))
    flux=_face_fluxes(model,u,prim,boundary)
    rhs=-(flux[1:]-flux[:-1])/model.dz_m[:,None,None]
    friction=np.zeros((model.nzone,19))
    if friction_enabled:
        mu=model.eos.viscosity_pa_s.at(prim.temperature_k)
        re=(prim.density_kg_m3*np.abs(prim.velocity_m_s)
            *model.diameter_m/mu)
        f=darcy_friction_factor(re,model.relative_roughness)
        friction=-(f*prim.density_kg_m3*prim.velocity_m_s
                   *np.abs(prim.velocity_m_s))/(2*model.diameter_m)
        rhs[...,1]+=friction
    # The absence of an energy friction sink is physical: adiabatic wall
    # friction converts directed kinetic energy into internal energy.
    rhs[...,2]+=heat_w/model.volume_m3
    return rhs,flux,friction


def advance_transient_hydrogen(
    model: TransientHydrogenModel, initial: ConservativeHydrogenState,
    *, dt_s: float, wall_heat_w: object=0.,
    boundary: HydrogenBoundary=HydrogenBoundary(),
    friction_enabled: bool=True,
) -> TransientHydrogenStep:
    """SSPRK2 for the conservative FV equations with exact boundary budgets.

    wall_heat_w is externally supplied thermal transfer into H2, in watts
    per axial segment/channel. Do not simultaneously advance Stage55 H2
    enthalpy for this same control volume and time interval.
    """
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("dt_s must be finite positive")
    heat=np.asarray(wall_heat_w,dtype=float)
    if heat.ndim==0:
        heat=np.full((model.nzone,19),float(heat))
    if heat.shape!=(model.nzone,19) or not np.all(np.isfinite(heat)):
        raise ValueError("wall heat W must be finite scalar or (nz,19)")
    old=np.asarray(initial.conserved,dtype=float)
    prim=primitive(model,initial)
    cfl=dt_s*float(np.max(
        (np.abs(prim.velocity_m_s)+prim.sound_speed_m_s)/model.dz_m[:,None]))
    if cfl>model.maximum_cfl*(1+1e-12):
        raise ValueError(f"CFL {cfl:g} exceeds limit; reduce timestep")
    du0,f0,fr0=_rhs(model,old,boundary,heat,friction_enabled)
    stage=old+dt_s*du0
    prim_stage=primitive(model,ConservativeHydrogenState(stage))
    cfl1=dt_s*float(np.max(
        (np.abs(prim_stage.velocity_m_s)+prim_stage.sound_speed_m_s)
        /model.dz_m[:,None]))
    if cfl1>model.maximum_cfl*(1+1e-12):
        raise ValueError(f"SSPRK2 stage CFL {cfl1:g} exceeds limit")
    du1,f1,fr1=_rhs(model,stage,boundary,heat,friction_enabled)
    new=.5*old+.5*(stage+dt_s*du1)
    primitive(model,ConservativeHydrogenState(new))
    avgflux=.5*(f0+f1)
    v=model.volume_m3
    delta=(new-old)*v[...,None]
    dm=float(np.sum(delta[...,0]))
    dp=float(np.sum(delta[...,1]))
    de=float(np.sum(delta[...,2]))
    minlet=avgflux[0,:,0]*model.area_m2
    moutlet=avgflux[-1,:,0]*model.area_m2
    pinlet=avgflux[0,:,1]*model.area_m2
    poutlet=avgflux[-1,:,1]*model.area_m2
    einlet=avgflux[0,:,2]*model.area_m2
    eoutlet=avgflux[-1,:,2]*model.area_m2
    friction_impulse=float(.5*dt_s*np.sum((fr0+fr1)*v))
    added=float(dt_s*np.sum(heat))
    rm=dm-dt_s*float(np.sum(minlet-moutlet))
    rp=dp-dt_s*float(np.sum(pinlet-poutlet))-friction_impulse
    re=de-dt_s*float(np.sum(einlet-eoutlet))-added
    massscale=max(1e-12,float(np.sum(old[...,0]*v)),abs(dm))
    momentscale=max(1e-10,float(np.sum(np.abs(old[...,1])*v)),abs(dp))
    energyscale=max(1e-6,float(np.sum(np.abs(old[...,2])*v)),abs(de))
    if (abs(rm)>1e-10*massscale or abs(rp)>2e-10*momentscale
            or abs(re)>2e-10*energyscale):
        raise ArithmeticError("hydrogen FV continuity/momentum/energy balance failed")
    return TransientHydrogenStep(
        ConservativeHydrogenState(new),dt_s,max(cfl,cfl1),
        minlet,moutlet,einlet,eoutlet,pinlet,poutlet,added,friction_impulse,
        dm,de,dp,rm,re,rp)
