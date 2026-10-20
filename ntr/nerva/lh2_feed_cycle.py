"""Cryogenic LH2 pump -> preheater -> real-fluid core -> choked nozzle.

Uses one consistent CoolProp HEOS::Hydrogen enthalpy convention from
20 K LH2 inlet to warm gaseous core outlet, including phase change
inside a lumped upstream PREHEATER (no two-phase channel-flow claim).
Real-fluid channel momentum includes acceleration and Darcy friction;
energy retains gas velocity kinetic energy.

Nozzle uses a *local* effective-gamma calorically-perfect surrogate at
the chamber state. It is not a real-fluid nozzle expansion calculation.

LIMITS: CoolProp Hydrogen EOS has Tmax=1000 K; this solver applies
a conservative <=900 K cutoff. Actual hot NERVA engine needs NASA7/
CEA thermochemistry at higher temperature and an accurately calibrated
two-phase/nozzle/reflector/tie-tube/turbopump/turbine network.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .nerva_engine import (
    NervaGasModel, NervaNozzleResult, evaluate_nerva_nozzle,
)
from .thermal import friction_factor


class CryogenicCycleError(ValueError):
    """No physically admissible single-phase warm-core operating point."""


class HydrogenEOS:
    """Explicit CoolProp dependency, with no property-reference mixing."""

    def __init__(self):
        try:
            from CoolProp.CoolProp import PropsSI
        except ImportError as exc:
            raise RuntimeError(
                "CoolProp is required: python -m pip install CoolProp"
            ) from exc
        self.props = PropsSI
        self.fluid = "HEOS::Hydrogen"
        self.maximum_temperature_k = min(
            900.,float(PropsSI("TMAX",self.fluid))
        )

    def get(self, key: str, x: str, xval: float, y: str, yval: float) -> float:
        try:
            val=float(self.props(key,x,xval,y,yval,self.fluid))
        except (ValueError,OverflowError) as exc:
            raise CryogenicCycleError(
                f"CoolProp Hydrogen property failure: {key} at "
                f"{x}={xval:.8g}, {y}={yval:.8g}: {exc}"
            ) from exc
        if not math.isfinite(val):
            raise CryogenicCycleError(f"nonfinite CoolProp {key}")
        return val

    def tp(self, key: str, t: float, p: float) -> float:
        return self.get(key,"T",t,"P",p)

    def hp(self, key: str, h: float, p: float) -> float:
        return self.get(key,"Hmass",h,"P",p)

    def sp(self, key: str, s: float, p: float) -> float:
        return self.get(key,"Smass",s,"P",p)


@dataclass(frozen=True)
class LH2Cycle:
    tank_temperature_k: float
    tank_pressure_pa: float
    pump_shutoff_pressure_rise_pa: float
    pump_head_quadratic_pa_s2_kg2: float
    pump_isentropic_efficiency: float
    feed_loss_quadratic_pa_s2_kg2: float
    chamber_loss_quadratic_pa_s2_kg2: float
    preheater_thermal_power_w: float
    channel_wall_power_w: tuple[float,...]
    channel_direct_hydrogen_power_w: tuple[float,...]
    z_edges_m: tuple[float,...]
    number_of_channels: int
    channel_diameter_m: float
    throat_area_m2: float
    nozzle_expansion_ratio: float
    maximum_mass_flow_kg_s: float
    channel_roughness_m: float = 1e-6
    allowable_mach: float = 0.8
    nozzle_efficiency: float = 0.95
    ambient_pressure_pa: float = 0.
    maximum_eos_temperature_k: float = 900.

    def validate(self) -> None:
        for name in (
            "tank_temperature_k","tank_pressure_pa",
            "pump_shutoff_pressure_rise_pa","preheater_thermal_power_w",
            "channel_diameter_m","throat_area_m2",
            "maximum_mass_flow_kg_s","maximum_eos_temperature_k",
        ):
            value=getattr(self,name)
            if not math.isfinite(value) or value<=0:
                raise ValueError(f"{name} must be positive and finite")
        if not 14.1<self.tank_temperature_k<33.:
            raise ValueError("tank inlet must be subcritical cryogenic liquid H2")
        if not 200.<self.maximum_eos_temperature_k<=900.:
            raise ValueError("EOS model temperature limit must be within (200,900] K")
        if not 0<self.pump_isentropic_efficiency<=1:
            raise ValueError("pump efficiency must lie in (0,1]")
        if not 0<self.allowable_mach<1:
            raise ValueError("allowable channel Mach must lie in (0,1)")
        if not 0<self.nozzle_efficiency<=1:
            raise ValueError("nozzle efficiency must lie in (0,1]")
        if self.nozzle_expansion_ratio<=1:
            raise ValueError("nozzle expansion ratio must exceed one")
        if not isinstance(self.number_of_channels,int) or self.number_of_channels<1:
            raise ValueError("number_of_channels must be positive integer")
        for name in (
            "pump_head_quadratic_pa_s2_kg2",
            "feed_loss_quadratic_pa_s2_kg2",
            "chamber_loss_quadratic_pa_s2_kg2",
            "channel_roughness_m","ambient_pressure_pa",
        ):
            x=getattr(self,name)
            if not math.isfinite(x) or x<0:
                raise ValueError(f"{name} must be nonnegative and finite")
        z=np.asarray(self.z_edges_m,dtype=float)
        w=np.asarray(self.channel_wall_power_w,dtype=float)
        g=np.asarray(self.channel_direct_hydrogen_power_w,dtype=float)
        if (w.ndim!=1 or w.size<1 or g.shape!=w.shape or
                z.shape!=(w.size+1,) or
                not np.all(np.isfinite(z)) or np.any(np.diff(z)<=0) or
                not np.all(np.isfinite(w)) or not np.all(np.isfinite(g)) or
                np.any(w<0) or np.any(g<0)):
            raise ValueError("axial gas-channel powers and coordinates inconsistent")


@dataclass(frozen=True)
class LH2OperatingPoint:
    mass_flow_kg_s: float
    pump_discharge_pressure_pa: float
    pump_discharge_temperature_k: float
    pump_shaft_power_w: float
    pump_isentropic_enthalpy_rise_j_kg: float
    pump_actual_enthalpy_rise_j_kg: float
    pump_inlet_density_kg_m3: float
    inlet_npsh_available_m: float
    core_inlet_pressure_pa: float
    core_inlet_temperature_k: float
    core_outlet_pressure_pa: float
    core_outlet_temperature_k: float
    chamber_pressure_pa: float
    chamber_temperature_k: float
    core_friction_loss_pa: float
    core_acceleration_loss_pa: float
    maximum_core_mach: float
    preheater_thermal_power_w: float
    core_thermal_power_w: float
    mass_residual_kg_s: float
    energy_residual_w: float
    pressure_residual_pa: float
    nozzle: NervaNozzleResult


def _warm_channel_segment(
    eos: HydrogenEOS, h0_in: float, t_in: float, p_in: float,
    mass_flux: float, power_per_channel_w: float, mdot_per_channel: float,
    dz: float, diameter: float, roughness: float,
    mach_max: float, t_max: float,
) -> tuple[float,float,float,float,float,float]:
    """Solve one gas control volume by total-enthalpy + momentum closure."""
    rho_in=eos.tp("Dmass",t_in,p_in)
    u_in=mass_flux/rho_in
    h_target=h0_in+power_per_channel_w/mdot_per_channel

    def outlet_state(p: float):
        if p<=0:
            raise CryogenicCycleError("channel pressure nonpositive")
        # At fixed p, solve h(T,p) + (G/rho(T,p))²/2 = h0_target
        lo=200.
        hi=t_max
        def residual(t: float) -> float:
            h=eos.tp("Hmass",t,p)
            rho=eos.tp("Dmass",t,p)
            return h+0.5*(mass_flux/rho)**2-h_target
        if residual(lo)>0 or residual(hi)<0:
            raise CryogenicCycleError(
                "gas core enthalpy exceeds 200K-to-EOS temperature bounds"
            )
        for _ in range(33):
            mid=(lo+hi)/2
            if residual(mid)>0:
                hi=mid
            else:
                lo=mid
        temp=(lo+hi)/2
        rho=eos.tp("Dmass",temp,p)
        vel=mass_flux/rho
        t_mid=0.5*(temp+t_in)
        p_mid=0.5*(p_in+p)
        visc=eos.tp("VISCOSITY",t_mid,p_mid)
        re=mass_flux*diameter/visc
        darcy=friction_factor(re,roughness/diameter)
        p_f=darcy*dz/diameter*mass_flux*(u_in+vel)/4.
        p_a=mass_flux*(vel-u_in)
        return p+p_f+p_a-p_in,temp,vel,p_f,p_a

    # Look for a subsonic root using damped finite-difference Newton,
    # reject loss of the physical low-Mach branch.
    p=p_in
    for _ in range(40):
        residual,temp,vel,p_f,p_a=outlet_state(p)
        sound=eos.tp("A",temp,p)
        if vel>=mach_max*sound:
            raise CryogenicCycleError("gas core approaching sonic/choked flow")
        if abs(residual)<1e-8*max(1.,p_in):
            return p,temp,vel,p_f,p_a,vel/sound
        eps=max(0.000001*p,0.01)
        if p-eps<=0:
            raise CryogenicCycleError("channel negative pressure")
        low=outlet_state(p-eps)[0]
        deriv=(residual-low)/eps
        if deriv<=0 or not math.isfinite(deriv):
            raise CryogenicCycleError("non-subsonic gas-channel pressure root")
        delta=max(-0.05*p_in,min(0.05*p_in,-residual/deriv))
        p+=delta
        if p<=0 or p>p_in*1.001:
            raise CryogenicCycleError("invalid pressure during gas core solve")
    raise CryogenicCycleError("gas core EOS energy/momentum solver not converged")


def evaluate_lh2_cycle(
    config: LH2Cycle, mdot: float, *,
    eos: HydrogenEOS | None = None
) -> tuple[LH2OperatingPoint,float]:
    """Single candidate mass flow; root accepted only by solve_lh2_cycle."""
    config.validate()
    eos=eos or HydrogenEOS()
    if not math.isfinite(mdot) or mdot<=0 or mdot>config.maximum_mass_flow_kg_s:
        raise ValueError("invalid LH2 mass-flow trial")
    t1,p1=config.tank_temperature_k,config.tank_pressure_pa
    satp=eos.get("P","T",t1,"Q",0.)
    if p1<=satp:
        raise CryogenicCycleError("tank inlet is saturated/vapor; liquid pump requires NPSH")
    rho1=eos.tp("Dmass",t1,p1)
    if rho1<=20:
        raise CryogenicCycleError("tank inlet is not dense liquid hydrogen")
    npsha=(p1-satp)/(rho1*9.80665)
    head=config.pump_shutoff_pressure_rise_pa-config.pump_head_quadratic_pa_s2_kg2*mdot**2
    if head<=0:
        raise CryogenicCycleError("pump head source exhausted")
    p2=p1+head
    h1=eos.tp("Hmass",t1,p1)
    s1=eos.tp("Smass",t1,p1)
    h2s=eos.sp("Hmass",s1,p2)
    if h2s<=h1:
        raise CryogenicCycleError("LH2 pump isentropic enthalpy increment nonpositive")
    h2=h1+(h2s-h1)/config.pump_isentropic_efficiency
    t2=eos.hp("T",h2,p2)
    rho2=eos.hp("Dmass",h2,p2)
    if t2>=34. or rho2<=20:
        raise CryogenicCycleError("pump discharge no longer dense liquid LH2")
    shaft=mdot*(h2-h1)
    feed_loss=config.feed_loss_quadratic_pa_s2_kg2*mdot**2
    pc_in=p2-feed_loss
    if pc_in<=p1:
        raise CryogenicCycleError("feed loss eliminates pump pressure increase")
    h3=h2+config.preheater_thermal_power_w/mdot
    t3=eos.hp("T",h3,pc_in)
    if not 200.<t3<config.maximum_eos_temperature_k:
        raise CryogenicCycleError(
            "preheater must deliver single-phase 200K-to-900K H2 gas "
            "before entering a gas-only core"
        )
    q_wall=np.asarray(config.channel_wall_power_w,dtype=float)
    q_direct=np.asarray(config.channel_direct_hydrogen_power_w,dtype=float)
    q_core=float(q_wall.sum()+q_direct.sum())
    h_static=h3
    t=t3
    p=pc_in
    dz=np.diff(np.asarray(config.z_edges_m,dtype=float))
    area=math.pi*config.channel_diameter_m**2/4.
    branch_mdot=mdot/config.number_of_channels
    flux=branch_mdot/area
    gas_rho=eos.tp("Dmass",t,p)
    v=flux/gas_rho
    if v>=config.allowable_mach*eos.tp("A",t,p):
        raise CryogenicCycleError("core inlet choked")
    # Adiabatic acceleration from a low-velocity preheater header.
    # h3 is the *stagnation enthalpy* at channel entry.
    h0=h_static
    # convert stagnation enthalpy into first-cell static temperature:
    lo,hi=200.,t3
    for _ in range(32):
        mid=(lo+hi)/2
        h_mid=eos.tp("Hmass",mid,p)
        rho_mid=eos.tp("Dmass",mid,p)
        if h_mid+0.5*(flux/rho_mid)**2>h0:
            hi=mid
        else:
            lo=mid
    t=0.5*(lo+hi)
    friction=0.
    acceleration=0.
    mach=0.
    for idx,length in enumerate(dz):
        p,t,v,p_f,p_a,mach=_warm_channel_segment(
            eos,h0,t,p,flux,
            (q_wall[idx]+q_direct[idx])/config.number_of_channels,
            branch_mdot,float(length),
            config.channel_diameter_m,config.channel_roughness_m,
            config.allowable_mach,config.maximum_eos_temperature_k
        )
        friction+=p_f
        acceleration+=p_a
        h0+=float(q_wall[idx]+q_direct[idx])/mdot
    core_p=p
    core_t=t
    pc=core_p-config.chamber_loss_quadratic_pa_s2_kg2*mdot**2
    if pc<=0:
        raise CryogenicCycleError("chamber pressure nonpositive")
    # Adiabatic diffuser: outlet velocity recovered in chamber enthalpy.
    chamber_t=eos.hp("T",h0,pc)
    if chamber_t>=config.maximum_eos_temperature_k or chamber_t<=200:
        raise CryogenicCycleError("chamber gas outside CoolProp EOS range")
    chamber_rho=eos.tp("Dmass",chamber_t,pc)
    cp=eos.tp("Cpmass",chamber_t,pc)
    cv=eos.tp("Cvmass",chamber_t,pc)
    if cv<=0 or cp<=cv:
        raise CryogenicCycleError("invalid chamber heat capacity ratio")
    gamma=cp/cv
    # Calorically perfect local nozzle surrogate; real EOS used upstream.
    r_eff=pc/(chamber_rho*chamber_t)
    gas=NervaGasModel(gamma=gamma,gas_constant_j_kg_k=r_eff,cp_j_kg_k=cp)
    nozzle=evaluate_nerva_nozzle(
        chamber_pressure_pa=pc,chamber_temperature_k=chamber_t,
        throat_area_m2=config.throat_area_m2,
        expansion_ratio=config.nozzle_expansion_ratio,
        ambient_pressure_pa=config.ambient_pressure_pa,
        nozzle_efficiency=config.nozzle_efficiency,gas=gas
    )
    h4=h0  # chamber stagnation enthalpy conserved through diffuser
    energy=mdot*(h4-h1)-(shaft+config.preheater_thermal_power_w+q_core)
    p_error=(p1+head)-(feed_loss+friction+acceleration+
                        config.chamber_loss_quadratic_pa_s2_kg2*mdot**2+pc)
    point=LH2OperatingPoint(
        mass_flow_kg_s=mdot,pump_discharge_pressure_pa=p2,
        pump_discharge_temperature_k=t2,pump_shaft_power_w=shaft,
        pump_isentropic_enthalpy_rise_j_kg=h2s-h1,
        pump_actual_enthalpy_rise_j_kg=h2-h1,
        pump_inlet_density_kg_m3=rho1,
        inlet_npsh_available_m=npsha,
        core_inlet_pressure_pa=pc_in,core_inlet_temperature_k=t3,
        core_outlet_pressure_pa=core_p,core_outlet_temperature_k=core_t,
        chamber_pressure_pa=pc,chamber_temperature_k=chamber_t,
        core_friction_loss_pa=friction,
        core_acceleration_loss_pa=acceleration,
        maximum_core_mach=mach,
        preheater_thermal_power_w=config.preheater_thermal_power_w,
        core_thermal_power_w=q_core,
        mass_residual_kg_s=mdot-nozzle.mass_flow_kg_s,
        energy_residual_w=energy,
        pressure_residual_pa=p_error,
        nozzle=nozzle
    )
    return point,point.mass_residual_kg_s


def solve_lh2_cycle(
    config: LH2Cycle,*, eos: HydrogenEOS | None = None,
    bracket_samples: int=44,relative_mass_tolerance: float=2e-6,
) -> LH2OperatingPoint:
    """Find common LH2 pump-flow/core-loss/nozzle operating point."""
    config.validate()
    eos=eos or HydrogenEOS()
    if bracket_samples<20 or bracket_samples>500:
        raise ValueError("bracket_samples outside [20,500]")
    if not 0<relative_mass_tolerance<1e-3:
        raise ValueError("invalid root tolerance")
    upper=config.maximum_mass_flow_kg_s
    if config.pump_head_quadratic_pa_s2_kg2>0:
        upper=min(upper,math.sqrt(
            config.pump_shutoff_pressure_rise_pa/
            config.pump_head_quadratic_pa_s2_kg2
        )*0.999)
    prev=None
    bracket=None
    for m in np.geomspace(max(upper*0.001,1e-5),upper,bracket_samples):
        try:
            pt,f=evaluate_lh2_cycle(config,float(m),eos=eos)
        except (ValueError,RuntimeError,OverflowError):
            continue
        if prev is not None and f*prev[1]<0:
            bracket=(prev[0],float(m),prev[1],f)
            break
        prev=(float(m),f)
    if bracket is None:
        raise CryogenicCycleError(
            "no physically valid LH2 cycle root: ensure enough preheat "
            "to vaporize and warm H2, achievable pump head/nozzle flow, "
            "and subsonic core below CoolProp 900 K limit"
        )
    lo,hi,flo,fhi=bracket
    result=None
    for _ in range(55):
        mid=(lo+hi)*0.5
        result,f=evaluate_lh2_cycle(config,mid,eos=eos)
        if abs(f)<relative_mass_tolerance*mid:
            break
        if (f>0)==(flo>0):
            lo,flo=mid,f
        else:
            hi,fhi=mid,f
    if result is None or abs(result.mass_residual_kg_s)>relative_mass_tolerance*result.mass_flow_kg_s:
        raise CryogenicCycleError("LH2 cycle nozzle/flow closure failed")
    energy_scale=max(1.,result.pump_shaft_power_w+
                     result.preheater_thermal_power_w+result.core_thermal_power_w)
    if abs(result.energy_residual_w)>1e-6*energy_scale:
        raise RuntimeError("LH2 cycle total enthalpy-energy budget failed")
    if abs(result.pressure_residual_pa)>1e-7*max(1.,result.pump_discharge_pressure_pa):
        raise RuntimeError("LH2 feed/core/chamber momentum budget failed")
    return result
