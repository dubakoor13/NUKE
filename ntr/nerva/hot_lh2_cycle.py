"""Cryogenic LH2 pump/preheater -> hot NASA-7 fuel channels -> H2 nozzle.

A phase-aware CoolProp LH2 feed/preheater is joined to hot molecular H2
NASA-7 gas-channel and isentropic NOZZLE models using a matched *enthalpy*
reference at the warm-gas handoff. The reference is shifted ONCE,
per operating-point inlet state:

    h_engine(T) = h_CoolProp(T_bridge,P_bridge)
                  + h_NASA7(T) - h_NASA7(T_bridge)

Therefore h_engine(T_bridge) == h_CoolProp(T_bridge,P_bridge) and
m*(h_engine(Tch)-h_tank) == pump_shaft + preheater + core_heat.

Pressure-dependent real-fluid properties ABOVE the bridge are replaced
with ideal-gas frozen molecular H2. This is an APPROXIMATION: the H2
enthalpy slope, density and viscosity may not be continuous at handoff.
The first-law audit proves matching reference conventions, NOT EOS
accuracy or a validated hot NERVA engine. No turbine power extraction,
regenerative nozzle/reflector/tie cooling, neutron kinetics, transients,
or detailed fuel-to-coolant conduction is included. Source curve and
energy deposition are prescribed engineering inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .compressible_channel import (
    R_H2, SubsonicFlowError, solve_fuel_channel_compressible,
)
from .gas_feed_network import (
    _static_channel_inlet_temperature, _temperature_at_enthalpy,
)
from .lh2_feed_cycle import (
    CryogenicCycleError, HydrogenEOS, LH2Cycle,
)
from .nasa_hydrogen import (
    T_MIN, T_MAX, hydrogen_state, choked_hydrogen_mass_flux,
    hydrogen_nozzle, HydrogenNozzle,
)


@dataclass(frozen=True)
class HotCoreOperatingPoint:
    mass_flow_kg_s: float
    tank_temperature_k: float
    tank_pressure_pa: float
    pump_discharge_pressure_pa: float
    pump_discharge_temperature_k: float
    pump_shaft_power_w: float
    inlet_npsh_available_m: float
    core_inlet_pressure_pa: float
    bridge_temperature_k: float
    channel_inlet_static_temperature_k: float
    bridge_enthalpy_offset_j_kg: float
    bridge_cp_relative_jump: float
    bridge_density_relative_jump: float
    core_outlet_pressure_pa: float
    core_outlet_static_temperature_k: float
    maximum_channel_mach: float
    friction_pressure_loss_pa: float
    acceleration_pressure_loss_pa: float
    chamber_pressure_pa: float
    chamber_temperature_k: float
    preheater_power_w: float
    core_heating_power_w: float
    nozzle_mass_flow_residual_kg_s: float
    overall_energy_residual_w: float
    pressure_residual_pa: float
    max_channel_energy_residual_w: float
    max_channel_momentum_residual_pa: float
    nozzle: HydrogenNozzle


def _pump_and_preheater(
    config: LH2Cycle, mdot: float, eos: HydrogenEOS,
) -> tuple[float,float,float,float,float,float,float]:
    """Liquid pump and preheater, all enthalpies in ONE CoolProp reference.

    Returns pump pressure, pump temperature, shaft power, preheater
    pressure, bridge T, bridge CoolProp h and tank h.
    """
    t1,p1=config.tank_temperature_k,config.tank_pressure_pa
    psat=eos.get("P","T",t1,"Q",0.)
    if p1 <= psat:
        raise CryogenicCycleError("tank hydrogen is not subcooled LH2")
    rho=eos.tp("Dmass",t1,p1)
    if rho <= 20:
        raise CryogenicCycleError("tank fluid is not dense liquid LH2")
    head=config.pump_shutoff_pressure_rise_pa-config.pump_head_quadratic_pa_s2_kg2*mdot**2
    if head <= 0:
        raise CryogenicCycleError("pump pressure head exhausted")
    p2=p1+head
    h1=eos.tp("Hmass",t1,p1)
    s1=eos.tp("Smass",t1,p1)
    h2s=eos.sp("Hmass",s1,p2)
    if h2s <= h1:
        raise CryogenicCycleError("isentropic pump enthalpy rise nonpositive")
    h2=h1+(h2s-h1)/config.pump_isentropic_efficiency
    t2=eos.hp("T",h2,p2)
    rho2=eos.hp("Dmass",h2,p2)
    if t2>=34 or rho2 <= 20:
        raise CryogenicCycleError("LH2 pump no longer discharges dense liquid")
    p3=p2-config.feed_loss_quadratic_pa_s2_kg2*mdot**2
    if p3 <= p1:
        raise CryogenicCycleError("feed system has insufficient pressure")
    h3=h2+config.preheater_thermal_power_w/mdot
    t3=eos.hp("T",h3,p3)
    if not T_MIN < t3 < min(900.,config.maximum_eos_temperature_k):
        raise CryogenicCycleError("preheater must supply 200K-900K hydrogen gas")
    if eos.tp("Dmass",t3,p3) <= 0 or eos.tp("Cpmass",t3,p3) <= 0:
        raise CryogenicCycleError("nonphysical preheater outlet")
    npsha=(p1-psat)/(rho*9.80665)
    return p2,t2,mdot*(h2-h1),p3,t3,h3,h1,npsha


def evaluate_hot_core_cycle(
    config: LH2Cycle, mass_flow_kg_s: float,
    *, eos: HydrogenEOS | None = None,
    calculate_full_nozzle: bool = False,
) -> tuple[HotCoreOperatingPoint | None,float]:
    """Evaluate one trial; full NASA-7 nozzle only on converged root.

    When calculate_full_nozzle=False, return (None, m - nozzle_demand)
    to avoid repeating a complete supersonic expansion for every
    mass-flow trial. This is still exact NASA-7 frozen choking.
    """
    config.validate()
    eos=eos or HydrogenEOS()
    m=float(mass_flow_kg_s)
    if not math.isfinite(m) or m<=0 or m>config.maximum_mass_flow_kg_s:
        raise ValueError("invalid candidate mass flow")
    p2,t2,wp,p3,t3,h3,h1,npsha=_pump_and_preheater(config,m,eos)

    # Reference bridge at the same T and P; do NOT equate the two EOS
    # absolute enthalpies directly (they use different zero references).
    nano3=hydrogen_state(t3,p3,mode="frozen")
    offset=h3-nano3.enthalpy_j_kg
    cp_cp=eos.tp("Cpmass",t3,p3)
    rho_cp=eos.tp("Dmass",t3,p3)
    cp_jump=(nano3.frozen_cp_j_kg_k-cp_cp)/cp_cp
    density_jump=(nano3.density_kg_m3-rho_cp)/rho_cp

    area=math.pi*config.channel_diameter_m**2/4
    branch_m=m/config.number_of_channels
    flux=branch_m/area
    t_channel=_static_channel_inlet_temperature(
        nano3.enthalpy_j_kg,p3,flux
    )
    heat_wall=np.asarray(config.channel_wall_power_w,dtype=float)
    heat_direct=np.asarray(config.channel_direct_hydrogen_power_w,dtype=float)
    channel=solve_fuel_channel_compressible(
        heat_wall,np.asarray(config.z_edges_m,dtype=float),
        config.number_of_channels,branch_m,t_channel,p3,
        config.channel_diameter_m,
        roughness_m=config.channel_roughness_m,
        axial_direct_coolant_power_w=heat_direct,
        max_mach=config.allowable_mach,
    )
    pe=channel.outlet_pressure_pa
    te=channel.outlet_temperature_k
    state_e=hydrogen_state(te,pe,mode="frozen")
    ve=flux/state_e.density_kg_m3
    h0_nasa=state_e.enthalpy_j_kg+0.5*ve**2
    pc=pe-config.chamber_loss_quadratic_pa_s2_kg2*m*m
    if pc<=0:
        raise CryogenicCycleError("chamber pressure nonpositive")
    tc=_temperature_at_enthalpy(h0_nasa,pc)
    if not T_MIN <= tc <= T_MAX:
        raise CryogenicCycleError("hot-core chamber outside NASA7 domain")
    # Exact frozen-H2 isentropic throat flow, not local constant gamma.
    gstar=choked_hydrogen_mass_flux(pc,tc,mode="frozen")
    nozzle_flow=gstar*config.throat_area_m2
    residual=m-nozzle_flow
    if not calculate_full_nozzle:
        return None,residual

    nozzle=hydrogen_nozzle(
        pc,tc,config.throat_area_m2,config.nozzle_expansion_ratio,
        mode="frozen",nozzle_efficiency=config.nozzle_efficiency,
        ambient_pressure_pa=config.ambient_pressure_pa,
    )
    qcore=float(heat_wall.sum()+heat_direct.sum())
    h_chamber_engine=h0_nasa+offset
    # Global first law includes shaft-work input. Exactly one bridge
    # offset appears on each side of the downstream total-h equation.
    energy=m*(h_chamber_engine-h1)-(
        wp+config.preheater_thermal_power_w+qcore
    )
    friction=float(channel.friction_pressure_loss_pa.sum())
    acceleration=float(channel.acceleration_pressure_loss_pa.sum())
    pressure=(p2-pc)-(
        config.feed_loss_quadratic_pa_s2_kg2*m*m+
        friction+acceleration+
        config.chamber_loss_quadratic_pa_s2_kg2*m*m
    )
    point=HotCoreOperatingPoint(
        mass_flow_kg_s=m,
        tank_temperature_k=config.tank_temperature_k,
        tank_pressure_pa=config.tank_pressure_pa,
        pump_discharge_pressure_pa=p2,
        pump_discharge_temperature_k=t2,
        pump_shaft_power_w=wp,
        inlet_npsh_available_m=npsha,
        core_inlet_pressure_pa=p3,
        bridge_temperature_k=t3,
        channel_inlet_static_temperature_k=t_channel,
        bridge_enthalpy_offset_j_kg=offset,
        bridge_cp_relative_jump=cp_jump,
        bridge_density_relative_jump=density_jump,
        core_outlet_pressure_pa=pe,
        core_outlet_static_temperature_k=te,
        maximum_channel_mach=float(max(channel.mach_number)),
        friction_pressure_loss_pa=friction,
        acceleration_pressure_loss_pa=acceleration,
        chamber_pressure_pa=pc,
        chamber_temperature_k=tc,
        preheater_power_w=config.preheater_thermal_power_w,
        core_heating_power_w=qcore,
        nozzle_mass_flow_residual_kg_s=m-nozzle.mass_flow_kg_s,
        overall_energy_residual_w=energy,
        pressure_residual_pa=pressure,
        max_channel_energy_residual_w=float(max(abs(channel.energy_residual_w))),
        max_channel_momentum_residual_pa=float(max(abs(channel.momentum_residual_pa))),
        nozzle=nozzle,
    )
    return point,point.nozzle_mass_flow_residual_kg_s


def solve_hot_core_cycle(
    config: LH2Cycle,
    *, eos: HydrogenEOS | None = None, bracket_samples: int = 38,
    relative_mass_tolerance: float = 2e-6,
) -> HotCoreOperatingPoint:
    """Solve pump-head / gas-channel / exact frozen-nozzle mass closure."""
    config.validate()
    eos=eos or HydrogenEOS()
    if not 16<=bracket_samples<=500:
        raise ValueError("bracket_samples must lie in [16,500]")
    if not 0<relative_mass_tolerance<1e-3:
        raise ValueError("invalid mass tolerance")
    upper=config.maximum_mass_flow_kg_s
    k=config.pump_head_quadratic_pa_s2_kg2
    if k>0:
        upper=min(upper,math.sqrt(config.pump_shutoff_pressure_rise_pa/k)*0.999)
    lower=max(upper*0.02,1e-6)
    prev=None
    bracket=None
    for m in np.geomspace(lower,upper,bracket_samples):
        try:
            _,fx=evaluate_hot_core_cycle(config,float(m),eos=eos)
        except (ValueError,RuntimeError,OverflowError):
            continue
        if prev is not None and fx*prev[1] <= 0:
            bracket=(prev[0],float(m),prev[1],fx)
            break
        prev=(float(m),fx)
    if bracket is None:
        raise CryogenicCycleError(
            "no valid LH2 hot-core mass-flow root; verify preheater "
            "gas handoff, channel choking, pump curve and NASA7 T bounds"
        )
    lo,hi,flo,fhi=bracket
    for _ in range(65):
        middle=(lo+hi)/2.
        _,fm=evaluate_hot_core_cycle(config,middle,eos=eos)
        if abs(fm)<=relative_mass_tolerance*middle*0.1:
            break
        if (fm>0)==(flo>0):
            lo,flo=middle,fm
        else:
            hi,fhi=middle,fm
    point,err=evaluate_hot_core_cycle(
        config,middle,eos=eos,calculate_full_nozzle=True
    )
    if point is None or abs(err)>relative_mass_tolerance*point.mass_flow_kg_s:
        raise CryogenicCycleError("hot-core nozzle/feed mass flow did not converge")
    scale=max(1.,point.pump_shaft_power_w+
              point.preheater_power_w+point.core_heating_power_w)
    if abs(point.overall_energy_residual_w)>1e-7*scale:
        raise RuntimeError("cryogenic-to-hot-core enthalpy audit FAILED")
    if abs(point.pressure_residual_pa)>1e-8*max(1.,point.pump_discharge_pressure_pa):
        raise RuntimeError("hot-core momentum audit FAILED")
    if point.maximum_channel_mach >= config.allowable_mach:
        raise RuntimeError("hot-core violated specified Mach gate")
    return point
