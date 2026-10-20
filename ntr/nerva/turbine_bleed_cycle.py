"""Thermodynamically closed LH2 pump / hot-H2 turbine bleed / core / nozzle.

The cryogenic pump raises pressure of ALL feed hydrogen. A prescribed
preheater vaporizes/warms it; a portion is then diverted through a
warm-gas turbine, whose shaft output (after mechanical losses) EXACTLY
drives the LH2 pump. The remaining H2 enters the NASA7 hot fuel channel
and the choked main nozzle. Both total feed flow and turbine bleed are
SOLVED, rather than separately imposed.

    W_pump = m_total * (h2-h1)
    w_turb = eta_t * [h3-h4s(s3,p_exhaust)]
    m_bleed = W_pump / (eta_shaft*w_turb)
    m_core = m_total - m_bleed
    m_core = A_throat * G_star(Pc,Tc)
    m_total = m_core + m_bleed

System first law (same CoolProp enthalpy zero matched to hot NASA7):
    m_core*h_chamber + m_bleed*h_turb_exhaust + W_shaft_loss
       - (m_total*h_tank + Q_preheater + Q_core) = 0
    W_shaft_loss = W_turb - W_pump.

This is a reduced-order shaft-coupling PHYSICS MODEL, not a genuine
full NERVA engine cycle: the preheater is a lumped supplied heat source
rather than a routed nozzle-jacket/reflector/tie return; turbine maps,
blade cooling, bypass valves, finite-rate chemistry, rotor dynamics and
turbine exhaust thrust are omitted. Effective engine Isp excludes any
uncomputed turbine-exhaust thrust, and is main-nozzle thrust divided by
TOTAL tank consumption, not core-only nozzle Isp.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .compressible_channel import (
    _temperature_from_total_enthalpy, solve_fuel_channel_compressible
)
from .hot_lh2_cycle import _pump_and_preheater
from .lh2_feed_cycle import CryogenicCycleError, HydrogenEOS, LH2Cycle
from .regenerative_cooling import (CoolingNetwork, CoolingNetworkPoint, pump_and_regenerative_cooling)
from .nasa_hydrogen import (
    G0, HydrogenNozzle, hydrogen_state, hydrogen_nozzle,
    choked_hydrogen_mass_flux
)


@dataclass(frozen=True)
class TurbineBleed:
    exhaust_pressure_pa: float
    isentropic_efficiency: float
    shaft_mechanical_efficiency: float
    maximum_bleed_fraction: float = 0.50

    def validate(self) -> None:
        if not math.isfinite(self.exhaust_pressure_pa) or self.exhaust_pressure_pa<=0:
            raise ValueError("turbine exhaust pressure must be positive")
        if (not math.isfinite(self.isentropic_efficiency)
                or not 0<self.isentropic_efficiency<=1):
            raise ValueError("turbine isentropic efficiency must lie in (0,1]")
        if (not math.isfinite(self.shaft_mechanical_efficiency)
                or not 0<self.shaft_mechanical_efficiency<=1):
            raise ValueError("mechanical efficiency must lie in (0,1]")
        if (not math.isfinite(self.maximum_bleed_fraction)
                or not 0<self.maximum_bleed_fraction<1):
            raise ValueError("maximum bleed fraction must lie in (0,1)")


@dataclass(frozen=True)
class ShaftClosedPoint:
    tank_mass_flow_kg_s: float
    core_mass_flow_kg_s: float
    turbine_bleed_mass_flow_kg_s: float
    turbine_bleed_fraction: float
    pump_discharge_pressure_pa: float
    pump_discharge_temperature_k: float
    core_inlet_pressure_pa: float
    warm_header_temperature_k: float
    warm_header_enthalpy_j_kg: float
    turbine_exhaust_pressure_pa: float
    turbine_exhaust_temperature_k: float
    turbine_exhaust_enthalpy_j_kg: float
    turbine_isentropic_specific_work_j_kg: float
    turbine_actual_specific_work_j_kg: float
    turbine_gas_power_w: float
    turbine_shaft_delivered_power_w: float
    pump_required_power_w: float
    mechanical_loss_w: float
    chamber_pressure_pa: float
    chamber_temperature_k: float
    maximum_core_mach: float
    core_friction_pressure_loss_pa: float
    core_acceleration_pressure_loss_pa: float
    core_heating_power_w: float
    preheater_power_w: float
    nozzle_mass_flow_residual_kg_s: float
    shaft_power_residual_w: float
    total_cycle_energy_residual_w: float
    channel_pressure_residual_pa: float
    main_nozzle_thrust_n: float
    main_nozzle_isp_s: float
    effective_isp_excluding_bleed_thrust_s: float
    chemistry_mode: str
    chamber_dissociation_fraction: float
    nozzle: HydrogenNozzle
    regenerative_cooling: CoolingNetworkPoint | None = None


def evaluate_turbine_bleed(
    config: LH2Cycle, bleed: TurbineBleed, tank_flow_kg_s: float,
    *, eos: HydrogenEOS | None = None, chemistry_mode: str = "frozen",
    full_nozzle: bool = False,
    cooling_network: CoolingNetwork | None = None,
) -> tuple[ShaftClosedPoint | None, float]:
    """Solve shaft bleed analytically at prescribed total mass, close nozzle.

    Return a cheap (None, mass residual) during root search, complete
    result only for the final root. The bleed path is separately
    exhausted and NEVER silently added to main-nozzle thrust.
    """
    config.validate()
    bleed.validate()
    if chemistry_mode not in ("frozen","equilibrium"):
        raise ValueError("invalid H2 chemistry mode")
    eos=eos or HydrogenEOS()
    m=float(tank_flow_kg_s)
    if not math.isfinite(m) or not 0<m<=config.maximum_mass_flow_kg_s:
        raise ValueError("invalid trial mass flow")
    cooling_result = None
    if cooling_network is None:
        p2,t2,wp,p3,t3,h3,h1,npsha=_pump_and_preheater(config,m,eos)
        cooling_pressure_drop = 0.
    else:
        conditioned=pump_and_regenerative_cooling(
            config,cooling_network,m,eos=eos
        )
        p2=conditioned.pump_pressure_pa
        t2=conditioned.pump_temperature_k
        wp=conditioned.pump_shaft_power_w
        p3=conditioned.header_pressure_pa
        t3=conditioned.header_temperature_k
        h3=conditioned.header_stagnation_enthalpy_j_kg
        h1=conditioned.tank_enthalpy_j_kg
        npsha=conditioned.inlet_npsh_available_m
        cooling_result=conditioned.cooling
        cooling_pressure_drop=(
            p2-config.feed_loss_quadratic_pa_s2_kg2*m*m-p3
        )

    if bleed.exhaust_pressure_pa >= p3:
        raise CryogenicCycleError("turbine exhaust pressure must be below gas header")
    s3=eos.tp("Smass",t3,p3)
    h4s=eos.sp("Hmass",s3,bleed.exhaust_pressure_pa)
    w_ideal=h3-h4s
    if not math.isfinite(w_ideal) or w_ideal<=0:
        raise CryogenicCycleError("warm H2 turbine has no positive isentropic work")
    w_actual=bleed.isentropic_efficiency*w_ideal
    h4=h3-w_actual
    t4=eos.hp("T",h4,bleed.exhaust_pressure_pa)
    # This turbine model applies only to single-phase H2 gas.
    # An exit at low T could require liquid/mixture turbine physics.
    if not 120 < t4 < config.maximum_eos_temperature_k:
        raise CryogenicCycleError("turbine H2 outlet outside warm gas domain")
    if bleed.exhaust_pressure_pa < 1.3e6:
        h_sat=eos.get("Hmass","P",bleed.exhaust_pressure_pa,"Q",1.)
        if h4 <= h_sat:
            raise CryogenicCycleError("turbine outlet would be liquid/two-phase")

    m_bleed=wp/(bleed.shaft_mechanical_efficiency*w_actual)
    beta=m_bleed/m
    if not 0<beta<=bleed.maximum_bleed_fraction:
        raise CryogenicCycleError(
            "pump requires excessive turbine bleed; shaft cycle not feasible"
        )
    m_core=m-m_bleed

    # Hot-core enthalpy offset: CoolProp h3 and NASA7 h(T3,p3)
    # have different absolute reference conventions.
    header=hydrogen_state(t3,p3,mode=chemistry_mode)
    h_offset=h3-header.enthalpy_j_kg
    diameter=config.channel_diameter_m
    area=math.pi*diameter*diameter/4.
    branch_m=m_core/config.number_of_channels
    G=branch_m/area
    channel_t=_temperature_from_total_enthalpy(
        header.enthalpy_j_kg,p3,G,t3,chemistry_mode
    )
    wall=np.asarray(config.channel_wall_power_w,dtype=float)
    direct=np.asarray(config.channel_direct_hydrogen_power_w,dtype=float)
    ch=solve_fuel_channel_compressible(
        wall,np.asarray(config.z_edges_m,dtype=float),
        config.number_of_channels,branch_m,channel_t,p3,diameter,
        roughness_m=config.channel_roughness_m,
        axial_direct_coolant_power_w=direct,
        max_mach=config.allowable_mach,
        chemistry_mode=chemistry_mode,
    )
    p_exit=ch.outlet_pressure_pa
    exit_state=hydrogen_state(ch.outlet_temperature_k,p_exit,mode=chemistry_mode)
    vel=G/exit_state.density_kg_m3
    h0_nasa=exit_state.enthalpy_j_kg+0.5*vel**2
    chamber_loss=config.chamber_loss_quadratic_pa_s2_kg2*m_core*m_core
    pc=p_exit-chamber_loss
    if pc<=0:
        raise CryogenicCycleError("no positive core chamber pressure")
    tc=_temperature_from_total_enthalpy(
        h0_nasa,pc,0.,ch.outlet_temperature_k,chemistry_mode
    )
    gstar=choked_hydrogen_mass_flux(pc,tc,mode=chemistry_mode)
    residual=m_core-config.throat_area_m2*gstar
    if not full_nozzle:
        return None,residual

    nozzle=hydrogen_nozzle(
        pc,tc,config.throat_area_m2,config.nozzle_expansion_ratio,
        mode=chemistry_mode,nozzle_efficiency=config.nozzle_efficiency,
        ambient_pressure_pa=config.ambient_pressure_pa,
    )
    turbine_gas=m_bleed*w_actual
    shaft_delivered=bleed.shaft_mechanical_efficiency*turbine_gas
    shaft_loss=turbine_gas-shaft_delivered
    qcore=float(wall.sum()+direct.sum())
    hchamber=h0_nasa+h_offset
    # System boundary contains pump, turbine, shaft, header and core.
    # Pump power is INTERNAL; only mechanical loss leaves this boundary.
    global_energy=(
        m_core*hchamber+m_bleed*h4+shaft_loss
        -(m*h1+config.preheater_thermal_power_w+qcore)
    )
    friction=float(np.sum(ch.friction_pressure_loss_pa))
    acceleration=float(np.sum(ch.acceleration_pressure_loss_pa))
    dp_check=p2-pc-(
        config.feed_loss_quadratic_pa_s2_kg2*m*m+
        cooling_pressure_drop+friction+acceleration+chamber_loss
    )
    pt=ShaftClosedPoint(
        tank_mass_flow_kg_s=m,
        core_mass_flow_kg_s=m_core,
        turbine_bleed_mass_flow_kg_s=m_bleed,
        turbine_bleed_fraction=beta,
        pump_discharge_pressure_pa=p2,
        pump_discharge_temperature_k=t2,
        core_inlet_pressure_pa=p3,
        warm_header_temperature_k=t3,
        warm_header_enthalpy_j_kg=h3,
        turbine_exhaust_pressure_pa=bleed.exhaust_pressure_pa,
        turbine_exhaust_temperature_k=t4,
        turbine_exhaust_enthalpy_j_kg=h4,
        turbine_isentropic_specific_work_j_kg=w_ideal,
        turbine_actual_specific_work_j_kg=w_actual,
        turbine_gas_power_w=turbine_gas,
        turbine_shaft_delivered_power_w=shaft_delivered,
        pump_required_power_w=wp,
        mechanical_loss_w=shaft_loss,
        chamber_pressure_pa=pc,
        chamber_temperature_k=tc,
        maximum_core_mach=float(max(ch.mach_number)),
        core_friction_pressure_loss_pa=friction,
        core_acceleration_pressure_loss_pa=acceleration,
        core_heating_power_w=qcore,
        preheater_power_w=config.preheater_thermal_power_w,
        nozzle_mass_flow_residual_kg_s=m_core-nozzle.mass_flow_kg_s,
        shaft_power_residual_w=shaft_delivered-wp,
        total_cycle_energy_residual_w=global_energy,
        channel_pressure_residual_pa=dp_check,
        main_nozzle_thrust_n=nozzle.thrust_n,
        main_nozzle_isp_s=nozzle.isp_s,
        effective_isp_excluding_bleed_thrust_s=nozzle.thrust_n/(m*G0),
        chemistry_mode=chemistry_mode,
        chamber_dissociation_fraction=nozzle.chamber.dissociation_fraction,
        nozzle=nozzle,
        regenerative_cooling=cooling_result,
    )
    return pt,pt.nozzle_mass_flow_residual_kg_s


def solve_turbine_bleed(
    config: LH2Cycle, bleed: TurbineBleed, *,
    eos: HydrogenEOS | None = None, chemistry_mode: str = "frozen",
    bracket_samples: int = 45, mass_relative_tolerance: float = 2e-6,
    cooling_network: CoolingNetwork | None = None,
) -> ShaftClosedPoint:
    """Find joint shaft-drive bleed fraction and core/nozzle mass flow."""
    config.validate()
    bleed.validate()
    eos=eos or HydrogenEOS()
    if not 16<=bracket_samples<=500:
        raise ValueError("invalid sample count")
    if not 0<mass_relative_tolerance<1e-3:
        raise ValueError("invalid mass convergence threshold")
    limit=config.maximum_mass_flow_kg_s
    if config.pump_head_quadratic_pa_s2_kg2>0:
        limit=min(limit, math.sqrt(
            config.pump_shutoff_pressure_rise_pa/
            config.pump_head_quadratic_pa_s2_kg2
        )*0.999)
    prev=None
    bracket=None
    for mass in np.geomspace(max(limit*0.02,1e-6),limit,bracket_samples):
        try:
            _,f=evaluate_turbine_bleed(
                config,bleed,float(mass),eos=eos,chemistry_mode=chemistry_mode,
                cooling_network=cooling_network
            )
        except (ValueError,RuntimeError,OverflowError):
            continue
        if prev is not None and prev[1]*f <=0:
            bracket=(prev[0],float(mass),prev[1])
            break
        prev=(float(mass),f)
    if bracket is None:
        raise CryogenicCycleError(
            "no subsonic shaft-driven NERVA root for this pressure source, "
            "turbine exit and preheater heating; do not infer an operating point"
        )
    lo,hi,flo=bracket
    mid=(lo+hi)/2
    for _ in range(62):
        mid=(lo+hi)/2
        _,f=evaluate_turbine_bleed(
            config,bleed,mid,eos=eos,chemistry_mode=chemistry_mode,
            cooling_network=cooling_network
        )
        if abs(f)<=mass_relative_tolerance*mid*0.10:
            break
        if (f>0)==(flo>0):
            lo,flo=mid,f
        else:
            hi=mid
    pt,error=evaluate_turbine_bleed(
        config,bleed,mid,eos=eos,chemistry_mode=chemistry_mode,full_nozzle=True,
        cooling_network=cooling_network
    )
    if pt is None or abs(error)>mass_relative_tolerance*pt.core_mass_flow_kg_s:
        raise CryogenicCycleError("shaft/nozzle mass closure did not converge")
    heat_scale=max(1.,pt.preheater_power_w+pt.core_heating_power_w)
    if abs(pt.total_cycle_energy_residual_w)>1e-8*heat_scale:
        raise RuntimeError("global turbine bleed-cycle enthalpy closure FAILED")
    if abs(pt.shaft_power_residual_w)>1e-9*max(1.,pt.pump_required_power_w):
        raise RuntimeError("turbine-pump shaft power balance FAILED")
    if abs(pt.channel_pressure_residual_pa)>1e-8*max(1.,pt.pump_discharge_pressure_pa):
        raise RuntimeError("core pressure balance FAILED")
    return pt
