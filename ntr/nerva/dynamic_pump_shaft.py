"""Dynamic pump flow/head and finite rotor energy for NERVA reduced engine.

Uses the already-established *synthetic* LH2 pump characteristic,
  Delta_p_pump = Delta_p_shutoff*(omega/omega_ref)^2 - K_pump*m_dot^2
  p_pump_out - K_feed*m_dot^2 = p_live_upstream_header .
At every acoustic timestep the live H2 pump-manifold pressure and
rotor speed determine m_dot, shaft power and delivered liquid enthalpy.
True CoolProp isentropic h(p,s) and the specified pump efficiency
determine liquid heating. The net shaft power updates rotating kinetic
energy explicitly:
  E_rot,new - E_rot,old = dt*(P_turbine,reference - P_pump)
  E_rot = 0.5*J*omega^2 .
The same P_pump is exactly added to delivered H2 enthalpy, so
combined fluid + rotor energy closes against tank enthalpy,
reference turbine drive, nuclear tie and nozzle cooling heat,
and reference engine discharge.

This is NOT a calibrated NERVA pump speed curve, turbine/speed map,
real NPSH transient, control system, gas-generator interaction, or
full changing nozzle/core flow demand. Turbine delivered power is a
fixed external drive from a solved *steady* reference point. No
fissile/criticality or reactor design optimization is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .dual_header_tie import (
    DualHeaderTieState, DualHeaderTieStep, advance_dual_header_tie,
)
from .dynamic_header_tie import DynamicHeaderParameters, header_properties
from .distributed_tie_euler import DistributedTieParameters
from .lh2_feed_cycle import LH2Cycle, HydrogenEOS
from .regenerative_cooling import CoolingNetwork
from .resolved_transient import ResolvedHeating
from .turbine_bleed_cycle import ShaftClosedPoint


@dataclass(frozen=True)
class DynamicPumpParameters:
    reference_angular_speed_rad_s: float
    rotor_inertia_kg_m2: float
    minimum_speed_ratio: float = 0.8
    maximum_speed_ratio: float = 1.2

    def validate(self) -> None:
        for key in ("reference_angular_speed_rad_s", "rotor_inertia_kg_m2",
                    "minimum_speed_ratio", "maximum_speed_ratio"):
            x=getattr(self,key)
            if not math.isfinite(x) or x<=0:
                raise ValueError(f"pump parameter {key} must be positive finite")
        if not (0<self.minimum_speed_ratio<1<self.maximum_speed_ratio<=3):
            raise ValueError("invalid dynamic pump speed domain")


@dataclass(frozen=True)
class DynamicPumpState:
    rotor_angular_speed_rad_s: float
    hydraulic: DualHeaderTieState


@dataclass(frozen=True)
class DynamicPumpDelivery:
    mass_flow_kg_s: float
    shaft_power_w: float
    delivered_stagnation_enthalpy_j_kg: float
    tank_enthalpy_j_kg: float
    pump_stage_pressure_pa: float
    manifold_pressure_pa: float
    pump_head_rise_pa: float
    feed_line_pressure_loss_pa: float
    manifold_pressure_residual_pa: float
    pump_enthalpy_power_residual_w: float
    available_npsh_m: float


@dataclass(frozen=True)
class DynamicPumpStep:
    state: DynamicPumpState
    fluid: DualHeaderTieStep
    pump: DynamicPumpDelivery
    reference_turbine_shaft_power_w: float
    rotor_power_storage_rate_w: float
    rotor_energy_balance_residual_j: float
    total_fluid_rotor_energy_balance_residual_j: float
    rotor_initial_energy_j: float
    rotor_final_energy_j: float
    pump_mass_flow_follows_pressure_speed: bool = True
    rotor_inertia_dynamic: bool = True
    turbine_shaft_power_reference_fixed: bool = True
    core_nozzle_mass_demand_reference_fixed: bool = True
    whole_engine_cycle_converged: bool = False
    historical_nasa_pump_map_validated: bool = False
    turbine_shaft_power_dynamic: bool = False


def pump_delivery_at_state(
    core: LH2Cycle, parameters: DynamicPumpParameters,
    angular_speed_rad_s: float, manifold_pressure_pa: float,
    *, eos: HydrogenEOS | None = None,
) -> DynamicPumpDelivery:
    """Evaluate positive-flow branch of the existing pump/feed map.

    The physical mass flow is the algebraic root of the pump curve
    and the live manifold pressure, not the former fixed shaft
    reference mass flow. No continuation past shutoff, 2-phase
    pump discharge, unsupported speed or flow envelope.
    """
    core.validate()
    parameters.validate()
    if not math.isfinite(angular_speed_rad_s):
        raise ValueError("pump rotor speed nonfinite")
    ratio=angular_speed_rad_s/parameters.reference_angular_speed_rad_s
    if not parameters.minimum_speed_ratio<=ratio<=parameters.maximum_speed_ratio:
        raise ValueError("pump rotor speed outside declared affinity-map range")
    if not math.isfinite(manifold_pressure_pa) or manifold_pressure_pa<=0:
        raise ValueError("invalid live pump-manifold pressure")
    p_tank=core.tank_pressure_pa
    t_tank=core.tank_temperature_k
    k_pump=core.pump_head_quadratic_pa_s2_kg2
    k_feed=core.feed_loss_quadratic_pa_s2_kg2
    k_total=k_pump+k_feed
    if not math.isfinite(k_total) or k_total<=0:
        raise ValueError("pump plus feed hydraulic resistance must be positive")
    shutoff=core.pump_shutoff_pressure_rise_pa*ratio**2
    surplus=p_tank+shutoff-manifold_pressure_pa
    if surplus<=0:
        raise ValueError("pump cannot deliver positive mass to elevated header")
    mdot=math.sqrt(surplus/k_total)
    if not math.isfinite(mdot) or not 0<mdot<=core.maximum_mass_flow_kg_s:
        raise ValueError("pump flow exceeds declared LH2 map limit")
    head=shutoff-k_pump*mdot**2
    feed_loss=k_feed*mdot**2
    p_stage=p_tank+head
    if head<=0 or p_stage-feed_loss<=p_tank:
        raise ValueError("pump cannot maintain admissible positive stage head")
    residual=p_stage-feed_loss-manifold_pressure_pa
    tol=1e-8*max(1.,p_stage)
    if abs(residual)>tol:
        raise RuntimeError("pump/feed/live-manifold pressure closure failed")
    eos=eos or HydrogenEOS()
    pvap=eos.get("P","T",t_tank,"Q",0.)
    if p_tank<=pvap:
        raise ValueError("tank H2 not subcooled; no pump cavitation closure")
    rho_tank=eos.tp("Dmass",t_tank,p_tank)
    if rho_tank<=20:
        raise ValueError("pump inlet must be dense liquid H2")
    h_tank=eos.tp("Hmass",t_tank,p_tank)
    s_tank=eos.tp("Smass",t_tank,p_tank)
    h_isen=eos.sp("Hmass",s_tank,p_stage)
    if h_isen<=h_tank:
        raise ValueError("isentropic LH2 stage has no positive work")
    h_stage=h_tank+(h_isen-h_tank)/core.pump_isentropic_efficiency
    t_stage=eos.hp("T",h_stage,p_stage)
    rho_stage=eos.hp("Dmass",h_stage,p_stage)
    # Feed-line pressure loss is adiabatic and does not add fictitious
    # thermodynamic heat to the CoolProp stagnation enthalpy ledger.
    t_manifold=eos.hp("T",h_stage,manifold_pressure_pa)
    rho_manifold=eos.hp("Dmass",h_stage,manifold_pressure_pa)
    if (t_stage>=34 or rho_stage<=20 or t_manifold>=34
            or rho_manifold<=20):
        raise ValueError("LH2 pump/feed outflow not admissible dense liquid")
    power=mdot*(h_stage-h_tank)
    npsh=(p_tank-pvap)/(rho_tank*9.80665)
    if power<=0 or npsh<=0:
        raise ValueError("pump work/NPSH outside physical operating range")
    return DynamicPumpDelivery(
        mass_flow_kg_s=mdot,
        shaft_power_w=power,
        delivered_stagnation_enthalpy_j_kg=h_stage,
        tank_enthalpy_j_kg=h_tank,
        pump_stage_pressure_pa=p_stage,
        manifold_pressure_pa=manifold_pressure_pa,
        pump_head_rise_pa=head,
        feed_line_pressure_loss_pa=feed_loss,
        manifold_pressure_residual_pa=residual,
        pump_enthalpy_power_residual_w=power-mdot*(h_stage-h_tank),
        available_npsh_m=npsh,
    )


def advance_dynamic_pump_tie(
    heating: ResolvedHeating,
    old: DynamicPumpState,
    core: LH2Cycle,
    pump_parameters: DynamicPumpParameters,
    distributed_parameters: DistributedTieParameters,
    upstream_parameters: DynamicHeaderParameters,
    downstream_parameters: DynamicHeaderParameters,
    network: CoolingNetwork,
    reference_shaft: ShaftClosedPoint,
    dt_s: float,
    *, eos: HydrogenEOS | None = None,
    turbine_shaft_power_override_w: float | None = None,
) -> DynamicPumpStep:
    """Couple real-fluid pump map, conservative headers, Euler tie and rotor."""
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("positive pump/shaft timestep required")
    pump_parameters.validate()
    eos=eos or HydrogenEOS()
    p_manifold,_,_,_=header_properties(
        old.hydraulic.pump,upstream_parameters,eos=eos,
    )
    delivery=pump_delivery_at_state(
        core,pump_parameters,old.rotor_angular_speed_rad_s,
        p_manifold,eos=eos,
    )
    fluid=advance_dual_header_tie(
        heating,old.hydraulic,distributed_parameters,
        upstream_parameters,downstream_parameters,
        network,reference_shaft,dt_s,eos=eos,
        pump_delivery_override=(
            delivery.mass_flow_kg_s,
            delivery.delivered_stagnation_enthalpy_j_kg,
        ),
    )
    turbine_drive=(
        reference_shaft.turbine_shaft_delivered_power_w
        if turbine_shaft_power_override_w is None
        else turbine_shaft_power_override_w
    )
    if not math.isfinite(turbine_drive) or turbine_drive<0:
        raise ValueError("invalid fixed reference turbine delivered power")
    initial_E=0.5*pump_parameters.rotor_inertia_kg_m2*(
        old.rotor_angular_speed_rad_s**2
    )
    delta_E=dt_s*(turbine_drive-delivery.shaft_power_w)
    final_E=initial_E+delta_E
    if final_E<=0 or not math.isfinite(final_E):
        raise ValueError("rotor stalled or energy invalid")
    speed_new=math.sqrt(
        2*final_E/pump_parameters.rotor_inertia_kg_m2
    )
    ratio_new=speed_new/pump_parameters.reference_angular_speed_rad_s
    if not pump_parameters.minimum_speed_ratio<=ratio_new<=pump_parameters.maximum_speed_ratio:
        raise ValueError("rotor exits supported speed range")
    rotor_error=final_E-initial_E-delta_E
    # Pump work is transferred into H2 via its actual specific
    # enthalpy rise: same quantity taken from the rotor shaft.
    fluid_plus_rotor_residual=(
        fluid.combined_system_total_energy_residual_j+
        rotor_error+
        dt_s*(
            delivery.mass_flow_kg_s*(
                delivery.delivered_stagnation_enthalpy_j_kg-
                delivery.tank_enthalpy_j_kg
            )-delivery.shaft_power_w
        )
    )
    eps=np.finfo(float).eps
    roundoff=(1e-6+16*eps*(
        initial_E+final_E+
        abs(old.hydraulic.pump.internal_energy_j)+
        abs(fluid.state.pump.internal_energy_j)
    ))
    if abs(rotor_error)>roundoff or abs(fluid_plus_rotor_residual)>roundoff:
        raise RuntimeError(
            "pump H2/rotor/tie/header total-energy ledger failed"
        )
    return DynamicPumpStep(
        state=DynamicPumpState(
            rotor_angular_speed_rad_s=speed_new,
            hydraulic=fluid.state,
        ),
        fluid=fluid,
        pump=delivery,
        reference_turbine_shaft_power_w=turbine_drive,
        rotor_power_storage_rate_w=(final_E-initial_E)/dt_s,
        rotor_energy_balance_residual_j=rotor_error,
        total_fluid_rotor_energy_balance_residual_j=fluid_plus_rotor_residual,
        rotor_initial_energy_j=initial_E,
        rotor_final_energy_j=final_E,
        turbine_shaft_power_reference_fixed=(
            turbine_shaft_power_override_w is None
        ),
        turbine_shaft_power_dynamic=(
            turbine_shaft_power_override_w is not None
        ),
    )
