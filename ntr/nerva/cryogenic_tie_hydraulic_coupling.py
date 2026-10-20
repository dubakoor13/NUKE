"""Conservative cryogenic tie energy + regenerative HYDRAULIC fixed point.

The tie solver is transient and stores fluid enthalpy; the existing
regenerative-network solver is steady and uses total-enthalpy rise.
To avoid counting transient hydrogen inventory as coolant heat reaching
the downstream mixing header, the hydraulic leg's EFFECTIVE advected
source is

 Q_advected,leg = Q_wall,leg + Q_direct_H2,leg - dE_H2,leg/dt.

This guarantees Q_nuclear,tie = Q_advected,supply + Q_advected,return
                            + dE_H2,tie/dt + dE_solid,tie/dt.

For each Picard trial, re-evaluate the real-fluid tie transient from the
SAME OLD STATE, then solve the hydraulic splitter with its two effective
leg sources. Return tie branch mass flow and representative pressure to
CoolProp and iterate. The engine-side nozzle/reflector heat sources and
the imposed pump outlet (mass, pressure, enthalpy) remain fixed.

Physical scope: network-closed branch split and lumped momentum residual,
one representative uniform tie fluid pressure per iteration, equilibrium
H2 EOS, fixed fluid inventory, fixed UA and cell solid capacities. It is
NOT spatial pressure momentum, two-phase slip/CHF, actual engine-pump or
shaft/nozzle closure, verified OpenMC transport, nor a NASA benchmark.
The real-fluid h and existing stagnation-h model use the same numerical
enthalpy at the boundary: kinetic/acceleration storage is NOT separately
modeled in the transient control volumes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .cryogenic_tie_enthalpy import (
    CryogenicTieParameters, CryogenicTieState, CryogenicTieStep,
    advance_cryogenic_tie,
)
from .lh2_feed_cycle import HydrogenEOS, LH2Cycle, CryogenicCycleError
from .regenerative_cooling import CoolingNetwork, CoolingNetworkPoint, solve_cooling_network
from .resolved_transient import ResolvedHeating


def pumped_regenerative_inlet(
    core: LH2Cycle, total_coolant_flow_kg_s: float,
    *, eos: HydrogenEOS | None = None,
) -> tuple[float,float]:
    """Return pressure and total H2 enthalpy after pump and feed loss.

    Uses the SAME isentropic-efficiency and shaft/head curve used by
    pump_and_regenerative_cooling, but does not prematurely solve the
    heating-dependent downstream splitter. Inputs are validated; this
    is a pump boundary evaluation, NOT a new pump operating-point solve.
    """
    core.validate()
    if not math.isfinite(total_coolant_flow_kg_s) or total_coolant_flow_kg_s<=0:
        raise ValueError("pump coolant flow must be positive finite")
    eos=eos or HydrogenEOS()
    m=total_coolant_flow_kg_s
    t,p=core.tank_temperature_k,core.tank_pressure_pa
    psat=eos.get("P","T",t,"Q",0.)
    if p<=psat or eos.tp("Dmass",t,p)<=20:
        raise CryogenicCycleError("pump inlet not subcooled dense H2")
    rise=core.pump_shutoff_pressure_rise_pa-core.pump_head_quadratic_pa_s2_kg2*m*m
    if rise<=0:
        raise CryogenicCycleError("pump head exhausted")
    p2=p+rise
    h1=eos.tp("Hmass",t,p)
    s1=eos.tp("Smass",t,p)
    h2s=eos.sp("Hmass",s1,p2)
    if h2s<=h1:
        raise CryogenicCycleError("pump isentropic work nonpositive")
    h2=h1+(h2s-h1)/core.pump_isentropic_efficiency
    if eos.hp("T",h2,p2)>=34 or eos.hp("Dmass",h2,p2)<=20:
        raise CryogenicCycleError("pump outlet not dense LH2")
    pin=p2-core.feed_loss_quadratic_pa_s2_kg2*m*m
    if pin<=p:
        raise CryogenicCycleError("feed loss eliminates pump pressure")
    return pin,h2


@dataclass(frozen=True)
class TieHydraulicIteration:
    iteration: int
    thermal_tie_flow_kg_s: float
    hydraulic_tie_flow_kg_s: float
    thermal_pressure_pa: float
    representative_hydraulic_pressure_pa: float
    relative_flow_residual: float
    relative_pressure_residual: float
    relative_heat_source_change: float
    tie_nuclear_deposition_w: float
    tie_advected_power_w: float
    hydrogen_storage_rate_w: float
    solid_storage_rate_w: float
    nuclear_energy_power_residual_w: float
    outlet_enthalpy_difference_j_kg: float


@dataclass(frozen=True)
class TieHydraulicCoupledResult:
    heating: ResolvedHeating
    tie_step: CryogenicTieStep
    cooling_point: CoolingNetworkPoint
    effective_cooling_network: CoolingNetwork
    iterations: int
    relative_flow_residual: float
    relative_pressure_residual: float
    relative_heat_source_change: float
    nuclear_energy_power_residual_w: float
    outlet_enthalpy_difference_j_kg: float
    diagnostics: tuple[TieHydraulicIteration,...]
    pressure_field_resolved: bool = False
    hydrogen_mass_inventory_updated: bool = False
    full_engine_cycle_converged: bool = False
    actual_openmc_transport_executed: bool = False


def couple_cryogenic_tie_hydraulics(
    heating: ResolvedHeating,
    old: CryogenicTieState,
    parameters: CryogenicTieParameters,
    cooling: CoolingNetwork,
    total_coolant_flow_kg_s: float,
    coolant_inlet_pressure_pa: float,
    coolant_inlet_enthalpy_j_kg: float,
    dt_s: float,
    *,
    nozzle_wall_heat_w: float,
    reflector_wall_heat_w: float,
    max_iterations: int = 15,
    relative_flow_tolerance: float = 0.002,
    relative_pressure_tolerance: float = 0.002,
    relative_source_tolerance: float = 0.002,
    relaxation: float = 0.6,
    eos: HydrogenEOS | None = None,
) -> TieHydraulicCoupledResult:
    heating.validate()
    parameters.validate()
    cooling.validate()
    for label,value in (
        ("total_coolant_flow_kg_s",total_coolant_flow_kg_s),
        ("coolant_inlet_pressure_pa",coolant_inlet_pressure_pa),
        ("dt_s",dt_s),
    ):
        if not math.isfinite(value) or value<=0:
            raise ValueError(f"{label} must be positive finite")
    if not math.isfinite(coolant_inlet_enthalpy_j_kg):
        raise ValueError("invalid coolant inlet enthalpy")
    for label,value in (
        ("nozzle_wall_heat_w",nozzle_wall_heat_w),
        ("reflector_wall_heat_w",reflector_wall_heat_w),
    ):
        if not math.isfinite(value) or value<0:
            raise ValueError(f"{label} must be nonnegative finite")
    if not isinstance(max_iterations,int) or max_iterations<2:
        raise ValueError("requires at least two thermal/hydraulic fixed point trials")
    for label,value in (
        ("relative_flow_tolerance",relative_flow_tolerance),
        ("relative_pressure_tolerance",relative_pressure_tolerance),
        ("relative_source_tolerance",relative_source_tolerance),
    ):
        if not math.isfinite(value) or not 0<value<=0.1:
            raise ValueError(f"invalid fixed-point {label}")
    if not math.isfinite(relaxation) or not 0<relaxation<=1:
        raise ValueError("invalid fixed-point relaxation")
    if heating.total_tie_flow_kg_s>=total_coolant_flow_kg_s:
        raise ValueError("tie flow cannot consume or exceed all parallel coolant")
    if parameters.pressure_pa>coolant_inlet_pressure_pa:
        raise ValueError("tie representative pressure exceeds pump inlet header")
    eos=eos or HydrogenEOS()
    # Validate real-fluid inlet against the actual pump/network state.
    actual_t=eos.hp("T",coolant_inlet_enthalpy_j_kg,coolant_inlet_pressure_pa)
    if not math.isfinite(actual_t) or actual_t<=0:
        raise ValueError("invalid real-fluid coolant inlet state")

    base_flow=heating.total_tie_flow_kg_s
    trial_flow=base_flow
    trial_p=parameters.pressure_pa
    last_sources=None
    trace=[]
    for iteration in range(1,max_iterations+1):
        trial_heating=replace(
            heating,
            tie_flow_kg_s=heating.tie_flow_kg_s*(trial_flow/base_flow),
        )
        trial_heating.validate()
        trial_params=replace(parameters,pressure_pa=trial_p)
        step=advance_cryogenic_tie(
            trial_heating,old,trial_params,
            coolant_inlet_enthalpy_j_kg,dt_s,eos=eos,
        )
        q_supply_real=float(
            np.sum(step.supply_wall_power_w)+
            np.sum(trial_heating.tie_supply_h2_w)
        )
        q_return_real=float(
            np.sum(step.return_wall_power_w)+
            np.sum(trial_heating.tie_return_h2_w)
        )
        supply_storage=(
            trial_params.supply_inventory_kg*
            float(np.sum(step.state.supply_enthalpy_j_kg-
                         old.supply_enthalpy_j_kg))/dt_s
        )
        return_storage=(
            trial_params.return_inventory_kg*
            float(np.sum(step.state.return_enthalpy_j_kg-
                         old.return_enthalpy_j_kg))/dt_s
        )
        q_supply_adv=q_supply_real-supply_storage
        q_return_adv=q_return_real-return_storage
        for label,value in (("supply",q_supply_adv),
                            ("return",q_return_adv)):
            if not math.isfinite(value) or value<0:
                raise ValueError(
                    f"{label} transient tie fluid is releasing heat "
                    "to inventory, not a positive regenerative leg; "
                    "the current network supports only nonnegative "
                    "effective leg heat input"
                )
        qadv=q_supply_adv+q_return_adv
        power_error=(step.nuclear_deposition_w
                     - qadv
                     - step.solid_storage_rate_w
                     - step.hydrogen_storage_rate_w)
        audit_scale=max(
            1.,abs(step.nuclear_deposition_w),abs(qadv),
            abs(step.solid_storage_rate_w),
            abs(step.hydrogen_storage_rate_w)
        )
        if abs(power_error)>1e-8*audit_scale+1e-5:
            raise RuntimeError("cryogenic source to hydraulic enthalpy ledger failed")
        mapped=replace(
            cooling,
            nozzle_jacket=replace(cooling.nozzle_jacket,
                                  heat_power_w=nozzle_wall_heat_w),
            reflector=replace(cooling.reflector,
                              heat_power_w=reflector_wall_heat_w),
            tie_supply=replace(cooling.tie_supply,
                               heat_power_w=q_supply_adv),
            tie_return=replace(cooling.tie_return,
                               heat_power_w=q_return_adv),
        )
        mapped.validate()
        point=solve_cooling_network(
            mapped,total_coolant_flow_kg_s,
            coolant_inlet_pressure_pa,coolant_inlet_enthalpy_j_kg,eos=eos,
        )
        sup=point.legs[2]
        ret=point.legs[3]
        target_flow=point.tie_tube_mass_flow_kg_s
        target_p=0.25*(
            sup.inlet_pressure_pa+sup.outlet_pressure_pa+
            ret.inlet_pressure_pa+ret.outlet_pressure_pa
        )
        if not 0<target_p<=coolant_inlet_pressure_pa:
            raise RuntimeError("hydraulic tie representative pressure invalid")
        flow_err=abs(trial_flow-target_flow)/target_flow
        pressure_err=abs(trial_p-target_p)/target_p
        source_change=(
            max(abs(q_supply_adv-last_sources[0]),
                abs(q_return_adv-last_sources[1]))/
            max(1.,abs(q_supply_adv),abs(q_return_adv),
                abs(last_sources[0]),abs(last_sources[1]))
            if last_sources is not None else math.inf
        )
        # Unsteady H2 source with finite storage must match the
        # *advected* enthalpy entering the solved downstream mixer,
        # not the deposited nuclear or wall heating alone.
        mixture_h=float(np.dot(
            trial_heating.tie_flow_kg_s,
            step.state.return_enthalpy_j_kg[:,0]
        )/trial_flow)
        # Each trial uses the transient flow for the tie energy
        # step and the newly solved hydraulic flow for its leg powers.
        # The two outlet enthalpies CANNOT coincide before flow
        # convergence; first independently verify the transient
        # advected power at its OWN flow.
        transient_h_expected=(coolant_inlet_enthalpy_j_kg+
                              qadv/trial_flow)
        local_h_tol=1e-8*max(1.,abs(mixture_h),
                              abs(transient_h_expected))
        if abs(mixture_h-transient_h_expected)>local_h_tol:
            raise RuntimeError(
                "tie transient outlet enthalpy violates its own "
                "advective source/inventory power ledger"
            )
        hydraulic_h_expected=(coolant_inlet_enthalpy_j_kg+
                              qadv/target_flow)
        if abs(ret.outlet_stagnation_enthalpy_j_kg-
               hydraulic_h_expected)>1e-8*max(
                   1.,abs(hydraulic_h_expected)
               ):
            raise RuntimeError(
                "hydraulic tie return enthalpy violates its own "
                "stagnation-source conservation"
            )
        h_error=mixture_h-ret.outlet_stagnation_enthalpy_j_kg
        # The exact mismatch identity at a finite outer iteration is
        # Qadv*(1/m_thermal - 1/m_hydraulic). Require this as an
        # independent residual check, not zero before convergence.
        predicted_h_error=qadv*(1./trial_flow-1./target_flow)
        if abs(h_error-predicted_h_error)>2e-8*max(
            1.,abs(h_error),abs(predicted_h_error),
            abs(mixture_h),abs(hydraulic_h_expected)
        ):
            raise RuntimeError("tie/hydraulic outlet mismatch is not explained by trial-flow residual")
        rec=TieHydraulicIteration(
            iteration=iteration,
            thermal_tie_flow_kg_s=trial_flow,
            hydraulic_tie_flow_kg_s=target_flow,
            thermal_pressure_pa=trial_p,
            representative_hydraulic_pressure_pa=target_p,
            relative_flow_residual=flow_err,
            relative_pressure_residual=pressure_err,
            relative_heat_source_change=source_change,
            tie_nuclear_deposition_w=step.nuclear_deposition_w,
            tie_advected_power_w=qadv,
            hydrogen_storage_rate_w=step.hydrogen_storage_rate_w,
            solid_storage_rate_w=step.solid_storage_rate_w,
            nuclear_energy_power_residual_w=power_error,
            outlet_enthalpy_difference_j_kg=h_error,
        )
        trace.append(rec)
        # In the accepted state cross-model enthalpy mismatch is
        # bounded by the specified FLOW tolerance times the actual
        # hydraulic leg enthalpy rise, not by machine precision.
        enthalpy_match=abs(h_error)<=(
            relative_flow_tolerance*
            max(1.,abs(ret.outlet_stagnation_enthalpy_j_kg-
                       coolant_inlet_enthalpy_j_kg))*(1+1e-7)
            + local_h_tol
        )
        if (last_sources is not None
                and flow_err<=relative_flow_tolerance
                and pressure_err<=relative_pressure_tolerance
                and source_change<=relative_source_tolerance
                and enthalpy_match):
            return TieHydraulicCoupledResult(
                heating=trial_heating,
                tie_step=step,cooling_point=point,
                effective_cooling_network=mapped,
                iterations=iteration,
                relative_flow_residual=flow_err,
                relative_pressure_residual=pressure_err,
                relative_heat_source_change=source_change,
                nuclear_energy_power_residual_w=power_error,
                outlet_enthalpy_difference_j_kg=h_error,
                diagnostics=tuple(trace),
            )
        trial_flow+=relaxation*(target_flow-trial_flow)
        trial_p+=relaxation*(target_p-trial_p)
        last_sources=(q_supply_adv,q_return_adv)
    last=trace[-1]
    raise RuntimeError(
        "cryogenic tie + regenerative hydraulics did not converge within "
        f"{max_iterations} iterations: flow={last.relative_flow_residual:.6g}, "
        f"pressure={last.relative_pressure_residual:.6g}, "
        f"source={last.relative_heat_source_change:.6g}"
    )
