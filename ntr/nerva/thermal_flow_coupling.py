"""Outer fixed point between resolved thermal flux and shaft-closed cycle FLOWS.

Repeated trials always advance from the same old transient state; they do
not integrate several time steps. Relative fuel/tie flows are preserved
inside each family while TOTAL family flow is returned by the engine
pump/turbine/hydraulic solver. Each iterate solves a new BE thermal step,
projects solid wall heat / solid storage, and solves the full steady
engine cycle. The retained nuclear source and first-law checks in both
solvers remain active.

Limit: the resolved model still accepts a SINGLE externally prescribed
hydrogen inlet temperature, not the separate real cryogenic tie inlet and
hot-core header inlet, and it does not solve distributed coolant momentum.
Therefore convergence here is a mass-flow/thermal-source fixed point,
NOT a fully consistent NERVA thermal-hydraulic or neutron kinetics model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

import numpy as np

from .lh2_feed_cycle import LH2Cycle
from .regenerative_cooling import CoolingNetwork
from .resolved_transient import (
    ResolvedHeating, ResolvedParameters, ResolvedState, ResolvedStep,
    advance_resolved,
)
from .thermal_cycle_feedback import (
    ResolvedWallPowerLedger, project_resolved_wall_power,
)
from .turbine_bleed_cycle import (
    TurbineBleed, ShaftClosedPoint, solve_turbine_bleed,
)


@dataclass(frozen=True)
class FlowIteration:
    iteration: int
    thermal_fuel_flow_kg_s: float
    thermal_tie_flow_kg_s: float
    cycle_fuel_flow_kg_s: float
    cycle_tie_flow_kg_s: float
    max_relative_flow_error: float
    max_thermal_profile_update_k: float
    cycle_header_temperature_k: float
    nuclear_power_balance_error_w: float
    cycle_energy_balance_error_w: float


@dataclass(frozen=True)
class FlowCouplingResult:
    engine: ShaftClosedPoint
    thermal_step: ResolvedStep
    thermal_heating: ResolvedHeating
    mapped_core: LH2Cycle
    mapped_network: CoolingNetwork
    power_ledger: ResolvedWallPowerLedger
    iterations: int
    max_relative_flow_error: float
    max_thermal_profile_update_k: float
    diagnostics: tuple[FlowIteration, ...]
    thermal_inlet_temperature_k: float
    temperature_boundary_not_closed: bool = True
    transport_performed: bool = False


def _max_profile_delta(a: ResolvedState, b: ResolvedState) -> float:
    names = ("fuel_solid_t_k", "fuel_h2_t_k", "tie_solid_t_k",
             "tie_supply_t_k", "tie_return_t_k")
    return max(
        float(np.max(np.abs(np.asarray(getattr(a,n)) -
                            np.asarray(getattr(b,n))))) for n in names
    )


def couple_resolved_thermal_flows(
    heating: ResolvedHeating,
    old: ResolvedState,
    parameters: ResolvedParameters,
    core: LH2Cycle,
    cooling: CoolingNetwork,
    turbine: TurbineBleed,
    inlet_temperature_k: float,
    dt_s: float,
    *,
    nozzle_wall_heat_w: float,
    reflector_wall_heat_w: float,
    chemistry_mode: str = "frozen",
    max_iterations: int = 12,
    relative_flow_tolerance: float = 0.002,
    profile_tolerance_k: float = 0.5,
    relaxation: float = 0.5,
    engine_solver: Callable | None = None,
) -> FlowCouplingResult:
    """Fail-closed flow fixed point with a conservation check per iterate.

    engine_solver is injectable only for synthetic numerical tests; the
    production/default path always invokes solve_turbine_bleed.
    """
    heating.validate()
    parameters.validate()
    core.validate()
    cooling.validate()
    turbine.validate()
    if not isinstance(max_iterations,int) or max_iterations < 2:
        raise ValueError("flow feedback requires at least two iterations")
    if (not math.isfinite(relative_flow_tolerance)
            or not 0 < relative_flow_tolerance <= 0.1
            or not math.isfinite(profile_tolerance_k)
            or profile_tolerance_k <= 0
            or not math.isfinite(relaxation)
            or not 0 < relaxation <= 1):
        raise ValueError("invalid flow fixed-point convergence parameters")
    if not math.isfinite(inlet_temperature_k) or inlet_temperature_k <= 0:
        raise ValueError("invalid prescribed resolved hydrogen inlet")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("positive transient timestep required")

    base_fuel=heating.total_fuel_flow_kg_s
    base_tie=heating.total_tie_flow_kg_s
    gf,gt=base_fuel,base_tie
    last_state=None
    trace=[]
    solver=engine_solver or solve_turbine_bleed

    for iteration in range(1,max_iterations+1):
        trial_heating=replace(
            heating,
            fuel_flow_kg_s=heating.fuel_flow_kg_s*(gf/base_fuel),
            tie_flow_kg_s=heating.tie_flow_kg_s*(gt/base_tie),
        )
        trial_heating.validate()
        # ALWAYS advance the SAME old transient state, not previous iterate.
        thermal_step=advance_resolved(
            trial_heating,old,parameters,inlet_temperature_k,dt_s,
        )
        mapped_core,mapped_network,ledger=project_resolved_wall_power(
            trial_heating,old,thermal_step,parameters,core,cooling,dt_s,
            nozzle_wall_heat_w=nozzle_wall_heat_w,
            reflector_wall_heat_w=reflector_wall_heat_w,
        )
        point=solver(
            mapped_core,turbine,cooling_network=mapped_network,
            chemistry_mode=chemistry_mode,
        )
        if point.regenerative_cooling is None:
            raise RuntimeError("flow feedback needs solved coolant branches")
        target_f=float(point.core_mass_flow_kg_s)
        target_t=float(point.regenerative_cooling.tie_tube_mass_flow_kg_s)
        if any(not math.isfinite(x) or x<=0 for x in (target_f,target_t)):
            raise RuntimeError("engine returned nonphysical coupled flow")
        flow_error=max(abs(gf-target_f)/target_f,
                       abs(gt-target_t)/target_t)
        profile_error=(
            _max_profile_delta(thermal_step.state,last_state)
            if last_state is not None else math.inf
        )
        rec=FlowIteration(
            iteration=iteration,
            thermal_fuel_flow_kg_s=gf,
            thermal_tie_flow_kg_s=gt,
            cycle_fuel_flow_kg_s=target_f,
            cycle_tie_flow_kg_s=target_t,
            max_relative_flow_error=flow_error,
            max_thermal_profile_update_k=profile_error,
            cycle_header_temperature_k=point.warm_header_temperature_k,
            nuclear_power_balance_error_w=ledger.nuclear_power_balance_error_w,
            cycle_energy_balance_error_w=point.total_cycle_energy_residual_w,
        )
        trace.append(rec)
        if (last_state is not None
                and flow_error<=relative_flow_tolerance
                and profile_error<=profile_tolerance_k):
            return FlowCouplingResult(
                engine=point,thermal_step=thermal_step,
                thermal_heating=trial_heating,mapped_core=mapped_core,
                mapped_network=mapped_network,power_ledger=ledger,
                iterations=iteration,
                max_relative_flow_error=flow_error,
                max_thermal_profile_update_k=profile_error,
                diagnostics=tuple(trace),
                thermal_inlet_temperature_k=inlet_temperature_k,
            )
        gf += relaxation*(target_f-gf)
        gt += relaxation*(target_t-gt)
        last_state=thermal_step.state

    raise RuntimeError(
        "resolved NERVA thermal/cycle flow fixed point did not converge "
        f"within {max_iterations} iterations: "
        f"last relative flow error={trace[-1].max_relative_flow_error:.6g}, "
        f"thermal profile change={trace[-1].max_thermal_profile_update_k:.6g} K"
    )
