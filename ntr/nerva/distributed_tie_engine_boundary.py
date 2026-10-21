"""Bridge a SHAFT-CLOSED NERVA engine operating point to transient tie Euler boundaries.

The established turbine-bleed + regenerative hydraulic network provides
an independent, fully solved STEADY mass/pressure/enthalpy reference:
 - tie supply inlet pressure and H2 stagnation enthalpy from pump header
 - tie return outlet pressure and H2 stagnation enthalpy from mixed header
 - tie branch mass-flow reference from the hydraulic splitter.

Those two thermodynamic states are applied as stationary reservoir ghost
states to the independent distributed compressible tie Euler step.
This is ONE-WAY boundary conditioning: the transient tie flux is allowed
to differ from the steady network's branch flow. The difference is
reported as a genuine mass-inventory/momentum closure diagnostic,
NEVER labeled a converged closed-loop engine operating point.

IMPORTANT: the Rusanov numerical *total-energy* boundary flux
contains kinetic/pressure work and diffusion and must NOT be replaced
by mdot*h, or treated as a new steady heating source. The Euler
conservative global energy ledger is retained unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .distributed_tie_euler import (
    DistributedTieParameters, DistributedTieState, DistributedTieStep,
    advance_distributed_tie,
)
from .lh2_feed_cycle import HydrogenEOS
from .turbine_bleed_cycle import ShaftClosedPoint
from .resolved_transient import ResolvedHeating


@dataclass(frozen=True)
class EngineTieReservoirs:
    upstream_pressure_pa: float
    upstream_stagnation_enthalpy_j_kg: float
    downstream_pressure_pa: float
    downstream_backflow_stagnation_enthalpy_j_kg: float
    steady_tie_branch_mass_flow_kg_s: float
    steady_total_pump_mass_flow_kg_s: float
    steady_common_header_pressure_pa: float
    steady_common_header_stagnation_enthalpy_j_kg: float


@dataclass(frozen=True)
class DistributedTieEngineBoundaryStep:
    reservoirs: EngineTieReservoirs
    transient: DistributedTieStep
    inlet_transient_mass_flow_kg_s: float
    outlet_transient_mass_flow_kg_s: float
    steady_tie_mass_flow_reference_kg_s: float
    inlet_mass_flow_reference_residual_kg_s: float
    outlet_mass_flow_reference_residual_kg_s: float
    transient_mass_accumulation_kg_s: float
    transient_mass_conservation_residual_kg_s: float
    transient_total_energy_conservation_residual_j: float
    transient_momentum_conservation_residual_n: float
    transient_flow_matches_steady_engine: bool = False
    full_engine_feedback_converged: bool = False
    external_openmc_transport_executed: bool = False


def engine_tie_reservoirs(
    shaft: ShaftClosedPoint,
) -> EngineTieReservoirs:
    """Extract hydraulic tie boundary without inventing header pressure.

    Supply/return hydraulic leg pressures and stagnation enthalpies
    are obtained from one and the same solved engine operating point.
    They are stationary H2 reservoirs when imposed on the transient.
    """
    hydraulic=shaft.regenerative_cooling
    if hydraulic is None:
        raise ValueError(
            "cannot supply distributed tie boundaries without "
            "regenerative-cooling shaft-cycle solution"
        )
    if len(hydraulic.legs)!=4 or [l.name for l in hydraulic.legs]!=[
        "nozzle_jacket","reflector","tie_supply","tie_return"
    ]:
        raise ValueError("unsupported cooling leg ordering or topology")
    supply=hydraulic.legs[2]
    ret=hydraulic.legs[3]
    m=hydraulic.tie_tube_mass_flow_kg_s
    total=shaft.tank_mass_flow_kg_s
    for name,val in (
        ("tie mass flow",m),
        ("pump total flow",total),
        ("supply pressure",supply.inlet_pressure_pa),
        ("return pressure",ret.outlet_pressure_pa),
        ("mixed header pressure",hydraulic.common_header_pressure_pa),
    ):
        if not math.isfinite(val) or val<=0:
            raise ValueError(f"invalid solved engine {name}")
    if not 0<m<total:
        raise ValueError("engine tie flow must be below pump total flow")
    if not (
        supply.inlet_pressure_pa>ret.outlet_pressure_pa
        and supply.inlet_pressure_pa>hydraulic.common_header_pressure_pa
    ):
        raise ValueError("solved tie header has nonpositive pressure drop")
    if abs(ret.outlet_pressure_pa-hydraulic.common_header_pressure_pa)>max(
        1.,1e-7*supply.inlet_pressure_pa
    ):
        raise ValueError("steady tie return and mixed header pressure disagree")
    if not math.isclose(supply.mass_flow_kg_s,m,rel_tol=1e-9) or not math.isclose(
        ret.mass_flow_kg_s,m,rel_tol=1e-9
    ):
        raise ValueError("steady tie hydraulic branches not mass closed")
    if not math.isclose(
        supply.inlet_pressure_pa,hydraulic.legs[0].inlet_pressure_pa,
        rel_tol=1e-9
    ):
        raise ValueError("tie and nozzle do not share the same pump header")
    for key,val in (
        ("tie inlet H2 enthalpy",supply.inlet_stagnation_enthalpy_j_kg),
        ("tie return H2 enthalpy",ret.outlet_stagnation_enthalpy_j_kg),
        ("mixed header H2 enthalpy",
         hydraulic.common_header_stagnation_enthalpy_j_kg),
    ):
        if not math.isfinite(val):
            raise ValueError(f"invalid {key}")
    return EngineTieReservoirs(
        upstream_pressure_pa=supply.inlet_pressure_pa,
        upstream_stagnation_enthalpy_j_kg=
            supply.inlet_stagnation_enthalpy_j_kg,
        downstream_pressure_pa=ret.outlet_pressure_pa,
        downstream_backflow_stagnation_enthalpy_j_kg=
            ret.outlet_stagnation_enthalpy_j_kg,
        steady_tie_branch_mass_flow_kg_s=m,
        steady_total_pump_mass_flow_kg_s=total,
        steady_common_header_pressure_pa=hydraulic.common_header_pressure_pa,
        steady_common_header_stagnation_enthalpy_j_kg=
            hydraulic.common_header_stagnation_enthalpy_j_kg,
    )


def advance_engine_conditioned_distributed_tie(
    heating: ResolvedHeating,
    previous: DistributedTieState,
    parameters: DistributedTieParameters,
    shaft: ShaftClosedPoint,
    dt_s: float,
    *,
    eos: HydrogenEOS | None = None,
) -> DistributedTieEngineBoundaryStep:
    """One transient step driven by validated shaft-cycle reservoir states.

    No feedback to the shaft solver occurs in this function; in
    particular the tie branch flow is a comparison target, not imposed
    on the Euler cell flux, and heating maps are not rescaled to it.
    """
    reservoirs=engine_tie_reservoirs(shaft)
    step=advance_distributed_tie(
        heating,previous,parameters,
        reservoirs.upstream_pressure_pa,
        reservoirs.upstream_stagnation_enthalpy_j_kg,
        reservoirs.downstream_pressure_pa,
        reservoirs.downstream_backflow_stagnation_enthalpy_j_kg,
        dt_s,eos=eos,
    )
    entering=float(np.sum(step.interface_mass_flux_kg_s[:,0]))
    leaving=float(np.sum(step.interface_mass_flux_kg_s[:,-1]))
    reference=reservoirs.steady_tie_branch_mass_flow_kg_s
    accumulated=float(np.sum(
        step.state.mass_kg-previous.mass_kg
    )/dt_s)
    mass_error=accumulated-(entering-leaving)
    if not math.isclose(mass_error,step.mass_residual_kg_s,
                        abs_tol=1e-8,rel_tol=1e-6):
        raise RuntimeError("engine boundary tie mass accumulation mismatch")
    # No transient/steady flow agreement is assumed: the initial
    # pressure-wave transient can differ strongly from a steady branch
    # flow and may have a nonzero total H2 inventory storage rate.
    return DistributedTieEngineBoundaryStep(
        reservoirs=reservoirs,
        transient=step,
        inlet_transient_mass_flow_kg_s=entering,
        outlet_transient_mass_flow_kg_s=leaving,
        steady_tie_mass_flow_reference_kg_s=reference,
        inlet_mass_flow_reference_residual_kg_s=entering-reference,
        outlet_mass_flow_reference_residual_kg_s=leaving-reference,
        transient_mass_accumulation_kg_s=accumulated,
        transient_mass_conservation_residual_kg_s=step.mass_residual_kg_s,
        transient_total_energy_conservation_residual_j=
            step.total_energy_residual_j,
        transient_momentum_conservation_residual_n=step.momentum_residual_n,
    )
