"""Quasi-static OpenMC -> *channel-resolved* transient Picard coupling.

This module owns only conservation, Picard convergence, state transition,
and provenance tests. Real OpenMC jobs are provided by the orchestration
script, one FRESH statepoint and reconstructed heating package per
iteration. Monte Carlo spatial heating is re-normalized to EXTERNALLY
PRESCRIBED reactor power. There is no reactor kinetics or reactivity-power
feedback: temperature/density influence only the normalized spatial map.

OpenMC material feedback is still UNIFORM in the fuel and H2 material
surrogates, using reductions from channel/tie axial thermal states.
This is not per-voxel temperature reconstruction in the transport mesh.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .resolved_transient import (
    ResolvedHeating, ResolvedParameters, ResolvedState, ResolvedStep,
    advance_resolved,
)
from .implicit_hydrogen import hydrogen_specific_enthalpy
from .nasa_hydrogen import T_MIN, T_MAX


@dataclass(frozen=True)
class TransportSnapshot:
    """Fresh transport result and reconstructed material heat powers."""
    heating: ResolvedHeating
    source_kind: str
    statepoint_sha256: str
    nuclear_data_sha256: str
    fuel_material_temperature_k: float
    hydrogen_material_temperature_k: float
    analysis_qa_passed: bool
    source_path: str = ""


@dataclass(frozen=True)
class PicardRecord:
    iteration: int
    snapshot_sha256: str
    requested_fuel_temperature_k: float
    requested_hydrogen_temperature_k: float
    computed_fuel_temperature_k: float
    computed_hydrogen_temperature_k: float
    temperature_update_k: float
    profile_update_k: float
    energy_residual_j: float
    unresolved_fraction: float


@dataclass(frozen=True)
class CoupledStep:
    result: ResolvedStep
    last_snapshot: TransportSnapshot
    iterations: int
    temperature_error_k: float
    profile_error_k: float
    diagnostics: tuple[PicardRecord, ...]


def reduce_openmc_material_state(
    state: ResolvedState, parameters: ResolvedParameters
) -> tuple[float, float]:
    """Convert resolved thermal fields to UNIFORM fuel and H2 transport states.

    Fuel-solid temperature is averaged over fuel solid nodes (uniform
    solid heat capacity). For coolant, the *inventories* are weighted by
    the fixed gas masses Cgas/cp_reference, not by coolant flow. For
    NASA-7 hydrogen, invert the mass-weighted specific enthalpy to the
    equivalent uniform temperature. An arithmetic temperature average
    would not conserve the stored thermal energy when cp(T) varies.
    Constant-cp mode retains the exact mass-weighted temperature mean.
    Tie solid is not treated as fuel in OpenMC: other solids remain at
    their existing transport-model material definitions.
    """
    parameters.validate()
    fuel = np.asarray(state.fuel_solid_t_k, dtype=float)
    fuel_gas = np.asarray(state.fuel_h2_t_k, dtype=float)
    tie_supply = np.asarray(state.tie_supply_t_k, dtype=float)
    tie_return = np.asarray(state.tie_return_t_k, dtype=float)
    arrays = (fuel, fuel_gas, tie_supply, tie_return)
    if any(a.size == 0 or not np.all(np.isfinite(a)) or np.any(a <= 0)
           for a in arrays):
        raise ValueError("nonphysical resolved thermal feedback temperatures")

    fg = parameters.fuel_h2_capacity_j_k * fuel_gas.size
    ts = parameters.tie_supply_h2_capacity_j_k * tie_supply.size
    tr = parameters.tie_return_h2_capacity_j_k * tie_return.size
    weight = fg + ts + tr
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("invalid hydrogen inventory for OpenMC projection")

    if parameters.enthalpy_model == "constant":
        # With constant cp, a mass-weighted enthalpy mean is exactly
        # the capacity-weighted arithmetic temperature mean.
        combined = (
            parameters.fuel_h2_capacity_j_k * float(fuel_gas.sum())
            + parameters.tie_supply_h2_capacity_j_k * float(tie_supply.sum())
            + parameters.tie_return_h2_capacity_j_k * float(tie_return.sum())
        ) / weight
        return float(fuel.mean()), float(combined)

    # The thermal solver stores m_g*h(T) for NASA-7 frozen H2, with
    # m_g=Cgas/cp_reference per cell. Since cp_reference is shared,
    # the reference heat capacities are proportional to fixed gas masses.
    if any(np.any((a < T_MIN) | (a > T_MAX))
           for a in (fuel_gas, tie_supply, tie_return)):
        raise ValueError("gas feedback temperature outside NASA-7 domain")
    h = lambda t: hydrogen_specific_enthalpy(
        t, parameters.hydrogen_cp_reference_j_kg_k, "nasa7-h2"
    )
    target_enthalpy = math.fsum((
        parameters.fuel_h2_capacity_j_k *
        math.fsum(h(float(t)) for t in fuel_gas.flat),
        parameters.tie_supply_h2_capacity_j_k *
        math.fsum(h(float(t)) for t in tie_supply.flat),
        parameters.tie_return_h2_capacity_j_k *
        math.fsum(h(float(t)) for t in tie_return.flat),
    )) / weight

    # h(T) is strictly increasing for the valid NASA-7 pure-H2 range.
    # Bisection cannot jump to a thermodynamically invalid branch.
    lower, upper = T_MIN, T_MAX
    for _ in range(55):
        mid = 0.5*(lower + upper)
        if h(mid) < target_enthalpy:
            lower = mid
        else:
            upper = mid
    return float(fuel.mean()), float(0.5*(lower + upper))


def state_infinity_difference(a: ResolvedState, b: ResolvedState) -> float:
    names = (
        "fuel_solid_t_k", "fuel_h2_t_k", "tie_solid_t_k",
        "tie_supply_t_k", "tie_return_t_k"
    )
    deltas = []
    for name in names:
        x, y = np.asarray(getattr(a, name)), np.asarray(getattr(b, name))
        if x.shape != y.shape:
            raise ValueError("transport changed resolved state topology")
        deltas.append(float(np.max(np.abs(x-y))))
    deltas.append(abs(a.unresolved_energy_j-b.unresolved_energy_j)
                  / max(1.0, abs(a.unresolved_energy_j),
                        abs(b.unresolved_energy_j)))
    return max(deltas)


def _assert_same_hydraulic_layout(
    current: ResolvedHeating, first: ResolvedHeating
) -> None:
    """External fixed-flow assumption: reject changing channel topology."""
    current.validate()
    first.validate()
    if (current.fuel_solid_w.shape != first.fuel_solid_w.shape
            or current.tie_solid_w.shape != first.tie_solid_w.shape):
        raise RuntimeError("OpenMC changed channel/tie heat mapping shape")
    if (not np.allclose(current.fuel_flow_kg_s, first.fuel_flow_kg_s,
                        atol=1e-12, rtol=1e-10)
            or not np.allclose(current.tie_flow_kg_s, first.tie_flow_kg_s,
                               atol=1e-12, rtol=1e-10)):
        raise RuntimeError("fuel/tie flows changed between Picard snapshots")
    if not math.isclose(current.total_openmc_power_w,
                        first.total_openmc_power_w, rel_tol=1e-8):
        raise RuntimeError("externally imposed reactor power changed")


def advance_resolved_with_openmc(
    state: ResolvedState,
    params: ResolvedParameters,
    inlet_temperature_k: float,
    dt_s: float,
    provider: Callable[[float, float], TransportSnapshot],
    *,
    max_iterations: int = 5,
    convergence_temperature_k: float = 0.5,
    relaxation: float = 1.0,
    max_unresolved_fraction: float = 0.6,
    allow_synthetic_test_provider: bool = False,
) -> CoupledStep:
    """Perform Picard iteration, resetting each trial to SAME old state.

    The provider is responsible for completing and hashing an actual
    transport run + reconstructing every relevant material power field.
    Its supplied source_kind must be 'openmc-statepoint' for real mode.
    Synthetic providers are accepted ONLY under an explicit test switch.

    Each requested material-state pair must result in a *new* snapshot
    identity even if the power shape is numerically identical. No
    transport re-use is accepted inside a step. Nonconvergence raises.
    """
    params.validate()
    if (max_iterations < 2 or not math.isfinite(convergence_temperature_k)
            or convergence_temperature_k <= 0):
        raise ValueError("invalid convergence parameters")
    if not math.isfinite(relaxation) or not (0 < relaxation <= 1):
        raise ValueError("relaxation must be in (0,1]")
    if not 0 <= max_unresolved_fraction <= 1:
        raise ValueError("invalid unresolved fraction")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("invalid thermal timestep")

    fuel_guess, gas_guess = reduce_openmc_material_state(state, params)
    old_topology: ResolvedHeating | None = None
    old_nuclear_data_hash: str | None = None
    old_step: ResolvedStep | None = None
    seen: set[str] = set()
    trace: list[PicardRecord] = []
    last_update = math.inf
    last_profile = math.inf

    for i in range(1, max_iterations+1):
        snap = provider(fuel_guess, gas_guess)
        if not isinstance(snap, TransportSnapshot):
            raise TypeError("provider must return a TransportSnapshot")
        if (snap.source_kind != "openmc-statepoint"
                and not (allow_synthetic_test_provider
                         and snap.source_kind == "synthetic-test")):
            raise ValueError("real feedback requires fresh OpenMC statepoint")
        if not snap.analysis_qa_passed:
            raise RuntimeError("OpenMC analysis QA rejected snapshot")
        if not snap.statepoint_sha256 or snap.statepoint_sha256 in seen:
            raise RuntimeError("stale/reused OpenMC snapshot in Picard iteration")
        seen.add(snap.statepoint_sha256)
        if not snap.nuclear_data_sha256:
            raise RuntimeError("missing verified cross-section library hash")
        if (old_nuclear_data_hash is not None
                and snap.nuclear_data_sha256 != old_nuclear_data_hash):
            raise RuntimeError("nuclear-data library changed within iteration")
        old_nuclear_data_hash = snap.nuclear_data_sha256
        if not math.isclose(snap.fuel_material_temperature_k, fuel_guess,
                            abs_tol=1e-9):
            raise RuntimeError("transport fuel-temperature input mismatch")
        if not math.isclose(snap.hydrogen_material_temperature_k, gas_guess,
                            abs_tol=1e-9):
            raise RuntimeError("transport H2-temperature input mismatch")
        pm = snap.heating
        pm.validate()
        if old_topology is None:
            old_topology = pm
        else:
            _assert_same_hydraulic_layout(pm, old_topology)
        unresolved = pm.unresolved_power_w / pm.total_openmc_power_w
        if unresolved > max_unresolved_fraction:
            raise RuntimeError("too much deposition outside modeled channels")

        # Critical: always advance from the ORIGINAL state; never
        # integrate multiple timesteps during convergence iterations.
        trial = advance_resolved(pm, state, params, inlet_temperature_k, dt_s)
        target_fuel, target_gas = reduce_openmc_material_state(trial.state, params)
        last_update = max(
            abs(target_fuel-fuel_guess), abs(target_gas-gas_guess)
        )
        last_profile = (
            state_infinity_difference(trial.state, old_step.state)
            if old_step is not None else math.inf
        )
        trace.append(PicardRecord(
            iteration=i, snapshot_sha256=snap.statepoint_sha256,
            requested_fuel_temperature_k=fuel_guess,
            requested_hydrogen_temperature_k=gas_guess,
            computed_fuel_temperature_k=target_fuel,
            computed_hydrogen_temperature_k=target_gas,
            temperature_update_k=last_update,
            profile_update_k=last_profile,
            energy_residual_j=trial.energy_residual_j,
            unresolved_fraction=unresolved,
        ))
        if (old_step is not None
                and last_update <= convergence_temperature_k
                and last_profile <= convergence_temperature_k):
            return CoupledStep(trial, snap, i, last_update,
                               last_profile, tuple(trace))
        fuel_guess += relaxation*(target_fuel-fuel_guess)
        gas_guess += relaxation*(target_gas-gas_guess)
        old_step = trial

    raise RuntimeError(
        "resolved OpenMC Picard nonconvergence; "
        f"{max_iterations} iterations, temperature update {last_update:.3g} K, "
        f"profile delta {last_profile:.3g} K"
    )
