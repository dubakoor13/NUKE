"""Independent time-refinement audit for resolved NERVA thermal transients.

Runs the *same* imposed OpenMC-derived heating field and constant-flow
thermal problem at dt, dt/2, dt/4, ... to exactly the SAME physical
end time, preserving all fuel-channel/tie states.

Numerical method: first-order backward Euler / implicit counterflow
Picard. Final-state infinity-norm differences between successive
time steps estimate the leading temporal truncation error. An observed
order is a diagnostic, NOT proof of agreement with NASA measurements.

These comparisons MUST NOT be called an OpenMC transport convergence
study: spatial neutron heating is intentionally frozen and no MC
statepoint is generated. Separate power/mesh/tally uncertainty studies
are needed for actual Monte Carlo convergence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .resolved_transient import (
    ResolvedHeating, ResolvedParameters, ResolvedState, advance_resolved,
)


_FIELDS = (
    "fuel_solid_t_k", "fuel_h2_t_k", "tie_solid_t_k",
    "tie_supply_t_k", "tie_return_t_k",
)


@dataclass(frozen=True)
class RefinementLevel:
    level: int
    steps: int
    dt_s: float
    end_time_s: float
    maximum_step_energy_residual_j: float
    accumulated_energy_residual_j: float
    maximum_fuel_outlet_k: float
    maximum_tie_return_outlet_k: float
    fuel_and_tie_temperature_k: tuple[float, float]


@dataclass(frozen=True)
class TemporalRefinement:
    levels: tuple[RefinementLevel, ...]
    difference_infinity_k: tuple[float, ...]
    difference_by_field_k: tuple[dict[str, float], ...]
    observed_order: tuple[float | None, ...]
    finest_difference_k: float
    extrapolated_fine_error_k: float | None
    tolerance_k: float
    tolerance_passed: bool
    difference_reducing: bool
    energy_residual_passed: bool
    status: str
    interpretation: str


def field_error(a: ResolvedState, b: ResolvedState) -> dict[str, float]:
    """Maximum absolute difference of every resolved thermal field, in K."""
    err = {}
    for name in _FIELDS:
        x = np.asarray(getattr(a, name), dtype=float)
        y = np.asarray(getattr(b, name), dtype=float)
        if x.shape != y.shape or x.size == 0:
            raise ValueError("time refinement changed resolved channel layout")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("time refinement contains non-finite temperatures")
        err[name] = float(np.max(np.abs(x-y)))
    return err


def refine_resolved_timestep(
    heating: ResolvedHeating,
    initial: ResolvedState,
    params: ResolvedParameters,
    inlet_temperature_k: float,
    end_time_s: float,
    *,
    base_steps: int = 2,
    levels: int = 4,
    tolerance_k: float = 0.25,
    require_decreasing: bool = True,
    max_steps: int = 4096,
) -> TemporalRefinement:
    """Integrate with refinement factors 1,2,4,... at fixed physical time.

    The finest vs penultimate grid difference approximates the leading
    error at the fine grid for first-order temporal integration.
    For near-zero differences, observed order is undefined (None).
    No timestep is shortened, extrapolated or substituted into the
    normal simulator's result: the audit is nondestructive.
    """
    heating.validate()
    params.validate()
    if not isinstance(base_steps, int) or base_steps < 1:
        raise ValueError("base_steps must be a positive integer")
    if not isinstance(levels, int) or not 3 <= levels <= 8:
        raise ValueError("levels must be an integer in [3,8]")
    if not math.isfinite(end_time_s) or end_time_s <= 0:
        raise ValueError("end_time_s must be positive and finite")
    if not math.isfinite(tolerance_k) or tolerance_k <= 0:
        raise ValueError("tolerance_k must be positive and finite")
    if base_steps * (2 ** (levels-1)) > max_steps:
        raise ValueError("refinement would require too many timesteps")
    if not math.isfinite(inlet_temperature_k) or inlet_temperature_k <= 0:
        raise ValueError("inlet temperature must be positive and finite")

    rows: list[RefinementLevel] = []
    differences: list[float] = []
    by_field: list[dict[str, float]] = []
    previous: ResolvedState | None = None
    energy_ok = True

    for level in range(levels):
        n = base_steps * (2**level)
        dt = end_time_s / n
        # State is immutable and advance_resolved allocates new arrays.
        state = initial
        max_residual = 0.0
        sum_residual = 0.0
        for _ in range(n):
            result = advance_resolved(
                heating, state, params, inlet_temperature_k, dt
            )
            state = result.state
            residual = result.energy_residual_j
            max_residual = max(max_residual, abs(residual))
            sum_residual += residual
            # The solver itself audits every individual step; additionally
            # track a conservative global, dimensionless energy threshold.
            scale = max(
                1.0, result.energy_stored_delta_j,
                dt*heating.total_openmc_power_w
            )
            if abs(residual) > 1e-8*scale + 1e-6:
                energy_ok = False

        rows.append(RefinementLevel(
            level=level, steps=n, dt_s=dt, end_time_s=n*dt,
            maximum_step_energy_residual_j=max_residual,
            accumulated_energy_residual_j=sum_residual,
            maximum_fuel_outlet_k=float(np.max(state.fuel_h2_t_k[:,:,-1])),
            maximum_tie_return_outlet_k=float(np.max(state.tie_return_t_k[:,0])),
            fuel_and_tie_temperature_k=(
                float(np.max(state.fuel_solid_t_k)),
                float(np.max(state.tie_solid_t_k)),
            ),
        ))
        if previous is not None:
            diff = field_error(state, previous)
            by_field.append(diff)
            differences.append(max(diff.values()))
        previous = state

    observed: list[float | None] = []
    for earlier, later in zip(differences, differences[1:]):
        if earlier <= 1e-10 or later <= 1e-10:
            observed.append(None)
        else:
            observed.append(math.log2(earlier/later))
    # The expected BE asymptotic order is p=1. Use the last observed
    # order if valid and positive for a diagnostic Richardson estimate.
    p = observed[-1] if observed else None
    estimated = None
    if p is not None and p > 0.1 and p < 5.0:
        estimated = differences[-1]/(2**p-1)
    decreasing = all(
        later <= earlier*(1+1e-9)+1e-9
        for earlier, later in zip(differences, differences[1:])
    )
    tolerance_passed = differences[-1] <= tolerance_k
    passed = (
        tolerance_passed and energy_ok
        and (decreasing or not require_decreasing)
    )
    return TemporalRefinement(
        levels=tuple(rows), difference_infinity_k=tuple(differences),
        difference_by_field_k=tuple(by_field),
        observed_order=tuple(observed),
        finest_difference_k=differences[-1],
        extrapolated_fine_error_k=estimated,
        tolerance_k=tolerance_k,
        tolerance_passed=tolerance_passed,
        difference_reducing=decreasing,
        energy_residual_passed=energy_ok,
        status="PASS" if passed else "FAIL",
        interpretation=(
            "Temporal convergence at fixed nuclear power/heating, "
            "flow and material parameters ONLY; not OpenMC neutron "
            "transport or historical NERVA validation."
        ),
    )
