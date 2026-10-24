"""FX2-TH-inspired, spatially reduced kinetics and flux-shape bookkeeping.

Source: Shober, Daly & Ferguson, ANL-78-97 (1978), chapter II.C.
Independent implementation, NOT historic FX2-TH FORTRAN, multigroup
diffusion, NERVA coefficients, OpenMC transient reactivity, or Doppler laws.

n' = ((rho-beta)/Lambda)*n + sum(lambda_i*C_i) + source
C_i' = (beta_i/Lambda)*n - lambda_i*C_i

Spatial/spectral shape S is normalized under an independently supplied
adjoint weight W, inverse neutron velocity 1/v and mesh cell volumes:
    sum(W * (1/v) * S * volume) = 1.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class KineticsCoefficients:
    """External effective parameters, NOT hard-coded nuclear data."""
    generation_time_s: float
    beta_groups: tuple[float, ...]
    lambda_groups_s_inv: tuple[float, ...]
    reactivity: float = 0.0
    source_amplitude_per_s: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.generation_time_s) or self.generation_time_s <= 0:
            raise ValueError("generation_time_s must be finite and positive")
        if not math.isfinite(self.reactivity) or self.reactivity >= 1:
            raise ValueError("reactivity must be finite and < 1")
        if not math.isfinite(self.source_amplitude_per_s) or self.source_amplitude_per_s < 0:
            raise ValueError("source must be finite and nonnegative")
        if len(self.beta_groups) != len(self.lambda_groups_s_inv):
            raise ValueError("delayed group count mismatch")
        if any(not math.isfinite(b) or b < 0 for b in self.beta_groups):
            raise ValueError("invalid delayed fractions")
        if sum(self.beta_groups) >= 1:
            raise ValueError("sum(beta_groups) must be < 1")
        if any(not math.isfinite(lam) or lam <= 0 for lam in self.lambda_groups_s_inv):
            raise ValueError("invalid decay constants")

    @property
    def beta_total(self) -> float:
        return sum(self.beta_groups)


@dataclass(frozen=True)
class KineticsState:
    time_s: float
    amplitude: float
    precursors: tuple[float, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.time_s) or self.time_s < 0:
            raise ValueError("invalid time")
        if not math.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError("invalid amplitude")
        if any(not math.isfinite(c) or c < 0 for c in self.precursors):
            raise ValueError("invalid precursor inventory")


@dataclass(frozen=True)
class KineticsStep:
    state: KineticsState
    substeps: int
    amplitude_balance_residual: float
    max_precursor_balance_residual: float


def critical_precursors(amplitude: float, coeffs: KineticsCoefficients) -> tuple[float, ...]:
    """Steady precursor inventories at a fixed amplitude."""
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("invalid amplitude")
    return tuple(amplitude * b / (coeffs.generation_time_s * lam)
                 for b, lam in zip(coeffs.beta_groups, coeffs.lambda_groups_s_inv))


def kinetics_rhs(
    state: KineticsState, coeffs: KineticsCoefficients
) -> tuple[float, tuple[float, ...]]:
    if len(state.precursors) != len(coeffs.beta_groups):
        raise ValueError("delayed-group count mismatch")
    n = state.amplitude
    dn = ((coeffs.reactivity - coeffs.beta_total)
          / coeffs.generation_time_s * n
          + sum(lam * c for lam, c in
                zip(coeffs.lambda_groups_s_inv, state.precursors))
          + coeffs.source_amplitude_per_s)
    dc = tuple(b / coeffs.generation_time_s * n - lam * c
               for b, lam, c in zip(coeffs.beta_groups,
                                    coeffs.lambda_groups_s_inv, state.precursors))
    return dn, dc


def _backward_euler_substep(
    state: KineticsState, coeffs: KineticsCoefficients, dt: float
) -> KineticsState:
    terms = tuple((1.0 + dt * lam, b, lam, c)
                  for b, lam, c in zip(coeffs.beta_groups,
                                       coeffs.lambda_groups_s_inv, state.precursors))
    # Algebraic precursor elimination gives the Schur complement:
    denominator = (1.0 - dt * coeffs.reactivity / coeffs.generation_time_s
                   + dt / coeffs.generation_time_s
                   * sum(b / d for d, b, _, _ in terms))
    if denominator <= 0 or not math.isfinite(denominator):
        raise ArithmeticError("nonpositive implicit kinetics denominator")
    numerator = (state.amplitude + dt * coeffs.source_amplitude_per_s
                 + dt * sum(lam * c / d for d, _, lam, c in terms))
    next_n = numerator / denominator
    next_c = tuple((c + dt * b / coeffs.generation_time_s * next_n) / d
                   for d, b, _, c in terms)
    if (not math.isfinite(next_n) or next_n < 0 or
            any(not math.isfinite(x) or x < 0 for x in next_c)):
        raise ArithmeticError("nonphysical implicit kinetics state")
    return KineticsState(state.time_s + dt, next_n, next_c)


def advance_kinetics(
    state: KineticsState, coeffs: KineticsCoefficients,
    duration_s: float, *, nominal_substeps: int = 1,
    max_substeps: int = 20000
) -> KineticsStep:
    """Fully implicit Euler with positivity protection.

    This protection does NOT establish temporal accuracy. Refine timestep
    independently. Coefficients are frozen throughout the requested duration.
    """
    if len(state.precursors) != len(coeffs.beta_groups):
        raise ValueError("delayed-group count mismatch")
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("invalid duration")
    if not isinstance(nominal_substeps, int) or nominal_substeps < 1:
        raise ValueError("nominal_substeps must be positive integer")
    if not isinstance(max_substeps, int) or max_substeps < 1:
        raise ValueError("max_substeps must be positive integer")
    count = nominal_substeps
    if coeffs.reactivity > 0:
        count = max(count, math.ceil(duration_s * coeffs.reactivity
                                    / (0.5 * coeffs.generation_time_s)))
    if count > max_substeps:
        raise ValueError("positive reactivity demands too many substeps")
    dt = duration_s / count
    current = state
    max_n_res = max_c_res = 0.0
    for _ in range(count):
        following = _backward_euler_substep(current, coeffs, dt)
        dn, dc = kinetics_rhs(following, coeffs)
        max_n_res = max(max_n_res,
                        abs((following.amplitude - current.amplitude) / dt - dn))
        for old, new, rate in zip(current.precursors, following.precursors, dc):
            max_c_res = max(max_c_res, abs((new - old) / dt - rate))
        current = following
    return KineticsStep(
        KineticsState(state.time_s + duration_s, current.amplitude,
                      current.precursors),
        count, max_n_res, max_c_res
    )


@dataclass(frozen=True)
class NormalizedShape:
    shape: np.ndarray
    amplitude: float
    weighted_normalization: float


def normalize_flux_shape(
    flux: Sequence[Sequence[float]] | np.ndarray,
    adjoint: Sequence[Sequence[float]] | np.ndarray,
    inverse_speeds_s_m: Sequence[float] | np.ndarray,
    cell_volumes_m3: Sequence[float] | np.ndarray,
) -> NormalizedShape:
    """Normalize a [cell, group] forward flux by external adjoint weights."""
    phi = np.asarray(flux, dtype=float)
    weight = np.asarray(adjoint, dtype=float)
    speed = np.asarray(inverse_speeds_s_m, dtype=float)
    volume = np.asarray(cell_volumes_m3, dtype=float)
    if phi.ndim != 2:
        raise ValueError("flux must be [cell,group]")
    if (weight.shape != phi.shape or speed.shape != (phi.shape[1],)
            or volume.shape != (phi.shape[0],)):
        raise ValueError("flux, adjoint, speed or mesh shape mismatch")
    for array in (phi, weight, speed, volume):
        if not np.all(np.isfinite(array)) or np.any(array < 0):
            raise ValueError("shape data must be finite and nonnegative")
    if np.any(speed <= 0) or np.any(volume <= 0):
        raise ValueError("inverse speeds and volumes must be positive")
    weights = weight * speed[None, :] * volume[:, None]
    amplitude = float(np.sum(phi * weights))
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("zero or invalid weighted flux amplitude")
    shape = phi / amplitude
    return NormalizedShape(shape, amplitude, float(np.sum(shape * weights)))


def reconstruct_flux(amplitude: float, shape: NormalizedShape) -> np.ndarray:
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError("invalid amplitude")
    if not np.isclose(shape.weighted_normalization, 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("shape normalization mismatch")
    return amplitude * shape.shape


def solve_piecewise_constant(
    start: KineticsState,
    segments: Sequence[tuple[float, KineticsCoefficients]],
    *, nominal_substeps: int = 1,
) -> tuple[KineticsState, ...]:
    """Integrate externally supplied effective parameters; no spatial solve."""
    states = [start]
    for duration, coeffs in segments:
        states.append(advance_kinetics(
            states[-1], coeffs, duration, nominal_substeps=nominal_substeps
        ).state)
    return tuple(states)
