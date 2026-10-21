"""Read-only numerical diagnostics from a REAL OpenMC eigenvalue statepoint.

No criticality, loading, enrichment, or drum-position search is performed.
These diagnostics cannot alone establish source convergence or NASA fidelity.
The caller is responsible for binding the report to a statepoint SHA-256.
"""
from __future__ import annotations

import math

import numpy as np

from .source_history_statistics import (
    analyze_correlated_history,
    batch_average_generation_k,
)


def _finite_scalar(value: object, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid OpenMC {name}") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"invalid OpenMC {name}")
    return number


def summarize_core_statepoint(statepoint: object) -> dict:
    """Summarize observables; NEVER substitute a small entropy drift for proof.

    Requires a genuine StatePoint in the production caller. Synthetic objects
    are permitted only in the separately labeled CI unit test.
    """
    if statepoint.run_mode != "eigenvalue":
        raise ValueError("NERVA core diagnostic requires eigenvalue mode")
    batches = int(statepoint.n_batches)
    inactive = int(statepoint.n_inactive)
    particles = int(statepoint.n_particles)
    if batches < 1 or not 0 <= inactive < batches or particles <= 0:
        raise ValueError("invalid OpenMC generation counts")
    keff = statepoint.keff
    if keff is None:
        raise ValueError("OpenMC eigenvalue estimator missing")
    k = _finite_scalar(keff.nominal_value, "keff", positive=True)
    sigma = _finite_scalar(keff.std_dev, "keff uncertainty")
    if sigma < 0:
        raise ValueError("negative OpenMC keff standard deviation")

    ent = getattr(statepoint, "entropy", None)
    if ent is None:
        values = None
        entropy_error = "No Shannon source entropy available"
    else:
        values = np.asarray(ent, dtype=float)
        if values.ndim != 1 or len(values) != batches or not np.isfinite(values).all():
            raise ValueError("invalid OpenMC source entropy history")
        entropy_error = None

    drift = None
    late_spread = None
    if values is not None:
        active = values[inactive:]
        half_window = min(10, len(active) // 2)
        if half_window >= 3:
            previous = active[-2 * half_window : -half_window]
            latest = active[-half_window:]
            drift = float(np.mean(latest) - np.mean(previous))
            late_spread = float(np.std(latest, ddof=1))
        else:
            entropy_error = "Too few active entropy samples for two windows"

    entropy_correlation = (
        analyze_correlated_history(values, inactive=inactive)
        if values is not None
        else None
    )
    k_history = getattr(statepoint, "k_generation", None)
    k_correlation = None
    if k_history is not None:
        k_batched = batch_average_generation_k(
            k_history,
            batches=batches,
            generations_per_batch=int(
                getattr(statepoint, "generations_per_batch", 1)
            ),
        )
        if k_batched.size:
            k_correlation = analyze_correlated_history(
                k_batched, inactive=inactive
            )

    seeds = getattr(statepoint, "seed", None)
    return {
        "schema_version": 1,
        "run_mode": "eigenvalue",
        "batches": batches,
        "inactive_batches": inactive,
        "active_batches": batches - inactive,
        "particles_per_generation": particles,
        "random_seed": None if seeds is None else int(seeds),
        "keff_mean": k,
        "keff_std_dev": sigma,
        "source_entropy_last": None if values is None else float(values[-1]),
        "source_entropy_active_window_shift": drift,
        "source_entropy_latest_window_std": late_spread,
        "source_entropy_note": entropy_error,
        "source_entropy_serial_correlation": entropy_correlation,
        "per_batch_keff_serial_correlation": k_correlation,
        "source_convergence_independently_demonstrated": False,
        "historical_nasa_benchmark_validated": False,
        "interpretation": (
            "Descriptive statistical observables from this OpenMC statepoint, "
            "NOT a proof of source convergence, criticality calibration, "
            "reactor design suitability, or historical NASA benchmark agreement. "
            "Use multiple independent seeds and source/mesh sensitivity studies. "
            "Reported autocorrelation time and effective sample count are "
            "heuristic numerical diagnostics, not uncertainty-certified acceptance."
        ),
    }
