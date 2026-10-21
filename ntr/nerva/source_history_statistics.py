"""Serial-correlation diagnostics for OpenMC eigenvalue generation histories.

This is *diagnostic* statistics, not a proof of source stationarity.
Consecutive Monte Carlo fission-source generations are not independent,
so s/sqrt(n) is not generally a valid estimate for their mean.
No eigenvalue, material, geometry or drum-setting optimization is performed.
"""
from __future__ import annotations

import math
import numpy as np


def analyze_correlated_history(
    history: np.ndarray,
    *,
    inactive: int = 0,
    comparison_window: int = 10,
    maximum_lag: int = 50,
) -> dict:
    """Inspect an ordered numerical generation series with no dropped samples.

    The integrated autocorrelation time (IAT) uses positive-sequence lag
    correlation truncated by a self-consistent 5*tau window. Its ESS and
    window z-ratio are approximate indicators, NOT confidence statements.
    A constant sequence has no estimable autocorrelation or meaningful ESS.
    The last two *nonoverlapping* windows are only an early drift indicator.
    """
    data = np.asarray(history, dtype=float)
    if data.ndim != 1:
        raise ValueError("OpenMC generation history must be one-dimensional")
    if not np.isfinite(data).all():
        raise ValueError("OpenMC generation history contains nonfinite values")
    if not isinstance(inactive, (int, np.integer)) or not 0 <= inactive <= data.size:
        raise ValueError("invalid inactive generation count")
    if not isinstance(comparison_window, (int, np.integer)) or comparison_window < 3:
        raise ValueError("comparison_window must be at least three")
    if not isinstance(maximum_lag, (int, np.integer)) or maximum_lag < 1:
        raise ValueError("maximum_lag must be positive")

    active = data[inactive:]
    n = len(active)
    output = {
        "total_history_count": int(data.size),
        "excluded_inactive_count": int(inactive),
        "active_count": int(n),
        "active_mean": float(np.mean(active)) if n else None,
        "active_standard_deviation": (
            float(np.std(active, ddof=1)) if n > 1 else None
        ),
        "integrated_autocorrelation_time": None,
        "effective_active_sample_count": None,
        "correlation_lags_used": 0,
        "last_window_count": 0,
        "last_window_mean_shift": None,
        "window_shift_over_correlated_noise": None,
        "last_window_standard_deviation": None,
        "diagnostic_note": None,
        "convergence_proven": False,
    }
    if n < 8:
        output["diagnostic_note"] = "Too few active generations to estimate serial correlation"
        return output

    center = active - np.mean(active)
    variance = float(np.dot(center, center) / n)
    tau = None
    if variance > 0 and math.isfinite(variance):
        # Biased autocovariances (divide by n), avoiding growing variance
        # with lag. Stop at the first nonpositive correlation, to avoid
        # mistaking a cancellation of noisy lag estimates for independence.
        limit = min(int(maximum_lag), n // 4)
        rho_sum = 0.0
        used = 0
        for lag in range(1, limit + 1):
            autocov = float(np.dot(center[:-lag], center[lag:]) / n)
            rho = autocov / variance
            if not math.isfinite(rho) or rho <= 0:
                break
            rho_sum += rho
            used = lag
            candidate = 1.0 + 2.0 * rho_sum
            if lag >= 5.0 * candidate:
                break
        tau = max(1.0, 1.0 + 2.0 * rho_sum)
        output["integrated_autocorrelation_time"] = float(tau)
        output["effective_active_sample_count"] = float(n / tau)
        output["correlation_lags_used"] = used
        if used == limit and limit > 0:
            output["diagnostic_note"] = (
                "Positive serial correlation reached truncation bound; "
                "effective sample count may be optimistic"
            )
    else:
        output["diagnostic_note"] = (
            "Constant active history: autocorrelation/ESS unidentifiable; "
            "zero fluctuation is not source-convergence evidence"
        )

    window = min(int(comparison_window), n // 2)
    if window >= 3:
        previous = active[-2 * window:-window]
        latest = active[-window:]
        shift = float(np.mean(latest) - np.mean(previous))
        output["last_window_count"] = window
        output["last_window_mean_shift"] = shift
        output["last_window_standard_deviation"] = float(
            np.std(latest, ddof=1)
        )
        # A conservative approximate drift/noise ratio: the overall IAT
        # inflates BOTH means' sample variance. Not a hypothesis test;
        # covariance between adjacent windows remains unmodeled.
        if tau is not None:
            variance_of_difference = (tau / window) * (
                float(np.var(previous, ddof=1)) +
                float(np.var(latest, ddof=1))
            )
            if variance_of_difference > 0 and math.isfinite(variance_of_difference):
                output["window_shift_over_correlated_noise"] = (
                    abs(shift) / math.sqrt(variance_of_difference)
                )
    return output


def batch_average_generation_k(
    k_generation: np.ndarray,
    *,
    batches: int,
    generations_per_batch: int = 1,
) -> np.ndarray:
    """Make exactly one per-batch k sample from per-generation estimators.

    OpenMC k_generation can have batches*generations_per_batch values.
    Dropping the first n_inactive GENERATIONS rather than inactive BATCHES
    corrupts the active statistical interval when generations_per_batch > 1.
    """
    if not isinstance(batches, (int, np.integer)) or batches < 1:
        raise ValueError("invalid OpenMC batch count")
    if (not isinstance(generations_per_batch, (int, np.integer))
            or generations_per_batch < 1):
        raise ValueError("invalid OpenMC generations-per-batch")
    values = np.asarray(k_generation, dtype=float)
    if values.size == 0:
        return np.asarray([], dtype=float)
    if (values.ndim == 1
            and values.size == batches * generations_per_batch):
        matrix = values.reshape(batches, generations_per_batch)
    elif values.ndim == 2 and values.shape == (batches, generations_per_batch):
        matrix = values
    else:
        raise ValueError(
            "OpenMC k_generation shape incompatible with "
            "batches and generations_per_batch"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite OpenMC k_generation")
    return np.mean(matrix, axis=1)
