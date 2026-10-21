"""Covariance-preserving uncertainty for independently sampled axial heat maps.

Power-normalized OpenMC mesh fractions sum to one IN EACH realization.
Their axial-bin errors are therefore correlated; summing independently
reported standard errors is incorrect. This module estimates covariance
from separate, identically configured, independently seeded replicas.

It does not establish neutron source convergence, tally independence,
nuclear-data accuracy, or any historical NERVA benchmark agreement.
"""
from __future__ import annotations

import math
import numpy as np


def axial_mean_covariance(
    axial_fractions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return axial mean fractions, covariance of the MEAN and axial SEM.

    The input has shape (replicas, axial_bins) and consists of NONNEGATIVE
    energy-deposition fractions normalized to unit total for every seed.

    Cov(mean) = (D^T D) / [n * (n-1)], where D contains centered rows.
    This empirical estimate preserves anticorrelation due to fixed total
    thermal power and requires at least two independent realizations.
    """
    data = np.asarray(axial_fractions, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 1:
        raise ValueError("axial fractions require >=2 replicas and >=1 bins")
    if not np.isfinite(data).all():
        raise ValueError("axial fractions contain nonfinite values")
    if np.any(data < 0):
        raise ValueError("axial fractions must be nonnegative")
    if not np.allclose(
        np.sum(data, axis=1), 1.0, rtol=0.0, atol=1e-7
    ):
        raise ValueError("each replica must conserve its axial power fraction")
    n = data.shape[0]
    mean = np.mean(data, axis=0)
    centered = data - mean
    covariance = centered.T @ centered / (n * (n - 1))
    covariance = 0.5 * (covariance + covariance.T)
    sem = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return mean, covariance, sem


def integrated_axial_fraction_sem(
    covariance_of_mean: np.ndarray,
    start: int,
    stop: int,
) -> float:
    """Sample-mean SEM for sum of axial fractions in [start, stop).

    Includes off-diagonal covariance, unlike quadrature summation of
    individual bin SEM values. Returns zero for exact conservation of all
    power fractions; slight negative roundoff is clamped to zero.
    """
    covariance = np.asarray(covariance_of_mean, dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] < 1
            or covariance.shape[0] != covariance.shape[1]
            or not np.isfinite(covariance).all()):
        raise ValueError("axial covariance matrix must be finite and square")
    n = covariance.shape[0]
    if (isinstance(start, (bool, np.bool_))
            or isinstance(stop, (bool, np.bool_))
            or not isinstance(start, (int, np.integer))
            or not isinstance(stop, (int, np.integer))
            or not (0 <= start < stop <= n)):
        raise ValueError("invalid axial interval")
    # Covariance is a numerical sample covariance; reject obviously
    # non-symmetric matrices to avoid misleading region uncertainties.
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if not np.allclose(covariance, covariance.T, atol=1e-12 * scale, rtol=0):
        raise ValueError("axial covariance matrix is not symmetric")
    variance = float(
        np.sum(covariance[start:stop, start:stop], dtype=np.float64)
    )
    tolerance = 1e-12 * max(
        float(np.sum(np.abs(covariance[start:stop, start:stop]))),
        1e-30,
    )
    if variance < -tolerance:
        raise ValueError("axial integrated covariance implies negative variance")
    return math.sqrt(max(variance, 0.0))
