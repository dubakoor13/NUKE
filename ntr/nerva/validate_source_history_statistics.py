"""Synthetic regression for OpenMC generation-history serial correlation.

Runs no OpenMC transport. Distinguishes correlated histories from nominal
independent sampling without claiming a validated nuclear reactor result.
"""
from __future__ import annotations

import math
import numpy as np

from .source_history_statistics import analyze_correlated_history


def reject(run, contains: str) -> None:
    try:
        run()
    except ValueError as exc:
        assert contains in str(exc), repr(exc)
    else:
        raise AssertionError("invalid generation history accepted: " + contains)


def main() -> int:
    random = np.random.default_rng(20261021)
    white = random.normal(0.0, 1.0, 500)
    correlated = np.zeros(500, dtype=float)
    innovations = random.normal(0.0, 1.0, 500)
    for i in range(1, 500):
        correlated[i] = 0.93 * correlated[i - 1] + innovations[i]

    independent = analyze_correlated_history(white)
    serial = analyze_correlated_history(correlated)
    assert independent["convergence_proven"] is False
    assert serial["convergence_proven"] is False
    assert independent["effective_active_sample_count"] > 0
    assert serial["integrated_autocorrelation_time"] > 3
    assert serial["effective_active_sample_count"] < 500 / 3
    assert serial["correlation_lags_used"] >= 2
    assert math.isfinite(serial["last_window_standard_deviation"])

    shifted = analyze_correlated_history(
        np.concatenate([np.zeros(20), np.linspace(2, 3, 80)]),
        inactive=20,
    )
    assert shifted["active_count"] == 80
    assert shifted["last_window_mean_shift"] > 0
    assert shifted["last_window_count"] == 10

    constant = analyze_correlated_history(np.full(50, 2.5))
    assert constant["effective_active_sample_count"] is None
    assert constant["integrated_autocorrelation_time"] is None
    assert "Constant" in constant["diagnostic_note"]
    short = analyze_correlated_history(np.ones(7))
    assert short["effective_active_sample_count"] is None
    assert "Too few" in short["diagnostic_note"]
    empty = analyze_correlated_history(np.array([]))
    assert empty["active_count"] == 0
    assert empty["active_mean"] is None

    for bad, msg in (
        (np.ones((2, 10)), "one-dimensional"),
        (np.array([1, np.nan, 2]), "nonfinite"),
        (np.array([1, np.inf, 2]), "nonfinite"),
    ):
        reject(lambda a=bad: analyze_correlated_history(a), msg)
    reject(lambda: analyze_correlated_history(white, inactive=501), "inactive")
    reject(lambda: analyze_correlated_history(white, inactive=-1), "inactive")
    reject(lambda: analyze_correlated_history(white, comparison_window=2), "comparison_window")
    reject(lambda: analyze_correlated_history(white, maximum_lag=0), "maximum_lag")
    print("OpenMC fission-source serial-correlation numerical regression: PASS")
    print("  real neutron transport executed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
