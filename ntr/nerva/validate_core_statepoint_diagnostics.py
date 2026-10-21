"""Synthetic fixture validation of OpenMC core statepoint QA extraction.

Does NOT run neutron transport or claim reactor validation.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from .core_statepoint_diagnostics import summarize_core_statepoint


def rejected(statepoint):
    try:
        summarize_core_statepoint(statepoint)
    except ValueError:
        return
    raise AssertionError("invalid OpenMC statepoint metadata accepted")


def main() -> int:
    fake = SimpleNamespace(
        run_mode="eigenvalue",
        n_batches=60,
        n_inactive=20,
        n_particles=4000,
        keff=SimpleNamespace(nominal_value=0.94, std_dev=0.003),
        entropy=np.array([2.0] * 20 + [3.0] * 40),
        seed=1234,
    )
    result = summarize_core_statepoint(fake)
    assert result["batches"] == 60
    assert result["active_batches"] == 40
    assert result["random_seed"] == 1234
    assert result["keff_mean"] == 0.94
    assert result["keff_std_dev"] == 0.003
    assert math.isclose(result["source_entropy_active_window_shift"], 0)
    assert not result["source_convergence_independently_demonstrated"]
    assert not result["historical_nasa_benchmark_validated"]

    for changed in (
        {"run_mode": "fixed source"},
        {"n_inactive": 60},
        {"n_particles": 0},
        {"keff": SimpleNamespace(nominal_value=float("nan"), std_dev=0.01)},
        {"keff": SimpleNamespace(nominal_value=1.0, std_dev=-0.01)},
        {"entropy": np.full(60, float("nan"))},
        {"entropy": np.ones(12)},
    ):
        rejected(SimpleNamespace(**{**vars(fake), **changed}))
    no_entropy = SimpleNamespace(**{**vars(fake), "entropy": None})
    incomplete = summarize_core_statepoint(no_entropy)
    assert incomplete["source_entropy_active_window_shift"] is None
    assert incomplete["source_entropy_note"]

    print("NERVA OpenMC core statepoint numerical QA fixtures: PASS")
    print("  real OpenMC transport executed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
