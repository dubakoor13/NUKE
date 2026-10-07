"""Validate parallel-flow redistribution helper math."""

from __future__ import annotations

import numpy as np

from .parallel_flow import (
    pressure_spread_fraction,
    update_parallel_mass_flow,
)


def main() -> int:
    resistance = np.array([1.0, 4.0, 9.0])
    current = np.array([1.0, 1.0, 1.0])
    delta_p = resistance * current**2

    result = update_parallel_mass_flow(
        current,
        delta_p,
        total_mass_flow_kg_s=3.0,
        relaxation=1.0,
    )

    expected = 1.0 / np.sqrt(resistance)
    expected *= 3.0 / np.sum(expected)

    assert np.allclose(
        result.mass_flow_kg_s,
        expected,
        rtol=1.0e-12,
    )
    assert np.isclose(
        np.sum(result.mass_flow_kg_s),
        3.0,
        rtol=1.0e-12,
    )

    balanced_dp = resistance * result.mass_flow_kg_s**2
    assert pressure_spread_fraction(balanced_dp) < 1.0e-12

    relaxed = update_parallel_mass_flow(
        current,
        delta_p,
        total_mass_flow_kg_s=3.0,
        relaxation=0.5,
    )
    assert np.isclose(
        np.sum(relaxed.mass_flow_kg_s),
        3.0,
        rtol=1.0e-12,
    )
    assert relaxed.max_relative_change > 0.0

    print("NERVA parallel-flow balancing validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
