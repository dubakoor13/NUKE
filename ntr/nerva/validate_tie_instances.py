"""Validate per-tie OpenMC thermal reconstruction helpers."""

from __future__ import annotations

import numpy as np

from .solve_tie_instances import _scale_to_total


def main() -> int:
    values = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 3.0],
        ]
    )
    scaled = _scale_to_total(values, 100.0)
    assert np.isclose(np.sum(scaled), 100.0)
    assert np.all(scaled >= 0.0)

    zero = _scale_to_total(values, 0.0)
    assert np.all(zero == 0.0)

    try:
        _scale_to_total(np.zeros_like(values), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "positive target with zero source should fail"
        )

    print("NERVA per-tie reconstruction validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
