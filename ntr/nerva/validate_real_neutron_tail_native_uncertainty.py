"""Synthetic adversarial tests for actual OpenMC neutron tail moment auditor.

The raw-moment arithmetic is tested here. The physical validation gate is
the separate CI step running all eight compiled OpenMC HDF5 statepoints.
"""
from __future__ import annotations

import math
import numpy as np

from .real_neutron_tail_native_uncertainty import native_batch_moments


def reject(fn, message: str) -> None:
    try:
        fn()
    except ValueError as err:
        assert message in str(err), (message, str(err))
    else:
        raise AssertionError("invalid native OpenMC tally moments accepted")


def main() -> int:
    n = 30
    true_mean = np.array([
        [0., 2.465587e-8, -0.2],
        [0.03, 0.25, 0.5],
    ])
    true_sem = np.array([
        [0., 2.465587e-8, 0.05],
        [0.01, 0.07, 0.04],
    ])
    raw = np.empty((2, 3, 2), dtype=float)
    raw[:, :, 0] = n * true_mean
    raw[:, :, 1] = (
        n * true_mean**2 + n*(n-1)*true_sem**2
    )
    mean, sem = native_batch_moments(raw, n)
    assert np.allclose(mean, true_mean, rtol=1e-13, atol=1e-22)
    assert np.allclose(sem, true_sem, rtol=1e-12, atol=1e-22)
    assert mean[0,0] == sem[0,0] == 0.
    assert sem[0,1] > 0. and sem[0,1]/mean[0,1] > .99
    assert math.isclose(sem[0,2], .05, rel_tol=1e-12)

    reject(lambda: native_batch_moments(raw, 2),
           "invalid native OpenMC scored")
    reject(lambda: native_batch_moments(raw[:,:,:1], n),
           "invalid native OpenMC scored")
    corrupted = raw.copy()
    corrupted[0,1,1] = -0.01
    reject(lambda: native_batch_moments(corrupted,n),
           "negative native squared")
    corrupted = raw.copy()
    corrupted[1,1,1] = corrupted[1,1,0]**2/n - 0.01
    reject(lambda: native_batch_moments(corrupted,n),
           "second moments violate")
    corrupted = raw.copy()
    corrupted[0,1,0] = float("nan")
    reject(lambda: native_batch_moments(corrupted,n),
           "invalid native OpenMC scored")
    # Small cancellation from hardware floating-point accumulation may be
    # clamped; a significant violation may NOT.
    very_small = np.zeros((1,1,2))
    very_small[0,0,0] = 30.e-9
    very_small[0,0,1] = (30.e-9)**2/30*(1.-1e-14)
    x,s = native_batch_moments(very_small,30)
    assert math.isclose(x[0,0], 1e-9) and s[0,0] == 0.
    print("OpenMC native neutron energy-tail SUM/SUMSQ regression: PASS (synthetic)")
    print("Actual neutron transport must be checked in the compiled C++ workflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
