"""Deterministic synthetic numerical tests; NO OpenMC transport is run."""
from __future__ import annotations

import math
import numpy as np

from .axial_covariance import (
    axial_mean_covariance, integrated_axial_fraction_sem,
)


def reject(fn, expected: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert expected in str(exc), str(exc)
    else:
        raise AssertionError("accepted invalid covariance input: " + expected)


def main() -> int:
    # Exactly normalized three-zone axial power with a random seed-dependent
    # transfer of heat between zones 1 and 3. Total deposited heat is fixed.
    shifts = np.asarray([-0.06, -0.02, 0.02, 0.06])
    samples = np.column_stack([
        (1 + shifts) / 3,
        np.ones(4) / 3,
        (1 - shifts) / 3,
    ])
    mean, covariance, sem = axial_mean_covariance(samples)
    analytic = math.sqrt(0.008 / 3) / 2 / 3
    assert np.allclose(mean, np.full(3, 1 / 3), atol=1e-14)
    assert math.isclose(sem[0], analytic, rel_tol=1e-12)
    assert sem[1] < 1e-14
    assert math.isclose(sem[2], analytic, rel_tol=1e-12)
    assert math.isclose(covariance[0, 2], -analytic**2, rel_tol=1e-12)
    assert np.allclose(covariance.sum(axis=0), 0, atol=1e-15)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-15

    early_sem = integrated_axial_fraction_sem(covariance, 0, 2)
    whole_sem = integrated_axial_fraction_sem(covariance, 0, 3)
    assert math.isclose(early_sem, analytic, rel_tol=1e-12)
    assert whole_sem < 1e-12
    assert math.sqrt(float(np.sum(sem**2))) > 0.01
    # The previous diagonal-only uncertainty sum would incorrectly report
    # nonzero error in an exactly conserved reactor-power fraction.

    # Under replica reordering, moments/covariances are unchanged.
    _, shuffled, shuffled_sem = axial_mean_covariance(samples[[2,0,3,1]])
    assert np.allclose(shuffled, covariance, atol=1e-15)
    assert np.allclose(shuffled_sem, sem, atol=1e-15)

    reject(lambda: axial_mean_covariance(samples[:1]), ">=2")
    reject(lambda: axial_mean_covariance(samples.reshape(2, 2, 3)), ">=2")
    reject(lambda: axial_mean_covariance(np.array([[float("nan")],[1.]])), "nonfinite")
    reject(lambda: axial_mean_covariance(np.array([[-.1,1.1],[.1,.9]])), "nonnegative")
    reject(lambda: axial_mean_covariance(np.array([[.4,.4],[.5,.5]])), "conserve")
    reject(lambda: integrated_axial_fraction_sem(covariance, 1, 1), "interval")
    reject(lambda: integrated_axial_fraction_sem(covariance, 0, 4), "interval")
    reject(lambda: integrated_axial_fraction_sem(
        np.array([[1.,2.],[0.,1.]]),0,2), "symmetric")
    reject(lambda: integrated_axial_fraction_sem(
        np.array([[1.,0.],[0.,-0.1]]),1,2), "negative variance")
    print("NERVA OpenMC axial-power covariance/normalization QA: PASS")
    print("  covariance-adjusted whole-core power-fraction SEM: 0")
    print("  genuine OpenMC neutron transport executed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
