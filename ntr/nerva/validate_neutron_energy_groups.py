"""Synthetic mathematical checks only; no NERVA neutron transport is run."""
from __future__ import annotations

import numpy as np
from .neutron_energy_groups import diagnostic_energy_edges, spectral_fractions


def reject(action, phrase: str) -> None:
    try:
        action()
    except ValueError as error:
        assert phrase in str(error), str(error)
    else:
        raise AssertionError("accepted invalid energy grid: " + phrase)


def main() -> int:
    for groups in (3, 8, 16, 80, 320):
        edges = diagnostic_energy_edges(groups)
        assert len(edges) == groups + 1
        assert np.all(np.diff(edges) > 0)
        assert edges[0] == 1.0e-5 and edges[-1] == 2e7
        assert np.count_nonzero(edges == 0.625) == 1
        assert np.count_nonzero(edges == 1e5) == 1
        bins = np.column_stack([edges[:-1], edges[1:]])
        exact = spectral_fractions(bins, np.ones(groups))
        assert np.isclose(sum(exact.values()), 1.0)
        assert all(value > 0 for value in exact.values())
        assert spectral_fractions(bins, np.zeros(groups)) == {
            "thermal_below_0p625_eV": 0.0,
            "epithermal_0p625eV_to_100keV": 0.0,
            "fast_above_100keV": 0.0,
        }

    simple = np.array([[1e-5,0.625],[0.625,1e5],[1e5,2e7]])
    f = spectral_fractions(simple, np.array([2.,3.,5.]))
    assert np.isclose(f["thermal_below_0p625_eV"], .2)
    assert np.isclose(f["epithermal_0p625eV_to_100keV"], .3)
    assert np.isclose(f["fast_above_100keV"], .5)
    reject(lambda: spectral_fractions(np.array([[.1,1],[1,2]]),np.array([1.,1.])), "0.625")
    reject(lambda: spectral_fractions(np.array([[1e4,2e5]]), np.array([1.])), "100000")
    reject(lambda: spectral_fractions(simple,np.array([1.,-1.,3.])), "nonnegative")
    reject(lambda: spectral_fractions(simple,np.array([1.,np.nan,3.])), "nonnegative")
    reject(lambda: spectral_fractions(simple,np.array([1.,3.])), "length")
    reject(lambda: spectral_fractions(simple[::-1],np.ones(3)), "ascending")
    reject(lambda: diagnostic_energy_edges(2), "integer >= 3")
    reject(lambda: diagnostic_energy_edges(3.0), "integer >= 3")
    print("NERVA exact-boundary spectral tallies and partition QA: PASS")
    print("  real OpenMC neutron transport executed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
