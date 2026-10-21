"""Neutron-energy spectrum bins with exact thermal/epithermal/fast boundaries.

OpenMC EnergyFilter classifies incoming particle energy in eV. The NERVA
reporting convention uses thresholds at 0.625 eV and 100 keV; a tally
whose bins cross either threshold cannot assign those bin counts exactly.
No within-bin spectral shape is invented or interpolated.
"""
from __future__ import annotations

import math
import numpy as np

ENERGY_LIMITS_EV = (1.0e-5, 0.625, 1.0e5, 2.0e7)


def diagnostic_energy_edges(groups: int) -> np.ndarray:
    """Produce exactly `groups` log-spaced bins including both cutoffs.

    Each of the three energy regions receives at least one bin, and
    additional bins are apportioned by logarithmic energy width using
    largest remainders. Endpoints are inserted exactly, not via rounding.
    """
    if isinstance(groups, (bool, np.bool_)) or not isinstance(groups, (int, np.integer)) or groups < 3:
        raise ValueError("diagnostic energy group count must be integer >= 3")
    bounds = np.asarray(ENERGY_LIMITS_EV, dtype=float)
    widths = np.diff(np.log(bounds))
    additional = int(groups) - 3
    expected = additional * widths / widths.sum()
    extras = np.floor(expected).astype(int)
    remainder = additional - int(np.sum(extras))
    for segment in np.argsort(-(expected - extras), kind="stable")[:remainder]:
        extras[segment] += 1
    bins_per_region = extras + 1
    all_edges = []
    for region, count in enumerate(bins_per_region):
        edges = np.geomspace(bounds[region], bounds[region + 1], count + 1)
        edges[0] = bounds[region]
        edges[-1] = bounds[region + 1]
        all_edges.extend(edges if region == 0 else edges[1:])
    result = np.asarray(all_edges, dtype=float)
    if len(result) != groups + 1 or np.any(np.diff(result) <= 0):
        raise RuntimeError("non-monotone or incorrect neutron group allocation")
    return result


def spectral_fractions(energy_bins_ev: np.ndarray, spectrum: np.ndarray) -> dict[str, float]:
    """Return fractions only when OpenMC incident-energy bins are unambiguous.

    Fail closed if any bin straddles 0.625 eV or 100 keV; inventing a
    proportional split would assume unknown within-bin flux. Fractions
    are reported only for the energy span scored by this EnergyFilter.
    """
    bins = np.asarray(energy_bins_ev, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    if bins.ndim != 2 or bins.shape[1] != 2:
        raise ValueError("EnergyFilter bins must have shape (N, 2)")
    if values.ndim != 1 or len(values) != len(bins):
        raise ValueError("spectrum length must match energy bins")
    if bins.size == 0 or not np.isfinite(bins).all():
        raise ValueError("empty or nonfinite neutron energy bins")
    lo, hi = bins[:, 0], bins[:, 1]
    if np.any(lo < 0) or np.any(hi <= lo) or np.any(lo[1:] < hi[:-1]):
        raise ValueError("neutron-energy bins must be ascending, disjoint and nonempty")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("neutron spectrum must be finite and nonnegative")
    for cut in ENERGY_LIMITS_EV[1:3]:
        eps = max(math.ulp(cut) * 16.0, cut * 1e-12)
        if np.any((lo < cut - eps) & (hi > cut + eps)):
            raise ValueError(
                f"EnergyFilter group crosses {cut:g} eV spectrum boundary; "
                "regenerate OpenMC tallies with edge-aligned groups"
            )
    total = float(np.sum(values, dtype=float))
    keys = (
        "thermal_below_0p625_eV",
        "epithermal_0p625eV_to_100keV",
        "fast_above_100keV",
    )
    if total == 0:
        return dict.fromkeys(keys, 0.0)
    thermal = float(np.sum(values[hi <= 0.625 * (1 + 1e-12)]))
    fast = float(np.sum(values[lo >= 1e5 * (1 - 1e-12)]))
    epi = total - thermal - fast
    if epi < -1e-12 * total:
        raise ValueError("neutron-spectrum numerical partition is inconsistent")
    return dict(zip(keys, (thermal / total, max(0.0, epi) / total, fast / total)))
