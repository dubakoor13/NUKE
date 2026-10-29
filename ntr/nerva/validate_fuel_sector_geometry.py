"""Validate deterministic NERVA fuel-sector areas."""

from __future__ import annotations

import math

import numpy as np

from .config import NervaConfig
from .fuel_sector_geometry import (
    fuel_sector_equivalent_outer_radii_m,
    fuel_sector_fuel_areas_cm2,
    fuel_sector_total_areas_cm2,
    fuel_sector_coated_channel_areas_cm2,
)


def main() -> int:
    config = NervaConfig()

    total = fuel_sector_total_areas_cm2(config)
    fuel = fuel_sector_fuel_areas_cm2(config)
    radii = fuel_sector_equivalent_outer_radii_m(config)

    assert total.shape == (19,)
    assert fuel.shape == (19,)
    assert radii.shape == (19,)
    assert np.all(total > 0.0)
    assert np.all(fuel > 0.0)

    hex_area = (
        3.0
        * math.sqrt(3.0)
        / 2.0
        * config.inner_fuel_edge_length_cm**2
    )
    coated_footprints = fuel_sector_coated_channel_areas_cm2(config)
    expected_fuel_area = hex_area - float(np.sum(coated_footprints))
    legacy_undercount = expected_fuel_area - (
        hex_area - 19.0*math.pi*config.coated_channel_radius_cm**2
    )
    # Six actual y-oriented hex faces clip annuli: previously this
    # geometric fuel area was silently discarded by the thermal model.
    assert np.isclose(legacy_undercount,0.0008995304697577,rtol=0,atol=2e-13)
    assert np.count_nonzero(
        coated_footprints < math.pi*config.coated_channel_radius_cm**2-1e-13
    ) == 6

    assert np.isclose(
        np.sum(total),
        hex_area,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert np.isclose(
        np.sum(fuel),
        expected_fuel_area,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    inner_radius_m = config.coated_channel_radius_cm / 100.0
    reconstructed = (
        math.pi
        * (radii**2 - inner_radius_m**2)
        * 1.0e4
    )
    assert np.allclose(
        reconstructed,
        fuel,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    print("NERVA deterministic fuel-sector area validation: PASS")
    print(f"  total inner-hex area: {hex_area:.9f} cm2")
    print(f"  fuel-material area: {expected_fuel_area:.9f} cm2")
    print(f"  historical fuel-sector area undercount corrected: {legacy_undercount:.12f} cm2")
    print("  actual CSG clipped coated channels: 6")
    print(
        f"  sector fuel area range: "
        f"{np.min(fuel):.9f}..{np.max(fuel):.9f} cm2"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
