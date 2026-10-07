"""Validate public NERVA reference presets and unit conversions."""

from __future__ import annotations

import math

from .historical_presets import (
    NERVA_75K_1972,
    NESS_R1,
    XE_PRIME,
)


def main() -> int:
    assert math.isclose(XE_PRIME.thermal_power_mw, 1140.0)
    assert math.isclose(XE_PRIME.thrust_n / 1000.0, 246.431477, rel_tol=1e-6)
    assert math.isclose(XE_PRIME.hydrogen_mass_flow_kg_s, 31.842184, rel_tol=1e-6)
    assert math.isclose(XE_PRIME.chamber_temperature_k, 2280.555556, rel_tol=1e-6)
    assert math.isclose(XE_PRIME.chamber_pressure_pa / 1e6, 3.901054, rel_tol=1e-6)
    assert XE_PRIME.isp_s == 710.0
    assert XE_PRIME.fuel_element_count == 1584
    assert XE_PRIME.coolant_channels_per_element == 19
    assert math.isclose(XE_PRIME.active_length_cm, 132.08)

    assert math.isclose(
        NERVA_75K_1972.thrust_n / 1000.0,
        333.616621,
        rel_tol=1e-6,
    )
    assert NERVA_75K_1972.isp_range_s == (825.0, 850.0)
    assert NERVA_75K_1972.chamber_temperature_range_k == (2350.0, 2500.0)
    assert math.isclose(
        NERVA_75K_1972.chamber_pressure_pa / 1e6,
        3.102641,
        rel_tol=1e-6,
    )

    assert math.isclose(NESS_R1.thermal_power_mw, 1500.0)
    assert math.isclose(NESS_R1.core_diameter_cm, 96.52)
    assert math.isclose(NESS_R1.active_length_cm, 132.08)
    assert NESS_R1.tie_heating_fraction_range == (0.03, 0.07)
    assert NESS_R1.reflector_heating_fraction_range == (0.01, 0.02)

    print("NERVA historical preset validation: PASS")
    print(f"  XE-Prime thrust: {XE_PRIME.thrust_n / 1000.0:.3f} kN")
    print(f"  XE-Prime Isp: {XE_PRIME.isp_s:.1f} s")
    print(
        f"  75-klbf NERVA Isp range: "
        f"{NERVA_75K_1972.isp_range_s[0]:.0f}-"
        f"{NERVA_75K_1972.isp_range_s[1]:.0f} s"
    )
    print(
        f"  NESS/R-1 core: {NESS_R1.core_diameter_cm:.2f} cm x "
        f"{NESS_R1.active_length_cm:.2f} cm"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
