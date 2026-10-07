"""Validate the four source-backed alternative-propellant NTP cases."""

from __future__ import annotations

import math

from .alternative_propellant_cases import (
    A_NTP,
    FOUR_CASES,
    H_NTP,
    METHANE_CASE_1,
    METHANE_CASE_2,
)


def _validate_methane_case(case) -> None:
    assert len(case.states) == 19
    assert case.state(21).pressure_mpa == 4.60
    assert math.isclose(
        case.state(21).temperature_k,
        case.chamber_temperature_k,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    assert math.isclose(
        case.state(1).mass_flow_kg_s,
        case.total_mass_flow_kg_s,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    assert math.isclose(
        case.state(22).mass_flow_kg_s,
        case.total_mass_flow_kg_s,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    assert math.isclose(
        case.state(3).pressure_mpa,
        case.maximum_system_pressure_mpa,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )
    assert case.pump_power_kw < case.turbine_power_kw
    assert case.turbine_pressure_ratio == 1.40


def main() -> int:
    assert len(FOUR_CASES) == 4

    assert H_NTP.propellant == "LH2"
    assert H_NTP.reference_isp_s == 900.0
    assert H_NTP.reference_temperature_k == 2700.0
    assert H_NTP.boost_pump is True
    assert H_NTP.serial_preheat is False

    assert A_NTP.propellant == "LNH3"
    assert A_NTP.reference_isp_s == 370.0
    assert A_NTP.reference_temperature_k == 2700.0
    assert A_NTP.boost_pump is False
    assert A_NTP.serial_preheat is True

    _validate_methane_case(METHANE_CASE_1)
    _validate_methane_case(METHANE_CASE_2)

    assert METHANE_CASE_1.isp_s == 353.71
    assert METHANE_CASE_2.isp_s == 400.16
    assert METHANE_CASE_1.reactor_power_mw == 177.65
    assert METHANE_CASE_2.reactor_power_mw == 214.66
    assert METHANE_CASE_1.maximum_system_pressure_mpa == 11.03
    assert METHANE_CASE_2.maximum_system_pressure_mpa == 11.28

    assert METHANE_CASE_2.chamber_temperature_k > METHANE_CASE_1.chamber_temperature_k
    assert METHANE_CASE_2.isp_s > METHANE_CASE_1.isp_s
    assert METHANE_CASE_2.total_mass_flow_kg_s < METHANE_CASE_1.total_mass_flow_kg_s
    assert METHANE_CASE_2.reactor_power_mw > METHANE_CASE_1.reactor_power_mw
    assert METHANE_CASE_2.pump_power_kw < METHANE_CASE_1.pump_power_kw

    print("Alternative-propellant NTP four-case validation: PASS")
    print(f"  H-NTP: {H_NTP.reference_isp_s:.0f} s @ {H_NTP.reference_temperature_k:.0f} K")
    print(f"  A-NTP: {A_NTP.reference_isp_s:.0f} s @ {A_NTP.reference_temperature_k:.0f} K")
    print(
        f"  M-NTP case 1: {METHANE_CASE_1.isp_s:.2f} s, "
        f"{METHANE_CASE_1.reactor_power_mw:.2f} MW, "
        f"{METHANE_CASE_1.total_mass_flow_kg_s:.2f} kg/s"
    )
    print(
        f"  M-NTP case 2: {METHANE_CASE_2.isp_s:.2f} s, "
        f"{METHANE_CASE_2.reactor_power_mw:.2f} MW, "
        f"{METHANE_CASE_2.total_mass_flow_kg_s:.2f} kg/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
