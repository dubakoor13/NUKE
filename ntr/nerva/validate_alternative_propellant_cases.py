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

    # Source diagram/state-table mass conservation.
    # Parallel regenerative/core branches: state 4 -> states 5 and 9 -> state 12.
    assert math.isclose(
        case.state(5).mass_flow_kg_s + case.state(9).mass_flow_kg_s,
        case.state(12).mass_flow_kg_s,
        rel_tol=0.0,
        abs_tol=0.02,
    )

    # Turbine/bypass split: state 12 -> states 15 and 17 -> state 19.
    assert math.isclose(
        case.state(15).mass_flow_kg_s + case.state(17).mass_flow_kg_s,
        case.state(19).mass_flow_kg_s,
        rel_tol=0.0,
        abs_tol=0.02,
    )

    # Hot chamber/nozzle path carries total flow.
    assert math.isclose(
        case.state(21).mass_flow_kg_s,
        case.state(22).mass_flow_kg_s,
        rel_tol=0.0,
        abs_tol=1.0e-9,
    )

    # Pump-side pressure is much higher than chamber pressure.
    assert case.maximum_system_pressure_mpa > case.chamber_pressure_mpa
    assert case.pump_pressure_rise_mpa > 10.0

    # Turbine has positive margin over pump requirement in both source cases.
    assert case.turbine_power_kw - case.pump_power_kw > 0.0

    # Thrust is a derived quantity from the source mdot and Isp.
    assert case.derived_thrust_n > 0.0


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

    assert 67.0 < METHANE_CASE_1.derived_thrust_n / 1000.0 < 69.0
    assert 68.0 < METHANE_CASE_2.derived_thrust_n / 1000.0 < 69.0

    print("Alternative-propellant NTP four-case validation: PASS")
    print(f"  H-NTP: {H_NTP.reference_isp_s:.0f} s @ {H_NTP.reference_temperature_k:.0f} K")
    print(f"  A-NTP: {A_NTP.reference_isp_s:.0f} s @ {A_NTP.reference_temperature_k:.0f} K")
    print(
        f"  M-NTP case 1: {METHANE_CASE_1.isp_s:.2f} s, "
        f"{METHANE_CASE_1.reactor_power_mw:.2f} MW, "
        f"{METHANE_CASE_1.total_mass_flow_kg_s:.2f} kg/s, "
        f"{METHANE_CASE_1.derived_thrust_n/1000.0:.2f} kN"
    )
    print(
        f"  M-NTP case 2: {METHANE_CASE_2.isp_s:.2f} s, "
        f"{METHANE_CASE_2.reactor_power_mw:.2f} MW, "
        f"{METHANE_CASE_2.total_mass_flow_kg_s:.2f} kg/s, "
        f"{METHANE_CASE_2.derived_thrust_n/1000.0:.2f} kN"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
