"""Validate the four requested NTP/LOX propulsion families."""

from __future__ import annotations

import math

from .propulsion_modes import (
    FOUR_PROPULSION_MODES,
    LCH4_LOX_AUGMENTED,
    LCH4_NTR,
    LH2_LOX_LANTR,
    LH2_NTR,
    lch4_lox_mass_bookkeeping,
    lch4_ntp_reference,
    lh2_lox_lantr_reference,
    lh2_ntp_reference,
)


def main() -> int:
    assert len(FOUR_PROPULSION_MODES) == 4
    assert LH2_NTR.lox_augmented is False
    assert LCH4_NTR.lox_augmented is False
    assert LH2_LOX_LANTR.lox_augmented is True
    assert LCH4_LOX_AUGMENTED.lox_augmented is True

    h2 = lh2_ntp_reference()
    assert h2.isp_s == 900.0
    assert h2.chamber_temperature_k == 2700.0
    assert h2.thrust_n is None

    ch4_1 = lch4_ntp_reference(1)
    ch4_2 = lch4_ntp_reference(2)
    assert ch4_1.isp_s == 353.71
    assert ch4_2.isp_s == 400.16
    assert ch4_1.chamber_pressure_mpa == 4.60
    assert ch4_2.chamber_pressure_mpa == 4.60
    assert ch4_2.thrust_n > ch4_1.thrust_n

    lantr0 = lh2_lox_lantr_reference(0.0)
    lantr3 = lh2_lox_lantr_reference(3.0)
    assert lantr0.isp_s == 900.0
    assert lantr3.isp_s == 588.0
    assert lantr3.thrust_n > lantr0.thrust_n

    high3 = lh2_lox_lantr_reference(3.0, high_pressure=True)
    assert high3.isp_s == 647.0
    assert math.isclose(high3.chamber_pressure_mpa, 13.789514586336)
    assert high3.thrust_n is None

    ch4lox = lch4_lox_mass_bookkeeping(1.0, methane_case_number=2)
    assert math.isclose(ch4lox.nuclear_propellant_flow_kg_s, 17.43)
    assert math.isclose(ch4lox.oxygen_flow_kg_s, 17.43)
    assert math.isclose(ch4lox.total_flow_kg_s, 34.86)
    assert ch4lox.isp_s is None
    assert ch4lox.thrust_n is None

    print("Four propulsion-mode validation: PASS")
    print("  LH2 NTR: source-backed reference")
    print("  LCH4 NTR: source-backed PBM cases 1/2")
    print("  LH2+LOX: source-backed NASA LANTR tables")
    print("  LCH4+LOX: mass bookkeeping only; chemistry required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
