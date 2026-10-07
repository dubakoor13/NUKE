"""Regression checks for the public LANTR / bimodal NTR reference model."""

from __future__ import annotations

import math

from .lantr import (
    BASE_CHAMBER_PRESSURE_PA,
    BASE_ISP_S,
    BASE_REACTOR_EXIT_TEMPERATURE_K,
    BASE_REACTOR_POWER_MW,
    BIMODAL_POWER,
    lantr_point,
    lantr_table,
)


def main() -> int:
    table = lantr_table()
    assert len(table) == 6

    p0 = lantr_point(0.0)
    p3 = lantr_point(3.0)
    p5 = lantr_point(5.0)
    p25 = lantr_point(2.5)

    assert p0.delivered_isp_s == 900.0
    assert p0.thrust_augmentation_factor == 1.0
    assert math.isclose(p0.thrust_n / 1000.0, 73.3956567, rel_tol=1e-8)

    assert p3.delivered_isp_s == 588.0
    assert p3.thrust_augmentation_factor == 2.616
    assert math.isclose(p3.thrust_n / 1000.0, 191.99999, rel_tol=2e-4)

    assert p5.delivered_isp_s == 516.0
    assert p5.thrust_augmentation_factor == 3.441
    assert math.isclose(p5.thrust_n / 1000.0, 252.562, rel_tol=2e-4)

    assert 588.0 < p25.delivered_isp_s < 637.0
    assert 2.123 < p25.thrust_augmentation_factor < 2.616

    assert BASE_REACTOR_POWER_MW == 365.0
    assert BASE_REACTOR_EXIT_TEMPERATURE_K == 2734.0
    assert math.isclose(BASE_CHAMBER_PRESSURE_PA / 1e6, 6.894757, rel_tol=1e-6)
    assert BASE_ISP_S == 900.0

    assert BIMODAL_POWER.electric_power_kwe_per_engine == 25.0
    assert BIMODAL_POWER.reference_stage_power_kwe == 50.0
    assert BIMODAL_POWER.idle_reactor_thermal_power_kwt == 125.0
    assert math.isclose(BIMODAL_POWER.conversion_efficiency, 0.20)

    print("LANTR / bimodal NTR reference validation: PASS")
    print(f"  MR=0: {p0.thrust_n/1000.0:.3f} kN, {p0.delivered_isp_s:.0f} s")
    print(f"  MR=3: {p3.thrust_n/1000.0:.3f} kN, {p3.delivered_isp_s:.0f} s")
    print(f"  MR=5: {p5.thrust_n/1000.0:.3f} kN, {p5.delivered_isp_s:.0f} s")
    print(
        f"  power mode: {BIMODAL_POWER.electric_power_kwe_per_engine:.0f} kWe/engine "
        f"from {BIMODAL_POWER.idle_reactor_thermal_power_kwt:.0f} kWt reference idle reactor"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
