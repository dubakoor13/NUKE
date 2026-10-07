"""Regression tests for NERVA engine-level performance estimates."""

from __future__ import annotations

from .engine_performance import (
    combine_outlet_streams,
    ideal_nozzle_performance,
)


def main() -> int:
    mixed = combine_outlet_streams(
        fuel_mass_flow_kg_s=2.0,
        fuel_temperature_k=524.475524,
        fuel_pressure_pa=7.95e6,
        tie_mass_flow_kg_s=0.4,
        tie_temperature_k=508.741259,
        tie_pressure_pa=7.90e6,
    )

    performance = ideal_nozzle_performance(
        mass_flow_kg_s=mixed.mass_flow_kg_s,
        chamber_temperature_k=mixed.temperature_k,
        chamber_pressure_pa=mixed.pressure_pa,
        exit_pressure_pa=1000.0,
        ambient_pressure_pa=0.0,
        gamma=1.35,
        nozzle_efficiency=0.95,
    )

    assert 520.0 < mixed.temperature_k < 524.0
    assert performance.thrust_n > 0.0
    assert performance.isp_s > 0.0
    assert performance.exit_velocity_m_s > 0.0
    assert performance.expansion_ratio > 1.0
    assert performance.exit_area_m2 > performance.throat_area_m2

    print("NERVA engine-performance validation: PASS")
    print(f"  mixed temperature: {mixed.temperature_k:.6f} K")
    print(f"  thrust: {performance.thrust_n:.6f} N")
    print(f"  Isp: {performance.isp_s:.6f} s")
    print(
        f"  effective exhaust velocity: "
        f"{performance.effective_exhaust_velocity_m_s:.6f} m/s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
