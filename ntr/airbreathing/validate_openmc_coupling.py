"""Validate OpenMC-power to airbreathing-cycle coupling."""

from __future__ import annotations

from .nuclear_ramjet import evaluate_nuclear_ramjet_from_thermal_power


def main() -> int:
    low = evaluate_nuclear_ramjet_from_thermal_power(
        mach=2.0,
        ambient_temperature_k=220.0,
        ambient_pressure_pa=25_000.0,
        air_mass_flow_kg_s=10.0,
        deposited_thermal_power_w=5.0e6,
        heat_transfer_efficiency=0.80,
    )
    high = evaluate_nuclear_ramjet_from_thermal_power(
        mach=2.0,
        ambient_temperature_k=220.0,
        ambient_pressure_pa=25_000.0,
        air_mass_flow_kg_s=10.0,
        deposited_thermal_power_w=10.0e6,
        heat_transfer_efficiency=0.80,
    )

    assert high.reactor_outlet_total_temperature_k > (
        low.reactor_outlet_total_temperature_k
    )
    assert high.nozzle_exit_velocity_m_s > low.nozzle_exit_velocity_m_s
    assert high.net_thrust_n > low.net_thrust_n

    print("OpenMC-power / airbreathing coupling validation: PASS")
    print(
        f"  5 MW -> Tt,out={low.reactor_outlet_total_temperature_k:.2f} K"
    )
    print(
        f"  10 MW -> Tt,out={high.reactor_outlet_total_temperature_k:.2f} K"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
