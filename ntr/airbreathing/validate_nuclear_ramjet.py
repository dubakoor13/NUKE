"""Regression checks for the generic nuclear-airbreathing cycle."""

from __future__ import annotations

from .nuclear_ramjet import evaluate_nuclear_ramjet


def main() -> int:
    baseline = evaluate_nuclear_ramjet(
        mach=2.0,
        ambient_temperature_k=220.0,
        ambient_pressure_pa=25_000.0,
        air_mass_flow_kg_s=10.0,
        reactor_outlet_total_temperature_k=1400.0,
    )

    hotter = evaluate_nuclear_ramjet(
        mach=2.0,
        ambient_temperature_k=220.0,
        ambient_pressure_pa=25_000.0,
        air_mass_flow_kg_s=10.0,
        reactor_outlet_total_temperature_k=1600.0,
    )

    more_flow = evaluate_nuclear_ramjet(
        mach=2.0,
        ambient_temperature_k=220.0,
        ambient_pressure_pa=25_000.0,
        air_mass_flow_kg_s=20.0,
        reactor_outlet_total_temperature_k=1400.0,
    )

    assert baseline.reactor_thermal_power_w > 0.0
    assert baseline.nozzle_exit_velocity_m_s > 0.0
    assert baseline.gross_thrust_n > 0.0
    assert baseline.ram_drag_n > 0.0
    assert baseline.net_thrust_n > 0.0

    # More reactor heat raises exit velocity and net specific thrust.
    assert hotter.reactor_thermal_power_w > baseline.reactor_thermal_power_w
    assert hotter.nozzle_exit_velocity_m_s > baseline.nozzle_exit_velocity_m_s
    assert hotter.specific_thrust_n_s_kg > baseline.specific_thrust_n_s_kg

    # At the same thermodynamic operating point, doubling supplied air flow
    # doubles reactor heat duty and thrust, but not specific thrust.
    assert abs(
        more_flow.reactor_thermal_power_w
        / baseline.reactor_thermal_power_w
        - 2.0
    ) < 1.0e-12
    assert abs(
        more_flow.net_thrust_n / baseline.net_thrust_n - 2.0
    ) < 1.0e-12
    assert abs(
        more_flow.specific_thrust_n_s_kg
        - baseline.specific_thrust_n_s_kg
    ) < 1.0e-12

    print("Generic nuclear-airbreathing validation: PASS")
    print(
        f"  baseline net thrust: "
        f"{baseline.net_thrust_n / 1000.0:.3f} kN"
    )
    print(
        f"  baseline reactor heat: "
        f"{baseline.reactor_thermal_power_w / 1.0e6:.3f} MW"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
