"""Deterministic validation for the first 1-D hydrogen channel model."""

from __future__ import annotations

import numpy as np

from .thermal import HydrogenProperties, solve_fuel_channel


def main() -> int:
    axial_power = np.full(12, 1.0e6 / 12.0)
    z_edges = np.linspace(0.0, 1.2, 13)
    channels = 100
    mass_flow_per_channel = 1.0e-3
    properties = HydrogenProperties()

    solution = solve_fuel_channel(
        axial_total_fuel_power_w=axial_power,
        z_edges_m=z_edges,
        fuel_channel_count=channels,
        mass_flow_per_channel_kg_s=mass_flow_per_channel,
        inlet_temperature_k=500.0,
        inlet_pressure_pa=8.0e6,
        channel_diameter_m=0.002565,
        properties=properties,
    )

    channel_power = float(np.sum(axial_power)) / channels
    recovered = (
        mass_flow_per_channel
        * properties.cp_j_kg_k
        * (solution.outlet_temperature_k - 500.0)
    )

    assert np.isclose(solution.absorbed_power_w, channel_power, rtol=1.0e-12)
    assert np.isclose(recovered, channel_power, rtol=1.0e-12)
    assert solution.outlet_temperature_k > 500.0
    assert solution.outlet_pressure_pa < 8.0e6
    bin_inlet_temperature = np.concatenate(
        ([500.0], solution.bulk_temperature_k[:-1])
    )
    bin_mean_temperature = 0.5 * (
        bin_inlet_temperature + solution.bulk_temperature_k
    )
    assert np.all(solution.wall_temperature_k > bin_mean_temperature)
    assert np.all(solution.reynolds > 0.0)

    print("NERVA 1-D fuel-channel thermal validation: PASS")
    print(f"  representative channel power: {channel_power:.6f} W")
    print(f"  outlet temperature: {solution.outlet_temperature_k:.6f} K")
    print(f"  outlet pressure: {solution.outlet_pressure_pa / 1.0e6:.6f} MPa")
    print(f"  max wall temperature: {np.max(solution.wall_temperature_k):.6f} K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
