"""Deterministic validation for the NERVA fuel-solid conduction estimate."""

from __future__ import annotations

import numpy as np

from .config import NervaConfig
from .fuel_conduction import (
    FuelSolidProperties,
    estimate_fuel_solid_temperatures,
)
from .thermal import HydrogenProperties, solve_fuel_channel


def main() -> int:
    config = NervaConfig()
    axial_power = np.full(12, 1.0e6 / 12.0)
    z_edges = np.linspace(0.0, 1.2, 13)

    channel = solve_fuel_channel(
        axial_total_fuel_power_w=axial_power,
        z_edges_m=z_edges,
        fuel_channel_count=100,
        mass_flow_per_channel_kg_s=1.0e-3,
        inlet_temperature_k=500.0,
        inlet_pressure_pa=8.0e6,
        channel_diameter_m=config.coolant_bore_diameter_cm / 100.0,
        properties=HydrogenProperties(),
    )

    estimate = estimate_fuel_solid_temperatures(
        channel,
        config=config,
        properties=FuelSolidProperties(),
    )

    assert estimate.effective_half_ligament_m > 0.0
    assert estimate.coating_thickness_m > 0.0
    assert np.all(
        estimate.fuel_surface_temperature_k
        >= estimate.coolant_wall_temperature_k
    )
    assert np.all(
        estimate.peak_fuel_temperature_k
        >= estimate.fuel_surface_temperature_k
    )

    zero_flux_channel = channel.__class__(
        z_center_m=channel.z_center_m,
        bulk_temperature_k=channel.bulk_temperature_k,
        wall_temperature_k=channel.wall_temperature_k,
        pressure_pa=channel.pressure_pa,
        reynolds=channel.reynolds,
        heat_transfer_coefficient_w_m2_k=channel.heat_transfer_coefficient_w_m2_k,
        heat_flux_w_m2=np.zeros_like(channel.heat_flux_w_m2),
        channel_power_w=np.zeros_like(channel.channel_power_w),
        wall_heat_power_w=np.zeros_like(channel.wall_heat_power_w),
        direct_coolant_power_w=np.zeros_like(channel.direct_coolant_power_w),
        mass_flow_kg_s=channel.mass_flow_kg_s,
        channel_diameter_m=channel.channel_diameter_m,
    )
    zero = estimate_fuel_solid_temperatures(zero_flux_channel, config=config)
    assert np.allclose(
        zero.peak_fuel_temperature_k,
        zero.coolant_wall_temperature_k,
    )

    print("NERVA fuel-solid conduction validation: PASS")
    print(
        f"  effective half-ligament: "
        f"{estimate.effective_half_ligament_m * 1.0e3:.6f} mm"
    )
    print(
        f"  peak fuel temperature: "
        f"{np.max(estimate.peak_fuel_temperature_k):.6f} K"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
