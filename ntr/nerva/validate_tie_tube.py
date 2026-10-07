"""Deterministic validation of the NERVA tie-tube counterflow solver."""

from __future__ import annotations

import numpy as np

from .config import NervaConfig
from .thermal import HydrogenProperties
from .tie_tube_thermal import solve_tie_tube_counterflow


def main() -> int:
    config = NervaConfig()
    properties = HydrogenProperties()

    axial_power = np.full(12, 20000.0 / 12.0)
    z_edges = np.linspace(0.0, 1.2, 13)
    tie_tubes = 10
    mass_flow_per_tie = 0.002

    solution = solve_tie_tube_counterflow(
        axial_total_tie_power_w=axial_power,
        z_edges_m=z_edges,
        tie_tube_count=tie_tubes,
        mass_flow_per_tie_kg_s=mass_flow_per_tie,
        inlet_temperature_k=500.0,
        inlet_pressure_pa=8.0e6,
        config=config,
        properties=properties,
    )

    representative_power = float(np.sum(axial_power)) / tie_tubes
    recovered_power = (
        mass_flow_per_tie
        * properties.cp_j_kg_k
        * (solution.outlet_temperature_k - 500.0)
    )

    assert np.isclose(solution.tie_power_w, representative_power, rtol=1.0e-12)
    assert np.isclose(
        solution.supply.absorbed_power_w
        + solution.return_path.absorbed_power_w,
        representative_power,
        rtol=1.0e-12,
    )
    assert np.isclose(recovered_power, representative_power, rtol=1.0e-12)
    assert solution.outlet_temperature_k > 500.0
    assert solution.outlet_pressure_pa < 8.0e6
    assert np.all(np.diff(solution.supply.pressure_pa) < 0.0)
    assert np.all(np.diff(solution.return_path.pressure_pa) < 0.0)
    assert np.all(solution.supply.reynolds > 0.0)
    assert np.all(solution.return_path.reynolds > 0.0)

    direct_supply_power = np.full(12, 2000.0 / 12.0)
    direct_return_power = np.full(12, 1000.0 / 12.0)
    direct_solution = solve_tie_tube_counterflow(
        axial_total_tie_power_w=axial_power,
        axial_direct_supply_hydrogen_power_w=direct_supply_power,
        axial_direct_return_hydrogen_power_w=direct_return_power,
        z_edges_m=z_edges,
        tie_tube_count=tie_tubes,
        mass_flow_per_tie_kg_s=mass_flow_per_tie,
        inlet_temperature_k=500.0,
        inlet_pressure_pa=8.0e6,
        config=config,
        properties=properties,
    )

    assert direct_solution.outlet_temperature_k > solution.outlet_temperature_k
    assert np.allclose(
        direct_solution.supply.heat_flux_w_m2,
        solution.supply.heat_flux_w_m2,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.allclose(
        direct_solution.return_path.heat_flux_w_m2,
        solution.return_path.heat_flux_w_m2,
        rtol=0.0,
        atol=1.0e-12,
    )
    assert np.isclose(
        direct_solution.direct_nuclear_coolant_power_w,
        (2000.0 + 1000.0) / tie_tubes,
        rtol=1.0e-12,
    )
    direct_recovered_power = (
        mass_flow_per_tie
        * properties.cp_j_kg_k
        * (direct_solution.outlet_temperature_k - 500.0)
    )
    assert np.isclose(
        direct_recovered_power,
        (20000.0 + 2000.0 + 1000.0) / tie_tubes,
        rtol=1.0e-12,
    )

    print("NERVA tie-tube counterflow validation: PASS")
    print(f"  representative tie power: {representative_power:.6f} W")
    print(f"  supply heat fraction: {solution.supply_heat_fraction:.6f}")
    print(f"  outlet temperature: {solution.outlet_temperature_k:.6f} K")
    print(f"  outlet pressure: {solution.outlet_pressure_pa / 1.0e6:.6f} MPa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
