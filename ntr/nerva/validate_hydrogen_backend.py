"""Regression tests for state-dependent hydrogen property plumbing."""

from __future__ import annotations

import numpy as np

from .config import NervaConfig
from .hydrogen_properties import HydrogenState
from .thermal import HydrogenProperties, solve_fuel_channel
from .tie_tube_thermal import solve_tie_tube_counterflow


class MockHydrogenModel:
    """Deterministic state-dependent backend used only by CI."""

    def __init__(self) -> None:
        self.calls = 0

    def state(self, temperature_k: float, pressure_pa: float) -> HydrogenState:
        self.calls += 1
        gas_constant = 4124.0
        density = pressure_pa / (gas_constant * temperature_k)
        cp = 13000.0 + 2.0 * temperature_k
        viscosity = 8.0e-6 + 6.0e-9 * temperature_k
        conductivity = 0.12 + 8.0e-5 * temperature_k
        prandtl = cp * viscosity / conductivity
        return HydrogenState(
            density_kg_m3=density,
            cp_j_kg_k=cp,
            dynamic_viscosity_pa_s=viscosity,
            thermal_conductivity_w_m_k=conductivity,
            prandtl=prandtl,
        )


def main() -> int:
    z_edges = np.linspace(0.0, 1.2, 13)
    fuel_power = np.linspace(0.5, 1.5, 12) * 1.0e6 / 12.0

    constant = solve_fuel_channel(
        axial_total_fuel_power_w=fuel_power,
        z_edges_m=z_edges,
        fuel_channel_count=100,
        mass_flow_per_channel_kg_s=1.0e-3,
        inlet_temperature_k=500.0,
        inlet_pressure_pa=8.0e6,
        channel_diameter_m=0.002565,
        properties=HydrogenProperties(),
    )

    mock = MockHydrogenModel()
    variable = solve_fuel_channel(
        axial_total_fuel_power_w=fuel_power,
        z_edges_m=z_edges,
        fuel_channel_count=100,
        mass_flow_per_channel_kg_s=1.0e-3,
        inlet_temperature_k=500.0,
        inlet_pressure_pa=8.0e6,
        channel_diameter_m=0.002565,
        properties=HydrogenProperties(),
        property_model=mock,
    )

    assert mock.calls > 12
    assert variable.outlet_temperature_k > 500.0
    assert variable.outlet_pressure_pa < 8.0e6
    assert not np.isclose(
        variable.outlet_temperature_k,
        constant.outlet_temperature_k,
    )

    tie_mock = MockHydrogenModel()
    tie = solve_tie_tube_counterflow(
        axial_total_tie_power_w=np.full(12, 20000.0 / 12.0),
        z_edges_m=z_edges,
        tie_tube_count=10,
        mass_flow_per_tie_kg_s=0.002,
        inlet_temperature_k=500.0,
        inlet_pressure_pa=8.0e6,
        config=NervaConfig(),
        properties=HydrogenProperties(),
        property_model=tie_mock,
    )

    assert tie_mock.calls > 24
    assert tie.outlet_temperature_k > 500.0
    assert tie.outlet_pressure_pa < 8.0e6

    print("NERVA state-dependent hydrogen property validation: PASS")
    print(f"  fuel backend calls: {mock.calls}")
    print(f"  tie backend calls: {tie_mock.calls}")
    print(
        f"  constant/variable fuel outlet K: "
        f"{constant.outlet_temperature_k:.6f} / "
        f"{variable.outlet_temperature_k:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
