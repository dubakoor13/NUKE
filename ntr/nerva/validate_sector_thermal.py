"""End-to-end synthetic validation of fuel-sector thermal coupling."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fields_path = root / "fields.npz"
        instances_path = root / "instances.npz"
        output = root / "out"

        dimension = np.array([1, 1, 4], dtype=int)
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 100.0])

        fuel_density = np.zeros((1, 1, 4), dtype=float)
        fuel_density[0, 0, :] = [10.0, 20.0, 30.0, 40.0]
        direct_h2_density = np.zeros_like(fuel_density)

        np.savez_compressed(
            fields_path,
            fuel_power_density_w_cm3=fuel_density,
            fuel_hydrogen_power_density_w_cm3=direct_h2_density,
            lower_left_cm=lower,
            upper_right_cm=upper,
            dimension=dimension,
        )

        n_elements = 2
        n_channels = 19
        n_axial = 4

        sector_integrated = np.zeros(
            (n_elements, n_channels),
            dtype=float,
        )
        sector_integrated[0, :] = np.linspace(
            1.0,
            2.0,
            n_channels,
        )
        sector_integrated[1, :] = np.linspace(
            2.0,
            4.0,
            n_channels,
        )

        element_fraction = np.sum(
            sector_integrated,
            axis=1,
        )
        element_fraction = (
            element_fraction / np.sum(element_fraction)
        )

        sector_axial = np.zeros(
            (n_elements, n_channels, n_axial),
            dtype=float,
        )
        base_shape = np.array([1.0, 2.0, 3.0, 2.0])
        for element in range(n_elements):
            for channel in range(n_channels):
                sector_axial[element, channel, :] = (
                    sector_integrated[element, channel]
                    * base_shape
                )

        element_axial = np.sum(sector_axial, axis=1)

        np.savez_compressed(
            instances_path,
            fuel_element_power_fraction=element_fraction,
            fuel_element_power_w=np.sum(
                sector_axial,
                axis=(1, 2),
            ),
            fuel_element_heating_std_w=np.zeros(n_elements),
            fuel_channel_sector_heating_w=sector_integrated,
            fuel_channel_sector_heating_std_w=np.zeros_like(
                sector_integrated
            ),
            fuel_channel_sector_axial_heating_w=sector_axial,
            fuel_channel_sector_axial_heating_std_w=np.zeros_like(
                sector_axial
            ),
            fuel_element_axial_heating_w=element_axial,
            fuel_element_axial_heating_std_w=np.zeros_like(
                element_axial
            ),
            channel_direct_nuclear_heating_w=np.zeros(
                (n_elements, n_channels),
            ),
            channel_direct_nuclear_heating_std_w=np.zeros(
                (n_elements, n_channels),
            ),
            channel_direct_nuclear_heating_axial_w=np.zeros(
                (n_elements, n_channels, n_axial),
            ),
            channel_direct_nuclear_heating_axial_std_w=np.zeros(
                (n_elements, n_channels, n_axial),
            ),
            instance_axial_z_edges_m=np.linspace(
                0.0,
                1.0,
                n_axial + 1,
            ),
        )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "ntr.nerva.solve_element_channels",
                str(fields_path),
                str(instances_path),
                "--fuel-mass-flow-kg-s",
                "0.20",
                "--inlet-temperature-k",
                "500",
                "--inlet-pressure-mpa",
                "8",
                "--hydrogen-model",
                "constant",
                "--output",
                str(output),
            ],
            check=True,
        )

        summary = json.loads(
            (output / "element_channel_summary.json").read_text(
                encoding="utf-8"
            )
        )

        voxel_volume_cm3 = 25.0
        expected_power_w = float(
            np.sum(fuel_density) * voxel_volume_cm3
        )

        assert summary["direct_fuel_sector_openmc_used"] is True
        assert summary["direct_element_axial_openmc_used"] is True
        assert summary["fuel_sector_axial_zero_fallback_count"] == 0
        assert summary["fuel_element_instances"] == n_elements
        assert summary["total_channels"] == n_elements * n_channels
        assert np.isclose(
            summary["reconstructed_wall_power_W"],
            expected_power_w,
            rtol=1.0e-12,
        )

        balanced_output = root / "balanced"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ntr.nerva.solve_element_channels",
                str(fields_path),
                str(instances_path),
                "--fuel-mass-flow-kg-s",
                "0.20",
                "--inlet-temperature-k",
                "500",
                "--inlet-pressure-mpa",
                "8",
                "--hydrogen-model",
                "constant",
                "--balance-flow",
                "--flow-balance-max-iterations",
                "3",
                "--flow-balance-tolerance",
                "1e-4",
                "--flow-balance-relaxation",
                "0.5",
                "--output",
                str(balanced_output),
            ],
            check=True,
        )

        balanced = json.loads(
            (
                balanced_output
                / "element_channel_summary.json"
            ).read_text(encoding="utf-8")
        )
        assert balanced["flow_balance_enabled"] is True
        assert balanced["flow_balance_iterations"] >= 1
        assert np.isclose(
            balanced[
                "reconstructed_total_fuel_mass_flow_kg_s"
            ],
            0.20,
            rtol=1.0e-12,
        )
        assert balanced[
            "minimum_channel_mass_flow_kg_s"
        ] > 0.0
        assert np.isfinite(
            balanced["final_pressure_drop_spread_fraction"]
        )
        assert (
            balanced["final_pressure_drop_spread_fraction"]
            >= 0.0
        )

    print("NERVA fuel-sector thermal coupling validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
