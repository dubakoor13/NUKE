"""Validate per-tie OpenMC thermal reconstruction helpers."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from .solve_tie_instances import _scale_to_total


def main() -> int:
    values = np.array(
        [
            [1.0, 2.0, 1.0],
            [2.0, 1.0, 3.0],
        ]
    )
    scaled = _scale_to_total(values, 100.0)
    assert np.isclose(np.sum(scaled), 100.0)
    assert np.all(scaled >= 0.0)

    zero = _scale_to_total(values, 0.0)
    assert np.all(zero == 0.0)

    try:
        _scale_to_total(np.zeros_like(values), 1.0)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "positive target with zero source should fail"
        )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fields_path = root / "fields.npz"
        instances_path = root / "instances.npz"
        output = root / "out"

        dimension = np.array([1, 1, 4], dtype=int)
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 100.0])

        tie_density = np.zeros((1, 1, 4), dtype=float)
        tie_density[0, 0, :] = [2.0, 4.0, 6.0, 4.0]
        zero_density = np.zeros_like(tie_density)

        np.savez_compressed(
            fields_path,
            tie_power_density_w_cm3=tie_density,
            tie_supply_hydrogen_power_density_w_cm3=(
                zero_density
            ),
            tie_return_hydrogen_power_density_w_cm3=(
                zero_density
            ),
            lower_left_cm=lower,
            upper_right_cm=upper,
            dimension=dimension,
        )

        tie_axial = np.array(
            [
                [1.0, 2.0, 3.0, 2.0],
                [2.0, 2.0, 1.0, 1.0],
            ],
            dtype=float,
        )
        np.savez_compressed(
            instances_path,
            tie_solid_axial_heating_w=tie_axial,
            tie_solid_axial_heating_std_w=np.zeros_like(
                tie_axial
            ),
            tie_supply_hydrogen_axial_heating_w=np.zeros_like(
                tie_axial
            ),
            tie_supply_hydrogen_axial_heating_std_w=np.zeros_like(
                tie_axial
            ),
            tie_return_hydrogen_axial_heating_w=np.zeros_like(
                tie_axial
            ),
            tie_return_hydrogen_axial_heating_std_w=np.zeros_like(
                tie_axial
            ),
            instance_axial_z_edges_m=np.linspace(
                0.0,
                1.0,
                5,
            ),
        )

        subprocess.run(
            [
                sys.executable,
                "-m",
                "ntr.nerva.solve_tie_instances",
                str(fields_path),
                str(instances_path),
                "--tie-mass-flow-kg-s",
                "0.04",
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
                str(output),
            ],
            check=True,
        )

        summary = json.loads(
            (output / "tie_instance_summary.json").read_text(
                encoding="utf-8"
            )
        )
        assert summary["flow_balance_enabled"] is True
        assert summary["flow_balance_iterations"] >= 1
        assert np.isclose(
            summary["reconstructed_total_tie_mass_flow_kg_s"],
            0.04,
            rtol=1.0e-12,
        )
        assert summary["minimum_tie_mass_flow_kg_s"] > 0.0
        assert np.isfinite(
            summary["final_pressure_drop_spread_fraction"]
        )
        assert (
            summary["final_pressure_drop_spread_fraction"]
            >= 0.0
        )

    print("NERVA per-tie reconstruction validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
