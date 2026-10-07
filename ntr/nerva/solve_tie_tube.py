"""Solve the NERVA-derived counterflow tie-tube thermal path."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import NervaConfig
from .hydrogen_properties import make_hydrogen_property_model
from .layout import mixed_core_counts
from .tie_tube_thermal import solve_tie_tube_counterflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve one representative NERVA tie-tube supply/return path."
    )
    parser.add_argument(
        "power_fields",
        type=Path,
        help="nerva_mesh_fields.npz from postprocess_statepoint.",
    )
    parser.add_argument("--rings", type=int, required=True)
    parser.add_argument(
        "--tie-mass-flow-kg-s",
        type=float,
        required=True,
        help="Total hydrogen mass flow allocated to all tie tubes.",
    )
    parser.add_argument("--inlet-temperature-k", type=float, required=True)
    parser.add_argument("--inlet-pressure-mpa", type=float, required=True)
    parser.add_argument(
        "--hydrogen-model",
        choices=("constant", "coolprop"),
        default="constant",
        help="Hydrogen property backend; CoolProp is optional.",
    )
    parser.add_argument(
        "--supply-heat-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of tie-solid heat transferred to the central supply path. "
            "Default uses wetted-perimeter weighting."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_tie_thermal"),
    )
    return parser.parse_args()


def _write_path(path: Path, solution) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "z_m",
                "bulk_temperature_K",
                "wall_temperature_K",
                "pressure_Pa",
                "reynolds",
                "htc_W_m2_K",
                "heat_flux_W_m2",
                "path_power_W",
            )
        )
        for row in zip(
            solution.z_center_m,
            solution.bulk_temperature_k,
            solution.wall_temperature_k,
            solution.pressure_pa,
            solution.reynolds,
            solution.heat_transfer_coefficient_w_m2_k,
            solution.heat_flux_w_m2,
            solution.path_power_w,
        ):
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.tie_mass_flow_kg_s <= 0.0:
        raise ValueError("--tie-mass-flow-kg-s must be positive")

    data = np.load(args.power_fields)
    required = {
        "tie_power_density_w_cm3",
        "lower_left_cm",
        "upper_right_cm",
        "dimension",
    }
    missing = required.difference(data.files)
    if missing:
        raise KeyError(
            "power field file is missing required arrays: "
            + ", ".join(sorted(missing))
        )

    tie_power_density = np.asarray(data["tie_power_density_w_cm3"], dtype=float)
    lower_left = np.asarray(data["lower_left_cm"], dtype=float)
    upper_right = np.asarray(data["upper_right_cm"], dtype=float)
    dimension = np.asarray(data["dimension"], dtype=int)
    shape = tuple(int(x) for x in dimension)
    if tie_power_density.shape != shape:
        raise ValueError(
            f"tie power field shape {tie_power_density.shape} "
            f"does not match stored dimension {shape}"
        )

    spacing_cm = (upper_right - lower_left) / dimension
    voxel_volume_cm3 = float(np.prod(spacing_cm))
    axial_tie_power_w = (
        np.sum(tie_power_density, axis=(0, 1))
        * voxel_volume_cm3
    )

    z_edges_m = np.linspace(
        lower_left[2] / 100.0,
        upper_right[2] / 100.0,
        int(dimension[2]) + 1,
    )

    _, tie_tubes = mixed_core_counts(args.rings)
    mass_flow_per_tie = args.tie_mass_flow_kg_s / tie_tubes

    config = NervaConfig(core_rings=args.rings)
    property_model = make_hydrogen_property_model(args.hydrogen_model)
    solution = solve_tie_tube_counterflow(
        axial_total_tie_power_w=axial_tie_power_w,
        z_edges_m=z_edges_m,
        tie_tube_count=tie_tubes,
        mass_flow_per_tie_kg_s=mass_flow_per_tie,
        inlet_temperature_k=args.inlet_temperature_k,
        inlet_pressure_pa=args.inlet_pressure_mpa * 1.0e6,
        config=config,
        property_model=property_model,
        supply_heat_fraction=args.supply_heat_fraction,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_path(args.output / "tie_supply_profile.csv", solution.supply)
    _write_path(args.output / "tie_return_profile.csv", solution.return_path)

    summary = {
        "tie_tubes": tie_tubes,
        "tie_mass_flow_kg_s": args.tie_mass_flow_kg_s,
        "mass_flow_per_tie_kg_s": mass_flow_per_tie,
        "total_tie_power_W": float(np.sum(axial_tie_power_w)),
        "representative_tie_power_W": solution.tie_power_w,
        "supply_heat_fraction": solution.supply_heat_fraction,
        "inlet_temperature_K": args.inlet_temperature_k,
        "outlet_temperature_K": solution.outlet_temperature_k,
        "inlet_pressure_Pa": args.inlet_pressure_mpa * 1.0e6,
        "outlet_pressure_Pa": solution.outlet_pressure_pa,
        "max_supply_wall_temperature_K": float(
            np.max(solution.supply.wall_temperature_k)
        ),
        "max_return_wall_temperature_K": float(
            np.max(solution.return_path.wall_temperature_k)
        ),
        "model": args.hydrogen_model,
    }
    (args.output / "tie_thermal_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA representative tie-tube counterflow solve complete:")
    print(f"  tie tubes: {tie_tubes}")
    print(f"  tie-solid power: {summary['total_tie_power_W'] / 1.0e6:.6g} MW")
    print(f"  supply heat fraction: {solution.supply_heat_fraction:.6f}")
    print(f"  outlet temperature: {solution.outlet_temperature_k:.3f} K")
    print(f"  outlet pressure: {solution.outlet_pressure_pa / 1.0e6:.6f} MPa")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
