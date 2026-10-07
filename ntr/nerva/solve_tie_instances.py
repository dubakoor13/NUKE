"""Solve every repeated NERVA tie tube using instance-axial OpenMC heating."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import NervaConfig
from .hydrogen_properties import make_hydrogen_property_model
from .parallel_flow import (
    pressure_spread_fraction,
    update_parallel_mass_flow,
)
from .tie_tube_thermal import solve_tie_tube_counterflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve all repeated NERVA tie tubes using direct OpenMC "
            "instance-axial solid and H2 heating tallies."
        )
    )
    parser.add_argument(
        "power_fields",
        type=Path,
        help="nerva_mesh_fields.npz from postprocess_statepoint.",
    )
    parser.add_argument(
        "instance_fields",
        type=Path,
        help="instance_power_fractions.npz from postprocess_instance_tallies.",
    )
    parser.add_argument(
        "--tie-mass-flow-kg-s",
        type=float,
        required=True,
        help="Total hydrogen mass flow through all tie tubes.",
    )
    parser.add_argument("--inlet-temperature-k", type=float, required=True)
    parser.add_argument("--inlet-pressure-mpa", type=float, required=True)
    parser.add_argument(
        "--hydrogen-model",
        choices=("constant", "coolprop"),
        default="constant",
    )
    parser.add_argument(
        "--supply-heat-fraction",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--balance-flow",
        action="store_true",
        help=(
            "Iteratively redistribute the fixed total tie-tube flow "
            "to approach a common pressure drop across parallel ties."
        ),
    )
    parser.add_argument(
        "--flow-balance-max-iterations",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--flow-balance-tolerance",
        type=float,
        default=1.0e-3,
    )
    parser.add_argument(
        "--flow-balance-relaxation",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_tie_instances"),
    )
    return parser.parse_args()


def _scale_to_total(
    values: np.ndarray,
    total_w: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(values < 0.0):
        raise ValueError("heating arrays must be non-negative")
    current = float(np.sum(values))
    if total_w < 0.0:
        raise ValueError("target total must be non-negative")
    if total_w == 0.0:
        return np.zeros_like(values)
    if current <= 0.0:
        raise ValueError(
            "cannot scale an empty/zero instance field to positive total"
        )
    return values * (total_w / current)


def _global_axial(
    density: np.ndarray,
    voxel_volume_cm3: float,
) -> np.ndarray:
    return (
        np.sum(np.asarray(density, dtype=float), axis=(0, 1))
        * voxel_volume_cm3
    )


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
                "total_absorbed_power_W",
                "wall_heat_power_W",
                "direct_nuclear_hydrogen_power_W",
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
            solution.wall_heat_power_w,
            solution.direct_coolant_power_w,
        ):
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.tie_mass_flow_kg_s <= 0.0:
        raise ValueError("--tie-mass-flow-kg-s must be positive")
    if args.inlet_temperature_k <= 0.0:
        raise ValueError("--inlet-temperature-k must be positive")
    if args.inlet_pressure_mpa <= 0.0:
        raise ValueError("--inlet-pressure-mpa must be positive")
    if args.flow_balance_max_iterations < 1:
        raise ValueError(
            "--flow-balance-max-iterations must be >= 1"
        )
    if args.flow_balance_tolerance <= 0.0:
        raise ValueError(
            "--flow-balance-tolerance must be positive"
        )
    if not (0.0 < args.flow_balance_relaxation <= 1.0):
        raise ValueError(
            "--flow-balance-relaxation must lie in (0, 1]"
        )

    fields = np.load(args.power_fields)
    instances = np.load(args.instance_fields)

    required_fields = {
        "tie_power_density_w_cm3",
        "lower_left_cm",
        "upper_right_cm",
        "dimension",
    }
    missing = required_fields - set(fields.files)
    if missing:
        raise KeyError(
            "power field package missing: "
            + ", ".join(sorted(missing))
        )

    if "tie_solid_axial_heating_w" not in instances.files:
        raise KeyError(
            "instance package missing tie_solid_axial_heating_w"
        )

    tie_raw = np.asarray(
        instances["tie_solid_axial_heating_w"],
        dtype=float,
    )
    if tie_raw.ndim != 2 or tie_raw.shape[0] < 1:
        raise ValueError(
            "tie_solid_axial_heating_w must be [tie, axial]"
        )

    n_ties, n_axial = tie_raw.shape

    if "instance_axial_z_edges_m" not in instances.files:
        raise KeyError("instance package missing instance_axial_z_edges_m")
    z_edges_m = np.asarray(
        instances["instance_axial_z_edges_m"],
        dtype=float,
    )
    if z_edges_m.size != n_axial + 1:
        raise ValueError(
            "instance axial z-grid does not match tie axial data"
        )

    supply_raw = np.asarray(
        instances["tie_supply_hydrogen_axial_heating_w"]
        if "tie_supply_hydrogen_axial_heating_w" in instances.files
        else np.zeros_like(tie_raw),
        dtype=float,
    )
    return_raw = np.asarray(
        instances["tie_return_hydrogen_axial_heating_w"]
        if "tie_return_hydrogen_axial_heating_w" in instances.files
        else np.zeros_like(tie_raw),
        dtype=float,
    )
    if supply_raw.shape != tie_raw.shape:
        raise ValueError("tie supply H2 axial shape mismatch")
    if return_raw.shape != tie_raw.shape:
        raise ValueError("tie return H2 axial shape mismatch")

    dimension = np.asarray(fields["dimension"], dtype=int)
    lower = np.asarray(fields["lower_left_cm"], dtype=float)
    upper = np.asarray(fields["upper_right_cm"], dtype=float)
    spacing_cm = (upper - lower) / dimension
    voxel_volume_cm3 = float(np.prod(spacing_cm))

    global_tie_axial = _global_axial(
        fields["tie_power_density_w_cm3"],
        voxel_volume_cm3,
    )
    global_tie_power_w = float(np.sum(global_tie_axial))

    if "tie_supply_hydrogen_power_density_w_cm3" in fields.files:
        global_supply_h2_axial = _global_axial(
            fields["tie_supply_hydrogen_power_density_w_cm3"],
            voxel_volume_cm3,
        )
    else:
        global_supply_h2_axial = np.zeros(int(dimension[2]))

    if "tie_return_hydrogen_power_density_w_cm3" in fields.files:
        global_return_h2_axial = _global_axial(
            fields["tie_return_hydrogen_power_density_w_cm3"],
            voxel_volume_cm3,
        )
    else:
        global_return_h2_axial = np.zeros(int(dimension[2]))

    global_supply_h2_power_w = float(
        np.sum(global_supply_h2_axial)
    )
    global_return_h2_power_w = float(
        np.sum(global_return_h2_axial)
    )

    tie_solid_axial = _scale_to_total(
        tie_raw,
        global_tie_power_w,
    )
    tie_supply_h2_axial = _scale_to_total(
        supply_raw,
        global_supply_h2_power_w,
    ) if global_supply_h2_power_w > 0.0 else np.zeros_like(tie_raw)
    tie_return_h2_axial = _scale_to_total(
        return_raw,
        global_return_h2_power_w,
    ) if global_return_h2_power_w > 0.0 else np.zeros_like(tie_raw)

    mass_flow_per_tie = args.tie_mass_flow_kg_s / n_ties
    config = NervaConfig()
    property_model = make_hydrogen_property_model(
        args.hydrogen_model
    )

    tie_mass_flow = np.full(
        n_ties,
        mass_flow_per_tie,
        dtype=float,
    )
    flow_balance_converged: bool | None = None
    flow_balance_iterations = 0
    flow_balance_max_relative_change = 0.0

    if args.balance_flow:
        flow_balance_converged = False
        for iteration in range(
            1,
            args.flow_balance_max_iterations + 1,
        ):
            delta_pressure = np.zeros(
                n_ties,
                dtype=float,
            )

            for tie_index in range(n_ties):
                trial = solve_tie_tube_counterflow(
                    axial_total_tie_power_w=(
                        tie_solid_axial[tie_index, :]
                    ),
                    axial_direct_supply_hydrogen_power_w=(
                        tie_supply_h2_axial[tie_index, :]
                    ),
                    axial_direct_return_hydrogen_power_w=(
                        tie_return_h2_axial[tie_index, :]
                    ),
                    z_edges_m=z_edges_m,
                    tie_tube_count=1,
                    mass_flow_per_tie_kg_s=float(
                        tie_mass_flow[tie_index]
                    ),
                    inlet_temperature_k=(
                        args.inlet_temperature_k
                    ),
                    inlet_pressure_pa=(
                        args.inlet_pressure_mpa * 1.0e6
                    ),
                    config=config,
                    property_model=property_model,
                    supply_heat_fraction=(
                        args.supply_heat_fraction
                    ),
                )
                delta_pressure[tie_index] = (
                    args.inlet_pressure_mpa * 1.0e6
                    - trial.outlet_pressure_pa
                )

            update = update_parallel_mass_flow(
                tie_mass_flow,
                delta_pressure,
                total_mass_flow_kg_s=args.tie_mass_flow_kg_s,
                relaxation=args.flow_balance_relaxation,
            )
            tie_mass_flow = update.mass_flow_kg_s
            flow_balance_iterations = iteration
            flow_balance_max_relative_change = (
                update.max_relative_change
            )

            if (
                update.max_relative_change
                <= args.flow_balance_tolerance
            ):
                flow_balance_converged = True
                break

    rows: list[dict] = []
    solutions = []
    hottest_index = -1
    hottest_wall_k = -np.inf

    for tie_index in range(n_ties):
        solution = solve_tie_tube_counterflow(
            axial_total_tie_power_w=tie_solid_axial[
                tie_index,
                :,
            ],
            axial_direct_supply_hydrogen_power_w=(
                tie_supply_h2_axial[tie_index, :]
            ),
            axial_direct_return_hydrogen_power_w=(
                tie_return_h2_axial[tie_index, :]
            ),
            z_edges_m=z_edges_m,
            tie_tube_count=1,
            mass_flow_per_tie_kg_s=float(
                tie_mass_flow[tie_index]
            ),
            inlet_temperature_k=args.inlet_temperature_k,
            inlet_pressure_pa=args.inlet_pressure_mpa * 1.0e6,
            config=config,
            property_model=property_model,
            supply_heat_fraction=args.supply_heat_fraction,
        )
        solutions.append(solution)

        max_supply_wall = float(
            np.max(solution.supply.wall_temperature_k)
        )
        max_return_wall = float(
            np.max(solution.return_path.wall_temperature_k)
        )
        max_wall = max(max_supply_wall, max_return_wall)

        row = {
            "tie_instance": tie_index,
            "mass_flow_kg_s": float(
                tie_mass_flow[tie_index]
            ),
            "pressure_drop_Pa": float(
                args.inlet_pressure_mpa * 1.0e6
                - solution.outlet_pressure_pa
            ),
            "solid_power_W": float(
                np.sum(tie_solid_axial[tie_index, :])
            ),
            "direct_supply_hydrogen_power_W": float(
                np.sum(tie_supply_h2_axial[tie_index, :])
            ),
            "direct_return_hydrogen_power_W": float(
                np.sum(tie_return_h2_axial[tie_index, :])
            ),
            "total_absorbed_power_W": float(
                solution.total_absorbed_power_w
            ),
            "outlet_temperature_K": float(
                solution.outlet_temperature_k
            ),
            "outlet_pressure_Pa": float(
                solution.outlet_pressure_pa
            ),
            "max_supply_wall_temperature_K": max_supply_wall,
            "max_return_wall_temperature_K": max_return_wall,
            "max_wall_temperature_K": max_wall,
            "min_supply_reynolds": float(
                np.min(solution.supply.reynolds)
            ),
            "min_return_reynolds": float(
                np.min(solution.return_path.reynolds)
            ),
        }
        rows.append(row)

        if max_wall > hottest_wall_k:
            hottest_wall_k = max_wall
            hottest_index = tie_index

    args.output.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output / "tie_instance_summary.csv"
    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)

    if hottest_index < 0:
        raise RuntimeError("no tie-tube solutions were produced")

    hottest = solutions[hottest_index]
    _write_path(
        args.output / "hottest_tie_supply_profile.csv",
        hottest.supply,
    )
    _write_path(
        args.output / "hottest_tie_return_profile.csv",
        hottest.return_path,
    )

    np.savez_compressed(
        args.output / "tie_instance_reconstruction.npz",
        tie_solid_axial_power_w=tie_solid_axial,
        tie_supply_direct_h2_axial_power_w=tie_supply_h2_axial,
        tie_return_direct_h2_axial_power_w=tie_return_h2_axial,
        tie_mass_flow_kg_s=tie_mass_flow,
        z_edges_m=z_edges_m,
    )

    outlet_temperature = np.asarray(
        [row["outlet_temperature_K"] for row in rows],
        dtype=float,
    )
    wall_temperature = np.asarray(
        [row["max_wall_temperature_K"] for row in rows],
        dtype=float,
    )
    tie_integrated_power = np.asarray(
        [row["solid_power_W"] for row in rows],
        dtype=float,
    )
    final_pressure_drop = np.asarray(
        [row["pressure_drop_Pa"] for row in rows],
        dtype=float,
    )

    summary = {
        "tie_instances": n_ties,
        "tie_mass_flow_kg_s": args.tie_mass_flow_kg_s,
        "mass_flow_per_tie_kg_s": mass_flow_per_tie,
        "minimum_tie_mass_flow_kg_s": float(
            np.min(tie_mass_flow)
        ),
        "maximum_tie_mass_flow_kg_s": float(
            np.max(tie_mass_flow)
        ),
        "reconstructed_total_tie_mass_flow_kg_s": float(
            np.sum(tie_mass_flow)
        ),
        "flow_balance_enabled": args.balance_flow,
        "flow_balance_converged": flow_balance_converged,
        "flow_balance_iterations": flow_balance_iterations,
        "flow_balance_tolerance": (
            args.flow_balance_tolerance
        ),
        "flow_balance_relaxation": (
            args.flow_balance_relaxation
        ),
        "flow_balance_max_relative_change": (
            flow_balance_max_relative_change
        ),
        "final_pressure_drop_spread_fraction": (
            pressure_spread_fraction(final_pressure_drop)
        ),
        "global_tie_solid_power_W": global_tie_power_w,
        "reconstructed_tie_solid_power_W": float(
            np.sum(tie_solid_axial)
        ),
        "global_direct_supply_hydrogen_power_W": (
            global_supply_h2_power_w
        ),
        "reconstructed_direct_supply_hydrogen_power_W": float(
            np.sum(tie_supply_h2_axial)
        ),
        "global_direct_return_hydrogen_power_W": (
            global_return_h2_power_w
        ),
        "reconstructed_direct_return_hydrogen_power_W": float(
            np.sum(tie_return_h2_axial)
        ),
        "tie_integrated_power_max_to_mean": float(
            np.max(tie_integrated_power)
            / np.mean(tie_integrated_power)
        ),
        "maximum_outlet_temperature_K": float(
            np.max(outlet_temperature)
        ),
        "minimum_outlet_temperature_K": float(
            np.min(outlet_temperature)
        ),
        "maximum_wall_temperature_K": float(
            np.max(wall_temperature)
        ),
        "hottest_tie_instance": hottest_index,
        "thermal_axial_bins": n_axial,
        "inlet_temperature_K": args.inlet_temperature_k,
        "inlet_pressure_Pa": args.inlet_pressure_mpa * 1.0e6,
        "hydrogen_property_model": args.hydrogen_model,
        "supply_heat_fraction": (
            hottest.supply_heat_fraction
        ),
        "interpretation": [
            (
                "Per-tie solid axial power shape comes from the summed "
                "OpenMC tie-component Distribcell x axial-mesh tallies."
            ),
            (
                "Per-tie direct supply and return H2 heating shapes come "
                "from dedicated OpenMC Distribcell x axial-mesh tallies."
            ),
            (
                "All instance fields are renormalized to the corresponding "
                "global OpenMC mesh totals for exact energy closure."
            ),
            (
                "Total tie mass flow is divided equally by default. "
                "With --balance-flow, an iterative effective-resistance "
                "update redistributes the fixed total flow toward a common "
                "parallel-branch pressure drop."
            ),
        ],
    }
    (args.output / "tie_instance_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA per-tie thermal solve complete:")
    print(f"  tie instances: {n_ties}")
    print(
        f"  tie solid power max/mean: "
        f"{summary['tie_integrated_power_max_to_mean']:.6f}"
    )
    print(
        f"  flow balance: enabled={args.balance_flow}, "
        f"converged={flow_balance_converged}, "
        f"iterations={flow_balance_iterations}, "
        f"pressure-drop spread="
        f"{summary['final_pressure_drop_spread_fraction']:.6g}"
    )
    print(
        f"  maximum outlet temperature: "
        f"{summary['maximum_outlet_temperature_K']:.3f} K"
    )
    print(
        f"  maximum wall temperature: "
        f"{summary['maximum_wall_temperature_K']:.3f} K"
    )
    print(f"  hottest tie instance: {hottest_index}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
