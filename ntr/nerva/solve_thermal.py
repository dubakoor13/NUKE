"""Couple normalized OpenMC fuel heating to a 1-D hydrogen fuel channel."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import NervaConfig
from .fuel_conduction import FuelSolidProperties, estimate_fuel_solid_temperatures
from .hydrogen_properties import make_hydrogen_property_model
from .layout import mixed_core_counts
from .thermal import HydrogenProperties, solve_fuel_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a representative NERVA fuel coolant channel."
    )
    parser.add_argument(
        "power_fields",
        type=Path,
        help="nerva_mesh_fields.npz from postprocess_statepoint.",
    )
    parser.add_argument(
        "--rings",
        type=int,
        required=True,
        help="Mixed-core ring count used in the OpenMC reactor model.",
    )
    parser.add_argument(
        "--fuel-mass-flow-kg-s",
        type=float,
        required=True,
        help="Total hydrogen mass flow assigned to fuel-element channels.",
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
        "--fuel-conductivity-w-m-k",
        type=float,
        default=25.0,
        help="Constant fuel-matrix thermal conductivity surrogate.",
    )
    parser.add_argument(
        "--zrc-conductivity-w-m-k",
        type=float,
        default=20.0,
        help="Constant channel-coating ZrC thermal conductivity surrogate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_thermal"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fuel_mass_flow_kg_s <= 0.0:
        raise ValueError("--fuel-mass-flow-kg-s must be positive")

    data = np.load(args.power_fields)
    required = {
        "fuel_power_density_w_cm3",
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

    fuel_power_density = np.asarray(
        data["fuel_power_density_w_cm3"],
        dtype=float,
    )
    lower_left = np.asarray(data["lower_left_cm"], dtype=float)
    upper_right = np.asarray(data["upper_right_cm"], dtype=float)
    dimension = np.asarray(data["dimension"], dtype=int)

    expected_shape = tuple(int(x) for x in dimension)
    if fuel_power_density.shape != expected_shape:
        raise ValueError(
            f"fuel power field shape {fuel_power_density.shape} "
            f"does not match stored dimension {expected_shape}"
        )

    spacing_cm = (upper_right - lower_left) / dimension
    voxel_volume_cm3 = float(np.prod(spacing_cm))
    axial_fuel_power_w = (
        np.sum(fuel_power_density, axis=(0, 1))
        * voxel_volume_cm3
    )

    z_edges_m = np.linspace(
        lower_left[2] / 100.0,
        upper_right[2] / 100.0,
        int(dimension[2]) + 1,
    )

    fuel_elements, _ = mixed_core_counts(args.rings)
    fuel_channels = fuel_elements * 19
    mass_flow_per_channel = args.fuel_mass_flow_kg_s / fuel_channels

    config = NervaConfig(core_rings=args.rings)
    property_model = make_hydrogen_property_model(args.hydrogen_model)
    solution = solve_fuel_channel(
        axial_total_fuel_power_w=axial_fuel_power_w,
        z_edges_m=z_edges_m,
        fuel_channel_count=fuel_channels,
        mass_flow_per_channel_kg_s=mass_flow_per_channel,
        inlet_temperature_k=args.inlet_temperature_k,
        inlet_pressure_pa=args.inlet_pressure_mpa * 1.0e6,
        channel_diameter_m=config.coolant_bore_diameter_cm / 100.0,
        properties=HydrogenProperties(),
        property_model=property_model,
    )

    solid = estimate_fuel_solid_temperatures(
        solution,
        config=config,
        properties=FuelSolidProperties(
            fuel_matrix_conductivity_w_m_k=args.fuel_conductivity_w_m_k,
            zrc_conductivity_w_m_k=args.zrc_conductivity_w_m_k,
        ),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "fuel_channel_profile.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "z_m",
                "bulk_temperature_K",
                "coolant_wall_temperature_K",
                "fuel_surface_temperature_K",
                "peak_fuel_temperature_K",
                "pressure_Pa",
                "reynolds",
                "htc_W_m2_K",
                "heat_flux_W_m2",
                "channel_power_W",
            )
        )
        for row in zip(
            solution.z_center_m,
            solution.bulk_temperature_k,
            solution.wall_temperature_k,
            solid.fuel_surface_temperature_k,
            solid.peak_fuel_temperature_k,
            solution.pressure_pa,
            solution.reynolds,
            solution.heat_transfer_coefficient_w_m2_k,
            solution.heat_flux_w_m2,
            solution.channel_power_w,
        ):
            writer.writerow(row)

    summary = {
        "fuel_elements": fuel_elements,
        "fuel_channels": fuel_channels,
        "fuel_mass_flow_kg_s": args.fuel_mass_flow_kg_s,
        "mass_flow_per_channel_kg_s": mass_flow_per_channel,
        "fuel_power_W": float(np.sum(axial_fuel_power_w)),
        "representative_channel_power_W": solution.absorbed_power_w,
        "inlet_temperature_K": args.inlet_temperature_k,
        "outlet_temperature_K": solution.outlet_temperature_k,
        "inlet_pressure_Pa": args.inlet_pressure_mpa * 1.0e6,
        "outlet_pressure_Pa": solution.outlet_pressure_pa,
        "maximum_coolant_wall_temperature_K": float(
            np.max(solution.wall_temperature_k)
        ),
        "maximum_fuel_surface_temperature_K": float(
            np.max(solid.fuel_surface_temperature_k)
        ),
        "maximum_peak_fuel_temperature_K": float(
            np.max(solid.peak_fuel_temperature_k)
        ),
        "effective_half_ligament_m": solid.effective_half_ligament_m,
        "fuel_matrix_conductivity_W_m_K": args.fuel_conductivity_w_m_k,
        "zrc_conductivity_W_m_K": args.zrc_conductivity_w_m_k,
        "hydrogen_property_model": args.hydrogen_model,
        "solid_conduction_model": "1-D ZrC + half-ligament symmetry-slab estimate",
    }
    summary_path = args.output / "thermal_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA representative fuel-channel thermal solve complete:")
    print(f"  fuel elements: {fuel_elements}")
    print(f"  fuel channels: {fuel_channels}")
    print(f"  fuel power: {summary['fuel_power_W'] / 1.0e6:.6g} MW")
    print(f"  outlet temperature: {solution.outlet_temperature_k:.3f} K")
    print(f"  outlet pressure: {solution.outlet_pressure_pa / 1.0e6:.6f} MPa")
    print(
        f"  maximum peak fuel temperature: "
        f"{summary['maximum_peak_fuel_temperature_K']:.3f} K"
    )
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
