"""Screening propagation of OpenMC instance uncertainty to hottest channel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import NervaConfig
from .fuel_conduction import (
    FuelSolidProperties,
    estimate_fuel_sector_temperatures,
    estimate_fuel_solid_temperatures,
)
from .hydrogen_properties import make_hydrogen_property_model
from .thermal import HydrogenProperties, solve_fuel_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Propagate ±1σ integrated OpenMC instance heating uncertainty "
            "through the hottest reconstructed fuel channel. This is a "
            "screening envelope, not a covariance-aware uncertainty analysis."
        )
    )
    parser.add_argument(
        "--element-summary",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reconstruction",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--instance-fields",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_uncertainty_envelope.json"),
    )
    return parser.parse_args()


def _relative_sigma(mean: float, std: float) -> float:
    if mean <= 0.0:
        return 0.0
    return abs(std / mean)


def _solve(
    wall_power: np.ndarray,
    direct_power: np.ndarray,
    z_edges: np.ndarray,
    summary: dict,
) -> dict[str, float]:
    config = NervaConfig()
    property_model = make_hydrogen_property_model(
        summary["hydrogen_property_model"]
    )
    solution = solve_fuel_channel(
        axial_total_fuel_power_w=wall_power,
        axial_direct_coolant_power_w=direct_power,
        z_edges_m=z_edges,
        fuel_channel_count=1,
        mass_flow_per_channel_kg_s=float(
            summary.get(
                "uncertainty_replay_mass_flow_kg_s",
                summary["mass_flow_per_channel_kg_s"],
            )
        ),
        inlet_temperature_k=summary["inlet_temperature_K"],
        inlet_pressure_pa=summary["inlet_pressure_Pa"],
        channel_diameter_m=(
            config.coolant_bore_diameter_cm / 100.0
        ),
        properties=HydrogenProperties(),
        property_model=property_model,
    )
    solid_properties = FuelSolidProperties(
        fuel_matrix_conductivity_w_m_k=summary[
            "fuel_matrix_conductivity_W_m_K"
        ],
        zrc_conductivity_w_m_k=summary[
            "zrc_conductivity_W_m_K"
        ],
    )
    if (
        summary.get("solid_conduction_model")
        == "equivalent-annulus-sector"
    ):
        sector_area_m2 = float(
            summary["uncertainty_replay_sector_fuel_area_m2"]
        )
        solid = estimate_fuel_sector_temperatures(
            solution,
            z_edges_m=z_edges,
            sector_fuel_area_m2=sector_area_m2,
            config=config,
            properties=solid_properties,
        )
    else:
        solid = estimate_fuel_solid_temperatures(
            solution,
            config=config,
            properties=solid_properties,
        )
    return {
        "outlet_temperature_K": solution.outlet_temperature_k,
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
        "absorbed_power_W": solution.absorbed_power_w,
        "wall_power_W": solution.wall_transferred_power_w,
        "direct_nuclear_hydrogen_power_W": (
            solution.direct_nuclear_coolant_power_w
        ),
    }


def main() -> int:
    args = parse_args()

    summary = json.loads(
        args.element_summary.read_text(encoding="utf-8")
    )
    reconstruction = np.load(args.reconstruction)
    instances = np.load(args.instance_fields)

    hottest = summary["hottest_channel"]
    element = int(hottest["element_instance"])
    channel = int(hottest["channel"]) - 1

    wall = np.asarray(
        reconstruction["channel_wall_axial_power_w"],
        dtype=float,
    )[element, channel, :]
    direct = np.asarray(
        reconstruction["channel_direct_h2_axial_power_w"],
        dtype=float,
    )[element, channel, :]
    z_edges = np.asarray(
        reconstruction["z_edges_m"],
        dtype=float,
    )

    if (
        "fuel_channel_sector_heating_w" in instances.files
        and "fuel_channel_sector_heating_std_w"
        in instances.files
    ):
        fuel_mean = np.asarray(
            instances["fuel_channel_sector_heating_w"],
            dtype=float,
        )[element, channel]
        fuel_std = np.asarray(
            instances["fuel_channel_sector_heating_std_w"],
            dtype=float,
        )[element, channel]
        solid_uncertainty_basis = "fuel_channel_sector"
    else:
        fuel_mean = np.asarray(
            instances["fuel_element_power_w"],
            dtype=float,
        )[element]
        fuel_std = np.asarray(
            instances["fuel_element_heating_std_w"],
            dtype=float,
        )[element]
        solid_uncertainty_basis = "fuel_element_legacy"

    direct_mean_matrix = np.asarray(
        instances["channel_direct_nuclear_heating_w"],
        dtype=float,
    )
    direct_std_matrix = np.asarray(
        instances["channel_direct_nuclear_heating_std_w"],
        dtype=float,
    )
    direct_mean = direct_mean_matrix[element, channel]
    direct_std = direct_std_matrix[element, channel]

    wall_rel_sigma = _relative_sigma(fuel_mean, fuel_std)
    direct_rel_sigma = _relative_sigma(direct_mean, direct_std)

    lower_wall = wall * max(0.0, 1.0 - wall_rel_sigma)
    upper_wall = wall * (1.0 + wall_rel_sigma)
    lower_direct = direct * max(0.0, 1.0 - direct_rel_sigma)
    upper_direct = direct * (1.0 + direct_rel_sigma)

    summary["uncertainty_replay_mass_flow_kg_s"] = float(
        hottest.get(
            "mass_flow_kg_s",
            summary["mass_flow_per_channel_kg_s"],
        )
    )

    nominal = _solve(wall, direct, z_edges, summary)
    lower = _solve(lower_wall, lower_direct, z_edges, summary)
    upper = _solve(upper_wall, upper_direct, z_edges, summary)

    result = {
        "element_instance": element,
        "channel": channel + 1,
        "solid_heating_uncertainty_basis": (
            solid_uncertainty_basis
        ),
        "solid_conduction_model": summary.get(
            "solid_conduction_model",
            "ligament-slab",
        ),
        "fuel_element_integrated_heating_relative_sigma": (
            wall_rel_sigma
        ),
        "direct_hydrogen_integrated_heating_relative_sigma": (
            direct_rel_sigma
        ),
        "nominal": nominal,
        "minus_1sigma_screening": lower,
        "plus_1sigma_screening": upper,
        "interpretation": (
            "Screening envelope using independent ±1σ scaling of the "
            "channel-associated fuel-sector OpenMC heating tally when "
            "available (legacy fallback: whole-element heating) and the "
            "direct-H2 tally while preserving "
            "their reconstructed axial shapes. It does not include tally "
            "covariance, cross-section uncertainty, geometry uncertainty, "
            "thermal-property uncertainty, or flow-distribution uncertainty."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA hottest-channel uncertainty screening complete:")
    print(
        f"  element/channel: {element}/{channel + 1}"
    )
    print(
        f"  fuel heating relative σ: {wall_rel_sigma:.6f}"
    )
    print(
        f"  direct H2 heating relative σ: {direct_rel_sigma:.6f}"
    )
    print(
        f"  peak fuel K nominal / -1σ / +1σ: "
        f"{nominal['maximum_peak_fuel_temperature_K']:.3f} / "
        f"{lower['maximum_peak_fuel_temperature_K']:.3f} / "
        f"{upper['maximum_peak_fuel_temperature_K']:.3f}"
    )
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
