"""Element- and channel-resolved NERVA thermal reconstruction.

For current statepoints, OpenMC supplies integrated repeated-cell power plus
direct fuel-element x axial and coolant-channel x axial heating tallies. The
solver uses those native axial shapes on their own mesh and renormalizes them
to the global power totals for strict energy closure.

Older statepoints that do not contain instance-axial tallies fall back to the
global axial shape. The remaining first-order assumption is equal solid-wall
power sharing among the 19 coolant channels inside one fuel element.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .config import NervaConfig
from .fuel_conduction import (
    FuelSolidProperties,
    estimate_fuel_solid_temperatures,
)
from .hydrogen_properties import make_hydrogen_property_model
from .thermal import HydrogenProperties, solve_fuel_channel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve every repeated NERVA fuel element × 19 coolant channels "
            "using OpenMC repeated-cell and direct instance-axial tallies."
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
        "--fuel-mass-flow-kg-s",
        type=float,
        required=True,
    )
    parser.add_argument("--inlet-temperature-k", type=float, required=True)
    parser.add_argument("--inlet-pressure-mpa", type=float, required=True)
    parser.add_argument(
        "--hydrogen-model",
        choices=("constant", "coolprop"),
        default="constant",
    )
    parser.add_argument(
        "--fuel-conductivity-w-m-k",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--zrc-conductivity-w-m-k",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_element_channels"),
    )
    return parser.parse_args()


def _normalized_shape(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    total = float(np.sum(values))
    if total <= 0.0:
        return np.zeros_like(values)
    return values / total


def _rebin_shape_to_edges(
    values: np.ndarray,
    old_edges: np.ndarray,
    new_edges: np.ndarray,
) -> np.ndarray:
    """Conservatively map bin-integrated values onto a new 1-D grid."""
    values = np.asarray(values, dtype=float)
    old_edges = np.asarray(old_edges, dtype=float)
    new_edges = np.asarray(new_edges, dtype=float)
    if old_edges.size != values.size + 1:
        raise ValueError("old_edges must bracket values")
    if new_edges.size < 2:
        raise ValueError("new_edges must contain at least two points")

    result = np.zeros(new_edges.size - 1, dtype=float)
    for i, value in enumerate(values):
        lo = old_edges[i]
        hi = old_edges[i + 1]
        width = hi - lo
        if width <= 0.0 or value == 0.0:
            continue
        for j in range(result.size):
            overlap = max(
                0.0,
                min(hi, new_edges[j + 1]) - max(lo, new_edges[j]),
            )
            if overlap > 0.0:
                result[j] += value * overlap / width
    return result


def main() -> int:
    args = parse_args()
    if args.fuel_mass_flow_kg_s <= 0.0:
        raise ValueError("--fuel-mass-flow-kg-s must be positive")

    fields = np.load(args.power_fields)
    instances = np.load(args.instance_fields)

    required_fields = {
        "fuel_power_density_w_cm3",
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
    if "fuel_element_power_fraction" not in instances.files:
        raise KeyError(
            "instance package missing fuel_element_power_fraction"
        )

    fuel_power_density = np.asarray(
        fields["fuel_power_density_w_cm3"],
        dtype=float,
    )
    if "fuel_hydrogen_power_density_w_cm3" in fields.files:
        direct_h2_density = np.asarray(
            fields["fuel_hydrogen_power_density_w_cm3"],
            dtype=float,
        )
    else:
        direct_h2_density = np.zeros_like(fuel_power_density)

    dimension = np.asarray(fields["dimension"], dtype=int)
    lower_left = np.asarray(fields["lower_left_cm"], dtype=float)
    upper_right = np.asarray(fields["upper_right_cm"], dtype=float)
    shape = tuple(int(v) for v in dimension)
    if fuel_power_density.shape != shape:
        raise ValueError("fuel power field shape mismatch")
    if direct_h2_density.shape != shape:
        raise ValueError("direct H2 power field shape mismatch")

    spacing_cm = (upper_right - lower_left) / dimension
    voxel_volume_cm3 = float(np.prod(spacing_cm))

    axial_fuel_power_w = (
        np.sum(fuel_power_density, axis=(0, 1))
        * voxel_volume_cm3
    )
    axial_direct_h2_power_w = (
        np.sum(direct_h2_density, axis=(0, 1))
        * voxel_volume_cm3
    )

    fuel_total_w = float(np.sum(axial_fuel_power_w))
    direct_total_w = float(np.sum(axial_direct_h2_power_w))
    if fuel_total_w <= 0.0:
        raise ValueError("fuel solid power must be positive")

    fuel_fraction = np.asarray(
        instances["fuel_element_power_fraction"],
        dtype=float,
    ).reshape(-1)
    if fuel_fraction.size == 0:
        raise ValueError("no fuel-element instance fractions")
    if np.any(fuel_fraction < 0.0):
        raise ValueError("fuel-element power fractions must be non-negative")
    fuel_fraction = _normalized_shape(fuel_fraction)

    n_elements = int(fuel_fraction.size)
    n_channels = 19

    if "channel_direct_nuclear_heating_w" in instances.files:
        channel_direct_integrated = np.asarray(
            instances["channel_direct_nuclear_heating_w"],
            dtype=float,
        )
        if channel_direct_integrated.shape != (n_elements, n_channels):
            raise ValueError(
                "channel direct-heating matrix must have shape "
                f"({n_elements}, {n_channels})"
            )
    else:
        channel_direct_integrated = np.zeros(
            (n_elements, n_channels),
            dtype=float,
        )

    direct_fraction = _normalized_shape(
        channel_direct_integrated.reshape(-1)
    ).reshape(n_elements, n_channels)

    fuel_axial_shape = _normalized_shape(axial_fuel_power_w)
    direct_axial_shape = _normalized_shape(axial_direct_h2_power_w)

    legacy_z_edges_m = np.linspace(
        lower_left[2] / 100.0,
        upper_right[2] / 100.0,
        int(dimension[2]) + 1,
    )

    instance_z_edges_m = np.asarray(
        instances["instance_axial_z_edges_m"],
        dtype=float,
    ) if "instance_axial_z_edges_m" in instances.files else np.asarray(
        [],
        dtype=float,
    )
    instance_axial_bins = max(0, instance_z_edges_m.size - 1)

    direct_instance_axial_available = (
        instance_axial_bins > 0
        and "fuel_element_axial_heating_w" in instances.files
        and np.asarray(
            instances["fuel_element_axial_heating_w"]
        ).shape == (n_elements, instance_axial_bins)
        and float(
            np.sum(instances["fuel_element_axial_heating_w"])
        ) > 0.0
    )

    z_edges_m = (
        instance_z_edges_m
        if direct_instance_axial_available
        else legacy_z_edges_m
    )
    thermal_axial_bins = z_edges_m.size - 1

    if direct_instance_axial_available:
        raw_element_axial = np.asarray(
            instances["fuel_element_axial_heating_w"],
            dtype=float,
        )
        element_solid_axial = np.zeros_like(raw_element_axial)
        for element in range(n_elements):
            shape_e = _normalized_shape(
                raw_element_axial[element, :]
            )
            element_solid_axial[element, :] = (
                fuel_fraction[element]
                * fuel_total_w
                * shape_e
            )
    else:
        element_solid_axial = (
            fuel_fraction[:, None]
            * fuel_total_w
            * fuel_axial_shape[None, :]
        )

    direct_fuel_sector_openmc_used = (
        "fuel_channel_sector_heating_w" in instances.files
        and "fuel_channel_sector_axial_heating_w" in instances.files
        and np.asarray(
            instances["fuel_channel_sector_heating_w"]
        ).shape == (n_elements, n_channels)
        and np.asarray(
            instances["fuel_channel_sector_axial_heating_w"]
        ).shape == (
            n_elements,
            n_channels,
            thermal_axial_bins,
        )
        and float(
            np.sum(instances["fuel_channel_sector_heating_w"])
        ) > 0.0
        and float(
            np.sum(
                instances[
                    "fuel_channel_sector_axial_heating_w"
                ]
            )
        ) > 0.0
    )

    sector_axial_zero_fallback_count = 0
    if direct_fuel_sector_openmc_used:
        raw_sector_integrated = np.asarray(
            instances["fuel_channel_sector_heating_w"],
            dtype=float,
        )
        sector_fraction = _normalized_shape(
            raw_sector_integrated.reshape(-1)
        ).reshape(n_elements, n_channels)
        raw_sector_axial = np.asarray(
            instances["fuel_channel_sector_axial_heating_w"],
            dtype=float,
        )

        channel_wall_axial = np.zeros_like(
            raw_sector_axial
        )
        for element in range(n_elements):
            element_shape = _normalized_shape(
                element_solid_axial[element, :]
            )
            for channel in range(n_channels):
                sector_shape = _normalized_shape(
                    raw_sector_axial[element, channel, :]
                )
                if (
                    float(np.sum(sector_shape)) <= 0.0
                    and sector_fraction[element, channel] > 0.0
                ):
                    sector_shape = element_shape
                    sector_axial_zero_fallback_count += 1
                channel_wall_axial[element, channel, :] = (
                    sector_fraction[element, channel]
                    * fuel_total_w
                    * sector_shape
                )
        element_solid_axial = np.sum(
            channel_wall_axial,
            axis=1,
        )
        fuel_fraction = np.sum(
            sector_fraction,
            axis=1,
        )
    else:
        sector_fraction = np.full(
            (n_elements, n_channels),
            1.0 / float(n_elements * n_channels),
            dtype=float,
        )
        channel_wall_axial = (
            element_solid_axial[:, None, :]
            / float(n_channels)
        )

    direct_channel_axial_available = (
        "channel_direct_nuclear_heating_axial_w" in instances.files
        and np.asarray(
            instances[
                "channel_direct_nuclear_heating_axial_w"
            ]
        ).shape == (
            n_elements,
            n_channels,
            thermal_axial_bins,
        )
        and float(
            np.sum(
                instances[
                    "channel_direct_nuclear_heating_axial_w"
                ]
            )
        ) > 0.0
    )

    fallback_direct_axial_shape = direct_axial_shape
    if (
        thermal_axial_bins != int(dimension[2])
        and direct_axial_shape.size > 0
    ):
        fallback_direct_axial_shape = _normalized_shape(
            _rebin_shape_to_edges(
                axial_direct_h2_power_w,
                legacy_z_edges_m,
                z_edges_m,
            )
        )

    if direct_total_w > 0.0 and np.sum(direct_fraction) > 0.0:
        if direct_channel_axial_available:
            raw_direct_axial = np.asarray(
                instances[
                    "channel_direct_nuclear_heating_axial_w"
                ],
                dtype=float,
            )
            channel_direct_axial = np.zeros_like(
                raw_direct_axial
            )
            for element in range(n_elements):
                for channel in range(n_channels):
                    shape_ec = _normalized_shape(
                        raw_direct_axial[element, channel, :]
                    )
                    channel_direct_axial[element, channel, :] = (
                        direct_fraction[element, channel]
                        * direct_total_w
                        * shape_ec
                    )
        else:
            channel_direct_axial = (
                direct_fraction[:, :, None]
                * direct_total_w
                * fallback_direct_axial_shape[None, None, :]
            )
    else:
        channel_direct_axial = np.zeros_like(channel_wall_axial)

    total_channels = n_elements * n_channels
    mass_flow_per_channel = (
        args.fuel_mass_flow_kg_s / total_channels
    )

    config = NervaConfig()
    property_model = make_hydrogen_property_model(
        args.hydrogen_model
    )
    solid_properties = FuelSolidProperties(
        fuel_matrix_conductivity_w_m_k=(
            args.fuel_conductivity_w_m_k
        ),
        zrc_conductivity_w_m_k=args.zrc_conductivity_w_m_k,
    )

    rows: list[dict] = []
    hottest = None
    hottest_solution = None
    hottest_solid = None

    for element in range(n_elements):
        for channel in range(n_channels):
            solution = solve_fuel_channel(
                axial_total_fuel_power_w=(
                    channel_wall_axial[element, channel, :]
                ),
                axial_direct_coolant_power_w=(
                    channel_direct_axial[element, channel, :]
                ),
                z_edges_m=z_edges_m,
                fuel_channel_count=1,
                mass_flow_per_channel_kg_s=(
                    mass_flow_per_channel
                ),
                inlet_temperature_k=args.inlet_temperature_k,
                inlet_pressure_pa=args.inlet_pressure_mpa * 1.0e6,
                channel_diameter_m=(
                    config.coolant_bore_diameter_cm / 100.0
                ),
                properties=HydrogenProperties(),
                property_model=property_model,
            )
            solid = estimate_fuel_solid_temperatures(
                solution,
                config=config,
                properties=solid_properties,
            )

            max_peak = float(np.max(solid.peak_fuel_temperature_k))
            row = {
                "element_instance": element,
                "channel": channel + 1,
                "fuel_element_power_fraction": float(
                    fuel_fraction[element]
                ),
                "wall_power_W": float(
                    solution.wall_transferred_power_w
                ),
                "direct_nuclear_hydrogen_power_W": float(
                    solution.direct_nuclear_coolant_power_w
                ),
                "total_channel_absorbed_power_W": float(
                    solution.absorbed_power_w
                ),
                "outlet_temperature_K": float(
                    solution.outlet_temperature_k
                ),
                "outlet_pressure_Pa": float(
                    solution.outlet_pressure_pa
                ),
                "max_coolant_wall_temperature_K": float(
                    np.max(solution.wall_temperature_k)
                ),
                "max_fuel_surface_temperature_K": float(
                    np.max(solid.fuel_surface_temperature_k)
                ),
                "max_peak_fuel_temperature_K": max_peak,
                "min_reynolds": float(np.min(solution.reynolds)),
                "max_reynolds": float(np.max(solution.reynolds)),
            }
            rows.append(row)

            if hottest is None or max_peak > hottest["max_peak_fuel_temperature_K"]:
                hottest = row
                hottest_solution = solution
                hottest_solid = solid

    args.output.mkdir(parents=True, exist_ok=True)

    with (args.output / "element_channel_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if hottest is None or hottest_solution is None or hottest_solid is None:
        raise RuntimeError("no element/channel solutions were produced")

    with (args.output / "hottest_channel_profile.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
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
                "heat_flux_W_m2",
                "wall_heat_W",
                "direct_nuclear_hydrogen_heat_W",
            )
        )
        for values in zip(
            hottest_solution.z_center_m,
            hottest_solution.bulk_temperature_k,
            hottest_solution.wall_temperature_k,
            hottest_solid.fuel_surface_temperature_k,
            hottest_solid.peak_fuel_temperature_k,
            hottest_solution.pressure_pa,
            hottest_solution.reynolds,
            hottest_solution.heat_flux_w_m2,
            hottest_solution.wall_heat_power_w,
            hottest_solution.direct_coolant_power_w,
        ):
            writer.writerow(values)

    np.savez_compressed(
        args.output / "element_channel_power_reconstruction.npz",
        fuel_element_power_fraction=fuel_fraction,
        channel_direct_power_fraction=direct_fraction,
        fuel_channel_sector_power_fraction=sector_fraction,
        global_fuel_axial_shape=fuel_axial_shape,
        global_direct_h2_axial_shape=direct_axial_shape,
        element_solid_axial_power_w=element_solid_axial,
        channel_wall_axial_power_w=channel_wall_axial,
        channel_direct_h2_axial_power_w=channel_direct_axial,
        z_edges_m=z_edges_m,
        native_instance_axial_mesh_used=np.asarray(
            [direct_instance_axial_available],
            dtype=bool,
        ),
    )

    outlet_temperatures = np.asarray(
        [row["outlet_temperature_K"] for row in rows]
    )
    peak_fuel_temperatures = np.asarray(
        [row["max_peak_fuel_temperature_K"] for row in rows]
    )

    summary = {
        "fuel_element_instances": n_elements,
        "channels_per_element": n_channels,
        "total_channels": total_channels,
        "fuel_mass_flow_kg_s": args.fuel_mass_flow_kg_s,
        "mass_flow_per_channel_kg_s": mass_flow_per_channel,
        "global_fuel_solid_power_W": fuel_total_w,
        "global_direct_hydrogen_nuclear_power_W": direct_total_w,
        "reconstructed_wall_power_W": float(
            np.sum(channel_wall_axial)
        ),
        "reconstructed_direct_hydrogen_power_W": float(
            np.sum(channel_direct_axial)
        ),
        "fuel_element_max_to_mean_integrated_power": float(
            np.max(fuel_fraction) / np.mean(fuel_fraction)
        ),
        "maximum_channel_outlet_temperature_K": float(
            np.max(outlet_temperatures)
        ),
        "minimum_channel_outlet_temperature_K": float(
            np.min(outlet_temperatures)
        ),
        "maximum_peak_fuel_temperature_K": float(
            np.max(peak_fuel_temperatures)
        ),
        "hottest_channel": hottest,
        "inlet_temperature_K": args.inlet_temperature_k,
        "inlet_pressure_Pa": args.inlet_pressure_mpa * 1.0e6,
        "fuel_matrix_conductivity_W_m_K": (
            args.fuel_conductivity_w_m_k
        ),
        "zrc_conductivity_W_m_K": (
            args.zrc_conductivity_w_m_k
        ),
        "hydrogen_property_model": args.hydrogen_model,
        "direct_element_axial_openmc_used": (
            direct_instance_axial_available
        ),
        "direct_fuel_sector_openmc_used": (
            direct_fuel_sector_openmc_used
        ),
        "fuel_sector_axial_zero_fallback_count": (
            sector_axial_zero_fallback_count
        ),
        "thermal_axial_bins": int(thermal_axial_bins),
        "direct_channel_axial_openmc_used": (
            direct_channel_axial_available
        ),
        "reconstruction_assumptions": [
            (
                "OpenMC integrated fuel-element fractions are exact tally "
                "outputs after source-rate normalization."
            ),
            (
                "Fuel-element-specific OpenMC axial shapes are used when "
                "the Distribcell x axial-mesh tallies are present; the "
                "global fuel axial shape is only a backward-compatible "
                "fallback."
            ),
            (
                "Current statepoints assign solid wall power from direct "
                "OpenMC nearest-channel fuel-sector integrated and axial "
                "heating tallies. Equal 1/19 sharing is retained only for "
                "legacy statepoints without those tallies."
            ),
            (
                "The Voronoi fuel-sector assignment is a scoring/thermal "
                "partition: all sector cells use the same fuel material and "
                "internal boundaries are transmission-only."
            ),
            (
                "Channel-specific OpenMC direct-H2 axial shapes are used "
                "when available; the global direct-H2 axial shape is only "
                "a backward-compatible fallback."
            ),
            (
                "No unmeasured element-specific axial shape is invented "
                "when direct instance-axial tallies are available."
            ),
        ],
    }
    (args.output / "element_channel_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA element/channel thermal reconstruction complete:")
    print(f"  fuel elements: {n_elements}")
    print(f"  channels solved: {total_channels}")
    print(
        f"  fuel element integrated-power peaking: "
        f"{summary['fuel_element_max_to_mean_integrated_power']:.6f}"
    )
    print(
        f"  maximum channel outlet temperature: "
        f"{summary['maximum_channel_outlet_temperature_K']:.3f} K"
    )
    print(
        f"  maximum peak fuel temperature: "
        f"{summary['maximum_peak_fuel_temperature_K']:.3f} K"
    )
    print(
        f"  hottest element/channel: "
        f"{hottest['element_instance']}/{hottest['channel']}"
    )
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
