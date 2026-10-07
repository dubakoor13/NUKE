"""Postprocess repeated-cell OpenMC tallies for the NERVA model.

This module extracts exact integrated Monte Carlo scores for repeated fuel,
hydrogen-channel, and tie-tube cell instances. It does not infer per-instance
axial shapes that were not explicitly tallied.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import openmc

from .power import EV_TO_J


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract fuel-element, coolant-channel, and tie-component "
            "instance scores from a NERVA OpenMC statepoint."
        )
    )
    parser.add_argument("statepoint", type=Path)
    parser.add_argument(
        "--power-metadata",
        type=Path,
        required=True,
        help="metadata.json from postprocess_statepoint.",
    )
    parser.add_argument(
        "--volume-results",
        type=Path,
        default=None,
        help=(
            "Optional volume_results.json from calculate_volumes. When "
            "provided, integrated cell flux tallies are divided by average "
            "repeated-cell volume to obtain cm^-2 s^-1."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_instances"),
    )
    return parser.parse_args()


def _score(
    tally: openmc.Tally,
    score: str,
    value: str = "mean",
) -> np.ndarray:
    sliced = tally.get_slice(scores=[score])
    data = np.asarray(
        sliced.get_reshaped_data(
            value=value,
            expand_dims=True,
        ),
        dtype=float,
    )
    if data.shape[-2:] != (1, 1):
        raise ValueError(
            f"{tally.name}: unexpected trailing tally shape "
            f"{data.shape[-2:]}"
        )
    return np.asarray(data[..., 0, 0], dtype=float).reshape(-1)




def _distrib_axial_score(
    tally: openmc.Tally,
    score: str,
    value: str = "mean",
) -> tuple[np.ndarray, list[str]]:
    """Return a Distribcell x 1x1xNz mesh score as [instance, z]."""
    distrib = tally.find_filter(openmc.DistribcellFilter)
    mesh_filter = tally.find_filter(openmc.MeshFilter)
    mesh = mesh_filter.mesh
    if not isinstance(mesh, openmc.RegularMesh):
        raise TypeError(f"{tally.name} axial filter must use RegularMesh")

    dimension = tuple(int(v) for v in mesh.dimension)
    if dimension[0] != 1 or dimension[1] != 1:
        raise ValueError(
            f"{tally.name} instance axial mesh must be 1x1xNz, got {dimension}"
        )

    sliced = tally.get_slice(scores=[score])
    data = np.asarray(
        sliced.get_reshaped_data(
            value=value,
            expand_dims=False,
        ),
        dtype=float,
    )
    data = np.asarray(data[..., 0, 0], dtype=float).reshape(
        int(distrib.num_bins),
        int(np.prod(dimension)),
    )
    paths = _paths(tally, int(distrib.num_bins))
    return data, paths


def _axial_z_edges_m(tally: openmc.Tally) -> np.ndarray:
    mesh_filter = tally.find_filter(openmc.MeshFilter)
    mesh = mesh_filter.mesh
    if not isinstance(mesh, openmc.RegularMesh):
        raise TypeError(f"{tally.name} axial filter must use RegularMesh")
    dimension = tuple(int(v) for v in mesh.dimension)
    if dimension[0] != 1 or dimension[1] != 1:
        raise ValueError(
            f"{tally.name} instance axial mesh must be 1x1xNz, got {dimension}"
        )
    return np.linspace(
        float(mesh.lower_left[2]) / 100.0,
        float(mesh.upper_right[2]) / 100.0,
        dimension[2] + 1,
    )


def _paths(
    tally: openmc.Tally,
    count: int,
) -> list[str]:
    try:
        filt = tally.find_filter(openmc.DistribcellFilter)
        values = list(filt.paths)
    except Exception:
        return [""] * count

    if len(values) != count:
        return [""] * count
    return [str(value) for value in values]


def _parent_path(path: str, instance: int) -> str:
    """Return the repeated-universe parent path used as an element key."""
    if path:
        parts = path.split("->")
        if len(parts) > 1:
            return "->".join(parts[:-1])
    return f"instance:{instance}"


def _relative_error(mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mean, dtype=float)
    mask = np.abs(mean) > 0.0
    result[mask] = np.abs(std[mask] / mean[mask])
    return result


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        args.power_metadata.read_text(encoding="utf-8")
    )

    repeated_cell_volumes: dict[str, float] = {}
    if args.volume_results is not None:
        volume_payload = json.loads(
            args.volume_results.read_text(encoding="utf-8")
        )
        repeated_cell_volumes = {
            str(row["name"]): float(
                row["total_repeated_volume_cm3"]
            )
            for row in volume_payload[
                "repeated_cell_total_volumes"
            ]
        }
    source_rate = float(metadata["source_rate_per_s"])
    if source_rate <= 0.0:
        raise ValueError("source_rate_per_s must be positive")

    fuel_rows: list[dict] = []
    channel_rows: list[dict] = []
    tie_rows: list[dict] = []

    fuel_axial_raw = None
    fuel_axial_std_raw = None
    fuel_axial_paths: list[str] = []
    instance_axial_z_edges_m = np.asarray([], dtype=float)
    channel_axial_raw: dict[int, np.ndarray] = {}
    channel_axial_std_raw: dict[int, np.ndarray] = {}
    channel_axial_paths: dict[int, list[str]] = {}
    tie_component_axial_raw: dict[str, np.ndarray] = {}
    tie_component_axial_std_raw: dict[str, np.ndarray] = {}
    tie_component_axial_paths: dict[str, list[str]] = {}
    tie_supply_axial_raw = None
    tie_supply_axial_std_raw = None
    tie_supply_axial_paths: list[str] = []
    tie_return_axial_raw = None
    tie_return_axial_std_raw = None
    tie_return_axial_paths: list[str] = []

    with openmc.StatePoint(args.statepoint) as sp:
        try:
            fuel_tally = sp.get_tally(
                name="nerva_fuel_element_instances"
            )
        except LookupError:
            fuel_tally = None

        if fuel_tally is not None:
            heating = _score(fuel_tally, "heating-local")
            heating_std = _score(
                fuel_tally,
                "heating-local",
                value="std_dev",
            )
            fission = _score(fuel_tally, "fission")
            nu_fission = _score(fuel_tally, "nu-fission")
            flux = _score(fuel_tally, "flux")
            paths = _paths(fuel_tally, len(heating))
            rel = _relative_error(heating, heating_std)

            for i in range(len(heating)):
                fuel_rows.append(
                    {
                        "instance": i,
                        "path": paths[i],
                        "element_key": _parent_path(paths[i], i),
                        "heating_W": (
                            heating[i] * EV_TO_J * source_rate
                        ),
                        "heating_std_W": (
                            heating_std[i] * EV_TO_J * source_rate
                        ),
                        "heating_rel_err": rel[i],
                        "fission_per_s": fission[i] * source_rate,
                        "nu_fission_per_s": (
                            nu_fission[i] * source_rate
                        ),
                        "flux_tally_x_source_rate": (
                            flux[i] * source_rate
                        ),
                        "flux_cm2_s": None,
                    }
                )

        try:
            fuel_axial_tally = sp.get_tally(
                name="nerva_fuel_element_axial_instances"
            )
        except LookupError:
            fuel_axial_tally = None

        if fuel_axial_tally is not None:
            instance_axial_z_edges_m = _axial_z_edges_m(
                fuel_axial_tally
            )
            fuel_axial_raw, fuel_axial_paths = _distrib_axial_score(
                fuel_axial_tally,
                "heating-local",
            )
            fuel_axial_std_raw, _ = _distrib_axial_score(
                fuel_axial_tally,
                "heating-local",
                value="std_dev",
            )
            fuel_axial_raw = (
                fuel_axial_raw * EV_TO_J * source_rate
            )
            fuel_axial_std_raw = (
                fuel_axial_std_raw * EV_TO_J * source_rate
            )

        for channel_index in range(1, 20):
            name = (
                f"nerva_hydrogen_channel_"
                f"{channel_index:02d}_instances"
            )
            try:
                tally = sp.get_tally(name=name)
            except LookupError:
                continue

            heating = _score(tally, "heating-local")
            heating_std = _score(
                tally,
                "heating-local",
                value="std_dev",
            )
            absorption = _score(tally, "absorption")
            flux = _score(tally, "flux")
            paths = _paths(tally, len(heating))
            rel = _relative_error(heating, heating_std)

            for i in range(len(heating)):
                channel_rows.append(
                    {
                        "channel": channel_index,
                        "instance": i,
                        "path": paths[i],
                        "element_key": _parent_path(paths[i], i),
                        "direct_nuclear_heating_W": (
                            heating[i] * EV_TO_J * source_rate
                        ),
                        "heating_std_W": (
                            heating_std[i] * EV_TO_J * source_rate
                        ),
                        "heating_rel_err": rel[i],
                        "absorption_per_s": (
                            absorption[i] * source_rate
                        ),
                        "flux_tally_x_source_rate": (
                            flux[i] * source_rate
                        ),
                        "flux_cm2_s": None,
                    }
                )

            try:
                axial_tally = sp.get_tally(
                    name=(
                        f"nerva_hydrogen_channel_"
                        f"{channel_index:02d}_axial_instances"
                    )
                )
            except LookupError:
                axial_tally = None

            if axial_tally is not None:
                axial_mean, axial_paths = _distrib_axial_score(
                    axial_tally,
                    "heating-local",
                )
                axial_std, _ = _distrib_axial_score(
                    axial_tally,
                    "heating-local",
                    value="std_dev",
                )
                channel_axial_raw[channel_index] = (
                    axial_mean * EV_TO_J * source_rate
                )
                channel_axial_std_raw[channel_index] = (
                    axial_std * EV_TO_J * source_rate
                )
                channel_axial_paths[channel_index] = axial_paths

        for tally in sp.tallies.values():
            if not tally.name.startswith("nerva_tie_instance_"):
                continue

            heating = _score(tally, "heating-local")
            heating_std = _score(
                tally,
                "heating-local",
                value="std_dev",
            )
            absorption = _score(tally, "absorption")
            paths = _paths(tally, len(heating))
            rel = _relative_error(heating, heating_std)
            component = tally.name.removeprefix(
                "nerva_tie_instance_"
            )

            for i in range(len(heating)):
                tie_rows.append(
                    {
                        "component": component,
                        "instance": i,
                        "path": paths[i],
                        "tie_key": _parent_path(paths[i], i),
                        "heating_W": (
                            heating[i] * EV_TO_J * source_rate
                        ),
                        "heating_std_W": (
                            heating_std[i] * EV_TO_J * source_rate
                        ),
                        "heating_rel_err": rel[i],
                        "absorption_per_s": (
                            absorption[i] * source_rate
                        ),
                    }
                )


        for tally in sp.tallies.values():
            if not tally.name.startswith(
                "nerva_tie_axial_instance_"
            ):
                continue
            component = tally.name.removeprefix(
                "nerva_tie_axial_instance_"
            )
            axial_mean, axial_paths = _distrib_axial_score(
                tally,
                "heating-local",
            )
            axial_std, _ = _distrib_axial_score(
                tally,
                "heating-local",
                value="std_dev",
            )
            tie_component_axial_raw[component] = (
                axial_mean * EV_TO_J * source_rate
            )
            tie_component_axial_std_raw[component] = (
                axial_std * EV_TO_J * source_rate
            )
            tie_component_axial_paths[component] = axial_paths
            if instance_axial_z_edges_m.size == 0:
                instance_axial_z_edges_m = _axial_z_edges_m(
                    tally
                )

        try:
            tie_supply_axial_tally = sp.get_tally(
                name="nerva_tie_supply_hydrogen_axial_instances"
            )
        except LookupError:
            tie_supply_axial_tally = None

        if tie_supply_axial_tally is not None:
            tie_supply_axial_raw, tie_supply_axial_paths = (
                _distrib_axial_score(
                    tie_supply_axial_tally,
                    "heating-local",
                )
            )
            tie_supply_axial_std_raw, _ = _distrib_axial_score(
                tie_supply_axial_tally,
                "heating-local",
                value="std_dev",
            )
            tie_supply_axial_raw = (
                tie_supply_axial_raw * EV_TO_J * source_rate
            )
            tie_supply_axial_std_raw = (
                tie_supply_axial_std_raw
                * EV_TO_J
                * source_rate
            )
            if instance_axial_z_edges_m.size == 0:
                instance_axial_z_edges_m = _axial_z_edges_m(
                    tie_supply_axial_tally
                )

        try:
            tie_return_axial_tally = sp.get_tally(
                name="nerva_tie_return_hydrogen_axial_instances"
            )
        except LookupError:
            tie_return_axial_tally = None

        if tie_return_axial_tally is not None:
            tie_return_axial_raw, tie_return_axial_paths = (
                _distrib_axial_score(
                    tie_return_axial_tally,
                    "heating-local",
                )
            )
            tie_return_axial_std_raw, _ = _distrib_axial_score(
                tie_return_axial_tally,
                "heating-local",
                value="std_dev",
            )
            tie_return_axial_raw = (
                tie_return_axial_raw * EV_TO_J * source_rate
            )
            tie_return_axial_std_raw = (
                tie_return_axial_std_raw
                * EV_TO_J
                * source_rate
            )
            if instance_axial_z_edges_m.size == 0:
                instance_axial_z_edges_m = _axial_z_edges_m(
                    tie_return_axial_tally
                )


    if fuel_rows and "fuel matrix" in repeated_cell_volumes:
        avg_volume = (
            repeated_cell_volumes["fuel matrix"]
            / len(fuel_rows)
        )
        if avg_volume > 0.0:
            for row in fuel_rows:
                row["flux_cm2_s"] = (
                    float(row["flux_tally_x_source_rate"])
                    / avg_volume
                )

    for channel_index in range(1, 20):
        name = f"hydrogen channel {channel_index}"
        channel_specific_rows = [
            row
            for row in channel_rows
            if int(row["channel"]) == channel_index
        ]
        if (
            channel_specific_rows
            and name in repeated_cell_volumes
        ):
            avg_volume = (
                repeated_cell_volumes[name]
                / len(channel_specific_rows)
            )
            if avg_volume > 0.0:
                for row in channel_specific_rows:
                    row["flux_cm2_s"] = (
                        float(row["flux_tally_x_source_rate"])
                        / avg_volume
                    )

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=list(rows[0]),
            )
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        args.output / "fuel_element_instances.csv",
        fuel_rows,
    )
    write_csv(
        args.output / "hydrogen_channel_instances.csv",
        channel_rows,
    )
    write_csv(
        args.output / "tie_component_instances.csv",
        tie_rows,
    )

    fuel_power = np.asarray(
        [row["heating_W"] for row in fuel_rows],
        dtype=float,
    )
    fuel_power_std = np.asarray(
        [row["heating_std_W"] for row in fuel_rows],
        dtype=float,
    )
    if fuel_power.size and float(np.sum(fuel_power)) > 0.0:
        fuel_fractions = fuel_power / float(np.sum(fuel_power))
        hottest = int(np.argmax(fuel_power))
        fuel_peaking = float(
            np.max(fuel_power) / np.mean(fuel_power)
        )
    else:
        fuel_fractions = np.asarray([], dtype=float)
        hottest = -1
        fuel_peaking = 0.0


    fuel_key_to_index = {
        str(row["element_key"]): index
        for index, row in enumerate(fuel_rows)
    }
    channel_direct_matrix = np.zeros(
        (len(fuel_rows), 19),
        dtype=float,
    )
    channel_direct_std_matrix = np.zeros(
        (len(fuel_rows), 19),
        dtype=float,
    )
    channel_alignment_mode = (
        "distribcell_parent_path"
        if fuel_rows and all(str(row["path"]) for row in fuel_rows)
        else "instance_index_fallback"
    )
    unmatched_channel_rows = 0
    for row in channel_rows:
        key = str(row["element_key"])
        element_index = fuel_key_to_index.get(key)
        if element_index is None:
            fallback = int(row["instance"])
            if 0 <= fallback < len(fuel_rows):
                element_index = fallback
                channel_alignment_mode = "instance_index_fallback"
            else:
                unmatched_channel_rows += 1
                continue
        channel_index = int(row["channel"]) - 1
        channel_direct_matrix[element_index, channel_index] += float(
            row["direct_nuclear_heating_W"]
        )
        channel_direct_std_matrix[element_index, channel_index] = (
            float(row["heating_std_W"])
        )


    axial_bins = 0
    if fuel_axial_raw is not None:
        axial_bins = int(fuel_axial_raw.shape[1])
    elif channel_axial_raw:
        axial_bins = int(next(iter(channel_axial_raw.values())).shape[1])
    elif tie_component_axial_raw:
        axial_bins = int(
            next(iter(tie_component_axial_raw.values())).shape[1]
        )
    elif tie_supply_axial_raw is not None:
        axial_bins = int(tie_supply_axial_raw.shape[1])

    fuel_element_axial_heating_w = np.zeros(
        (len(fuel_rows), axial_bins),
        dtype=float,
    )
    fuel_element_axial_heating_std_w = np.zeros_like(
        fuel_element_axial_heating_w
    )
    unmatched_fuel_axial_rows = 0

    if fuel_axial_raw is not None:
        for i in range(fuel_axial_raw.shape[0]):
            key = _parent_path(
                fuel_axial_paths[i] if i < len(fuel_axial_paths) else "",
                i,
            )
            element_index = fuel_key_to_index.get(key)
            if element_index is None and i < len(fuel_rows):
                element_index = i
            if element_index is None:
                unmatched_fuel_axial_rows += 1
                continue
            fuel_element_axial_heating_w[element_index, :] = (
                fuel_axial_raw[i, :]
            )
            fuel_element_axial_heating_std_w[element_index, :] = (
                fuel_axial_std_raw[i, :]
            )

    channel_direct_axial_w = np.zeros(
        (len(fuel_rows), 19, axial_bins),
        dtype=float,
    )
    channel_direct_axial_std_w = np.zeros_like(
        channel_direct_axial_w
    )
    unmatched_channel_axial_rows = 0

    for channel_index, axial_values in channel_axial_raw.items():
        paths = channel_axial_paths[channel_index]
        axial_std_values = channel_axial_std_raw[channel_index]
        for i in range(axial_values.shape[0]):
            key = _parent_path(
                paths[i] if i < len(paths) else "",
                i,
            )
            element_index = fuel_key_to_index.get(key)
            if element_index is None and i < len(fuel_rows):
                element_index = i
            if element_index is None:
                unmatched_channel_axial_rows += 1
                continue
            channel_direct_axial_w[
                element_index,
                channel_index - 1,
                :,
            ] = axial_values[i, :]
            channel_direct_axial_std_w[
                element_index,
                channel_index - 1,
                :,
            ] = axial_std_values[i, :]


    tie_keys: list[str] = []
    tie_key_to_index: dict[str, int] = {}
    for row in tie_rows:
        key = str(row["tie_key"])
        if key not in tie_key_to_index:
            tie_key_to_index[key] = len(tie_keys)
            tie_keys.append(key)

    if not tie_keys:
        path_sources = list(tie_component_axial_paths.values())
        if tie_supply_axial_paths:
            path_sources.append(tie_supply_axial_paths)
        if tie_return_axial_paths:
            path_sources.append(tie_return_axial_paths)
        for paths in path_sources:
            for i, path in enumerate(paths):
                key = _parent_path(path, i)
                if key not in tie_key_to_index:
                    tie_key_to_index[key] = len(tie_keys)
                    tie_keys.append(key)

    tie_count = len(tie_keys)
    tie_solid_axial_w = np.zeros(
        (tie_count, axial_bins),
        dtype=float,
    )
    tie_solid_axial_variance_w2 = np.zeros_like(
        tie_solid_axial_w
    )
    unmatched_tie_axial_rows = 0

    for component, axial_values in tie_component_axial_raw.items():
        paths = tie_component_axial_paths[component]
        std_values = tie_component_axial_std_raw[component]
        for i in range(axial_values.shape[0]):
            key = _parent_path(
                paths[i] if i < len(paths) else "",
                i,
            )
            tie_index = tie_key_to_index.get(key)
            if tie_index is None and i < tie_count:
                tie_index = i
            if tie_index is None:
                unmatched_tie_axial_rows += 1
                continue
            tie_solid_axial_w[tie_index, :] += axial_values[i, :]
            tie_solid_axial_variance_w2[tie_index, :] += (
                std_values[i, :] ** 2
            )

    tie_solid_axial_std_w = np.sqrt(
        tie_solid_axial_variance_w2
    )

    def align_tie_hydrogen(
        values: np.ndarray | None,
        std_values: np.ndarray | None,
        paths: list[str],
    ) -> tuple[np.ndarray, np.ndarray, int]:
        aligned = np.zeros((tie_count, axial_bins), dtype=float)
        aligned_std = np.zeros_like(aligned)
        unmatched = 0
        if values is None or std_values is None:
            return aligned, aligned_std, unmatched
        for i in range(values.shape[0]):
            key = _parent_path(
                paths[i] if i < len(paths) else "",
                i,
            )
            tie_index = tie_key_to_index.get(key)
            if tie_index is None and i < tie_count:
                tie_index = i
            if tie_index is None:
                unmatched += 1
                continue
            aligned[tie_index, :] = values[i, :]
            aligned_std[tie_index, :] = std_values[i, :]
        return aligned, aligned_std, unmatched

    (
        tie_supply_hydrogen_axial_w,
        tie_supply_hydrogen_axial_std_w,
        unmatched_tie_supply_rows,
    ) = align_tie_hydrogen(
        tie_supply_axial_raw,
        tie_supply_axial_std_raw,
        tie_supply_axial_paths,
    )
    (
        tie_return_hydrogen_axial_w,
        tie_return_hydrogen_axial_std_w,
        unmatched_tie_return_rows,
    ) = align_tie_hydrogen(
        tie_return_axial_raw,
        tie_return_axial_std_raw,
        tie_return_axial_paths,
    )

    channel_power_by_number = {}
    for channel_index in range(1, 20):
        values = [
            float(row["direct_nuclear_heating_W"])
            for row in channel_rows
            if row["channel"] == channel_index
        ]
        if values:
            channel_power_by_number[str(channel_index)] = float(
                np.sum(values)
            )

    tie_power_by_component: dict[str, float] = {}
    for row in tie_rows:
        component = str(row["component"])
        tie_power_by_component[component] = (
            tie_power_by_component.get(component, 0.0)
            + float(row["heating_W"])
        )

    np.savez_compressed(
        args.output / "instance_power_fractions.npz",
        fuel_element_power_w=fuel_power,
        fuel_element_heating_std_w=fuel_power_std,
        fuel_element_power_fraction=fuel_fractions,
        channel_direct_nuclear_heating_w=channel_direct_matrix,
        channel_direct_nuclear_heating_std_w=channel_direct_std_matrix,
        fuel_element_axial_heating_w=fuel_element_axial_heating_w,
        fuel_element_axial_heating_std_w=(
            fuel_element_axial_heating_std_w
        ),
        channel_direct_nuclear_heating_axial_w=(
            channel_direct_axial_w
        ),
        channel_direct_nuclear_heating_axial_std_w=(
            channel_direct_axial_std_w
        ),
        instance_axial_z_edges_m=instance_axial_z_edges_m,
        tie_solid_axial_heating_w=tie_solid_axial_w,
        tie_solid_axial_heating_std_w=tie_solid_axial_std_w,
        tie_supply_hydrogen_axial_heating_w=(
            tie_supply_hydrogen_axial_w
        ),
        tie_supply_hydrogen_axial_heating_std_w=(
            tie_supply_hydrogen_axial_std_w
        ),
        tie_return_hydrogen_axial_heating_w=(
            tie_return_hydrogen_axial_w
        ),
        tie_return_hydrogen_axial_heating_std_w=(
            tie_return_hydrogen_axial_std_w
        ),
    )

    summary = {
        "statepoint": str(args.statepoint),
        "source_rate_per_s": source_rate,
        "fuel_element_instance_count": len(fuel_rows),
        "fuel_element_power_sum_W": float(
            np.sum(fuel_power)
        ) if fuel_power.size else 0.0,
        "fuel_element_power_quadrature_std_W": float(
            np.sqrt(np.sum(fuel_power_std**2))
        ) if fuel_power_std.size else 0.0,
        "fuel_element_max_to_mean_power": fuel_peaking,
        "hottest_fuel_element_instance": hottest,
        "hydrogen_channel_row_count": len(channel_rows),
        "channel_element_alignment_mode": channel_alignment_mode,
        "unmatched_channel_rows": unmatched_channel_rows,
        "direct_element_axial_tally_available": (
            fuel_axial_raw is not None
        ),
        "direct_channel_axial_tallies_available": (
            len(channel_axial_raw) == 19
        ),
        "instance_axial_bins": axial_bins,
        "unmatched_fuel_axial_rows": unmatched_fuel_axial_rows,
        "unmatched_channel_axial_rows": unmatched_channel_axial_rows,
        "direct_hydrogen_nuclear_heating_by_channel_W": (
            channel_power_by_number
        ),
        "tie_component_row_count": len(tie_rows),
        "tie_instance_count": tie_count,
        "direct_tie_solid_axial_tallies_available": (
            len(tie_component_axial_raw) > 0
        ),
        "direct_tie_supply_axial_tally_available": (
            tie_supply_axial_raw is not None
        ),
        "direct_tie_return_axial_tally_available": (
            tie_return_axial_raw is not None
        ),
        "unmatched_tie_axial_rows": unmatched_tie_axial_rows,
        "unmatched_tie_supply_rows": unmatched_tie_supply_rows,
        "unmatched_tie_return_rows": unmatched_tie_return_rows,
        "tie_heating_by_component_W": tie_power_by_component,
        "important_interpretation": {
            "fuel_instance_heating": (
                "Exact integrated OpenMC heating-local score for each "
                "repeated fuel-matrix cell instance."
            ),
            "channel_heating": (
                "Direct nuclear energy deposited in hydrogen. It is not "
                "the convective heat transferred from the fuel matrix."
            ),
            "axial_distribution": (
                "When present, fuel-element and channel-instance axial "
                "heating arrays come directly from Distribcell x axial-mesh "
                "OpenMC tallies. Integrated instance tallies remain the "
                "normalization reference for element/channel totals."
            ),
            "flux_units": (
                "flux_tally_x_source_rate is the cell-integrated "
                "tracklength tally scaled by source rate. flux_cm2_s is "
                "populated only when stochastic repeated-cell volumes are "
                "provided and uses average volume per repeated instance."
            ),
        },
    }

    (args.output / "instance_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA repeated-cell OpenMC tallies extracted:")
    print(f"  fuel element instances: {len(fuel_rows)}")
    print(f"  channel instance rows: {len(channel_rows)}")
    print(f"  tie component rows: {len(tie_rows)}")
    if fuel_power.size:
        print(
            f"  fuel element max/mean integrated heating: "
            f"{fuel_peaking:.6f}"
        )
        print(f"  hottest fuel instance: {hottest}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
