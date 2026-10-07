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
    source_rate = float(metadata["source_rate_per_s"])
    if source_rate <= 0.0:
        raise ValueError("source_rate_per_s must be positive")

    fuel_rows: list[dict] = []
    channel_rows: list[dict] = []
    tie_rows: list[dict] = []

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
                    }
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
                    }
                )

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
        fuel_element_power_fraction=fuel_fractions,
        channel_direct_nuclear_heating_w=channel_direct_matrix,
    )

    summary = {
        "statepoint": str(args.statepoint),
        "source_rate_per_s": source_rate,
        "fuel_element_instance_count": len(fuel_rows),
        "fuel_element_power_sum_W": float(
            np.sum(fuel_power)
        ) if fuel_power.size else 0.0,
        "fuel_element_max_to_mean_power": fuel_peaking,
        "hottest_fuel_element_instance": hottest,
        "hydrogen_channel_row_count": len(channel_rows),
        "channel_element_alignment_mode": channel_alignment_mode,
        "unmatched_channel_rows": unmatched_channel_rows,
        "direct_hydrogen_nuclear_heating_by_channel_W": (
            channel_power_by_number
        ),
        "tie_component_row_count": len(tie_rows),
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
                "No per-instance axial shape is inferred in this file. "
                "Use the separate axial OpenMC tally or a clearly labeled "
                "reconstruction model for axial thermal coupling."
            ),
            "flux_units": (
                "flux_tally_x_source_rate is the raw cell-integrated "
                "tracklength tally scaled by source rate; divide by a "
                "validated cell volume before interpreting as cm^-2 s^-1."
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
