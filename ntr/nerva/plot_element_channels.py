"""Plot element/channel-resolved NERVA thermal reconstruction outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot NERVA element × 19-channel thermal reconstruction."
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--hottest-profile",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_element_channel_plots"),
    )
    return parser.parse_args()


def _matplotlib():
    import matplotlib.pyplot as plt
    return plt


def _read_rows(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    key: float(value)
                    for key, value in row.items()
                }
            )
    return rows


def _matrix(
    rows: list[dict[str, float]],
    field: str,
) -> np.ndarray:
    n_elements = 1 + max(int(row["element_instance"]) for row in rows)
    n_channels = max(int(row["channel"]) for row in rows)
    values = np.full((n_elements, n_channels), np.nan)
    for row in rows:
        values[
            int(row["element_instance"]),
            int(row["channel"]) - 1,
        ] = row[field]
    return values


def main() -> int:
    args = parse_args()
    rows = _read_rows(args.summary_csv)
    if not rows:
        raise ValueError("element/channel summary is empty")

    args.output.mkdir(parents=True, exist_ok=True)
    plt = _matplotlib()

    for field, title, label, filename in (
        (
            "outlet_temperature_K",
            "Fuel-channel outlet temperature",
            "K",
            "element_channel_outlet_temperature.png",
        ),
        (
            "max_peak_fuel_temperature_K",
            "Peak fuel temperature by element/channel",
            "K",
            "element_channel_peak_fuel_temperature.png",
        ),
        (
            "direct_nuclear_hydrogen_power_W",
            "Direct nuclear H2 heating by element/channel",
            "W",
            "element_channel_direct_h2_heating.png",
        ),
    ):
        values = _matrix(rows, field)
        fig, ax = plt.subplots()
        image = ax.imshow(values, aspect="auto", origin="lower")
        ax.set_xlabel("Channel index")
        ax.set_ylabel("Fuel-element instance")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label=label)
        fig.tight_layout()
        fig.savefig(args.output / filename, dpi=160)
        plt.close(fig)

    element_fraction: dict[int, float] = {}
    for row in rows:
        idx = int(row["element_instance"])
        element_fraction[idx] = row["fuel_element_power_fraction"]

    fig, ax = plt.subplots()
    keys = sorted(element_fraction)
    ax.bar(keys, [element_fraction[key] for key in keys])
    ax.set_xlabel("Fuel-element instance")
    ax.set_ylabel("Integrated fuel power fraction")
    ax.set_title("OpenMC repeated fuel-element power distribution")
    fig.tight_layout()
    fig.savefig(
        args.output / "fuel_element_power_fraction.png",
        dpi=160,
    )
    plt.close(fig)

    profile = np.genfromtxt(
        args.hottest_profile,
        delimiter=",",
        names=True,
    )
    fig, ax = plt.subplots()
    ax.plot(profile["z_m"], profile["bulk_temperature_K"], label="H2 bulk")
    ax.plot(
        profile["z_m"],
        profile["coolant_wall_temperature_K"],
        label="coolant wall",
    )
    ax.plot(
        profile["z_m"],
        profile["peak_fuel_temperature_K"],
        label="peak fuel",
    )
    ax.set_xlabel("Axial position [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Hottest reconstructed fuel channel")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        args.output / "hottest_channel_temperature_profile.png",
        dpi=160,
    )
    plt.close(fig)

    print("NERVA element/channel plots written:")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
