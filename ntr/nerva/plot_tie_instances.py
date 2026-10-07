"""Plot per-tie NERVA thermal results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-tie NERVA thermal results."
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--supply-profile", type=Path, required=True)
    parser.add_argument("--return-profile", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_tie_instance_plots"),
    )
    return parser.parse_args()


def _read_summary(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {key: float(value) for key, value in row.items()}
            )
    return rows


def main() -> int:
    args = parse_args()
    rows = _read_summary(args.summary_csv)
    if not rows:
        raise ValueError("tie-instance summary is empty")

    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)

    instance = np.asarray(
        [row["tie_instance"] for row in rows],
        dtype=float,
    )

    fig, ax = plt.subplots()
    ax.bar(
        instance,
        [row["solid_power_W"] for row in rows],
    )
    ax.set_xlabel("Tie-tube instance")
    ax.set_ylabel("Integrated solid power [W]")
    ax.set_title("OpenMC tie-tube solid heating distribution")
    fig.tight_layout()
    fig.savefig(
        args.output / "tie_instance_solid_power.png",
        dpi=160,
    )
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(
        instance,
        [row["outlet_temperature_K"] for row in rows],
        marker="o",
    )
    ax.set_xlabel("Tie-tube instance")
    ax.set_ylabel("Outlet H2 temperature [K]")
    ax.set_title("Tie-tube outlet temperature distribution")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        args.output / "tie_instance_outlet_temperature.png",
        dpi=160,
    )
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(
        instance,
        [row["max_supply_wall_temperature_K"] for row in rows],
        label="supply wall",
    )
    ax.plot(
        instance,
        [row["max_return_wall_temperature_K"] for row in rows],
        label="return wall",
    )
    ax.set_xlabel("Tie-tube instance")
    ax.set_ylabel("Maximum wall temperature [K]")
    ax.set_title("Tie-tube wall-temperature distribution")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        args.output / "tie_instance_wall_temperature.png",
        dpi=160,
    )
    plt.close(fig)

    supply = np.genfromtxt(
        args.supply_profile,
        delimiter=",",
        names=True,
    )
    return_path = np.genfromtxt(
        args.return_profile,
        delimiter=",",
        names=True,
    )

    fig, ax = plt.subplots()
    ax.plot(
        supply["z_m"],
        supply["bulk_temperature_K"],
        label="supply H2 bulk",
    )
    ax.plot(
        supply["z_m"],
        supply["wall_temperature_K"],
        label="supply wall",
    )
    ax.plot(
        return_path["z_m"],
        return_path["bulk_temperature_K"],
        label="return H2 bulk",
    )
    ax.plot(
        return_path["z_m"],
        return_path["wall_temperature_K"],
        label="return wall",
    )
    ax.set_xlabel("Axial position [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Hottest reconstructed tie tube")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(
        args.output / "hottest_tie_temperature_profile.png",
        dpi=160,
    )
    plt.close(fig)

    print("NERVA tie-instance plots written:")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
