"""Plot normalized OpenMC and thermal-coupling outputs for NERVA studies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it with: "
            "python -m pip install matplotlib"
        ) from exc
    return plt


def plot_power_fields(power_fields: Path, output: Path) -> list[Path]:
    data = np.load(power_fields)
    required = {
        "power_density_w_cm3",
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

    output.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()

    lower = np.asarray(data["lower_left_cm"], dtype=float)
    upper = np.asarray(data["upper_right_cm"], dtype=float)
    total = np.asarray(data["power_density_w_cm3"], dtype=float)
    fuel = np.asarray(data["fuel_power_density_w_cm3"], dtype=float)

    created: list[Path] = []

    for name, field, title in (
        ("total_power_midplane.png", total, "Total deposited power density"),
        ("fuel_power_midplane.png", fuel, "Fuel power density"),
    ):
        k = field.shape[2] // 2
        fig, ax = plt.subplots()
        image = ax.imshow(
            field[:, :, k].T,
            origin="lower",
            extent=(lower[0], upper[0], lower[1], upper[1]),
            aspect="equal",
        )
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("y [cm]")
        ax.set_title(f"{title} at axial midplane")
        fig.colorbar(image, ax=ax, label="W/cm^3")
        fig.tight_layout()
        path = output / name
        fig.savefig(path, dpi=180)
        plt.close(fig)
        created.append(path)

    spacing = (upper - lower) / np.asarray(total.shape, dtype=float)
    voxel_volume = float(np.prod(spacing))
    z_edges = np.linspace(lower[2], upper[2], total.shape[2] + 1)
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    total_axial = np.sum(total, axis=(0, 1)) * voxel_volume
    fuel_axial = np.sum(fuel, axis=(0, 1)) * voxel_volume

    fig, ax = plt.subplots()
    ax.plot(z, total_axial / 1.0e6, label="total")
    ax.plot(z, fuel_axial / 1.0e6, label="fuel")
    ax.set_xlabel("z [cm]")
    ax.set_ylabel("Power per axial bin [MW]")
    ax.set_title("Axial deposited power")
    ax.legend()
    fig.tight_layout()
    path = output / "axial_power.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    return created


def _load_csv(path: Path):
    return np.genfromtxt(path, delimiter=",", names=True)


def plot_fuel_thermal(profile: Path, output: Path) -> list[Path]:
    data = _load_csv(profile)
    output.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    created: list[Path] = []

    fig, ax = plt.subplots()
    ax.plot(data["z_m"], data["bulk_temperature_K"], label="bulk H2")
    ax.plot(
        data["z_m"],
        data["coolant_wall_temperature_K"],
        label="coolant wall",
    )
    if "fuel_surface_temperature_K" in data.dtype.names:
        ax.plot(
            data["z_m"],
            data["fuel_surface_temperature_K"],
            label="fuel surface",
        )
    if "peak_fuel_temperature_K" in data.dtype.names:
        ax.plot(
            data["z_m"],
            data["peak_fuel_temperature_K"],
            label="peak fuel estimate",
        )
    ax.set_xlabel("z [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Representative fuel-channel thermal profile")
    ax.legend()
    fig.tight_layout()
    path = output / "fuel_temperatures.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots()
    ax.plot(data["z_m"], data["pressure_Pa"] / 1.0e6)
    ax.set_xlabel("z [m]")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_title("Representative fuel-channel pressure")
    fig.tight_layout()
    path = output / "fuel_pressure.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    return created


def plot_tie_thermal(
    supply_profile: Path,
    return_profile: Path,
    output: Path,
) -> list[Path]:
    supply = _load_csv(supply_profile)
    return_path = _load_csv(return_profile)
    output.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    created: list[Path] = []

    fig, ax = plt.subplots()
    ax.plot(
        supply["z_m"],
        supply["bulk_temperature_K"],
        label="supply bulk H2",
    )
    ax.plot(
        return_path["z_m"],
        return_path["bulk_temperature_K"],
        label="return bulk H2",
    )
    ax.set_xlabel("z [m]")
    ax.set_ylabel("Temperature [K]")
    ax.set_title("Representative tie-tube counterflow temperature")
    ax.legend()
    fig.tight_layout()
    path = output / "tie_temperatures.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    fig, ax = plt.subplots()
    ax.plot(
        supply["z_m"],
        supply["pressure_Pa"] / 1.0e6,
        label="supply",
    )
    ax.plot(
        return_path["z_m"],
        return_path["pressure_Pa"] / 1.0e6,
        label="return",
    )
    ax.set_xlabel("z [m]")
    ax.set_ylabel("Pressure [MPa]")
    ax.set_title("Representative tie-tube counterflow pressure")
    ax.legend()
    fig.tight_layout()
    path = output / "tie_pressure.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(path)

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot NERVA OpenMC and thermal-coupling result products."
    )
    parser.add_argument("--power-fields", type=Path)
    parser.add_argument("--fuel-profile", type=Path)
    parser.add_argument("--tie-supply-profile", type=Path)
    parser.add_argument("--tie-return-profile", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_plots"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    created: list[Path] = []

    if args.power_fields is not None:
        created.extend(plot_power_fields(args.power_fields, args.output))

    if args.fuel_profile is not None:
        created.extend(plot_fuel_thermal(args.fuel_profile, args.output))

    tie_args = (args.tie_supply_profile, args.tie_return_profile)
    if any(value is not None for value in tie_args):
        if any(value is None for value in tie_args):
            raise ValueError(
                "both --tie-supply-profile and --tie-return-profile are required"
            )
        created.extend(
            plot_tie_thermal(
                args.tie_supply_profile,
                args.tie_return_profile,
                args.output,
            )
        )

    if not created:
        raise ValueError(
            "provide --power-fields, --fuel-profile, or both tie profiles"
        )

    print("NERVA result plots written:")
    for path in created:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
