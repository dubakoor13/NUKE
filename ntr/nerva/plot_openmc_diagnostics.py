"""Plot advanced NERVA OpenMC diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot advanced NERVA OpenMC diagnostic products."
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        required=True,
        help="openmc_diagnostics.npz",
    )
    parser.add_argument(
        "--material-csv",
        type=Path,
        required=True,
        help="material_transport.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_openmc_plots"),
    )
    return parser.parse_args()


def _matplotlib():
    import matplotlib.pyplot as plt
    return plt


def _material_heating(path: Path) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            if row["score"] != "heating-local":
                continue
            names.append(row["material"])
            values.append(float(row["value"]) / 1.0e6)
    return names, values


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = np.load(args.diagnostics)
    plt = _matplotlib()

    z = data["z_centers_cm"]
    axial_power = data["axial_power_w"] / 1.0e6
    fig, ax = plt.subplots()
    ax.plot(z, axial_power)
    ax.set_xlabel("Axial position [cm]")
    ax.set_ylabel("Power per axial bin [MW]")
    ax.set_title("NERVA OpenMC axial deposited power")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(args.output / "openmc_axial_power.png", dpi=160)
    plt.close(fig)

    energy_bins = data["energy_bins_ev"]
    energy_mid = np.sqrt(
        energy_bins[:, 0] * energy_bins[:, 1]
    )

    fig, ax = plt.subplots()
    ax.loglog(
        energy_mid,
        np.maximum(data["fuel_flux_spectrum_per_s"], 1.0e-300),
    )
    ax.set_xlabel("Neutron energy [eV]")
    ax.set_ylabel("Fuel flux tally × source rate [1/s]")
    ax.set_title("NERVA fuel neutron spectrum")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(args.output / "openmc_fuel_spectrum.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.loglog(
        energy_mid,
        np.maximum(
            data["hydrogen_flux_spectrum_per_s"],
            1.0e-300,
        ),
    )
    ax.set_xlabel("Neutron energy [eV]")
    ax.set_ylabel("Hydrogen flux tally × source rate [1/s]")
    ax.set_title("NERVA hydrogen-region neutron spectrum")
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig(
        args.output / "openmc_hydrogen_spectrum.png",
        dpi=160,
    )
    plt.close(fig)

    names, heating_mw = _material_heating(args.material_csv)
    fig, ax = plt.subplots()
    ax.bar(names, heating_mw)
    ax.set_ylabel("Local deposited power [MW]")
    ax.set_title("NERVA OpenMC material heat deposition")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(
        args.output / "openmc_material_heating.png",
        dpi=160,
    )
    plt.close(fig)

    power = data["power_density_w_cm3"]
    std = data["power_density_std_w_cm3"]
    k = power.shape[2] // 2

    fig, ax = plt.subplots()
    image = ax.imshow(
        power[:, :, k].T,
        origin="lower",
        aspect="equal",
    )
    ax.set_title("OpenMC local heating — axial midplane")
    fig.colorbar(image, ax=ax, label="W/cm³")
    fig.tight_layout()
    fig.savefig(
        args.output / "openmc_power_midplane.png",
        dpi=160,
    )
    plt.close(fig)

    rel = np.zeros_like(power)
    mask = np.abs(power) > 1.0e-12 * np.max(np.abs(power))
    rel[mask] = np.abs(std[mask] / power[mask])

    fig, ax = plt.subplots()
    image = ax.imshow(
        rel[:, :, k].T,
        origin="lower",
        aspect="equal",
    )
    ax.set_title("OpenMC heating relative uncertainty — axial midplane")
    fig.colorbar(image, ax=ax, label="σ / mean")
    fig.tight_layout()
    fig.savefig(
        args.output / "openmc_heating_uncertainty_midplane.png",
        dpi=160,
    )
    plt.close(fig)

    print("NERVA OpenMC diagnostic plots written:")
    for name in (
        "openmc_axial_power.png",
        "openmc_fuel_spectrum.png",
        "openmc_hydrogen_spectrum.png",
        "openmc_material_heating.png",
        "openmc_power_midplane.png",
        "openmc_heating_uncertainty_midplane.png",
    ):
        print(f"  {args.output / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
