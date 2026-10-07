"""Convert a NERVA OpenMC statepoint into normalized engineering fields."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import openmc

from .power import EV_TO_J, normalize_regular_mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize the NERVA mesh tally to a requested reactor thermal power."
    )
    parser.add_argument("statepoint", type=Path, help="OpenMC statepoint HDF5 file.")
    parser.add_argument(
        "--power-mw",
        type=float,
        required=True,
        help="Total reactor thermal power used for normalization, in MW.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_power"),
        help="Directory for NPZ/CSV/JSON normalized outputs.",
    )
    return parser.parse_args()


def _score_mesh_field(tally: openmc.Tally, score: str) -> np.ndarray:
    """Return a mesh score, summing any filters that follow the mesh filter."""
    mesh_filter = tally.find_filter(openmc.MeshFilter)
    mesh_shape = tuple(int(value) for value in mesh_filter.mesh.dimension)

    sliced = tally.get_slice(scores=[score])
    values = np.asarray(
        sliced.get_reshaped_data(value="mean", expand_dims=True),
        dtype=float,
    )

    # Last two dimensions are nuclide and score. Both are singleton after
    # score slicing in the tallies used here.
    if values.shape[-2:] != (1, 1):
        raise ValueError(
            f"unexpected trailing tally dimensions {values.shape[-2:]}"
        )
    values = values[..., 0, 0]

    if values.shape[: len(mesh_shape)] != mesh_shape:
        raise ValueError(
            f"expanded mesh tally starts with shape {values.shape}, "
            f"expected mesh prefix {mesh_shape}"
        )

    if values.ndim > len(mesh_shape):
        axes = tuple(range(len(mesh_shape), values.ndim))
        values = np.sum(values, axis=axes)

    return values


def main() -> int:
    args = parse_args()
    if args.power_mw <= 0.0:
        raise ValueError("--power-mw must be positive")

    args.output.mkdir(parents=True, exist_ok=True)

    with openmc.StatePoint(args.statepoint) as statepoint:
        tally = statepoint.get_tally(name="nerva_3d_neutronics")
        mesh_filter = tally.find_filter(openmc.MeshFilter)
        mesh = mesh_filter.mesh

        if not isinstance(mesh, openmc.RegularMesh):
            raise TypeError("nerva_3d_neutronics must use an OpenMC RegularMesh")

        dimension = np.asarray(mesh.dimension, dtype=int)
        lower_left = np.asarray(mesh.lower_left, dtype=float)
        upper_right = np.asarray(mesh.upper_right, dtype=float)

        spacing = (upper_right - lower_left) / dimension
        cell_volume_cm3 = float(np.prod(spacing))

        heating = _score_mesh_field(tally, "heating-local")
        fission = _score_mesh_field(tally, "fission")
        flux = _score_mesh_field(tally, "flux")

        fuel_tally = statepoint.get_tally(name="nerva_3d_fuel_heating")
        fuel_heating = _score_mesh_field(fuel_tally, "heating-local")

        try:
            tie_tally = statepoint.get_tally(name="nerva_3d_tie_heating")
        except LookupError:
            tie_heating = np.zeros_like(fuel_heating)
        else:
            tie_heating = _score_mesh_field(tie_tally, "heating-local")

        normalized = normalize_regular_mesh(
            heating_ev_per_source=heating,
            fission_per_source=fission,
            flux_tracklength_per_source=flux,
            cell_volume_cm3=cell_volume_cm3,
            reactor_power_w=args.power_mw * 1.0e6,
        )

        shape = tuple(int(value) for value in dimension)
        if heating.shape != shape:
            raise ValueError(
                f"expanded mesh tally has shape {heating.shape}, expected {shape}"
            )
        if fuel_heating.shape != shape:
            raise ValueError(
                f"fuel heating mesh has shape {fuel_heating.shape}, expected {shape}"
            )
        if tie_heating.shape != shape:
            raise ValueError(
                f"tie heating mesh has shape {tie_heating.shape}, expected {shape}"
            )

        power_density = normalized.power_density_w_cm3
        fission_rate = normalized.fission_rate_cm3_s
        flux_rate = normalized.flux_cm2_s
        fuel_power_density = (
            fuel_heating * EV_TO_J * normalized.source_rate_s / cell_volume_cm3
        )
        tie_power_density = (
            tie_heating * EV_TO_J * normalized.source_rate_s / cell_volume_cm3
        )

    np.savez_compressed(
        args.output / "nerva_mesh_fields.npz",
        power_density_w_cm3=power_density,
        fission_rate_cm3_s=fission_rate,
        flux_cm2_s=flux_rate,
        fuel_power_density_w_cm3=fuel_power_density,
        tie_power_density_w_cm3=tie_power_density,
        lower_left_cm=lower_left,
        upper_right_cm=upper_right,
        dimension=dimension,
    )

    centers = [
        lower_left[axis] + (np.arange(dimension[axis]) + 0.5) * spacing[axis]
        for axis in range(3)
    ]

    with (args.output / "nerva_mesh_fields.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "i",
                "j",
                "k",
                "x_cm",
                "y_cm",
                "z_cm",
                "power_density_W_cm3",
                "fission_rate_cm3_s",
                "flux_cm2_s",
                "fuel_power_density_W_cm3",
                "tie_power_density_W_cm3",
            )
        )
        for i in range(dimension[0]):
            for j in range(dimension[1]):
                for k in range(dimension[2]):
                    writer.writerow(
                        (
                            i,
                            j,
                            k,
                            centers[0][i],
                            centers[1][j],
                            centers[2][k],
                            power_density[i, j, k],
                            fission_rate[i, j, k],
                            flux_rate[i, j, k],
                            fuel_power_density[i, j, k],
                            tie_power_density[i, j, k],
                        )
                    )

    fuel_power_w = float(np.sum(fuel_power_density) * cell_volume_cm3)
    tie_power_w = float(np.sum(tie_power_density) * cell_volume_cm3)

    metadata = {
        "statepoint": str(args.statepoint),
        "requested_power_MW": args.power_mw,
        "normalized_power_W": normalized.total_power_w,
        "source_rate_per_s": normalized.source_rate_s,
        "fuel_power_W": fuel_power_w,
        "fuel_power_fraction": fuel_power_w / normalized.total_power_w,
        "tie_power_W": tie_power_w,
        "tie_power_fraction": tie_power_w / normalized.total_power_w,
        "cell_volume_cm3": normalized.cell_volume_cm3,
        "dimension": [int(value) for value in dimension],
        "lower_left_cm": [float(value) for value in lower_left],
        "upper_right_cm": [float(value) for value in upper_right],
        "field_units": {
            "power_density": "W/cm3",
            "fission_rate": "reactions/cm3/s",
            "flux": "particles/cm2/s",
            "fuel_power_density": "W/cm3",
            "tie_power_density": "W/cm3",
        },
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Normalized NERVA mesh fields written:")
    print(f"  requested thermal power: {args.power_mw:.6g} MW")
    print(f"  normalized power: {normalized.total_power_w / 1.0e6:.6g} MW")
    print(f"  source rate: {normalized.source_rate_s:.6e} source/s")
    print(f"  fuel-deposited power: {fuel_power_w / 1.0e6:.6g} MW")
    print(f"  fuel power fraction: {fuel_power_w / normalized.total_power_w:.6f}")
    print(f"  tie-solid power: {tie_power_w / 1.0e6:.6g} MW")
    print(f"  tie power fraction: {tie_power_w / normalized.total_power_w:.6f}")
    print(f"  mesh: {tuple(int(value) for value in dimension)}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
