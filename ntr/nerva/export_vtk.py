"""Export normalized NERVA OpenMC fields to legacy VTK rectilinear-grid format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export cell-centered NERVA OpenMC NPZ fields as an ASCII VTK "
            "rectilinear grid for ParaView/VisIt."
        )
    )
    parser.add_argument(
        "fields",
        type=Path,
        help="nerva_mesh_fields.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_fields.vtk"),
    )
    return parser.parse_args()


def _write_values(stream, values: np.ndarray) -> None:
    flat = np.asarray(values, dtype=float).ravel(order="F")
    for start in range(0, flat.size, 8):
        stream.write(
            " ".join(f"{value:.12e}" for value in flat[start:start + 8])
            + "\n"
        )


def main() -> int:
    args = parse_args()
    data = np.load(args.fields)

    for name in ("dimension", "lower_left_cm", "upper_right_cm"):
        if name not in data.files:
            raise KeyError(f"missing required array {name!r}")

    dimension = np.asarray(data["dimension"], dtype=int)
    lower = np.asarray(data["lower_left_cm"], dtype=float)
    upper = np.asarray(data["upper_right_cm"], dtype=float)
    if dimension.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("dimension/lower_left/upper_right must be length 3")

    shape = tuple(int(v) for v in dimension)
    edges = [
        np.linspace(lower[i], upper[i], int(dimension[i]) + 1)
        for i in range(3)
    ]

    candidate_fields = (
        "power_density_w_cm3",
        "power_density_std_w_cm3",
        "total_heating_power_density_w_cm3",
        "fuel_power_density_w_cm3",
        "tie_power_density_w_cm3",
        "fuel_hydrogen_power_density_w_cm3",
        "tie_hydrogen_power_density_w_cm3",
        "tie_supply_hydrogen_power_density_w_cm3",
        "tie_return_hydrogen_power_density_w_cm3",
        "flux_cm2_s",
        "absorption_rate_cm3_s",
        "fission_rate_cm3_s",
        "nu_fission_rate_cm3_s",
    )

    arrays: dict[str, np.ndarray] = {}
    for name in candidate_fields:
        if name not in data.files:
            continue
        array = np.asarray(data[name], dtype=float)
        if array.shape != shape:
            raise ValueError(
                f"{name} shape {array.shape} != mesh shape {shape}"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values")
        arrays[name] = array

    if not arrays:
        raise ValueError("no supported scalar fields found")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# vtk DataFile Version 3.0\n")
        stream.write("NERVA normalized OpenMC fields\n")
        stream.write("ASCII\n")
        stream.write("DATASET RECTILINEAR_GRID\n")
        stream.write(
            f"DIMENSIONS {shape[0] + 1} {shape[1] + 1} {shape[2] + 1}\n"
        )

        for axis, values in zip(("X", "Y", "Z"), edges):
            stream.write(
                f"{axis}_COORDINATES {len(values)} double\n"
            )
            _write_values(stream, values)

        cell_count = int(np.prod(dimension))
        stream.write(f"CELL_DATA {cell_count}\n")
        for name, array in arrays.items():
            stream.write(f"SCALARS {name} double 1\n")
            stream.write("LOOKUP_TABLE default\n")
            _write_values(stream, array)

    print("NERVA OpenMC VTK export complete:")
    print(f"  mesh: {shape}")
    print(f"  scalar fields: {len(arrays)}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
