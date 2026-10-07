"""Generate deterministic synthetic NERVA coupling fields for CI integration tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate non-physical synthetic fields for NERVA pipeline tests."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_synthetic/nerva_mesh_fields.npz"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    dimension = np.array([12, 12, 12], dtype=int)
    lower_left_cm = np.array([-12.0, -12.0, -66.0])
    upper_right_cm = np.array([12.0, 12.0, 66.0])

    x = np.linspace(-1.0, 1.0, dimension[0], endpoint=False) + 1.0 / dimension[0]
    y = np.linspace(-1.0, 1.0, dimension[1], endpoint=False) + 1.0 / dimension[1]
    z = np.linspace(-1.0, 1.0, dimension[2], endpoint=False) + 1.0 / dimension[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    radial = np.exp(-2.5 * (xx**2 + yy**2))
    axial = np.maximum(0.0, np.cos(0.5 * np.pi * zz))
    shape = radial * axial

    spacing_cm = (upper_right_cm - lower_left_cm) / dimension
    voxel_volume_cm3 = float(np.prod(spacing_cm))

    def normalized_density(total_power_w: float) -> np.ndarray:
        field = np.asarray(shape, dtype=float)
        scale = total_power_w / (np.sum(field) * voxel_volume_cm3)
        return field * scale

    total_power_density = normalized_density(1.0e6)
    fuel_power_density = normalized_density(0.70e6)
    tie_power_density = normalized_density(0.05e6)

    np.savez_compressed(
        args.output,
        power_density_w_cm3=total_power_density,
        fuel_power_density_w_cm3=fuel_power_density,
        tie_power_density_w_cm3=tie_power_density,
        fission_rate_cm3_s=np.zeros_like(total_power_density),
        flux_cm2_s=np.zeros_like(total_power_density),
        lower_left_cm=lower_left_cm,
        upper_right_cm=upper_right_cm,
        dimension=dimension,
    )

    print("Synthetic NERVA coupling fields written:")
    print(f"  output: {args.output}")
    print("  total deposited power: 1.0 MW")
    print("  fuel deposited power: 0.70 MW")
    print("  tie-solid deposited power: 0.05 MW")
    print("  NOTE: synthetic CI data; not a physical neutronics solution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
