"""Prepare/run stochastic OpenMC volume calculations for NERVA diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import openmc

from .config import NervaConfig
from .model import build_model


_INSTANCE_CELL_NAMES = {
    "fuel matrix",
    *{
        f"fuel matrix channel sector {i:02d}"
        for i in range(1, 20)
    },
    *{f"hydrogen channel {i}" for i in range(1, 20)},
    "tie-tube inner Inconel tube",
    "tie-tube ZrH moderator",
    "tie-tube outer Inconel tube",
    "tie-tube ZrC sleeve",
    "tie-tube graphite filler",
    "tie-tube outer ZrC coating",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or run OpenMC stochastic volume calculations for "
            "NERVA materials and repeated-cell definitions."
        )
    )
    parser.add_argument(
        "--cross-sections",
        type=Path,
        default=None,
        help="cross_sections.xml; defaults to OPENMC_CROSS_SECTIONS.",
    )
    parser.add_argument(
        "--assembly",
        choices=("core", "cluster", "mixed-core", "reactor"),
        default="reactor",
    )
    parser.add_argument("--rings", type=int, default=5)
    parser.add_argument("--drum-angle", type=float, default=0.0)
    parser.add_argument("--samples", type=int, default=1_000_000)
    parser.add_argument(
        "--rel-err-trigger",
        type=float,
        default=None,
        help=(
            "Optional stochastic-volume relative-error trigger."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_volumes"),
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Invoke OpenMC stochastic volume calculation.",
    )
    return parser.parse_args()


def _bounding_box(
    config: NervaConfig,
    assembly: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    half = 0.5 * config.active_length_cm
    if assembly == "cluster":
        radius = config.cluster_reflector_outer_radius_cm
    elif assembly == "reactor":
        radius = config.pressure_vessel_outer_radius_cm
    else:
        radius = config.reflector_outer_radius_cm
    return (-radius, -radius, -half), (radius, radius, half)


def _pair(value) -> tuple[float, float]:
    if hasattr(value, "n") and hasattr(value, "s"):
        return float(value.n), float(value.s)
    if hasattr(value, "nominal_value") and hasattr(value, "std_dev"):
        return float(value.nominal_value), float(value.std_dev)
    return float(value), 0.0


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.rel_err_trigger is not None:
        if not (0.0 < args.rel_err_trigger < 1.0):
            raise ValueError("--rel-err-trigger must lie in (0, 1)")

    if args.cross_sections is not None:
        if not args.cross_sections.is_file():
            raise FileNotFoundError(args.cross_sections)
        os.environ["OPENMC_CROSS_SECTIONS"] = str(
            args.cross_sections.resolve()
        )

    config = NervaConfig(
        core_rings=args.rings,
        particles=100,
        batches=12,
        inactive=4,
        control_drum_angle_deg=args.drum_angle,
    )
    model = build_model(config, assembly=args.assembly)
    lower_left, upper_right = _bounding_box(
        config,
        args.assembly,
    )

    material_domains = list(model.materials)
    material_calc = openmc.VolumeCalculation(
        material_domains,
        args.samples,
        lower_left=lower_left,
        upper_right=upper_right,
    )

    all_cells = model.geometry.get_all_cells().values()
    cell_domains = [
        cell
        for cell in all_cells
        if cell.name in _INSTANCE_CELL_NAMES
    ]
    if not cell_domains:
        raise RuntimeError(
            "no repeated-cell diagnostic domains were found"
        )

    cell_calc = openmc.VolumeCalculation(
        cell_domains,
        args.samples,
        lower_left=lower_left,
        upper_right=upper_right,
    )

    if args.rel_err_trigger is not None:
        material_calc.set_trigger(
            args.rel_err_trigger,
            "rel_err",
        )
        cell_calc.set_trigger(
            args.rel_err_trigger,
            "rel_err",
        )

    model.settings.volume_calculations = [
        material_calc,
        cell_calc,
    ]

    args.output.mkdir(parents=True, exist_ok=True)
    model.export_to_xml(directory=args.output)

    specification = {
        "assembly": args.assembly,
        "rings": args.rings,
        "control_drum_angle_deg": args.drum_angle,
        "samples_per_calculation": args.samples,
        "relative_error_trigger": args.rel_err_trigger,
        "bounding_box_cm": {
            "lower_left": list(lower_left),
            "upper_right": list(upper_right),
        },
        "material_domains": [
            {
                "id": material.id,
                "name": material.name,
            }
            for material in material_domains
        ],
        "cell_domains": [
            {
                "id": cell.id,
                "name": cell.name,
            }
            for cell in cell_domains
        ],
        "interpretation": (
            "Cell-domain volumes represent all occurrences of a repeated "
            "cell ID in the geometry. Divide by the independently observed "
            "Distribcell instance count to obtain average identical-instance "
            "volume where appropriate."
        ),
    }
    (args.output / "volume_specification.json").write_text(
        json.dumps(specification, indent=2) + "\n",
        encoding="utf-8",
    )

    if not args.run:
        print("NERVA stochastic volume calculations prepared:")
        print(f"  materials: {len(material_domains)}")
        print(f"  repeated-cell definitions: {len(cell_domains)}")
        print(f"  samples/calculation: {args.samples}")
        print("  execution: skipped (use --run)")
        print(f"  output: {args.output}")
        return 0

    model.calculate_volumes(
        cwd=args.output,
        apply_volumes=True,
        export_model_xml=False,
    )

    material_rows = []
    for material in material_domains:
        value = material_calc.volumes.get(material.id)
        if value is None:
            continue
        mean, std = _pair(value)
        material_rows.append(
            {
                "id": material.id,
                "name": material.name,
                "volume_cm3": mean,
                "std_dev_cm3": std,
                "relative_error": (
                    abs(std / mean) if mean != 0.0 else None
                ),
            }
        )

    cell_by_id = {cell.id: cell for cell in cell_domains}
    cell_rows = []
    for cell_id, value in cell_calc.volumes.items():
        mean, std = _pair(value)
        cell = cell_by_id[cell_id]
        cell_rows.append(
            {
                "id": cell_id,
                "name": cell.name,
                "total_repeated_volume_cm3": mean,
                "std_dev_cm3": std,
                "relative_error": (
                    abs(std / mean) if mean != 0.0 else None
                ),
            }
        )

    result = {
        "specification": specification,
        "material_volumes": material_rows,
        "repeated_cell_total_volumes": cell_rows,
        "volume_files": [
            str(args.output / "volume_1.h5"),
            str(args.output / "volume_2.h5"),
        ],
    }
    (args.output / "volume_results.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA stochastic volume calculations complete:")
    print(f"  material results: {len(material_rows)}")
    print(f"  repeated-cell results: {len(cell_rows)}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
