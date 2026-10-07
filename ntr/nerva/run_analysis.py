"""Run the complete post-OpenMC NERVA analysis workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run NERVA statepoint normalization, fuel/tie thermal solves, "
            "and optional plotting."
        )
    )
    parser.add_argument("statepoint", type=Path)
    parser.add_argument("--power-mw", type=float, required=True)
    parser.add_argument("--rings", type=int, required=True)
    parser.add_argument("--fuel-mass-flow-kg-s", type=float, required=True)
    parser.add_argument("--tie-mass-flow-kg-s", type=float, required=True)
    parser.add_argument("--inlet-temperature-k", type=float, required=True)
    parser.add_argument("--inlet-pressure-mpa", type=float, required=True)
    parser.add_argument(
        "--hydrogen-model",
        choices=("constant", "coolprop"),
        default="constant",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("build/nerva_analysis"),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib result generation.",
    )
    return parser.parse_args()


def _run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def main() -> int:
    args = parse_args()

    if args.power_mw <= 0.0:
        raise ValueError("--power-mw must be positive")
    if args.rings < 1:
        raise ValueError("--rings must be >= 1")
    if args.fuel_mass_flow_kg_s <= 0.0:
        raise ValueError("--fuel-mass-flow-kg-s must be positive")
    if args.tie_mass_flow_kg_s <= 0.0:
        raise ValueError("--tie-mass-flow-kg-s must be positive")
    if args.inlet_temperature_k <= 0.0:
        raise ValueError("--inlet-temperature-k must be positive")
    if args.inlet_pressure_mpa <= 0.0:
        raise ValueError("--inlet-pressure-mpa must be positive")

    root = args.output_root
    power_dir = root / "power"
    fuel_dir = root / "fuel"
    tie_dir = root / "tie"
    plots_dir = root / "plots"
    root.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    _run(
        [
            python,
            "-m",
            "ntr.nerva.postprocess_statepoint",
            str(args.statepoint),
            "--power-mw",
            str(args.power_mw),
            "--output",
            str(power_dir),
        ]
    )

    fields = power_dir / "nerva_mesh_fields.npz"

    _run(
        [
            python,
            "-m",
            "ntr.nerva.solve_thermal",
            str(fields),
            "--rings",
            str(args.rings),
            "--fuel-mass-flow-kg-s",
            str(args.fuel_mass_flow_kg_s),
            "--inlet-temperature-k",
            str(args.inlet_temperature_k),
            "--inlet-pressure-mpa",
            str(args.inlet_pressure_mpa),
            "--hydrogen-model",
            args.hydrogen_model,
            "--output",
            str(fuel_dir),
        ]
    )

    _run(
        [
            python,
            "-m",
            "ntr.nerva.solve_tie_tube",
            str(fields),
            "--rings",
            str(args.rings),
            "--tie-mass-flow-kg-s",
            str(args.tie_mass_flow_kg_s),
            "--inlet-temperature-k",
            str(args.inlet_temperature_k),
            "--inlet-pressure-mpa",
            str(args.inlet_pressure_mpa),
            "--hydrogen-model",
            args.hydrogen_model,
            "--output",
            str(tie_dir),
        ]
    )

    if not args.no_plots:
        _run(
            [
                python,
                "-m",
                "ntr.nerva.plot_results",
                "--power-fields",
                str(fields),
                "--fuel-profile",
                str(fuel_dir / "fuel_channel_profile.csv"),
                "--tie-supply-profile",
                str(tie_dir / "tie_supply_profile.csv"),
                "--tie-return-profile",
                str(tie_dir / "tie_return_profile.csv"),
                "--output",
                str(plots_dir),
            ]
        )

    print("NERVA analysis pipeline complete:")
    print(f"  root: {root}")
    print(f"  power fields: {power_dir}")
    print(f"  fuel thermal: {fuel_dir}")
    print(f"  tie thermal: {tie_dir}")
    if not args.no_plots:
        print(f"  plots: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
