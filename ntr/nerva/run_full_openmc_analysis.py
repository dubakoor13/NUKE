"""Run one complete real-data NERVA OpenMC -> engineering analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one explicit NERVA OpenMC transport configuration and then "
            "the complete diagnostics/thermal QA pipeline. No optimization "
            "or criticality search is performed."
        )
    )
    parser.add_argument("--cross-sections", type=Path, required=True)
    parser.add_argument(
        "--assembly",
        choices=("core", "cluster", "mixed-core", "reactor"),
        default="reactor",
    )
    parser.add_argument("--rings", type=int, default=5)
    parser.add_argument("--particles", type=int, default=4000)
    parser.add_argument("--batches", type=int, default=80)
    parser.add_argument("--inactive", type=int, default=20)
    parser.add_argument("--drum-angle", type=float, default=0.0)
    parser.add_argument("--photon-transport", action="store_true")
    parser.add_argument("--diagnostic-energy-groups", type=int, default=80)
    parser.add_argument("--axial-mesh-bins", type=int, default=96)

    parser.add_argument("--power-mw", type=float, required=True)
    parser.add_argument("--fuel-mass-flow-kg-s", type=float, required=True)
    parser.add_argument("--tie-mass-flow-kg-s", type=float, required=True)
    parser.add_argument("--inlet-temperature-k", type=float, required=True)
    parser.add_argument("--inlet-pressure-mpa", type=float, required=True)
    parser.add_argument(
        "--hydrogen-model",
        choices=("constant", "coolprop"),
        default="constant",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("build/nerva_full_run"),
    )
    return parser.parse_args()


def _run(arguments: list[str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True)


def _statepoint_from_provenance(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    statepoint = payload.get("statepoint")
    if not isinstance(statepoint, dict):
        raise RuntimeError(
            "transport provenance does not contain a completed statepoint"
        )
    statepoint_path = Path(statepoint["path"])
    if not statepoint_path.is_file():
        raise FileNotFoundError(statepoint_path)
    return statepoint_path


def main() -> int:
    args = parse_args()

    transport_dir = args.output_root / "transport"
    analysis_dir = args.output_root / "analysis"
    args.output_root.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    transport_command = [
        python,
        "-m",
        "ntr.nerva.run_openmc_transport",
        "--cross-sections",
        str(args.cross_sections),
        "--assembly",
        args.assembly,
        "--rings",
        str(args.rings),
        "--particles",
        str(args.particles),
        "--batches",
        str(args.batches),
        "--inactive",
        str(args.inactive),
        "--drum-angle",
        str(args.drum_angle),
        "--diagnostic-energy-groups",
        str(args.diagnostic_energy_groups),
        "--axial-mesh-bins",
        str(args.axial_mesh_bins),
        "--output",
        str(transport_dir),
    ]
    if args.photon_transport:
        transport_command.append("--photon-transport")

    _run(transport_command)

    provenance = transport_dir / "transport_provenance.json"
    statepoint = _statepoint_from_provenance(provenance)

    analysis_command = [
        python,
        "-m",
        "ntr.nerva.run_analysis",
        str(statepoint),
        "--power-mw",
        str(args.power_mw),
        "--rings",
        str(args.rings),
        "--fuel-mass-flow-kg-s",
        str(args.fuel_mass_flow_kg_s),
        "--tie-mass-flow-kg-s",
        str(args.tie_mass_flow_kg_s),
        "--inlet-temperature-k",
        str(args.inlet_temperature_k),
        "--inlet-pressure-mpa",
        str(args.inlet_pressure_mpa),
        "--hydrogen-model",
        args.hydrogen_model,
        "--output-root",
        str(analysis_dir),
    ]
    if args.no_plots:
        analysis_command.append("--no-plots")

    _run(analysis_command)

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "transport_directory": str(transport_dir),
        "analysis_directory": str(analysis_dir),
        "statepoint": str(statepoint),
        "transport_provenance": str(provenance),
        "analysis_validation": str(
            analysis_dir / "analysis_validation.json"
        ),
        "interpretation": (
            "One explicit OpenMC transport + engineering analysis. "
            "PASS means software/data/analysis QA passed; it does not make "
            "the surrogate a validated historical NERVA reactor."
        ),
    }
    summary_path = args.output_root / "full_run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Complete NERVA OpenMC analysis: PASS")
    print(f"  statepoint: {statepoint}")
    print(f"  QA: {summary['analysis_validation']}")
    print(f"  summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
