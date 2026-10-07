"""Reproducible single-configuration OpenMC transport runner for NERVA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

import openmc

from .config import NervaConfig
from .geometry import (
    cluster_summary,
    geometry_summary,
    mixed_core_summary,
)
from .model import build_model
from .nuclear_data import inspect_nuclear_data, write_report
from .periphery import reactor_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight, export, and optionally run one explicit NERVA "
            "OpenMC configuration using an external nuclear-data library. "
            "No enrichment/control/geometry search is performed."
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
    parser.add_argument("--particles", type=int, default=4000)
    parser.add_argument("--batches", type=int, default=80)
    parser.add_argument("--inactive", type=int, default=20)
    parser.add_argument("--drum-angle", type=float, default=0.0)
    parser.add_argument("--photon-transport", action="store_true")
    parser.add_argument("--diagnostic-energy-groups", type=int, default=80)
    parser.add_argument("--axial-mesh-bins", type=int, default=96)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_transport"),
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help=(
            "Perform nuclear-data preflight and export XML/provenance, "
            "but do not invoke the OpenMC executable."
        ),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def _summary(config: NervaConfig, assembly: str) -> dict:
    if assembly == "cluster":
        return cluster_summary(config)
    if assembly == "mixed-core":
        return mixed_core_summary(config)
    if assembly == "reactor":
        return reactor_summary(config)
    return geometry_summary(config)


def _xml_checksums(output: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for name in (
        "geometry.xml",
        "materials.xml",
        "settings.xml",
        "tallies.xml",
        "plots.xml",
    ):
        path = output / name
        if path.is_file():
            checksums[name] = _sha256(path)
    return checksums


def main() -> int:
    args = parse_args()

    config = NervaConfig(
        core_rings=args.rings,
        particles=args.particles,
        batches=args.batches,
        inactive=args.inactive,
        control_drum_angle_deg=args.drum_angle,
        photon_transport=args.photon_transport,
        diagnostic_energy_groups=args.diagnostic_energy_groups,
        axial_mesh_bins=args.axial_mesh_bins,
    )
    config.validate()

    report = inspect_nuclear_data(
        cross_sections=args.cross_sections,
        config=config,
        photon_transport=args.photon_transport,
        require_referenced_files=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_report(
        report,
        args.output / "nuclear_data_report.json",
    )
    if not report.ready:
        raise RuntimeError(
            "nuclear-data preflight failed; inspect "
            f"{args.output / 'nuclear_data_report.json'}"
        )

    cross_sections = Path(report.cross_sections_xml)
    os.environ["OPENMC_CROSS_SECTIONS"] = str(cross_sections)

    model = build_model(config, assembly=args.assembly)
    model.export_to_xml(directory=args.output)

    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository": "dubakoor13/NUKE",
        "git_commit": _git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "openmc_version": getattr(openmc, "__version__", "unknown"),
        "assembly": args.assembly,
        "config": {
            "core_rings": config.core_rings,
            "particles": config.particles,
            "batches": config.batches,
            "inactive": config.inactive,
            "control_drum_angle_deg": config.control_drum_angle_deg,
            "photon_transport": config.photon_transport,
            "diagnostic_energy_groups": config.diagnostic_energy_groups,
            "axial_mesh_bins": config.axial_mesh_bins,
            "uranium_enrichment_wt_percent": (
                config.uranium_enrichment_wt_percent
            ),
        },
        "geometry_summary": _summary(config, args.assembly),
        "nuclear_data": {
            "cross_sections_xml": report.cross_sections_xml,
            "cross_sections_sha256": report.sha256,
            "required_neutron_nuclides": list(
                report.required_neutron_nuclides
            ),
            "photon_transport_requested": (
                report.photon_transport_requested
            ),
        },
        "xml_sha256": _xml_checksums(args.output),
        "run_policy": (
            "single explicit configuration only; no enrichment, geometry, "
            "or control-worth optimization"
        ),
        "statepoint": None,
    }

    provenance_path = args.output / "transport_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA OpenMC transport bundle prepared:")
    print(f"  assembly: {args.assembly}")
    print(f"  nuclear data: {report.cross_sections_xml}")
    print(f"  nuclear-data SHA-256: {report.sha256}")
    print(f"  photon transport: {config.photon_transport}")
    print(f"  XML: {args.output}")
    print(f"  provenance: {provenance_path}")

    if args.export_only:
        print("  OpenMC execution: SKIPPED (--export-only)")
        return 0

    statepoint = Path(model.run(cwd=args.output)).resolve()
    if not statepoint.is_file():
        raise FileNotFoundError(
            f"OpenMC returned missing statepoint: {statepoint}"
        )

    provenance["statepoint"] = {
        "path": str(statepoint),
        "sha256": _sha256(statepoint),
        "size_bytes": statepoint.stat().st_size,
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print("OpenMC transport completed:")
    print(f"  statepoint: {statepoint}")
    print(f"  SHA-256: {provenance['statepoint']['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
