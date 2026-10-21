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
from .audit_nuclear_data_temperatures import (
    audit_temperature_coverage,
)
from .audit_bound_thermal_scattering import audit_bound_thermal_scattering
from .physical_core_reference_validation import assess_core_fidelity
from .geometry import (
    cluster_summary,
    geometry_summary,
    mixed_core_summary,
)
from .model import build_model
from .core_statepoint_diagnostics import summarize_core_statepoint
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
    parser.add_argument(
        "--statepoint-batches", nargs="+", type=int, default=None,
        help=(
            "Save actual fission-source HDF5 banks at these batch numbers "
            "(including the last batch); permits source shape checks over time."
        ),
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Explicit independent OpenMC random seed for statistical replicas.")
    parser.add_argument("--fuel-temperature-k", type=float,
                        help="Uniform OpenMC fuel surrogate temperature for snapshot feedback.")
    parser.add_argument("--hydrogen-temperature-k", type=float,
                        help="Uniform OpenMC hydrogen material temperature for snapshot feedback.")
    parser.add_argument("--hydrogen-density-g-cm3", type=float,
                        help="Uniform OpenMC hydrogen material density for snapshot feedback.")
    parser.add_argument("--photon-transport", action="store_true")
    parser.add_argument(
        "--thermal-scattering", choices=("free-gas", "evaluated-bound"),
        default="free-gas",
        help="Evaluated bound thermal neutron scattering, or explicit free-gas surrogate.",
    )
    parser.add_argument(
        "--bound-solid-temperature-k", type=float,
        help="Required temperature for Be, graphite and ZrH evaluated bound kernels.",
    )
    parser.add_argument(
        "--bound-nearest-temperature-tolerance-k", type=float, default=0.0,
        help=(
            "Explicit maximum allowed nearest evaluated neutron and S(a,b) "
            "temperature offset, 0..5 K, for evaluated-bound mode only. "
            "The OpenMC transport settings use this same tolerance."
        ),
    )
    parser.add_argument(
        "--heating-mesh-dimensions", nargs=3, type=int, default=(48, 48, 24),
        metavar=("NX", "NY", "NZ"),
        help=(
            "OpenMC regular heating tally bins in x/y/z (default 48 48 24). "
            "For nested-mesh convergence, use integer-multiple dimensions "
            "and identical geometry, nuclear data, score, power, and bounds."
        ),
    )
    parser.add_argument("--diagnostic-energy-groups", type=int, default=80)
    parser.add_argument("--axial-mesh-bins", type=int, default=96)
    parser.add_argument(
        "--target-rel-error",
        type=float,
        default=None,
        help=(
            "Optional OpenMC heating-local relative-error trigger. "
            "This affects statistical stopping only."
        ),
    )
    parser.add_argument("--trigger-max-batches", type=int, default=500)
    parser.add_argument("--trigger-batch-interval", type=int, default=5)
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
        thermal_scattering_treatment=args.thermal_scattering,
        bound_solid_temperature_k=args.bound_solid_temperature_k,
        bound_nearest_temperature_tolerance_k=args.bound_nearest_temperature_tolerance_k,
        fuel_temperature_k=args.fuel_temperature_k,
        hydrogen_temperature_k=(args.hydrogen_temperature_k if args.hydrogen_temperature_k is not None
                                else NervaConfig.hydrogen_temperature_k),
        hydrogen_density_g_cm3=(args.hydrogen_density_g_cm3 if args.hydrogen_density_g_cm3 is not None
                                else NervaConfig.hydrogen_density_g_cm3),
        particles=args.particles,
        batches=args.batches,
        inactive=args.inactive,
        control_drum_angle_deg=args.drum_angle,
        photon_transport=args.photon_transport,
        heating_mesh_dimension=tuple(args.heating_mesh_dimensions),
        diagnostic_energy_groups=args.diagnostic_energy_groups,
        axial_mesh_bins=args.axial_mesh_bins,
        tally_rel_err_trigger=args.target_rel_error,
        trigger_max_batches=args.trigger_max_batches,
        trigger_batch_interval=args.trigger_batch_interval,
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
    thermal_audit = audit_bound_thermal_scattering(cross_sections, config)
    (args.output / "thermal_scattering_physics_audit.json").write_text(
        json.dumps(thermal_audit, indent=2) + "\n",
        encoding="utf-8",
    )
    if (config.thermal_scattering_treatment == "evaluated-bound"
            and not thermal_audit["ready"]):
        raise RuntimeError(
            "evaluated bound neutron scattering preflight failed; "
            "check thermal_scattering_physics_audit.json"
        )

    temperature_audit = audit_temperature_coverage(
        cross_sections=cross_sections,
        config=config,
    )
    temperature_audit_path = (
        args.output / "nuclear_data_temperature_audit.json"
    )
    temperature_audit_path.write_text(
        json.dumps(temperature_audit, indent=2) + "\n",
        encoding="utf-8",
    )

    if (args.fuel_temperature_k is not None
            or args.hydrogen_temperature_k is not None
            or config.thermal_scattering_treatment == "evaluated-bound") and not temperature_audit[
        "model_explicit_temperature_coverage_ok"
    ]:
        raise RuntimeError("OpenMC temperature feedback outside available neutron XS temperature range")

    physical_audit = None
    if args.assembly in ("core", "mixed-core"):
        physical_audit = assess_core_fidelity(config, assembly=args.assembly)
        (args.output / "nasa_core_physical_reference_audit.json").write_text(
            json.dumps(physical_audit, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.statepoint_batches is not None:
        stages = sorted(set(args.statepoint_batches))
        if (len(stages) != len(args.statepoint_batches)
                or stages[0] <= 0 or stages[-1] != args.batches
                or any(stage <= args.inactive for stage in stages)):
            raise ValueError(
                "statepoint batches must be unique active-batch numbers "
                "and must include the final requested batch"
            )
    if args.seed is not None and not 1 <= args.seed < 2**63:
        raise ValueError("OpenMC seed must be a positive 63-bit integer")
    model = build_model(config, assembly=args.assembly)
    if args.statepoint_batches is not None:
        model.settings.statepoint = {"batches": stages}
    if args.seed is not None:
        model.settings.seed = args.seed
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
            "seed": args.seed,
            "batches": config.batches,
            "inactive": config.inactive,
            "statepoint_batches": args.statepoint_batches,
            "control_drum_angle_deg": config.control_drum_angle_deg,
            "fuel_temperature_k": config.fuel_temperature_k,
            "hydrogen_temperature_k": config.hydrogen_temperature_k,
            "thermal_scattering_treatment": config.thermal_scattering_treatment,
            "bound_solid_temperature_k": config.bound_solid_temperature_k,
            "bound_nearest_temperature_tolerance_k": config.bound_nearest_temperature_tolerance_k,
            "hydrogen_density_g_cm3": config.hydrogen_density_g_cm3,
            "photon_transport": config.photon_transport,
            "heating_mesh_dimension": list(config.heating_mesh_dimension),
            "diagnostic_energy_groups": config.diagnostic_energy_groups,
            "axial_mesh_bins": config.axial_mesh_bins,
            "tally_rel_err_trigger": config.tally_rel_err_trigger,
            "trigger_max_batches": config.trigger_max_batches,
            "trigger_batch_interval": config.trigger_batch_interval,
            "uranium_enrichment_wt_percent": (
                config.uranium_enrichment_wt_percent
            ),
        },
        "geometry_summary": _summary(config, args.assembly),
        "nasa_core_physical_reference_audit": physical_audit,
        "nuclear_data": {
            "cross_sections_xml": report.cross_sections_xml,
            "cross_sections_sha256": report.sha256,
            "required_neutron_nuclides": list(
                report.required_neutron_nuclides
            ),
            "photon_transport_requested": (
                report.photon_transport_requested
            ),
            "temperature_audit": str(temperature_audit_path),
            "explicit_temperature_coverage_ok": (
                temperature_audit[
                    "model_explicit_temperature_coverage_ok"
                ]
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
    print(
        f"  tally relative-error trigger: "
        f"{config.tally_rel_err_trigger}"
    )
    print(
        f"  explicit XS temperature coverage: "
        f"{temperature_audit['model_explicit_temperature_coverage_ok']}"
    )
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
    # OpenMC completed an actual executable run: extract its eigenvalue
    # and Shannon entropy observables, but do NOT assert NASA validation.
    with openmc.StatePoint(statepoint) as result:
        core_diagnostics = summarize_core_statepoint(result)
    core_qa_path = args.output / "core_statepoint_diagnostics.json"
    core_qa_path.write_text(
        json.dumps(core_diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    provenance["statepoint"]["core_diagnostics"] = str(core_qa_path)
    provenance["statepoint"]["source_convergence_independently_demonstrated"] = False
    provenance["statepoint"]["historical_nasa_benchmark_validated"] = False
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )

    print("OpenMC transport completed:")
    print(f"  statepoint: {statepoint}")
    print(f"  SHA-256: {provenance['statepoint']['sha256']}")
    print(f"  core numerical diagnostics: {core_qa_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
