"""Execute REAL NERVA OpenMC core eigenvalue transport for multiple RNG seeds.

Requires an installed OpenMC transport executable and evaluated continuous-
energy neutron HDF5 cross-section files. Never substitutes synthetic tallies
or an invented k-effective. Records exact required HDF5 SHA-256 digests,
OpenMC executable identity, statepoint digests, and independent seed settings.

Small/default run settings are for numerical smoke testing ONLY; no result
is represented as a historical NASA NERVA model or source-convergence proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .config import NervaConfig
from .nuclear_data import inspect_nuclear_data
from .audit_nuclear_data_temperatures import audit_temperature_coverage
from .statepoint_content_audit import audit_eigenvalue_statepoint_content


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_nuclear_data(xs: Path, config: NervaConfig, output: Path) -> dict:
    report = inspect_nuclear_data(
        cross_sections=xs, config=config, photon_transport=False,
        require_referenced_files=True,
    )
    if not report.ready:
        raise RuntimeError(
            "official nuclear-data archive is incomplete for this core: "
            f"missing neutron nuclides {list(report.missing_neutron_nuclides)}, "
            f"missing physical HDF5 files {len(report.referenced_files_missing)}"
        )
    audit = audit_temperature_coverage(cross_sections=xs, config=config)
    required = set(report.required_neutron_nuclides)
    invalid = [
        row["nuclide"] for row in audit["nuclides"]
        if row["nuclide"] in required and (
            row["load_error"] is not None or not row["available_temperatures_K"]
        )
    ]
    if invalid:
        raise RuntimeError(
            "OpenMC could not load evaluated neutron HDF5 data for "
            + ", ".join(sorted(invalid))
        )
    if audit["model_explicit_temperature_coverage_ok"] is not True:
        raise RuntimeError("explicit core temperature not covered by evaluated XS")

    paths: dict[str, Path] = {}
    for entry in report.entries:
        if entry.data_type == "neutron" and required.intersection(entry.materials):
            paths[str(Path(entry.resolved_path).resolve())] = Path(entry.resolved_path)
    files = [
        {"path": name, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for name,path in sorted(paths.items())
    ]
    if not files:
        raise RuntimeError("no physical neutron HDF5 file hashes generated")
    manifest = {
        "nuclear_data_source": (
            "Evaluated cross sections supplied to the running OpenMC container; "
            "the image and data-release identity must be independently recorded"
        ),
        "cross_sections_xml": str(xs),
        "cross_sections_sha256": _sha256(xs),
        "required_neutron_nuclides": sorted(required),
        "neutron_hdf5_files": files,
        "all_required_neutron_hdf5_loaded": True,
        "explicit_temperature_coverage_ok": True,
        "nuclear_data_release_edition_verified_externally": False,
        "assumption": "Not an independently certified nuclear data benchmark",
    }
    output.write_text(json.dumps(manifest, indent=2) + "\n",encoding="utf-8")
    return manifest


def _aggregate(reports: list[dict]) -> dict:
    if len(reports) < 2:
        raise ValueError("requires >=2 independent seeds")
    seeds = [int(r["seed"]) for r in reports]
    if any(seed <= 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("OpenMC independent seeds missing/duplicated")
    means = [float(r["keff_mean"]) for r in reports]
    sigmas = [float(r["keff_std_dev"]) for r in reports]
    if any(not math.isfinite(k) or k <= 0 for k in means):
        raise ValueError("nonpositive or nonfinite real OpenMC k-effective")
    if any(not math.isfinite(s) or s <= 0 for s in sigmas):
        raise ValueError("positive OpenMC statistical error required per seed")
    weights = [1 / (s * s) for s in sigmas]
    total_weights = math.fsum(weights)
    weighted = math.fsum(k * w for k, w in zip(means,weights)) / total_weights
    formal_sem = math.sqrt(1 / total_weights)
    q = math.fsum(w * (k - weighted)**2 for k,w in zip(means,weights))
    dof = len(means) - 1
    reduced_q = q / dof
    inflation = math.sqrt(max(1, reduced_q))
    return {
        "independent_seed_count": len(seeds),
        "seeds": seeds,
        "inverse_variance_mean_keff": weighted,
        "formal_mean_keff_standard_error": formal_sem,
        "seed_dispersion_birge_factor": inflation,
        "dispersion_inflated_mean_keff_error": formal_sem * inflation,
        "between_seed_chi_square": q,
        "between_seed_degrees_of_freedom": dof,
        "between_seed_reduced_chi_square": reduced_q,
        "interpretation": (
            "OpenMC eigenvalue estimator consistency across distinct starting "
            "random seeds. Inverse-variance averaging and Birge dispersion "
            "are diagnostics, not proof of source convergence; correlations "
            "within fission-chain generations and nuclear-data systematics "
            "are not independently resolved."
        ),
        "source_convergence_independently_demonstrated": False,
        "historical_nasa_nerva_benchmark_validated": False,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cross-sections",type=Path,default=None,
                   help="OpenMC cross_sections.xml; otherwise OPENMC_CROSS_SECTIONS")
    p.add_argument("--output",type=Path,default=Path("build/nerva_real_core"))
    p.add_argument("--rings",type=int,default=1)
    p.add_argument("--particles",type=int,default=300)
    p.add_argument("--batches",type=int,default=24)
    p.add_argument("--inactive",type=int,default=8)
    p.add_argument("--seeds",type=int,nargs="+",default=(13007,23011,33013,43019))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if shutil.which("openmc") is None:
        raise RuntimeError(
            "Real OpenMC executable unavailable (Python package alone "
            "does not perform neutron transport)"
        )
    if len(args.seeds) < 4 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("real transport evidence needs >=4 different random seeds")
    if any(seed <= 0 for seed in args.seeds):
        raise ValueError("OpenMC seed must be positive")
    xs = args.cross_sections or os.environ.get("OPENMC_CROSS_SECTIONS")
    if not xs:
        raise RuntimeError(
            "Missing continuous-energy evaluated nuclear data; refusing "
            "fake or synthetic k-effective"
        )
    xs = Path(xs).expanduser().resolve()
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    cfg = NervaConfig(
        core_rings=args.rings,particles=args.particles,batches=args.batches,
        inactive=args.inactive,heating_mesh_dimension=(8,8,8),
        axial_mesh_bins=8,diagnostic_energy_groups=8,
    )
    cfg.validate()
    manifest = _verify_nuclear_data(xs,cfg,output/"verified_xs_files.json")
    binary = subprocess.run(["openmc","--version"],capture_output=True,text=True)
    if binary.returncode:
        raise RuntimeError("OpenMC installed binary cannot report its version")
    (output/"openmc_binary_version.txt").write_text(
        binary.stdout + binary.stderr,encoding="utf-8"
    )

    all_runs = []
    for seed in args.seeds:
        run_directory = output / f"seed_{seed}"
        cmd = [
            sys.executable,"-m","ntr.nerva.run_openmc_transport",
            "--cross-sections",str(xs),
            "--assembly","core","--rings",str(args.rings),
            "--particles",str(args.particles),
            "--batches",str(args.batches),"--inactive",str(args.inactive),
            "--heating-mesh-dimensions","8","8","8",
            "--diagnostic-energy-groups","8",
            "--axial-mesh-bins","8",
            "--seed",str(seed),"--output",str(run_directory),
        ]
        print("ACTUAL OPENMC NEUTRON TRANSPORT seed =",seed,flush=True)
        with (output/f"seed_{seed}.log").open("w",encoding="utf-8") as handle:
            proc = subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,
                                  check=False)
        if proc.returncode:
            raise RuntimeError(
                f"Real OpenMC failed for seed={seed}, return code "
                f"{proc.returncode}; inspect {output/f'seed_{seed}.log'}"
            )
        provenance = json.loads(
            (run_directory/"transport_provenance.json").read_text()
        )
        sp_info = provenance.get("statepoint")
        if not isinstance(sp_info,dict):
            raise RuntimeError("No real OpenMC statepoint from completed neutron transport")
        file = Path(sp_info["path"])
        if not file.is_file() or sp_info.get("sha256") != _sha256(file):
            raise RuntimeError("Real OpenMC statepoint bytes mismatch")
        if provenance.get("config",{}).get("seed") != seed:
            raise RuntimeError("OpenMC transport input seed mismatch")
        if provenance["nuclear_data"]["cross_sections_sha256"] != (
            manifest["cross_sections_sha256"]
        ):
            raise RuntimeError("Nuclear data index changed across OpenMC seeds")
        structural = audit_eigenvalue_statepoint_content(file)
        if structural["run_mode"] if "run_mode" in structural else False:
            raise RuntimeError("Unexpected statepoint audit result")
        if structural["photon_transport"]:
            raise RuntimeError("Unrequested photon transport in neutron-only core")
        diagnostics = json.loads(
            (run_directory/"core_statepoint_diagnostics.json").read_text()
        )
        if not math.isfinite(float(diagnostics["keff_mean"])):
            raise RuntimeError("Nonfinite actual eigenvalue estimate")
        all_runs.append({
            "seed":seed,
            "statepoint_sha256":sp_info["sha256"],
            "statepoint_bytes":sp_info["size_bytes"],
            "statepoint_path":str(file),
            "keff_mean":diagnostics["keff_mean"],
            "keff_std_dev":diagnostics["keff_std_dev"],
            "shannon_entropy_active_window_shift":
                diagnostics["source_entropy_active_window_shift"],
            "keff_serial_correlation":diagnostics.get(
                "per_batch_keff_serial_correlation"
            ),
            "entropy_serial_correlation":diagnostics.get(
                "source_entropy_serial_correlation"
            ),
            "hdf5_eigenvalue_integrity":structural,
        })
        (output/"progress.json").write_text(json.dumps(all_runs,indent=2)+"\n")

    if len({r["statepoint_sha256"] for r in all_runs}) != len(all_runs):
        raise RuntimeError("Duplicate real neutron transport statepoint contents")
    summary = {
        "execution": "ACTUAL_OPENMC_TRANSPORT",
        "model_scope": "NERVA-derived core; not calibrated historical geometry",
        "assembly": "core",
        "rings":args.rings,
        "particles_per_generation":args.particles,
        "batches":args.batches,"inactive_batches":args.inactive,
        "cross_sections_xml_sha256":manifest["cross_sections_sha256"],
        "run_statistics":_aggregate(all_runs),
        "runs":all_runs,
    }
    (output/"real_core_transport_summary.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n"
    )
    print(json.dumps(summary["run_statistics"],indent=2),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
