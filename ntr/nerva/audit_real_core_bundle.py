"""Verify ACTUAL retained OpenMC four-seed reactor evidence ZIP, read-only.

The input is a downloaded GitHub Actions *real neutron-transport artifact*
(such as the archived NERVA cold surrogate evidence from run 39933618638).
Read each statepoint directly from the ZIP, hash its bytes and inspect HDF5
native run_mode, seed, batch, particle population, and combined k estimator;
cross-check the repository's transport JSON and existing physical material,
source-convergence, criticality and neutron local-heating QA reports.

Important: a real successful OpenMC run and exact archive hashes establish
execution and identity, not criticality, source stationarity, qualified
hot-core nuclear data, or actual neutron/photon local heat. This auditor
cannot promote a synthetic or physically unqualified model into a NASA
NERVA benchmark. Its flags explicitly separate independent evidence gates.

Only standard-library ZIP/JSON and h5py are needed. No network, evaluated
library, OpenMC Python API, geometry generation or reactor parameter tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import h5py
import numpy as np


MAX_ZIP_ENTRIES = 10000
MAX_UNCOMPRESSED_BYTES = 1_000_000_000
MAX_ENTRY_BYTES = 250_000_000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact_json(z: zipfile.ZipFile, path: str) -> dict:
    try:
        raw = z.read(path)
    except KeyError as error:
        raise ValueError("required real-transport report missing: " + path) from error
    if len(raw) > 20_000_000:
        raise ValueError("oversized transport report: " + path)
    try:
        obj = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid transport JSON: " + path) from error
    if not isinstance(obj, dict):
        raise ValueError("transport report must be a JSON object: " + path)
    return obj


def _positive(name: str, obj) -> float:
    x = float(obj)
    if not math.isfinite(x) or x <= 0:
        raise ValueError(name + " must be finite and positive")
    return x


def _statepoint_hdf5(raw: bytes, seed: int, batch: int,
                     particles: int, inactive: int) -> dict:
    try:
        with h5py.File(io.BytesIO(raw), "r") as sp:
            filetype = sp.attrs["filetype"]
            if bytes(filetype).rstrip(b"\0") != b"statepoint":
                raise ValueError("not an actual OpenMC statepoint")
            mode = bytes(sp["run_mode"][()]).rstrip(b"\0")
            if mode != b"eigenvalue":
                raise ValueError("real evidence requires OpenMC eigenvalue statepoint")
            if (int(sp["seed"][()]) != seed or
                int(sp["current_batch"][()]) != batch or
                int(sp["n_inactive"][()]) != inactive or
                int(sp["n_particles"][()]) != particles):
                raise ValueError("OpenMC HDF5 seed or run settings disagree with manifest")
            if int(sp.attrs.get("source_present", 0)) != 1:
                raise ValueError("OpenMC fission source bank is absent")
            active = int(sp["n_realizations"][()])
            if active != batch-inactive:
                raise ValueError("OpenMC active realizations inconsistent with run settings")
            measured = np.asarray(sp["k_combined"][()], dtype=float)
            if measured.shape != (2,):
                raise ValueError("k_combined must have mean and reported standard error")
            keff = _positive("measured k_effective", measured[0])
            sigma = _positive("reported k standard error", measured[1])
            version = [int(x) for x in np.asarray(sp.attrs["openmc_version"])]
            if len(version) != 3 or version[0] < 0:
                raise ValueError("invalid native OpenMC version")
            return {
                "seed": seed,
                "keff_mean": keff,
                "reported_standard_error": sigma,
                "active_realizations": active,
                "openmc_version": version,
                "photon_transport": bool(int(sp.attrs.get("photon_transport", 0))),
                "has_saved_source_bank": True,
                "hdf5_sha256": sha256_bytes(raw),
            }
    except (OSError, KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("incomplete or corrupted actual OpenMC HDF5 evidence") from error


def inspect_real_core_bundle(path: str | Path,
                             *, workflow_run_id: int | None = None) -> dict:
    """Check retained real data, and FAIL CLOSED on physical readiness claims.

    Successful return indicates HDF5 + provenance integrity, NOT hot-core
    readiness. Existing repository qualification reports retain independent
    responsibility for source convergence, evaluated XS and heating.
    """
    archive = Path(path)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if workflow_run_id is not None and (type(workflow_run_id) is not int
                                       or workflow_run_id <= 0):
        raise ValueError("workflow run ID must be a positive integer")
    zip_hash = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            zip_hash.update(block)

    with zipfile.ZipFile(archive) as z:
        members = z.infolist()
        names = [m.filename for m in members]
        if (len(members) < 4 or len(members) > MAX_ZIP_ENTRIES
                or len(set(names)) != len(names)):
            raise ValueError("unexpected or duplicate evidence ZIP entries")
        if any(m.is_dir() or m.file_size > MAX_ENTRY_BYTES or
               m.filename.startswith("/") or
               ".." in Path(m.filename).parts for m in members):
            raise ValueError("invalid/unsafe evidence archive entry")
        if sum(m.file_size for m in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("evidence archive has excessive uncompressed size")

        summary_raw = z.read("real_core_transport_summary.json")
        summary = _artifact_json(z, "real_core_transport_summary.json")
        temperature = _artifact_json(z, "core_material_temperature_qualification.json")
        heating = _artifact_json(z, "heating_data_qualification.json")
        critical = _artifact_json(z, "core_criticality_physical_qualification.json")
        nuclear = _artifact_json(z, "verified_xs_files.json")
        if (summary.get("execution") != "ACTUAL_OPENMC_TRANSPORT" or
            summary.get("assembly") != "core" or
            not summary.get("cold_xs_smoke_test_only", False)):
            raise ValueError("real COLD core transport evidence is required, no synthetic/hot claim")
        batches = int(summary["batches"])
        inactive = int(summary["inactive_batches"])
        particles = int(summary["particles_per_generation"])
        if batches <= inactive or inactive < 0 or particles < 2:
            raise ValueError("invalid source history accounting")

        runs = summary.get("runs")
        if not isinstance(runs, list) or len(runs) < 4:
            raise ValueError("at least four genuinely distinct seeds required")
        expected_hash = sha256_bytes(summary_raw)
        observed = []
        versions = set()
        all_seeds = set()
        all_hashes = set()
        for row in runs:
            if not isinstance(row, dict):
                raise ValueError("malformed transport run metadata")
            seed = int(row["seed"])
            if seed <= 0 or seed in all_seeds:
                raise ValueError("duplicate or invalid source seed")
            all_seeds.add(seed)
            name = f"seed_{seed}/statepoint.{batches:03d}.h5"
            if name not in names:
                raise ValueError("missing real statepoint: " + name)
            raw = z.read(name)
            value = _statepoint_hdf5(raw, seed, batches, particles, inactive)
            if (value["hdf5_sha256"] != row.get("statepoint_sha256") or
                not math.isclose(value["keff_mean"],float(row["keff_mean"]),
                                 abs_tol=1e-12,rel_tol=1e-11) or
                not math.isclose(value["reported_standard_error"],
                                 float(row["keff_std_dev"]),
                                 abs_tol=1e-12,rel_tol=1e-11)):
                raise ValueError("executed HDF5 k or checksum differs from run summary")
            if value["hdf5_sha256"] in all_hashes:
                raise ValueError("independent seeds may not share identical statepoints")
            all_hashes.add(value["hdf5_sha256"])
            versions.add(tuple(value["openmc_version"]))
            observed.append(value)
        if len(versions) != 1:
            raise ValueError("different OpenMC versions among the same transport ensemble")

        weights = np.array([1.0/x["reported_standard_error"]**2 for x in observed])
        means = np.array([x["keff_mean"] for x in observed])
        aggregate = float(np.dot(weights, means)/np.sum(weights))
        reported = _positive(
            "reported aggregate k",
            summary["run_statistics"]["inverse_variance_mean_keff"]
        )
        if not math.isclose(aggregate,reported,abs_tol=1e-12,rel_tol=1e-10):
            raise ValueError("weighted real k estimate differs from reported source ensemble")

        for label, report in (
            ("temperature",temperature),
            ("heating",heating),
            ("criticality",critical),
        ):
            if report.get("real_transport_summary_sha256",
                          report.get("transport_summary_sha256")) != expected_hash:
                raise ValueError(label + " qualification references another transport summary")
        library_sha = summary.get("cross_sections_xml_sha256")
        if (library_sha != nuclear.get("cross_sections_sha256") or
            library_sha != temperature.get("cross_sections_xml_sha256") or
            library_sha != critical.get("cross_sections_xml_sha256")):
            raise ValueError("physical evaluated-data index hashes inconsistent")
        if int(temperature.get("actual_openmc_seed_count",0)) != len(observed):
            raise ValueError("material audit seed count differs")
        if int(critical.get("independent_seed_count",0)) != len(observed):
            raise ValueError("criticality audit seed count differs")
        if int(heating.get("number_of_independent_seeds",0)) != len(observed):
            raise ValueError("local heating audit seed count differs")

        source_proven = bool(summary["run_statistics"].get(
            "source_convergence_independently_demonstrated",False))
        hot_qualified = bool(temperature.get(
            "hot_operating_temperature_neutron_physics_qualified",False))
        deposited = bool(heating.get(
            "all_real_statepoints_have_usable_neutron_local_heating",False)
            and heating.get("thermal_power_normalization_allowed",False))
        criticality = bool(critical.get(
            "nuclear_reactor_physical_criticality_validated",False))
        if (not source_proven and critical.get(
                "fission_source_stationarity_demonstrated") is True):
            raise ValueError("contradictory claimed neutron source stationarity")
        if hot_qualified and temperature.get(
                "evaluated_neutron_temperature_grid_has_multiple_temperatures") is False:
            raise ValueError("hot qualification contradicts evaluated temperature grid")

        return {
            "audit": "INDEPENDENT_RETAINED_REAL_OPENMC_CORE_ARTIFACT_INTEGRITY",
            "artifact_sha256": zip_hash.hexdigest(),
            "workflow_run_id": workflow_run_id,
            "transport_summary_sha256": expected_hash,
            "evaluated_xs_index_sha256": library_sha,
            "actual_openmc_HDF5_transport": True,
            "actual_executed_core_scope": summary["model_scope"],
            "core_state": "cold_surrogate",
            "cold_input_hydrogen_temperature_k": float(
                summary["hydrogen_temperature_K"]),
            "openmc_version": list(next(iter(versions))),
            "seeds": [item["seed"] for item in observed],
            "independent_seed_count": len(observed),
            "active_generations": batches-inactive,
            "particles_per_generation": particles,
            "statepoints": observed,
            "recomputed_inverse_variance_k_mean": aggregate,
            "source_convergence_demonstrated": source_proven,
            "hot_material_data_qualified": hot_qualified,
            "real_local_heating_available_for_thermal_power": deposited,
            "local_heating_eV_per_source": float(
                heating.get("independent_seed_local_heating_mean_eV_per_source",0)),
            "local_heating_status": heating.get("status"),
            "reported_criticality_condition": critical.get("observed_state"),
            "critical_operating_core_validated": criticality,
            "historical_NERVA_validation": False,
            "qualified_hot_core_power_prediction": bool(
                source_proven and hot_qualified and deposited and criticality),
            "interpretation": (
                "Actual OpenMC 0.x numerical neutron transport evidence was "
                "read directly from retained HDF5 and matched to source and "
                "nuclear-data manifests. Readiness gates are copied from "
                "independent original physical qualification audits and "
                "cannot be promoted by this read-only checker. Real transport "
                "does not establish hot reactor or historical fidelity."
            )
        }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact",type=Path,required=True)
    p.add_argument("--workflow-run-id",type=int)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    report=inspect_real_core_bundle(
        args.artifact,workflow_run_id=args.workflow_run_id)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Genuine OpenMC HDF5 verified:",report["independent_seed_count"],
          "seeds; cold k_mean:",report["recomputed_inverse_variance_k_mean"])
    print("Source stationary:",report["source_convergence_demonstrated"])
    print("Hot nuclear data qualified:",report["hot_material_data_qualified"])
    print("Local nuclear heating qualified:",report["real_local_heating_available_for_thermal_power"])
    print("Historical NERVA reactor benchmark: NOT VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
