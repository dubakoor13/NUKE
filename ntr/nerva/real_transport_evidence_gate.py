"""Read-only, fail-closed qualification of archived ACTUAL OpenMC core statepoints.

Audits the *original binary* four-seed OpenMC HDF5 statepoints, XML model
identity, scientific qualification JSON, and SHA256 provenance. This is an
observational gate, NEVER a core design/criticality tuning procedure.

A successful evidence-integrity audit can legitimately report ALL physical
qualification gates BLOCKED. In particular, a cold subcritical surrogate
with unconverged fission source and zero usable local heating MUST NOT be
promoted to a physically validated hot NERVA core.

Inputs are a ZIP downloaded from a real OpenMC GitHub workflow, or the
directory produced by unpacking that archive. There is NO synthetic data
fallback, and no neutron-transport simulation is performed in this module.
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

_REQUIRED_REPORTS = (
    "real_core_transport_summary.json",
    "core_criticality_physical_qualification.json",
    "core_material_temperature_qualification.json",
    "heating_data_qualification.json",
    "source_resolution_audit.json",
    "real_core_keff_history_qualification.json",
    "verified_xs_files.json",
)
_SHA_LEN = 64


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class _Bundle:
    def __init__(self, root: Path):
        self.root = Path(root)
        if self.root.is_file():
            self.zipfile = zipfile.ZipFile(self.root)
            names = self.zipfile.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate ZIP entry names")
            self.names = frozenset(names)
            self.archive_sha256 = sha256_bytes(self.root.read_bytes())
        elif self.root.is_dir():
            self.zipfile = None
            self.names = None
            self.archive_sha256 = None
        else:
            raise FileNotFoundError(root)

    def read(self, path: str) -> bytes:
        if (not path or path.startswith("/") or
            any(part in ("", ".", "..") for part in path.split("/"))):
            raise ValueError("unsafe evidence path")
        if self.zipfile is not None:
            if path not in self.names:
                raise ValueError(f"missing REAL OpenMC evidence: {path}")
            return self.zipfile.read(path)
        candidate = self.root.joinpath(*path.split("/"))
        if not candidate.is_file() or not candidate.resolve().is_relative_to(
            self.root.resolve()
        ):
            raise ValueError(f"missing REAL OpenMC evidence: {path}")
        return candidate.read_bytes()

    def report(self, path: str) -> dict:
        try:
            record = json.loads(self.read(path))
        except (UnicodeError, json.JSONDecodeError) as err:
            raise ValueError("malformed evidence JSON: " + path) from err
        if not isinstance(record, dict):
            raise ValueError("evidence JSON is not an object: " + path)
        return record

    def close(self):
        if self.zipfile is not None:
            self.zipfile.close()


def _match_sha(actual: str, expected: str, context: str) -> None:
    if (not isinstance(expected, str) or len(expected) != _SHA_LEN or
        any(c not in "0123456789abcdef" for c in expected) or
        actual != expected):
        raise ValueError("SHA256 evidence mismatch: " + context)


def _h5_binary(seed: int, blob: bytes, record: dict, summary: dict) -> dict:
    _match_sha(sha256_bytes(blob), record["statepoint_sha256"],
               f"seed {seed} statepoint")
    with h5py.File(io.BytesIO(blob), "r") as statepoint:
        filetype = statepoint.attrs.get("filetype")
        if bytes(filetype).rstrip(b"\x00") != b"statepoint":
            raise ValueError("not a real OpenMC statepoint")
        version = tuple(int(n) for n in statepoint.attrs["openmc_version"])
        if version != (0, 15, 3):
            raise ValueError("unsupported actual OpenMC statepoint version")
        if (int(statepoint.attrs.get("source_present", 0)) != 1 or
            int(statepoint.attrs.get("photon_transport", -1)) != 0 or
            bytes(statepoint["run_mode"][()]).rstrip(b"\x00") != b"eigenvalue"):
            raise ValueError("missing actual neutron eigenvalue-source evidence")
        values = {
            "seed": int(statepoint["seed"][()]),
            "batch": int(statepoint["current_batch"][()]),
            "inactive": int(statepoint["n_inactive"][()]),
            "particles": int(statepoint["n_particles"][()]),
            "realizations": int(statepoint["n_realizations"][()]),
        }
        kpair = np.asarray(statepoint["k_combined"][()], dtype=float)
    if (values["seed"] != seed or values["batch"] != summary["batches"] or
        values["inactive"] != summary["inactive_batches"] or
        values["particles"] != summary["particles_per_generation"] or
        values["realizations"] != values["batch"]-values["inactive"] or
        values["realizations"] < 2 or
        kpair.shape != (2,) or not np.isfinite(kpair).all() or
        kpair[0] <= 0 or kpair[1] <= 0):
        raise ValueError("binary neutron statepoint settings or estimator invalid")
    for measured, declared in (
        (float(kpair[0]), float(record["keff_mean"])),
        (float(kpair[1]), float(record["keff_std_dev"])),
    ):
        if not math.isclose(measured, declared, rel_tol=1e-11, abs_tol=1e-14):
            raise ValueError("reported k differs from executed binary statepoint")
    return {
        "seed": seed,
        "statepoint_sha256": sha256_bytes(blob),
        "k_effective": float(kpair[0]),
        "reported_openmc_sem": float(kpair[1]),
        "batch": values["batch"],
        "inactive_batches": values["inactive"],
        "particles_per_generation": values["particles"],
        "active_realizations": values["realizations"],
        "openmc_version": list(version),
    }


def audit_real_transport_evidence(path: str | Path,
                                   *, expected_archive_sha256: str | None = None) -> dict:
    """Require genuine 4-seed binary/OpenMC evidence; report physical blockers."""
    evidence = _Bundle(Path(path))
    try:
        if expected_archive_sha256 is not None:
            if evidence.archive_sha256 is None:
                raise ValueError("ZIP SHA256 required; directory has no ZIP digest")
            _match_sha(evidence.archive_sha256, expected_archive_sha256,
                       "archive from published real transport run")
        reports = {key: evidence.report(key) for key in _REQUIRED_REPORTS}
        summary = reports["real_core_transport_summary.json"]
        critical = reports["core_criticality_physical_qualification.json"]
        temperature = reports["core_material_temperature_qualification.json"]
        heating = reports["heating_data_qualification.json"]
        source = reports["source_resolution_audit.json"]
        history = reports["real_core_keff_history_qualification.json"]
        xs = reports["verified_xs_files.json"]
        if (summary.get("execution") != "ACTUAL_OPENMC_TRANSPORT" or
            summary.get("assembly") != "core" or
            summary.get("cold_xs_smoke_test_only") is not True or
            not isinstance(summary.get("runs"), list) or len(summary["runs"]) < 4):
            raise ValueError("not the actual cold core independent-seed transport set")
        summary_sha = sha256_bytes(evidence.read("real_core_transport_summary.json"))
        for report,name in (
            (critical, "transport_summary_sha256"),
            (temperature, "real_transport_summary_sha256"),
            (heating, "transport_summary_sha256"),
            (source, "real_transport_summary_sha256"),
            (history, "real_transport_summary_sha256"),
        ):
            _match_sha(summary_sha, report.get(name), name)
        cross_sha = summary.get("cross_sections_xml_sha256")
        _match_sha(cross_sha, xs.get("cross_sections_sha256"),
                   "evaluated neutron nuclear-data index identity")
        for report in (critical, temperature, source):
            _match_sha(cross_sha, report.get("cross_sections_xml_sha256"),
                       "cross-section manifest across physics audits")
        if (xs.get("all_required_neutron_hdf5_loaded") is not True or
            xs.get("nuclear_data_release_edition_verified_externally") is not False or
            critical.get("actual_neutron_transport_execution") is not True):
            raise ValueError("not a genuine, fully identified cold neutron-data run")
        rows = []
        seen = set()
        for record in summary["runs"]:
            seed = record.get("seed")
            if type(seed) is not int or seed <= 0 or seed in seen:
                raise ValueError("duplicate or invalid independent OpenMC RNG seed")
            seen.add(seed)
            file = f"seed_{seed}/statepoint.{summary['batches']:03d}.h5"
            rows.append(_h5_binary(seed, evidence.read(file), record, summary))
        xml_fields = ("geometry.xml", "materials.xml", "tallies.xml")
        xml_sha = {}
        for field in xml_fields:
            hashes = {
                sha256_bytes(evidence.read(f"seed_{row['seed']}/{field}"))
                for row in rows
            }
            if len(hashes) != 1:
                raise ValueError("executed geometry/material/tally differs among seeds")
            xml_sha[field] = next(iter(hashes))
        k = np.asarray([row["k_effective"] for row in rows])
        std = np.asarray([row["reported_openmc_sem"] for row in rows])
        w = 1.0 / np.square(std)
        mean = float(np.sum(w*k)/np.sum(w))
        reported = summary.get("run_statistics", {})
        if (reported.get("independent_seed_count") != len(rows) or
            set(reported.get("seeds", [])) != seen or
            not math.isclose(mean, float(reported["inverse_variance_mean_keff"]),
                             rel_tol=1e-10, abs_tol=1e-14)):
            raise ValueError("published ensemble mean contradicts HDF5 eigenvalues")
        # These booleans are evidence-derived findings, not automatic passes.
        converged = (
            reported.get("source_convergence_independently_demonstrated") is True
            and source.get("source_stationarity_validated") is True
            and history.get("spatial_fission_source_stationarity_validated") is True
        )
        hot = temperature.get("hot_operating_temperature_neutron_physics_qualified") is True
        heat = (
            heating.get("all_real_statepoints_have_usable_neutron_local_heating") is True
            and heating.get("thermal_power_normalization_allowed") is True
        )
        criticality = critical.get("nuclear_reactor_physical_criticality_validated") is True
        historical = (
            reported.get("historical_nasa_nerva_benchmark_validated") is True
            and critical.get("historical_nerva_model_fidelity_validated") is True
        )
        gates = {
            "independent_source_stationarity": converged,
            "hot_material_neutron_data": hot,
            "local_energy_deposition_and_normalization": heat,
            "physical_criticality": criticality,
            "independent_historical_benchmark": historical,
        }
        report = {
            "schema": "nuke-real-core-evidence-v1",
            "evidence_kind": "ACTUAL_EXECUTED_OPENMC_COLD_NEUTRON_TRANSPORT",
            "archive_sha256": evidence.archive_sha256,
            "transport_summary_sha256": summary_sha,
            "cross_sections_xml_sha256": cross_sha,
            "independent_seed_count": len(rows),
            "statepoints": rows,
            "same_executed_xml_sha256": xml_sha,
            "recomputed_inverse_variance_k_mean": mean,
            "reported_openmc_formal_sem": float(1/np.sqrt(np.sum(w))),
            "combined_error_is_formal_only": True,
            "physical_gates": gates,
            "reactor_core_historical_or_hot_state_qualified": all(gates.values()),
            "criticality_observation": critical.get("observed_state"),
            "local_heating_observation": heating.get("status"),
            "temperature_observation": {
                "evaluated_K": temperature.get("evaluated_neutron_temperature_values_K"),
                "root_reachable_materials": temperature.get("actual_reachable_material_count"),
                "explicitly_temperature_assigned": temperature.get(
                    "number_with_explicit_exported_temperature"),
            },
            "limitations": [
                "Reported OpenMC k standard errors do not establish fission-source independence.",
                "Cold subcritical surrogate is not a validated historical NERVA operating core.",
                "Neutron-only statepoints and zero heating-local are not an accepted hot-core heating normalization.",
                "Archive hashes establish data identity, not correctness of historical geometry or nuclear data.",
            ],
        }
        return report
    finally:
        evidence.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path, help="Archived real OpenMC ZIP or extracted directory")
    parser.add_argument("--expected-archive-sha256", help="Verified GitHub Actions ZIP digest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-physical-qualification", action="store_true",
                        help="Return code 3 if any physical qualification gate fails")
    args = parser.parse_args()
    result = audit_real_transport_evidence(
        args.evidence, expected_archive_sha256=args.expected_archive_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                           encoding="utf-8")
    print("ACTUAL statepoints integrity:", len(result["statepoints"]), "verified")
    print("Observed mean k:", result["recomputed_inverse_variance_k_mean"])
    for name, passed in result["physical_gates"].items():
        print(name, "QUALIFIED" if passed else "BLOCKED")
    if args.require_physical_qualification and not result[
        "reactor_core_historical_or_hot_state_qualified"
    ]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
