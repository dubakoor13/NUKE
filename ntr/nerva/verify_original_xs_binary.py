"""Stage 28: independently re-open REAL neutron isotope HDF5 binaries.

The 2026-10-23 OpenMC neutron transport archive contains SHA256/byte-length
claims for 29 isotope files, not those isotope binaries. This verifier must
run in a genuinely installed OpenMC 0.15.3 environment with the ORIGINAL
cross_sections.xml candidate and REAL isotope HDF5 bytes available.

Hash matching validates *binary identity*, not evaluated library edition,
high-temperature nuclear data, physical reactor criticality or heating.
Never interpolate, synthesize, substitute, or quietly skip neutron data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import openmc.data

from .nuclear_data import _library_entries
from .mt901_numeric import inspect_mt901_numeric_curves
from .real_native_material_xs_audit import audit_native_material_xs
from .real_transport_evidence_gate import _Bundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_original_hash_index(
    records: list[dict], indexed_isotopes: list[str],
    candidate_isotope_paths: dict[str, Path],
) -> dict[str, dict]:
    """Check ORIGINAL manifest against ACTUAL byte content, no fake fallback.

    The independently selected isotope -> path map must come from a
    same-SHA cross_sections.xml, not a user-supplied replacement list.
    """
    if (not isinstance(records, list) or not records
            or not isinstance(indexed_isotopes, list)
            or len(set(indexed_isotopes)) != len(indexed_isotopes)
            or not isinstance(candidate_isotope_paths, dict)):
        raise ValueError("invalid indexed evaluated neutron manifest")
    expected = {}
    for row in records:
        path = Path(row["path"])
        isotope = path.stem
        digest = row.get("sha256")
        byte_count = row.get("bytes")
        if (isotope in expected or isotope not in indexed_isotopes
                or not isinstance(digest, str) or len(digest)!=64
                or any(x not in "0123456789abcdef" for x in digest)
                or type(byte_count) is not int or byte_count <= 0):
            raise ValueError("invalid/duplicate original isotope HDF5 hash declaration")
        expected[isotope] = {"path": path, "sha256": digest, "bytes": byte_count}
    if set(expected) != set(indexed_isotopes):
        raise ValueError("original isotope index has missing or extra HDF5 records")
    if set(candidate_isotope_paths) != set(expected):
        raise ValueError("candidate evaluated isotope library does not match original index")
    matched = {}
    for name in sorted(expected):
        record = expected[name]
        path = Path(candidate_isotope_paths[name]).resolve()
        if not path.is_file():
            raise ValueError("actual candidate evaluated isotope HDF5 bytes missing: " + name)
        if path != record["path"].resolve():
            # Same index SHA and original Docker should resolve the same files.
            raise ValueError("candidate isotope path differs from original manifest: " + name)
        byte_count = path.stat().st_size
        if byte_count != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError("actual evaluated isotope HDF5 SHA256/size mismatch: " + name)
        matched[name] = {"path": str(path), "sha256": record["sha256"],
                         "verified_bytes": byte_count}
    return matched


def _neutron_paths_from_verified_xml(index: Path) -> dict[str, Path]:
    by_isotope = {}
    for entry in _library_entries(index):
        if entry.data_type != "neutron":
            continue
        p = Path(entry.resolved_path).resolve()
        for isotope in entry.materials:
            if isotope in by_isotope:
                raise ValueError("ambiguous evaluated isotope index: " + isotope)
            by_isotope[isotope] = p
    if not by_isotope:
        raise ValueError("no evaluated neutron entries in ORIGINAL XML")
    return by_isotope


def verify_candidate_neutron_data(
    artifact: Path, cross_sections: Path,
) -> dict:
    """Require original 4-seed HDF5 and unmodified real isotope binaries."""
    original = audit_native_material_xs(artifact)
    bundle = _Bundle(Path(artifact))
    try:
        manifest = bundle.report("verified_xs_files.json")
        reported = bundle.report("neutron_temperature_coverage_audit.json")
        index = Path(cross_sections).resolve()
        if not index.is_file():
            raise ValueError("original neutron cross_sections.xml not available")
        sha = _sha256(index)
        if (sha != manifest["cross_sections_sha256"]
                or manifest["cross_sections_sha256"] !=
                    bundle.report("real_core_transport_summary.json")["cross_sections_xml_sha256"]):
            raise ValueError("real candidate cross_sections.xml SHA256 differs from ORIGINAL source")
        candidate = _neutron_paths_from_verified_xml(index)
        index_names = list(manifest["required_neutron_nuclides"])
        if set(candidate) < set(index_names):
            raise ValueError("candidate original XML lacks isotope data")
        candidate = {name: candidate[name] for name in index_names}
        hashes = verify_original_hash_index(
            manifest["neutron_hdf5_files"], index_names, candidate)
        original_grids = {}
        for row in reported["nuclides"]:
            name = row["nuclide"]
            if name in original_grids:
                raise ValueError("duplicate original isotope temperature record")
            original_grids[name] = row["available_temperatures_K"]
        if set(original_grids) != set(hashes):
            raise ValueError("original temperature records do not match isotope bytes index")
        reached = set(original["executed_reachable_neutron_isotopes"])
        if not reached.issubset(hashes):
            raise ValueError("original executed material isotope data incomplete")

        rows = []
        for name, file in sorted(hashes.items()):
            path = Path(file["path"])
            # Do not trust file names or manifest records alone:
            # deserialize each matched isotope's ACTUAL data with OpenMC.
            data = openmc.data.IncidentNeutron.from_hdf5(path)
            if data.name != name:
                raise ValueError("genuine nuclear isotope HDF5 identity differs: " + name)
            kTs = sorted(float(kT / openmc.data.K_BOLTZMANN) for kT in data.kTs)
            exp = sorted(float(x) for x in original_grids[name])
            if (not kTs or len(kTs)!=len(set(kTs))
                    or len(kTs)!=len(exp) or
                    any(not math.isfinite(x) or x<=0 for x in kTs) or
                    not np.allclose(kTs,exp,rtol=0,atol=1e-8)):
                raise ValueError("actual isotope temperature knots differ from original: " + name)
            # Real numerical MT901 coefficients are required to support
            # neutron-only local KERMA. Missing MT901 is OBSERVED, not repaired.
            kerma = inspect_mt901_numeric_curves(data)
            rows.append({
                "isotope": name,
                **file,
                "executed_material_isotope": name in reached,
                "actually_deserialized_from_hdf5": True,
                "actual_evaluated_temperatures_K": kTs,
                "native_MT901_present": kerma["mt901_reaction_present"],
                "native_MT901_numerically_usable":
                    kerma["local_kerma_mt901_numerical_usable"],
                "native_MT901_temperatures_checked": kerma["temperatures_checked"],
            })
        if len(rows)!=len(index_names):
            raise ValueError("original evaluated isotope inventory incomplete")
        reached_rows = [x for x in rows if x["executed_material_isotope"]]
        # A negative result is a scientifically valid outcome.
        all_reached_kerma = all(x["native_MT901_numerically_usable"]
                                for x in reached_rows)
        return {
            "schema": "nuke-original-evaluated-isotope-binary-stage28-v1",
            "source_evidence": "ACTUAL_ORIGINAL_FOUR_SEED_OPENMC_HDF5_AND_REAL_EVALUATED_ISOTOPE_BYTES",
            "original_evidence_openmc_seed_count": original["actual_openmc_seeds"],
            "original_cross_sections_xml_sha256": sha,
            "real_isotope_HDF5_hashes_and_byte_sizes_independently_verified": True,
            "real_neutron_data_deserialized_by_openmc": True,
            "original_manifest_matched_isotope_count": len(rows),
            "physically_root_connected_isotope_count": len(reached),
            "candidate_reachable_MT901_numeric_presence":
                bool(all_reached_kerma),
            "actual_transport_evaluated_library_edition_verified_externally": False,
            "original_HDF5_isotope_binaries_retained_in_original_archive": False,
            "genuine_independent_source_stationarity_qualified": False,
            "genuine_hot_material_temperature_coverage_qualified": False,
            "genuine_material_resolved_heating_local_qualified": False,
            "hot_core_power_normalization_allowed": False,
            "historical_nerva_physics_qualified": False,
            "isotopes": rows,
            "limitations": [
                "Matching true source-run isotope SHA identifies re-retrieved binary bytes, not evaluated-data release provenance.",
                "Source archive still does not include isotope files; rerunning requires a pinned accessible data container.",
                "Physical MT901 local heating qualification ALSO requires nonzero executed spatial energy tallies.",
                "One cold neutron temperature is not a NERVA hot-core temperature library.",
                "Fission source stationarity, photon deposition and NASA operating-point measurements remain unqualified.",
            ],
        }
    finally:
        bundle.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original_artifact", type=Path)
    p.add_argument("--cross-sections", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = verify_candidate_neutron_data(a.original_artifact, a.cross_sections)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("REAL evaluated neutron files matched:",result["original_manifest_matched_isotope_count"])
    print("Actual root-used isotope HDF5:",result["physically_root_connected_isotope_count"])
    print("All actual evaluated reachables MT901 numerical present:",
          result["candidate_reachable_MT901_numeric_presence"])
    print("Qualified hot NERVA neutron/photon heating: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
