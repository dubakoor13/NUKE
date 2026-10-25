"""Audit ACTUALLY EXECUTED OpenMC geometry/material and cold neutron-data evidence.

Reads original four-seed OpenMC summary.h5 native material/cell datasets,
materials.xml, qualified temperature reports and isotope HDF5 hash manifest.
Does not mistake declared (but unplaced) elements for executed physics.
Does not pretend listed isotope HDF5 SHA strings have been verified against
actual isotope bytes: those evaluated libraries are NOT in this archive.

Read-only and observational: no reactor design, fuel, criticality, XS
temperature extrapolation, source-rate normalization or hot NERVA claims.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .real_transport_evidence_gate import _Bundle, audit_real_transport_evidence, sha256_bytes


def _decode(value):
    if not isinstance(value, (bytes, np.bytes_)):
        raise ValueError("native OpenMC string must be HDF5 bytes")
    return bytes(value).decode("utf-8").rstrip("\x00")


def _xml_materials(blob):
    root = ET.fromstring(blob)
    if root.tag != "materials":
        raise ValueError("expected original executed materials.xml")
    found = {}
    for element in root.findall("material"):
        ident = int(element.attrib["id"])
        nuclides = [item.attrib["name"] for item in element.findall("nuclide")]
        density = element.find("density")
        if (ident in found or ident < 1 or not nuclides or
            len(set(nuclides)) != len(nuclides) or density is None or
            float(density.attrib["value"]) <= 0):
            raise ValueError("duplicate/invalid executed nuclear material")
        temperature = (float(element.attrib["temperature"])
                       if "temperature" in element.attrib else None)
        if temperature is not None and (not math.isfinite(temperature) or
                                        temperature <= 0):
            raise ValueError("invalid explicitly assigned nuclear temperature")
        found[ident] = {
            "id": ident,
            "name": element.attrib["name"],
            "nuclides": nuclides,
            "explicit_temperature_K": temperature,
            "sab": sorted(item.attrib["name"] for item in element.findall("sab")),
        }
    if not found:
        raise ValueError("no executed nuclear materials")
    return found


def _summary_materials_and_cells(blob, xml):
    cells = []
    native = {}
    with h5py.File(io.BytesIO(blob), "r") as h5:
        for key, g in h5["materials"].items():
            if not key.startswith("material "):
                continue
            mid = int(key.split()[-1])
            isotopes = [_decode(x) for x in g["nuclides"][()]]
            populations = np.asarray(g["nuclide_densities"][()], float)
            n = float(g["atom_density"][()])
            if (mid in native or mid not in xml or
                _decode(g["name"][()]) != xml[mid]["name"] or
                isotopes != xml[mid]["nuclides"] or
                populations.shape != (len(isotopes),) or
                not np.isfinite(populations).all() or
                np.any(populations <= 0) or
                not math.isclose(populations.sum(), n, rel_tol=1e-8, abs_tol=1e-11)):
                raise ValueError("native summary HDF5 isotope or density differs from XML")
            native[mid] = True
        if set(native) != set(xml):
            raise ValueError("native declared nuclear material IDs differ from XML")
        for key, g in h5["geometry/cells"].items():
            if not key.startswith("cell ") or _decode(g["fill_type"][()]) != "material":
                continue
            x = g["material"][()]
            material_id = (int(bytes(x).decode().split()[-1])
                           if isinstance(x, (bytes, np.bytes_)) else int(x))
            if material_id not in native:
                raise ValueError("executed material-filled cell has undeclared nuclear ID")
            temperature = np.asarray(g["temperature"][()], float)
            if (temperature.size == 0 or not np.isfinite(temperature).all() or
                np.any(temperature <= 0)):
                raise ValueError("invalid native executed cell temperature")
            cells.append((material_id, float(temperature.mean())))
    if not cells:
        raise ValueError("no native executed material-filled core cells")
    return sorted({mid for mid, _ in cells}), cells


def audit_native_material_xs(artifact):
    """Require original HDF5/XML consistency; report NOT-hot-data qualification."""
    validated = audit_real_transport_evidence(artifact)
    source = _Bundle(Path(artifact))
    try:
        xs = source.report("verified_xs_files.json")
        nuclear = source.report("neutron_temperature_coverage_audit.json")
        temp = source.report("core_material_temperature_qualification.json")
        if (xs["cross_sections_sha256"] != validated["cross_sections_xml_sha256"]
            or xs.get("nuclear_data_release_edition_verified_externally") is not False
            or xs.get("all_required_neutron_hdf5_loaded") is not True
            or temp.get("hot_operating_temperature_neutron_physics_qualified") is not False):
            raise ValueError("nuclear-data/temperature provenance inconsistent")
        indexed = {}
        for row in xs["neutron_hdf5_files"]:
            name = Path(row["path"]).stem
            digest = row.get("sha256")
            if (name in indexed or not isinstance(digest, str) or len(digest) != 64
                or any(letter not in "0123456789abcdef" for letter in digest)
                or not isinstance(row.get("bytes"), int) or row["bytes"] <= 0):
                raise ValueError("invalid or duplicated isotope HDF5 manifest entry")
            indexed[name] = row
        if set(indexed) != set(xs["required_neutron_nuclides"]):
            raise ValueError("requested isotope set mismatches evaluated-data manifest")
        grids = {}
        for row in nuclear["nuclides"]:
            name = row["nuclide"]
            values = row["available_temperatures_K"]
            if (name in grids or name not in indexed or not values
                or any(not math.isfinite(float(x)) or float(x) <= 0 for x in values)):
                raise ValueError("invalid temperature coverage for isotope")
            grids[name] = tuple(float(x) for x in values)
        if set(grids) != set(indexed):
            raise ValueError("missing isotope temperature coverage")
        qualified = {m["material_id"]: m for m in temp["physical_materials"]}
        observations = []
        for seed in [row["seed"] for row in validated["statepoints"]]:
            xml_bytes = source.read(f"seed_{seed}/materials.xml")
            summary_bytes = source.read(f"seed_{seed}/summary.h5")
            xml = _xml_materials(xml_bytes)
            used, cells = _summary_materials_and_cells(summary_bytes, xml)
            if (used != sorted(qualified) or
                used != sorted(temp["actual_reachable_material_ids"])):
                raise ValueError("native executed material reachability differs from audit")
            for mid in used:
                record = xml[mid]
                ref = qualified[mid]
                if (record["name"] != ref["material_name"] or
                    record["nuclides"] != ref["nuclides"] or
                    record["explicit_temperature_K"] !=
                        ref["explicit_material_temperature_K"] or
                    record["sab"] != ref["bound_thermal_scattering_tables"] or
                    not set(record["nuclides"]).issubset(indexed)):
                    raise ValueError("native HDF5 material, evaluated XS or phase mismatched")
            for mid, t in cells:
                explicit = xml[mid]["explicit_temperature_K"]
                if (explicit is not None and
                    not math.isclose(t, explicit, rel_tol=0, abs_tol=1e-5)):
                    raise ValueError("native cell temperature contradicts input XML")
            scattering = source.report(
                f"seed_{seed}/thermal_scattering_physics_audit.json")
            if (scattering.get("treatment") != "free-gas" or
                scattering.get("ready") is not False or
                scattering.get("physically_complete_bound_thermal_scattering")
                    is not False):
                raise ValueError("bound thermal-scattering evidence unexpectedly changed")
            observations.append({
                "seed": seed,
                "executed_summary_hdf5_sha256": sha256_bytes(summary_bytes),
                "executed_material_xml_sha256": sha256_bytes(xml_bytes),
                "material_filled_cells": len(cells),
                "reachable_material_ids": used,
                "explicit_temperature_by_material_K": {
                    str(mid): xml[mid]["explicit_temperature_K"] for mid in used
                },
                "native_minimum_cell_temperature_K": min(t for _, t in cells),
                "native_maximum_cell_temperature_K": max(t for _, t in cells),
            })
        xml = _xml_materials(source.read("seed_13007/materials.xml"))
        reached = set(observations[0]["reachable_material_ids"])
        used_isotopes = set().union(
            *(set(x["nuclides"]) for mid, x in xml.items() if mid in reached))
        temperature_grids = sorted({v for name in used_isotopes for v in grids[name]})
        explicit_count = sum(xml[mid]["explicit_temperature_K"] is not None
                             for mid in reached)
        if (len(xml) != 9 or len(reached) != 4 or len(used_isotopes) != 13
            or explicit_count != 1 or len(temperature_grids) != 1):
            raise ValueError("archived actual cold-material inventory unexpectedly changed")
        return {
            "schema": "nuke-native-material-xs-stage25-v1",
            "evidence_kind": "ACTUAL_OPENMC_EXECUTED_SUMMARY_HDF5_AND_XS_MANIFEST",
            "actual_openmc_seeds": len(observations),
            "declared_material_count": len(xml),
            "executed_reachable_material_count": len(reached),
            "unused_material_ids": sorted(set(xml)-reached),
            "executed_reachable_neutron_isotopes": sorted(used_isotopes),
            "executed_reachable_neutron_isotope_count": len(used_isotopes),
            "declared_neutron_isotope_file_count": len(indexed),
            "available_executed_neutron_temperatures_K": temperature_grids,
            "executed_materials_with_explicit_temperature": explicit_count,
            "executed_materials_using_implicit_temperature": len(reached)-explicit_count,
            "claimed_evaluated_XS_file_hashes_independently_reverified": False,
            "actual_isotope_HDF5_binary_bytes_in_source_archive": False,
            "nuclear_data_release_edition_verified": False,
            "qualified_bound_scattering_and_hot_materials": False,
            "historical_hot_nerva_qualification": False,
            "per_seed_native_evidence": observations,
            "physical_blockers": [
                "Only one cold evaluated nuclide temperature, not a hot XS temperature table",
                "3 of 4 physically reached core materials have no explicit exported temperature",
                "No qualified bound solid S(alpha,beta) thermal scattering in executed case",
                "Referenced isotope HDF5 files are not actually present in archived binary evidence",
                "Evaluated nuclear data release edition not independently identified",
                "Transport source stationarity and historical NERVA core fidelity are unqualified",
            ],
        }
    finally:
        source.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("artifact", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = audit_native_material_xs(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("Executed materials/XML provenance: VERIFIED",
          result["actual_openmc_seeds"], "seeds")
    print("Reachable material count:", result["executed_reachable_material_count"])
    print("Evaluated neutron temperatures:", result["available_executed_neutron_temperatures_K"])
    print("Physical hot-core material/Xs qualification: BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
