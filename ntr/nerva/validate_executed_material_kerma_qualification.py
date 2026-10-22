"""Synthetic provenance/tampering tests for executed-material nuclear KERMA QA.

The fixture files are NOT neutron transport or evaluated nuclear data.
Only the separate Docker/OpenMC workflow establishes real reactor evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from .executed_material_kerma_qualification import _sha, evaluate_executed_core_mt901


def reject(fn, expected: str) -> None:
    try:
        fn()
    except ValueError as err:
        assert expected in str(err), str(err)
    else:
        raise AssertionError("invalid real-core KERMA evidence accepted: " + expected)


def main() -> int:
    with TemporaryDirectory() as td:
        root = Path(td)
        index_hash = "1" * 64
        paths = []
        for seed in (13007, 23011, 33013, 43019):
            directory = root / str(seed)
            directory.mkdir()
            statepoint = directory / "statepoint.120.h5"
            statepoint.write_bytes(f"synthetic-statepoint-{seed}".encode())
            xml = ET.Element("materials")
            for mid, name, nuclides in (
                (19, "synthetic fuel", ("U235", "C0")),
                (20, "synthetic hydrogen", ("H1",)),
            ):
                mat = ET.SubElement(xml, "material", {"id": str(mid), "name": name})
                for iso in nuclides:
                    ET.SubElement(mat, "nuclide", {"name": iso, "ao": "1.0"})
            ET.ElementTree(xml).write(directory / "materials.xml", encoding="utf-8")
            paths.append({"seed": seed, "statepoint_path": str(statepoint),
                          "statepoint_sha256": _sha(statepoint)})
        summary_file = root / "summary.json"
        inventory_file = root / "inventory.json"
        summary = {
            "execution": "ACTUAL_OPENMC_TRANSPORT", "assembly": "core",
            "cross_sections_xml_sha256": index_hash,
            "runs": paths,
        }
        inventory = {
            "mode": "ACTUAL_EVALUATED_NEUTRON_CROSS_SECTION_HDF5_INSPECTION",
            "cross_sections_xml_sha256": index_hash,
            "nuclides": [
                {"nuclide": n, "local_kerma_mt901_present": True,
                 "local_kerma_mt901_numerical_usable": True,
                 "physical_neutron_hdf5_sha256": str(i + 2) * 64,
                 "library_load_error": None}
                for i, n in enumerate(("U235", "C0", "H1"))
            ],
        }
        def save():
            summary_file.write_text(json.dumps(summary), encoding="utf-8")
            inventory_file.write_text(json.dumps(inventory), encoding="utf-8")
        def audit():
            save()
            return evaluate_executed_core_mt901(summary_file, inventory_file)
        covered = audit()
        assert covered["real_independent_seed_count"] == 4
        assert covered["physical_material_count"] == 2
        assert covered["required_executed_neutron_nuclide_count"] == 3
        assert covered["all_executed_materials_have_numerically_usable_mt901"]
        assert not covered["thermal_power_normalization_allowed"]
        assert not covered["nasa_nerva_core_energy_deposition_validated"]

        inventory["nuclides"][0]["local_kerma_mt901_present"] = False
        inventory["nuclides"][0]["local_kerma_mt901_numerical_usable"] = False
        absent = audit()
        assert absent["material_regions_without_complete_evaluated_mt901"] == [
            "synthetic fuel"
        ]
        assert absent["physical_materials"][0][
            "nuclides_missing_verified_numeric_mt901"
        ] == ["U235"]
        assert not absent["all_executed_materials_have_numerically_usable_mt901"]

        # Fake-positive coverage without a physical source HDF5 hash.
        inventory["nuclides"][0]["local_kerma_mt901_present"] = True
        inventory["nuclides"][0]["local_kerma_mt901_numerical_usable"] = True
        inventory["nuclides"][0]["physical_neutron_hdf5_sha256"] = None
        reject(audit, "unproven evaluated HDF5")
        inventory["nuclides"][0]["physical_neutron_hdf5_sha256"] = "2" * 64

        # Missing one evaluated isotope may not be hidden by another material.
        removed = inventory["nuclides"].pop()
        missing = audit()
        assert missing["required_executed_neutron_nuclides_missing_index"] == ["H1"]
        assert missing["material_regions_without_complete_evaluated_mt901"] == [
            "synthetic hydrogen"
        ]
        inventory["nuclides"].append(removed)

        inventory["cross_sections_xml_sha256"] = "0" * 64
        reject(audit, "cross-section library identity mismatch")
        inventory["cross_sections_xml_sha256"] = index_hash
        inventory["nuclides"].append(inventory["nuclides"][0].copy())
        reject(audit, "duplicate")
        inventory["nuclides"].pop()

        # Changed material during one seed is not a comparable source ensemble.
        xml_path = Path(paths[1]["statepoint_path"]).parent / "materials.xml"
        tree = ET.parse(xml_path)
        tree.getroot().findall("material")[1].set("name", "different material")
        tree.write(xml_path)
        reject(audit, "changed across seeds")
        tree.getroot().findall("material")[1].set("name", "synthetic hydrogen")
        tree.write(xml_path)

        summary["runs"][3]["seed"] = paths[0]["seed"]
        reject(audit, "duplicate real core RNG seed")
        summary["runs"][3]["seed"] = 43019
        state = Path(paths[0]["statepoint_path"])
        state.write_bytes(b"tampered source")
        reject(audit, "statepoint SHA256 provenance mismatch")

    print("Executed-material evaluated MT901 physical evidence regressions: PASS")
    print("  synthetic provenance fixtures only; real OpenMC CI separately required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
