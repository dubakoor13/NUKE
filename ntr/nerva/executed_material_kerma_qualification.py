"""Trace genuine evaluated MT901 local KERMA coverage to EXECUTED OpenMC materials.

Read actual four-seed eigenvalue statepoints, exported materials.xml AND
geometry.xml including root-universe and hex-lattice reachability, and an
independently HDF5-inspected evaluated-neutron MT901 inventory. Numerical
MT901 absence is a physical *data-coverage blocker*, never zero physical heat.

No reactor parameter changes, fabricated KERMA, energy substitution, thermal
normalization, criticality optimization or historical qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _real_materials(path: Path) -> list[dict]:
    root = ET.parse(path).getroot()
    if root.tag != "materials":
        raise ValueError("not executed OpenMC materials.xml")
    materials = []
    ids, names = set(), set()
    for node in root.findall("material"):
        mid = int(node.attrib["id"])
        name = node.attrib.get("name", "")
        nuclides = [x.attrib.get("name") for x in node.findall("nuclide")]
        if (mid <= 0 or not name or mid in ids or name in names
                or not nuclides or any(not n for n in nuclides)
                or len(set(nuclides)) != len(nuclides)):
            raise ValueError("invalid or duplicate executed core material/nuclide")
        ids.add(mid)
        names.add(name)
        materials.append({"material_id": mid, "material_name": name,
                          "nuclides": sorted(nuclides)})
    if len(materials) < 2:
        raise ValueError("too few executed physical materials")
    return sorted(materials, key=lambda x: x["material_id"])



def _real_reachable_material_ids(geometry_xml: Path, defined_ids: set[int]) -> tuple[set[int], dict[int, list[str]]]:
    """Trace actual OpenMC root universe through lattice/filled-universe graph.

    Merely listing a material in materials.xml does NOT put it in a reactor:
    material-filled cells must be reachable from the geometric root. Count
    geometric *placements*, not all defined nuclear material resources.
    An unsupported graph is rejected, never silently counted as physical.
    """
    root = ET.parse(geometry_xml).getroot()
    if root.tag != "geometry":
        raise ValueError("not executed OpenMC geometry.xml")
    cells_by_universe: dict[int, list[ET.Element]] = {}
    for c in root.findall("cell"):
        u = int(c.attrib["universe"])
        if u <= 0:
            raise ValueError("nonphysical OpenMC universe ID")
        cells_by_universe.setdefault(u, []).append(c)
    if not cells_by_universe:
        raise ValueError("no executed OpenMC geometry cells")

    lattices: dict[int, set[int]] = {}
    referenced_universes: set[int] = set()
    for lattice in root:
        if lattice.tag not in ("hex_lattice", "lattice"):
            continue
        lid = int(lattice.attrib["id"])
        if lid in lattices:
            raise ValueError("duplicate OpenMC lattice")
        universes = lattice.findtext("universes")
        if not universes or not universes.strip():
            raise ValueError("empty lattice universe placement")
        members = {int(x) for x in universes.split()}
        outside = lattice.findtext("outer")
        if outside is not None:
            members.add(int(outside.strip()))
        if any(x <= 0 for x in members):
            raise ValueError("invalid hex-lattice universe placement")
        lattices[lid] = members
        referenced_universes.update(members)
    for cells in cells_by_universe.values():
        for c in cells:
            if "fill" in c.attrib:
                fill = int(c.attrib["fill"])
                if fill not in lattices:
                    referenced_universes.add(fill)
    roots = set(cells_by_universe) - referenced_universes
    if len(roots) != 1:
        raise ValueError("cannot uniquely identify physical root OpenMC universe")

    reachable: set[int] = set()
    active_mids: set[int] = set()
    named_cells: dict[int, list[str]] = {}
    pending = list(roots)
    while pending:
        universe = pending.pop()
        if universe in reachable:
            continue
        if universe not in cells_by_universe:
            raise ValueError("reachable OpenMC geometry universe undefined")
        reachable.add(universe)
        for c in cells_by_universe[universe]:
            has_material = "material" in c.attrib
            has_fill = "fill" in c.attrib
            if has_material and has_fill:
                raise ValueError("OpenMC cell has both material and universe fill")
            if has_material:
                value = c.attrib["material"]
                if value == "void":
                    continue
                ids = [int(x) for x in value.split()]
                if not ids or any(mid <= 0 or mid not in defined_ids for mid in ids):
                    raise ValueError("executed geometry references missing physical material")
                for mid in ids:
                    active_mids.add(mid)
                    named_cells.setdefault(mid, []).append(c.attrib.get("name", ""))
            if has_fill:
                fill = int(c.attrib["fill"])
                if fill in lattices:
                    pending.extend(lattices[fill])
                else:
                    pending.append(fill)
    if not active_mids:
        raise ValueError("actual OpenMC root contains no physically instantiated materials")
    return active_mids, {mid: sorted(set(names)) for mid,names in named_cells.items()}


def evaluate_executed_core_mt901(summary_file: Path, inventory_file: Path) -> dict:
    summary_file = Path(summary_file)
    inventory_file = Path(inventory_file)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    inventory = json.loads(inventory_file.read_text(encoding="utf-8"))
    if summary.get("execution") != "ACTUAL_OPENMC_TRANSPORT" or summary.get("assembly") != "core":
        raise ValueError("requires genuine OpenMC core eigenvalue summary")
    if inventory.get("mode") != "ACTUAL_EVALUATED_NEUTRON_CROSS_SECTION_HDF5_INSPECTION":
        raise ValueError("requires actual evaluated neutron HDF5 MT901 inventory")
    expected_sha = summary.get("cross_sections_xml_sha256")
    if (not isinstance(expected_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
            or expected_sha != inventory.get("cross_sections_xml_sha256")):
        raise ValueError("evaluated neutron cross-section library identity mismatch")

    rows = inventory.get("nuclides")
    if not isinstance(rows, list) or not rows:
        raise ValueError("missing actual evaluated isotope inventory")
    index = {}
    for row in rows:
        nuclide = row.get("nuclide")
        if not isinstance(nuclide, str) or not nuclide or nuclide in index:
            raise ValueError("duplicate or missing evaluated neutron nuclide")
        present = row.get("local_kerma_mt901_present")
        usable = row.get("local_kerma_mt901_numerical_usable")
        if present not in (None, True, False) or usable not in (None, True, False):
            raise ValueError("invalid evaluated MT901 reaction availability")
        identity = row.get("physical_neutron_hdf5_sha256")
        if identity is not None and (
            not isinstance(identity, str) or not re.fullmatch(r"[0-9a-f]{64}", identity)
        ):
            raise ValueError("invalid actual evaluated neutron HDF5 digest")
        if (present is True or usable is True) and (
            identity is None or row.get("library_load_error") is not None
        ):
            raise ValueError("unproven evaluated HDF5 reaction availability")
        index[nuclide] = row

    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) < 4:
        raise ValueError("requires >=4 separately executed OpenMC seeds")
    seen_seeds = set()
    executed = None
    declared = None
    reachable_ids = None
    physical_xmls = []
    for run in runs:
        seed = run.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed <= 0 or seed in seen_seeds:
            raise ValueError("missing or duplicate real core RNG seed")
        seen_seeds.add(seed)
        statepoint = Path(run["statepoint_path"])
        if (not statepoint.is_file()
                or _sha(statepoint) != run.get("statepoint_sha256")):
            raise ValueError("real core OpenMC statepoint SHA256 provenance mismatch")
        xml = statepoint.parent / "materials.xml"
        if not xml.is_file():
            raise ValueError("missing executed OpenMC materials XML")
        geometry = statepoint.parent / "geometry.xml"
        if not geometry.is_file():
            raise ValueError("missing executed OpenMC geometry XML")
        materials = _real_materials(xml)
        mids = {m["material_id"] for m in materials}
        active_ids, cell_names = _real_reachable_material_ids(geometry, mids)
        actual = [
            {**m, "reachable_executed_cells": cell_names[m["material_id"]]}
            for m in materials if m["material_id"] in active_ids
        ]
        if executed is None:
            executed = actual
            declared = materials
            reachable_ids = active_ids
        elif materials != declared or actual != executed:
            raise ValueError("physical material/nuclide or geometry reachability changed across seeds")
        physical_xmls.append({
            "seed": seed,
            "materials_xml_sha256": _sha(xml),
            "geometry_xml_sha256": _sha(geometry),
            "actual_reachable_material_ids": sorted(active_ids),
            "statepoint_sha256": run["statepoint_sha256"],
        })

    required_nuclides = sorted({n for mat in executed for n in mat["nuclides"]})
    missing_isotopes = [n for n in required_nuclides if n not in index]
    materials_report = []
    for mat in executed:
        insufficient = []
        for n in mat["nuclides"]:
            row = index.get(n)
            if (row is None or row.get("library_load_error") is not None
                    or row.get("local_kerma_mt901_present") is not True
                    or row.get("local_kerma_mt901_numerical_usable") is not True):
                insufficient.append(n)
        materials_report.append({
            **mat,
            "nuclides_missing_verified_numeric_mt901": insufficient,
            "all_material_nuclides_have_evaluated_mt901": not insufficient,
            "neutron_local_heating_physics_qualified": False,
        })
    # A valid HDF5 MT901 table would remain only a *necessary* input gate.
    # Deposited-power fidelity also needs actual tally results, appropriate
    # material temperature cross sections, source convergence and benchmarking.
    all_inputs = not missing_isotopes and all(
        m["all_material_nuclides_have_evaluated_mt901"] for m in materials_report
    )
    return {
        "audit": "INDEPENDENT_EXECUTED_OPENMC_MATERIAL_RESOLVED_MT901_PHYSICS",
        "real_four_seed_model_and_evaluated_data_identical_cross_sections_xml": True,
        "evaluated_neutron_cross_sections_xml_sha256": expected_sha,
        "evaluated_mt901_inventory_sha256": _sha(inventory_file),
        "transport_summary_sha256": _sha(summary_file),
        "real_independent_seed_count": len(seen_seeds),
        "physical_material_count": len(executed),
        "declared_material_count": len(declared),
        "uninstantiated_declared_material_count": len(declared) - len(executed),
        "executed_reachable_material_ids": sorted(reachable_ids),
        "uninstantiated_declared_materials": [
            {"material_id":m["material_id"],"material_name":m["material_name"]}
            for m in declared if m["material_id"] not in reachable_ids
        ],
        "required_executed_neutron_nuclide_count": len(required_nuclides),
        "required_executed_neutron_nuclides_missing_index": missing_isotopes,
        "material_regions_without_complete_evaluated_mt901": [
            r["material_name"] for r in materials_report
            if not r["all_material_nuclides_have_evaluated_mt901"]
        ],
        "all_executed_materials_have_numerically_usable_mt901": all_inputs,
        "physical_materials": materials_report,
        "actual_executed_source_evidence": physical_xmls,
        "thermal_power_normalization_allowed": False,
        "nasa_nerva_core_energy_deposition_validated": False,
        "interpretation": (
            "Computed against actual root-reachable OpenMC geometry and material "
            "XML plus HDF5 isotope inventory, never counting defined-but-unused "
            "materials as instantiated core physics. Missing evaluated "
            "MT901 prevents neutron-only local energy-deposition physics; "
            "zeros in heating-local tallies are NOT evidence of zero actual "
            "reactor power. A nonzero MT901 in some future library is necessary "
            "but does not by itself validate reactor energy deposition."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = evaluate_executed_core_mt901(args.summary, args.inventory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("Defined nuclear materials:", report["declared_material_count"])
    print("Actually instantiated geometry materials:", report["physical_material_count"])
    print("Defined but unreachable materials:",report["uninstantiated_declared_materials"])
    print("Actual evaluated neutron isotope inputs:",
          report["required_executed_neutron_nuclide_count"])
    print("Material regions without physical local-neutron KERMA:",
          report["material_regions_without_complete_evaluated_mt901"])
    print("Thermal power physically qualified: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
