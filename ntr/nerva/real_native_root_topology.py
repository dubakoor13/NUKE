"""Independent native OpenMC summary.h5 root-connectivity evidence audit.

The Stage 25 material tally describes native material-filled cell definitions,
not proof that a cell is root-connected or visited by neutron histories.
This module traverses the actual HDF5 Universe->Cell->Universe/Lattice graph,
and compares root-connected material IDs/cell definitions to Stage 25.

Read-only physics QA. It does not tune criticality, modify the reactor, supply
nuclear data, or claim that topological membership proves occupied volume.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re

import h5py
import numpy as np

from .real_native_material_xs_audit import audit_native_material_xs
from .real_transport_evidence_gate import _Bundle, sha256_bytes


def _group_id(label: str, prefix: str) -> int:
    m = re.fullmatch(re.escape(prefix) + r" ([1-9][0-9]*)", label)
    if m is None:
        raise ValueError("invalid native OpenMC group ID: " + label)
    return int(m.group(1))


def _ref(value: object) -> int:
    arr = np.asarray(value)
    if arr.ndim:
        raise ValueError("non-scalar native reference")
    raw = arr[()]
    if isinstance(raw, (bytes, np.bytes_)):
        m = re.fullmatch(
            r"(?:(?:material|universe|lattice) )?(-?[0-9]+)",
            bytes(raw).decode("ascii").rstrip("\x00"))
        if m is None:
            raise ValueError("non-numeric native reference")
        return int(m.group(1))
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    raise ValueError("unsupported native reference type")


def _graph_from_native_summary(blob: bytes) -> dict:
    """Enumerate referenced HDF5 cell *definitions*, not lattice instances."""
    with h5py.File(io.BytesIO(blob), "r") as h5:
        if ("materials" not in h5 or "geometry" not in h5 or
            any(name not in h5["geometry"]
                for name in ("cells", "universes", "lattices")) or
            "dagmc" in h5["geometry"].attrs):
            raise ValueError("missing or unsupported native CSG HDF5 geometry")
        materials = {_group_id(name, "material") for name in h5["materials"]}
        cells = {}
        for name, group in h5["geometry/cells"].items():
            cid = _group_id(name, "cell")
            raw = group["fill_type"][()]
            if not isinstance(raw, (bytes, np.bytes_)):
                raise ValueError("invalid native fill type encoding")
            kind = bytes(raw).decode("ascii").rstrip("\x00")
            field = {"material": "material", "universe": "fill",
                     "lattice": "lattice"}.get(kind)
            if field is None:
                raise ValueError("unknown native fill type")
            refs = tuple(_ref(x) for x in np.asarray(group[field][()]).reshape(-1))
            if (not refs or (kind != "material" and len(refs) != 1)
                or any(x < -1 or x == 0 for x in refs)):
                raise ValueError("invalid native cell fill references")
            cells[cid] = (kind, refs)
        universes = {}
        for name, group in h5["geometry/universes"].items():
            uid = _group_id(name, "universe")
            ids = np.asarray(group["cells"][()], dtype=np.int64).ravel().tolist()
            if any(x <= 0 for x in ids) or len(set(ids)) != len(ids):
                raise ValueError("invalid or duplicate universe cell")
            universes[uid] = tuple(int(x) for x in ids)
        lattices = {}
        for name, group in h5["geometry/lattices"].items():
            lid = _group_id(name, "lattice")
            raw_type = group["type"][()]
            if (not isinstance(raw_type, (bytes, np.bytes_)) or
                bytes(raw_type).decode("ascii").rstrip("\x00")
                    not in ("hexagonal", "rectangular")):
                raise ValueError("unsupported lattice type")
            ids = np.asarray(group["universes"][()], dtype=np.int64).ravel()
            outer = _ref(group["outer"][()])
            if (ids.size == 0 or outer < -1 or outer == 0 or
                np.any((ids < -1) | (ids == 0))):
                raise ValueError("invalid lattice universe indices")
            linked = {int(x) for x in ids if x > 0}
            if outer > 0:
                linked.add(outer)
            lattices[lid] = tuple(sorted(linked))

    if not cells or not universes or not materials:
        raise ValueError("empty native CSG graph")
    membership = {}
    for uid, cell_ids in universes.items():
        for cid in cell_ids:
            if cid not in cells:
                raise ValueError("universe references nonexistent cell")
            if cid in membership and membership[cid] != uid:
                raise ValueError("cell assigned to multiple universes")
            membership[cid] = uid
    referenced_universes = set()
    for kind, refs in cells.values():
        if kind == "material":
            if any(x > 0 and x not in materials for x in refs):
                raise ValueError("cell references nonexistent material")
        elif kind == "universe":
            if refs[0] not in universes:
                raise ValueError("cell references nonexistent universe")
            referenced_universes.add(refs[0])
        else:
            if refs[0] not in lattices:
                raise ValueError("cell references nonexistent lattice")
    for linked in lattices.values():
        if any(u not in universes for u in linked):
            raise ValueError("lattice references nonexistent universe")
        referenced_universes.update(linked)
    roots = set(universes) - referenced_universes
    if len(roots) != 1:
        raise ValueError("nonunique native root universe")
    root = next(iter(roots))
    visited, active, reached_cells, material_cells, reached_materials, reached_lattices = (
        set(), set(), set(), set(), set(), set())

    def visit(uid):
        if uid in active:
            raise ValueError("cyclic native universe/lattice graph")
        if uid in visited:
            return
        active.add(uid)
        for cid in universes[uid]:
            reached_cells.add(cid)
            kind, refs = cells[cid]
            if kind == "material":
                if any(x > 0 for x in refs):
                    material_cells.add(cid)
                    reached_materials.update(x for x in refs if x > 0)
            elif kind == "universe":
                visit(refs[0])
            else:
                reached_lattices.add(refs[0])
                for child in lattices[refs[0]]:
                    visit(child)
        active.remove(uid)
        visited.add(uid)

    visit(root)
    all_material_cells = {
        cid for cid, (kind, refs) in cells.items()
        if kind == "material" and any(x > 0 for x in refs)
    }
    graph = {"root": root, "cells": sorted((cid, kind, refs)
                                          for cid, (kind, refs) in cells.items()),
             "universes": sorted(universes.items()),
             "lattices": sorted(lattices.items())}
    return {
        "root_universe_id": root,
        "root_connected_universe_ids": sorted(visited),
        "root_connected_lattice_ids": sorted(reached_lattices),
        "root_connected_cell_definition_ids": sorted(reached_cells),
        "root_connected_material_cell_definition_ids": sorted(material_cells),
        "root_connected_material_ids": sorted(reached_materials),
        "non_root_material_cell_definition_ids": sorted(all_material_cells - material_cells),
        "unassigned_cell_definition_ids": sorted(set(cells) - set(membership)),
        "non_root_universe_ids": sorted(set(universes) - visited),
        "all_native_material_cell_definition_count": len(all_material_cells),
        "declared_native_material_count": len(materials),
        "topology_sha256": hashlib.sha256(
            json.dumps(graph, sort_keys=True).encode("utf-8")).hexdigest(),
    }


def audit_native_root_topology(artifact: str | Path) -> dict:
    """Require original OpenMC evidence and reconcile all four real seeds."""
    previous = audit_native_material_xs(artifact)
    src = _Bundle(Path(artifact))
    try:
        observations = []
        for original in previous["per_seed_native_evidence"]:
            seed = original["seed"]
            native = src.read(f"seed_{seed}/summary.h5")
            topology = _graph_from_native_summary(native)
            if (topology["root_connected_material_ids"] != original["reachable_material_ids"]
                or len(topology["root_connected_material_cell_definition_ids"])
                    != original["material_filled_cells"]
                or topology["all_native_material_cell_definition_count"]
                    != original["material_filled_cells"]
                or topology["non_root_material_cell_definition_ids"]):
                raise ValueError("Stage 25 inventory includes non-root cell definitions")
            observations.append({
                "seed": seed,
                "summary_sha256": sha256_bytes(native),
                **topology,
            })
        if len({item["topology_sha256"] for item in observations}) != 1:
            raise ValueError("native topology changes between independent seeds")
        return {
            "schema": "nuke-openmc-native-root-stage26-v1",
            "evidence_kind": "FOUR_ORIGINAL_OPENMC_SUMMARY_HDF5_GEOMETRY_GRAPHS",
            "original_openmc_seed_count": len(observations),
            "cross_seed_native_geometry_consistent": True,
            "stage25_materials_root_connected": True,
            "occupied_physical_cell_volume_established": False,
            "neutron_history_visitation_established": False,
            "qualified_hot_nerva_core": False,
            "qualified_local_neutron_photon_heating": False,
            "per_seed_native_geometry": observations,
            "limitations": [
                "Root graph connectivity alone cannot establish positive cell volume or neutron visitation.",
                "Unique cell definitions are NOT lattice instance counts.",
                "Cold surrogate is far subcritical; source stationarity and heating remain unqualified.",
                "No original isotope HDF5 binary SHA or hot material data independently validated.",
            ],
        }
    finally:
        src.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit_native_root_topology(a.artifact)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print("Original OpenMC native geometry audited:", result["original_openmc_seed_count"], "seeds")
    print("NERVA hot-core physical qualification: BLOCKED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
