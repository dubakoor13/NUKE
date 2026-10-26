"""Stage 26 parser unit checks and checks against genuine four-seed OpenMC HDF5.

No mock statepoint may satisfy the real-evidence tests. Synthetic HDF5
fixtures test only correctness of native graph traversal and rejection.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
import unittest

import h5py
import numpy as np

from .real_native_root_topology import _graph_from_native_summary, audit_native_root_topology


def _fixture(*, orphan=False, another_root=False, cycle=False,
             bad_material=False, bad_lattice=False, change_fill=False):
    data = io.BytesIO()
    with h5py.File(data, "w") as h5:
        mats = h5.create_group("materials")
        for mid in (10, 20, 30):
            mats.create_group(f"material {mid}")
        geo = h5.create_group("geometry")
        cells = geo.create_group("cells")
        universes = geo.create_group("universes")
        lattices = geo.create_group("lattices")

        def cell(cid, kind, ref):
            group = cells.create_group(f"cell {cid}")
            group.create_dataset("fill_type", data=np.bytes_(kind))
            name = {"material": "material", "universe": "fill", "lattice": "lattice"}[kind]
            group.create_dataset(name, data=ref)

        cell(1, "universe", 2)
        cell(2, "material", 999 if bad_material else 10)
        cell(3, "lattice", 5)
        cell(4, "material", 20 if not change_fill else 30)
        if orphan or another_root:
            cell(5, "material", 30)
        if cycle:
            cell(6, "universe", 2)
        for uid, ids in ((1, [1, 3]), (2, [2] + ([6] if cycle else [])),
                         (3, [4])):
            group = universes.create_group(f"universe {uid}")
            group.create_dataset("cells", data=np.array(ids, dtype="i8"))
        if another_root:
            group = universes.create_group("universe 4")
            group.create_dataset("cells", data=np.array([5], dtype="i8"))
        group = lattices.create_group("lattice 5")
        group.create_dataset("type", data=np.bytes_("rectangular"))
        group.create_dataset("outer", data=-1)
        group.create_dataset("universes", data=np.array([[[999 if bad_lattice else 3]]]))
    return data.getvalue()


class NativeRootGraphParserTests(unittest.TestCase):
    def test_lattice_and_nested_universe_reachability(self):
        result = _graph_from_native_summary(_fixture())
        self.assertEqual(result["root_universe_id"], 1)
        self.assertEqual(result["root_connected_universe_ids"], [1, 2, 3])
        self.assertEqual(result["root_connected_lattice_ids"], [5])
        self.assertEqual(result["root_connected_material_ids"], [10, 20])
        self.assertEqual(result["root_connected_material_cell_definition_ids"], [2, 4])
        self.assertEqual(result["all_native_material_cell_definition_count"], 2)

    def test_orphan_material_cell_is_not_root_connected(self):
        result = _graph_from_native_summary(_fixture(orphan=True))
        self.assertEqual(result["root_connected_material_ids"], [10, 20])
        self.assertEqual(result["unassigned_cell_definition_ids"], [5])
        self.assertEqual(result["non_root_material_cell_definition_ids"], [5])
        self.assertEqual(result["all_native_material_cell_definition_count"], 3)

    def test_ambiguous_root_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "nonunique native root universe"):
            _graph_from_native_summary(_fixture(another_root=True))

    def test_missing_material_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "nonexistent material"):
            _graph_from_native_summary(_fixture(bad_material=True))

    def test_missing_lattice_universe_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "lattice references nonexistent universe"):
            _graph_from_native_summary(_fixture(bad_lattice=True))

    def test_cycle_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "cyclic native universe"):
            _graph_from_native_summary(_fixture(cycle=True))

    def test_topology_hash_changes_when_natively_recorded_fill_changes(self):
        old = _graph_from_native_summary(_fixture())
        new = _graph_from_native_summary(_fixture(change_fill=True))
        self.assertNotEqual(old["topology_sha256"], new["topology_sha256"])
        self.assertEqual(old["root_connected_material_ids"], [10, 20])
        self.assertEqual(new["root_connected_material_ids"], [10, 30])


class OriginalOpenMCNativeGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = os.getenv("NUKE_REAL_CORE_EVIDENCE_PATH")
        if not raw or not Path(raw).exists():
            raise RuntimeError("Must use original four-seed OpenMC HDF5 evidence, no replacement")
        cls.path = Path(raw)

    def test_four_native_openmc_root_geometries_and_stage25_agree(self):
        result = audit_native_root_topology(self.path)
        self.assertEqual(result["original_openmc_seed_count"], 4)
        self.assertTrue(result["cross_seed_native_geometry_consistent"])
        self.assertTrue(result["stage25_materials_root_connected"])
        self.assertEqual([x["seed"] for x in result["per_seed_native_geometry"]],
                         [13007, 23011, 33013, 43019])
        for run in result["per_seed_native_geometry"]:
            self.assertEqual(run["root_connected_material_ids"], [19, 20, 21, 22])
            self.assertEqual(len(run["root_connected_material_cell_definition_ids"]), 60)
            self.assertEqual(run["non_root_material_cell_definition_ids"], [])

    def test_topology_does_not_claim_observed_neutron_visits_or_qualified_heat(self):
        result = audit_native_root_topology(self.path)
        self.assertFalse(result["occupied_physical_cell_volume_established"])
        self.assertFalse(result["neutron_history_visitation_established"])
        self.assertFalse(result["qualified_hot_nerva_core"])
        self.assertFalse(result["qualified_local_neutron_photon_heating"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
