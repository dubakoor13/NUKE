"""Checks against ACTUAL retained OpenMC C++ HDF5 evidence (not fake arrays).

Set NUKE_REAL_CORE_ARTIFACT_ZIP to the four-seed GitHub Actions artifact
ZIP downloaded from a successful real transport run. The test NEVER
generates an OpenMC HDF5 fixture and refuses to silently fall back to a
synthetic source. If the real artifact is unavailable, required CI fails.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

from .audit_real_core_bundle import inspect_real_core_bundle


class GenuineOpenMCCoreArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get("NUKE_REAL_CORE_ARTIFACT_ZIP")
        if not path or not Path(path).is_file():
            raise RuntimeError(
                "REAL four-seed OpenMC evidence missing. Set "
                "NUKE_REAL_CORE_ARTIFACT_ZIP to downloaded ACTUAL Actions ZIP; "
                "synthetic test doubles are forbidden for this test."
            )
        cls.path=Path(path)

    def test_original_hdf5_evidence_integrity_and_physical_nonclaims(self):
        x=inspect_real_core_bundle(self.path,workflow_run_id=39933618638)
        self.assertTrue(x["actual_openmc_HDF5_transport"])
        self.assertEqual(x["independent_seed_count"],4)
        self.assertEqual(x["seeds"],[13007,23011,33013,43019])
        self.assertEqual(x["active_generations"],90)
        self.assertEqual(x["particles_per_generation"],3000)
        self.assertEqual(x["openmc_version"],[0,15,3])
        self.assertAlmostEqual(
            x["recomputed_inverse_variance_k_mean"],.01169802857789661,
            places=11)
        self.assertFalse(x["source_convergence_demonstrated"])
        self.assertFalse(x["hot_material_data_qualified"])
        self.assertFalse(x["real_local_heating_available_for_thermal_power"])
        self.assertEqual(x["local_heating_eV_per_source"],0.)
        self.assertEqual(x["reported_criticality_condition"],"FAR_BELOW_CRITICAL")
        self.assertFalse(x["critical_operating_core_validated"])
        self.assertFalse(x["qualified_hot_core_power_prediction"])
        self.assertFalse(x["historical_NERVA_validation"])
        self.assertEqual(
            x["transport_summary_sha256"],
            "830a006751aa3a10bd4b5637a6e0a3a481fba20cc1e26a065791a6a72f6bfa21"
        )
        for row in x["statepoints"]:
            self.assertEqual(len(row["hdf5_sha256"]),64)
            self.assertTrue(row["has_saved_source_bank"])

    def test_bundle_does_not_mutate_when_read(self):
        before=hashlib.sha256(self.path.read_bytes()).hexdigest()
        inspect_real_core_bundle(self.path)
        after=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(before,after)

    def _tamper(self, mutate):
        with zipfile.ZipFile(self.path) as source:
            entries={name:source.read(name) for name in source.namelist()}
        mutate(entries)
        with tempfile.TemporaryDirectory() as tmp:
            changed=Path(tmp)/"tampered.zip"
            with zipfile.ZipFile(changed,"w",compression=zipfile.ZIP_DEFLATED) as z:
                for name,raw in entries.items():
                    z.writestr(name,raw)
            return inspect_real_core_bundle(changed)

    def test_changed_actual_statepoint_hash_is_rejected(self):
        def tamper(entries):
            target="seed_13007/statepoint.120.h5"
            raw=bytearray(entries[target])
            raw[-32]^=1
            entries[target]=bytes(raw)
        with self.assertRaisesRegex(ValueError,"checksum|corrupted|HDF5"):
            self._tamper(tamper)

    def test_duplicate_rng_seeds_are_rejected(self):
        def tamper(entries):
            name="real_core_transport_summary.json"
            doc=json.loads(entries[name])
            doc["runs"][1]["seed"]=doc["runs"][0]["seed"]
            entries[name]=json.dumps(doc).encode()
        with self.assertRaisesRegex(ValueError,"duplicate"):
            self._tamper(tamper)

    def test_transport_summary_relabelled_as_synthetic_is_rejected(self):
        def tamper(entries):
            name="real_core_transport_summary.json"
            doc=json.loads(entries[name])
            doc["execution"]="SYNTHETIC"
            entries[name]=json.dumps(doc).encode()
        with self.assertRaisesRegex(ValueError,"REAL|COLD"):
            self._tamper(tamper)

    def test_wrong_criticality_report_link_rejected(self):
        def tamper(entries):
            name="core_criticality_physical_qualification.json"
            doc=json.loads(entries[name])
            doc["transport_summary_sha256"]="0"*64
            entries[name]=json.dumps(doc).encode()
        with self.assertRaisesRegex(ValueError,"qualification references"):
            self._tamper(tamper)

    def test_wrong_xs_provenance_rejected(self):
        def tamper(entries):
            name="verified_xs_files.json"
            doc=json.loads(entries[name])
            doc["cross_sections_sha256"]="f"*64
            entries[name]=json.dumps(doc).encode()
        with self.assertRaisesRegex(ValueError,"index hashes"):
            self._tamper(tamper)

    def test_false_hot_qualification_without_temperature_grid_is_rejected(self):
        def tamper(entries):
            name="core_material_temperature_qualification.json"
            doc=json.loads(entries[name])
            doc["hot_operating_temperature_neutron_physics_qualified"]=True
            doc["real_transport_summary_sha256"]=hashlib.sha256(
                entries["real_core_transport_summary.json"]).hexdigest()
            entries[name]=json.dumps(doc).encode()
        with self.assertRaisesRegex(ValueError,"contradicts"):
            self._tamper(tamper)


if __name__=="__main__":
    unittest.main(verbosity=2)
