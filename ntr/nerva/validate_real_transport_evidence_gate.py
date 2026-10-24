"""Regression against genuinely executed four-seed OpenMC HDF5 evidence.

The test fixture is downloaded from GitHub Actions run 39933618638,
artifact 12492499186. No synthetic statepoint is a stand-in for transport.
If the required artifact is absent, this module FAILS (not silently skips).
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from .real_transport_evidence_gate import audit_real_transport_evidence

_REAL_RUN_ARCHIVE_SHA = (
    "cf05ab907bee76b198d3dc4736a84c6d560f3a8c2d753b78b5605c4826e53274"
)


class ExecutedOpenMCEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("NUKE_REAL_CORE_EVIDENCE_PATH")
        if not path or not Path(path).exists():
            raise RuntimeError(
                "Real four-seed OpenMC evidence absent: "
                "set NUKE_REAL_CORE_EVIDENCE_PATH to downloaded 39933618638 "
                "artifact ZIP or extracted directory. Synthetic substitutes forbidden."
            )
        cls.path = Path(path)

    def test_real_hdf5_statepoints_and_xs_provenance(self):
        report = audit_real_transport_evidence(self.path)
        self.assertEqual(report["evidence_kind"],
                         "ACTUAL_EXECUTED_OPENMC_COLD_NEUTRON_TRANSPORT")
        self.assertEqual(report["independent_seed_count"],4)
        self.assertEqual([s["seed"] for s in report["statepoints"]],
                         [13007,23011,33013,43019])
        self.assertTrue(all(s["openmc_version"]==[0,15,3]
                            for s in report["statepoints"]))
        self.assertTrue(all(s["active_realizations"]==90
                            for s in report["statepoints"]))
        self.assertTrue(all(s["particles_per_generation"]==3000
                            for s in report["statepoints"]))
        self.assertEqual(len(report["same_executed_xml_sha256"]),3)

    def test_real_eigenvalues_match_independent_binary_observations(self):
        result = audit_real_transport_evidence(self.path)
        k = np.array([entry["k_effective"] for entry in result["statepoints"]])
        sem = np.array([entry["reported_openmc_sem"]
                        for entry in result["statepoints"]])
        np.testing.assert_allclose(k,
             [.01181822339995014,.01163543885601469,
              .0116943712526958,.0116419966504851],
              rtol=1e-5, atol=2e-7)
        self.assertTrue(np.all(sem>0))
        self.assertAlmostEqual(result["recomputed_inverse_variance_k_mean"],
                               .01169802857789661,places=12)
        self.assertAlmostEqual(result["reported_openmc_formal_sem"],
                               7.69260125465282e-05,places=12)

    def test_authentic_execution_still_fails_hot_reactor_claim(self):
        result = audit_real_transport_evidence(self.path)
        self.assertEqual(result["criticality_observation"],
                         "FAR_BELOW_CRITICAL")
        self.assertEqual(result["local_heating_observation"],
                         "BLOCKED_NO_VERIFIED_LOCAL_HEAT_DEPOSITION")
        self.assertEqual(result["temperature_observation"]["evaluated_K"],
                         [293.6059129982851])
        self.assertEqual(result["temperature_observation"]["root_reachable_materials"],4)
        self.assertEqual(result["temperature_observation"]["explicitly_temperature_assigned"],1)
        self.assertTrue(all(accepted is False for accepted in
                            result["physical_gates"].values()))
        self.assertIs(result["reactor_core_historical_or_hot_state_qualified"],False)

    def test_archive_is_original_GitHub_artifact_bytes_if_zip(self):
        if not self.path.is_file():
            self.skipTest("Extracted GitHub workflow artifact; original ZIP SHA not available")
        report = audit_real_transport_evidence(
            self.path,expected_archive_sha256=_REAL_RUN_ARCHIVE_SHA)
        self.assertEqual(report["archive_sha256"],_REAL_RUN_ARCHIVE_SHA)

    def test_missing_physical_evidence_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,"missing REAL OpenMC evidence"):
                audit_real_transport_evidence(Path(td))

    def test_real_statepoint_mutation_is_detected_by_sha(self):
        if not self.path.is_file():
            self.skipTest("ZIP mutation regression requires original downloaded ZIP")
        with zipfile.ZipFile(self.path) as source:
            with tempfile.TemporaryDirectory() as td:
                edited=Path(td)/"bad-statepoint.zip"
                with zipfile.ZipFile(edited,"w",compression=zipfile.ZIP_DEFLATED) as target:
                    for entry in source.namelist():
                        payload=source.read(entry)
                        if entry=="seed_13007/statepoint.120.h5":
                            payload=payload[:-1]+bytes([payload[-1]^1])
                        target.writestr(entry,payload)
                with self.assertRaisesRegex(ValueError,"SHA256 evidence mismatch"):
                    audit_real_transport_evidence(edited)

    def test_real_summary_provenance_mutation_is_detected(self):
        if not self.path.is_file():
            self.skipTest("ZIP mutation regression requires original downloaded ZIP")
        with zipfile.ZipFile(self.path) as source:
            with tempfile.TemporaryDirectory() as td:
                edited=Path(td)/"bad-summary.zip"
                with zipfile.ZipFile(edited,"w",compression=zipfile.ZIP_DEFLATED) as target:
                    for entry in source.namelist():
                        payload=source.read(entry)
                        if entry=="real_core_transport_summary.json":
                            record=json.loads(payload)
                            record["hydrogen_temperature_K"]=303.
                            payload=json.dumps(record).encode()
                        target.writestr(entry,payload)
                with self.assertRaisesRegex(ValueError,"SHA256 evidence mismatch"):
                    audit_real_transport_evidence(edited)

    def test_changed_reactor_geometry_between_seeds_is_detected(self):
        if not self.path.is_file():
            self.skipTest("ZIP mutation regression requires original downloaded ZIP")
        with zipfile.ZipFile(self.path) as source:
            with tempfile.TemporaryDirectory() as td:
                edited=Path(td)/"bad-model.zip"
                with zipfile.ZipFile(edited,"w",compression=zipfile.ZIP_DEFLATED) as target:
                    for entry in source.namelist():
                        payload=source.read(entry)
                        if entry=="seed_23011/geometry.xml":
                            payload+=b"<!-- changed reactor geometry evidence -->\n"
                        target.writestr(entry,payload)
                with self.assertRaisesRegex(ValueError,"geometry/material/tally differs"):
                    audit_real_transport_evidence(edited)


if __name__=="__main__":
    unittest.main(verbosity=2)
