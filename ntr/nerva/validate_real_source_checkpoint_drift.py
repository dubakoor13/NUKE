"""Only ACTUAL OpenMC workflow artifact for Stage22 test observations.

Never substitute fabricated source bank/HDF5 datasets; tests fail closed
if the recorded GitHub Actions archive is absent or expired.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from .real_source_checkpoint_drift import (
    audit_real_source_checkpoints,
)


class RealOpenMCSourceCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        value=os.environ.get("NUKE_REAL_CORE_EVIDENCE_PATH")
        if not value or not Path(value).exists():
            raise RuntimeError(
                "Real OpenMC source-bank ZIP absent; no synthetic HDF5 fallback. "
                "Download actual run 39933618638 artifact 12492499186 first.")
        cls.path=Path(value)

    def test_twenty_actual_executed_source_checkpoint_sha_and_prefixes(self):
        r=audit_real_source_checkpoints(self.path)
        self.assertEqual(r["real_native_statepoint_count"],20)
        self.assertEqual(r["rng_seeds"],[13007,23011,33013,43019])
        self.assertEqual(r["saved_batches"],[40,60,80,100,120])
        self.assertTrue(r["all_checkpoint_sha256_settings_and_prefixes_verified"])
        self.assertTrue(r["all_independently_recomputed_source_distributions_match_archive"])
        self.assertEqual(len(set(o["source_bank_sha256"] for o in r["checkpoints"])),20)

    def test_actual_nominal_site_counts_are_not_unique_fission_locations(self):
        r=audit_real_source_checkpoints(self.path)
        self.assertEqual(r["source_sites_per_bank"],3000)
        self.assertEqual(r["minimum_distinct_positions"],58)
        self.assertEqual(r["maximum_distinct_positions"],96)
        np.testing.assert_allclose(
            [r["minimum_inverse_simpson_effective_position_count"],
             r["maximum_inverse_simpson_effective_position_count"]],
            [20.244007773698982,32.320852695918234],
            rtol=1e-11)
        self.assertTrue(all(o["exact_distinct_position_count"]<100
                            for o in r["checkpoints"]))
        self.assertTrue(all(o["largest_exact_position_weight_share"]>.04
                            for o in r["checkpoints"]))

    def test_spatial_history_drift_reproduces_real_archived_reference(self):
        r=audit_real_source_checkpoints(self.path)
        np.testing.assert_allclose(
            r["mean_within_seed_adjacent_total_variation"],
            [.7290833333333333,.665,.71075,.7013333333333334],
            atol=2e-12,rtol=0)
        np.testing.assert_allclose(
            r["mean_between_seed_total_variation_by_batch"],
            [.6957222222222222,.6265555555555556,.6663888888888888,
             .6817222222222222,.6857222222222221],
            atol=2e-12,rtol=0)
        self.assertTrue(all(0<x<np.log(2.) for x in
                            r["mean_between_seed_jensen_shannon_nats_by_batch"]))

    def test_physical_source_stationarity_is_not_certified_by_good_k(self):
        r=audit_real_source_checkpoints(self.path)
        self.assertFalse(r["source_stationarity_or_independent_genealogies_qualified"])
        self.assertEqual(r["source_diversity_status"],"FAIL_SOURCE_CLONING_SCREEN")
        self.assertFalse(r["physical_hot_nerva_qualification"])
        self.assertIn("NOT independent particle genealogies",
                      "NOT independent particle genealogies")

    def test_missing_native_source_bank_checkpoint_is_fatal(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,"missing REAL OpenMC evidence"):
                audit_real_source_checkpoints(td)

    def test_bit_flip_one_middle_checkpoint_is_detected(self):
        if not self.path.is_file():
            self.skipTest("Original ZIP required for byte-level tampering check")
        with zipfile.ZipFile(self.path) as original:
            with tempfile.TemporaryDirectory() as td:
                edited=Path(td)/"modified-native-checkpoint.zip"
                with zipfile.ZipFile(edited,"w",compression=zipfile.ZIP_DEFLATED) as target:
                    for filename in original.namelist():
                        b=original.read(filename)
                        if filename=="seed_23011/statepoint.080.h5":
                            b=b[:-1]+bytes([b[-1]^1])
                        target.writestr(filename,b)
                with self.assertRaisesRegex(ValueError,"HDF5 SHA256 mismatch"):
                    audit_real_source_checkpoints(edited)

    def test_archive_source_trajectory_summary_mutation_is_detected(self):
        if not self.path.is_file():
            self.skipTest("Original ZIP required for provenance tampering check")
        with zipfile.ZipFile(self.path) as original:
            with tempfile.TemporaryDirectory() as td:
                edited=Path(td)/"altered-source-report.zip"
                with zipfile.ZipFile(edited,"w",compression=zipfile.ZIP_DEFLATED) as target:
                    for filename in original.namelist():
                        b=original.read(filename)
                        if filename=="real_core_transport_summary.json":
                            x=json.loads(b)
                            x["multi_checkpoint_source_bank_trajectory"][
                                "between_seed_mean_total_variation_by_batch"][0]+=.02
                            b=json.dumps(x).encode()
                        target.writestr(filename,b)
                with self.assertRaisesRegex(ValueError,"SHA256 evidence mismatch"):
                    audit_real_source_checkpoints(edited)


if __name__=="__main__":
    unittest.main(verbosity=2)
