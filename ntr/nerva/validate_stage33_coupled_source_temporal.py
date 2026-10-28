"""Stage33 adversarial numerics and mandatory original OpenMC HDF5 tests."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .real_coupled_source_temporal import (
    CHECKPOINTS,
    SEEDS,
    audit_real_coupled_temporal_source,
    temporal_source_statistics,
    verify_native_checkpoint_prefixes,
)


def _record(seed: int, n: int, *, shift: float = 0) -> dict:
    return {"seed": seed, "batch": n,
            "k": np.arange(n, dtype=float) * 0.001 + 0.01 + shift,
            "entropy": np.arange(n, dtype=float) * 0.01 + 1.}


class TemporalSourceMathTests(unittest.TestCase):
    def test_static_probability_distributions_have_zero_temporal_drift(self):
        a = np.full((4, 3, 4, 4, 10), 1/160.)
        r = temporal_source_statistics(a)
        self.assertEqual(r["mean_30_to_65_within_seed_TV"], 0.)
        self.assertFalse(r["source_stationarity_independently_qualified"])

    def test_disjoint_early_and_final_banks_expose_maximal_drift(self):
        a = np.zeros((4, 3, 4, 4, 10))
        a[:, 0, 0, 0, 0] = 1.
        a[:, 1, 1, 1, 1] = 1.
        a[:, 2, 3, 3, 9] = 1.
        r = temporal_source_statistics(a)
        self.assertEqual(r["mean_30_to_65_within_seed_TV"], 1.)
        self.assertEqual(r["source_drift_and_resolution"] if "source_drift_and_resolution" in r else
                         r["same_seed_temporal_source_variation"][0]["same_seed_TV_batch45_to_65"], 1.)

    def test_coarsening_can_reduce_fine_variation(self):
        a = np.zeros((4, 3, 4, 4, 10))
        a[:, :, 0, 0, 0] = 1.
        a[:, 2, 0, 0, 0] = 0.
        a[:, 2, 1, 0, 0] = 1.
        r = temporal_source_statistics(a)
        self.assertEqual(r["mean_30_to_65_within_seed_TV"], 1.)
        self.assertEqual(r["same_seed_temporal_source_variation"][0][
            "coarse_2x2x5_same_seed_TV_batch30_to_65"], 0.)

    def test_invalid_source_weights_and_dimensions_fail(self):
        for f in (np.zeros((4, 3, 4, 4, 10)),
                  np.full((4, 3, 4, 4, 10), -1.),
                  np.full((4, 3, 4, 4, 10), np.nan),
                  np.zeros((4, 2, 4, 4, 10)),
                  np.zeros((4, 3, 2, 2, 5))):
            with self.subTest(shape=f.shape), self.assertRaises(ValueError):
                temporal_source_statistics(f)

    def test_exact_prefix_and_disjoint_active_keff_windows(self):
        recs = [_record(13007, k) for k in CHECKPOINTS]
        result = verify_native_checkpoint_prefixes(recs)
        self.assertTrue(result["exact_history_prefixes_verified"])
        self.assertEqual([x["batches"] for x in result[
            "descriptive_generation_keff_and_entropy_windows"]],
            [[21, 30], [31, 45], [46, 65]])
        self.assertFalse(result["generation_k_samples_independent"])

    def test_modified_past_k_generation_rejected(self):
        recs = [_record(13007, k) for k in CHECKPOINTS]
        recs[1]["k"][6] += 0.01
        with self.assertRaisesRegex(ValueError, "prefix"):
            verify_native_checkpoint_prefixes(recs)

    def test_modified_past_entropy_rejected(self):
        recs = [_record(13007, k) for k in CHECKPOINTS]
        recs[2]["entropy"][1] += .02
        with self.assertRaisesRegex(ValueError, "prefix"):
            verify_native_checkpoint_prefixes(recs)

    def test_cross_seed_checkpoint_splicing_rejected(self):
        recs = [_record(13007, k) for k in CHECKPOINTS]
        recs[1]["seed"] = 23011
        with self.assertRaisesRegex(ValueError, "different RNG seeds"):
            verify_native_checkpoint_prefixes(recs)

    def test_missing_real_openmc_checkpoint_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                audit_real_coupled_temporal_source(tmp)

    def test_forged_heating_qualification_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/"actual_four_seed_coupled_heating_covariance.json").write_text(
                json.dumps({"schema":"not_original", "real_native_coupled_hdf5_count":4}))
            with self.assertRaisesRegex(ValueError, "unqualified"):
                audit_real_coupled_temporal_source(tmp)


class ActualOpenMCCoupledTemporalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = os.environ.get("NUKE_STAGE33_REAL_COUPLED_ROOT")
        if not base:
            raise RuntimeError("real 12 statepoints from four new OpenMC coupled physics runs required")
        cls.report = audit_real_coupled_temporal_source(base)

    def test_original_12_neutron_photon_statepoints_and_exact_keff_history(self):
        r = self.report
        self.assertEqual(r["native_original_checkpoint_count"], 12)
        self.assertEqual(r["seed_order"], list(SEEDS))
        self.assertEqual(r["batch_checkpoints"], list(CHECKPOINTS))
        self.assertTrue(r["all_same_seed_native_generation_history_prefixes_exact"])
        self.assertTrue(r["all_final_native_sha_match_genuine_coupled_heating"])
        for seed in r["per_seed_real_transport"]:
            self.assertEqual(len(seed["checkpoints"]), 3)
            self.assertTrue(seed["eigen_history"]["exact_history_prefixes_verified"])
            self.assertGreater(seed["checkpoints"][-1]["source_bank"]["source_weights"], 0.)

    def test_original_temporal_drift_and_physical_qualification_rejected(self):
        r = self.report
        result = r["source_drift_and_resolution"]
        self.assertEqual(len(result["same_seed_temporal_source_variation"]), 4)
        self.assertEqual(len(result["between_seed_source_variation_at_each_batch"]), 3)
        self.assertTrue(all(0 <= x["same_seed_TV_batch30_to_65"] <= 1.0 + 1e-12
                            for x in result["same_seed_temporal_source_variation"]))
        self.assertFalse(r["physically_stationary_source_qualified"])
        self.assertFalse(r["independent_critical_hot_nerva_watts_qualified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
