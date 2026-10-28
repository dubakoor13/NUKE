"""Stage34 disjoint OpenMC tally moment math and mandatory native HDF5 tests."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from .real_coupled_disjoint_heating import (
    audit_real_disjoint_heating,
    disjoint_raw_moments,
)


def _example():
    # Ten, fifteen, twenty nonoverlapping independent *fixture* batches.
    # Real Monte Carlo neutron generations are NOT statistically independent.
    a = np.tile(np.array([[1., 2.], [3., 4.]]), (5, 1))
    b = np.tile(np.array([[2., 3.], [4., 5.], [6., 7.]]), (5, 1))
    c = np.tile(np.array([[5., 4.], [3., 2.]]), (10, 1))
    data = np.concatenate((a, b, c))
    cut = (10, 25, 45)
    return data, [np.sum(data[:k], axis=0) for k in cut], [
        np.sum(data[:k]**2, axis=0) for k in cut
    ]


class DisjointHeatingMomentTests(unittest.TestCase):
    def test_original_cumulative_sum_and_sum_sq_reconstruct_three_windows(self):
        x, sums, sq = _example()
        w = disjoint_raw_moments(sums, sq, [10, 25, 45], label="fixture")
        self.assertEqual([y["active_realizations"] for y in w], [10, 15, 20])
        for row, subset in zip(w, (x[:10], x[10:25], x[25:45])):
            np.testing.assert_allclose(row["mean"], subset.mean(axis=0))
            np.testing.assert_allclose(row["descriptive_sample_batch_variance_not_IID"],
                                       subset.var(axis=0, ddof=1), atol=1e-12)

    def test_window_means_require_batch_weighting(self):
        _, sums, sq = _example()
        w = disjoint_raw_moments(sums, sq, [10, 25, 45], label="fixture")
        recovery = sum(row["raw_sum"] for row in w)/45.
        np.testing.assert_allclose(recovery, sums[-1]/45.)
        self.assertFalse(np.allclose(sum(row["mean"] for row in w)/3, recovery))

    def test_constant_batch_values_have_zero_sample_variance(self):
        x = np.tile(np.array([1.25, 6.]), (45, 1))
        w = disjoint_raw_moments(
            [x[:n].sum(axis=0) for n in (10, 25, 45)],
            [(x[:n]**2).sum(axis=0) for n in (10, 25, 45)],
            [10, 25, 45], label="constant"
        )
        for row in w:
            np.testing.assert_allclose(row["descriptive_sample_batch_variance_not_IID"],
                                       0., rtol=0, atol=1e-12)

    def test_correlation_cannot_be_built_from_marginal_variances(self):
        # Exactly anticorrelated energy bins have a constant total.
        x = np.column_stack((np.arange(1, 46, dtype=float),
                             46.-np.arange(1, 46, dtype=float)))
        w = disjoint_raw_moments(
            [x[:n].sum(axis=0) for n in (10, 25, 45)],
            [(x[:n]**2).sum(axis=0) for n in (10, 25, 45)],
            [10, 25, 45], label="anticorrelated"
        )
        self.assertGreater(float(w[2]["descriptive_sample_batch_variance_not_IID"].sum()), 0)
        self.assertEqual(float(np.var(x[25:45].sum(axis=1))), 0.)

    def test_wrong_realization_counts_fail_closed(self):
        _, sums, sq = _example()
        for bad in ([10, 25, 44], [10, 10, 45], [10, 45], []):
            with self.subTest(counts=bad), self.assertRaisesRegex(ValueError, "invalid"):
                disjoint_raw_moments(sums, sq, bad, label="invalid")

    def test_decreasing_native_cumulative_sum_fails_closed(self):
        _, sums, sq = _example()
        bad = [x.copy() for x in sums]
        bad[1][0] = 0.
        with self.assertRaisesRegex(ValueError, "decreased"):
            disjoint_raw_moments(bad, sq, [10, 25, 45], label="decreasing")

    def test_negative_sum_sq_or_nonfinite_moments_fail_closed(self):
        _, sums, sq = _example()
        badsq = [x.copy() for x in sq]
        badsq[1][0] = -1.
        with self.assertRaisesRegex(ValueError, "invalid"):
            disjoint_raw_moments(sums, badsq, [10, 25, 45], label="bad squared")
        badsum = [x.copy() for x in sums]
        badsum[2][1] = np.nan
        with self.assertRaisesRegex(ValueError, "invalid"):
            disjoint_raw_moments(badsum, sq, [10, 25, 45], label="nonfinite")

    def test_impossible_negative_variance_from_forged_second_moments_fails(self):
        _, sums, sq = _example()
        bad = [x.copy() for x in sq]
        bad[0] *= .01  # violates E[X^2] >= E[X]^2 on first window
        with self.assertRaisesRegex(ValueError, "Cauchy"):
            disjoint_raw_moments(sums, bad, [10, 25, 45], label="forged")

    def test_missing_real_openmc_checkpoints_cannot_use_mock(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                audit_real_disjoint_heating(temporary)


class ActualRealDisjointOpenMCHeatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("NUKE_STAGE34_ORIGINAL_REAL_HDF5_ROOT")
        if not root:
            raise RuntimeError("Stage33 original 12 coupled OpenMC HDF5 are mandatory")
        cls.data = audit_real_disjoint_heating(root)

    def test_all_12_original_statepoints_and_full_raw_tally_moment_recovery(self):
        r = self.data
        self.assertEqual(r["genuine_openmc_statepoints_reverified"], 12)
        self.assertTrue(r["all_raw_moment_telescope_and_final_heat_identical"])
        self.assertEqual(r["active_windows"], [[21,30],[31,45],[46,65]])
        for s in r["per_seed"]:
            self.assertEqual([w["active_realizations"] for w in s["disjoint_heating_windows"]],
                             [10,15,20])
            self.assertTrue(s["weighted_window_heating_recovers_original_final"])
            for w in s["disjoint_heating_windows"]:
                self.assertLess(abs(w["material_mesh_relative_residual"]),1e-9)
                self.assertLess(abs(w["all_four_particle_mesh_relative_residual"]),1e-9)

    def test_real_spatial_deposition_never_qualifies_hot_nerva(self):
        r = self.data
        self.assertEqual(len(r["per_seed"]), 4)
        self.assertFalse(r["cumulative_checkpoint_means_are_independent"])
        self.assertFalse(r["separate_active_generation_batches_statistically_independent"])
        self.assertFalse(r["cross_bin_within_window_covariances_available"])
        self.assertFalse(r["fission_source_stationarity_qualified"])
        self.assertFalse(r["real_hot_nerva_physical_watts_qualified"])
        self.assertFalse(r["nasa_historical_benchmark_validated"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
