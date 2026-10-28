"""Stage35 conservative 3D heating numerics and required genuine-HDF5 checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .real_coupled_spatial_heating_stability import (
    audit_actual_3d_heating,
    conservative_block_sum,
    measure_3d_heating_stability,
    total_variation,
)


def example_uniform() -> np.ndarray:
    a = np.ones((4, 3, 12, 12, 10), dtype=float)
    a *= np.asarray([1., 2., 3., 4.])[:, None, None, None, None]
    a *= np.asarray([1., 1.5, 2.])[None, :, None, None, None]
    return a


class SpatialHeatingMathTests(unittest.TestCase):
    def test_constant_spatial_shape_with_change_in_amplitude_has_no_drift(self):
        report = measure_3d_heating_stability(example_uniform())
        for scale in report["scale_sensitivity"]:
            self.assertEqual(scale["mean_early_late_TV"], 0.)
        self.assertFalse(report["spatial_voxel_heating_uncertainty_calibrated"])
        self.assertFalse(report["hot_nerva_reactor_power_density_in_watts_qualified"])

    def test_disjoint_hotspots_have_maximum_fine_resolution_total_variation(self):
        a = np.zeros((4, 3, 12, 12, 10), dtype=float)
        a[:, 0, 0, 0, 0] = 1.
        a[:, 1, 6, 6, 4] = 1.
        a[:, 2, 11, 11, 9] = 1.
        result = measure_3d_heating_stability(a)
        self.assertEqual(result["scale_sensitivity"][0]["mean_early_late_TV"], 1.)
        self.assertEqual(result["scale_sensitivity"][-1]["mean_early_late_TV"], 1.)

    def test_fine_spatial_drift_disappears_under_valid_coarsening(self):
        a = np.zeros((4, 3, 12, 12, 10), dtype=float)
        a[:, 0, 0, 0, 5] = 1.
        a[:, 1, 0, 0, 5] = 1.
        a[:, 2, 1, 1, 5] = 1.
        result = measure_3d_heating_stability(a)
        self.assertEqual(result["scale_sensitivity"][0]["mean_early_late_TV"], 1.)
        for coarser in result["scale_sensitivity"][1:]:
            self.assertEqual(coarser["mean_early_late_TV"], 0.)

    def test_conservative_3d_mesh_exact_energy_conservation(self):
        rng = np.random.default_rng(113)
        a = rng.uniform(size=(12, 12, 10))
        for target in ((12,12,10),(6,6,10),(3,3,10),(1,1,10)):
            b = conservative_block_sum(a, target)
            self.assertEqual(b.shape, target)
            self.assertAlmostEqual(float(b.sum()), float(a.sum()), places=11)

    def test_conservative_coarsening_from_two_to_one_matches_direct(self):
        a = np.arange(1440, dtype=float).reshape((12,12,10))
        one_step = conservative_block_sum(a,(1,1,10))
        two_step = conservative_block_sum(
            conservative_block_sum(a,(6,6,10)),(1,1,10))
        np.testing.assert_allclose(one_step, two_step, rtol=0, atol=1e-12)

    def test_between_seed_spatial_difference_respects_coarsening(self):
        a = example_uniform()
        a[0, :, :, :, :] = 0.
        a[0, :, 0, 0, 2] = 3.
        a[1, :, :, :, :] = 0.
        a[1, :, 1, 1, 2] = 3.
        report = measure_3d_heating_stability(a)
        fine = report["scale_sensitivity"][0]["between_seed_by_window"][0]
        coarse = report["scale_sensitivity"][1]["between_seed_by_window"][0]
        self.assertGreater(fine["all_six_pairwise_between_seed_TV"][0], 0.)
        self.assertLessEqual(coarse["all_six_pairwise_between_seed_TV"][0],
                             fine["all_six_pairwise_between_seed_TV"][0])

    def test_invalid_3d_probabilities_rejected(self):
        for invalid in (
            np.array([-1.,2.]),np.array([np.nan,1.]),np.array([1.,1.]),
            np.array([]),
        ):
            with self.subTest(bad=invalid), self.assertRaises(ValueError):
                total_variation(invalid, np.ones_like(invalid))

    def test_zero_or_negative_or_incomplete_spatial_data_rejected(self):
        for arr in (np.zeros((4,3,12,12,10)),
                    np.full((4,3,12,12,10),-1.),
                    np.full((4,3,12,12,10),np.nan),
                    np.ones((3,3,12,12,10))):
            with self.subTest(shape=arr.shape), self.assertRaises(ValueError):
                measure_3d_heating_stability(arr)

    def test_impossible_spatial_mesh_coarsening_rejected(self):
        with self.assertRaises(ValueError):
            conservative_block_sum(np.ones((12,12,10)),(5,6,5))
        with self.assertRaises(ValueError):
            conservative_block_sum(np.ones((12,12,10)),(4,4,7))

    def test_missing_genuine_original_stage34_report_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                audit_actual_3d_heating(d, Path(d)/"missing_stage34.json")

    def test_forged_stage34_physically_qualified_report_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d)/"fake.json"
            file.write_text(json.dumps({"schema":"fake", "genuine_openmc_statepoints_reverified":12}))
            with self.assertRaisesRegex(ValueError, "evidence invalid"):
                audit_actual_3d_heating(d,file)


class ActualOriginalOpenMC3DHeatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.getenv("NUKE_STAGE35_ACTUAL_OPENMC_12_HDF5")
        report = os.getenv("NUKE_STAGE35_ACTUAL_STAGE34_JSON")
        if not root or not report:
            raise RuntimeError("Stage35 needs all 12 actual OpenMC HDF5 and original Stage34 physical JSON")
        cls.result, cls.fields = audit_actual_3d_heating(root, report)

    def test_native_hdf5_original_and_all_voxel_totals_conserve(self):
        r, f = self.result, self.fields
        self.assertEqual(r["original_native_hdf5_checked"],12)
        self.assertTrue(r["independent_accepted_stage34_axial_heating_verified"])
        self.assertEqual(f["deposited_energy_ev_per_mc_start"].shape, (4,3,12,12,10))
        self.assertTrue(np.isfinite(f["deposited_energy_density_ev_per_mc_start_per_cm3"]).all())
        np.testing.assert_allclose(f["spatial_heating_probability"].sum(axis=(2,3,4)),
                                   1., rtol=0, atol=1e-12)

    def test_matched_3d_axial_multiscale_and_no_false_reactor_qualification(self):
        r=self.result
        m=r["spatial_heating_statistics"]
        self.assertEqual(len(m["scale_sensitivity"]), 4)
        self.assertGreaterEqual(m["axial_mean_early_late_TV"], 0)
        self.assertLessEqual(m["axial_mean_early_late_TV"], 1)
        for fine,coarse in zip(m["scale_sensitivity"],m["scale_sensitivity"][1:]):
            self.assertGreaterEqual(fine["mean_early_late_TV"]+1e-12,
                                    coarse["mean_early_late_TV"])
        self.assertFalse(m["native_independent_source_stationarity_qualified"])
        self.assertFalse(r["absolute_physical_reactor_watts_known"])


if __name__=="__main__":
    unittest.main(verbosity=2)
