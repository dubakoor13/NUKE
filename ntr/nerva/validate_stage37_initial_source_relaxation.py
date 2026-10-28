"""Stage37 numerical adversarial checks and mandatory genuine OpenMC HDF5 tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .real_coupled_initial_source_relaxation import (
    audit_original_initialization_experiment,
    disjoint_windows,
    numerical_comparison,
    _identity,
)


def fixture():
    source=np.full((2,4,3,4,4,10),1/160.,dtype=float)
    heat=np.full((2,4,3,12,12,10),1.,dtype=float)
    return source,heat


class InitialSourceRelaxationMathTests(unittest.TestCase):
    def test_exact_disjoint_cumulative_heating_window_means(self):
        x=np.concatenate((
            np.ones((10,4))*2,
            np.ones((30,4))*5,
            np.ones((40,4))*11,
        ))
        sums=[x[:n].sum(axis=0) for n in (10,40,80)]
        w=disjoint_windows(sums)
        np.testing.assert_allclose(w, np.array([
            [2]*4,[5]*4,[11]*4],dtype=float),atol=1e-12)

    def test_unequal_windows_must_not_be_equally_averaged(self):
        x=np.concatenate((
            np.ones((10,2))*2,
            np.ones((30,2))*5,
            np.ones((40,2))*11,
        ))
        sums=[x[:n].sum(axis=0) for n in (10,40,80)]
        m=disjoint_windows(sums)
        np.testing.assert_allclose((10*m[0]+30*m[1]+40*m[2])/80,sums[-1]/80)
        self.assertFalse(np.allclose(m.mean(axis=0),sums[-1]/80))

    def test_decreasing_original_cumulative_tallies_rejected(self):
        with self.assertRaisesRegex(ValueError,"nonmonotonic"):
            disjoint_windows([np.array([10.]),np.array([9.]),np.array([100.])])

    def test_wrong_realization_schedule_rejected(self):
        with self.assertRaisesRegex(ValueError,"requires genuine"):
            disjoint_windows([np.ones(2)]*3,counts=(10,25,45))
        with self.assertRaisesRegex(ValueError,"requires genuine"):
            disjoint_windows([np.ones(2)]*2)

    def test_nonfinite_or_negative_native_energy_rejected(self):
        for invalid in (np.array([-1.]),np.array([float("nan")])):
            with self.subTest(bad=invalid),self.assertRaises(ValueError):
                disjoint_windows([invalid]*3)

    def test_identical_initialization_shapes_yield_zero_spatial_TV(self):
        s,h=fixture()
        out=numerical_comparison(s,h)
        for row in out["per_case"]:
            self.assertEqual(row["mean_source_TV_batch30_to_100"],0.)
            self.assertEqual(row["mean_3d_heating_TV_active_21_30_to_61_100"],0.)
        for row in out["cross_initialization"]["source_bank_comparison_at_three_batch_checkpoints"]:
            self.assertEqual(row["cross_initialization_seed_average_source_TV"],0.)
        self.assertFalse(out["source_stationarity_proven"])

    def test_nonoverlapping_initial_source_banks_yield_unit_TV(self):
        s,h=fixture()
        s.fill(0.)
        s[0,:,: ,0,0,0]=1.
        s[1,:,: ,3,3,9]=1.
        out=numerical_comparison(s,h)
        for row in out["cross_initialization"]["source_bank_comparison_at_three_batch_checkpoints"]:
            self.assertEqual(row["cross_initialization_seed_average_source_TV"],1.)

    def test_larger_3D_heating_TV_disappears_on_coarsening(self):
        s,h=fixture()
        h.fill(0.)
        h[0,:,: ,0,0,5]=1.
        h[1,:,: ,1,1,5]=1.
        out=numerical_comparison(s,h)
        vals=out["cross_initialization"]["late_window_heating_shape_by_mesh_resolution"]
        self.assertEqual(vals[0]["cross_initialization_last_window_heating_shape_TV"],1.)
        self.assertTrue(all(v["cross_initialization_last_window_heating_shape_TV"]==0.
                            for v in vals[1:]))

    def test_temporal_source_drift_can_exist_without_heat_drift(self):
        s,h=fixture()
        s.fill(0.)
        s[:,:,0,0,0,0]=1.
        s[:,:,1,1,1,4]=1.
        s[:,:,2,3,3,9]=1.
        out=numerical_comparison(s,h)
        for row in out["per_case"]:
            self.assertEqual(row["mean_source_TV_batch30_to_100"],1.)
            self.assertEqual(row["mean_3d_heating_TV_active_21_30_to_61_100"],0.)

    def test_invalid_shape_probability_and_heat_are_rejected(self):
        s,h=fixture()
        for a,b in ((s[:1],h),(s,np.zeros_like(h)),
                    (np.zeros_like(s),h),(s,np.full_like(h,np.nan)),
                    (s,np.full_like(h,-1))):
            with self.subTest(shape=a.shape),self.assertRaises(ValueError):
                numerical_comparison(a,b)

    def test_native_missing_or_wrong_original_statepoints_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            missing=Path(directory)/"missing.h5"
            with self.assertRaises(ValueError):
                _identity(missing,seed=13007,batch=30)
            with self.assertRaises(FileNotFoundError):
                audit_original_initialization_experiment(directory)

    def test_forged_native_provenance_without_real_HDF5_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            first=Path(directory)/"legacy-broad"/"seed_13007"
            first.mkdir(parents=True)
            (first/"transport_provenance.json").write_text(json.dumps(
                {"config":{"initial_fission_source":"central-compact"}}))
            (first/"nuclear_data_report.json").write_text("{}")
            with self.assertRaises(ValueError):
                audit_original_initialization_experiment(directory)


class ActualOriginalOpenMCInitialSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.getenv("NUKE_STAGE37_REAL_EIGHT_OPENMC_ROOT")
        if not root:
            raise RuntimeError("Stage37 requires all 24 genuine coupled native HDF5 statepoints")
        cls.report,cls.arrays=audit_original_initialization_experiment(root)

    def test_eight_original_coupled_neutron_photon_runs_and_native_history(self):
        report=self.report
        self.assertEqual(report["genuine_coupled_native_HDF5_count"],24)
        self.assertEqual(report["real_independent_openmc_source_histories"],8)
        self.assertTrue(report["physical_reactor_geometry_material_tally_and_evaluated_data_identical"])
        self.assertEqual(self.arrays["native_source_probability_4x4x10"].shape,(2,4,3,4,4,10))
        self.assertEqual(self.arrays["native_coupled_heating_ev_per_start_12x12x10"].shape,(2,4,3,12,12,10))
        for cases in report["each_original_source_history"]:
            for case in cases:
                self.assertTrue(case["exact_k_generation_and_shannon_source_entropy_prefixes_verified"])

    def test_real_spatial_conservation_and_source_dependence_observed_not_certified(self):
        result=self.report["source_heating_diagnostics"]
        self.assertEqual(len(result["per_case"]),2)
        self.assertEqual(len(result["cross_initialization"]["source_bank_comparison_at_three_batch_checkpoints"]),3)
        self.assertTrue(all(0.<=row["mean_source_TV_batch30_to_100"]<=1.
                            for row in result["per_case"]))
        self.assertTrue(all(0.<=row["mean_3d_heating_TV_active_21_30_to_61_100"]<=1.
                            for row in result["per_case"]))
        self.assertFalse(result["source_stationarity_proven"])
        self.assertFalse(result["real_nerva_operating_power_or_temperature_qualified"])
        self.assertFalse(self.report["statistically_stationary_source_qualified"])


if __name__=="__main__":
    unittest.main(verbosity=2)
