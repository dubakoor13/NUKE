"""Stage38 exact source-genealogy and native deposited-energy burn-in QA."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .real_coupled_burnin_sensitivity import (
    audit_real_inactive_threshold,
    compare_burnin_heating,
    recover_disjoint_sums,
    strict_native_identity,
    trajectory_identity,
)


def fake_checkpoint(seed: int, inactive: int, batch: int):
    # ONLY unit-test fixture; never accepted as genuine HDF5 evidence.
    rng = np.random.default_rng(19)
    return {
        "seed": seed, "inactive": inactive, "batch": batch,
        "k": rng.uniform(size=batch),
        "entropy": rng.uniform(size=batch),
        "source_bank": np.array([1.5,2.5,3.5]),
    }


def identical_trajectory():
    full=np.arange(100,dtype=float)/100
    entropy=np.arange(100,dtype=float)/30
    out=[]
    for inactive in (20,40):
        record=[]
        for batch in (60,100):
            record.append({
                "seed": 13007, "inactive": inactive, "batch": batch,
                "k": full[:batch].copy(),
                "entropy": entropy[:batch].copy(),
                "source_bank": np.array([2.,3.,4.]),
            })
        out.append(record)
    return out


class BurninMathTests(unittest.TestCase):
    def test_realization_weighted_disjoint_windows_exact(self):
        batch=np.array([1.25,2.,4.])
        first,late=recover_disjoint_sums((40*batch,40*batch+40*batch*2),(40,80))
        np.testing.assert_allclose(first,batch)
        np.testing.assert_allclose(late,2*batch)
        first_new,last_new=recover_disjoint_sums(
            (20*batch*3,20*batch*3+40*batch*2),(20,60))
        np.testing.assert_allclose(first_new,3*batch)
        np.testing.assert_allclose(last_new,2*batch)

    def test_cloned_monte_carlo_sources_must_be_verified_not_inferred(self):
        old,new=identical_trajectory()
        r=trajectory_identity(old,new)
        self.assertTrue(r["native_3000_fission_source_sites_bit_identical_at_both_checkpoints"])
        self.assertTrue(r["native_k_generation_bit_identical"])

    def test_identical_seed_but_different_generator_history_fails(self):
        old,new=identical_trajectory()
        new[-1]["k"][99]+=0.01
        with self.assertRaisesRegex(ValueError,"same seed"):
            trajectory_identity(old,new)

    def test_different_source_sites_with_same_k_entropy_fail(self):
        old,new=identical_trajectory()
        new[-1]["source_bank"][0]+=0.1
        with self.assertRaisesRegex(ValueError,"same seed"):
            trajectory_identity(old,new)

    def test_inconsistent_generation_prefix_within_original_HDF5_fails(self):
        old,new=identical_trajectory()
        old[0]["k"][0]+=1.
        new[0]["k"][0]+=1.
        with self.assertRaisesRegex(ValueError,"prefix"):
            trajectory_identity(old,new)

    def test_decreasing_heating_cumulative_native_tally_rejected(self):
        with self.assertRaisesRegex(ValueError,"non-monotonic"):
            recover_disjoint_sums((np.array([100.]),np.array([99.])),(40,80))

    def test_negative_nonfinite_or_wrong_realizations_rejected(self):
        for earlier,later in ((np.array([-1.]),np.array([10.])),
                              (np.array([np.nan]),np.array([10.]))):
            with self.assertRaises(ValueError):
                recover_disjoint_sums((earlier,later),(20,60))
        with self.assertRaisesRegex(ValueError,"count pair"):
            recover_disjoint_sums((np.ones(2),np.ones(2)),(30,80))

    def test_identical_heat_all_windows_no_tv_or_bias(self):
        a=np.ones((12,12,10),dtype=float)
        r=compare_burnin_heating(a,2*a,a,a)
        self.assertTrue(r["late_61_100_native_heating_reproduced_across_inactive_thresholds"])
        for lev in r["heating_shape_sensitivity_to_BurnIn_and_time_at_four_resolutions"]:
            self.assertEqual(
                lev["mean_heating_first_active_window_shape_TV_21_60_vs_41_60"],0.)

    def test_withholding_initial_20_batches_changes_spatial_heat_shape(self):
        baseline=np.zeros((12,12,10))
        late=np.zeros_like(baseline)
        baseline[0,0,0]=1.
        late[1,1,0]=1.
        r=compare_burnin_heating(baseline,late,late,late)
        self.assertEqual(
            r["heating_shape_sensitivity_to_BurnIn_and_time_at_four_resolutions"][0][
                "mean_heating_first_active_window_shape_TV_21_60_vs_41_60"],1.)
        self.assertEqual(
            r["heating_shape_sensitivity_to_BurnIn_and_time_at_four_resolutions"][-1][
                "mean_heating_first_active_window_shape_TV_21_60_vs_41_60"],0.)

    def test_unreproduced_identical_late_generation_heating_rejected(self):
        a=np.ones((12,12,10),dtype=float)
        inconsistent=a.copy()
        inconsistent[0,0,0]*=2
        with self.assertRaisesRegex(ValueError,"61-100"):
            compare_burnin_heating(a,a,a,inconsistent)

    def test_missing_original_openmc_hdf5_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/"statepoint.060.h5"
            with self.assertRaises(ValueError):
                strict_native_identity(file,seed=13007,inactive=20,batch=60)
            with self.assertRaises(FileNotFoundError):
                audit_real_inactive_threshold(
                    Path(folder)/"original",Path(folder)/"candidate")


class ActualOpenMCCoupledBurninTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=os.environ.get("NUKE_STAGE38_ORIGINAL_STAGE37_HDF5_ROOT")
        new=os.environ.get("NUKE_STAGE38_REAL_40_INACTIVE_HDF5_ROOT")
        if not old or not new:
            raise RuntimeError("Stage38 mandatory eight original 20+40 inactive source histories absent")
        cls.report=audit_real_inactive_threshold(old,new)

    def test_eight_native_sources_and_exact_k_entropy_source_bank_reproduction(self):
        report=self.report
        self.assertEqual(report["original_native_hdf5_count"],16)
        self.assertEqual(report["new_reexecuted_openmc_neutron_photon_histories"],4)
        self.assertTrue(report["same_actual_neutron_transport_trajectories_verified"])
        for run in report["per_original_run"]:
            proof=run["native_transport_genealogy_reproducibility"]
            self.assertTrue(proof["native_k_generation_bit_identical"])
            self.assertTrue(proof["native_shannon_entropy_bit_identical"])
            self.assertTrue(proof["native_3000_fission_source_sites_bit_identical_at_both_checkpoints"])

    def test_active_burnin_shift_and_energy_partition_conservation_not_watts(self):
        report=self.report
        self.assertTrue(report["genuine_native_material_mesh_particle_heating_closure"])
        for run in report["per_original_run"]:
            self.assertGreater(
                run["native_20_vs_40_inactive_deposited_energy"]["deposited_energy_61_100_ev_per_MC_source"],0.)
            self.assertTrue(run["native_20_vs_40_inactive_deposited_energy"][
                "late_61_100_native_heating_reproduced_across_inactive_thresholds"])
        self.assertFalse(report["burnin_20_vs_40_improves_fission_source_stationarity_PROVEN"])
        self.assertFalse(report["qualified_original_hot_nerva_reactor_power_watts"])


if __name__=="__main__":
    unittest.main(verbosity=2)
