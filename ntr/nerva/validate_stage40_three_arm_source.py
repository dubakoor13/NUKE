"""Stage40 three-arm genuine neutron source reproducibility and heating tests."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

from .real_coupled_three_arm_source_control import (
    audit_three_arm_native_controls,
    compare_native_three_arm_paths,
    compare_original_late_heating,
)


def three_arms():
    k=np.arange(100,dtype=float)/10000.
    e=np.arange(100,dtype=float)/25.
    observed=[]
    bins=[]
    for inactive in (20,20,40):
        arm=[]
        hist=[]
        for batch in (60,100):
            bank=np.arange(30,dtype=float)
            arm.append({
                "seed":13007,"batch":batch,"inactive":inactive,
                "k":k[:batch].copy(),"entropy":e[:batch].copy(),
                "source_bank":bank.copy(),
            })
            p=np.zeros((4,4,10))
            p[0,0,0]=1.
            hist.append(p)
        observed.append(arm)
        bins.append(hist)
    return observed,bins


class ThreeArmSourceControlMathTests(unittest.TestCase):
    def test_true_same_configuration_hdf5_source_control_is_exact(self):
        r,h=three_arms()
        a=compare_native_three_arm_paths(*r,h)
        self.assertTrue(a["unchanged_replay_all_fission_bank_bytes_and_entropy_exact"])
        self.assertFalse(a["changed_inactive_bank_difference_observed"])
        self.assertTrue(all(b[
            "unchanged20_vs_20_k_within_1e_14_absolute"]
            for b in a["per_native_checkpoint"]))

    def test_last_bit_k_scoring_noise_not_physical_source_divergence(self):
        r,h=three_arms()
        for b in r[1]:
            b["k"][2]+=1e-16
        a=compare_native_three_arm_paths(*r,h)
        self.assertTrue(a["unchanged_replay_all_fission_bank_bytes_and_entropy_exact"])
        self.assertIsNone(a["per_native_checkpoint"][-1][
            "unchanged20_vs_20_first_material_k_difference_gt_1e_8"])

    def test_changed_inactive_can_have_real_k_entropy_and_bank_divergence(self):
        r,h=three_arms()
        for b in r[2]:
            b["k"][22]+=0.002
            b["entropy"][22]+=1.
            b["source_bank"][0]=9.
        for p in h[2]:
            p[0,0,0]=0.
            p[1,1,4]=1.
        a=compare_native_three_arm_paths(*r,h)
        self.assertTrue(a["changed_inactive_bank_difference_observed"])
        self.assertEqual(a["per_native_checkpoint"][-1][
            "changed20_vs_40_first_material_k_difference_gt_1e_8"],23)
        self.assertEqual(a["per_native_checkpoint"][-1][
            "changed20_vs_40_first_entropy_difference"],23)
        self.assertEqual(a["per_native_checkpoint"][-1][
            "changed20_vs_40_spatial_fission_bank_TV"],1.)

    def test_same_seed_but_different_unchanged_replay_source_must_fail_control(self):
        r,h=three_arms()
        for b in r[1]:
            b["source_bank"][0]=12345.
        a=compare_native_three_arm_paths(*r,h)
        self.assertFalse(a["unchanged_replay_all_fission_bank_bytes_and_entropy_exact"])

    def test_k_float_difference_beyond_roundoff_fails_exact_replay_qualification(self):
        r,h=three_arms()
        for b in r[1]:
            b["k"][2]+=1e-6
        a=compare_native_three_arm_paths(*r,h)
        self.assertFalse(a["unchanged_replay_all_fission_bank_bytes_and_entropy_exact"])
        self.assertEqual(a["per_native_checkpoint"][-1][
            "unchanged20_vs_20_first_material_k_difference_gt_1e_8"],3)

    def test_corrupted_actual_inactive_count_rejected(self):
        r,h=three_arms()
        r[2][0]["inactive"]=20
        with self.assertRaisesRegex(ValueError,"20/20/40"):
            compare_native_three_arm_paths(*r,h)

    def test_missing_native_checkpoint_rejected(self):
        r,h=three_arms()
        with self.assertRaisesRegex(ValueError,"60/100"):
            compare_native_three_arm_paths(r[0][:1],r[1],r[2],h)

    def test_exact_same_heat_shape_is_zero_TV_even_if_amplitudes_differ(self):
        a=np.ones((12,12,10))
        d=compare_original_late_heating([a,a,2*a])
        self.assertEqual(d["unchanged20_vs_20_final_heating_shape_TV"],0.)
        self.assertEqual(d["changed20_vs_40_final_heating_shape_TV"],0.)
        self.assertFalse(d["changed_40_inactive_heat_differences_are_causal_PROVEN"])

    def test_changed_inactive_spatial_heat_shape_is_separately_measured(self):
        a=np.zeros((12,12,10))
        b=np.zeros_like(a)
        a[0,0,0]=1.
        b[2,2,0]=1.
        d=compare_original_late_heating([a,a,b])
        self.assertEqual(d["unchanged20_vs_20_final_heating_shape_TV"],0.)
        self.assertEqual(d["changed20_vs_40_final_heating_shape_TV"],1.)

    def test_nan_negative_or_empty_real_heating_not_accepted(self):
        a=np.ones((12,12,10))
        for invalid in (np.zeros_like(a),np.full_like(a,np.nan),-a):
            with self.assertRaises(ValueError):
                compare_original_late_heating([a,a,invalid])

    def test_real_openmc_HDF5_is_mandatory_and_cannot_use_fake_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                audit_three_arm_native_controls(Path(d)/"a",Path(d)/"b",Path(d)/"c")


class ActualOriginalOpenMCThreeArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ref=os.getenv("NUKE_STAGE40_GENUINE_STAGE37")
        repeat=os.getenv("NUKE_STAGE40_GENUINE_STAGE39")
        changed=os.getenv("NUKE_STAGE40_GENUINE_STAGE38")
        if not ref or not repeat or not changed:
            raise RuntimeError("Stage40 requires all three executed OpenMC original HDF5 sets")
        cls.r=audit_three_arm_native_controls(ref,repeat,changed)

    def test_actual_original_twenty_four_native_control_pairs_and_conservation(self):
        r=self.r
        self.assertEqual(r["actual_native_60_100_source_checkpoints_reopened"],24)
        self.assertTrue(r["all_original_native_neutron_photon_material_mesh_energy_partitions_conserved"])
        self.assertTrue(r["all_unchanged_source_bank_and_entropy_paths_repeat_exactly"])
        self.assertEqual(len(r["per_original_source_configuration"]),4)

    def test_changed_inactive_real_source_paths_diverge_beyond_unchanged_controls(self):
        r=self.r
        self.assertTrue(r["all_changed_inactive_source_banks_diverge_from_original"])
        self.assertEqual(r["average_unchanged_replay_final_source_bank_TV"],0.)
        self.assertGreater(r["average_changed_inactive_final_source_bank_TV"],0.)
        self.assertLess(r["average_unchanged_replay_late_3D_heat_TV"],1e-12)
        self.assertGreater(r["average_changed_inactive_late_3D_heat_TV"],0.)
        self.assertFalse(r["source_stationarity_and_inactive_bias_statistically_qualified"])
        self.assertFalse(r["hot_nasa_nerva_reactor_watts_qualified"])


if __name__=="__main__":
    unittest.main(verbosity=2)
