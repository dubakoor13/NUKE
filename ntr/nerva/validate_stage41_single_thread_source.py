"""Stage41 one-OpenMP-thread original neutron/photon HDF5 physical QA.

Unit fixtures check numerical identity logic only. Real acceptance reopens
twelve ORIGINAL C++ OpenMC statepoints from six new actual transport runs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .real_coupled_single_thread_source_control import (
    audit_native_one_thread_experiment,
    verify_three_native_original_inputs,
)
from .real_coupled_three_arm_source_control import (
    compare_native_three_arm_paths,
    compare_original_late_heating,
)


def native_cases():
    # Synthetic fixtures are ONLY numerical unit tests, never physical HDF5.
    results=[]
    source=[]
    k=np.arange(100,dtype=float)*1e-4
    entropy=np.arange(100,dtype=float)*0.01
    for inactive in (20,20,40):
        pair=[]
        hist=[]
        for batch in (60,100):
            pair.append({
                "seed":13007,"batch":batch,"inactive":inactive,
                "k":k[:batch].copy(),"entropy":entropy[:batch].copy(),
                "source_bank":np.arange(3000,dtype=float),
            })
            p=np.zeros((4,4,10))
            p[0,0,0]=1.
            hist.append(p)
        results.append(pair)
        source.append(hist)
    return results,source


class SingleThreadControlMathTests(unittest.TestCase):
    def test_bit_identical_source_bank_control_accepts_equal_native_histories(self):
        a,h=native_cases()
        r=compare_native_three_arm_paths(*a,h)
        self.assertTrue(r["unchanged_replay_all_fission_bank_bytes_and_entropy_exact"])

    def test_small_k_roundoff_is_not_a_new_fission_source(self):
        a,h=native_cases()
        for v in a[1]:
            v["k"][0]+=1e-16
        r=compare_native_three_arm_paths(*a,h)
        self.assertTrue(r["unchanged_replay_all_fission_bank_bytes_and_entropy_exact"])

    def test_changed_inactive_can_change_native_source_without_control_failure(self):
        a,h=native_cases()
        for v in a[2]:
            v["k"][22]+=0.002
            v["entropy"][22]+=0.03
            v["source_bank"][0]+=1
        for x in h[2]:
            x[0,0,0]=0
            x[3,3,9]=1
        r=compare_native_three_arm_paths(*a,h)
        self.assertTrue(r["unchanged_replay_all_fission_bank_bytes_and_entropy_exact"])
        self.assertTrue(r["changed_inactive_bank_difference_observed"])
        self.assertEqual(r["per_native_checkpoint"][-1][
            "changed20_vs_40_first_material_k_difference_gt_1e_8"],23)
        self.assertEqual(r["per_native_checkpoint"][-1][
            "changed20_vs_40_spatial_fission_bank_TV"],1.)

    def test_spatial_heat_shape_disjoint_and_conservatively_normalized(self):
        a=np.ones((12,12,10))
        r=compare_original_late_heating([a,a,a*2])
        self.assertEqual(r["unchanged20_vs_20_final_heating_shape_TV"],0.)
        self.assertEqual(r["changed20_vs_40_final_heating_shape_TV"],0.)
        self.assertFalse(r["changed_40_inactive_heat_differences_are_causal_PROVEN"])

    def test_physical_heat_spatial_change_detected_separately(self):
        a=np.zeros((12,12,10))
        b=np.zeros_like(a)
        a[0,0,5]=1.
        b[6,6,5]=1.
        r=compare_original_late_heating([a,a,b])
        self.assertEqual(r["unchanged20_vs_20_final_heating_shape_TV"],0.)
        self.assertEqual(r["changed20_vs_40_final_heating_shape_TV"],1.)

    def test_invalid_inactive_mode_or_seed_rejected(self):
        a,h=native_cases()
        a[1][-1]["seed"]=23011
        with self.assertRaises(ValueError):
            compare_native_three_arm_paths(*a,h)

    def test_missing_real_native_xml_and_statepoints_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):
                verify_three_native_original_inputs(
                    [Path(t)/a/"legacy-broad"/"seed_13007"
                     for a in ("original20","repeat20","changed40")],
                    "legacy-broad")
            with self.assertRaises(FileNotFoundError):
                audit_native_one_thread_experiment(t)


class GenuineSingleThreadOpenMCPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.getenv("NUKE_STAGE41_REAL_SINGLE_THREAD_HDF5_ROOT")
        if not root:
            raise RuntimeError("Stage41 real native OpenMC 12-statepoint archive required")
        cls.report=audit_native_one_thread_experiment(root)

    def test_all_six_real_neutron_photon_executions_and_native_physical_inputs(self):
        r=self.report
        self.assertEqual(r["native_OpenMC_neutron_photon_histories_executed"],6)
        self.assertEqual(r["genuine_original_hdf5_checkpoints_60_100"],12)
        self.assertTrue(r["identical_transport_input_and_nuclear_data_fingerprints_verified"])
        self.assertEqual(len(r["source_and_heat_by_initial_source"]),2)
        for case in r["source_and_heat_by_initial_source"]:
            p=case["same_actual_reactor_evaluated_data_and_engine_provenance"]
            self.assertTrue(p["physical_geometry_materials_tally_XML_identical"])
            self.assertTrue(p["evaluated_nuclear_data_index_SHA_identical"])
            self.assertTrue(p["unchanged_20_and_repeat_20_settings_SHA_exact"])

    def test_native_source_and_energy_differences_are_descriptive_not_qualified(self):
        r=self.report
        self.assertFalse(r["one_thread_source_history_stationarity_qualified"])
        self.assertFalse(r["two_thread_to_single_thread_causal_RNG_mechanism_isolated"])
        self.assertFalse(r["real_hot_nasa_nerva_core_or_watts_qualified"])
        for case in r["source_and_heat_by_initial_source"]:
            src=case["native_source_trajectory"]
            self.assertEqual(len(src["per_native_checkpoint"]),2)
            h=case["native_photon_neutron_disjoint_heating"]
            self.assertEqual(h["per_disjoint_active_window"][0],61)
            self.assertEqual(h["per_disjoint_active_window"][1],100)
            self.assertTrue(np.isfinite(h["source_normalized_total_ev_per_mc_start"]).all())
            self.assertFalse(h["changed_40_inactive_heat_differences_are_causal_PROVEN"])


if __name__=="__main__":
    unittest.main(verbosity=2)
