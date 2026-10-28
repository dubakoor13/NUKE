"""Stage39 numerical tests plus mandatory native OpenMC replay evidence."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .real_coupled_identical_replay import (
    audit_real_replay,
    heating_replication_diagnostics,
    source_replication_diagnostics,
)


def source_records():
    k=np.arange(100,dtype=float)/10000.
    ent=np.arange(100,dtype=float)/200.
    out=[]
    for batch in (30,60,100):
        out.append({"seed":13007,"batch":batch,"k":k[:batch].copy(),
                    "entropy":ent[:batch].copy()})
    banks=[np.arange(30,dtype=float),np.arange(60,dtype=float),
           np.arange(100,dtype=float)]
    probs=[]
    for _ in range(3):
        a=np.zeros((4,4,10),dtype=float)
        a[0,0,0]=1.
        probs.append(a)
    return out,banks,probs


class IdenticalReplayMathTests(unittest.TestCase):
    def test_equal_real_source_histories_can_be_exact(self):
        a,b,h=source_records()
        result=source_replication_diagnostics(a,a,b,b,h,h)
        self.assertTrue(result[
            "original_executed_same_setting_k_entropy_source_banks_ALL_exact"])
        self.assertIsNone(result["first_generation_of_k_difference"])
        self.assertEqual(result["per_original_statepoint"][-1][
            "native_4x4x10_source_bank_shape_TV"],0.)

    def test_seed_equality_cannot_override_different_original_k(self):
        a,b,h=source_records()
        other=[dict(v,k=v["k"].copy()) for v in a]
        for v in other:
            v["k"][5]+=0.003
        result=source_replication_diagnostics(a,other,b,b,h,h)
        self.assertFalse(result[
            "original_executed_same_setting_k_entropy_source_banks_ALL_exact"])
        self.assertEqual(result["first_generation_of_k_difference"],6)

    def test_source_entropy_drift_detected_independently_of_k(self):
        a,b,h=source_records()
        other=[dict(v,entropy=v["entropy"].copy()) for v in a]
        for v in other:
            v["entropy"][2]+=0.01
        result=source_replication_diagnostics(a,other,b,b,h,h)
        self.assertFalse(result[
            "original_executed_same_setting_k_entropy_source_banks_ALL_exact"])
        self.assertIsNone(result["first_generation_of_k_difference"])
        self.assertFalse(result["per_original_statepoint"][-1][
            "native_generation_entropy_bit_identical"])

    def test_original_source_bank_mismatch_cannot_be_hidden_by_histogram(self):
        a,b,h=source_records()
        changed=[x.copy() for x in b]
        changed[2][0]+=3.
        result=source_replication_diagnostics(a,a,b,changed,h,h)
        self.assertFalse(result[
            "original_executed_same_setting_k_entropy_source_banks_ALL_exact"])
        self.assertEqual(result["per_original_statepoint"][-1][
            "native_4x4x10_source_bank_shape_TV"],0.)

    def test_original_generation_prefix_corruption_rejected(self):
        a,b,h=source_records()
        broken=[dict(v,k=v["k"].copy()) for v in a]
        broken[1]["k"][0]+=1.
        with self.assertRaisesRegex(ValueError,"prefix"):
            source_replication_diagnostics(a,broken,b,b,h,h)

    def test_original_seed_mismatch_rejected(self):
        a,b,h=source_records()
        wrong=[dict(v,seed=23011) for v in a]
        with self.assertRaisesRegex(ValueError,"seed"):
            source_replication_diagnostics(a,wrong,b,b,h,h)

    def test_wrong_checkpoint_count_rejected(self):
        a,b,h=source_records()
        with self.assertRaisesRegex(ValueError,"three native"):
            source_replication_diagnostics(a[:2],a,b,b,h,h)

    def test_identical_3D_heat_profiles_no_spatial_change(self):
        a=np.ones((3,12,12,10),dtype=float)
        b=2*a
        result=heating_replication_diagnostics(a,b)
        self.assertEqual(len(result["per_disjoint_active_window"]),3)
        for x in result["per_disjoint_active_window"]:
            for r in x["source_normalized_spatial_shape"]:
                self.assertEqual(
                    r["cross_identical_setup_replays_heating_shape_TV"],0.)
        self.assertFalse(result["reactor_thermal_power_watts_known"])

    def test_disjoint_spatial_heating_maximum_tv_and_coarsening(self):
        a=np.zeros((3,12,12,10))
        b=np.zeros_like(a)
        a[:,:, :, :] = 0
        a[:,0,0,4]=1.
        b[:,1,1,4]=1.
        result=heating_replication_diagnostics(a,b)
        for row in result["per_disjoint_active_window"]:
            shape=row["source_normalized_spatial_shape"]
            self.assertEqual(shape[0][
                "cross_identical_setup_replays_heating_shape_TV"],1.)
            self.assertTrue(all(x[
                "cross_identical_setup_replays_heating_shape_TV"]==0.
                for x in shape[1:]))

    def test_unphysical_heat_data_rejected(self):
        a=np.ones((3,12,12,10))
        for b in (np.zeros_like(a),np.full_like(a,np.nan),
                  np.full_like(a,-1.),np.ones((3,10,12,10))):
            with self.subTest(shape=b.shape),self.assertRaises(ValueError):
                heating_replication_diagnostics(a,b)

    def test_original_HDF5_cannot_be_replaced_by_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                audit_real_replay(Path(tmp)/"reference",Path(tmp)/"replay")


class ActualIdenticalOpenMCReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=os.getenv("NUKE_STAGE39_REAL_STAGE37_REFERENCE")
        new=os.getenv("NUKE_STAGE39_REAL_IDENTICAL_REPLAY")
        if not old or not new:
            raise RuntimeError("Stage39 requires four actual replay OpenMC runs plus Stage37 native HDF5")
        cls.results=audit_real_replay(old,new)

    def test_twenty_four_native_statepoints_and_reactor_input_fingerprints(self):
        r=self.results
        self.assertEqual(r["actual_native_HDF5_statepoints_reopened"],24)
        self.assertTrue(r["all_actual_reactor_geometry_materials_tallies_settings_data_identical"])
        self.assertTrue(r["all_actual_native_coupled_heat_energy_partition_closed"])
        self.assertEqual(len(r["per_actual_original_replay"]),4)
        for item in r["per_actual_original_replay"]:
            self.assertTrue(item["actual_original_and_replay_xml_all_identical"])
            self.assertEqual(len(item["native_replayed_fission_source_history"][
                "per_original_statepoint"]),3)

    def test_native_replay_source_and_heating_does_not_falsely_certify_hot_nerva(self):
        r=self.results
        self.assertEqual(r["new_identical_configuration_openmc_executions"],4)
        self.assertFalse(r["source_stationarity_and_neutron_history_independence_qualified"])
        self.assertFalse(r["full_hot_nasa_nerva_physical_core_or_reactor_watts_qualified"])
        for item in r["per_actual_original_replay"]:
            self.assertEqual(len(item[
                "native_replayed_coupled_deposited_energy"]["per_disjoint_active_window"]),3)


if __name__=="__main__":
    unittest.main(verbosity=2)
