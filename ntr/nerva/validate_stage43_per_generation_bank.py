"""Stage43 mathematical and ORIGINAL 56-source-bank HDF5 acceptance."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

from .real_original_per_generation_fission_bank import (
    BATCHES, audit_per_generation_original_sources,
    compare_bank_sequences,read_genuine_source_checkpoint,
)


def _source_fixture():
    out=[]
    for change in (False,True):
        d={}
        for batch in BATCHES:
            k=np.ones(batch)*0.012
            entropy=np.ones(batch)*1.
            source=np.arange(3000,dtype=float)
            if change and batch>=23:
                k[22:]+=0.002
                entropy[22:]+=0.2
                source[0]+=2.
            d[batch]={"k":k,"entropy":entropy,"source":source,
                      "source_bank_sha256":"a" if not change or batch<23 else "b"}
        out.append(d)
    return out


class PerGenerationNativeFissionMathTests(unittest.TestCase):
    def test_first_exact_native_source_bank_divergence_batch_23(self):
        a,b=_source_fixture()
        o=compare_bank_sequences(a,b)
        self.assertEqual(o["first_completed_batch_with_different_real_fission_bank"],23)
        self.assertEqual(o["first_completed_batch_with_materially_different_generation_k"],23)
        self.assertEqual(o["first_completed_batch_with_different_source_entropy"],23)
        self.assertTrue(o["per_original_checkpoint"][0]["entire_original_source_bank_bytes_equal"])

    def test_different_source_bank_can_exist_without_k_change(self):
        a,b=_source_fixture()
        for batch in BATCHES:
            b[batch]["k"]=a[batch]["k"].copy()
            b[batch]["entropy"]=a[batch]["entropy"].copy()
        r=compare_bank_sequences(a,b)
        self.assertEqual(r["first_completed_batch_with_different_real_fission_bank"],23)
        self.assertIsNone(r["first_completed_batch_with_materially_different_generation_k"])
        self.assertIsNone(r["first_completed_batch_with_different_source_entropy"])

    def test_k_difference_below_numerical_threshold_not_material(self):
        a,b=_source_fixture()
        for batch in BATCHES:
            b[batch]["k"]=a[batch]["k"]+1e-16
            b[batch]["source"]=a[batch]["source"].copy()
            b[batch]["entropy"]=a[batch]["entropy"].copy()
        r=compare_bank_sequences(a,b)
        self.assertIsNone(r["first_completed_batch_with_different_real_fission_bank"])
        self.assertIsNone(r["first_completed_batch_with_materially_different_generation_k"])

    def test_missing_intermediate_checkpoint_cannot_claim_source_divergence(self):
        a,b=_source_fixture()
        del b[23]
        with self.assertRaisesRegex(ValueError,"all native"):
            compare_bank_sequences(a,b)

    def test_no_original_statepoint_or_provenance_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                read_genuine_source_checkpoint(Path(d)/"not-real.h5",
                                                batch=19,inactive=40)
            with self.assertRaises(FileNotFoundError):
                audit_per_generation_original_sources(
                    Path(d)/"old",Path(d)/"new")


class ActualNativeOriginalFissionBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=os.getenv("NUKE_STAGE43_REAL_PREVIOUS_ORIGINAL_HDF5")
        new=os.getenv("NUKE_STAGE43_REAL_SAVED_EVERY_GENERATION")
        if not old or not new:
            raise RuntimeError("56 original OpenMC native fission-source StatePoints REQUIRED")
        cls.r=audit_per_generation_original_sources(old,new)

    def test_real_56_native_neutron_photon_statepoints_and_prefixes(self):
        r=self.r
        self.assertEqual(r["original_new_openmc_Cplusplus_HDF5_checkpoints"],56)
        self.assertTrue(r["all_extra_HDF5_writes_proven_nonintrusive_against_original"])
        self.assertTrue(r["full_original_core_material_and_nuclear_data_equivalence"])
        self.assertEqual(len(r["source_initialization_per_mode"]),2)
        for p in r["source_initialization_per_mode"]:
            self.assertEqual(len(p["per_generation_source_bank_comparison"][
                "per_original_checkpoint"]),14)

    def test_observed_first_source_change_does_not_certify_hot_reactor(self):
        r=self.r
        self.assertFalse(r["source_stationarity_statistically_established"])
        self.assertFalse(r["operating_NASA_NERVA_thermal_power_qualified"])
        for p in r["source_initialization_per_mode"]:
            rows=p["per_generation_source_bank_comparison"]["per_original_checkpoint"]
            self.assertTrue(all(0<=x[
                "maximum_k_generation_absolute_difference_so_far"]
                for x in rows))


if __name__=="__main__":
    unittest.main(verbosity=2)
