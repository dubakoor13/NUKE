"""Stage44 fixed-reactor OpenMC v0.15.3 inverse-k genealogy mechanism QA."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .real_original_openmc_k_feedback_mechanism import (
    BATCHES,
    audit_original_cpp_feedback,
    physical_source_feedback,
    _cpp_provenance,
    _native,
)


def fixture():
    source_dtype=np.dtype([("r",np.dtype([("x","<f8"),("y","<f8"),("z","<f8")])),
                           ("wgt","<f8")])
    baseline=np.zeros(3000,dtype=source_dtype)
    baseline["wgt"]=1.
    full=np.full(100,0.012,dtype=float)
    full[20]=0.011
    full[21]=0.013
    current=full.copy()
    current[22]+=0.0002
    pair=[]
    for inactive,k in ((20,full),(40,current)):
        data={}
        for b in BATCHES:
            bank=baseline.copy()
            if inactive==40 and b>=23:
                bank["r"]["x"][0]+=0.1
            data[b]={"source":bank,"k":k[:b].copy(),
                     "entropy":np.full(b,1.0)}
        pair.append(data)
    return pair


class InverseKSourceMechanismMathTests(unittest.TestCase):
    def test_active_mean_vs_inactive_current_k_predicts_next_fission_yield(self):
        a,b=fixture()
        r=physical_source_feedback(a,b)
        self.assertEqual(r["first_observed_original_source_bank_divergence_batch"],23)
        self.assertEqual(r["first_original_fission_bank_site_difference_1_based"],1)
        self.assertAlmostEqual(r[
            "k_feedback_for_generation23_after_20_inactive"],0.012)
        self.assertAlmostEqual(r[
            "k_feedback_for_generation23_after_40_inactive"],0.013)
        self.assertAlmostEqual(r[
            "relative_expected_fission_site_production_factor_active20_over_inactive40"],
            0.013/0.012)

    def test_exact_bank_equality_through22_is_mandatory(self):
        a,b=fixture()
        b[22]["source"]["r"]["x"][0]+=1.
        with self.assertRaisesRegex(ValueError,"agree before"):
            physical_source_feedback(a,b)

    def test_identical_source_at23_must_not_claim_divergence(self):
        a,b=fixture()
        b[23]["source"]=a[23]["source"].copy()
        with self.assertRaisesRegex(ValueError,"no observed"):
            physical_source_feedback(a,b)

    def test_tiny_k_roundoff_does_not_qualify_feedback_divergence(self):
        a,b=fixture()
        a[22]["k"][20]=0.013
        b[22]["k"][20]=0.013
        with self.assertRaisesRegex(ValueError,"did not diverge"):
            physical_source_feedback(a,b)

    def test_k_generation_prefix_must_match_original_cpp_hdf5(self):
        a,b=fixture()
        b[21]["k"][0]+=0.1
        with self.assertRaisesRegex(ValueError,"prefix"):
            physical_source_feedback(a,b)

    def test_missing_checkpoints_fail_closed(self):
        a,b=fixture()
        del b[22]
        with self.assertRaisesRegex(ValueError,"all 19"):
            physical_source_feedback(a,b)

    def test_fake_upstream_cpp_cannot_pass_pin(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"src"
            p.mkdir()
            (p/"physics.cpp").write_text("simulation::keff = 1.0;")
            (p/"eigenvalue.cpp").write_text("simulation::keff = 1.0;")
            with self.assertRaisesRegex(ValueError,"Git blob SHA"):
                _cpp_provenance(Path(folder))

    def test_original_OpenMC_not_replaced_by_missing_or_mock_hdf5(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                _native(Path(folder)/"fake.h5",inactive=20,batch=22)
            with self.assertRaises(FileNotFoundError):
                audit_original_cpp_feedback(
                    Path(folder)/"actual",Path(folder)/"upstream")


class ActualOpenMCOriginalKFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native=os.getenv("NUKE_STAGE44_REAL_STAGE43_OPENMC_ROOT")
        upstream=os.getenv("NUKE_STAGE44_OFFICIAL_V0153_CPP_ROOT")
        if not native or not upstream:
            raise RuntimeError("Stage44 requires original physical OpenMC HDF5 and pinned official C++")
        cls.r=audit_original_cpp_feedback(native,upstream)

    def test_original_source_banks_and_authentic_official_Cplusplus_input(self):
        r=self.r
        self.assertEqual(r["genuine_native_openmc_neutron_photon_hdf5_checked"],32)
        self.assertEqual(len(r["upstream_official_openmc_v0153_CPP_code_sha_verified"]),2)
        self.assertTrue(r["original_scientific_HDF5_first_source_difference_verified"])
        for case in r["original_source_modes"]:
            physics=case["source_feedback"]
            self.assertTrue(physics[
                "the_two_neutron_sources_match_exactly_through_batch22"])
            self.assertEqual(physics[
                "first_observed_original_source_bank_divergence_batch"],23)
            self.assertNotEqual(physics[
                "relative_expected_fission_site_production_factor_active20_over_inactive40"],
                1.)

    def test_source_numerics_not_stationarity_or_hot_NERVA_validation(self):
        r=self.r
        self.assertFalse(r["the_exact_stochastic_source_site_ancestry_Causally_proven"])
        self.assertFalse(r["hot_nasa_nerva_reactor_geometry_or_thermal_power_qualified"])


if __name__=="__main__":
    unittest.main(verbosity=2)
