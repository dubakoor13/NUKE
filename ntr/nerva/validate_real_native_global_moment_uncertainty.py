"""Stage 27: moment formula/parser tests plus genuine 4-seed HDF5 QA.

Synthetic data are used ONLY to unit-test arithmetic and corrupt-moment
rejection; scientific result tests fail if original OpenMC evidence is absent.
"""
from __future__ import annotations

import io
import math
import os
from pathlib import Path
import unittest

import h5py
import numpy as np

from .real_native_global_moment_uncertainty import (
    audit_real_global_uncertainty, between_seed_descriptives,
    conditional_raw_moments, extract_native_global_moments,
)


def _fixture(*, two_bins=False, corrupt_score=False, bad_estimator=False):
    f = io.BytesIO()
    with h5py.File(f, "w") as h5:
        tallies = h5.create_group("tallies")
        for name, estimator, scores in [
            ("nerva_global_neutron_balance", "analog",
             ("absorption","scatter","nu-scatter","nu-fission")),
            ("nerva_global_reaction_tracklength", "tracklength",
             ("absorption","nu-fission")),
        ]:
            g=tallies.create_group("tally "+str(len(tallies)+1))
            g.create_dataset("name",data=np.bytes_(name))
            g.create_dataset("estimator",data=np.bytes_(
                "analog" if bad_estimator and name=="nerva_global_reaction_tracklength"
                else estimator))
            g.create_dataset("n_realizations",data=4)
            if corrupt_score and name=="nerva_global_neutron_balance":
                scores=tuple(x for x in scores if x!="nu-scatter")
            g.create_dataset("score_bins",data=np.array(
                [x.encode("utf-8") for x in scores]))
            shape=(2 if two_bins and name=="nerva_global_neutron_balance" else 1,
                   len(scores),2)
            vals=np.zeros(shape,dtype=float)
            vals[:,:,0]=4.
            vals[:,:,1]=4.
            g.create_dataset("results",data=vals)
        globals_=np.zeros((4,3),float)
        globals_[3,1:]=[4.,4.]
        h5.create_dataset("global_tallies",data=globals_)
    return f.getvalue()


class MomentMathematicsTests(unittest.TestCase):
    def test_openmc_native_sum_sum_sq_sem(self):
        out=conditional_raw_moments(10.,30.,4)
        self.assertEqual(out["mean_per_starting_neutron"],2.5)
        self.assertAlmostEqual(out["native_conditional_batch_sem_NOT_SOURCE_CORRECTED"],
                               math.sqrt(5./12.),places=13)

    def test_zero_variance_not_false_precision(self):
        out=conditional_raw_moments(4.,4.,4)
        self.assertEqual(out["native_conditional_batch_sem_NOT_SOURCE_CORRECTED"],0.)

    def test_impossible_second_moment_rejected(self):
        with self.assertRaisesRegex(ValueError,"impossible raw tally"):
            conditional_raw_moments(10.,1.,4)

    def test_nonfinite_or_single_batch_rejected(self):
        for args in ((10.,20.,1),(10.,float("nan"),4),
                     (float("inf"),10.,4)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                conditional_raw_moments(*args)

    def test_single_global_bins_parse_correctly(self):
        with h5py.File(io.BytesIO(_fixture()),"r") as h5:
            scores=extract_native_global_moments(h5,4)
        self.assertEqual(len(scores),7)
        self.assertEqual(scores["vacuum_neutron_leakage"]["mean_per_starting_neutron"],1.)
        self.assertEqual(scores["analog_nu_fission"]["mean_per_starting_neutron"],1.)

    def test_multibin_sum_sq_refused_without_covariance(self):
        with h5py.File(io.BytesIO(_fixture(two_bins=True)),"r") as h5:
            with self.assertRaisesRegex(ValueError,"one bin"):
                extract_native_global_moments(h5,4)

    def test_missing_score_or_wrong_estimator_rejected(self):
        for kw in ({"corrupt_score":True},{"bad_estimator":True}):
            with self.subTest(kwargs=kw):
                with h5py.File(io.BytesIO(_fixture(**kw)),"r") as h5:
                    with self.assertRaisesRegex(ValueError,"invalid native global tally"):
                        extract_native_global_moments(h5,4)

    def test_between_seed_variance_uses_paired_differences(self):
        seeds=[]
        for k in range(4):
            seeds.append({
                "seed":100+k,
                "moments":{
                    "analog_absorption":{
                        "mean_per_starting_neutron":float(k),
                        "native_conditional_batch_sem_NOT_SOURCE_CORRECTED":.1},
                    "tracklength_absorption":{
                        "mean_per_starting_neutron":float(k)+.02,
                        "native_conditional_batch_sem_NOT_SOURCE_CORRECTED":.1},
                }
            })
        output=between_seed_descriptives(seeds)
        paired=output["paired_tracklength_minus_analog_absorption"]
        self.assertAlmostEqual(paired["observed_paired_seed_mean_difference"],.02)
        self.assertLess(paired["observed_paired_seed_standard_deviation"],1e-15)
        self.assertFalse(paired["score_level_analog_tracklength_covariance_from_raw_marginals_available"])


class GenuineOriginalOpenMCGlobalMomentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=os.environ.get("NUKE_REAL_CORE_EVIDENCE_PATH")
        if not path or not Path(path).exists():
            raise RuntimeError("Original four-seed OpenMC HDF5 required; no mocked science data")
        cls.path=Path(path)

    def test_all_four_original_native_moment_estimators_and_population_identity(self):
        result=audit_real_global_uncertainty(self.path)
        self.assertEqual(result["original_neutron_only_cold_surrogate_seed_count"],4)
        self.assertEqual([x["seed"] for x in result["per_seed_native_global_moments"]],
                         [13007,23011,33013,43019])
        for row in result["per_seed_native_global_moments"]:
            self.assertEqual(row["active_batch_realizations"],90)
            self.assertAlmostEqual(row["observed_neutron_population_closure"],1.,places=10)
            self.assertEqual(len(row["moments"]),7)
            for score in row["moments"].values():
                self.assertTrue(math.isfinite(
                    score["native_conditional_batch_sem_NOT_SOURCE_CORRECTED"]))
                self.assertGreaterEqual(
                    score["native_conditional_batch_sem_NOT_SOURCE_CORRECTED"],0.)

    def test_observed_global_scores_and_no_hot_qualification(self):
        result=audit_real_global_uncertainty(self.path)
        d=result["between_seed_descriptive_statistics"]
        self.assertAlmostEqual(
            d["analog_absorption"]["observed_unweighted_seed_mean"],
            .039666666666666656,places=12)
        self.assertAlmostEqual(
            d["tracklength_absorption"]["observed_unweighted_seed_mean"],
            .03991528763875892,places=12)
        self.assertAlmostEqual(
            d["vacuum_neutron_leakage"]["observed_unweighted_seed_mean"],
            1.0238555555555557,places=12)
        self.assertFalse(result["mesh_bin_covariance_available"])
        self.assertFalse(result["independent_stationary_fission_source_qualified"])
        self.assertFalse(result["native_batch_conditional_sem_physically_qualified"])
        self.assertFalse(result["local_neutron_photon_heating_qualified"])
        self.assertFalse(result["absolute_reactor_watts_normalization_permitted"])


if __name__=="__main__":
    unittest.main(verbosity=2)
