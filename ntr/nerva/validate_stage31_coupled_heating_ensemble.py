"""Stage31 paired-covariance mathematics and mandatory original HDF5 checks.

Numerical unit fixtures are not neutron/photon transport.
Real-science assertions below MUST reopen all four original official OpenMC
statepoints from this exact CI execution; no synthetic substitute accepted.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import unittest

import numpy as np

from .real_coupled_photon_heating_ensemble import (
    audit_actual_four_seed_coupled_heating,
    seed_vector_covariance,
)


class PairedSeedStatisticsMathTests(unittest.TestCase):
    def test_correlated_bins_total_variance_zero_even_when_each_bin_varies(self):
        data=np.array([[1.,4.],[2.,3.],[3.,2.],[4.,1.]])
        report=seed_vector_covariance(data,label="test")
        self.assertGreater(report["between_seed_sample_standard_deviation"][0],0)
        self.assertGreater(report["between_seed_sample_standard_deviation"][1],0)
        self.assertEqual(report["total_between_seed_sample_sd"],0.)
        self.assertLess(report["paired_between_seed_sample_covariance"][0][1],0.)
        self.assertFalse(report["formal_confidence_coverage_established"])

    def test_perfectly_correlated_bins_joint_variance_is_larger(self):
        data=np.array([[1.,2.],[2.,4.],[3.,6.],[4.,8.]])
        report=seed_vector_covariance(data,label="test")
        c=np.asarray(report["paired_between_seed_sample_covariance"])
        self.assertAlmostEqual(report["total_between_seed_sample_sd"]**2,np.sum(c))
        self.assertGreater(c[0,1],0.)

    def test_cross_covariance_identity_four_seed_realizations(self):
        data=np.array([[1.,2.,1.],[2.,1.,4.],[4.,3.,2.],[2.,5.,2.]])
        report=seed_vector_covariance(data,label="test")
        observed=np.var(data.sum(axis=1),ddof=1)
        self.assertAlmostEqual(report["total_between_seed_sample_sd"]**2,observed)
        self.assertEqual(report["observed_covariance_rank_upper_bound"],3)
        self.assertTrue(report["native_within_seed_spatial_bin_SEMs_not_summed"])

    def test_sample_sem_is_descriptive_not_confidence_qualification(self):
        x=np.array([[5.],[8.],[4.],[7.]])
        out=seed_vector_covariance(x,label="test")
        self.assertAlmostEqual(out["total_descriptive_seed_mean_SEM_IF_STATIONARY"],
                               np.std(x[:,0],ddof=1)/2.)
        self.assertFalse(out["formal_confidence_coverage_established"])

    def test_nonfinite_or_negative_sample_rejected(self):
        for values in (np.array([[1.],[2.],[3.],[math.nan]]),
                       np.array([[1.],[2.],[-2.],[4.]])):
            with self.subTest(values=values),self.assertRaisesRegex(ValueError,"invalid original"):
                seed_vector_covariance(values,label="negative")

    def test_three_fake_replicas_cannot_substitute_for_four_original_seeds(self):
        with self.assertRaisesRegex(ValueError,"invalid original"):
            seed_vector_covariance(np.array([[1.],[2.],[3.]]),label="not enough")

    def test_empty_bins_are_invalid(self):
        with self.assertRaisesRegex(ValueError,"invalid original"):
            seed_vector_covariance(np.empty((4,0)),label="empty")

    def test_zero_scored_bins_allow_nonnegative_variance(self):
        x=np.array([[0.,0.],[0.,0.],[0.,0.],[0.,0.]])
        out=seed_vector_covariance(x,label="zero")
        self.assertEqual(out["total_between_seed_sample_sd"],0.)


class ActualOriginalCoupledOpenMCFourSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=os.environ.get("NUKE_STAGE31_REAL_COUPLED_SOURCE_ROOT")
        if not raw or not Path(raw).exists():
            raise RuntimeError("Four actual independently executed OpenMC coupled source HDF5 are mandatory")
        cls.report=audit_actual_four_seed_coupled_heating(Path(raw))

    def test_genuine_four_source_binaries_and_all_physical_energy_partitions(self):
        r=self.report
        self.assertEqual(r["real_native_coupled_hdf5_count"],4)
        self.assertEqual(r["original_independent_openmc_rng_seeds"],[13007,23011,33013,43019])
        self.assertTrue(r["all_actual_geometry_material_and_tally_XML_SHA_unchanged"])
        self.assertTrue(r["all_electron_positron_neutron_photon_energy_deposition_partitions_conserved"])
        for seed in r["per_actual_seed"]:
            self.assertLess(abs(seed["native_material_mesh_closure_relative"]),1e-9)
            self.assertLess(abs(seed["native_all_particle_mesh_closure_relative"]),1e-9)
            self.assertGreater(seed["native_particle_ev_per_source"]["electron"],0.)
            self.assertGreater(seed["native_particle_ev_per_source"]["neutron"],0.)

    def test_paired_cross_axial_bin_covariance_and_hot_qualification_blocked(self):
        r=self.report
        self.assertEqual(r["correlated_axial_heating_seed_statistics"]["bin_count"],10)
        self.assertEqual(r["correlated_particle_heating_seed_statistics"]["bin_count"],4)
        self.assertEqual(len(r["per_actual_seed"]),4)
        self.assertFalse(r["statistically_calibrated_fission_source_convergence"])
        self.assertFalse(r["real_neutron_photon_heating_hot_NERVA_physically_qualified"])
        self.assertFalse(r["independent_reactor_watt_normalization_exists"])
        self.assertFalse(r["nasa_historical_core_benchmark_validated"])
        mat=r["correlated_material_heating_seed_statistics"]
        self.assertLess(mat["max_absolute_covariance_vs_direct_total_variance_residual"],
                        max(1.,mat["total_between_seed_sample_sd"]**2)*1e-10)


if __name__=="__main__":
    unittest.main(verbosity=2)
