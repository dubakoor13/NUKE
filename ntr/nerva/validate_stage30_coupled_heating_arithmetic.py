"""Synthetic arithmetic tests ONLY for Stage30 genuine HDF5 heating audit.

Never substitutes fake arrays for actual executable OpenMC transport.
"""
from __future__ import annotations

import unittest

from .real_coupled_photon_heating_audit import reconcile_energy_partitions


class PairedCoupledEnergyScoreTests(unittest.TestCase):
    def test_exact_neutron_plus_photon_material_mesh_mean_closure(self):
        r=reconcile_energy_partitions(14.,[2.,4.,8.],6.,8.,
                                      all_scores_share_estimator=True)
        self.assertTrue(r["same_domain_partition_means_close"])
        self.assertTrue(r["photon_energy_deposition_observed"])
        self.assertAlmostEqual(r["direct_photon_event_fraction_of_recorded_heating"],8./14.)

    def test_charged_secondaries_complete_physical_heating_balance(self):
        r=reconcile_energy_partitions(
            14.,[2.,4.,8.],10.,1.,electron_heating=2.5,
            positron_heating=.5,all_scores_share_estimator=True)
        self.assertTrue(r["same_domain_partition_means_close"])
        self.assertAlmostEqual(
            r["photon_and_secondary_charged_heating_ev_per_source"],4.)
        self.assertAlmostEqual(r["photon_plus_charged_fraction_of_recorded_heating"],4/14)

    def test_neutron_photon_only_misses_charged_energy_by_design(self):
        r=reconcile_energy_partitions(
            14.,[2.,4.,8.],10.,1.,all_scores_share_estimator=True)
        self.assertFalse(r["same_domain_partition_means_close"])
        self.assertAlmostEqual(r["mesh_vs_particle_relative_residual"],-3/14)

    def test_zero_photon_score_does_not_claim_observed_gamma_heating(self):
        r=reconcile_energy_partitions(14.,[7.,7.],14.,0.,
                                      all_scores_share_estimator=True)
        self.assertFalse(r["photon_energy_deposition_observed"])
        self.assertFalse(r["physical_reactor_watts_known"])

    def test_unmodeled_material_partition_reported_not_silently_renormalized(self):
        r=reconcile_energy_partitions(14.,[6.,6.],6.,8.,
                                      all_scores_share_estimator=True)
        self.assertFalse(r["same_domain_partition_means_close"])
        self.assertAlmostEqual(r["mesh_vs_material_relative_residual"],-2/14)

    def test_correlated_estimator_mismatch_not_claimed_identical(self):
        r=reconcile_energy_partitions(14.,[2.,4.,8.],6.,8.,
                                      all_scores_share_estimator=False)
        self.assertFalse(r["material_particle_mesh_estimator_consistent"])

    def test_nonfinite_or_negative_energy_rejected(self):
        for mesh,mats,n,p in (
            (float("nan"),[1.],1.,1.),(2.,[-1.,3.],1.,1.),
            (2.,[1.,1.],float("inf"),1.),(2.,[1.,1.],1.,-1.)):
            with self.subTest(mesh=mesh,mats=mats,n=n,p=p):
                with self.assertRaisesRegex(ValueError,"invalid/negative"):
                    reconcile_energy_partitions(mesh,mats,n,p,
                                               all_scores_share_estimator=True)

    def test_exact_zero_total_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"zero/negative"):
            reconcile_energy_partitions(0.,[0.],0.,0.,
                                       all_scores_share_estimator=True)

    def test_invalid_rel_tolerance_and_no_material_bins_refused(self):
        with self.assertRaises(ValueError):
            reconcile_energy_partitions(1.,[],1.,0.,
                                       all_scores_share_estimator=True)
        with self.assertRaises(ValueError):
            reconcile_energy_partitions(1.,[1.],1.,0.,
                                       all_scores_share_estimator=True,
                                       relative_tolerance=.2)


if __name__=="__main__":
    unittest.main(verbosity=2)
