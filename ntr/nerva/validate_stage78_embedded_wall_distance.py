"""Stage78 independent curved embedded-boundary fuel-contact tests.

Opt-in local half-depth geometry is a FIRST-ORDER approximation based
on each real hex-clipped fuel cell and the bore arc-normal. It replaces
one nonphysical global fuel wall length but DOES NOT prove grid
independence or exact 2D curved-interface reconstruction.

All material laws, nuclear W/m3 and initial conditions are SYNTHETIC.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from .coating_angular_interfaces import (
    local_fuel_wall_half_depth_m, angular_interface_power,
)
from .coating_radial_interfaces import radial_interface_power
from .coating_mesh_convergence import CoupledMeshResolution
from .fuel_cutcell_xy_convergence import run_fuel_xy_convergence
from .fuel_hydrogen_coupled_transient import (
    with_resolved_coating, advance_coupled_core,
)
from .hydrogen_transient_fvm import primitive
from .validate_stage76_coupled_mesh_convergence import (
    BASE, HEAT, physical_fixture,
)


GRIDS=((20,24),(28,32),(36,40))


def local_factory(mesh:CoupledMeshResolution):
    """Hold geometry, source distribution and material data fixed."""
    m,initial=physical_fixture(mesh)
    m=replace(
        m,coating=replace(
            m.coating,fuel_wall_distance_model="local_half_depth"))
    return m,initial


def global_factory(mesh:CoupledMeshResolution):
    return physical_fixture(mesh)


def uniform_interface_flux(model, *, fuel_k=505.,coating_k=455.,h2_k=395.):
    shape=model.radial_coating.sector_mass_kg.shape
    fuel=np.full((model.nzone,model.solid.transverse.n_fuel),fuel_k)
    coat=np.full(shape,coating_k)
    h2=np.full((model.nzone,19),h2_k)
    return radial_interface_power(
        model.coating,model.solid,model.properties,
        model.angular_contacts,model.radial_coating,
        fuel,coat,h2)


class EmbeddedCurvedWallDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local=run_fuel_xy_convergence(
            local_factory,base_mesh=BASE,heating=HEAT,
            xy_sizes=GRIDS)
        cls.legacy=run_fuel_xy_convergence(
            global_factory,base_mesh=BASE,heating=HEAT,
            xy_sizes=GRIDS)

    def test_exact_planar_embedded_half_depth_limit(self):
        # Fuel occupies the entire rectangle x in [0,2] cm, and
        # the wall is at x=0 cm with normal +x.
        polygon=[(0.,0.),(2.,0.),(2.,1.),(0.,1.)]
        self.assertAlmostEqual(
            local_fuel_wall_half_depth_m(
                polygon,(0.,.5),(1.,0.)),
            .01,delta=1e-14)
        self.assertAlmostEqual(
            local_fuel_wall_half_depth_m(
                polygon,(.25,.5),(1.,0.)),
            .00875,delta=1e-14)
        self.assertAlmostEqual(
            local_fuel_wall_half_depth_m(
                polygon,(1.,0.),(0.,1.)),
            .005,delta=1e-14)

    def test_nonphysical_cutface_or_nonunit_normal_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"physical clipped"):
            local_fuel_wall_half_depth_m(
                [],(0.,0.),(1.,0.))
        with self.assertRaisesRegex(ValueError,"unit"):
            local_fuel_wall_half_depth_m(
                [(0.,0.),(1.,0.),(1.,1.)],
                (0.,0.),(2.,0.))
        with self.assertRaisesRegex(ArithmeticError,"no physical"):
            local_fuel_wall_half_depth_m(
                [(0.,0.),(0.,1.),(-1.,1.),(-1.,0.)],
                (0.,.5),(1.,0.))

    def test_every_active_contact_has_finite_positive_geometric_depth(self):
        for nx,ny in GRIDS:
            m,_=local_factory(CoupledMeshResolution(
                nx,ny,BASE.axial_zones,
                BASE.angular_sectors,BASE.radial_layers))
            d=m.angular_contacts.fuel_wall_half_depth_m
            area=m.angular_contacts.fuel_outer_full_m2
            self.assertEqual(d.shape,area.shape)
            np.testing.assert_array_equal(d.indptr,area.indptr)
            np.testing.assert_array_equal(d.indices,area.indices)
            self.assertTrue(np.all(np.isfinite(d.data)))
            self.assertTrue(np.all(d.data>0.))
            self.assertGreater(float(np.ptp(d.data)),1e-7)
            self.assertEqual(d.nnz,area.nnz)

    def test_geometric_depth_depends_on_face_not_one_global_half_dx(self):
        m,_=local_factory(BASE)
        d=m.angular_contacts.fuel_wall_half_depth_m.data
        old=m.coating.half_fuel_distance_m
        self.assertGreater(float(np.ptp(d)),old*.2)
        self.assertFalse(np.allclose(d,old,rtol=1e-7,atol=1e-10))

    def test_identical_physical_contact_area_and_source_on_all_six_runs(self):
        for l,o in zip(self.local.cases,self.legacy.cases):
            self.assertEqual(l.mesh,o.mesh)
            self.assertAlmostEqual(
                l.observation.total_nuclear_energy_j,
                o.observation.total_nuclear_energy_j,delta=2e-13)
            np.testing.assert_allclose(
                l.fuel_coating_contact_area_by_channel_m2,
                o.fuel_coating_contact_area_by_channel_m2,
                rtol=0,atol=2e-13)
            self.assertAlmostEqual(
                l.fuel_volume_m3,o.fuel_volume_m3,delta=2e-14)

    def test_both_contact_policies_conserve_fuel_zrc_hydrogen_energy(self):
        for study in (self.local,self.legacy):
            self.assertLess(study.max_relative_energy_residual,5e-8)
            self.assertLess(study.max_nuclear_source_rebin_difference_j,2e-11)
            for case in study.cases:
                self.assertLess(
                    abs(case.observation.coating_first_law_residual_j),3e-7)

    def test_wall_flux_response_differs_under_local_embedded_geometry(self):
        model_a,_=local_factory(BASE)
        model_b,_=global_factory(BASE)
        a=uniform_interface_flux(model_a)
        b=uniform_interface_flux(model_b)
        self.assertGreater(float(a.fuel_to_coating_by_fuel_w.sum()),0.)
        self.assertGreater(float(b.fuel_to_coating_by_fuel_w.sum()),0.)
        self.assertNotAlmostEqual(
            float(a.fuel_to_coating_by_fuel_w.sum()),
            float(b.fuel_to_coating_by_fuel_w.sum()),
            delta=.1)
        self.assertLess(abs(a.fuel_antisymmetric_residual_w),1e-8)
        self.assertLess(abs(a.hydrogen_antisymmetric_residual_w),1e-8)
        np.testing.assert_allclose(
            a.fuel_to_coating_by_fuel_w.sum(axis=1),
            a.fuel_to_coating_by_radial_w.sum(axis=(1,2,3)),
            atol=1e-8,rtol=0)

    def test_both_refinement_sequences_measure_actual_local_wall_flux(self):
        for study in (self.local,self.legacy):
            for case in study.cases:
                measurements=case.observation.measurements
                self.assertTrue(all(math.isfinite(v)
                                    for v in measurements.values()))
                self.assertGreater(
                    measurements["fuel_outer_wall_peak_abs_w_m2"],0.)
                self.assertGreater(
                    measurements["coating_to_hydrogen_energy_j"],0.)
                self.assertGreater(case.observation.wall_flux_face_count,0)
            self.assertIn(
                "fuel_outer_wall_peak_abs_w_m2",study.metrics)

    def test_synthetic_reference_local_wall_refinement_improvement_measured(self):
        # Recorded empirical improvement is a REGRESSION on this exact
        # fixed synthetic benchmark, not evidence of continuum accuracy.
        flux="fuel_outer_wall_peak_abs_w_m2"
        energy="fuel_to_coating_energy_j"
        improved_flux=self.local.metrics[flux].relative_mid_to_fine
        legacy_flux=self.legacy.metrics[flux].relative_mid_to_fine
        improved_energy=self.local.metrics[energy].relative_mid_to_fine
        legacy_energy=self.legacy.metrics[energy].relative_mid_to_fine
        self.assertLess(improved_flux,legacy_flux)
        self.assertLess(improved_energy,legacy_energy)
        # Still not converged to 1% in the local wall flux.
        self.assertGreater(improved_flux,.01)
        self.assertIn(flux,self.local.metrics_requiring_further_refinement)
        # The per-cell source normalization and fuel inventory remain
        # identical across both physically equivalent models.
        self.assertLess(self.local.max_nuclear_source_rebin_difference_j,2e-11)

    def test_no_false_grid_independence_from_unchanged_peak_fuel_temp(self):
        # Actual observed wall flux differences are reported, never
        # replaced by the tiny peak-fuel-temperature difference.
        for study in (self.local,self.legacy):
            delta=study.metrics["fuel_outer_wall_peak_abs_w_m2"]
            self.assertGreaterEqual(delta.coarse_to_mid_abs,0.)
            self.assertGreaterEqual(delta.mid_to_fine_abs,0.)
            self.assertTrue(study.interpretation)
            self.assertTrue(
                "fuel_outer_wall_peak_abs_w_m2" in
                study.metrics_requiring_further_refinement or
                delta.relative_mid_to_fine<=0.01)


if __name__=="__main__":
    unittest.main(verbosity=2)
