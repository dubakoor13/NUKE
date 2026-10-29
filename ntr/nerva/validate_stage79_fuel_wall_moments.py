"""Stage79 actual polygon-minus-hole centroid and curved-arc normal tests.

A geometric wall-normal centroid distance replaces the Stage78
half-of-maximal-polygon-extent proxy. This is still a two-point
fuel-to-wall heat-flux approximation with measurable tangential offset.
Exact nonorthogonal gradient reconstruction and independent curved-hole
heat-equation validation remain NOT done.

The coupled comparison uses synthetic heating and material properties.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from .fuel_embedded_boundary_moments import (
    polygon_area_moments_cm,disk_polygon_area_moments_cm,
    fuel_cutcell_volume_centroid_cm,
    circular_arc_centroid_and_normal_cm,
    fuel_wall_moment_normal_depth_m,
)
from .coating_mesh_convergence import CoupledMeshResolution
from .fuel_cutcell_xy_convergence import run_fuel_xy_convergence
from .coating_radial_interfaces import radial_interface_power
from .validate_stage78_embedded_wall_distance import (
    local_factory,global_factory,uniform_interface_flux,GRIDS,
)
from .validate_stage76_coupled_mesh_convergence import BASE,HEAT,physical_fixture


def moment_factory(mesh:CoupledMeshResolution):
    m,state=physical_fixture(mesh)
    m=replace(
        m,coating=replace(m.coating,
                          fuel_wall_distance_model="moment_normal_depth"))
    return m,state


class TrueEmbeddedFuelFirstMoments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study=run_fuel_xy_convergence(
            moment_factory,base_mesh=BASE,heating=HEAT,xy_sizes=GRIDS)

    def test_polygon_rectangle_centroid_from_signed_first_moments(self):
        poly=[(0.,0.),(2.,0.),(2.,1.),(0.,1.)]
        area,mx,my=polygon_area_moments_cm(poly)
        self.assertAlmostEqual(area,2.,delta=1e-14)
        self.assertAlmostEqual(mx/area,1.,delta=1e-14)
        self.assertAlmostEqual(my/area,.5,delta=1e-14)
        negative=list(reversed(poly))
        ma,mmx,mmy=polygon_area_moments_cm(negative)
        np.testing.assert_allclose([ma,mmx,mmy],
                                   [-area,-mx,-my],atol=1e-14,rtol=0)

    def test_full_disk_inside_polygon_has_zero_shifted_first_moments(self):
        r=1.
        poly=[(-2.,-2.),(2.,-2.),(2.,2.),(-2.,2.)]
        a,mx,my=disk_polygon_area_moments_cm(poly,(.2,-.1),r)
        self.assertAlmostEqual(a,math.pi,delta=5e-14)
        self.assertAlmostEqual(mx/a,.2,delta=5e-14)
        self.assertAlmostEqual(my/a,-.1,delta=5e-14)

    def test_quarter_disk_analytic_area_and_centroid(self):
        poly=[(0.,0.),(2.,0.),(2.,2.),(0.,2.)]
        a,mx,my=disk_polygon_area_moments_cm(poly,(0.,0.),1.)
        self.assertAlmostEqual(a,math.pi/4.,delta=2e-14)
        self.assertAlmostEqual(mx,1./3.,delta=2e-14)
        self.assertAlmostEqual(my,1./3.,delta=2e-14)
        self.assertAlmostEqual(mx/a,4./(3.*math.pi),delta=3e-14)

    def test_polygon_minus_half_disk_exact_centroid(self):
        poly=[(0.,-1.),(2.,-1.),(2.,1.),(0.,1.)]
        a,x,y=fuel_cutcell_volume_centroid_cm(poly,[(0.,0.)],1.)
        expected_area=4.-math.pi/2.
        # x-moment of right half unit disk is exactly 2/3 cm3.
        self.assertAlmostEqual(a,expected_area,delta=2e-14)
        self.assertAlmostEqual(x,(4.-2./3.)/expected_area,delta=2e-14)
        self.assertAlmostEqual(y,0.,delta=2e-14)

    def test_exact_circular_contact_arc_centroid_and_mean_normal(self):
        a=circular_arc_centroid_and_normal_cm(
            (0.,0.),1.,-math.pi/2,math.pi/2)
        self.assertAlmostEqual(a[0],math.pi,delta=1e-14)
        self.assertAlmostEqual(a[1],2./math.pi,delta=1e-14)
        self.assertAlmostEqual(a[2],0.,delta=1e-14)
        self.assertAlmostEqual(a[3],2./math.pi,delta=1e-14)
        self.assertAlmostEqual(a[4],0.,delta=1e-14)
        with self.assertRaisesRegex(ValueError,"invalid physical"):
            circular_arc_centroid_and_normal_cm((0.,0.),1.,1.,0.)

    def test_planar_limit_and_tangential_offset_are_independently_resolved(self):
        wall=(1.,0.,.5,1.,0.)
        norm,tang=fuel_wall_moment_normal_depth_m((1.5,.5),wall)
        self.assertAlmostEqual(norm,.015,delta=1e-14)
        self.assertAlmostEqual(tang,0.,delta=1e-14)
        norm,tang=fuel_wall_moment_normal_depth_m((1.5,1.5),wall)
        self.assertAlmostEqual(norm,.015,delta=1e-14)
        self.assertAlmostEqual(tang,.01,delta=1e-14)
        with self.assertRaisesRegex(ArithmeticError,"nonpositive"):
            fuel_wall_moment_normal_depth_m((-.1,.5),wall)

    def test_actual_all_channel_moment_geometry_has_positive_depth(self):
        for nx,ny in GRIDS:
            m,_=moment_factory(CoupledMeshResolution(
                nx,ny,BASE.axial_zones,
                BASE.angular_sectors,BASE.radial_layers))
            contacts=m.angular_contacts
            area=contacts.fuel_outer_full_m2
            for matrix in (
                contacts.fuel_wall_moment_depth_m,
                contacts.fuel_wall_tangential_offset_m,
            ):
                self.assertEqual(matrix.shape,area.shape)
                np.testing.assert_array_equal(matrix.indptr,area.indptr)
                np.testing.assert_array_equal(matrix.indices,area.indices)
                self.assertTrue(np.all(np.isfinite(matrix.data)))
            self.assertTrue(
                np.all(contacts.fuel_wall_moment_depth_m.data>0))
            self.assertGreater(float(
                np.ptp(contacts.fuel_wall_moment_depth_m.data)),1e-7)

    def test_real_nonorthogonality_is_not_hidden(self):
        m,_=moment_factory(BASE)
        d=m.angular_contacts.fuel_wall_moment_depth_m.data
        tangent=m.angular_contacts.fuel_wall_tangential_offset_m.data
        self.assertGreater(float(np.max(np.abs(tangent))),0.)
        self.assertGreater(float(np.ptp(d)),0.)
        self.assertGreater(float(np.ptp(tangent)),0.)

    def test_true_moment_wall_contact_is_antisymmetric(self):
        m,_=moment_factory(BASE)
        p=uniform_interface_flux(m)
        self.assertGreater(float(p.fuel_to_coating_by_fuel_w.sum()),0.)
        self.assertLess(abs(p.fuel_antisymmetric_residual_w),1e-8)
        self.assertLess(abs(p.hydrogen_antisymmetric_residual_w),1e-8)
        np.testing.assert_allclose(
            p.fuel_to_coating_by_fuel_w.sum(axis=1),
            p.fuel_to_coating_by_radial_w.sum(axis=(1,2,3)),
            rtol=0,atol=1e-8)

    def test_repeated_xy_sweeps_preserve_nuclear_source_and_ledger(self):
        study=self.study
        self.assertLess(study.max_relative_energy_residual,5e-8)
        self.assertLess(study.max_nuclear_source_rebin_difference_j,2e-11)
        self.assertLess(study.max_wall_contact_discrepancy_m2,4e-11)
        self.assertLess(study.max_fuel_inventory_discrepancy_m3,5e-12)
        self.assertEqual(len(study.cases),3)
        for case in study.cases:
            self.assertLess(abs(
                case.observation.coating_first_law_residual_j),4e-7)

    def test_observed_wall_flux_is_reported_without_convergence_claim(self):
        results=self.study
        for name in (
            "fuel_outer_wall_peak_abs_w_m2",
            "fuel_to_coating_energy_j",
            "coating_peak_k",
        ):
            d=results.metrics[name]
            self.assertGreaterEqual(d.mid_to_fine_abs,0.)
            self.assertTrue(d.interpretation)
        self.assertIn("NOT imply",results.interpretation)

    def test_nonphysical_material_moment_reference_is_rejected(self):
        poly=[(-1.,-1.),(1.,-1.),(1.,1.),(-1.,1.)]
        with self.assertRaisesRegex(ArithmeticError,"differs"):
            fuel_cutcell_volume_centroid_cm(
                poly,[],.1,exact_fuel_area_cm2=3.)
        with self.assertRaisesRegex(ValueError,"invalid physical"):
            circular_arc_centroid_and_normal_cm((0.,0.),-1.,0.,1.)


if __name__=="__main__":
    unittest.main(verbosity=2)
