"""Stage82: true curved-hole harmonic manufactured Fourier-flux tests.

The reference field solves Laplace(T)=0 in fuel exactly, since its
logarithmic singularity lies inside the excluded coated coolant hole.
Local open face integrals and central circular wall power are analytically
or independently Gaussian-quadrature derived.

These tests do NOT solve the manufactured discrete boundary-value
problem; they measure *operator flux consistency*, a necessary but not
sufficient condition for full spatial PDE convergence.
"""
from __future__ import annotations

import math
import unittest
import numpy as np

from .fuel_curved_harmonic_manufactured import (
    HarmonicHoleField,_line_material_intervals,
    evaluate_harmonic_curved_hole,run_harmonic_flux_refinement,
)
from .validate_stage81_fuel_interior import full_factory
from .validate_stage76_coupled_mesh_convergence import BASE
from .fuel_cutcell_conduction_2d import _line_fuel_length_cm
from .layout import coolant_channel_positions


class TrueCurvedHoleHarmonicFluxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study=run_harmonic_flux_refinement(full_factory)
        cls.model,_=full_factory(BASE)
        cfg=cls.model.config
        cls.field=HarmonicHoleField(
            hole_center_cm=coolant_channel_positions(cfg)[0],
            hole_radius_cm=cfg.coated_channel_radius_cm,
            amplitude_k=60.,conductivity_w_m_k=19.,
            axial_length_m=cfg.active_length_cm*.01)

    def test_source_free_harmonic_temperature_and_gradient_exact(self):
        field=self.field
        cx,cy=field.hole_center_cm
        r=field.hole_radius_cm
        points=np.array([[cx+1.5*r,cy],[cx,cy+2*r],
                         [cx+2*r,cy+2*r]])
        t=field.temperature(points)
        expected=field.wall_temperature_k+field.amplitude_k*np.log(
            np.linalg.norm(points-np.array([cx,cy]),axis=1)/r)
        np.testing.assert_allclose(t,expected,rtol=0,atol=1e-12)
        g=field.gradient_per_cm(points)
        delta=points-np.array([cx,cy])
        np.testing.assert_allclose(
            g,field.amplitude_k*delta/(np.sum(delta*delta,axis=1)[:,None]),
            atol=1e-12,rtol=2e-15)
        # Exact differential: div(grad log(r))=0 away from hole.
        squared=np.sum(delta*delta,axis=1)
        laplacian=field.amplitude_k*(2./squared-
                    2.*squared/(squared*squared))
        self.assertLess(float(np.max(np.abs(laplacian))),1e-9)

    def test_analytic_full_circle_flux_is_exact_2pi_k_L_A(self):
        field=self.field
        exact=field.exact_central_wall_power_w()
        self.assertAlmostEqual(
            exact,2*math.pi*19.*(self.model.config.active_length_cm*.01)*60.,
            delta=1e-10)
        q=field.exact_central_wall_flux_w_m2()
        expected_area=2*math.pi*field.hole_radius_cm*.01*field.axial_length_m
        self.assertAlmostEqual(q*expected_area,exact,delta=1e-10)
        self.assertGreater(q,0.)

    def test_independent_line_intersection_avoids_coolant_hole(self):
        cfg=self.model.config
        centers=coolant_channel_positions(cfg)
        ap=cfg.inner_fuel_flat_to_flat_cm*.5
        radius=cfg.coated_channel_radius_cm
        p=(-ap,0.)
        q=(ap,0.)
        intervals=_line_material_intervals(p,q,ap,centers,radius)
        self.assertGreater(len(intervals),2)
        total=sum((high-low)*(q[0]-p[0]) for low,high in intervals)
        reference=_line_fuel_length_cm(p,q,ap,centers,radius)
        self.assertAlmostEqual(total,reference,delta=1e-12)
        for a,b in intervals:
            x=p[0]+(q[0]-p[0])*.5*(a+b)
            self.assertTrue(all(
                math.hypot(x-cx,-cy)>=radius-1e-12
                for cx,cy in centers))

    def test_independently_masked_exact_open_face_areas_close_stage53(self):
        for observation in self.study.levels:
            self.assertGreater(observation.face_count,200)
            self.assertLess(
                observation.open_area_reference_max_discrepancy_m2,2e-12)
            self.assertEqual(
                observation.stage81_flux_w.shape,
                observation.exact_interior_flux_w.shape)
            self.assertTrue(np.all(np.isfinite(
                observation.exact_interior_flux_w)))

    def test_exact_central_wall_arc_area_closes_at_every_resolution(self):
        for observation in self.study.levels:
            self.assertLess(
                abs(observation.central_arc_area_residual_m2),2e-12)
            self.assertLess(
                observation.central_wall_reference_temperature_max_error_k,
                2e-12)
            self.assertAlmostEqual(
                observation.exact_central_hole_wall_power_w,
                self.field.exact_central_wall_power_w(),delta=1e-10)

    def test_stage81_internal_fv_faces_cancel_without_any_source(self):
        for observation in self.study.levels:
            self.assertLess(
                observation.stage81_internal_flux_global_cancellation_w,
                1e-7)
            self.assertTrue(np.all(np.isfinite(observation.stage81_flux_w)))
            self.assertTrue(np.all(np.isfinite(observation.stage53_flux_w)))

    def test_all_three_true_harmonic_mesh_errors_are_measured(self):
        study=self.study
        self.assertEqual(
            tuple(o.mesh_xy for o in study.levels),
            ((20,24),(28,32),(36,40)))
        self.assertTrue(
            study.levels[0].face_count<
            study.levels[1].face_count<
            study.levels[2].face_count)
        for o in study.levels:
            for error in (
                o.stage53_normalized_flux_l1_error,
                o.stage81_normalized_flux_l1_error,
                o.stage53_normalized_flux_l2_error,
                o.stage81_normalized_flux_l2_error,
                o.stage79_central_wall_relative_error,
                o.stage80_central_wall_relative_error,
            ):
                self.assertTrue(math.isfinite(error))
                self.assertGreaterEqual(error,0.)

    def test_no_spurious_claim_of_manufactured_PDE_solution(self):
        study=self.study
        self.assertIn("NOT",study.disclaimer)
        self.assertIn("not established",study.disclaimer)
        self.assertEqual(len(study.interior_flux_l1_improvement),2)
        self.assertEqual(len(study.wall_total_flux_errors),3)
        self.assertIsInstance(study.unresolved_nonmonotonic_wall_flux,bool)

    def test_invalid_hole_and_inside_singularity_rejected(self):
        f=self.field
        with self.assertRaisesRegex(ValueError,"finite physical"):
            HarmonicHoleField(
                hole_center_cm=(0.,0.),hole_radius_cm=0.,
                amplitude_k=60.,conductivity_w_m_k=19.,axial_length_m=1.)
        with self.assertRaisesRegex(ValueError,"inside excluded"):
            f.temperature(np.asarray([f.hole_center_cm]))
        with self.assertRaisesRegex(ValueError,"inside excluded"):
            f.gradient_per_cm(np.asarray([f.hole_center_cm]))
        with self.assertRaisesRegex(ValueError,"quadrature"):
            evaluate_harmonic_curved_hole(self.model,f,gauss_points=4)

    def test_manufactured_field_must_match_actual_central_hole(self):
        from dataclasses import replace
        with self.assertRaisesRegex(ValueError,"centered"):
            evaluate_harmonic_curved_hole(
                self.model,replace(self.field,hole_center_cm=(100.,0.)))


if __name__=="__main__":
    unittest.main(verbosity=2)
