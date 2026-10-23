"""Synthetic FX2-TH-inspired kinetics verification: NOT a NERVA benchmark."""
from __future__ import annotations
import math
import unittest
import numpy as np
from .fx2th_kinetics import (
    KineticsCoefficients, KineticsState, advance_kinetics,
    critical_precursors, kinetics_rhs, normalize_flux_shape,
    reconstruct_flux, solve_piecewise_constant,
)

def demo_coeffs(reactivity=0.0, source=0.0):
    # Fabricated numerical-test values, never historical/NERVA data.
    return KineticsCoefficients(
        0.1, (0.0010, 0.0015, 0.0012), (0.1, 0.7, 2.0),
        reactivity=reactivity, source_amplitude_per_s=source
    )

class FX2THKineticsTests(unittest.TestCase):
    def test_stationary_critical_amplitude_precursors(self):
        p = demo_coeffs()
        initial = KineticsState(0., 1.5, critical_precursors(1.5, p))
        step = advance_kinetics(initial, p, 10.0)
        self.assertAlmostEqual(step.state.amplitude, initial.amplitude, places=11)
        np.testing.assert_allclose(step.state.precursors, initial.precursors, rtol=1e-11)
        self.assertLess(step.amplitude_balance_residual, 1e-11)

    def test_rhs_critical_is_zero(self):
        p = demo_coeffs()
        a, delayed = kinetics_rhs(KineticsState(0., 1., critical_precursors(1., p)), p)
        self.assertAlmostEqual(a, 0.)
        np.testing.assert_allclose(delayed, 0, atol=1e-14)

    def test_prompt_only_refinement_against_analytic(self):
        p = KineticsCoefficients(1., (), (), reactivity=-0.2)
        s = KineticsState(0., 3., ())
        exact = 3 * math.exp(-0.2)
        coarse = advance_kinetics(s, p, 1., nominal_substeps=10).state.amplitude
        fine = advance_kinetics(s, p, 1., nominal_substeps=100).state.amplitude
        self.assertLess(abs(fine - exact), abs(coarse - exact) / 5)
        self.assertLess(abs(fine - exact), 1e-3)

    def test_steady_external_source(self):
        p = demo_coeffs(-0.002, 0.02)
        n = -p.source_amplitude_per_s * p.generation_time_s / p.reactivity
        s = KineticsState(0., n, critical_precursors(n, p))
        next_s = advance_kinetics(s, p, 100.).state
        self.assertAlmostEqual(next_s.amplitude, n, places=10)
        np.testing.assert_allclose(next_s.precursors, s.precursors, rtol=1e-10)

    def test_independent_matrix_exponential_eigen_reference(self):
        p = demo_coeffs(-0.0006)
        init = KineticsState(0., 1., critical_precursors(1., p))
        beta, lam = np.array(p.beta_groups), np.array(p.lambda_groups_s_inv)
        matrix = np.zeros((4,4))
        matrix[0,0] = (p.reactivity - p.beta_total)/p.generation_time_s
        matrix[0,1:] = lam
        matrix[1:,0] = beta/p.generation_time_s
        matrix[1:,1:] = -np.diag(lam)
        values, vectors = np.linalg.eig(matrix)
        exact = np.real_if_close(vectors @ (
            np.exp(values * 1.5) *
            np.linalg.solve(vectors, np.asarray((init.amplitude,)+init.precursors))
        ))
        coarse = advance_kinetics(init,p,1.5,nominal_substeps=20).state
        fine = advance_kinetics(init,p,1.5,nominal_substeps=400).state
        ec = np.linalg.norm(np.asarray((coarse.amplitude,)+coarse.precursors)-exact)
        ef = np.linalg.norm(np.asarray((fine.amplitude,)+fine.precursors)-exact)
        self.assertLess(ef, ec/10)
        self.assertLess(ef, 2e-5)

    def test_positive_reactivity_positivity_guard(self):
        p = KineticsCoefficients(0.02,(),(),reactivity=0.05)
        step = advance_kinetics(KineticsState(0.,1.,()),p,0.5)
        self.assertGreater(step.substeps,1)
        self.assertGreater(step.state.amplitude,1.)
        self.assertTrue(math.isfinite(step.state.amplitude))

    def test_bad_parameters_and_group_counts(self):
        with self.assertRaises(ValueError): KineticsCoefficients(0,(),())
        with self.assertRaises(ValueError): KineticsCoefficients(0.1,(0.8,0.3),(1.,2.))
        with self.assertRaises(ValueError): KineticsCoefficients(0.1,(0.2,),())
        with self.assertRaises(ValueError): KineticsCoefficients(0.1,(0.01,),(-2.,))
        with self.assertRaises(ValueError):
            advance_kinetics(KineticsState(0.,1.,()),demo_coeffs(),1.)

    def test_backwards_euler_equation_residuals(self):
        p=demo_coeffs(-0.001)
        s=KineticsState(0.,1.,critical_precursors(1.,p))
        step=advance_kinetics(s,p,0.7,nominal_substeps=8)
        self.assertLess(step.amplitude_balance_residual,1e-10)
        self.assertLess(step.max_precursor_balance_residual,1e-10)
        self.assertLess(step.state.amplitude,1.)

    def test_adjoint_shape_normalize_and_reconstruct(self):
        phi=np.array([[2.,3.],[4.,6.]])
        adj=np.array([[1.,2.],[1.5,1.]])
        result=normalize_flux_shape(phi,adj,[0.005,0.002],[2.,3.])
        self.assertAlmostEqual(result.weighted_normalization,1.,places=13)
        np.testing.assert_allclose(reconstruct_flux(result.amplitude,result),phi)

    def test_shape_gauge_scaling(self):
        phi=np.array([[1.,4.],[5.,3.]])
        a=normalize_flux_shape(phi,np.ones_like(phi),[1.,1.],[1.,1.])
        b=normalize_flux_shape(10*phi,np.ones_like(phi),[1.,1.],[1.,1.])
        np.testing.assert_allclose(a.shape,b.shape)
        self.assertAlmostEqual(b.amplitude,10*a.amplitude)

    def test_bad_shapes(self):
        with self.assertRaises(ValueError):
            normalize_flux_shape([[1.,2.]],[[0.,0.]],[1.,1.],[1.])
        with self.assertRaises(ValueError):
            normalize_flux_shape([[1.,2.]],[[1.,1.]],[1.],[1.])
        with self.assertRaises(ValueError):
            normalize_flux_shape([[1.,-2.]],[[1.,1.]],[1.,1.],[1.])

    def test_piecewise_reactivity_schedule(self):
        p=demo_coeffs()
        s=KineticsState(0.,1.,critical_precursors(1.,p))
        states=solve_piecewise_constant(s,[(1.,p),(2.,demo_coeffs(-0.001))],
                                        nominal_substeps=10)
        self.assertEqual([v.time_s for v in states],[0.,1.,3.])
        self.assertLess(states[-1].amplitude,states[1].amplitude)

if __name__ == "__main__":
    unittest.main(verbosity=2)
