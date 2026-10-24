"""Independent adaptive-step numerical accuracy tests.

All inputs are synthetic. These independently compare the Stage-13 controller
against full-state dense ODE matrix exponentials (including a constant-source
augmented system), not the controller's own coarse/fine solutions.
"""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
from scipy.linalg import expm

from .fx2th_adaptive_core_kinetics import (
    AdaptiveTolerances, advance_adaptive_core,
)
from .fx2th_family_spacetime import (
    SpatialFamilyState,stationary_precursors,
)
from .validate_fx2th_family_spacetime import (
    model,initial,dense_generator,
)
from .validate_fx2th_rz_family_diffusion import (
    grid,family_data,
)


def linear_oracle(d,s,T,source=None):
    y=np.r_[s.flux.reshape(-1),s.precursors.reshape(-1)]
    H=dense_generator(d)
    if source is None:
        return expm(T*H)@y
    # Independent augmented homogeneous matrix solution of
    #   dy/dt = H y + [V*q/M, 0], d1/dt=0.
    n=len(y)
    A=np.zeros((n+1,n+1),dtype=float)
    A[:n,:n]=H
    q=np.asarray(source)*np.asarray(d.volumes_m3)[:,None]
    A[:d.cells*d.groups,-1]=q.flatten()/d.mass_diagonal
    return (expm(T*A)@np.r_[y,1.])[:n]


def vector(state):
    return np.r_[state.flux.flatten(),state.precursors.flatten()]


class AdaptiveFamilyTests(unittest.TestCase):
    def test_critical_mode_stays_stationary_for_variable_steps(self):
        d=model(cells=1,groups=1)
        old=initial(d)
        result=advance_adaptive_core(d,old,.7,initial_dt_s=.015,
                    tolerances=AdaptiveTolerances(max_dt_s=.21))
        np.testing.assert_allclose(result.state.flux,old.flux,rtol=0,atol=2e-12)
        np.testing.assert_allclose(result.state.precursors,old.precursors,
                                   rtol=0,atol=2e-12)
        self.assertAlmostEqual(result.state.time_s,.7,places=14)
        self.assertEqual(result.rejected_steps,0)
        self.assertGreater(len(result.steps),1)

    def test_two_group_dense_matrix_exponential_reference(self):
        d=model();s=initial(d);T=.34
        result=advance_adaptive_core(d,s,T,initial_dt_s=.2,
             tolerances=AdaptiveTolerances(rtol=2e-4,atol_flux=1e-10,
                      atol_precursors=1e-10,max_dt_s=.25))
        exact=linear_oracle(d,s,T)
        self.assertLess(np.linalg.norm(vector(result.state)-exact)/
                        np.linalg.norm(exact),.002)
        self.assertLessEqual(result.max_local_error_accepted,1.)

    def test_tightening_error_tolerance_reduces_global_error(self):
        d=model();s=initial(d);T=.6
        exact=linear_oracle(d,s,T)
        loose=advance_adaptive_core(d,s,T,initial_dt_s=.4,
                        tolerances=AdaptiveTolerances(
                            rtol=.025,atol_flux=1e-7,
                            atol_precursors=1e-7,max_dt_s=.4))
        tight=advance_adaptive_core(d,s,T,initial_dt_s=.4,
                        tolerances=AdaptiveTolerances(
                            rtol=5e-5,atol_flux=1e-10,
                            atol_precursors=1e-10,max_dt_s=.4))
        err_l=np.linalg.norm(vector(loose.state)-exact)
        err_t=np.linalg.norm(vector(tight.state)-exact)
        self.assertLess(err_t,err_l*.45)
        self.assertGreater(len(tight.steps),len(loose.steps))

    def test_rejected_large_step_can_recover_and_reach_endpoint(self):
        d=model();s=initial(d)
        result=advance_adaptive_core(d,s,.30,initial_dt_s=.30,
               tolerances=AdaptiveTolerances(rtol=2e-5,
                     atol_flux=1e-10,atol_precursors=1e-10,
                     max_dt_s=.3,min_dt_s=1e-7))
        self.assertGreater(result.rejected_steps,0)
        self.assertAlmostEqual(result.state.time_s,.3,places=14)
        self.assertEqual(result.attempted_steps,
                         result.rejected_steps+len(result.steps))

    def test_each_accepted_step_passes_neutron_and_precursor_balance(self):
        d=model()
        r=advance_adaptive_core(d,initial(d),.35,initial_dt_s=.2,
              tolerances=AdaptiveTolerances(max_dt_s=.2))
        for segment in r.steps:
            self.assertLess(segment.first_half_neutron_balance_residual,1e-11)
            self.assertLess(segment.second_half_neutron_balance_residual,1e-11)
            self.assertLess(segment.first_half_precursor_residual,1e-11)
            self.assertLess(segment.second_half_precursor_residual,1e-11)
            self.assertLessEqual(segment.scaled_local_error,1.)

    def test_exact_total_time_and_consecutive_positive_dt(self):
        d=model()
        r=advance_adaptive_core(d,initial(d),.56789,initial_dt_s=.033,
              tolerances=AdaptiveTolerances(max_dt_s=.097))
        self.assertEqual(r.state.time_s,.56789)
        self.assertAlmostEqual(sum(x.dt_s for x in r.steps),.56789,places=12)
        self.assertTrue(all(x.dt_s>0 for x in r.steps))
        self.assertTrue(all(r.steps[i].time_s>r.steps[i-1].time_s
                            for i in range(1,len(r.steps))))

    def test_one_group_critical_with_constant_external_source_against_exact(self):
        d=model(cells=1,groups=1);s=initial(d);T=.25
        q=np.array([[.15]])
        r=advance_adaptive_core(d,s,T,initial_dt_s=.1,
                 tolerances=AdaptiveTolerances(rtol=1e-4,max_dt_s=.12),
                 external_source_per_m3=q)
        ref=linear_oracle(d,s,T,q)
        self.assertLess(np.linalg.norm(vector(r.state)-ref),.007)
        self.assertGreater(r.state.flux[0,0],s.flux[0,0])

    def test_zero_source_decay(self):
        d=model(cells=1,groups=1)
        d=replace(d,nu_sigma_f_inv_m=np.zeros((1,1)))
        s=SpatialFamilyState(0.,np.array([[3.]]),np.zeros((1,2)))
        r=advance_adaptive_core(d,s,.5,initial_dt_s=.2,
              tolerances=AdaptiveTolerances(rtol=1e-4,max_dt_s=.2))
        reference=3.*np.exp(-.8/(2./2.)*.5)
        self.assertAlmostEqual(r.state.flux[0,0],reference,delta=.025)
        np.testing.assert_allclose(r.state.precursors,0.,atol=0)

    def test_rz_two_group_spatial_precuror_matrix_exponential(self):
        g=grid(nr=2,nz=3,ng=2,boundary="reflecting",fission=.3)
        d=family_data(g)
        phi=np.ones((d.cells,d.groups))
        phi[0,0]=2.;phi[-1,1]=.3
        s=SpatialFamilyState(0.,phi,stationary_precursors(d,phi))
        r=advance_adaptive_core(d,s,.12,initial_dt_s=.08,
             tolerances=AdaptiveTolerances(rtol=1e-4,atol_flux=1e-11,
                 atol_precursors=1e-11,max_dt_s=.1))
        truth=linear_oracle(d,s,.12)
        error=np.linalg.norm(vector(r.state)-truth)/np.linalg.norm(truth)
        self.assertLess(error,.007)
        self.assertEqual(r.state.flux.shape,(6,2))
        self.assertEqual(r.state.precursors.shape,(6,2))

    def test_different_delayed_emission_spectra_are_preserved(self):
        d=model();s=initial(d)
        changed=replace(d,delayed_spectra_by_family=
            np.tile(np.asarray(d.prompt_spectrum)[:,None,:],(1,d.families,1)))
        cfg=AdaptiveTolerances(rtol=2e-3,max_dt_s=.1)
        first=advance_adaptive_core(d,s,.2,initial_dt_s=.08,tolerances=cfg)
        second=advance_adaptive_core(changed,s,.2,initial_dt_s=.08,tolerances=cfg)
        self.assertGreater(np.linalg.norm(first.state.flux-second.state.flux),1e-4)

    def test_invalid_source_and_initial_state_rejected(self):
        d=model();s=initial(d)
        for source in (np.ones((2,3)),np.array([[-1.,1.],[0.,0.]])):
            with self.subTest(source=source),self.assertRaises(ValueError):
                advance_adaptive_core(d,s,.1,initial_dt_s=.1,
                                      external_source_per_m3=source)
        with self.assertRaises(ValueError):
            advance_adaptive_core(d,SpatialFamilyState(0.,np.ones((1,2)),
                 s.precursors),.1,initial_dt_s=.1)

    def test_reject_invalid_step_and_endpoints(self):
        d=model();s=initial(d)
        for end,dt in ((0.,.1),(-1.,.1),(.1,0.),(.1,-1.),
                        (float("nan"),.1),(.1,float("inf"))):
            with self.subTest(end=end,dt=dt),self.assertRaises(ValueError):
                advance_adaptive_core(d,s,end,initial_dt_s=dt)

    def test_invalid_tolerances_rejected(self):
        for kwargs in ({"rtol":0.},{"atol_flux":0.},
                       {"atol_precursors":float("nan")},
                       {"min_dt_s":.2,"max_dt_s":.1},
                       {"max_steps":0},{"max_rejected":-1},
                       {"max_balance_residual":-1.}):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                AdaptiveTolerances(**kwargs)

    def test_min_dt_underresolved_fails_instead_of_accepting(self):
        d=model();s=initial(d)
        with self.assertRaisesRegex(RuntimeError,"min_dt_s"):
            advance_adaptive_core(d,s,.4,initial_dt_s=.4,
                tolerances=AdaptiveTolerances(rtol=1e-12,
                     atol_flux=1e-15,atol_precursors=1e-15,
                     min_dt_s=.2,max_dt_s=.4))

    def test_max_attempt_limit_enforced(self):
        d=model();s=initial(d)
        with self.assertRaisesRegex(RuntimeError,"maximum attempted"):
            advance_adaptive_core(d,s,.4,initial_dt_s=.02,
                 tolerances=AdaptiveTolerances(max_steps=1,max_dt_s=.02))

    def test_input_initial_arrays_are_not_modified(self):
        d=model();s=initial(d)
        oldphi=s.flux.copy();oldc=s.precursors.copy()
        advance_adaptive_core(d,s,.2,initial_dt_s=.1,
              tolerances=AdaptiveTolerances(max_dt_s=.1))
        np.testing.assert_array_equal(s.flux,oldphi)
        np.testing.assert_array_equal(s.precursors,oldc)

    def test_reproducible_with_same_initial_data(self):
        d=model();s=initial(d)
        kw=dict(initial_dt_s=.09,tolerances=AdaptiveTolerances(max_dt_s=.2))
        a=advance_adaptive_core(d,s,.25,**kw)
        b=advance_adaptive_core(d,s,.25,**kw)
        np.testing.assert_array_equal(a.state.flux,b.state.flux)
        np.testing.assert_array_equal(a.state.precursors,b.state.precursors)
        np.testing.assert_allclose([p.dt_s for p in a.steps],
                                   [p.dt_s for p in b.steps],rtol=0,atol=0)

    def test_source_provenance_preserved(self):
        d=model()
        result=advance_adaptive_core(d,initial(d),.1,initial_dt_s=.04)
        self.assertEqual(result.source_id,d.source_id)
        self.assertEqual(result.attempted_steps,len(result.steps)+result.rejected_steps)


if __name__=="__main__":
    unittest.main(verbosity=2)
