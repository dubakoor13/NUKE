"""Synthetic independent verification of second-order spatial neutron kinetics.

Dense matrix exponential is independently assembled by Stage 7's coupled
ODE oracle, not the new sparse SDIRK2 generator or its timestep code.
Nothing in this file is a NERVA reactor transport validation.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest
import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from .fx2th_sdirk2_neutron_kinetics import (
    GAMMA, build_core_operator, advance_sdirk2,
    advance_sdirk2_adaptive, SDIRK2Tolerances,
)
from .fx2th_family_spacetime import (
    SpatialFamilyState,stationary_precursors,advance_spatial_families,
)
from .validate_fx2th_family_spacetime import model,initial,dense_generator
from .validate_fx2th_rz_family_diffusion import grid,family_data


def vec(state):
    return np.r_[state.flux.flatten(),state.precursors.flatten()]


def independent_exact(d,s,dt,source=None):
    y=vec(s)
    H=dense_generator(d)
    if source is None:
        return expm(dt*H)@y
    n=len(y)
    extended=np.zeros((n+1,n+1))
    extended[:n,:n]=H
    q=np.asarray(source)*np.asarray(d.volumes_m3)[:,None]
    extended[:d.cells*d.groups,-1]=q.flatten()/d.mass_diagonal
    return (expm(dt*extended)@np.r_[y,1.])[:-1]


class SDIRK2Tests(unittest.TestCase):
    def test_generator_matches_independent_full_ode(self):
        for d in (model(), family_data(grid(nr=2,nz=3,ng=2,boundary="zero_flux"))):
            with self.subTest(cells=d.cells):
                op=build_core_operator(d)
                np.testing.assert_allclose(op.generator.toarray(),dense_generator(d),
                    rtol=3e-14,atol=3e-14)

    def test_critical_single_cell_stationary(self):
        d=model(cells=1,groups=1);s=initial(d);op=build_core_operator(d)
        for dt in (.001,.1,.9):
            step=advance_sdirk2(op,d,s,dt)
            np.testing.assert_allclose(step.state.flux,s.flux,atol=2e-12)
            np.testing.assert_allclose(step.state.precursors,s.precursors,atol=2e-12)
            self.assertLess(step.global_balance_relative_residual,1e-12)

    def test_second_order_time_convergence(self):
        d=model();s=initial(d);op=build_core_operator(d);exact=independent_exact(d,s,.4)
        errors=[]
        for n in (4,8,16):
            state=s
            for _ in range(n):
                state=advance_sdirk2(op,d,state,.4/n).state
            errors.append(np.linalg.norm(vec(state)-exact))
        self.assertGreater(errors[0],0)
        self.assertLess(errors[1],errors[0]/3.0)
        self.assertLess(errors[2],errors[1]/3.0)

    def test_more_accurate_than_backward_euler_at_same_h(self):
        d=model();s=initial(d);op=build_core_operator(d)
        exact=independent_exact(d,s,.4)
        second=first=s
        for _ in range(4):
            second=advance_sdirk2(op,d,second,.1).state
            first=advance_spatial_families(d,first,.1).state
        self.assertLess(np.linalg.norm(vec(second)-exact),
                        .35*np.linalg.norm(vec(first)-exact))

    def test_nonzero_source_compares_with_augmented_matrix_exponential(self):
        d=model(cells=1,groups=1);s=initial(d)
        q=np.array([[.25]]);op=build_core_operator(d,q)
        errors=[]
        for n in (8,32):
            state=s
            for _ in range(n):
                state=advance_sdirk2(op,d,state,.3/n).state
            errors.append(np.linalg.norm(vec(state)-independent_exact(d,s,.3,q)))
        self.assertLess(errors[1],errors[0]/10)

    def test_population_and_stage_conservation(self):
        d=model();s=initial(d)
        op=build_core_operator(d,np.array([[.2,.1],[.1,.3]]))
        step=advance_sdirk2(op,d,s,.05)
        self.assertLess(step.stage1_relative_residual,1e-11)
        self.assertLess(step.stage2_relative_residual,1e-11)
        self.assertLess(step.global_balance_relative_residual,1e-11)
        self.assertGreater(step.neutron_inventory,0)
        self.assertGreater(step.precursor_inventory,0)

    def test_rz_multigroup_spatial_transient(self):
        d=family_data(grid(nr=2,nz=3,ng=2,boundary="reflecting"))
        phi=np.ones((d.cells,d.groups));phi[0,0]=3.;phi[-1,1]=.25
        s=SpatialFamilyState(0.,phi,stationary_precursors(d,phi))
        op=build_core_operator(d);state=s
        for _ in range(12):
            state=advance_sdirk2(op,d,state,.01).state
        self.assertLess(np.linalg.norm(vec(state)-independent_exact(d,s,.12))/
                        np.linalg.norm(independent_exact(d,s,.12)),.001)

    def test_delayed_emission_spectra_affect_flux(self):
        d=model();s=initial(d)
        a=advance_sdirk2(build_core_operator(d),d,s,.15).state
        other=replace(d,delayed_spectra_by_family=
                     np.tile(d.prompt_spectrum[:,None,:],(1,d.families,1)))
        b=advance_sdirk2(build_core_operator(other),other,s,.15).state
        self.assertGreater(np.linalg.norm(a.flux-b.flux),1e-5)

    def test_stiff_scalar_l_stability_and_nonnegativity_guard(self):
        z=-1e6
        amplification=(1.+(1.-2.*GAMMA)*z)/(1.-GAMMA*z)**2
        self.assertLess(abs(amplification),1e-5)
        d=model(cells=1,groups=1)
        d=replace(d,integrated_loss=csc_matrix([[25.]]),
                  nu_sigma_f_inv_m=np.array([[0.]]))
        s=SpatialFamilyState(0.,np.array([[1.]]),np.zeros((1,2)))
        with self.assertRaisesRegex(ArithmeticError,"negative"):
            advance_sdirk2(build_core_operator(d),d,s,1.)

    def test_adaptive_stiff_rejection_and_recovery(self):
        d=model(cells=1,groups=1)
        d=replace(d,integrated_loss=csc_matrix([[25.]]),
                  nu_sigma_f_inv_m=np.array([[0.]]))
        s=SpatialFamilyState(0.,np.array([[1.]]),np.zeros((1,2)))
        r=advance_sdirk2_adaptive(build_core_operator(d),d,s,.15,initial_dt_s=.15,
            tolerances=SDIRK2Tolerances(relative=3e-4,
                    flux_absolute=1e-10,precursor_absolute=1e-10,max_dt_s=.2))
        self.assertGreater(r.rejected_steps,0)
        self.assertTrue(np.all(r.state.flux>=0))
        self.assertAlmostEqual(r.state.time_s,.15,places=13)

    def test_tolerance_tightening_improves_true_error(self):
        d=model();s=initial(d);op=build_core_operator(d)
        exact=independent_exact(d,s,.5)
        loose=advance_sdirk2_adaptive(op,d,s,.5,initial_dt_s=.4,
            tolerances=SDIRK2Tolerances(relative=.1,max_dt_s=.4))
        tight=advance_sdirk2_adaptive(op,d,s,.5,initial_dt_s=.4,
            tolerances=SDIRK2Tolerances(relative=4e-6,flux_absolute=1e-11,
                    precursor_absolute=1e-11,max_dt_s=.4))
        self.assertLess(np.linalg.norm(vec(tight.state)-exact),
                        .3*np.linalg.norm(vec(loose.state)-exact))
        self.assertGreater(len(tight.steps),len(loose.steps))
        self.assertLessEqual(tight.max_accepted_estimated_error,1.)

    def test_adaptive_cylindrical_and_balances(self):
        d=family_data(grid(nr=2,nz=2,ng=2))
        phi=np.ones((d.cells,d.groups))
        s=SpatialFamilyState(0.,phi,stationary_precursors(d,phi))
        r=advance_sdirk2_adaptive(build_core_operator(d),d,s,.25,initial_dt_s=.12,
            tolerances=SDIRK2Tolerances(relative=1e-4,max_dt_s=.15))
        self.assertLess(np.linalg.norm(vec(r.state)-independent_exact(d,s,.25))/
                        np.linalg.norm(independent_exact(d,s,.25)),.002)
        self.assertAlmostEqual(sum(x.duration_s for x in r.steps),.25,places=12)
        for segment in r.steps:
            self.assertLess(segment.global_balance_relative_residual,1e-11)
            self.assertLessEqual(segment.estimated_local_error,1.)
        self.assertEqual(r.attempted_steps,r.rejected_steps+len(r.steps))
        self.assertEqual(r.implicit_stage_solves,6*r.attempted_steps)

    def test_repeated_runs_are_deterministic(self):
        d=model();s=initial(d);op=build_core_operator(d)
        kw=dict(initial_dt_s=.11,tolerances=SDIRK2Tolerances(max_dt_s=.15))
        a=advance_sdirk2_adaptive(op,d,s,.35,**kw)
        b=advance_sdirk2_adaptive(op,d,s,.35,**kw)
        np.testing.assert_array_equal(a.state.flux,b.state.flux)
        np.testing.assert_array_equal(a.state.precursors,b.state.precursors)
        np.testing.assert_array_equal([x.duration_s for x in a.steps],
                                      [x.duration_s for x in b.steps])

    def test_bad_source_and_params_are_rejected(self):
        d=model()
        for q in (np.ones((2,3)),np.array([[-1.,1.],[1.,1.]])):
            with self.subTest(q=q),self.assertRaises(ValueError):
                build_core_operator(d,q)
        for kw in ({"relative":0.},{"flux_absolute":0.},
                    {"min_dt_s":.4,"max_dt_s":.2},{"max_attempts":0}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                SDIRK2Tolerances(**kw)

    def test_mismatched_times_and_min_dt_fail_closed(self):
        d=model();s=initial(d);op=build_core_operator(d)
        with self.assertRaises(ValueError):
            advance_sdirk2(op,d,SpatialFamilyState(-1.,s.flux,s.precursors),.1)
        with self.assertRaises(ValueError):
            advance_sdirk2_adaptive(op,d,s,1e-16,initial_dt_s=1e-16)
        with self.assertRaisesRegex(RuntimeError,"minimum"):
            advance_sdirk2_adaptive(op,d,s,.3,initial_dt_s=.3,
                tolerances=SDIRK2Tolerances(relative=1e-14,
                    flux_absolute=1e-16,precursor_absolute=1e-16,
                    min_dt_s=.2,max_dt_s=.3))

    def test_attempt_budget_enforced(self):
        d=model();s=initial(d);op=build_core_operator(d)
        with self.assertRaisesRegex(RuntimeError,"attempt"):
            advance_sdirk2_adaptive(op,d,s,.3,initial_dt_s=.01,
                tolerances=SDIRK2Tolerances(max_attempts=1,max_dt_s=.01))


if __name__=="__main__":
    unittest.main(verbosity=2)
