"""13 synthetic numerical tests for IQS-inspired two-rate evolution.

No nuclear data, historical FX2-TH, OpenMC, or NERVA benchmark claims.
"""
from __future__ import annotations
import unittest
import numpy as np

from .fx2th_diffusion import DiffusionGrid, assemble_diffusion, solve_keff
from .fx2th_spacetime import (DelayedData, SpaceTimeState, advance_space_time,
                              critical_precursors_spatial, run_space_time)
from .fx2th_iqs import advance_iqs_shape, run_iqs_shape


def synthetic_grid(n=2, group_count=1, nu=0.37, leak=False):
    dims=(n,n,group_count)
    spectrum=np.zeros(dims);spectrum[...,0]=1
    scattering=np.zeros(dims+(group_count,))
    if group_count==2:
        scattering[...,0,1]=0.16
        scattering[...,1,0]=0.04
        spectrum[...,0]=0.8
        spectrum[...,1]=0.2
    return DiffusionGrid(
        np.linspace(0.,1.,n+1),np.linspace(0.,1.,n+1),
        np.full(dims,0.3),np.full(dims,0.4),scattering,
        np.full(dims,nu/group_count),spectrum,
        {side:'zero_flux' if leak else 'reflecting'
         for side in ('left','right','top','bottom')})


def initial(grid, delayed, heterogeneous=True):
    phi=np.ones(grid.diffusion_m.shape)
    if heterogeneous and phi.shape[:2]==(2,2):
        phi[:,:,0]=[[1.,3.],[2.,0.5]]
        if phi.shape[-1]==2:
            phi[:,:,1]=[[0.4,2.],[1.,0.3]]
    return SpaceTimeState(0.,phi,critical_precursors_spatial(grid,delayed,phi))


def reference(grid, delayed, start, dt, steps, source=None):
    state=start
    for _ in range(steps):
        state=advance_space_time(grid,delayed,state,dt,source).state
    return state


class EvolvingIQSTests(unittest.TestCase):
    def test_single_microstep_equals_full_implicit_pde(self):
        g=synthetic_grid()
        d=DelayedData((0.003,0.002),(0.2,0.9),(1.,))
        s=initial(g,d)
        w=solve_keff(g).adjoint
        result=advance_iqs_shape(g,d,s,0.05,w,microsteps=1)
        full=advance_space_time(g,d,s,0.05)
        np.testing.assert_allclose(result.state.flux,full.state.flux,rtol=1e-8,atol=1e-9)
        np.testing.assert_allclose(result.state.precursors,full.state.precursors,rtol=1e-8,atol=1e-9)
        self.assertLess(result.spatial_integral_residual,1e-8)

    def test_two_rate_reduces_frozen_shape_error(self):
        g=synthetic_grid();d=DelayedData((0.003,),(0.8,),(1.,))
        s=initial(g,d);w=solve_keff(g).adjoint
        fine=reference(g,d,s,0.1,4)
        evolving=run_iqs_shape(g,d,s,0.1,4,w,shape_interval=4)[-1].state
        frozen=run_space_time([g]*4,d,s,0.1,refresh_every=5,adjoint_weight=w)[-1].state
        ee=np.linalg.norm(evolving.flux-fine.flux)/np.linalg.norm(fine.flux)
        ef=np.linalg.norm(frozen.flux-fine.flux)/np.linalg.norm(fine.flux)
        self.assertLess(ee,0.10)
        self.assertLess(ee,0.15*ef)

    def test_more_shape_updates_reduces_error(self):
        g=synthetic_grid();d=DelayedData((0.003,),(0.8,),(1.,))
        s=initial(g,d);w=solve_keff(g).adjoint
        truth=reference(g,d,s,0.1,4).flux
        errors=[]
        for interval in (4,2,1):
            phi=run_iqs_shape(g,d,s,0.1,4,w,shape_interval=interval)[-1].state.flux
            errors.append(np.linalg.norm(phi-truth))
        self.assertLess(errors[2],1e-7)
        self.assertLess(errors[2],errors[1])
        self.assertLess(errors[1],errors[0])

    def test_integrated_balance_and_precursor_residual(self):
        g=synthetic_grid(leak=True,nu=0.6)
        d=DelayedData((0.005,0.001),(0.2,0.8),(4.,))
        s=initial(g,d);w=solve_keff(g).adjoint
        r=advance_iqs_shape(g,d,s,0.015,w,microsteps=4)
        self.assertLess(r.spatial_integral_residual,1e-8)
        self.assertLess(abs(r.global_neutron_balance_residual),1e-9)
        self.assertLess(r.maximum_precursor_residual,1e-10)
        self.assertGreater(r.shape_iterations,1)

    def test_adjoint_normalization(self):
        g=synthetic_grid();d=DelayedData((0.003,),(0.4,),(1.,))
        s=initial(g,d);w=solve_keff(g).adjoint
        vol=assemble_diffusion(g).cell_volume_m3[:,:,None]
        for r in run_iqs_shape(g,d,s,0.1,5,w,shape_interval=2):
            self.assertLess(r.shape_normalization_residual,1e-10)
            self.assertAlmostEqual(float(np.sum(w*r.state.flux*vol)),
                                   r.final_amplitude,places=9)

    def test_uniform_critical_state(self):
        g=synthetic_grid(n=3,nu=0.4)
        d=DelayedData((0.004,0.002),(0.1,0.7),(1.,))
        s=initial(g,d,False);w=solve_keff(g).adjoint
        r=run_iqs_shape(g,d,s,0.15,8,w,shape_interval=4)[-1]
        np.testing.assert_allclose(r.state.flux,s.flux,atol=1e-9)
        np.testing.assert_allclose(r.state.precursors,s.precursors,atol=1e-9)

    def test_no_delayed_one_cell_analytic(self):
        g=synthetic_grid(n=1,nu=0.2);d=DelayedData((),(),(2.,))
        s=initial(g,d,False);w=solve_keff(g).adjoint
        r=run_iqs_shape(g,d,s,0.01,10,w,shape_interval=5)[-1]
        expected=(1+0.01*2*(0.4-0.2))**(-10)
        self.assertAlmostEqual(float(r.state.flux[0,0,0]),expected,places=9)
        self.assertEqual(r.state.precursors.shape,(1,1,0))

    def test_precursor_local_implicit_equation(self):
        g=synthetic_grid();d=DelayedData((0.005,0.002),(0.4,1.2),(1.,))
        s=initial(g,d);w=solve_keff(g).adjoint
        r=advance_iqs_shape(g,d,s,0.05,w,microsteps=3)
        old=s.precursors;vol=assemble_diffusion(g).cell_volume_m3
        for state in r.micro_states:
            rate=(state.precursors-old)/0.05
            p=np.sum(g.nu_fission_inv_m*state.flux,axis=2)*vol
            rhs=(np.array(d.beta)[None,None,:]*p[:,:,None]
                 -np.array(d.decay_s_inv)[None,None,:]*state.precursors)
            np.testing.assert_allclose(rate,rhs,atol=1e-11)
            old=state.precursors

    def test_two_group_heterogeneous(self):
        g=synthetic_grid(group_count=2,nu=0.55)
        sa=np.array(g.absorption_inv_m,copy=True);sa[0,0,0]*=1.15
        g=DiffusionGrid(g.x_edges_m,g.y_edges_m,g.diffusion_m,sa,
                        g.scattering_inv_m,g.nu_fission_inv_m,
                        g.fission_spectrum,g.boundaries)
        d=DelayedData((0.003,),(0.6,),(2.,5.))
        s=initial(g,d);w=solve_keff(g).adjoint
        full=reference(g,d,s,0.025,4)
        r=run_iqs_shape(g,d,s,0.025,4,w,shape_interval=2)[-1]
        self.assertLess(r.spatial_integral_residual,1e-8)
        self.assertLess(np.linalg.norm(r.state.flux-full.flux)/np.linalg.norm(full.flux),0.03)
        self.assertTrue(np.all(r.state.flux>0))

    def test_external_source(self):
        g=synthetic_grid(nu=0.2);d=DelayedData((0.002,),(0.3,),(1.,))
        s=initial(g,d);w=solve_keff(g).adjoint
        q=np.full(s.flux.shape,0.05)
        r=run_iqs_shape(g,d,s,0.02,4,w,shape_interval=2,source=q)[-1]
        full=reference(g,d,s,0.02,4,q)
        np.testing.assert_allclose(r.state.flux,full.flux,atol=0.02)
        self.assertLess(abs(r.global_neutron_balance_residual),1e-9)

    def test_time_refinement(self):
        g=synthetic_grid();d=DelayedData((0.003,),(0.8,),(1.,))
        s=initial(g,d);w=solve_keff(g).adjoint
        ref=reference(g,d,s,0.001,200).flux
        coarse=run_iqs_shape(g,d,s,0.05,4,w,shape_interval=2)[-1].state.flux
        fine=run_iqs_shape(g,d,s,0.0125,16,w,shape_interval=2)[-1].state.flux
        self.assertLess(np.linalg.norm(fine-ref),np.linalg.norm(coarse-ref))

    def test_invalid_parameters(self):
        g=synthetic_grid();d=DelayedData((),(),(1.,))
        s=initial(g,d);w=solve_keff(g).adjoint
        with self.assertRaises(ValueError): advance_iqs_shape(g,d,s,-0.1,w)
        with self.assertRaises(ValueError): advance_iqs_shape(g,d,s,0.1,np.zeros_like(w))
        with self.assertRaises(ValueError): advance_iqs_shape(g,d,s,0.1,w,microsteps=0)
        with self.assertRaises(ValueError): advance_iqs_shape(g,d,s,0.1,w,source=-np.ones_like(s.flux))
        with self.assertRaises(ValueError): run_iqs_shape(g,d,s,0.1,0,w)
        with self.assertRaises(ValueError): run_iqs_shape(g,d,s,0.1,1,w,shape_interval=0)

    def test_does_not_mutate_inputs(self):
        g=synthetic_grid();d=DelayedData((0.003,),(0.5,),(1.,))
        s=initial(g,d);w=solve_keff(g).adjoint
        flux=s.flux.copy();prec=s.precursors.copy()
        advance_iqs_shape(g,d,s,0.1,w,microsteps=3)
        np.testing.assert_array_equal(flux,s.flux)
        np.testing.assert_array_equal(prec,s.precursors)


if __name__=='__main__':
    unittest.main(verbosity=2)
