"""Synthetic 2D space-time diffusion tests, NOT historical reactor benchmarks."""
from __future__ import annotations
import math
import unittest
import numpy as np
from scipy.linalg import expm
from .fx2th_diffusion import DiffusionGrid, assemble_diffusion, solve_keff
from .fx2th_spacetime import (
    DelayedData,SpaceTimeState,advance_space_time,critical_precursors_spatial,
    mass_diagonal,project_amplitude,advance_frozen_shape,run_space_time,
)


def grid(n=1,ng=1,nu=0.4,absorption=0.4,leak=False):
    shape=(n,n,ng)
    scattering=np.zeros(shape+(ng,))
    chi=np.zeros(shape)
    chi[...,0]=1.
    return DiffusionGrid(np.linspace(0,1,n+1),np.linspace(0,1,n+1),
        np.full(shape,0.3),np.full(shape,absorption),scattering,
        np.full(shape,nu/ng),chi,
        {side:'zero_flux' if leak else 'reflecting' for side in ('left','right','bottom','top')})


def state(g,d,a=1.):
    phi=np.full(assemble_diffusion(g).shape,a)
    prec=critical_precursors_spatial(g,d,phi)
    return SpaceTimeState(0.,phi,prec)


class SpaceTimeTests(unittest.TestCase):
    def test_stationary_critical_with_delayed(self):
        g=grid(n=3)
        d=DelayedData((0.004,0.002),(0.1,0.6),(1.,))
        st=state(g,d,3.)
        for _ in range(5):
            result=advance_space_time(g,d,st,0.4)
            st=result.state
        np.testing.assert_allclose(st.flux,3,rtol=1e-11,atol=1e-11)
        self.assertLess(result.flux_equation_residual,1e-11)
        self.assertLess(abs(result.total_neutron_balance_residual),1e-11)

    def test_prompt_only_single_cell_backward_euler(self):
        g=grid(n=1,nu=0.3,absorption=0.4)
        d=DelayedData((),(),(2.,))
        s=state(g,d,a=2.)
        result=advance_space_time(g,d,s,0.5)
        expected=2./(1+0.5*2*(0.4-0.3))
        self.assertAlmostEqual(float(result.state.flux[0,0,0]),expected)
        self.assertEqual(result.state.precursors.shape,(1,1,0))

    def test_analytic_matrix_exponential_convergence(self):
        g=grid(n=1,nu=0.42)
        d=DelayedData((0.004,0.002),(0.2,0.8),(3.,))
        s=state(g,d,a=1.)
        beta=sum(d.beta)
        A=np.array([[-3*(0.4-(1-beta)*0.42),3*d.decay_s_inv[0],3*d.decay_s_inv[1]],
                    [d.beta[0]*0.42,-d.decay_s_inv[0],0],
                    [d.beta[1]*0.42,0,-d.decay_s_inv[1]]])
        exact=expm(A*0.6)@np.array([1.,s.precursors[0,0,0],s.precursors[0,0,1]])
        def go(substeps):
            st=s
            for _ in range(substeps): st=advance_space_time(g,d,st,0.6/substeps).state
            return np.array([st.flux.item(),*st.precursors.flatten()])
        coarse=go(10);fine=go(100)
        self.assertLess(np.linalg.norm(fine-exact),np.linalg.norm(coarse-exact)/5)
        self.assertLess(np.linalg.norm(fine-exact),0.0007)

    def test_spatial_precursor_inventory_local(self):
        g=grid(n=2)
        d=DelayedData((0.005,),(0.5,),(1.,))
        s=state(g,d)
        s=SpaceTimeState(0.,np.array([[[1.],[2.]],[[3.],[4.]]]),s.precursors)
        result=advance_space_time(g,d,s,0.01)
        vol=assemble_diffusion(g).cell_volume_m3
        expected=(s.precursors+0.01*(vol*.4*result.state.flux[:,:,0])[:,:,None]*.005)/(1+.01*.5)
        np.testing.assert_allclose(result.state.precursors,expected,rtol=1e-12)
        self.assertLess(result.precursor_equation_residual,1e-11)

    def test_closed_neutron_inventory_and_boundary_loss(self):
        g=grid(n=3,nu=0.55,absorption=0.3,leak=True)
        d=DelayedData((0.006,),(0.7,),(8.,))
        s=state(g,d)
        result=advance_space_time(g,d,s,0.005)
        self.assertGreater(result.state.flux.sum(),0)
        self.assertLess(abs(result.total_neutron_balance_residual),1e-10)
        self.assertLess(result.flux_equation_residual,1e-10)

    def test_fixed_source_long_time_asymptote(self):
        g=grid(n=2,nu=0.2)
        d=DelayedData((0.003,),(0.2,),(1.,))
        old=SpaceTimeState(0.,np.zeros((2,2,1)),np.zeros((2,2,1)))
        q=np.full((2,2,1),0.08)
        for _ in range(300): old=advance_space_time(g,d,old,0.4,q).state
        np.testing.assert_allclose(old.flux,0.08/(0.4-0.2),atol=4e-3)

    def test_cell_heterogeneity_and_group_scattering(self):
        base=grid(2,2,nu=0.5)
        scatter=np.zeros((2,2,2,2));scatter[...,0,1]=0.18;scatter[...,1,0]=0.06
        absorp=np.full((2,2,2),0.4);absorp[0,0,0]*=1.6
        chi=np.zeros((2,2,2));chi[...,0]=0.7;chi[...,1]=0.3
        g=DiffusionGrid(base.x_edges_m,base.y_edges_m,base.diffusion_m,absorp,
                        scatter,base.nu_fission_inv_m,chi,base.boundaries)
        d=DelayedData((0.003,),(0.5,),(3.,7.))
        s=state(g,d)
        result=advance_space_time(g,d,s,0.05)
        self.assertLess(result.flux_equation_residual,1e-9)
        self.assertLess(abs(result.total_neutron_balance_residual),1e-9)
        self.assertTrue(np.all(result.state.flux>0))

    def test_adjoint_weighted_projection(self):
        g=grid(n=2)
        d=DelayedData((0.003,),(0.4,),(1.,))
        s=state(g,d,a=2.)
        adj=solve_keff(g).adjoint
        a,shape=project_amplitude(g,d,s,adj)
        np.testing.assert_allclose(shape*a,s.flux)
        self.assertAlmostEqual(float(np.dot((adj*shape).flatten(),mass_diagonal(g,d))),1.)

    def test_refresh_each_microstep_is_full_pde(self):
        g=grid(2,nu=0.45)
        d=DelayedData((0.004,),(0.2,),(4.,))
        s=state(g,d)
        s=SpaceTimeState(0.,np.array([[[1.],[2.]],[[3.],[1.]]]),s.precursors)
        grids=[g]*4
        ref=[s]
        for _ in range(4):ref.append(advance_space_time(g,d,ref[-1],0.03).state)
        run=run_space_time(grids,d,s,0.03,refresh_every=1)
        np.testing.assert_allclose(run[-1].state.flux,ref[-1].flux)
        self.assertTrue(all(step.spatial_refreshed for step in run))

    def test_projected_uniform_shape_matches_full(self):
        g=grid(n=3)
        d=DelayedData((0.005,),(0.1,),(3.,))
        s=state(g,d,a=2.)
        adj=solve_keff(g).adjoint
        result=run_space_time([g]*5,d,s,0.08,refresh_every=99,adjoint_weight=adj)
        ref=s
        for _ in range(5):ref=advance_space_time(g,d,ref,0.08).state
        np.testing.assert_allclose(result[-1].state.flux,ref.flux,atol=1e-10)
        self.assertTrue(all(not r.spatial_refreshed for r in result))

    def test_periodic_shape_refresh_resolves_spatial_deviation(self):
        g=grid(n=2,nu=0.37)
        d=DelayedData((0.003,),(0.8,),(1.,))
        s=state(g,d)
        s=SpaceTimeState(0.,np.array([[[1.],[3.]],[[2.],[0.5]]]),s.precursors)
        adj=solve_keff(g).adjoint
        full=run_space_time([g]*4,d,s,0.1,refresh_every=1)
        cheap=run_space_time([g]*4,d,s,0.1,refresh_every=3,adjoint_weight=adj)
        self.assertTrue(cheap[2].spatial_refreshed)
        self.assertFalse(cheap[3].spatial_refreshed)
        self.assertGreater(cheap[0].flux_equation_residual,1e-3)
        self.assertGreater(np.linalg.norm(cheap[-1].state.flux-full[-1].state.flux),1e-5)

    def test_user_supplied_time_varying_materials(self):
        a=grid(n=2,nu=0.4)
        b=grid(n=2,nu=0.42)
        d=DelayedData((0.001,),(0.5,),(1.,))
        s=state(a,d)
        steps=run_space_time([a,b],d,s,0.05)
        self.assertLess(steps[0].flux_equation_residual,1e-10)
        self.assertGreater(float(steps[1].state.flux.mean()),float(steps[0].state.flux.mean()))

    def test_time_varying_mesh_without_remap_is_rejected(self):
        a=grid(n=2)
        b=grid(n=3)
        d=DelayedData((),(),(1.,))
        s=state(a,d)
        with self.assertRaisesRegex(ValueError, 'remapping'):
            run_space_time([a,b],d,s,0.1)

    def test_bad_inputs_and_shapes(self):
        g=grid(n=2)
        d=DelayedData((),(),(1.,))
        s=state(g,d)
        with self.assertRaises(ValueError): DelayedData((0.3,0.8),(0.1,0.2),(1.,))
        with self.assertRaises(ValueError): DelayedData((0.1,),(0.,),(1.,))
        with self.assertRaises(ValueError): advance_space_time(g,d,s,-1.)
        with self.assertRaises(ValueError): advance_space_time(g,d,s,1.,np.full((2,2,1),-1.))
        with self.assertRaises(ValueError): project_amplitude(g,d,s,np.zeros((2,2,1)))
        with self.assertRaises(ValueError): run_space_time([g],d,s,.1,refresh_every=2)


if __name__=='__main__':
    unittest.main(verbosity=2)
