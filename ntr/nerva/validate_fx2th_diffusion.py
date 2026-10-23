"""Synthetic finite-volume tests; no physical reactor benchmark."""
from __future__ import annotations
import math
import unittest
import numpy as np
from scipy.linalg import eig
from .fx2th_diffusion import DiffusionGrid, assemble_diffusion, solve_keff, solve_fixed_source


def simple(nx=2,ny=2,G=1,bound='reflecting'):
    shape=(ny,nx,G)
    s=np.zeros(shape+(G,))
    chi=np.zeros(shape);chi[...,0]=1
    return DiffusionGrid(np.linspace(0,1,nx+1),np.linspace(0,1,ny+1),
        np.full(shape,0.35),np.full(shape,0.4),s,
        np.full(shape,0.6),chi,
        {side:bound for side in ('left','right','bottom','top')})


class DiffusionTests(unittest.TestCase):
    def test_reflecting_single_group_uniform_analytic(self):
        m=simple(nx=4,ny=3)
        r=solve_keff(m)
        self.assertAlmostEqual(r.keff,1.5,places=9)
        np.testing.assert_allclose(r.forward,np.ones_like(r.forward),rtol=1e-8,atol=1e-9)
        self.assertAlmostEqual(r.boundary_leakage_neutrons_per_s,0,places=12)
        self.assertLess(abs(r.global_neutron_balance_residual),1e-9)

    def test_one_cell_zero_face_flux_analytic(self):
        grid=simple(1,1,bound='zero_flux')
        r=solve_keff(grid)
        self.assertAlmostEqual(r.keff,0.6/(0.4+8*0.35),places=10)
        self.assertAlmostEqual(r.boundary_leakage_neutrons_per_s,8*0.35,places=10)
        self.assertLess(abs(r.global_neutron_balance_residual),1e-9)

    def test_harmonic_face_coefficient_nonsymmetric_mesh(self):
        grid=simple(2,1)
        D=np.array([[[0.5],[2.]]])
        grid=DiffusionGrid(np.array([0,1,3.]),np.array([0,2.]),D,
            grid.absorption_inv_m,grid.scattering_inv_m,
            grid.nu_fission_inv_m,grid.fission_spectrum,grid.boundaries)
        A=assemble_diffusion(grid).loss.toarray()
        expected=2./(1/(2*0.5)+2/(2*2.))
        self.assertAlmostEqual(A[0,1],-expected,places=12)
        self.assertAlmostEqual(A[1,0],-expected,places=12)
        np.testing.assert_allclose(A.sum(axis=1),[0.4*2.,0.4*4.],atol=1e-12)

    def test_groups_reflecting_dense_analytic_eigenvalue(self):
        g=simple(1,1,2)
        scatter=np.array([[[[0.,0.25],[0.05,0.]]]])
        chi=np.array([[[0.75,0.25]]])
        nf=np.array([[[0.55,0.11]]])
        abs_=np.array([[[0.38,0.20]]])
        grid=DiffusionGrid(g.x_edges_m,g.y_edges_m,g.diffusion_m,
                           abs_,scatter,nf,chi,g.boundaries)
        r=solve_keff(grid)
        m=assemble_diffusion(grid)
        ref=max(eig(m.fission.toarray(),m.loss.toarray(),right=False).real)
        self.assertAlmostEqual(r.keff,ref,places=9)
        self.assertLess(abs(r.global_neutron_balance_residual),1e-9)
        self.assertAlmostEqual(r.biorthogonal_fission_norm,1.,places=10)

    def test_adjoint_matches_dense_transposed_generalized_problem(self):
        g=simple(2,2,2,'zero_flux')
        chi=np.zeros_like(g.fission_spectrum);chi[...,0]=0.7;chi[...,1]=0.3
        scatter=np.zeros_like(g.scattering_inv_m)
        scatter[...,0,1]=0.18
        scatter[...,1,0]=0.07
        nf=np.zeros_like(g.nu_fission_inv_m)
        nf[...,0]=0.6;nf[...,1]=0.15
        nf[0,0,0]*=1.4
        grid=DiffusionGrid(g.x_edges_m,g.y_edges_m,g.diffusion_m,
                           g.absorption_inv_m,scatter,nf,chi,g.boundaries)
        m=assemble_diffusion(grid)
        r=solve_keff(grid)
        p=r.forward.flatten();q=r.adjoint.flatten()
        self.assertLess(np.linalg.norm(m.loss @p - (m.fission@p)/r.keff),1e-8)
        self.assertLess(np.linalg.norm(m.loss.T@q - (m.fission.T@q)/r.keff),1e-8)
        self.assertAlmostEqual(q @ (m.fission @ p),1.,places=8)
        ref=max(eig(m.fission.toarray(),m.loss.toarray(),right=False).real)
        self.assertAlmostEqual(r.keff,ref,places=8)

    def test_mesh_convergence_bare_sine_eigenfunction(self):
        exact=0.6/(0.4+2*0.35*math.pi**2)
        coarse=solve_keff(simple(5,5,bound='zero_flux'),tolerance=2e-9).keff
        fine=solve_keff(simple(10,10,bound='zero_flux'),tolerance=2e-9).keff
        self.assertLess(abs(fine-exact),abs(coarse-exact)*0.35)

    def test_fixed_source_analytic_reflected(self):
        g=simple(2,2)
        # make subcritical: production 0.2 < absorption 0.4
        new_fis=np.full_like(g.nu_fission_inv_m,0.2)
        grid=DiffusionGrid(g.x_edges_m,g.y_edges_m,g.diffusion_m,
                          g.absorption_inv_m,g.scattering_inv_m,new_fis,
                          g.fission_spectrum,g.boundaries)
        phi=solve_fixed_source(grid,np.full_like(new_fis,3.))
        np.testing.assert_allclose(phi,15.,rtol=1e-9)
        m=assemble_diffusion(grid)
        np.testing.assert_allclose((m.loss-m.fission)@phi.flatten(),
                                   (3*m.cell_volume_m3).flatten(),rtol=1e-8)

    def test_fixed_source_supercritical_rejected(self):
        with self.assertRaises(ValueError):
            solve_fixed_source(simple(),np.ones((2,2,1)))

    def test_asymmetric_mixed_boundaries(self):
        g=simple(1,1)
        boundaries={'left':'zero_flux','right':'reflecting',
                    'bottom':'reflecting','top':'reflecting'}
        grid=DiffusionGrid(g.x_edges_m,g.y_edges_m,g.diffusion_m,
                           g.absorption_inv_m,g.scattering_inv_m,
                           g.nu_fission_inv_m,g.fission_spectrum,boundaries)
        r=solve_keff(grid)
        self.assertAlmostEqual(r.keff,0.6/(0.4+2*0.35),places=9)

    def test_invalid_inputs(self):
        g=simple()
        kwargs=dict(x_edges_m=g.x_edges_m,y_edges_m=g.y_edges_m,
                    diffusion_m=g.diffusion_m,absorption_inv_m=g.absorption_inv_m,
                    scattering_inv_m=g.scattering_inv_m,
                    nu_fission_inv_m=g.nu_fission_inv_m,
                    fission_spectrum=g.fission_spectrum,boundaries=g.boundaries)
        for key,bad in [
            ('x_edges_m',[0,1,0.5]),('diffusion_m',-g.diffusion_m),
            ('scattering_inv_m',np.ones((2,2,1,1))),
            ('fission_spectrum',np.zeros_like(g.fission_spectrum)),
            ('nu_fission_inv_m',np.zeros_like(g.nu_fission_inv_m)),
            ('boundaries',{'left':'vacuum'})]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                DiffusionGrid(**{**kwargs,key:bad})

    def test_rotation_invariance_of_geometry(self):
        a=solve_keff(simple(3,5,bound='zero_flux')).keff
        b=solve_keff(simple(5,3,bound='zero_flux')).keff
        self.assertAlmostEqual(a,b,places=10)


if __name__ == '__main__':
    unittest.main(verbosity=2)
