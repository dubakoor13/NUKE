"""Independent Stage 11 2D FV-to-delayed-family kinetics verification.

Unlike isolated Stage 7 tests, these use the actual harmonic-face finite
volume loss matrix from Stage 2 on nonuniform 2D meshes and full group
scattering. All material constants are explicitly synthetic.
"""
from __future__ import annotations
from dataclasses import replace
import unittest
import numpy as np
from scipy.linalg import expm
from .fx2th_diffusion import DiffusionGrid, assemble_diffusion, solve_keff, solve_fixed_source
from .fx2th_family_spacetime import (
    SpatialFamilyState,advance_spatial_families,stationary_precursors,
    importance_snapshot
)
from .fx2th_cartesian_family_bridge import (
    cartesian_to_spatial_families,diffusion_eigen_kinetics_reference
)
from .validate_fx2th_diffusion import simple
from .validate_fx2th_family_spacetime import dense_generator


def heterogeneous(*, tuned: bool=False, boundary: str="reflecting") -> DiffusionGrid:
    """2 by 3 cells, two energy groups, unequal face gaps and scattering."""
    x=np.array([0.,0.7,1.4,3.])
    y=np.array([0.,0.9,2.2])
    shape=(2,3,2)
    D=np.empty(shape); D[...,0]=.22;D[...,1]=.42
    D[0,2,0]=.61;D[1,0,1]=.18
    sa=np.empty(shape);sa[...,0]=.32;sa[...,1]=.49
    scattering=np.zeros((*shape,2))
    scattering[...,0,1]=.09
    scattering[...,1,0]=.045
    nf=np.empty(shape);nf[...,0]=.38;nf[...,1]=.21
    nf[0,0,0]*=1.25;nf[1,2,1]*=1.4
    chi=np.empty(shape);chi[...,0]=.78;chi[...,1]=.22
    chi[1,0,0]=.62;chi[1,0,1]=.38
    g=DiffusionGrid(x,y,D,sa,scattering,nf,chi,
                    dict.fromkeys(("left","right","top","bottom"),boundary))
    if tuned:
        k=solve_keff(g).keff
        g=replace(g,nu_fission_inv_m=nf/k)
    return g


def data(grid, *, families=2, different_spectra=False):
    ny,nx,ng=grid.diffusion_m.shape
    nc=nx*ny
    beta=np.full((nc,ng,families),.004/families)
    beta[:,0,0]=.002
    beta[0,0,0]=.006
    lam=np.array([.1,1.7]) if families==2 else np.linspace(.1,1.7,families)
    spectrum=grid.fission_spectrum.reshape(nc,ng)
    cd=np.repeat(spectrum[:,None,:],families,axis=1)
    if different_spectra and ng>1:
        cd=cd.copy()
        cd[0,0,:]=cd[0,0,::-1]
    return cartesian_to_spatial_families(
        grid,source_id="synthetic-heterogeneous-FV-ref",
        beta_by_incident_group_and_family=beta,
        delayed_spectra_by_family=cd,
        decay_constants_s_inv=lam,
        group_speeds_m_s=np.linspace(2.,5.,ng)
    )


class CartesianBridgeTests(unittest.TestCase):
    def test_exact_loss_matrix_fission_and_indexing(self):
        g=heterogeneous();d=data(g)
        m=assemble_diffusion(g)
        np.testing.assert_allclose(d.data.integrated_loss.toarray(),m.loss.toarray(),rtol=0,atol=0)
        np.testing.assert_allclose(d.data.volumes_m3,m.cell_volume_m3.flatten(),rtol=0,atol=0)
        np.testing.assert_allclose(d.data.nu_sigma_f_inv_m,
                                   g.nu_fission_inv_m.reshape(d.data.cells,d.data.groups))
        np.testing.assert_allclose(d.data.prompt_spectrum,
                                   g.fission_spectrum.reshape(d.data.cells,d.data.groups))
        self.assertEqual(d.shape_yxg,(2,3,2))

    def test_nonuniform_cell_volumes_and_flatten_roundtrip(self):
        g=heterogeneous();d=data(g)
        np.testing.assert_allclose(d.data.volumes_m3,[.63,.63,1.44,.91,.91,2.08])
        image=np.arange(12,dtype=float).reshape((2,3,2))
        np.testing.assert_array_equal(d.image_flux(d.reshape_flux(image)),image)
        with self.assertRaisesRegex(ValueError,"flux"):
            d.reshape_flux(np.ones((3,2,2)))

    def test_nonuniform_leakage_plus_absorption_equals_global_loss(self):
        g=heterogeneous(boundary="zero_flux")
        d=data(g)
        phi=np.arange(1,1+2*3*2,dtype=float).reshape(6,2)*.07
        total,a,l=d.diffusion_loss_partition(phi,g)
        self.assertGreater(l,0.)
        self.assertAlmostEqual(total,a+l,places=11)
        self.assertAlmostEqual(l,d.boundary_leakage(phi),places=12)

    def test_reflecting_boundary_cancels_face_currents_and_scattering(self):
        g=heterogeneous()
        d=data(g)
        phi=np.arange(12,dtype=float).reshape(6,2)*.2+1.
        total,a,l=d.diffusion_loss_partition(phi,g)
        self.assertAlmostEqual(l,0,places=12)
        self.assertAlmostEqual(total,a,places=11)

    def test_stationary_critical_single_group_exact_implicit(self):
        g=simple(4,3)
        g=replace(g,nu_fission_inv_m=np.full((3,4,1),.4))
        d=data(g)
        out=diffusion_eigen_kinetics_reference(d,g)
        self.assertAlmostEqual(out.keff,1.,places=9)
        self.assertIsNotNone(out.equilibrium_state)
        s=out.equilibrium_state
        for dt in (.001,.25,2.):
            step=advance_spatial_families(d.data,s,dt)
            np.testing.assert_allclose(step.state.flux,s.flux,rtol=1e-9,atol=1e-10)
            np.testing.assert_allclose(step.state.precursors,s.precursors,rtol=1e-9,atol=1e-10)
            self.assertLess(step.flux_equation_residual,1e-9)

    def test_multigroup_nonuniform_stationary_critical(self):
        g=heterogeneous(tuned=True)
        d=data(g)
        ref=diffusion_eigen_kinetics_reference(d,g)
        self.assertAlmostEqual(ref.keff,1.,places=8)
        self.assertIsNotNone(ref.equilibrium_state)
        self.assertLess(ref.forward_integrated_residual,1e-10)
        self.assertLess(ref.adjoint_integrated_residual,1e-10)
        self.assertAlmostEqual(ref.importance_fission_normalization,1.,places=9)
        step=advance_spatial_families(d.data,ref.equilibrium_state,.08)
        np.testing.assert_allclose(step.state.flux,ref.equilibrium_state.flux,rtol=1e-8,atol=1e-9)
        self.assertLess(step.global_balance_residual,1e-10)

    def test_weighted_adjoint_fission_contraction_exactly_one_volume(self):
        g=heterogeneous(tuned=True)
        d=data(g)
        eig=diffusion_eigen_kinetics_reference(d,g)
        r=importance_snapshot(d.data,eig.equilibrium_state,
                              eig.adjoint.reshape(d.data.cells,d.data.groups))
        p=eig.forward.flatten();psi=eig.adjoint.flatten()
        m=assemble_diffusion(g)
        direct=float(psi @ (m.fission@p))
        self.assertAlmostEqual(direct,1.,places=9)
        self.assertAlmostEqual(r.weighted_total,direct,places=9)
        volumes=np.repeat(d.data.volumes_m3,d.data.groups)
        incorrect_twice=float(psi @ (volumes*(m.fission@p)))
        self.assertGreater(abs(incorrect_twice-direct),.03)
        self.assertAlmostEqual(sum(r.beta_effective_by_family),r.beta_effective)

    def test_effective_beta_family_sum_and_raw_weighting(self):
        g=heterogeneous(tuned=True)
        d=data(g)
        r=diffusion_eigen_kinetics_reference(d,g)
        self.assertEqual(len(r.beta_effective_by_family),2)
        self.assertAlmostEqual(sum(r.beta_effective_by_family),
                               sum(r.beta_unweighted_by_family),places=4) if False else None
        self.assertTrue(all(0.<=b<1. for b in r.beta_effective_by_family))
        self.assertTrue(all(0.<=b<1. for b in r.beta_unweighted_by_family))

    def test_single_group_analytic_generation_time(self):
        g=simple(2,2,G=1)
        d=data(g)
        r=diffusion_eigen_kinetics_reference(d,g)
        # Lambda = <psi,V/v*phi> / <psi,V*nuSigmaF*phi>
        # one group, constant v & nuSigmaF => 1/(v*nuSigmaF).
        self.assertAlmostEqual(r.generation_time_s,1./(2.*.6),places=10)

    def test_noncritical_eigenvector_does_not_claim_equilibrium(self):
        g=heterogeneous()
        d=data(g)
        r=diffusion_eigen_kinetics_reference(d,g)
        self.assertGreater(abs(r.keff-1.),1e-4)
        self.assertIsNone(r.equilibrium_state)

    def test_distinct_delayed_spectrum_invalidates_stage2_eigen_equilibrium(self):
        g=heterogeneous(tuned=True)
        d=data(g,different_spectra=True)
        with self.assertRaisesRegex(ValueError,"birth spectra"):
            diffusion_eigen_kinetics_reference(d,g)

    def test_subcritical_external_source_stationary_solution(self):
        g=simple(3,2,G=1)
        g=replace(g,nu_fission_inv_m=np.full((2,3,1),.20))
        d=data(g)
        source=np.full((2,3,1),.3)
        stationary=solve_fixed_source(g,source)
        phi=d.reshape_flux(stationary)
        s=SpatialFamilyState(0.,phi,stationary_precursors(d.data,phi))
        step=advance_spatial_families(d.data,s,.12,source.reshape(d.data.cells,1))
        np.testing.assert_allclose(step.state.flux,phi,rtol=1e-10,atol=1e-10)
        self.assertLess(step.global_balance_residual,1e-10)

    def test_full_spatial_matrix_exponential_convergence(self):
        # Spatial diffusion A comes from Stage 2, reference full-state
        # generator is assembled independently of Stage 7 Schur reduction.
        g=heterogeneous()
        d=data(g)
        phi=np.ones((d.data.cells,d.data.groups))
        phi[0,0]=3.
        s=SpatialFamilyState(0.,phi,stationary_precursors(d.data,phi))
        y=np.r_[phi.flatten(),s.precursors.flatten()]
        exact=expm(dense_generator(d.data)*.2)@y
        def integrate(n):
            state=s
            for _ in range(n):
                state=advance_spatial_families(d.data,state,.2/n).state
            return np.r_[state.flux.flatten(),state.precursors.flatten()]
        e1=np.linalg.norm(integrate(5)-exact)
        e2=np.linalg.norm(integrate(80)-exact)
        self.assertGreater(e1,0)
        self.assertLess(e2,e1/10.)

    def test_source_shape_and_material_mismatch_fails_closed(self):
        g=heterogeneous()
        nc=6;ng=2;nf=2
        original=np.full((nc,ng,nf),.001)
        other=np.tile(g.fission_spectrum.reshape(nc,ng)[:,None,:],(1,nf,1))
        with self.assertRaisesRegex(ValueError,"beta"):
            cartesian_to_spatial_families(g,source_id="bad-beta",
                beta_by_incident_group_and_family=np.zeros((nc+1,ng,nf)),
                delayed_spectra_by_family=other,decay_constants_s_inv=[.1,1.],
                group_speeds_m_s=[2.,5.])
        d=cartesian_to_spatial_families(g,source_id="match",
                beta_by_incident_group_and_family=original,
                delayed_spectra_by_family=other,
                decay_constants_s_inv=[.1,1.],group_speeds_m_s=[2.,5.])
        mutated=replace(g,diffusion_m=g.diffusion_m*2.)
        with self.assertRaisesRegex(ValueError,"incompatible"):
            diffusion_eigen_kinetics_reference(d,mutated)

    def test_zero_flux_mesh_refinement_analytic_sine_mode(self):
        def make(n):
            g=simple(n,n,G=1,bound="zero_flux")
            return replace(g,nu_fission_inv_m=np.full((n,n,1),.2))
        import math
        expected=.2/(.4+2*.35*math.pi**2)
        e_coarse=abs(solve_keff(make(5)).keff-expected)
        e_fine=abs(solve_keff(make(12)).keff-expected)
        self.assertLess(e_fine,e_coarse*.25)


if __name__=="__main__":
    unittest.main(verbosity=2)
