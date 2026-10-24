"""Independent cylindrical finite-volume and delayed-family tests.

All cross sections/kinetic data are SYNTHETIC; the analytic benchmark is
the separable J0(alpha*r/R)*sin(pi*z/H) bare cylindrical eigenmode.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np
from scipy.linalg import eig, expm
from scipy.sparse.linalg import spsolve
from scipy.special import jn_zeros,j0,j1

from .fx2th_rz_family_diffusion import (
    RZGrid,assemble_rz,solve_rz_keff,
    rz_to_spatial_families,rz_critical_family_reference,
)
from .fx2th_family_spacetime import (
    SpatialFamilyState,advance_spatial_families,stationary_precursors
)
from .validate_fx2th_family_spacetime import dense_generator


def grid(nr=3,nz=4,ng=1,*,boundary="reflecting",fission=.8):
    r=np.linspace(0,1.,nr+1)
    z=np.linspace(0,2.,nz+1)
    shape=(nr,nz,ng)
    D=np.full(shape,.4);Sa=np.full(shape,.7)
    scatter=np.zeros(shape+(ng,))
    chi=np.zeros(shape);chi[...,0]=1.
    nf=np.full(shape,fission)
    if ng==2:
        Sa[...,0]=.8;Sa[...,1]=.6
        nf[...,0]=.55;nf[...,1]=.3
        chi[...,0]=.78;chi[...,1]=.22
        scatter[...,0,1]=.12
        scatter[...,1,0]=.04
    return RZGrid(r,z,D,Sa,scatter,nf,chi,
                  outer_boundary=boundary,
                  lower_boundary=boundary,upper_boundary=boundary)


def family_data(g):
    nr,nz,ng=g.diffusion_m.shape
    nc=nr*nz;nf=2
    beta=np.full((nc,ng,nf),.008)
    beta[:,:,1]=.003
    chi=np.asarray(g.fission_spectrum).reshape(nc,ng)
    delayed=np.repeat(chi[:,None,:],nf,axis=1)
    return rz_to_spatial_families(
        g,source_id="SYNTHETIC-bessel-neutron-family-reference",
        group_speeds_m_s=np.linspace(2.4,5.6,ng),
        beta_by_incident_group_and_family=beta,
        delayed_spectra_by_family=delayed,
        decay_constants_s_inv=np.array([.09,1.7])
    )


class RZNumerics(unittest.TestCase):
    def test_total_annular_volume_exact(self):
        g=grid(7,9)
        m=assemble_rz(g)
        self.assertAlmostEqual(float(m.cell_volumes_m3.sum()),2*math.pi,places=12)
        self.assertAlmostEqual(float(m.cell_volumes_m3[0,0]),
                               math.pi*(1/7)**2*(2/9),places=13)

    def test_axis_symmetry_and_all_reflecting_uniform_mode(self):
        g=grid(8,5)
        m=assemble_rz(g)
        p=np.ones(m.loss.shape[0])
        np.testing.assert_allclose(m.loss@p,.7*m.cell_volumes_m3.flatten(),rtol=1e-12,atol=1e-12)
        self.assertEqual(float(m.boundary_current_coefficients.sum()),0.)

    def test_analytic_reflecting_keff_and_global_balance(self):
        g=grid(4,6)
        x=solve_rz_keff(g)
        self.assertAlmostEqual(x.keff,.8/.7,places=9)
        self.assertLess(abs(x.balance_residual),1e-10)
        self.assertEqual(x.leakage_rate,0.)
        self.assertAlmostEqual(x.biorthogonal_fission_norm,1.,places=9)

    def test_one_annulus_one_axial_dirichlet_exact(self):
        g=grid(1,1,boundary="zero_flux")
        V=2*math.pi
        radial=4*math.pi*.4*2
        axial=2*(math.pi*.4)
        expected=.8*V/(.7*V+radial+axial)
        m=assemble_rz(g)
        self.assertAlmostEqual(m.boundary_current_coefficients.sum(),radial+axial,places=11)
        self.assertAlmostEqual(solve_rz_keff(g).keff,expected,places=10)

    def test_nonuniform_harmonic_radial_faces(self):
        g=grid(2,1)
        g=replace(g,radial_edges_m=np.array([0.,.3,1.0]),
                  diffusion_m=np.array([[[.2]],[[.9]]]))
        m=assemble_rz(g)
        a=m.loss.toarray()
        face_area=2*math.pi*.3*2
        t=face_area/(.3/(2*.2)+.7/(2*.9))
        self.assertAlmostEqual(a[0,1],-t,places=12)
        self.assertAlmostEqual(a[1,0],-t,places=12)

    def test_full_scattering_and_face_current_balance_nonuniform(self):
        g=grid(3,2,2,boundary="zero_flux")
        D=np.asarray(g.diffusion_m).copy()
        D[1,1,0]=.13
        g=replace(g,radial_edges_m=np.array([0.,.1,.7,1.]),
                  axial_edges_m=np.array([0.,.5,2.]),diffusion_m=D)
        m=assemble_rz(g)
        phi=np.arange(1,m.loss.shape[0]+1,dtype=float)*.1
        total=float(np.sum(m.loss@phi))
        a=float(np.sum(g.absorption_inv_m.flatten()*
                       np.repeat(m.cell_volumes_m3.flatten(),2)*phi))
        leakage=float(np.dot(m.boundary_current_coefficients.flatten(),phi))
        self.assertAlmostEqual(total,a+leakage,places=10)
        self.assertGreater(leakage,0.)

    def test_dense_two_group_generalized_forward_adjoint(self):
        g=grid(2,3,2,boundary="zero_flux")
        m=assemble_rz(g)
        out=solve_rz_keff(g,tolerance=1e-11)
        dense=max(eig(m.fission.toarray(),m.loss.toarray(),right=False).real)
        self.assertAlmostEqual(out.keff,dense,places=9)
        p=out.forward.flatten();q=out.adjoint.flatten()
        self.assertLess(np.linalg.norm(m.loss@p-m.fission@p/out.keff),1e-9)
        self.assertLess(np.linalg.norm(m.loss.T@q-m.fission.T@q/out.keff),1e-9)
        self.assertAlmostEqual(float(q@(m.fission@p)),1.,places=9)

    def test_bare_cylinder_bessel_sine_eigenvalue_convergence(self):
        root=float(jn_zeros(0,1)[0])
        exact=.8/(.7+.4*((root/1.)**2+(math.pi/2.)**2))
        co=solve_rz_keff(grid(8,10,boundary="zero_flux")).keff
        fi=solve_rz_keff(grid(20,25,boundary="zero_flux")).keff
        self.assertLess(abs(fi-exact),abs(co-exact)*.30)
        self.assertLess(abs(fi/exact-1),.012)

    def test_analytic_radial_volume_weighted_profile(self):
        g=grid(24,28,boundary="zero_flux")
        x=solve_rz_keff(g)
        phi=x.forward[:,:,0]
        V=assemble_rz(g).cell_volumes_m3
        bands=np.add.reduceat((phi*V).sum(axis=1),np.arange(0,24,6))
        bands=bands/bands.sum()
        a=float(jn_zeros(0,1)[0])
        e=np.linspace(0,1,5)
        integral=e*j1(a*e)
        expected=np.diff(integral)/integral[-1]
        np.testing.assert_allclose(bands,expected,atol=.01,rtol=0)
        # Distinguish a central flux-density peak from shell-integrated power.
        self.assertGreater(bands[1],bands[0])

    def test_analytic_axial_sine_profile(self):
        g=grid(24,28,boundary="zero_flux")
        x=solve_rz_keff(g)
        shape=(x.forward[:,:,0]*assemble_rz(g).cell_volumes_m3).sum(axis=0)
        bands=np.add.reduceat(shape,np.arange(0,28,7))
        bands/=bands.sum()
        e=np.linspace(0,1,5)
        expected=(np.cos(math.pi*e[:-1])-np.cos(math.pi*e[1:]))/2
        np.testing.assert_allclose(bands,expected,atol=.01,rtol=0)

    def test_one_and_two_group_critical_spatial_precursor_modes(self):
        for ng in (1,2):
            with self.subTest(groups=ng):
                raw=grid(3,4,ng)
                k=solve_rz_keff(raw).keff
                critical=replace(raw,nu_fission_inv_m=raw.nu_fission_inv_m/k)
                d=family_data(critical)
                ref,state=rz_critical_family_reference(critical,d)
                self.assertAlmostEqual(ref.keff,1.,places=8)
                self.assertEqual(state.precursors.shape,(12,2))
                for dt in (.01, .5):
                    after=advance_spatial_families(d,state,dt)
                    np.testing.assert_allclose(after.state.flux,state.flux,
                                               atol=1e-9,rtol=1e-9)
                    self.assertLess(after.global_balance_residual,1e-10)

    def test_family_transient_against_independent_dense_exponential(self):
        g=grid(2,2,2)
        d=family_data(g)
        phi=np.ones((4,2));phi[0,0]=3.
        st=SpatialFamilyState(0.,phi,stationary_precursors(d,phi))
        vec=np.r_[phi.flatten(),st.precursors.flatten()]
        truth=expm(dense_generator(d)*.12)@vec
        def iterate(n):
            s=st
            for _ in range(n):
                s=advance_spatial_families(d,s,.12/n).state
            return np.r_[s.flux.flatten(),s.precursors.flatten()]
        e1=float(np.linalg.norm(iterate(6)-truth))
        e2=float(np.linalg.norm(iterate(96)-truth))
        self.assertGreater(e1,0)
        self.assertLess(e2,e1*.15)

    def test_subcritical_reflected_source_exact(self):
        g=grid(3,4,fission=.3)
        m=assemble_rz(g)
        src=.2
        rhs=src*np.repeat(m.cell_volumes_m3.flatten(),1)
        p=spsolve(m.loss-m.fission,rhs)
        np.testing.assert_allclose(p,src/(.7-.3),rtol=2e-12,atol=2e-12)

    def test_distinct_delayed_birth_spectrum_cannot_claim_static_equilibrium(self):
        g=grid(2,3,2)
        k=solve_rz_keff(g).keff
        critical=replace(g,nu_fission_inv_m=g.nu_fission_inv_m/k)
        d=family_data(critical)
        ch=np.array(d.delayed_spectra_by_family,copy=True)
        ch[0,0]=ch[0,0,::-1]
        d=replace(d,delayed_spectra_by_family=ch)
        with self.assertRaisesRegex(ValueError,"distinct family birth spectra"):
            rz_critical_family_reference(critical,d)

    def test_invalid_inputs_are_rejected(self):
        g=grid()
        with self.assertRaisesRegex(ValueError,"axis"):
            replace(g,radial_edges_m=np.linspace(.1,1,4))
        with self.assertRaisesRegex(ValueError,"scattering"):
            replace(g,scattering_inv_m=np.ones_like(g.scattering_inv_m))
        with self.assertRaisesRegex(ValueError,"boundar"):
            replace(g,outer_boundary="vacuum")
        with self.assertRaises(ValueError):
            rz_to_spatial_families(g,source_id="synthetic",group_speeds_m_s=[2.],
                beta_by_incident_group_and_family=np.ones((2,1,2)),
                delayed_spectra_by_family=np.ones((12,2,1)),
                decay_constants_s_inv=[.1,1.])


if __name__ == '__main__':
    unittest.main(verbosity=2)
