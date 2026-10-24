"""Independent synthetic tests for generalized prompt + delayed eigen operators.

A dense generalized eigensolver, direct per-cell operator arithmetic and two
existing independent transient methods are used as external numerical oracles.
No historical NERVA material parameters or real nuclear transport data.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np
from scipy.linalg import eig
from scipy.sparse import csc_matrix

from .fx2th_full_family_fission_eigen import (
    assemble_family_fission, solve_family_eigen,
    verify_stationary_family_eigen,
)
from .fx2th_family_spacetime import (
    SpatialFamilyData, stationary_precursors,
    advance_spatial_families,
)
from .fx2th_rz_family_diffusion import (
    assemble_rz,solve_rz_keff,
    rz_critical_family_reference,
)
from .validate_fx2th_rz_family_diffusion import grid,family_data
from .validate_fx2th_family_spacetime import model


def unequal_births(*, nr=2,nz=3,boundary="reflecting"):
    g=grid(nr,nz,ng=2,boundary=boundary)
    d=family_data(g)
    beta=np.empty_like(d.beta_by_incident_group_and_family)
    beta[...,0]=.16
    beta[...,1]=.07
    beta[:,0,1]=.11
    chi=np.array(d.delayed_spectra_by_family,copy=True)
    chi[:,0,:]=[.12,.88]
    chi[:,1,:]=[.92,.08]
    return g,replace(d,beta_by_incident_group_and_family=beta,
                     delayed_spectra_by_family=chi)


def direct_births(data,flux):
    """Direct loops independent of the new block-diagonal matrix builder."""
    cells,groups,fams=data.cells,data.groups,data.families
    p=np.zeros((cells,groups))
    delayed=np.zeros((fams,cells,groups))
    for c in range(cells):
        for incident in range(groups):
            strength=(data.volumes_m3[c]*data.nu_sigma_f_inv_m[c,incident]*
                      flux[c,incident])
            for emitted in range(groups):
                p[c,emitted]+=strength*(1-sum(
                    data.beta_by_incident_group_and_family[c,incident,:]))*(
                    data.prompt_spectrum[c,emitted])
                for f in range(fams):
                    delayed[f,c,emitted]+=strength*(
                        data.beta_by_incident_group_and_family[c,incident,f])*(
                        data.delayed_spectra_by_family[c,f,emitted])
    return p,delayed


def dominant_generalized_dense(A,F):
    values=eig(F.toarray(),A.toarray(),left=False,right=False)
    physical=[float(z.real) for z in values if abs(z.imag)<1e-8
              and z.real>0 and np.isfinite(z.real)]
    return max(physical)


class FullFamilyEigenTests(unittest.TestCase):
    def test_prompt_delayed_matrix_matches_independent_loop(self):
        _,d=unequal_births()
        mats=assemble_family_fission(d)
        phi=np.arange(1,d.cells*d.groups+1,dtype=float).reshape(d.cells,d.groups)*.1
        p,dfam=direct_births(d,phi)
        np.testing.assert_allclose((mats.prompt@phi.flatten()).reshape(phi.shape),p,rtol=2e-14)
        for f,mat in enumerate(mats.delayed_by_family):
            np.testing.assert_allclose((mat@phi.flatten()).reshape(phi.shape),
                                       dfam[f],rtol=2e-14)
        np.testing.assert_allclose((mats.total@phi.flatten()).reshape(phi.shape),
                                   p+dfam.sum(axis=0),rtol=2e-14)

    def test_fission_production_integrated_exactly_one_volume(self):
        _,d=unequal_births()
        phi=np.arange(1,d.cells*d.groups+1,dtype=float).reshape(d.cells,d.groups)*.15
        m=assemble_family_fission(d)
        expected=float(np.sum(d.volumes_m3[:,None]*
                              d.nu_sigma_f_inv_m*phi))
        self.assertAlmostEqual(float(np.sum(m.total@phi.flatten())),expected,places=11)

    def test_distinct_birth_spectra_change_fission_operator(self):
        g,d=unequal_births()
        new=assemble_family_fission(d).total
        original=assemble_rz(g).fission
        self.assertGreater(np.max(np.abs((new-original).toarray())),1e-3)
        self.assertGreater(abs(solve_family_eigen(d).keff-solve_rz_keff(g).keff),1e-5)

    def test_common_birth_spectra_equal_original_rz_operator(self):
        g=grid(2,3,ng=2)
        d=family_data(g)
        new=assemble_family_fission(d)
        old=assemble_rz(g)
        np.testing.assert_allclose(new.total.toarray(),old.fission.toarray(),
                                   atol=2e-14,rtol=2e-14)
        self.assertAlmostEqual(solve_family_eigen(d).keff,solve_rz_keff(g).keff,places=9)

    def test_zero_delayed_fraction_reduces_to_prompt_fission(self):
        _,d=unequal_births()
        d=replace(d,beta_by_incident_group_and_family=np.zeros_like(
                  d.beta_by_incident_group_and_family))
        m=assemble_family_fission(d)
        np.testing.assert_array_equal(m.total.toarray(),m.prompt.toarray())
        for x in m.delayed_by_family:
            self.assertEqual(x.nnz,0)
        r=solve_family_eigen(d)
        self.assertEqual(sum(r.beta_effective_by_family),0.)

    def test_generalized_eigenvalue_agrees_with_independent_dense(self):
        for boundary in ("reflecting","zero_flux"):
            with self.subTest(boundary=boundary):
                _,d=unequal_births(nr=2,nz=2,boundary=boundary)
                f=assemble_family_fission(d).total
                got=solve_family_eigen(d)
                dense=dominant_generalized_dense(d.integrated_loss,f)
                self.assertAlmostEqual(got.keff,dense,places=9)
                self.assertLess(got.forward_residual,1e-9)
                self.assertLess(got.adjoint_residual,1e-9)

    def test_forward_and_left_generalized_operator_residuals(self):
        _,d=unequal_births()
        x=solve_family_eigen(d)
        p=x.forward_flux.flatten();q=x.adjoint_flux.flatten()
        F=assemble_family_fission(d).total
        A=d.integrated_loss
        np.testing.assert_allclose(A@p,F@p/x.keff,rtol=1e-9,atol=1e-11)
        np.testing.assert_allclose(A.T@q,F.T@q/x.keff,rtol=1e-9,atol=1e-11)
        self.assertAlmostEqual(float(q@(F@p)),1.,places=9)
        self.assertAlmostEqual(x.fission_adjoint_pairing,1.,places=9)

    def test_effective_family_fractions_match_direct_adjoint_matrix_products(self):
        _,d=unequal_births()
        x=solve_family_eigen(d)
        p=x.forward_flux.flatten();q=x.adjoint_flux.flatten()
        ms=assemble_family_fission(d)
        denom=float(q@(ms.total@p))
        by_family=tuple(float(q@(mat@p))/denom for mat in ms.delayed_by_family)
        np.testing.assert_allclose(x.beta_effective_by_family,by_family,
                                   atol=1e-12,rtol=1e-12)
        raw=tuple(float(np.sum(mat@p))/float(np.sum(ms.total@p))
                  for mat in ms.delayed_by_family)
        np.testing.assert_allclose(x.beta_unweighted_by_family,raw,
                                   atol=1e-12,rtol=1e-12)
        self.assertGreater(abs(sum(raw)-sum(by_family)),1e-5)

    def test_effective_beta_can_be_above_or_below_unweighted(self):
        _,d=unequal_births()
        x=solve_family_eigen(d)
        self.assertGreater(x.prompt_generation_time_s,0.)
        self.assertGreater(sum(x.beta_effective_by_family),0.)
        self.assertLess(sum(x.beta_effective_by_family),1.)
        self.assertAlmostEqual(x.prompt_fission_births+x.delayed_fission_births,
                               x.total_fission_births,places=11)

    def test_decay_constants_do_not_change_equilibrium_fission_eigenvalue(self):
        _,d=unequal_births()
        changed=replace(d,decay_constants_s_inv=np.array([.006,15.]))
        x=solve_family_eigen(d)
        y=solve_family_eigen(changed)
        self.assertAlmostEqual(x.keff,y.keff,places=11)
        np.testing.assert_allclose(x.forward_flux,y.forward_flux,rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(x.adjoint_flux,y.adjoint_flux,rtol=1e-11,atol=1e-12)
        before=stationary_precursors(d,x.forward_flux)
        after=stationary_precursors(changed,y.forward_flux)
        self.assertGreater(np.max(np.abs(before-after)),.01)

    def test_critical_different_spectra_is_stationary_for_both_integrators(self):
        _,d=unequal_births()
        reference=solve_family_eigen(d)
        critical=replace(d,nu_sigma_f_inv_m=d.nu_sigma_f_inv_m/reference.keff)
        x=verify_stationary_family_eigen(critical,timestep_s=.3)
        self.assertAlmostEqual(x.reference.keff,1.,places=9)
        self.assertLess(x.implicit_euler_flux_error,1e-9)
        self.assertLess(x.implicit_euler_precursor_error,1e-9)
        self.assertLess(x.sdirk2_flux_error,1e-9)
        self.assertLess(x.sdirk2_precursor_error,1e-9)
        self.assertLess(x.sdirk2_balance_residual,1e-10)

    def test_old_common_spectrum_reference_rejects_new_valid_equilibrium(self):
        g,d=unequal_births()
        reference=solve_family_eigen(d)
        critical=replace(d,nu_sigma_f_inv_m=d.nu_sigma_f_inv_m/reference.keff)
        with self.assertRaisesRegex(ValueError,"distinct family birth spectra"):
            rz_critical_family_reference(g,critical)
        self.assertIsNotNone(verify_stationary_family_eigen(critical).initial)

    def test_noncritical_mode_not_claimed_stationary(self):
        _,d=unequal_births()
        with self.assertRaisesRegex(ValueError,"near 1"):
            verify_stationary_family_eigen(d,critical_tolerance=1e-10)

    def test_prompt_only_unequal_cell_volumes(self):
        d=model()
        original=assemble_family_fission(d)
        p=np.ones((d.cells,d.groups))
        q=np.asarray(original.total@p.flatten()).reshape(p.shape)
        _,fam=direct_births(d,p)
        self.assertAlmostEqual(float(fam.sum())+float((original.prompt@p.flatten()).sum()),
                               float(q.sum()),places=12)

    def test_nonuniform_cylindrical_fv_geometry_preserved(self):
        g,d=unequal_births(nr=3,nz=2,boundary="zero_flux")
        g=replace(g,radial_edges_m=np.array([0.,.11,.43,1.]),
                  axial_edges_m=np.array([0.,.53,2.]))
        from .fx2th_rz_family_diffusion import rz_to_spatial_families
        new=rz_to_spatial_families(g,source_id=d.source_id,
              group_speeds_m_s=d.group_speeds_m_s,
              beta_by_incident_group_and_family=d.beta_by_incident_group_and_family,
              delayed_spectra_by_family=d.delayed_spectra_by_family,
              decay_constants_s_inv=d.decay_constants_s_inv)
        x=solve_family_eigen(new)
        self.assertLess(x.forward_residual,1e-9)
        self.assertLess(x.adjoint_residual,1e-9)
        self.assertEqual(x.forward_flux.shape,(6,2))
        self.assertGreater(np.ptp(new.volumes_m3),.1)

    def test_tolerance_and_invalid_kinetics_guards(self):
        _,d=unequal_births()
        for opts in ({"tolerance":0.},{"max_iterations":0},
                     {"tolerance":float("nan")}):
            with self.subTest(opts=opts),self.assertRaises(ValueError):
                solve_family_eigen(d,**opts)
        with self.assertRaises(ValueError):
            verify_stationary_family_eigen(d,timestep_s=-.1)

    def test_fission_spectrum_power_conservation_independent(self):
        _,d=unequal_births()
        xs=np.asarray(d.nu_sigma_f_inv_m)
        p=np.arange(1,xs.size+1,dtype=float).reshape(xs.shape)
        matrices=assemble_family_fission(d)
        total=np.asarray(matrices.total@p.flatten()).reshape(xs.shape)
        reference=np.sum(d.volumes_m3[:,None]*xs*p)
        self.assertAlmostEqual(float(total.sum()),float(reference),places=11)

    def test_stationary_state_unaffected_by_timestep_choice(self):
        _,d=unequal_births(nr=2,nz=2)
        k=solve_family_eigen(d).keff
        critical=replace(d,nu_sigma_f_inv_m=d.nu_sigma_f_inv_m/k)
        a=verify_stationary_family_eigen(critical,timestep_s=.01)
        b=verify_stationary_family_eigen(critical,timestep_s=.8)
        np.testing.assert_allclose(a.initial.flux,b.initial.flux,
                                   rtol=1e-11,atol=1e-12)
        np.testing.assert_allclose(a.initial.precursors,b.initial.precursors,
                                   rtol=1e-11,atol=1e-12)

    def test_fission_source_id_retained(self):
        _,d=unequal_births()
        x=solve_family_eigen(d)
        self.assertEqual(x.source_id,d.source_id)


if __name__=="__main__":
    unittest.main(verbosity=2)
