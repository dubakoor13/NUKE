"""Synthetic space-time, spatial delayed-neutron family validation."""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from .fx2th_family_spacetime import (
    SpatialFamilyData, SpatialFamilyState, advance_spatial_families,
    evolve_external_states, importance_snapshot, stationary_precursors,
)


def model(*, cells: int = 2, groups: int = 2, families: int = 2) -> SpatialFamilyData:
    if cells == 1 and groups == 1:
        return SpatialFamilyData(
            "test-one-cell-one-group",csc_matrix([[0.8]]),
            np.array([2.]),np.array([2.]),
            np.array([[0.4]]),np.array([[[.03,.02]]]),
            np.array([[1.]]),np.array([[[1.],[1.]]]),
            np.array([.2,1.5])
        )
    if (cells,groups,families)!=(2,2,2):
        raise ValueError("test helper only supports 2x2x2")
    A = np.array([
        [1.2, -.20, -.10, 0.],
        [-.05, .85, 0., -.10],
        [-.10, 0., 1.0, -.10],
        [0., -.10, -.03, .90]
    ])
    return SpatialFamilyData(
        "synthetic-material-schedule-v1",csc_matrix(A),
        np.array([2.,3.]),np.array([1.5,3.0]),
        np.array([[.50,.35],[.40,.30]]),
        np.array([[[.03,.02],[.02,.01]],[[.015,.015],[.02,.025]]]),
        np.array([[.8,.2],[.65,.35]]),
        np.array([[[.2,.8],[.7,.3]],[[.1,.9],[.6,.4]]]),
        np.array([.20,1.50]),
    )


def initial(d: SpatialFamilyData) -> SpatialFamilyState:
    flux=np.array([[1.,2.],[2.,1.]]) if d.cells==2 else np.array([[4.]])
    C=stationary_precursors(d,flux)
    return SpatialFamilyState(0.,flux,C)


def dense_generator(d: SpatialFamilyData) -> np.ndarray:
    """Independently construct the full ODE Jacobian WITHOUT Schur elimination."""
    nc, ng, nf=d.cells,d.groups,d.families
    nphi=nc*ng
    H=np.zeros((nphi+nc*nf,nphi+nc*nf))
    vol=np.asarray(d.volumes_m3)
    beta=np.asarray(d.beta_by_incident_group_and_family)
    lam=np.asarray(d.decay_constants_s_inv)
    M=d.mass_diagonal
    A=d.integrated_loss.toarray()
    H[:nphi,:nphi] = -A/M[:,None]
    for c in range(nc):
        for gout in range(ng):
            row=c*ng+gout
            for gin in range(ng):
                col=c*ng+gin
                rate=vol[c]*d.nu_sigma_f_inv_m[c,gin]
                H[row,col]+=rate*(1-beta[c,gin].sum())*d.prompt_spectrum[c,gout]/M[row]
            for f in range(nf):
                col_c=nphi+c*nf+f
                H[row,col_c]+=lam[f]*d.delayed_spectra_by_family[c,f,gout]/M[row]
        for f in range(nf):
            row=nphi+c*nf+f
            H[row,row]-=lam[f]
            for gin in range(ng):
                H[row,c*ng+gin]+=vol[c]*beta[c,gin,f]*d.nu_sigma_f_inv_m[c,gin]
    return H


class SpatialFamilyTests(unittest.TestCase):
    def test_critical_one_cell_preserves_family_inventories(self):
        d=model(cells=1,groups=1)
        s=initial(d)
        a=advance_spatial_families(d,s,2.)
        np.testing.assert_allclose(a.state.flux,s.flux,rtol=1e-12)
        np.testing.assert_allclose(a.state.precursors,s.precursors,rtol=1e-12)
        self.assertLess(a.flux_equation_residual,1e-12)
        self.assertLess(a.precursor_equation_residual,1e-12)

    def test_explicit_matrix_exponential_reference_and_refinement(self):
        d=model()
        s=initial(d)
        q=np.r_[s.flux.flatten(),s.precursors.flatten()]
        exact=expm(dense_generator(d)*0.4)@q
        def integrate(n):
            x=s
            for _ in range(n): x=advance_spatial_families(d,x,.4/n).state
            return np.r_[x.flux.flatten(),x.precursors.flatten()]
        coarse=integrate(10)
        fine=integrate(160)
        ec=np.linalg.norm(coarse-exact)
        ef=np.linalg.norm(fine-exact)
        self.assertLess(ef,ec/8.)
        self.assertLess(ef,0.01)

    def test_exact_backward_euler_dense_matrix_solution(self):
        d=model();s=initial(d);dt=.25
        y=np.r_[s.flux.flatten(),s.precursors.flatten()]
        exact=np.linalg.solve(np.eye(len(y))-dt*dense_generator(d),y)
        step=advance_spatial_families(d,s,dt)
        np.testing.assert_allclose(
            np.r_[step.state.flux.flatten(),step.state.precursors.flatten()],
            exact,rtol=3e-12,atol=3e-12)

    def test_full_balance_neutron_and_precursor_equations(self):
        d=model();s=initial(d)
        a=advance_spatial_families(d,s,.2)
        self.assertLess(a.flux_equation_residual,1e-12)
        self.assertLess(a.precursor_equation_residual,1e-12)
        self.assertLess(a.global_balance_residual,1e-12)
        self.assertLess(abs(a.absolute_global_balance),1e-10)

    def test_precursor_spatial_shifts_are_not_collapsed(self):
        d=model();s=initial(d)
        a=advance_spatial_families(d,s,.3)
        self.assertEqual(a.state.precursors.shape,(2,2))
        self.assertFalse(np.allclose(a.state.precursors[0],a.state.precursors[1]))
        self.assertNotAlmostEqual(a.state.precursors[0,0],a.state.precursors[0,1])

    def test_independent_family_decay_rates(self):
        d=model();s=SpatialFamilyState(0.,np.zeros((2,2)),
                   np.array([[2.,3.],[1.,4.]]))
        d=replace(d,nu_sigma_f_inv_m=np.zeros((2,2)))
        a=advance_spatial_families(d,s,.5)
        expected=s.precursors/(1+.5*d.decay_constants_s_inv[None,:])
        np.testing.assert_allclose(a.state.precursors,expected,rtol=2e-14)

    def test_prompt_vs_delayed_family_spectrum_changes_solution(self):
        d=model();s=initial(d)
        a=advance_spatial_families(d,s,.7)
        d2=replace(d,delayed_spectra_by_family=
             np.tile(np.asarray(d.prompt_spectrum)[:,None,:],(1,d.families,1)))
        b=advance_spatial_families(d2,s,.7)
        self.assertGreater(np.linalg.norm(a.state.flux-b.state.flux),1e-4)

    def test_source_only_implicit_analytic(self):
        d=model(cells=1,groups=1)
        d=replace(d,nu_sigma_f_inv_m=np.array([[0.]]))
        s=SpatialFamilyState(0.,np.array([[1.]]),np.zeros((1,2)))
        q=np.array([[1.]])
        a=advance_spatial_families(d,s,.5,q)
        expected=(1./.5+2.)/(1./.5+.8)
        self.assertAlmostEqual(a.state.flux[0,0],expected)
        self.assertLess(a.global_balance_residual,1e-12)

    def test_critical_state_with_no_delay_families_but_zero_beta(self):
        d=model(cells=1,groups=1)
        d=replace(d,beta_by_incident_group_and_family=np.zeros((1,1,2)))
        s=SpatialFamilyState(0.,np.array([[4.]]),np.zeros((1,2)))
        a=advance_spatial_families(d,s,.25)
        np.testing.assert_allclose(a.state.flux,s.flux,atol=1e-12)
        np.testing.assert_allclose(a.state.precursors,0)

    def test_external_material_schedule_changes_solution(self):
        d=model();s=initial(d)
        a=evolve_external_states([d,d],s,.1)
        altered=replace(d,integrated_loss=d.integrated_loss+
                        csc_matrix(np.diag([.5,.5,.5,.5])))
        b=evolve_external_states([d,altered],s,.1)
        self.assertAlmostEqual(a[0].state.flux[0,0],b[0].state.flux[0,0])
        self.assertLess(b[-1].state.flux.sum(),a[-1].state.flux.sum())

    def test_mismatched_mesh_schedule_rejected(self):
        d=model()
        altered=replace(d,volumes_m3=np.array([2.,4.]))
        with self.assertRaisesRegex(ValueError,"mesh"):
            evolve_external_states([d,altered],initial(d),.1)

    def test_frozen_input_provenance_required(self):
        d=model()
        with self.assertRaisesRegex(ValueError,"source_id"):
            replace(d,source_id=" ")

    def test_stage6_family_importance_exact_adapter(self):
        d=model();s=initial(d);psi=np.array([[1.,2.],[2.,3.]])
        result=importance_snapshot(d,s,psi)
        self.assertEqual(result.decay_constants_s_inv,(.2,1.5))
        self.assertAlmostEqual(sum(result.beta_effective_by_family),result.beta_effective)
        self.assertEqual(result.source_id,d.source_id)

    def test_invalid_family_spectra_rejected(self):
        d=model()
        invalid=np.asarray(d.delayed_spectra_by_family).copy()
        invalid[0,1,0]=.8
        with self.assertRaisesRegex(ValueError,"spectrum"):
            replace(d,delayed_spectra_by_family=invalid)

    def test_invalid_group_fractions_rejected(self):
        d=model()
        beta=np.asarray(d.beta_by_incident_group_and_family).copy()
        beta[0,0]=[.7,.5]
        with self.assertRaisesRegex(ValueError,"beta"):
            replace(d,beta_by_incident_group_and_family=beta)

    def test_invalid_loss_matrix_rejected(self):
        d=model()
        wrong=np.asarray(d.integrated_loss.toarray())
        wrong[0,1]=+.1
        with self.assertRaisesRegex(ValueError,"off-diagonal"):
            replace(d,integrated_loss=csc_matrix(wrong))

    def test_bad_source_dimensions_or_negative_rejected(self):
        d=model();s=initial(d)
        for q in (np.array([1.,2.]),np.array([[-1.,1.],[1.,1.]])):
            with self.subTest(q=q),self.assertRaises(ValueError):
                advance_spatial_families(d,s,.2,q)
        with self.assertRaises(ValueError):
            advance_spatial_families(d,s,-1.)

    def test_incident_energy_dependence_not_lost(self):
        d=model();s=initial(d)
        raw=stationary_precursors(d,s.flux)
        independent=np.empty((d.cells,d.families))
        for c in range(d.cells):
            for f in range(d.families):
                production=sum(d.volumes_m3[c]*d.nu_sigma_f_inv_m[c,g]*
                               s.flux[c,g]*d.beta_by_incident_group_and_family[c,g,f]
                               for g in range(d.groups))
                independent[c,f]=production/d.decay_constants_s_inv[f]
        np.testing.assert_allclose(raw,independent,rtol=2e-14)

    def test_input_has_single_integration_of_volume(self):
        d=model();s=initial(d)
        expected=np.einsum("cg,cg,cgf,c->cf",s.flux,d.nu_sigma_f_inv_m,
                           d.beta_by_incident_group_and_family,d.volumes_m3)
        a=advance_spatial_families(d,s,.15)
        np.testing.assert_allclose(a.produced_precursors,
            np.einsum("cg,cg,cgf,c->cf",a.state.flux,d.nu_sigma_f_inv_m,
                      d.beta_by_incident_group_and_family,d.volumes_m3),rtol=2e-14)
        self.assertTrue(np.all(expected>0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
