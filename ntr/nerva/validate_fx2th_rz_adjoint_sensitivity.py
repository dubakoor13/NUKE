"""Independent numerical tests of Stage 16 adjoint reactor-core eigen sensitivity.

Finite differences use the OLD Stage-12 finite-volume assembler, Stage-15
fission assembler, and SciPy's independent DENSE generalized eigensolver.
The analytic derivatives are not used to calculate finite-difference targets.
All nuclear data/directions are synthetic, with NO NERVA thermal worth claim.
"""
from __future__ import annotations

from dataclasses import replace
import unittest
import numpy as np
from scipy.linalg import eigvals
from scipy.sparse import csc_matrix

from .fx2th_rz_adjoint_sensitivity import (
    assemble_rz_loss_derivative, assemble_family_fission_derivative,
    evaluate_core_eigen_sensitivity, evaluate_rz_material_sensitivity,
)
from .fx2th_full_family_fission_eigen import (
    assemble_family_fission,solve_family_eigen,
)
from .fx2th_rz_family_diffusion import assemble_rz,rz_to_spatial_families
from .validate_fx2th_full_family_fission_eigen import unequal_births
from .validate_fx2th_rz_family_diffusion import grid


def material_of(grid0, data):
    return rz_to_spatial_families(
        grid0,source_id=data.source_id,
        group_speeds_m_s=data.group_speeds_m_s,
        beta_by_incident_group_and_family=data.beta_by_incident_group_and_family,
        delayed_spectra_by_family=data.delayed_spectra_by_family,
        decay_constants_s_inv=data.decay_constants_s_inv,
        prompt_spectrum=data.prompt_spectrum,
    )


def dense_keff(data):
    A=data.integrated_loss.toarray()
    F=assemble_family_fission(data).total.toarray()
    e=eigvals(F,A)
    positive=[float(x.real) for x in e if abs(x.imag)<1e-9 and x.real>0]
    if not positive:
        raise ArithmeticError("no positive reference eigenvalue")
    return max(positive)


def perturb_material(g,d,epsilon,**directions):
    """Independently rebuild Stage-12 loss matrix and Stage-15 fission F."""
    name_grid={
        "d_diffusion_m":"diffusion_m",
        "d_absorption_inv_m":"absorption_inv_m",
        "d_scattering_inv_m":"scattering_inv_m",
    }
    name_data={
        "d_nu_sigma_f_inv_m":"nu_sigma_f_inv_m",
        "d_beta_by_incident_group_and_family":"beta_by_incident_group_and_family",
        "d_prompt_spectrum":"prompt_spectrum",
        "d_delayed_spectra_by_family":"delayed_spectra_by_family",
    }
    overrides={}
    for key,attr in name_grid.items():
        if key in directions:
            overrides[attr]=np.asarray(getattr(g,attr))+epsilon*directions[key]
    gp=replace(g,**overrides)
    out=material_of(gp,d)
    overrides={}
    for key,attr in name_data.items():
        if key in directions:
            overrides[attr]=np.asarray(getattr(out,attr))+epsilon*directions[key]
    return replace(out,**overrides),gp


def difference_against_dense(g,d,dirs,rtol=2e-5):
    eps=2e-5
    plus,_=perturb_material(g,d,eps,**dirs)
    minus,_=perturb_material(g,d,-eps,**dirs)
    fd=(dense_keff(plus)-dense_keff(minus))/(2*eps)
    computed=evaluate_rz_material_sensitivity(g,d,**dirs)
    np.testing.assert_allclose(computed.d_keff_d_parameter,fd,rtol=rtol,atol=2e-8)
    return computed,fd


def signed_arrays(g,d):
    diffusion=np.zeros_like(g.diffusion_m)
    diffusion[1,0,0]=.17
    absorption=np.zeros_like(g.absorption_inv_m)
    absorption[0,1,1]=.12
    scatter=np.zeros_like(g.scattering_inv_m)
    scatter[0,0,0,1]=.03
    nu=np.zeros_like(d.nu_sigma_f_inv_m)
    nu[-1,0]=.16
    beta=np.zeros_like(d.beta_by_incident_group_and_family)
    beta[0,1,0]=.045
    cp=np.zeros_like(d.prompt_spectrum)
    cp[0]=[.07,-.07]
    cd=np.zeros_like(d.delayed_spectra_by_family)
    cd[1,0]=[-.05,.05]
    return dict(d_diffusion_m=diffusion,d_absorption_inv_m=absorption,
                d_scattering_inv_m=scatter,d_nu_sigma_f_inv_m=nu,
                d_beta_by_incident_group_and_family=beta,
                d_prompt_spectrum=cp,d_delayed_spectra_by_family=cd)


class MaterialSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.grid,self.data=unequal_births(nr=2,nz=3,boundary="zero_flux")
        self.directions=signed_arrays(self.grid,self.data)

    def test_loss_tangent_matches_independent_fv_matrix_difference(self):
        g,d=self.grid,self.data
        names=("d_diffusion_m","d_absorption_inv_m","d_scattering_inv_m")
        dirs={n:self.directions[n] for n in names}
        tangent=assemble_rz_loss_derivative(g,**dirs).matrix
        h=1e-6
        gp=replace(g,
             diffusion_m=g.diffusion_m+h*dirs["d_diffusion_m"],
             absorption_inv_m=g.absorption_inv_m+h*dirs["d_absorption_inv_m"],
             scattering_inv_m=g.scattering_inv_m+h*dirs["d_scattering_inv_m"])
        gm=replace(g,
             diffusion_m=g.diffusion_m-h*dirs["d_diffusion_m"],
             absorption_inv_m=g.absorption_inv_m-h*dirs["d_absorption_inv_m"],
             scattering_inv_m=g.scattering_inv_m-h*dirs["d_scattering_inv_m"])
        fd=(assemble_rz(gp).loss-assemble_rz(gm).loss)/(2*h)
        np.testing.assert_allclose(tangent.toarray(),fd.toarray(),
                                   atol=2e-9,rtol=2e-8)

    def test_prompt_delayed_family_tangent_matches_independent_matrix_difference(self):
        g,d=self.grid,self.data
        names=("d_nu_sigma_f_inv_m","d_beta_by_incident_group_and_family",
               "d_prompt_spectrum","d_delayed_spectra_by_family")
        dirs={n:self.directions[n] for n in names}
        tangent=assemble_family_fission_derivative(d,**dirs)
        h=1e-6
        plus,_=perturb_material(g,d,h,**dirs)
        minus,_=perturb_material(g,d,-h,**dirs)
        mp=assemble_family_fission(plus)
        mm=assemble_family_fission(minus)
        np.testing.assert_allclose(tangent.total.toarray(),
                                  ((mp.total-mm.total)/(2*h)).toarray(),
                                  atol=1e-9,rtol=2e-8)
        for f in range(d.families):
            np.testing.assert_allclose(tangent.delayed_by_family[f].toarray(),
                ((mp.delayed_by_family[f]-mm.delayed_by_family[f])/(2*h)).toarray(),
                atol=1e-9,rtol=2e-8)

    def test_absorption_eigen_derivative_matches_dense(self):
        difference_against_dense(self.grid,self.data,
            {"d_absorption_inv_m":self.directions["d_absorption_inv_m"]})

    def test_diffusion_leakage_eigen_derivative_matches_dense(self):
        difference_against_dense(self.grid,self.data,
            {"d_diffusion_m":self.directions["d_diffusion_m"]})

    def test_group_scattering_eigen_derivative_matches_dense(self):
        difference_against_dense(self.grid,self.data,
            {"d_scattering_inv_m":self.directions["d_scattering_inv_m"]})

    def test_nu_fission_eigen_derivative_matches_dense(self):
        difference_against_dense(self.grid,self.data,
            {"d_nu_sigma_f_inv_m":self.directions["d_nu_sigma_f_inv_m"]})

    def test_incident_group_delayed_beta_derivative_matches_dense(self):
        computed,_=difference_against_dense(self.grid,self.data,
            {"d_beta_by_incident_group_and_family":
                self.directions["d_beta_by_incident_group_and_family"]})
        self.assertNotAlmostEqual(computed.d_keff_d_parameter,0.,places=8)

    def test_prompt_emission_energy_derivative_matches_dense(self):
        difference_against_dense(self.grid,self.data,
            {"d_prompt_spectrum":self.directions["d_prompt_spectrum"]})

    def test_delayed_family_emission_derivative_matches_dense(self):
        difference_against_dense(self.grid,self.data,
            {"d_delayed_spectra_by_family":
                self.directions["d_delayed_spectra_by_family"]})

    def test_mixed_all_material_directions_matches_dense(self):
        result,_=difference_against_dense(self.grid,self.data,self.directions)
        self.assertAlmostEqual(
            result.d_log_keff_d_parameter,
            result.loss_contribution_dlogk
            +result.prompt_fission_contribution_dlogk
            +sum(result.delayed_fission_contribution_dlogk),places=12)
        self.assertAlmostEqual(result.d_reactivity_d_parameter,
                               result.d_keff_d_parameter/(result.keff**2),places=12)

    def test_reactivity_derivative_matches_dense_independent_difference(self):
        g,d=self.grid,self.data
        dirs=self.directions
        h=2e-5
        plus,_=perturb_material(g,d,h,**dirs)
        minus,_=perturb_material(g,d,-h,**dirs)
        fd=((1.-1./dense_keff(plus))-(1.-1./dense_keff(minus)))/(2*h)
        result=evaluate_rz_material_sensitivity(g,d,**dirs)
        np.testing.assert_allclose(result.d_reactivity_d_parameter,fd,
                                   atol=1e-8,rtol=3e-5)

    def test_uniform_fission_scale_analytical_k_derivative(self):
        d=self.data
        tangent=assemble_family_fission_derivative(
            d,d_nu_sigma_f_inv_m=d.nu_sigma_f_inv_m)
        result=evaluate_core_eigen_sensitivity(
            d,csc_matrix(d.integrated_loss.shape),tangent)
        self.assertAlmostEqual(result.d_keff_d_parameter,result.keff,places=10)
        self.assertAlmostEqual(result.d_log_keff_d_parameter,1.,places=10)

    def test_uniform_one_group_reflected_absorption_analytic(self):
        g=grid(nr=1,nz=1,ng=1,boundary="reflecting",fission=.8)
        d=rz_to_spatial_families(g,source_id="synthetic-one-cell",
             group_speeds_m_s=np.array([2.]),
             beta_by_incident_group_and_family=np.zeros((1,1,2)),
             delayed_spectra_by_family=np.ones((1,2,1)),
             decay_constants_s_inv=np.array([.1,1.]))
        direction=np.ones_like(g.absorption_inv_m)
        output=evaluate_rz_material_sensitivity(g,d,d_absorption_inv_m=direction)
        self.assertAlmostEqual(output.keff,.8/.7,places=10)
        self.assertAlmostEqual(output.d_keff_d_parameter,-.8/(.7**2),places=10)

    def test_scattering_tangent_cancels_in_total_neutron_balance(self):
        g=self.grid
        mat=assemble_rz_loss_derivative(
            g,d_scattering_inv_m=self.directions["d_scattering_inv_m"]).matrix
        for phi in (np.ones(mat.shape[0]),np.arange(1,mat.shape[0]+1)*.1):
            self.assertAlmostEqual(float(np.sum(mat@phi)),0.,places=11)

    def test_diffusion_tangent_reflecting_constant_shape_zero(self):
        g=grid(nr=3,nz=4,ng=2,boundary="reflecting")
        tangent=np.ones_like(g.diffusion_m)*.3
        deriv=assemble_rz_loss_derivative(g,d_diffusion_m=tangent)
        self.assertLess(float(np.max(np.abs(deriv.matrix@np.ones(
            deriv.matrix.shape[0])))),1e-11)

    def test_one_cell_dirichlet_face_conductance_derivative(self):
        g=grid(nr=1,nz=1,ng=1,boundary="zero_flux")
        deriv=assemble_rz_loss_derivative(g,
             d_diffusion_m=np.ones_like(g.diffusion_m)).matrix.toarray()[0,0]
        # Radius R=1, height H=2, dr=1, dz=2, D perturb 1 m:
        # radial conductance derivative = (2*pi*R*H)*2/dr = 8*pi
        # two axial conductances = 2 * (pi*R**2)*2/dz = 2*pi
        self.assertAlmostEqual(deriv,10*np.pi,places=10)

    def test_beta_only_tangent_conserves_total_emitted_births(self):
        d=self.data
        deriv=assemble_family_fission_derivative(d,
           d_beta_by_incident_group_and_family=
             self.directions["d_beta_by_incident_group_and_family"]).total
        self.assertLess(float(np.max(np.abs(
            np.asarray(deriv.sum(axis=0)).ravel()))),1e-12)

    def test_spectrum_tangent_conserves_total_emitted_births(self):
        d=self.data
        deriv=assemble_family_fission_derivative(d,
             d_prompt_spectrum=self.directions["d_prompt_spectrum"],
             d_delayed_spectra_by_family=
                self.directions["d_delayed_spectra_by_family"]).total
        self.assertLess(float(np.max(np.abs(
            np.asarray(deriv.sum(axis=0)).ravel()))),1e-12)

    def test_zero_derivative_is_exact_zero(self):
        g,d=self.grid,self.data
        out=evaluate_rz_material_sensitivity(g,d)
        self.assertEqual(out.d_keff_d_parameter,0.)
        self.assertEqual(out.d_reactivity_d_parameter,0.)
        self.assertEqual(out.loss_contribution_dlogk,0.)

    def test_bad_input_shapes_and_spectrum_tangents_are_rejected(self):
        g,d=self.grid,self.data
        with self.assertRaisesRegex(ValueError,"shape"):
            assemble_rz_loss_derivative(g,d_diffusion_m=np.ones((3,3,3)))
        with self.assertRaisesRegex(ValueError,"zero diagonal"):
            assemble_rz_loss_derivative(g,d_scattering_inv_m=np.ones_like(
                g.scattering_inv_m))
        with self.assertRaisesRegex(ValueError,"sum to zero"):
            assemble_family_fission_derivative(d,d_prompt_spectrum=
                np.ones_like(d.prompt_spectrum))
        with self.assertRaises(ValueError):
            assemble_family_fission_derivative(d,d_nu_sigma_f_inv_m=
                np.full_like(d.nu_sigma_f_inv_m,np.nan))

    def test_rz_material_snapshot_mismatch_rejected(self):
        g,d=self.grid,self.data
        corrupted=replace(d,nu_sigma_f_inv_m=d.nu_sigma_f_inv_m*1.01)
        with self.assertRaisesRegex(ValueError,"differ"):
            evaluate_rz_material_sensitivity(g,corrupted)

    def test_parameter_linearity_of_adjoint_tangent(self):
        d=self.data
        dx=self.directions["d_nu_sigma_f_inv_m"]
        f1=assemble_family_fission_derivative(d,d_nu_sigma_f_inv_m=dx)
        f2=assemble_family_fission_derivative(d,d_nu_sigma_f_inv_m=2.75*dx)
        s1=evaluate_core_eigen_sensitivity(d,csc_matrix(d.integrated_loss.shape),f1)
        s2=evaluate_core_eigen_sensitivity(d,csc_matrix(d.integrated_loss.shape),f2)
        self.assertAlmostEqual(s2.d_keff_d_parameter,
                               2.75*s1.d_keff_d_parameter,places=11)


    def test_reject_stale_adjoint_and_forward_eigen_reference(self):
        d=self.data
        ref=solve_family_eigen(d)
        changed=replace(d,nu_sigma_f_inv_m=d.nu_sigma_f_inv_m*1.03)
        tangent=assemble_family_fission_derivative(
            changed,d_nu_sigma_f_inv_m=changed.nu_sigma_f_inv_m)
        with self.assertRaisesRegex(ValueError,"stale"):
            evaluate_core_eigen_sensitivity(
                changed,csc_matrix(changed.integrated_loss.shape),
                tangent,eigen=ref)

    def test_adjoints_and_forward_normalization_cancel_from_sensitivity(self):
        d=self.data
        f=assemble_family_fission_derivative(
            d,d_nu_sigma_f_inv_m=self.directions["d_nu_sigma_f_inv_m"])
        z=csc_matrix(d.integrated_loss.shape)
        ref=solve_family_eigen(d)
        first=evaluate_core_eigen_sensitivity(d,z,f,eigen=ref)
        scaled=replace(ref,forward_flux=ref.forward_flux*3.7,
                            adjoint_flux=ref.adjoint_flux*0.15)
        second=evaluate_core_eigen_sensitivity(d,z,f,eigen=scaled)
        self.assertAlmostEqual(first.d_keff_d_parameter,
                               second.d_keff_d_parameter,places=12)
        self.assertAlmostEqual(first.d_log_keff_d_parameter,
                               second.d_log_keff_d_parameter,places=12)

if __name__=="__main__":
    unittest.main(verbosity=2)
