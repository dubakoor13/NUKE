"""Synthetic unit tests for delayed-family kinetics data crosswalk."""
from __future__ import annotations
import unittest
import numpy as np
from .delayed_family_contract import family_importance, legacy_spacetime_contract
from .delayed_importance_operator import delayed_importance


def basic():
    return dict(source_id="synthetic-two-family-test-v1",
                forward_flux=np.array([[3.,1.],[2.,4.]]),
                adjoint_flux=np.array([[1.,5.],[3.,2.]]),
                cell_volumes_m3=np.array([2.,4.]),
                nu_sigma_f_inv_m=np.array([[2.,4.],[1.,3.]]),
                beta_by_incident_group_and_family=np.array([
                    [[.03,.07],[.015,.035]],
                    [[.005,.020],[.025,.035]],
                ]),
                prompt_spectrum=np.array([[1.,0.],[.5,.5]]),
                delayed_spectra_by_family=np.array([
                    [[0.,1.],[1.,0.]],
                    [[.25,.75],[.75,.25]],
                ]),
                decay_constants_s_inv=np.array([.1,2.]))


class DelayedFamilyContractTests(unittest.TestCase):
    def test_family_partition_matches_direct_tensor_contraction(self):
        p=basic();r=family_importance(**p)
        fission=p['forward_flux']*p['nu_sigma_f_inv_m']
        births=np.einsum('cg,cgf->cf',fission,p['beta_by_incident_group_and_family'])
        emitted=births[:,:,None]*p['delayed_spectra_by_family']*p['cell_volumes_m3'][:,None,None]
        weighted=np.einsum('cg,cfg->f',p['adjoint_flux'],emitted)
        np.testing.assert_allclose(r.delayed_emission_by_family,emitted)
        np.testing.assert_allclose(r.weighted_delayed_by_family,weighted)
        np.testing.assert_allclose(r.beta_effective_by_family,weighted/r.weighted_total)
        self.assertAlmostEqual(sum(r.beta_effective_by_family),r.beta_effective)
        self.assertAlmostEqual(sum(r.beta_unweighted_by_family),r.beta_unweighted)

    def test_total_birth_neutron_conservation(self):
        p=basic();r=family_importance(**p)
        total=float(np.sum(p['cell_volumes_m3'][:,None]*p['forward_flux']*p['nu_sigma_f_inv_m']))
        self.assertAlmostEqual(r.prompt_emission.sum()+r.delayed_emission_by_family.sum(),total)

    def test_equal_family_spectra_reduces_exactly_to_stage5_operator(self):
        p=basic();p['delayed_spectra_by_family']=np.tile([[.3,.7]],(2,2,1))
        f=family_importance(**p)
        s=delayed_importance(p['forward_flux'],p['adjoint_flux'],p['cell_volumes_m3'],
                             p['nu_sigma_f_inv_m'],
                             p['beta_by_incident_group_and_family'].sum(axis=-1),
                             p['prompt_spectrum'],p['delayed_spectra_by_family'][:,0,:])
        self.assertAlmostEqual(f.beta_effective,s.beta_effective)
        self.assertAlmostEqual(f.beta_unweighted,s.beta_unweighted)
        np.testing.assert_allclose(f.prompt_emission,s.prompt_emission)
        np.testing.assert_allclose(f.delayed_emission_by_family.sum(axis=1),s.delayed_emission)

    def test_single_group_adjoint_cancels(self):
        p=dict(source_id='synthetic',forward_flux=[[2.],[3.]],adjoint_flux=[[4.],[4.]],
               cell_volumes_m3=[1.,3.],nu_sigma_f_inv_m=[[2.],[4.]],
               beta_by_incident_group_and_family=[[[.001,.004]],[[.008,.012]]],
               prompt_spectrum=[[1.],[1.]],delayed_spectra_by_family=[[[1.],[1.]],[[1.],[1.]]],
               decay_constants_s_inv=[.1,1.0]);r=family_importance(**p)
        self.assertAlmostEqual(r.beta_effective,r.beta_unweighted)
        np.testing.assert_allclose(r.beta_effective_by_family,r.beta_unweighted_by_family)

    def test_flux_and_adjoint_global_gauge_invariance(self):
        p=basic();r=family_importance(**p)
        p['forward_flux']=p['forward_flux']*123.
        p['adjoint_flux']=p['adjoint_flux']*7.
        other=family_importance(**p)
        np.testing.assert_allclose(r.beta_effective_by_family,other.beta_effective_by_family,atol=1e-14)
        self.assertAlmostEqual(other.weighted_total,r.weighted_total*861.)

    def test_family_rates_remain_distinct(self):
        p=basic();r=family_importance(**p)
        self.assertEqual(r.decay_constants_s_inv,(.1,2.))
        self.assertEqual(len(r.beta_effective_by_family),2)
        self.assertNotAlmostEqual(r.beta_effective_by_family[0],r.beta_effective_by_family[1])

    def test_stage1_kinetics_named_contract(self):
        r=family_importance(**basic())
        k=r.point_kinetics_kwargs(generation_time_s=.2,reactivity=0.,source_amplitude_per_s=0.)
        self.assertEqual(k['beta_groups'],r.beta_effective_by_family)
        self.assertEqual(k['lambda_groups_s_inv'],(.1,2.))
        self.assertEqual(k['generation_time_s'],.2)

    def test_stage1_contract_does_not_infer_generation_time_or_reactivity(self):
        r=family_importance(**basic())
        with self.assertRaises(TypeError):r.point_kinetics_kwargs(reactivity=0.,source_amplitude_per_s=0.)
        with self.assertRaises(ValueError):r.point_kinetics_kwargs(generation_time_s=0.,reactivity=0.,source_amplitude_per_s=0.)
        with self.assertRaises(ValueError):r.point_kinetics_kwargs(generation_time_s=.2,reactivity=1.,source_amplitude_per_s=0.)

    def test_stage3_legacy_bridge_only_exact_representability(self):
        p=basic();p['beta_by_incident_group_and_family']=np.broadcast_to([.002,.003],(2,2,2)).copy()
        p['prompt_spectrum']=np.array([[.8,.2],[.8,.2]])
        p['delayed_spectra_by_family']=np.tile([[.8,.2]],(2,2,1))
        r=family_importance(**p)
        kw=legacy_spacetime_contract(family_result=r,
             beta_by_incident_group_and_family=p['beta_by_incident_group_and_family'],
             delayed_spectra_by_family=p['delayed_spectra_by_family'],
             prompt_spectrum=p['prompt_spectrum'],grid_fission_spectrum=p['prompt_spectrum'])
        self.assertEqual(kw['beta'],(.002,.003))
        self.assertEqual(kw['decay_s_inv'],(.1,2.))

    def test_stage3_different_family_birth_spectra_are_rejected(self):
        p=basic();r=family_importance(**p)
        with self.assertRaisesRegex(ValueError,'distinct delayed/prompt'):
            legacy_spacetime_contract(family_result=r,
                beta_by_incident_group_and_family=np.broadcast_to([.002,.003],(2,2,2)).copy(),
                delayed_spectra_by_family=p['delayed_spectra_by_family'],
                prompt_spectrum=p['prompt_spectrum'],grid_fission_spectrum=p['prompt_spectrum'])

    def test_stage3_nonuniform_beta_is_rejected(self):
        p=basic();p['prompt_spectrum']=np.tile([.8,.2],(2,1));p['delayed_spectra_by_family']=np.tile([.8,.2],(2,2,1))
        r=family_importance(**p)
        with self.assertRaisesRegex(ValueError,'beta constant'):
            legacy_spacetime_contract(family_result=r,
                beta_by_incident_group_and_family=p['beta_by_incident_group_and_family'],
                delayed_spectra_by_family=p['delayed_spectra_by_family'],
                prompt_spectrum=p['prompt_spectrum'],grid_fission_spectrum=p['prompt_spectrum'])

    def test_invalid_group_sums_and_shapes(self):
        p=basic()
        variations=[('spectrum',dict(prompt_spectrum=[[.9,.2],[.5,.5]])),
                    ('family',dict(delayed_spectra_by_family=[[[.5,.8],[1.,0.]],[[.25,.75],[.75,.25]]])),
                    ('fraction',dict(beta_by_incident_group_and_family=[[[.6,.5],[.01,.02]],[[.02,.03],[.03,.04]]])),
                    ('decay',dict(decay_constants_s_inv=[.1,0.])),
                    ('volume',dict(cell_volumes_m3=[0.,4.])),
                    ('shape',dict(beta_by_incident_group_and_family=[[[.1,.2]]])),
                    ('empty source',dict(source_id=' '))]
        for name,change in variations:
            with self.subTest(name=name),self.assertRaises(ValueError):family_importance(**{**p,**change})

    def test_zero_fission_fails_closed(self):
        p=basic();p['nu_sigma_f_inv_m']=np.zeros_like(p['nu_sigma_f_inv_m'])
        with self.assertRaisesRegex(ValueError,'production'):
            family_importance(**p)

    def test_zero_importance_fails_closed(self):
        p=basic();p['adjoint_flux']=np.zeros_like(p['adjoint_flux'])
        with self.assertRaisesRegex(ValueError,'production'):
            family_importance(**p)

    def test_all_zero_beta_allowed_but_ratio_unavailable(self):
        p=basic();p['beta_by_incident_group_and_family']=np.zeros_like(p['beta_by_incident_group_and_family'])
        r=family_importance(**p)
        self.assertEqual(r.beta_effective,0.)
        self.assertTrue(np.isnan(r.importance_ratio))


if __name__=='__main__':unittest.main(verbosity=2)
