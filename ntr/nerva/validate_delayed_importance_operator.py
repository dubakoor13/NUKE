"""Synthetic, independent operator tests: no physical reactor parameters."""
from __future__ import annotations
import unittest
import numpy as np
from .delayed_importance_operator import delayed_importance


def case(phi=None, psi=None, beta=None, prompt=None, delayed=None, xs=None, volumes=None):
    phi=np.array([[3., 1.]],dtype=float) if phi is None else np.asarray(phi,float)
    n,g=phi.shape
    psi=np.array([[1.,5.]]) if psi is None else np.asarray(psi,float)
    beta=np.array([[0.1,0.05]]) if beta is None else np.asarray(beta,float)
    prompt=np.array([[1.,0.]]) if prompt is None else np.asarray(prompt,float)
    delayed=np.array([[0.,1.]]) if delayed is None else np.asarray(delayed,float)
    xs=np.array([[2.,4.]]) if xs is None else np.asarray(xs,float)
    volumes=np.array([2.]) if volumes is None else np.asarray(volumes,float)
    return delayed_importance(phi,psi,volumes,xs,beta,prompt,delayed)


class DelayedImportanceTests(unittest.TestCase):
    def test_two_group_hand_calculated_emission(self):
        r=case()
        self.assertAlmostEqual(r.beta_effective,4./13.2)
        self.assertAlmostEqual(r.beta_unweighted,0.8/10.)
        np.testing.assert_allclose(r.prompt_emission,[[18.4,0.]])
        np.testing.assert_allclose(r.delayed_emission,[[0.,1.6]])
        self.assertAlmostEqual(r.prompt_weighted_production,18.4)
        self.assertAlmostEqual(r.delayed_weighted_production,8.)
        self.assertAlmostEqual(r.beta_effective,8./26.4)

    def test_single_group_identity(self):
        r=case(phi=[[2.]],psi=[[7.]],beta=[[.0065]],xs=[[0.3]],volumes=[4.],prompt=[[1.]],delayed=[[1.]])
        self.assertAlmostEqual(r.beta_effective,.0065)
        self.assertAlmostEqual(r.beta_unweighted,.0065)
        self.assertAlmostEqual(r.importance_ratio,1.)

    def test_adjoints_scale_out(self):
        a=case()
        b=case(psi=[[2.,10.]])
        self.assertAlmostEqual(a.beta_effective,b.beta_effective)
        self.assertAlmostEqual(b.total_weighted_production,2*a.total_weighted_production)

    def test_forward_flux_scales_out(self):
        a=case()
        b=case(phi=[[30.,10.]])
        self.assertAlmostEqual(a.beta_effective,b.beta_effective)
        self.assertAlmostEqual(a.beta_unweighted,b.beta_unweighted)

    def test_effective_exceeds_unweighted_when_delayed_high_importance(self):
        r=case()
        self.assertGreater(r.importance_ratio,1.)

    def test_effective_below_unweighted_when_delayed_low_importance(self):
        r=case(psi=[[5.,1.]])
        self.assertLess(r.importance_ratio,1.)

    def test_identical_spectra_uniform_importance_gives_unweighted(self):
        r=case(psi=[[2.,2.]],delayed=[[.7,.3]],prompt=[[.7,.3]])
        self.assertAlmostEqual(r.beta_effective,r.beta_unweighted)

    def test_source_zero_delayed_is_zero(self):
        r=case(beta=[[0.,0.]])
        self.assertEqual(r.beta_effective,0.)
        self.assertTrue(np.isnan(r.importance_ratio))

    def test_two_cells_manual_volume_weighting(self):
        r=case(phi=[[3.,1.],[3.,1.]],psi=[[1.,5.],[5.,1.]],
               beta=[[.1,.05],[.1,.05]],xs=[[2.,4.],[2.,4.]],
               volumes=[2.,6.],prompt=[[1.,0.],[1.,0.]],delayed=[[0.,1.],[0.,1.]])
        self.assertAlmostEqual(r.total_unweighted_production,80.)
        self.assertAlmostEqual(r.delayed_unweighted_production,6.4)
        self.assertAlmostEqual(r.delayed_weighted_production,2.*.8*5+6.*.8*1)
        self.assertAlmostEqual(r.prompt_weighted_production,2.*9.2*1+6.*9.2*5)

    def test_group_contribution_closure(self):
        r=case()
        self.assertAlmostEqual(sum(r.beta_group_contributions),r.beta_effective)
        self.assertEqual(len(r.beta_group_contributions),2)

    def test_invalid_spectra_guard(self):
        with self.assertRaisesRegex(ValueError,"prompt birth spectrum"):
            case(prompt=[[.5,.3]])
        with self.assertRaisesRegex(ValueError,"delayed birth spectrum"):
            case(delayed=[[.4,.7]])

    def test_bad_dimensions_and_values_guard(self):
        cases=[dict(psi=[[1.,2.,3.]]),dict(beta=[[-.1,.1]]),dict(xs=[[0.,-1.]]),
               dict(volumes=[-1.]),dict(phi=[[0.,0.]]),dict(beta=[[1.,.1]])]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):case(**kwargs)


if __name__=='__main__':unittest.main(verbosity=2)
