"""Stage23 evidence tests on genuine 20 OpenMC statepoints and pure probability math.

Real-statepoint tests REQUIRE the executed original 2026-10-23 artifact
files, downloaded from GitHub Actions 39933618638. Mathematical source
histogram examples are NOT synthetic nuclear transport validation.
"""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .real_source_multiresolution import (
    coarsen_source_cube, spatial_tv, spatial_js,
    multiresolution_source_report,
)


class PureProbabilityCoarseningTests(unittest.TestCase):
    def test_probability_exactly_conserved_on_repeated_coarsening(self):
        rng=np.random.default_rng(731)
        cube=rng.random((16,16,16))
        cube/=cube.sum()
        for n in (2,4,8,16):
            result=coarsen_source_cube(cube,n)
            self.assertEqual(result.shape,(n,n,n))
            self.assertAlmostEqual(float(result.sum()),1.,places=14)
        np.testing.assert_allclose(
            coarsen_source_cube(coarsen_source_cube(cube,8),2),
            coarsen_source_cube(cube,2),rtol=0,atol=2e-16)

    def test_tv_and_information_contraction_for_coarsening(self):
        rng=np.random.default_rng(323)
        a=rng.random((16,16,16));a/=a.sum()
        b=rng.random((16,16,16));b/=b.sum()
        tv=[];js=[]
        for n in (2,4,8,16):
            pa=coarsen_source_cube(a,n)
            pb=coarsen_source_cube(b,n)
            tv.append(spatial_tv(pa,pb))
            js.append(spatial_js(pa,pb))
        self.assertTrue(all(x<=y+1e-13 for x,y in zip(tv,tv[1:])))
        self.assertTrue(all(x<=y+1e-13 for x,y in zip(js,js[1:])))
        self.assertLessEqual(tv[-1],1)
        self.assertLessEqual(js[-1],np.log(2))

    def test_identical_nonuniform_distributions_have_zero_divergence(self):
        c=np.zeros((4,4,4));c[1,1,1]=.75;c[2,1,1]=.25
        self.assertEqual(spatial_tv(c,c),0)
        self.assertEqual(spatial_js(c,c),0)

    def test_invalid_histograms_and_incompatible_grids_fail(self):
        a=np.ones((8,8,8))/512.
        for corrupt in (
            a*.2,np.zeros_like(a),
            np.full((8,8,8),np.nan),
            np.ones((3,4,5)),
        ):
            with self.subTest(shape=corrupt.shape),self.assertRaises(ValueError):
                coarsen_source_cube(corrupt,4)
        with self.assertRaises(ValueError):
            coarsen_source_cube(a,3)
        with self.assertRaises(ValueError):
            spatial_tv(a,np.ones((4,4,4))/64.)
        with self.assertRaises(ValueError):
            spatial_js(a,-a)


class RealMultiResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src=os.environ.get("NUKE_REAL_CORE_EVIDENCE_PATH")
        if not src or not Path(src).exists():
            raise RuntimeError("Missing ORIGINAL executed OpenMC HDF5 artifact. "
                               "No mock or fabricated statepoint fallback.")
        cls.source=Path(src)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_all_twenty_original_openmc_checkpoint_records(self):
        r=multiresolution_source_report(self.source)
        self.assertEqual(r["real_statepoints"],20)
        self.assertEqual(r["four_independent_rng_seeds"],
                         [13007,23011,33013,43019])
        self.assertEqual(r["cartesian_resolution_per_axis"],[2,4,8,16])
        self.assertTrue(r["coarsening_probability_conserved_and_tv_js_contraction"])
        self.assertFalse(r["fission_source_stationarity_physically_qualified"])
        self.assertFalse(r["historical_hot_nerva_qualified"])

    def test_legacy_four_cube_real_drift_is_identically_reproduced(self):
        r=multiresolution_source_report(self.source)
        np.testing.assert_allclose(
            r["cartesian_between_seed_mean_total_variation_by_batch"]["4"],
            [.6957222222222222,.6265555555555556,.6663888888888888,
             .6817222222222222,.6857222222222221],
            rtol=0,atol=2e-12)

    def test_real_multiscale_spatial_source_drift_and_sparsity(self):
        r=multiresolution_source_report(self.source)
        last=[r["cartesian_between_seed_mean_total_variation_by_batch"][str(n)][-1]
              for n in (2,4,8,16)]
        np.testing.assert_allclose(last,
                [.3637777778,.6857222222,.9319444444,.9882222222],
                atol=2e-5,rtol=0)
        emptiness=r["cartesian_mean_empty_bin_fraction"]
        self.assertGreater(emptiness["16"],.97)
        self.assertLess(emptiness["2"],.02)
        self.assertGreater(last[0],.3)
        self.assertTrue(all(x<=y+1e-12 for x,y in zip(last,last[1:])))

    def test_real_axial_and_radial_source_projection(self):
        r=multiresolution_source_report(self.source)
        proj=r["cylindrical_radial_and_axial_projection_between_seed_mean_tv_by_batch"]
        np.testing.assert_allclose(proj["radial_8_bins"][-1],
                                   .2730555556,atol=2e-5,rtol=0)
        np.testing.assert_allclose(proj["axial_8_bins"][-1],
                                   .3096666667,atol=2e-5,rtol=0)
        self.assertEqual(len(proj["radial_8_bins"]),5)
        self.assertEqual(len(proj["axial_8_bins"]),5)

    def test_missing_native_archive_is_not_promoted_to_physical_evidence(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError,"missing REAL OpenMC evidence"):
                multiresolution_source_report(path)


if __name__=="__main__":
    unittest.main(verbosity=2)
