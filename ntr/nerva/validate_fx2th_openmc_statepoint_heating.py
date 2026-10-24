"""Independent numerical tests for OpenMC-style StatePoint heating extraction.

Uses a test double of the documented OpenMC high-level object layout.
These tests do NOT execute OpenMC or prove real statepoint fidelity.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import unittest

import numpy as np

from .fx2th_openmc_statepoint_heating import (
    extract_statepoint_objects, compare_independent_statepoints,
)
from .fx2th_transport_heat_projection import (
    HeatingNormalization, project_mesh_heating,
)

class RegularMesh:
    def __init__(self, dims=(2,1,2), upper=(2.,1.,4.),mesh_id=1):
        self.dimension=dims
        self.lower_left=(0.,0.,0.)
        self.upper_right=upper
        self.id=mesh_id

class MeshFilter:
    def __init__(self, mesh=None):
        self.mesh=mesh or RegularMesh()
        self.num_bins=int(np.prod(self.mesh.dimension))
        self.translation=None
        self.rotation=None

class MaterialFilter:
    def __init__(self, n=1):
        self.num_bins=n
        self.translation=None
        self.rotation=None

class FakeTally:
    def __init__(self, mean,*,sem=None,filters=None,n=40,
                 score='heating',nuclide='total'):
        self.data=np.asarray(mean,dtype=float)
        self.sem=np.asarray(sem if sem is not None else .01*self.data,dtype=float)
        self.num_realizations=n
        self.filters=filters or [MeshFilter()]
        self.scores=[score]
        self.nuclides=[nuclide]
    def get_reshaped_data(self, value='mean', expand_dims=False):
        assert expand_dims
        src=self.data if value=='mean' else self.sem
        extra=(1,)*max(0,len(self.filters)-1)
        return src.reshape(src.shape+extra+(1,1))

class FakeStatePoint:
    def __init__(self,seed=100):
        total=np.array([[[20.,40.]],[[60.,80.]]])
        solid=total*.25
        hydrogen=total*.15
        self.tallies={101:FakeTally(total),102:FakeTally(solid),
                      103:FakeTally(hydrogen)}
        self.run_mode='fixed source'
        self.seed=seed
        self.photon_transport=True

def evidence(sp=None, *, n=10,source_kind='synthetic',**kwargs):
    if sp is None:sp=FakeStatePoint()
    return extract_statepoint_objects(
        sp,tally_ids=(101,102,103),mesh_filter_type=MeshFilter,
        source_id='synthetic-statepoint-test',
        material_fingerprint_sha256='1'*64,
        statepoint_sha256=hashlib.sha256(str(sp.seed).encode()).hexdigest(),
        nuclear_data_sha256='3'*64,min_realizations=n,
        source_kind=source_kind,**kwargs)


class StatepointReaderTests(unittest.TestCase):
    def test_mesh_and_score_values_to_thermal_energies(self):
        e=evidence()
        self.assertEqual(e.score.total_ev_per_history_bin.shape,(2,1,2))
        self.assertEqual(e.n_realizations,40)
        self.assertFalse(e.is_actual_statepoint_reader)
        p=e.project(HeatingNormalization('externally-prescribed-power',200.),
                    [0.,1.,3.,4.])
        np.testing.assert_allclose(p.deposition.solid_power_w,[10.,25.,15.])
        np.testing.assert_allclose(p.deposition.hydrogen_power_w,[6.,15.,9.])
        np.testing.assert_allclose(p.deposition.other_power_w,[24.,60.,36.])

    def test_high_error_in_mean_is_rejected(self):
        sp=FakeStatePoint()
        sp.tallies[101].sem[0,0,0]=20.
        with self.assertRaisesRegex(ValueError,'uncertainty'):
            evidence(sp)

    def test_ambiguous_zero_mean_nonzero_error_rejected(self):
        sp=FakeStatePoint()
        sp.tallies[102].data[0,0,0]=0.
        with self.assertRaisesRegex(ValueError,'zero-mean'):
            evidence(sp)

    def test_zero_error_zero_mean_allowed(self):
        sp=FakeStatePoint()
        sp.tallies[102].data[0,0,0]=0.
        sp.tallies[102].sem[0,0,0]=0.
        e=evidence(sp)
        self.assertEqual(e.score.solid_ev_per_history_bin[0,0,0],0)

    def test_not_enough_realizations_rejected(self):
        sp=FakeStatePoint()
        sp.tallies[103].num_realizations=2
        with self.assertRaisesRegex(ValueError,'insufficient'):
            evidence(sp,n=10)

    def test_cross_tally_batch_disagreement_rejected(self):
        sp=FakeStatePoint()
        sp.tallies[103].num_realizations=60
        with self.assertRaisesRegex(ValueError,'mesh/realizations'):
            evidence(sp)

    def test_wrong_mesh_bins_and_bounds_rejected(self):
        sp=FakeStatePoint()
        sp.tallies[103].filters=[MeshFilter(RegularMesh(upper=(2.,1.,3.)))]
        with self.assertRaisesRegex(ValueError,'different mesh'):
            evidence(sp)

    def test_malformed_filter_layout_rejected(self):
        sp=FakeStatePoint()
        sp.tallies[101].filters=[MaterialFilter(),MeshFilter()]
        with self.assertRaisesRegex(ValueError,'first tally filter'):
            evidence(sp)

    def test_single_material_partition_filter_supported(self):
        sp=FakeStatePoint()
        sp.tallies[102].filters=[MeshFilter(),MaterialFilter(1)]
        self.assertAlmostEqual(evidence(sp).score.solid_ev_per_history_bin.sum(),50.)

    def test_multi_material_filter_cannot_be_silently_summed(self):
        sp=FakeStatePoint()
        sp.tallies[102].filters=[MeshFilter(),MaterialFilter(2)]
        with self.assertRaisesRegex(ValueError,'singleton'):
            evidence(sp)

    def test_invalid_score_and_nuclide_refused(self):
        for field,value,message in [('scores',['flux'],'heating score'),
                                    ('nuclides',['U235'],'total nuclide')]:
            sp=FakeStatePoint()
            setattr(sp.tallies[102],field,value)
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,message):
                evidence(sp)

    def test_eigenvalue_needs_prescribed_power_not_fixed_source_rate(self):
        sp=FakeStatePoint()
        sp.run_mode='eigenvalue'
        e=evidence(sp)
        with self.assertRaisesRegex(ValueError,'eigenvalue'):
            e.project(HeatingNormalization('fixed-source-rate',1e6),[0.,4.])
        self.assertAlmostEqual(
            e.project(HeatingNormalization('externally-prescribed-power',200.),[0.,4.])
              .mesh_total_power_w,200.)

    def test_fake_evidence_not_promoted_to_real_by_label(self):
        with self.assertRaisesRegex(ValueError,'actual StatePoint loader'):
            evidence(source_kind='synthetic',actual_statepoint_reader=True)

    def test_negative_heating_rejected_not_clipped(self):
        sp=FakeStatePoint()
        sp.tallies[103].data[0,0,0]=-1.
        with self.assertRaisesRegex(ValueError,'negative'):
            evidence(sp)

    def test_local_double_counting_rejected(self):
        sp=FakeStatePoint()
        sp.tallies[103].data[0,0,0]=20.
        with self.assertRaisesRegex(ValueError,'overcounts'):
            evidence(sp)

    def test_source_metadata_must_be_consistent(self):
        sp=FakeStatePoint()
        sp.run_mode='volume'
        with self.assertRaisesRegex(ValueError,'run mode'):
            evidence(sp)
        sp.run_mode='fixed source'
        with self.assertRaisesRegex(ValueError,'tally IDs'):
            extract_statepoint_objects(
                sp,tally_ids=(101,101,103),mesh_filter_type=MeshFilter,
                source_id='synthetic',material_fingerprint_sha256='1'*64,
                statepoint_sha256='2'*64,nuclear_data_sha256='3'*64)

    def test_optional_bin_threshold_does_not_rewrite_means(self):
        sp=FakeStatePoint()
        sp.tallies[102].data[0,0,0]=.0001
        sp.tallies[102].sem[0,0,0]=.0001
        e=evidence(sp,relative_bin_cutoff=.01)
        self.assertAlmostEqual(e.score.solid_ev_per_history_bin[0,0,0],.0001)
        self.assertGreater(e.near_zero_bins,0)


class IndependentSeedTests(unittest.TestCase):
    def test_distinct_seed_sem_computed_from_whole_projections(self):
        ev=[]
        for i in range(4):
            sp=FakeStatePoint(seed=300+i)
            sp.tallies[102].data[0,0,0]+=i
            ev.append(evidence(sp))
        r=compare_independent_statepoints(tuple(ev),
                 HeatingNormalization('externally-prescribed-power',200.),[0.,2.,4.])
        self.assertEqual(r.independent_runs,4)
        self.assertEqual(r.seeds,(300,301,302,303))
        self.assertAlmostEqual(r.total_mean_w,200.)
        self.assertGreater(r.sem_solid_power_w[0],0)
        self.assertTrue(np.isfinite(r.sem_hydrogen_power_w).all())
        self.assertEqual(r.source_kind,'synthetic')

    def test_duplicate_seeds_or_hashes_rejected(self):
        ev=tuple(evidence(FakeStatePoint(seed=500)) for _ in range(4))
        with self.assertRaisesRegex(ValueError,'distinct seeds'):
            compare_independent_statepoints(ev,HeatingNormalization(
                'externally-prescribed-power',200.),[0.,4.])

    def test_different_material_rejected(self):
        ev=[evidence(FakeStatePoint(seed=600+i)) for i in range(4)]
        changed=replace(ev[-1],score=replace(ev[-1].score,
                              material_fingerprint_sha256='4'*64))
        ev[-1]=changed
        with self.assertRaisesRegex(ValueError,'share mesh'):
            compare_independent_statepoints(tuple(ev),
                HeatingNormalization('externally-prescribed-power',200.),[0.,4.])

    def test_fewer_than_four_seeds_not_certified(self):
        ev=tuple(evidence(FakeStatePoint(seed=700+i)) for i in range(3))
        with self.assertRaisesRegex(ValueError,'insufficient'):
            compare_independent_statepoints(ev,
                HeatingNormalization('externally-prescribed-power',200.),[0.,4.])

    def test_matched_seeds_same_scores_yield_zero_between_seed_sem(self):
        ev=tuple(evidence(FakeStatePoint(seed=800+i)) for i in range(4))
        r=compare_independent_statepoints(ev,HeatingNormalization(
             'externally-prescribed-power',200.),[0.,1.,3.,4.])
        np.testing.assert_allclose(r.sem_solid_power_w,0)
        np.testing.assert_allclose(r.sem_hydrogen_power_w,0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
