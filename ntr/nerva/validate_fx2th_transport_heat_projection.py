"""Independent synthetic checks for per-source heating to thermal-cell power.

Tests do not import, simulate or authenticate OpenMC statepoints. All
meshes and heat rates are synthetic algebraic verification fixtures.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from .fx2th_transport_heat_projection import (
    EV_TO_J, HeatingMeshScore, HeatingNormalization,
    material_fingerprint, make_checked_deposition_provider,
    project_mesh_heating,
)
from .fx2th_thermal_energy import (
    ThermalCells, ThermalState, advance_thermal_cells,
)
from .fx2th_thermal_feedback import advance_coupled_boundary
from .validate_fx2th_family_spacetime import model, initial


def score(*, kind="synthetic", fingerprint=None):
    """2 x 1 x 2 mesh: integrated bin totals are 80/120 eV per history."""
    return HeatingMeshScore(
        source_id="synthetic-transport-mesh-test-v1",
        source_kind=kind,
        material_fingerprint_sha256=fingerprint or "1"*64,
        statepoint_sha256="2"*64,
        nuclear_data_sha256="3"*64,
        tally_score="heating",
        score_units="eV/source-history/mesh-bin",
        lower_left_cm=np.array([0.,0.,0.]),
        upper_right_cm=np.array([2.,1.,4.]),
        total_ev_per_history_bin=np.array([[[20.,40.]],[[60.,80.]]]),
        solid_ev_per_history_bin=np.array([[[5.,10.]],[[15.,20.]]]),
        hydrogen_ev_per_history_bin=np.array([[[3.,6.]],[[9.,12.]]]),
    )


class ProjectionTests(unittest.TestCase):
    def test_prescribed_power_irregular_axial_bins(self):
        r=project_mesh_heating(score(),HeatingNormalization("externally-prescribed-power",200.),
                               [0.,1.,3.,4.])
        np.testing.assert_allclose(r.deposition.solid_power_w,[10.,25.,15.],rtol=0,atol=1e-12)
        np.testing.assert_allclose(r.deposition.hydrogen_power_w,[6.,15.,9.],rtol=0,atol=1e-12)
        np.testing.assert_allclose(r.deposition.other_power_w,[24.,60.,36.],rtol=0,atol=1e-12)
        self.assertAlmostEqual(r.mesh_total_power_w,200.)
        self.assertAlmostEqual(r.solid_total_power_w,50.)
        self.assertAlmostEqual(r.hydrogen_total_power_w,30.)
        self.assertAlmostEqual(r.unmodeled_total_power_w,120.)

    def test_fixed_source_rate_is_exact_ev_to_j_not_assumed_watts(self):
        rate=8.9e11
        r=project_mesh_heating(score(),HeatingNormalization("fixed-source-rate",rate),[0.,2.,4.])
        expected=200.*rate*EV_TO_J
        self.assertAlmostEqual(r.mesh_total_power_w,expected,delta=1e-13*expected)
        self.assertAlmostEqual(float(r.deposition.solid_power_w.sum()),expected*.25,
                               delta=1e-13*expected)
        self.assertEqual(r.source_mode,"fixed-source-rate")

    def test_bin_count_independent_projection_and_global_conservation(self):
        s=score()
        for edges in (
            [0.,4.],[0.,.25,.5,1.25,2.5,4.],
            np.linspace(0,4,17),
            [0.,1.99,2.,2.01,4.],
        ):
            with self.subTest(edges=len(edges)):
                r=project_mesh_heating(s,HeatingNormalization("externally-prescribed-power",73.4),edges)
                self.assertAlmostEqual(r.mesh_total_power_w,73.4)
                self.assertLess(abs(r.global_power_residual_w),1e-12)
                np.testing.assert_allclose(r.axial_overlap_weights.sum(axis=1),1.,
                                           rtol=0,atol=1e-13)
                self.assertEqual(len(r.deposition.solid_power_w),len(edges)-1)

    def test_material_power_partition_remains_disjoint(self):
        r=project_mesh_heating(score(),HeatingNormalization("externally-prescribed-power",200.),
                               [0.,1.,3.,4.])
        np.testing.assert_allclose(r.deposition.solid_power_w+
                                   r.deposition.hydrogen_power_w+
                                   r.deposition.other_power_w,[40.,100.,60.])
        self.assertTrue(r.deposition.synthetic)

    def test_thermal_first_law_with_projected_power(self):
        r=project_mesh_heating(score(),HeatingNormalization("externally-prescribed-power",200.),
                               [0.,1.,3.,4.])
        p=ThermalCells(
            solid_heat_capacity_j_k=np.full(3,100.),
            hydrogen_heat_capacity_j_k=np.full(3,20.),
            wall_conductance_w_k=np.full(3,2.),
            hydrogen_mass_flow_kg_s=np.full(3,.01),
            hydrogen_cp_j_kg_k=np.full(3,1000.),
            inlet_temperature_k=np.full(3,300.),
        )
        t=advance_thermal_cells(
            p,ThermalState(0.,np.full(3,350.),np.full(3,300.)),
            r.deposition,0.25)
        self.assertLess(t.relative_energy_residual,1e-11)
        self.assertAlmostEqual(t.declared_deposited_energy_j,50.)
        self.assertAlmostEqual(t.unmodeled_energy_j,30.)

    def test_reject_mesh_local_partition_overcount_even_if_global_small(self):
        s=score()
        bad=np.array(s.solid_ev_per_history_bin,copy=True)
        bad[0,0,0]=s.total_ev_per_history_bin[0,0,0]
        with self.assertRaisesRegex(ValueError,"overcounts"):
            project_mesh_heating(replace(s,solid_ev_per_history_bin=bad),
               HeatingNormalization("externally-prescribed-power",200.),[0.,4.])

    def test_reject_negative_and_nonfinite_scores(self):
        s=score()
        for val in (-1.,float("nan"),float("inf")):
            bad=s.hydrogen_ev_per_history_bin.copy()
            bad[1,0,0]=val
            with self.subTest(val=val),self.assertRaises(ValueError):
                project_mesh_heating(replace(s,hydrogen_ev_per_history_bin=bad),
                      HeatingNormalization("externally-prescribed-power",200.),[0.,4.])

    def test_reject_score_shape_and_missing_thermal_coverage(self):
        s=score()
        with self.assertRaisesRegex(ValueError,"match mesh"):
            project_mesh_heating(replace(s,solid_ev_per_history_bin=np.zeros((1,1,2))),
                         HeatingNormalization("externally-prescribed-power",200.),[0.,4.])
        for edges in ([.5,4.],[0.,3.],[0.,2.,1.,4.],[0.,1.,1.,4.]):
            with self.subTest(edges=edges),self.assertRaises(ValueError):
                project_mesh_heating(s,HeatingNormalization("externally-prescribed-power",200.),edges)

    def test_reject_power_density_w_per_cc_disguised_as_score(self):
        s=score()
        with self.assertRaisesRegex(ValueError,"integrated eV"):
            project_mesh_heating(replace(s,score_units="W/cm3"),
                  HeatingNormalization("externally-prescribed-power",200.),[0.,4.])

    def test_no_hidden_power_normalization(self):
        s=score()
        for mode,magnitude in (("infer-from-keff",200.),("fixed-source-rate",0.),
                               ("externally-prescribed-power",-1.),
                               ("externally-prescribed-power",float("nan"))):
            with self.subTest(mode=mode,magnitude=magnitude),self.assertRaises(ValueError):
                project_mesh_heating(s,HeatingNormalization(mode,magnitude),[0.,4.])

    def test_hash_provenance_inputs_are_required_not_verified(self):
        s=score()
        with self.assertRaisesRegex(ValueError,"SHA-256"):
            project_mesh_heating(replace(s,nuclear_data_sha256="mock"),
                  HeatingNormalization("externally-prescribed-power",200.),[0.,4.])
        with self.assertRaisesRegex(ValueError,"source_kind"):
            project_mesh_heating(replace(s,source_kind="certified-nerva"),
                  HeatingNormalization("externally-prescribed-power",200.),[0.,4.])
        with self.assertRaisesRegex(ValueError,"heating tally"):
            project_mesh_heating(replace(s,tally_score="fission-q-recoverable"),
                  HeatingNormalization("externally-prescribed-power",200.),[0.,4.])

    def test_external_kind_is_declared_not_numerically_certified(self):
        s=score(kind="extracted-openmc-arrays")
        r=project_mesh_heating(s,HeatingNormalization("externally-prescribed-power",200.),[0.,4.])
        self.assertFalse(r.deposition.synthetic)
        self.assertEqual(r.source_kind,"extracted-openmc-arrays")

    def test_no_energy_lost_when_thermal_bins_outnumber_transport(self):
        r=project_mesh_heating(score(),HeatingNormalization("externally-prescribed-power",200.),
                 np.linspace(0,4,101))
        self.assertAlmostEqual(sum(r.deposition.solid_power_w),50.,places=11)
        self.assertAlmostEqual(sum(r.deposition.hydrogen_power_w),30.,places=11)
        self.assertAlmostEqual(sum(r.deposition.other_power_w),120.,places=11)


class MaterialSnapshotTests(unittest.TestCase):
    def test_material_fingerprint_deterministic_and_sensitive(self):
        m=model(cells=1,groups=1)
        a=material_fingerprint(m)
        self.assertEqual(len(a),64)
        self.assertEqual(material_fingerprint(m),a)
        from scipy.sparse import csc_matrix
        changed=replace(m,integrated_loss=csc_matrix([[.9]]))
        self.assertNotEqual(material_fingerprint(changed),a)

    def test_checked_provider_rejects_stale_materials(self):
        m=model(cells=1,groups=1)
        s=score(fingerprint="4"*64)
        provider=make_checked_deposition_provider(
            lambda state,material:s,
            HeatingNormalization("externally-prescribed-power",200.),
            [0.,4.]
        )
        with self.assertRaisesRegex(ValueError,"stale"):
            provider(initial(m),m)

    def test_checked_provider_rejects_incompatible_thermal_cell_count(self):
        m=model() # 2 spatial cells; one thermal slab is insufficient
        s=score(fingerprint=material_fingerprint(m))
        callback=make_checked_deposition_provider(
            lambda state,material:s,
            HeatingNormalization("externally-prescribed-power",200.),
            [0.,4.]
        )
        with self.assertRaisesRegex(ValueError,"cell count"):
            callback(initial(m),m)

    def test_checked_provider_integrates_with_stage8_feedback(self):
        m=model(cells=1,groups=1)
        snapshots=[]
        def provider(state,data):
            snapshots.append((state.time_s,material_fingerprint(data)))
            return score(fingerprint=material_fingerprint(data))
        source=make_checked_deposition_provider(
            provider,HeatingNormalization("externally-prescribed-power",200.),[0.,4.])
        properties=ThermalCells([100.],[40.],[3.],[.04],[800.],[300.])
        t0=ThermalState(0.,np.array([350.]),np.array([320.]))
        result=advance_coupled_boundary(
            initial(m),t0,properties,.1,
            materials_from_temperature=lambda temperature:m,
            deposition_from_kinetics=source,
        )
        self.assertAlmostEqual(result.thermal.declared_deposited_energy_j,20.)
        self.assertAlmostEqual(result.thermal.unmodeled_energy_j,12.)
        self.assertLess(result.thermal.relative_energy_residual,1e-10)
        self.assertLess(result.kinetic.global_balance_residual,1e-11)
        self.assertEqual(result.kinetic.state.time_s,.1)
        self.assertGreaterEqual(len(snapshots),1)
        self.assertTrue(result.synthetic)


if __name__=="__main__":
    unittest.main(verbosity=2)
