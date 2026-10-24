"""Synthetic resolved core neutron/thermal feedback mathematical references.

All heat and nuclear data are SYNTHETIC. Test oracles use the established
Stage-7 neutron integrator and Stage-19 independent conjugate FV operator.
These verify discrete equations, not a hot NERVA engine or OpenMC heating.
"""
from __future__ import annotations

from dataclasses import replace
import unittest
import numpy as np

from .validate_fx2th_conjugate_fuel_transport import fixture, heat
from .validate_fx2th_rz_temperature_tables import synthetic_tables
from .fx2th_rz_temperature_tables import sample_rz_temperature_tables
from .fx2th_conjugate_fuel_transport import (
    ConjugateCoreState, ConjugateCoreDeposition,
    advance_conjugate_core, effective_solid_temperature_for_xs,
)
from .fx2th_family_spacetime import (
    SpatialFamilyState,stationary_precursors,advance_spatial_families
)
from .fx2th_resolved_rz_feedback import advance_resolved_rz_feedback


def setup():
    table=synthetic_tables()
    g, thermal, _ = fixture(npath=2,nz=3,nr=3,flow=.002,wall=20.)
    data=sample_rz_temperature_tables(
        table,effective_solid_temperature_for_xs(g,thermal)).material
    phi=np.ones((6,2));phi[0,0]=1.4;phi[-1,1]=.6
    neutron=SpatialFamilyState(0.,phi,stationary_precursors(data,phi))
    return table,g,neutron,thermal


def deposition(geo):
    def f(neutrons,data):
        source=np.asarray(neutrons.flux).sum(axis=1).reshape((2,3))
        volumes=geo.solid_volumes_m3
        frac=volumes/volumes.sum(axis=2,keepdims=True)
        qs=(80.+15.*source)[:,:,None]*frac
        qh=15.+4.*source
        other=np.full((2,3),6.)
        return ConjugateCoreDeposition(
            "synthetic-mesh-aware-partition",qs,qh,other,
            float(qs.sum()+qh.sum()+other.sum()),True)
    return f


class ResolvedFeedbackTests(unittest.TestCase):
    def test_resolved_feedback_advances_spatial_neutrons_and_fuel(self):
        table,g,neutrons,thermal=setup()
        out=advance_resolved_rz_feedback(
            table,g,neutrons,thermal,.1,deposition(g))
        self.assertEqual(out.neutron.state.flux.shape,(6,2))
        self.assertEqual(out.thermal.state.fuel_temperature_k.shape,(2,3,3))
        self.assertEqual(out.thermal.state.hydrogen_temperature_k.shape,(2,3))
        self.assertLess(out.maximum_temperature_residual_k,1e-8)
        self.assertLess(out.thermal.thermal_relative_residual,1e-10)
        self.assertLess(out.neutron.global_balance_residual,1e-10)
        self.assertTrue(out.synthetic)
        self.assertAlmostEqual(out.neutron.state.time_s,.1)
        self.assertAlmostEqual(out.thermal.state.time_s,.1)

    def test_converged_neutron_step_matches_independent_BE_from_old_state(self):
        table,g,neutrons,thermal=setup()
        step=advance_resolved_rz_feedback(
            table,g,neutrons,thermal,.15,deposition(g))
        final=effective_solid_temperature_for_xs(g,step.thermal.state)
        data=sample_rz_temperature_tables(table,final).material
        direct=advance_spatial_families(data,neutrons,.15)
        np.testing.assert_allclose(step.neutron.state.flux,
                                  direct.state.flux,rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(step.neutron.state.precursors,
                                  direct.state.precursors,rtol=1e-9,atol=1e-9)

    def test_final_fuel_and_coolant_equal_independently_resolved_thermal_step(self):
        table,g,neutrons,thermal=setup()
        power=deposition(g)
        out=advance_resolved_rz_feedback(table,g,neutrons,thermal,.13,power)
        p=power(out.neutron.state,
                sample_rz_temperature_tables(table,
                    effective_solid_temperature_for_xs(
                        g,out.thermal.state)).material)
        reference=advance_conjugate_core(g,thermal,p,.13)
        np.testing.assert_allclose(
            out.thermal.state.fuel_temperature_k,
            reference.state.fuel_temperature_k,rtol=0,atol=2e-8)
        np.testing.assert_allclose(
            out.thermal.state.hydrogen_temperature_k,
            reference.state.hydrogen_temperature_k,rtol=0,atol=2e-8)

    def test_zero_heating_reduces_to_independent_one_way_operator(self):
        table,g,neutrons,thermal=setup()
        def zero(state,data): return heat(g)
        result=advance_resolved_rz_feedback(table,g,neutrons,thermal,
                                            .08,zero)
        expected_thermal=advance_conjugate_core(g,thermal,heat(g),.08)
        expected_material=sample_rz_temperature_tables(table,
              effective_solid_temperature_for_xs(g,expected_thermal.state)).material
        expected_neutrons=advance_spatial_families(expected_material,neutrons,.08)
        np.testing.assert_allclose(result.thermal.state.fuel_temperature_k,
            expected_thermal.state.fuel_temperature_k,rtol=0,atol=1e-9)
        np.testing.assert_allclose(result.thermal.state.hydrogen_temperature_k,
            expected_thermal.state.hydrogen_temperature_k,rtol=0,atol=1e-9)
        np.testing.assert_allclose(result.neutron.state.flux,
            expected_neutrons.state.flux,rtol=1e-9,atol=1e-9)

    def test_external_neutron_source_changes_new_flux_not_source_normalization(self):
        table,g,neutrons,thermal=setup()
        q=np.zeros((6,2));q[0,0]=.25
        a=advance_resolved_rz_feedback(table,g,neutrons,thermal,.05,
                                        deposition(g))
        b=advance_resolved_rz_feedback(table,g,neutrons,thermal,.05,
                                        deposition(g),
                                        external_source_per_m3=q)
        self.assertGreater(np.linalg.norm(
            a.neutron.state.flux-b.neutron.state.flux),1e-5)
        self.assertLess(b.thermal.thermal_relative_residual,1e-10)

    def test_responds_to_resolved_ring_temperature_feedback(self):
        table,g,neutrons,thermal=setup()
        changed=ConjugateCoreState(0.,thermal.fuel_temperature_k.copy(),
                                  thermal.hydrogen_temperature_k.copy())
        warmer=changed.fuel_temperature_k.copy()
        warmer[:,:,2]+=90.
        changed=replace(changed,fuel_temperature_k=warmer)
        before=sample_rz_temperature_tables(table,
                    effective_solid_temperature_for_xs(g,thermal)).material
        after=sample_rz_temperature_tables(table,
                    effective_solid_temperature_for_xs(g,changed)).material
        self.assertNotEqual(before.source_id,after.source_id)
        self.assertGreater(np.max(np.abs(
            (before.integrated_loss-after.integrated_loss).toarray())),1e-6)

    def test_nonconvergent_feedback_does_not_accept_partial_step(self):
        table,g,neutrons,thermal=setup()
        with self.assertRaisesRegex(RuntimeError,"did not converge"):
            advance_resolved_rz_feedback(
                table,g,neutrons,thermal,.2,deposition(g),
                temperature_tolerance_k=1e-13,max_iterations=1)

    def test_incompatible_mesh_and_time_rejected(self):
        table,g,neutrons,thermal=setup()
        with self.assertRaisesRegex(ValueError,"times"):
            advance_resolved_rz_feedback(
                table,g,neutrons,replace(thermal,time_s=.6),.1,deposition(g))
        other_g,_,_=fixture(npath=1,nz=3,nr=3)
        with self.assertRaisesRegex(ValueError,"do not match"):
            advance_resolved_rz_feedback(
                table,other_g,neutrons,thermal,.1,deposition(other_g))
        with self.assertRaises(ValueError):
            advance_resolved_rz_feedback(
                table,g,neutrons,thermal,.1,deposition(g),
                external_source_per_m3=np.ones((2,3)))

    def test_callback_and_source_partition_fail_closed(self):
        table,g,neutrons,thermal=setup()
        with self.assertRaisesRegex(TypeError,"ConjugateCoreDeposition"):
            advance_resolved_rz_feedback(
                table,g,neutrons,thermal,.1,lambda state,mat:1.0)
        with self.assertRaises(TypeError):
            advance_resolved_rz_feedback(table,g,neutrons,thermal,.1,None)

    def test_repeatability_and_old_state_immutability(self):
        table,g,neutrons,thermal=setup()
        phi=neutrons.flux.copy()
        solid=thermal.fuel_temperature_k.copy()
        a=advance_resolved_rz_feedback(
            table,g,neutrons,thermal,.11,deposition(g))
        b=advance_resolved_rz_feedback(
            table,g,neutrons,thermal,.11,deposition(g))
        np.testing.assert_array_equal(a.neutron.state.flux,b.neutron.state.flux)
        np.testing.assert_array_equal(
            a.thermal.state.fuel_temperature_k,
            b.thermal.state.fuel_temperature_k)
        np.testing.assert_array_equal(neutrons.flux,phi)
        np.testing.assert_array_equal(thermal.fuel_temperature_k,solid)


if __name__=="__main__":
    unittest.main(verbosity=2)
