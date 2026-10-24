"""Synthetic second-stage verification: NASA7 H2 enthalpy and r-z neutron feedback.

Compared to the independently tested Stage19 constant cp model, Stage7
spatial precursor solver and Stage20 standalone enthalpy FV solver.
No NERVA material, fuel, evaluated neutron or heating input is assumed.
"""
from __future__ import annotations

from dataclasses import replace
import unittest
import numpy as np

from .fx2th_enthalpy_rz_feedback import advance_enthalpy_resolved_rz_feedback
from .fx2th_enthalpy_conjugate_core import (
    EnthalpyCoreControls, EnthalpyCoreStep,advance_enthalpy_conjugate_core,
)
from .fx2th_resolved_rz_feedback import advance_resolved_rz_feedback
from .fx2th_family_spacetime import advance_spatial_families
from .fx2th_rz_temperature_tables import sample_rz_temperature_tables
from .fx2th_conjugate_fuel_transport import (
    advance_conjugate_core,effective_solid_temperature_for_xs,
)
from .validate_fx2th_resolved_rz_feedback import setup,deposition
from .validate_fx2th_conjugate_fuel_transport import heat


class EnthalpyRZFeedbackTests(unittest.TestCase):
    def test_nasa7_full_neutron_family_fuel_and_coolant_coupling(self):
        table,g,y0,t0=setup()
        result=advance_enthalpy_resolved_rz_feedback(
            table,g,y0,t0,.1,deposition(g))
        self.assertIsInstance(result.thermal,EnthalpyCoreStep)
        self.assertEqual(result.thermal.enthalpy_model,"nasa7-h2")
        self.assertEqual(result.neutron.state.flux.shape,(6,2))
        self.assertEqual(result.neutron.state.precursors.shape,(6,2))
        self.assertEqual(result.thermal.state.fuel_temperature_k.shape,(2,3,3))
        self.assertLess(result.maximum_temperature_residual_k,1e-8)
        self.assertLess(result.thermal.thermal_relative_residual,1e-10)
        self.assertLess(result.neutron.global_balance_residual,1e-10)
        self.assertTrue(result.synthetic)
        self.assertAlmostEqual(result.neutron.state.time_s,.1)
        self.assertAlmostEqual(result.thermal.state.time_s,.1)

    def test_constant_cp_enthalpy_feedback_matches_original_stage19(self):
        table,g,y0,t0=setup()
        baseline=advance_resolved_rz_feedback(
            table,g,y0,t0,.13,deposition(g))
        wrapped=advance_enthalpy_resolved_rz_feedback(
            table,g,y0,t0,.13,deposition(g),
            enthalpy_controls=EnthalpyCoreControls(
                enthalpy_model="constant"))
        np.testing.assert_allclose(
            wrapped.neutron.state.flux,baseline.neutron.state.flux,
            atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(
            wrapped.neutron.state.precursors,
            baseline.neutron.state.precursors,atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(
            wrapped.thermal.state.fuel_temperature_k,
            baseline.thermal.state.fuel_temperature_k,
            atol=1e-8,rtol=0)
        np.testing.assert_allclose(
            wrapped.thermal.state.hydrogen_temperature_k,
            baseline.thermal.state.hydrogen_temperature_k,
            atol=1e-8,rtol=0)

    def test_final_nuclear_state_matches_independent_stage7(self):
        table,g,y0,t0=setup()
        dt=.14
        result=advance_enthalpy_resolved_rz_feedback(
            table,g,y0,t0,dt,deposition(g))
        finalT=effective_solid_temperature_for_xs(g,result.thermal.state)
        data=sample_rz_temperature_tables(table,finalT).material
        independent=advance_spatial_families(data,y0,dt)
        np.testing.assert_allclose(
            result.neutron.state.flux,independent.state.flux,
            atol=2e-9,rtol=2e-9)
        np.testing.assert_allclose(
            result.neutron.state.precursors,independent.state.precursors,
            atol=2e-9,rtol=2e-9)

    def test_final_enthalpy_equals_standalone_nonlinear_thermal(self):
        table,g,y0,t0=setup()
        dt=.12
        cb=deposition(g)
        result=advance_enthalpy_resolved_rz_feedback(
            table,g,y0,t0,dt,cb)
        finalT=effective_solid_temperature_for_xs(g,result.thermal.state)
        data=sample_rz_temperature_tables(table,finalT).material
        power=cb(result.neutron.state,data)
        independent=advance_enthalpy_conjugate_core(
            g,t0,power,dt)
        np.testing.assert_allclose(
            result.thermal.state.fuel_temperature_k,
            independent.state.fuel_temperature_k,
            atol=3e-9,rtol=0)
        np.testing.assert_allclose(
            result.thermal.state.hydrogen_temperature_k,
            independent.state.hydrogen_temperature_k,
            atol=3e-9,rtol=0)
        self.assertLess(result.thermal.thermal_relative_residual,1e-10)

    def test_nasa7_nonconstant_cp_changes_thermal_feedback_solution(self):
        table,g,y0,t0=setup()
        dt=.19
        nasa=advance_enthalpy_resolved_rz_feedback(
            table,g,y0,t0,dt,deposition(g))
        constant=advance_enthalpy_resolved_rz_feedback(
            table,g,y0,t0,dt,deposition(g),
            enthalpy_controls=EnthalpyCoreControls(
                enthalpy_model="constant"))
        difference=float(np.max(np.abs(
            nasa.thermal.state.hydrogen_temperature_k-
            constant.thermal.state.hydrogen_temperature_k)))
        self.assertGreater(difference,.2)
        self.assertNotEqual(nasa.material_source_id,constant.material_source_id)

    def test_zero_deposition_reduces_to_independent_three_solvers(self):
        table,g,y0,t0=setup()
        dt=.08
        def zero(state,data): return heat(g)
        result=advance_enthalpy_resolved_rz_feedback(
            table,g,y0,t0,dt,zero)
        expected_thermal=advance_enthalpy_conjugate_core(
            g,t0,heat(g),dt)
        material=sample_rz_temperature_tables(
            table,effective_solid_temperature_for_xs(
                g,expected_thermal.state)).material
        expected_neutron=advance_spatial_families(material,y0,dt)
        np.testing.assert_allclose(
            result.thermal.state.hydrogen_temperature_k,
            expected_thermal.state.hydrogen_temperature_k,
            atol=2e-9,rtol=0)
        np.testing.assert_allclose(
            result.neutron.state.flux,expected_neutron.state.flux,
            atol=1e-9,rtol=1e-9)

    def test_fail_closed_outside_molecular_hydrogen_domain(self):
        table,g,y0,t0=setup()
        cold=replace(t0,hydrogen_temperature_k=np.full(
            g.shape[:2],150.))
        with self.assertRaises(ValueError):
            advance_enthalpy_resolved_rz_feedback(
                table,g,y0,cold,.1,deposition(g))

    def test_nonlinear_picard_iteration_budget_enforced(self):
        table,g,y0,t0=setup()
        with self.assertRaisesRegex(RuntimeError,"did not converge"):
            advance_enthalpy_resolved_rz_feedback(
                table,g,y0,t0,.15,deposition(g),
                temperature_tolerance_k=1e-13,max_iterations=1)

    def test_external_source_and_heat_provenance(self):
        table,g,y0,t0=setup()
        q=np.zeros((6,2));q[0,0]=.3
        b=advance_enthalpy_resolved_rz_feedback(
            table,g,y0,t0,.07,deposition(g),
            external_source_per_m3=q)
        a=advance_enthalpy_resolved_rz_feedback(
            table,g,y0,t0,.07,deposition(g))
        self.assertGreater(np.linalg.norm(
            b.neutron.state.flux-a.neutron.state.flux),1e-5)
        self.assertEqual(b.deposition_source_id,
                         "synthetic-mesh-aware-partition")
        self.assertLess(b.thermal.thermal_relative_residual,1e-10)

    def test_original_states_unchanged_and_reproducible(self):
        table,g,y0,t0=setup()
        phi=y0.flux.copy();T=t0.fuel_temperature_k.copy()
        a=advance_enthalpy_resolved_rz_feedback(
            table,g,y0,t0,.08,deposition(g))
        b=advance_enthalpy_resolved_rz_feedback(
            table,g,y0,t0,.08,deposition(g))
        np.testing.assert_array_equal(a.neutron.state.flux,b.neutron.state.flux)
        np.testing.assert_array_equal(
            a.thermal.state.hydrogen_temperature_k,
            b.thermal.state.hydrogen_temperature_k)
        np.testing.assert_array_equal(y0.flux,phi)
        np.testing.assert_array_equal(t0.fuel_temperature_k,T)


if __name__=="__main__":
    unittest.main(verbosity=2)
