"""Numerical regression for Stage 18 fully coupled backward-Euler reactor model.

Uses clearly SYNTHETIC multigroup temperature tables and explicit synthetic
heating. Comparisons use the independent Stage-8 Picard implementation and
the Stage-7 spatial neutron integrator, as well as the Stage-8 thermal
first-law conservation reference. NO historical NERVA physics verified.
"""
from __future__ import annotations

from dataclasses import replace
import unittest
import numpy as np

from .fx2th_monolithic_rz_core import (
    advance_monolithic_rz_core, MonolithicControls, _ResidualEvaluator,
    _finite_difference_jacobian,
)
from .fx2th_rz_temperature_tables import (
    sample_rz_temperature_tables,
)
from .validate_fx2th_rz_temperature_tables import (
    synthetic_tables, heterogeneous_temperature,
)
from .fx2th_family_spacetime import (
    SpatialFamilyState, stationary_precursors, advance_spatial_families,
)
from .fx2th_thermal_feedback import advance_coupled_boundary
from .fx2th_thermal_energy import (
    ThermalCells, ThermalState, ThermalDeposition, advance_thermal_cells,
)


def fixture():
    table=synthetic_tables()
    Ts=heterogeneous_temperature().ravel()
    Th=np.full(6,335.)
    data=sample_rz_temperature_tables(table,Ts.reshape(2,3)).material
    phi=np.ones((6,2),dtype=float)
    phi[0,0]=2.4
    phi[-1,1]=.43
    y0=SpatialFamilyState(0.,phi,stationary_precursors(data,phi))
    t0=ThermalState(0.,Ts.copy(),Th)
    cells=ThermalCells(
        solid_heat_capacity_j_k=np.full(6,120.),
        hydrogen_heat_capacity_j_k=np.full(6,40.),
        wall_conductance_w_k=np.full(6,8.),
        hydrogen_mass_flow_kg_s=np.full(6,.02),
        hydrogen_cp_j_kg_k=np.full(6,200.),
        inlet_temperature_k=np.full(6,300.),
    )
    return table,y0,t0,cells


def explicit_synthetic_deposition(state,data):
    # This is NOT an evaluated reactor heating or nuclear Q model.
    strength=np.asarray(state.flux).sum(axis=1)
    solid=300.+80.*strength
    hydrogen=80.+12.*strength
    other=np.full(len(strength),30.)
    return ThermalDeposition(
        "synthetic-external-core-heating-v1",
        solid,hydrogen,other,
        float(np.sum(solid+hydrogen+other)),synthetic=True)


def zero_deposition(state,data):
    return ThermalDeposition(
        "synthetic-zero-power-for-analytic-thermal-check",
        np.zeros(data.cells),np.zeros(data.cells),
        np.zeros(data.cells),0.,synthetic=True)


def cold_material_callback(table):
    def f(thermal):
        return sample_rz_temperature_tables(
            table,np.asarray(thermal.solid_temperature_k).reshape(
                table.spatial_shape)).material
    return f


class MonolithicFeedbackTests(unittest.TestCase):
    def test_simultaneous_physics_and_energy_balance(self):
        tab,y0,t0,c=fixture()
        step=advance_monolithic_rz_core(
            tab,y0,t0,c,.15,explicit_synthetic_deposition)
        self.assertLess(step.scaled_residual,1e-10)
        self.assertLess(step.thermal_ledger.relative_energy_residual,1e-10)
        self.assertLess(abs(step.neutron_population_balance),1e-8)
        self.assertEqual(step.neutron_state.flux.shape,(6,2))
        self.assertEqual(step.neutron_state.precursors.shape,(6,2))
        self.assertTrue(step.synthetic)
        self.assertGreater(step.thermal_ledger.unmodeled_energy_j,0.)
        self.assertAlmostEqual(step.neutron_state.time_s,.15,places=13)
        self.assertAlmostEqual(step.thermal_state.time_s,.15,places=13)

    def test_monolithic_solution_agrees_with_independent_picard(self):
        tab,y0,t0,c=fixture()
        dt=.12
        nl=advance_monolithic_rz_core(
            tab,y0,t0,c,dt,explicit_synthetic_deposition)
        picard=advance_coupled_boundary(
            y0,t0,c,dt,cold_material_callback(tab),
            explicit_synthetic_deposition,
            temperature_tolerance_k=1e-9,relaxation=.75,max_iterations=100)
        np.testing.assert_allclose(
            nl.neutron_state.flux,picard.kinetic.state.flux,
            rtol=8e-8,atol=8e-9)
        np.testing.assert_allclose(
            nl.neutron_state.precursors,picard.kinetic.state.precursors,
            rtol=8e-8,atol=8e-9)
        np.testing.assert_allclose(
            nl.thermal_state.solid_temperature_k,
            picard.thermal.state.solid_temperature_k,
            rtol=0,atol=1e-7)
        np.testing.assert_allclose(
            nl.thermal_state.hydrogen_temperature_k,
            picard.thermal.state.hydrogen_temperature_k,
            rtol=0,atol=1e-7)

    def test_zero_heat_exactly_matches_independent_thermal_and_neutron_solves(self):
        tab,y0,t0,c=fixture()
        dt=.08
        t_direct=advance_thermal_cells(c,t0,zero_deposition(y0,
              sample_rz_temperature_tables(tab,t0.solid_temperature_k.reshape(2,3)).material),dt)
        sample=sample_rz_temperature_tables(tab,
                    t_direct.state.solid_temperature_k.reshape(2,3))
        expected=advance_spatial_families(sample.material,y0,dt)
        coupled=advance_monolithic_rz_core(
            tab,y0,t0,c,dt,zero_deposition)
        np.testing.assert_allclose(coupled.thermal_state.solid_temperature_k,
            t_direct.state.solid_temperature_k,rtol=0,atol=1e-8)
        np.testing.assert_allclose(coupled.thermal_state.hydrogen_temperature_k,
            t_direct.state.hydrogen_temperature_k,rtol=0,atol=1e-8)
        np.testing.assert_allclose(coupled.neutron_state.flux,
            expected.state.flux,atol=1e-9,rtol=1e-9)
        np.testing.assert_allclose(coupled.neutron_state.precursors,
            expected.state.precursors,atol=1e-9,rtol=1e-9)

    def test_temperature_change_alters_actual_nuclear_material_matrix(self):
        tab,y0,t0,c=fixture()
        t_initial=sample_rz_temperature_tables(
            tab,t0.solid_temperature_k.reshape(2,3)).material
        x=advance_monolithic_rz_core(tab,y0,t0,c,.2,
                                     explicit_synthetic_deposition)
        self.assertNotEqual(x.material_source_id,t_initial.source_id)
        self.assertGreater(np.max(np.abs(
            x.material.nu_sigma_f_inv_m-t_initial.nu_sigma_f_inv_m)),1e-9)
        self.assertGreater(np.max(np.abs(
            (x.material.integrated_loss-t_initial.integrated_loss).toarray())),1e-9)

    def test_constant_externally_supplied_source_preserved(self):
        tab,y0,t0,c=fixture()
        q=np.zeros((6,2));q[0,0]=.3;q[5,1]=.12
        step=advance_monolithic_rz_core(
            tab,y0,t0,c,.08,explicit_synthetic_deposition,
            external_source_per_m3=q)
        self.assertLess(step.scaled_residual,1e-10)
        no=advance_monolithic_rz_core(tab,y0,t0,c,.08,
                                     explicit_synthetic_deposition)
        self.assertGreater(np.linalg.norm(
            step.neutron_state.flux-no.neutron_state.flux),1e-4)

    def test_power_partition_and_first_law_are_independent(self):
        tab,y0,t0,c=fixture()
        dt=.11
        answer=advance_monolithic_rz_core(
            tab,y0,t0,c,dt,explicit_synthetic_deposition)
        heat=explicit_synthetic_deposition(answer.neutron_state,answer.material)
        budget=advance_thermal_cells(c,t0,heat,dt)
        self.assertAlmostEqual(answer.thermal_ledger.declared_deposited_energy_j,
                               heat.declared_total_w*dt,places=10)
        self.assertAlmostEqual(answer.thermal_ledger.unmodeled_energy_j,
                               dt*sum(heat.other_power_w),places=10)
        np.testing.assert_allclose(
            answer.thermal_state.solid_temperature_k,
            budget.state.solid_temperature_k,atol=1e-8,rtol=0)

    def test_no_temperature_extrapolation_or_silent_clipping(self):
        tab,y0,t0,c=fixture()
        initial_bad=replace(t0,solid_temperature_k=np.full(6,1250.))
        with self.assertRaisesRegex(ValueError,"extrapolation"):
            advance_monolithic_rz_core(
                tab,y0,initial_bad,c,.1,explicit_synthetic_deposition)
        def absurd_power(state,data):
            q=np.full(data.cells,1e9)
            return ThermalDeposition("synthetic-absurd-W",q,
                np.zeros(data.cells),np.zeros(data.cells),
                float(q.sum()),True)
        with self.assertRaisesRegex(RuntimeError,"line search"):
            advance_monolithic_rz_core(
                tab,y0,t0,c,.3,absurd_power)

    def test_explicit_power_provenance_required(self):
        tab,y0,t0,c=fixture()
        with self.assertRaisesRegex(TypeError,"ThermalDeposition"):
            advance_monolithic_rz_core(
                tab,y0,t0,c,.05,lambda state,data:np.ones(data.cells))
        def invalid_partition(state,data):
            return ThermalDeposition("synthetic-unbalanced-power",
                np.ones(data.cells),np.ones(data.cells),
                np.ones(data.cells),100.)
        with self.assertRaisesRegex(ValueError,"partition"):
            advance_monolithic_rz_core(
                tab,y0,t0,c,.05,invalid_partition)

    def test_invalid_state_mesh_and_timesteps_refused(self):
        tab,y0,t0,c=fixture()
        with self.assertRaisesRegex(ValueError,"cells"):
            advance_monolithic_rz_core(tab,y0,t0,
                replace(c,solid_heat_capacity_j_k=np.ones(5)),.1,
                explicit_synthetic_deposition)
        with self.assertRaisesRegex(ValueError,"times"):
            advance_monolithic_rz_core(tab,y0,replace(t0,time_s=.2),c,.1,
                explicit_synthetic_deposition)
        for dt in (0.,-1.,float("nan")):
            with self.subTest(dt=dt),self.assertRaises(ValueError):
                advance_monolithic_rz_core(tab,y0,t0,c,dt,
                                           explicit_synthetic_deposition)

    def test_dense_newton_system_size_limit_fails_closed(self):
        tab,y0,t0,c=fixture()
        with self.assertRaisesRegex(ValueError,"unknown limit"):
            advance_monolithic_rz_core(
                tab,y0,t0,c,.1,explicit_synthetic_deposition,
                controls=MonolithicControls(max_unknowns=12))

    def test_nonlinear_residual_jacobian_matches_independent_directional_difference(self):
        tab,y0,t0,c=fixture()
        evaluator=_ResidualEvaluator(tab,y0,t0,c,.09,
                   explicit_synthetic_deposition,None)
        x=evaluator.x0.copy()
        base=evaluator.evaluate(x).scaled
        J=_finite_difference_jacobian(evaluator,x,base,MonolithicControls())
        dr=np.linspace(-1.,1.,len(x))
        dr/=np.linalg.norm(dr)
        eps=3e-5
        actual=(evaluator.evaluate(x+eps*dr).scaled-
                evaluator.evaluate(x-eps*dr).scaled)/(2*eps)
        np.testing.assert_allclose(J@dr,actual,rtol=6e-4,atol=2e-7)

    def test_solver_is_deterministic_and_does_not_mutate_inputs(self):
        tab,y0,t0,c=fixture()
        phi=y0.flux.copy();C=y0.precursors.copy();Ts=t0.solid_temperature_k.copy()
        a=advance_monolithic_rz_core(tab,y0,t0,c,.13,
                                     explicit_synthetic_deposition)
        b=advance_monolithic_rz_core(tab,y0,t0,c,.13,
                                     explicit_synthetic_deposition)
        np.testing.assert_array_equal(a.neutron_state.flux,b.neutron_state.flux)
        np.testing.assert_array_equal(a.neutron_state.precursors,b.neutron_state.precursors)
        np.testing.assert_array_equal(a.thermal_state.solid_temperature_k,
                                      b.thermal_state.solid_temperature_k)
        np.testing.assert_array_equal(y0.flux,phi)
        np.testing.assert_array_equal(y0.precursors,C)
        np.testing.assert_array_equal(t0.solid_temperature_k,Ts)

    def test_nonconvergent_newton_budget_is_not_accepted(self):
        tab,y0,t0,c=fixture()
        with self.assertRaisesRegex(RuntimeError,"did not converge"):
            advance_monolithic_rz_core(
                tab,y0,t0,c,.1,explicit_synthetic_deposition,
                controls=MonolithicControls(max_iterations=1,
                    scaled_residual_tolerance=1e-14))


if __name__=="__main__":
    unittest.main(verbosity=2)
