"""Stage20 independent variable-cp frozen-H2 finite-volume equation checks.

Only SYNTHETIC heating, geometry and fuel materials. NASA-7 GRI-Mech H2
is an existing reference approximation valid 200..3500 K; these tests
DO NOT validate a NERVA fuel design, molecular dissociation, or reactor
neutron/photon deposition.
"""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
from scipy.optimize import brentq

from .fx2th_enthalpy_conjugate_core import (
    EnthalpyCoreControls,_EnthalpySystem,advance_enthalpy_conjugate_core,
)
from .fx2th_conjugate_fuel_transport import (
    ConjugateCoreState,ConjugateCoreDeposition,advance_conjugate_core
)
from .validate_fx2th_conjugate_fuel_transport import fixture,heat
from .implicit_hydrogen import hydrogen_specific_enthalpy
from .nasa_hydrogen import hydrogen_state,T_MIN,T_MAX


def h(t):
    return hydrogen_specific_enthalpy(t,1000.,"nasa7-h2")


def direct_deposition(geo, qh):
    p,z,ring=geo.shape
    heatgas=np.broadcast_to(np.asarray(qh,dtype=float),(p,z)).copy()
    return ConjugateCoreDeposition(
        "synthetic-direct-h2-heating-only",
        np.zeros((p,z,ring)),heatgas,np.zeros((p,z)),
        float(heatgas.sum()),True)


class EnthalpyResolvedCoreTests(unittest.TestCase):
    def test_analytic_nasa7_enthalpy_derivative_equals_frozen_cp(self):
        for t in (250.,330.,550.,999.,1001.,1800.,3000.):
            with self.subTest(temperature=t):
                central=(h(t+.001)-h(t-.001))/.002
                cp=hydrogen_state(t,101325.,mode="frozen").frozen_cp_j_kg_k
                np.testing.assert_allclose(central,cp,rtol=2e-7,atol=1e-4)

    def test_constant_cp_reduces_exactly_to_previous_resolved_FV(self):
        geo,old,power=fixture(npath=2,nz=3,nr=3)
        linear=advance_conjugate_core(geo,old,power,.17)
        nonlinear=advance_enthalpy_conjugate_core(
            geo,old,power,.17,
            controls=EnthalpyCoreControls(enthalpy_model="constant"))
        np.testing.assert_allclose(
            nonlinear.state.fuel_temperature_k,
            linear.state.fuel_temperature_k,atol=1e-9,rtol=3e-12)
        np.testing.assert_allclose(
            nonlinear.state.hydrogen_temperature_k,
            linear.state.hydrogen_temperature_k,atol=1e-9,rtol=3e-12)
        self.assertAlmostEqual(nonlinear.fuel_storage_j,
                               linear.fuel_storage_j,delta=1e-8)
        self.assertAlmostEqual(nonlinear.outlet_net_enthalpy_j,
                               linear.outlet_net_enthalpy_j,delta=1e-8)
        self.assertLess(nonlinear.maximum_equation_residual_w,1e-7)

    def test_nasa7_one_cell_transient_matches_independent_scalar_root(self):
        geo,old,_=fixture(npath=1,nz=1,nr=1,flow=.002,wall=0.)
        old=ConjugateCoreState(0.,np.full(geo.shape,440.),
                              np.full((1,1),410.))
        q=direct_deposition(geo,[[850.]])
        dt=.16
        out=advance_enthalpy_conjugate_core(geo,old,q,dt)
        m=geo.hydrogen_mass_kg[0,0]
        flow=geo.hydrogen_mass_flow_kg_s[0]
        oldh=h(410.);inleth=h(geo.inlet_temperature_k[0])
        def eq(t):
            return m*(h(t)-oldh)/dt+flow*(h(t)-inleth)-850.
        target=brentq(eq,T_MIN,T_MAX,xtol=2e-11)
        self.assertAlmostEqual(out.state.hydrogen_temperature_k[0,0],
                               target,places=8)
        self.assertAlmostEqual(out.state.fuel_temperature_k[0,0,0],440.,places=9)
        self.assertLess(out.thermal_relative_residual,1e-10)

    def test_nasa7_steady_axial_upwind_enthalpy_exact_reference(self):
        geo,old,_=fixture(npath=1,nz=3,nr=1,flow=.004,wall=0.)
        q=direct_deposition(geo,[[900.,1500.,2300.]])
        output=advance_enthalpy_conjugate_core(geo,old,q,1e9)
        flow=geo.hydrogen_mass_flow_kg_s[0]
        up=h(geo.inlet_temperature_k[0])
        expected=[]
        for power in (900.,1500.,2300.):
            up+=power/flow
            expected.append(brentq(lambda t:h(t)-up,T_MIN,T_MAX,xtol=2e-11))
        np.testing.assert_allclose(output.state.hydrogen_temperature_k[0],
                                  expected,atol=3e-4,rtol=0)
        self.assertAlmostEqual(output.outlet_net_enthalpy_j/1e9,4700.,
                               delta=.002)

    def test_variable_enthalpy_changes_coolant_temperature_vs_constant_cp(self):
        geo,old,_=fixture(npath=1,nz=2,nr=1,flow=.004,wall=0.)
        power=direct_deposition(geo,[[800.,1400.]])
        variable=advance_enthalpy_conjugate_core(geo,old,power,.3)
        linear=advance_enthalpy_conjugate_core(
            geo,old,power,.3,
            controls=EnthalpyCoreControls(enthalpy_model="constant"))
        self.assertGreater(np.linalg.norm(
            variable.state.hydrogen_temperature_k-
            linear.state.hydrogen_temperature_k),3.)
        self.assertLess(variable.thermal_relative_residual,1e-10)

    def test_global_first_law_with_full_solid_radial_axial_and_gas(self):
        geo,old,power=fixture(npath=2,nz=3,nr=3,wall=80.)
        output=advance_enthalpy_conjugate_core(geo,old,power,.27)
        self.assertLess(output.thermal_relative_residual,1e-10)
        self.assertAlmostEqual(
            output.fuel_storage_j+output.hydrogen_enthalpy_storage_j+
            output.outlet_net_enthalpy_j+output.other_deposition_j,
            output.deposited_total_j,
            delta=1e-8*max(1.,output.deposited_total_j))
        self.assertTrue(output.synthetic)
        self.assertEqual(output.source_id,power.source_id)

    def test_reverse_wall_heat_can_warm_fuel(self):
        geo,old,_=fixture(npath=1,nz=1,nr=2,flow=0.,wall=110.)
        old=ConjugateCoreState(0.,np.full(geo.shape,310.),
                              np.full((1,1),650.))
        result=advance_enthalpy_conjugate_core(geo,old,heat(geo),.05)
        self.assertLess(result.wall_transfer_j,0)
        self.assertGreater(result.state.fuel_temperature_k[0,0,0],310.)
        self.assertLess(result.state.hydrogen_temperature_k[0,0],650.)
        self.assertLess(result.thermal_relative_residual,1e-10)

    def test_zero_flow_zero_heat_uniform_temperature_is_exact_steady(self):
        geo,_,_=fixture(npath=1,nz=3,nr=2,flow=0.,wall=55.)
        old=ConjugateCoreState(0.,np.full(geo.shape,725.),
                              np.full((1,3),725.))
        result=advance_enthalpy_conjugate_core(geo,old,heat(geo),.5)
        np.testing.assert_allclose(result.state.fuel_temperature_k,725.,
                                   rtol=0,atol=2e-9)
        np.testing.assert_allclose(result.state.hydrogen_temperature_k,725.,
                                   rtol=0,atol=2e-9)
        self.assertLess(result.thermal_relative_residual,1e-10)

    def test_exact_jacobian_matches_independent_centered_difference(self):
        geo,old,power=fixture(npath=1,nz=2,nr=2)
        task=_EnthalpySystem(geo,old,power,.13,EnthalpyCoreControls())
        x=np.r_[old.fuel_temperature_k.ravel(),
                old.hydrogen_temperature_k.ravel()]
        residual,operator=task.equation(x,jacobian=True)
        direction=np.array([.7,-.3,.2,-.4,.6,-.9])
        step=1e-4
        fp,_=task.equation(x+step*direction)
        fm,_=task.equation(x-step*direction)
        measured=(fp-fm)/(2*step)
        np.testing.assert_allclose(operator@direction,measured,
                                   atol=2e-6,rtol=3e-7)

    def test_molecular_nasa7_enthalpy_reference_not_cp_reference(self):
        geo,old,p=fixture(npath=1,nz=2,nr=2,flow=.004,wall=50.)
        othergeo=replace(geo,hydrogen_cp_j_kg_k=np.array([3000.]))
        a=advance_enthalpy_conjugate_core(geo,old,p,.16)
        b=advance_enthalpy_conjugate_core(othergeo,old,p,.16)
        np.testing.assert_allclose(
            a.state.hydrogen_temperature_k,b.state.hydrogen_temperature_k,
            rtol=0,atol=3e-10)
        np.testing.assert_allclose(
            a.state.fuel_temperature_k,b.state.fuel_temperature_k,
            rtol=0,atol=3e-10)

    def test_staged_fuel_wall_coolant_energy_is_not_double_counted(self):
        geo,old,_=fixture(npath=1,nz=3,nr=3,flow=.003,wall=70.)
        power=heat(geo,solid=100.,h2=5.,other=11.)
        result=advance_enthalpy_conjugate_core(geo,old,power,.2)
        self.assertAlmostEqual(result.other_deposition_j,
                               .2*3*11.,places=11)
        self.assertAlmostEqual(result.deposited_total_j,
                               .2*power.declared_total_w,places=11)
        self.assertLess(result.thermal_relative_residual,1e-10)

    def test_nonuniform_fuel_properties_and_parallel_channel_count(self):
        geo,old,p=fixture(npath=2,nz=3,nr=3)
        conductivity=np.asarray(geo.solid_conductivity_w_m_k).copy()
        conductivity[1,2,0]=.8
        geo=replace(geo,solid_conductivity_w_m_k=conductivity,
                    channels_per_radial_zone=np.array([2,5]))
        result=advance_enthalpy_conjugate_core(geo,old,p,.13)
        self.assertLess(result.thermal_relative_residual,1e-9)
        self.assertTrue(np.all(np.isfinite(result.state.fuel_temperature_k)))

    def test_nasa7_domain_rejects_cryogenic_h2_initial(self):
        geo,old,p=fixture(npath=1,nz=1,nr=1)
        old=replace(old,hydrogen_temperature_k=np.array([[100.]]))
        with self.assertRaisesRegex(ValueError,"NASA-7|NASA|range|enthalpy"):
            advance_enthalpy_conjugate_core(geo,old,p,.1)

    def test_nasa7_domain_rejects_out_of_range_inlet(self):
        geo,old,p=fixture(npath=1,nz=1,nr=1)
        bad=replace(geo,inlet_temperature_k=np.array([190.]))
        with self.assertRaises(ValueError):
            advance_enthalpy_conjugate_core(bad,old,p,.1)

    def test_extreme_heating_outside_nasa7_domain_fails_closed(self):
        geo,old,p=fixture(npath=1,nz=1,nr=1,flow=0,wall=0)
        huge=direct_deposition(geo,[[1e9]])
        with self.assertRaisesRegex(RuntimeError,"line search|did not converge"):
            advance_enthalpy_conjugate_core(geo,old,huge,.25)

    def test_invalid_deposition_and_time_fail_closed(self):
        geo,old,p=fixture(npath=1,nz=2,nr=2)
        with self.assertRaises(ValueError):
            advance_enthalpy_conjugate_core(
                geo,old,replace(p,declared_total_w=5.),.1)
        for dt in (0.,-.1,float("nan")):
            with self.subTest(dt=dt),self.assertRaises(ValueError):
                advance_enthalpy_conjugate_core(geo,old,p,dt)

    def test_max_unknowns_and_newton_budget_enforced(self):
        geo,old,p=fixture(npath=1,nz=2,nr=2)
        with self.assertRaisesRegex(ValueError,"maximum"):
            advance_enthalpy_conjugate_core(
                geo,old,p,.1,controls=EnthalpyCoreControls(max_unknowns=1))
        with self.assertRaisesRegex(RuntimeError,"did not converge"):
            advance_enthalpy_conjugate_core(
                geo,old,p,.1,controls=EnthalpyCoreControls(
                    max_iterations=1,residual_tolerance_w=1e-13))

    def test_repeatability_and_old_state_immutability(self):
        geo,old,p=fixture(npath=2,nz=3,nr=3)
        previous=old.fuel_temperature_k.copy()
        a=advance_enthalpy_conjugate_core(geo,old,p,.09)
        b=advance_enthalpy_conjugate_core(geo,old,p,.09)
        np.testing.assert_array_equal(a.state.fuel_temperature_k,
                                      b.state.fuel_temperature_k)
        np.testing.assert_array_equal(a.state.hydrogen_temperature_k,
                                      b.state.hydrogen_temperature_k)
        np.testing.assert_array_equal(old.fuel_temperature_k,previous)


if __name__=="__main__":
    unittest.main(verbosity=2)
