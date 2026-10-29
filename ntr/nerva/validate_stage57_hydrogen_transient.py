"""Stage57 transient 19-channel compressible H2 numerical acceptance tests.

All gas transport and thermodynamic tables in this file are SYNTHETIC,
not NERVA operating conditions or validated real-gas correlations.
"""
from __future__ import annotations

import math
import unittest
import numpy as np

from .config import NervaConfig
from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .hydrogen_channel_hydraulics import IdealHydrogenProperties
from .hydrogen_transient_fvm import (
    ConservativeHydrogenState, HydrogenBoundary,
    _e_internal, _T_from_e,
    advance_transient_hydrogen, build_transient_hydrogen_model,
    make_state, primitive,
)


def synthetic_eos() -> IdealHydrogenProperties:
    return IdealHydrogenProperties(
        cp_j_kg_k=TemperatureTable(
            [240.,300.,500.,800.,1100.],
            [14000.,14200.,14600.,15100.,15700.]),
        viscosity_pa_s=TemperatureTable(
            [240.,300.,500.,800.,1100.],
            [8e-6,9e-6,12e-6,15e-6,19e-6]))


def model(nz: int=4,roughness: float=0.):
    c=NervaConfig()
    return build_transient_hydrogen_model(
        c,synthetic_eos(),
        axial_edges_m=np.linspace(0.,c.active_length_cm/100,nz+1),
        absolute_roughness_m=roughness)


def totals(m,state):
    return np.sum(state.conserved*m.volume_m3[...,None],axis=(0,1))


class CaloricallyImperfectHydrogenTests(unittest.TestCase):
    def test_exact_piecewise_linear_cv_primitive_and_inverse(self):
        e=synthetic_eos()
        t=np.array([240.,241.,280.,300.,410.,500.,699.,800.,1100.])
        internal=_e_internal(e,t)
        self.assertAlmostEqual(float(internal[0]),0.)
        self.assertTrue(np.all(np.diff(internal)>0))
        np.testing.assert_allclose(_T_from_e(e,internal),t,atol=2e-11,rtol=0)
        tmid=np.array([251.,360.,750.])
        eps=.001
        derivative=(_e_internal(e,tmid+eps)-_e_internal(e,tmid-eps))/(2*eps)
        np.testing.assert_allclose(
            derivative,e.cp_j_kg_k.at(tmid)-e.gas_constant,
            rtol=1e-8,atol=1e-3)

    def test_state_density_sound_speed_and_pressure_recover_stage56_eos(self):
        m=model()
        s=make_state(m,pressure_pa=3e6,temperature_k=450.,velocity_m_s=25.)
        v=primitive(m,s)
        np.testing.assert_allclose(v.pressure_pa,3e6,atol=1e-8,rtol=5e-15)
        np.testing.assert_allclose(
            v.density_kg_m3,m.eos.density_kg_m3(3e6,450.),atol=1e-14)
        np.testing.assert_allclose(
            v.sound_speed_m_s,m.eos.sound_speed_m_s(450.),atol=1e-11)

    def test_negative_internal_energy_temperature_and_supersonic_guards(self):
        m=model()
        s=make_state(m,pressure_pa=3e6,temperature_k=400.)
        corrupt=s.conserved.copy()
        corrupt[:,:,2]=-1.
        with self.assertRaises(ValueError):
            primitive(m,ConservativeHydrogenState(corrupt))
        with self.assertRaises(ValueError):
            make_state(m,pressure_pa=3e6,temperature_k=1150.)
        with self.assertRaises(ValueError):
            make_state(m,pressure_pa=3e6,temperature_k=400.,velocity_m_s=2500.)


class ConservativeFiniteVolumeTests(unittest.TestCase):
    def test_resting_adiabatic_sealed_channel_is_stationary(self):
        m=model(5)
        old=make_state(m,pressure_pa=3e6,temperature_k=420.)
        r=advance_transient_hydrogen(m,old,dt_s=1e-6)
        np.testing.assert_allclose(r.state.conserved,old.conserved,atol=1e-12,rtol=0)
        np.testing.assert_allclose(r.inlet_mass_flux_kg_s,0.,atol=0,rtol=0)
        np.testing.assert_allclose(r.outlet_energy_flux_w,0.,atol=0,rtol=0)
        self.assertLess(abs(r.mass_balance_residual_kg),1e-13)
        self.assertLess(abs(r.energy_balance_residual_j),1e-9)

    def test_constant_transmissive_solution_with_no_wall_friction(self):
        m=model(6)
        old=make_state(m,pressure_pa=3e6,temperature_k=430.,velocity_m_s=30.)
        r=advance_transient_hydrogen(
            m,old,dt_s=2e-6,boundary=HydrogenBoundary("transmissive"),
            friction_enabled=False)
        np.testing.assert_allclose(r.state.conserved,old.conserved,atol=2e-10,rtol=0)
        self.assertGreater(float(np.sum(r.inlet_mass_flux_kg_s)),0)
        np.testing.assert_allclose(r.inlet_mass_flux_kg_s,r.outlet_mass_flux_kg_s,
                                   atol=1e-13,rtol=0)
        np.testing.assert_allclose(r.inlet_energy_flux_w,r.outlet_energy_flux_w,
                                   atol=1e-8,rtol=0)
        self.assertLess(abs(r.energy_balance_residual_j),1e-7)

    def test_exact_cv_enthalpy_wall_heating_increases_pressure(self):
        m=model(4)
        initial=make_state(m,pressure_pa=3e6,temperature_k=350.)
        rho=primitive(m,initial).density_kg_m3
        de=float(_e_internal(m.eos,351.)-_e_internal(m.eos,350.))
        dt=2e-6
        power=rho*m.volume_m3*de/dt
        r=advance_transient_hydrogen(m,initial,dt_s=dt,wall_heat_w=power)
        p=primitive(m,r.state)
        np.testing.assert_allclose(p.temperature_k,351.,atol=3e-10,rtol=0)
        np.testing.assert_allclose(p.pressure_pa,3e6*351./350.,atol=8e-6,rtol=0)
        np.testing.assert_allclose(p.density_kg_m3,rho,atol=2e-14,rtol=0)
        self.assertAlmostEqual(r.mass_change_kg,0.,delta=2e-13)
        self.assertAlmostEqual(r.total_energy_change_j,r.wall_heat_j,delta=4e-8)
        self.assertLess(abs(r.energy_balance_residual_j),4e-8)

    def test_pressure_wave_conserves_sealed_total_mass_and_energy(self):
        m=model(12)
        p=np.full((12,19),3e6)
        p[:6,:]*=1.002
        initial=make_state(m,pressure_pa=p,temperature_k=420.)
        old=totals(m,initial)
        r=advance_transient_hydrogen(m,initial,dt_s=2e-6)
        updated=primitive(m,r.state)
        self.assertGreater(float(updated.velocity_m_s[6,0]),0.)
        self.assertGreater(float(updated.pressure_pa[6,0]),float(p[6,0]))
        self.assertLess(abs(r.mass_balance_residual_kg),1e-13)
        self.assertLess(abs(r.energy_balance_residual_j),1e-7)
        np.testing.assert_allclose(totals(m,r.state)[[0,2]],old[[0,2]],
                                   atol=2e-9,rtol=2e-13)

    def test_adabatic_friction_dissipates_kinetic_not_total_energy(self):
        m=model(1,roughness=1e-6)
        initial=make_state(
            m,pressure_pa=3e6,temperature_k=400.,velocity_m_s=100.)
        r=advance_transient_hydrogen(m,initial,dt_s=1e-6)
        p=primitive(m,r.state)
        self.assertLess(float(p.velocity_m_s[0,0]),100.)
        self.assertGreater(float(p.temperature_k[0,0]),400.)
        self.assertLess(r.momentum_wall_friction_impulse_n_s,0.)
        self.assertLess(abs(r.momentum_balance_residual_kg_m_s),1e-10)
        self.assertLess(abs(r.energy_balance_residual_j),1e-8)
        self.assertAlmostEqual(r.total_energy_change_j,0.,delta=1e-8)

    def test_reservoir_pressure_wave_balance_uses_actual_face_fluxes(self):
        m=model(6)
        initial=make_state(m,pressure_pa=3e6,temperature_k=400.)
        bc=HydrogenBoundary(
            "reservoir",inlet_pressure_pa=3.1e6,
            inlet_temperature_k=400.,inlet_velocity_m_s=0.,
            outlet_pressure_pa=2.9e6,
            outlet_temperature_k=400.,outlet_velocity_m_s=0.)
        r=advance_transient_hydrogen(
            m,initial,dt_s=1e-6,boundary=bc,friction_enabled=False)
        self.assertGreater(float(np.sum(r.inlet_mass_flux_kg_s)),0.)
        self.assertGreater(float(np.sum(r.outlet_mass_flux_kg_s)),0.)
        expected=1e-6*float(np.sum(
            r.inlet_mass_flux_kg_s-r.outlet_mass_flux_kg_s))
        self.assertAlmostEqual(r.mass_change_kg,expected,delta=1e-13)
        self.assertLess(abs(r.mass_balance_residual_kg),1e-13)
        self.assertLess(abs(r.energy_balance_residual_j),1e-7)
        self.assertGreater(float(primitive(m,r.state).velocity_m_s[0,0]),0.)

    def test_heating_specific_channel_does_not_heat_neighbors(self):
        m=model(4)
        initial=make_state(m,pressure_pa=3e6,temperature_k=400.)
        q=np.zeros((4,19))
        q[1,5]=1000.
        r=advance_transient_hydrogen(m,initial,dt_s=1e-6,wall_heat_w=q)
        t=primitive(m,r.state).temperature_k
        self.assertGreater(float(t[1,5]),400.)
        np.testing.assert_allclose(t[:,0],400.,atol=1e-11,rtol=0)
        self.assertLess(abs(r.energy_balance_residual_j),1e-8)

    def test_two_half_steps_agree_with_full_step_for_smooth_pressure_pulse(self):
        m=model(8)
        x=(np.arange(8)+.5)/8
        p=(3e6*(1+.002*np.sin(2*np.pi*x)))[:,None]*np.ones((1,19))
        initial=make_state(m,pressure_pa=p,temperature_k=410.)
        dt=2e-6
        full=advance_transient_hydrogen(m,initial,dt_s=dt,friction_enabled=False)
        half=advance_transient_hydrogen(m,initial,dt_s=dt/2,friction_enabled=False)
        two=advance_transient_hydrogen(m,half.state,dt_s=dt/2,
                                       friction_enabled=False)
        diff=np.max(np.abs(full.state.conserved-two.state.conserved),axis=(0,1))
        scale=np.maximum(np.max(np.abs(initial.conserved),axis=(0,1)),
                         [1.,2000.,1e6])
        self.assertLess(float(np.max(diff/scale)),1e-5)
        self.assertLess(abs(full.energy_balance_residual_j),1e-7)

    def test_rejects_oversized_cfl_bad_heating_geometry_and_out_of_bounds(self):
        m=model(4)
        initial=make_state(m,pressure_pa=3e6,temperature_k=400.)
        with self.assertRaisesRegex(ValueError,"CFL"):
            advance_transient_hydrogen(m,initial,dt_s=.1)
        with self.assertRaises(ValueError):
            advance_transient_hydrogen(
                m,initial,dt_s=1e-6,wall_heat_w=np.zeros((2,18)))
        with self.assertRaises(ValueError):
            advance_transient_hydrogen(m,initial,dt_s=-1)
        with self.assertRaises(ValueError):
            advance_transient_hydrogen(
                m,initial,dt_s=1e-6,boundary=HydrogenBoundary("bad"))
        with self.assertRaises(ValueError):
            advance_transient_hydrogen(m,initial,dt_s=1e-6,wall_heat_w=1e13)

    def test_preserves_actual_nerva_bore_diameter_and_19_channels(self):
        m=model(16)
        exact=math.pi*(NervaConfig().coolant_bore_diameter_cm/100)**2/4
        self.assertAlmostEqual(m.area_m2,exact,delta=1e-20)
        self.assertEqual(m.volume_m3.shape,(16,19))
        self.assertAlmostEqual(float(np.sum(m.volume_m3)),
                               19*exact*1.32,delta=1e-15)


if __name__=="__main__":
    unittest.main(verbosity=2)
