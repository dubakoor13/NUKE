"""Stage56 numerical acceptance: genuine 1D compressible momentum, no power tuning.

The declared gas data and operating points are SYNTHETIC checks, not
high-pressure H2 reference properties, qualified flight conditions,
or historical NERVA pressure-drop measurements.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np
from scipy.optimize import brentq

from .config import NervaConfig
from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .hydrogen_channel_hydraulics import (
    IdealHydrogenProperties,darcy_friction_factor,
    solve_ideal_hydrogen_hydraulics,
)


def synthetic_h2():
    return IdealHydrogenProperties(
        cp_j_kg_k=TemperatureTable(
            [240.,300.,500.,800.,1100.],
            [14000.,14200.,14600.,15100.,15700.]),
        viscosity_pa_s=TemperatureTable(
            [240.,300.,500.,800.,1100.],
            [8.e-6,9.e-6,12.e-6,15.e-6,19.e-6]),
    )


def simulate(*, nz=16, temperature=430., flow=.00025,
             pressure=3.e6, roughness=0., eos=None, inlet=None):
    c=NervaConfig()
    z=np.linspace(0.,c.active_length_cm/100,nz+1)
    T=np.full((nz,19),temperature) if np.isscalar(temperature) else temperature
    if inlet is None:
        inlet=T[0,:]
    return solve_ideal_hydrogen_hydraulics(
        c,synthetic_h2() if eos is None else eos,
        axial_edges_m=z,hydrogen_temperature_k=T,
        inlet_temperature_k=inlet,inlet_pressure_pa=pressure,
        mass_flow_kg_s=flow,absolute_roughness_m=roughness)


class IdealGasHydrogenThermodynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eos=synthetic_h2()

    def test_pressure_temperature_density_eos(self):
        e=self.eos
        t=400.
        p=2.e6
        rho=float(e.density_kg_m3(p,t))
        self.assertAlmostEqual(rho*e.gas_constant*t,p,delta=1e-9)
        self.assertAlmostEqual(
            float(e.density_kg_m3(2*p,t)),2*rho,delta=2e-14)
        self.assertAlmostEqual(
            float(e.density_kg_m3(p,2*t)),0.5*rho,delta=2e-14)

    def test_sound_speed_and_gamma_are_thermodynamically_consistent(self):
        e=self.eos
        t=460.
        cp=float(e.cp_j_kg_k.at(t))
        gamma=float(e.gamma(t))
        self.assertAlmostEqual(gamma,cp/(cp-e.gas_constant),delta=1e-13)
        self.assertAlmostEqual(float(e.sound_speed_m_s(t))**2,
                               gamma*e.gas_constant*t,delta=1e-8)
        self.assertGreater(gamma,1.)

    def test_hydrogen_enthalpy_is_exact_integral_of_supplied_cp(self):
        e=self.eos
        t=np.array([300.,405.,600.])
        eps=1e-4
        numerical=(e.enthalpy_j_kg(t+eps)-e.enthalpy_j_kg(t-eps))/(2*eps)
        np.testing.assert_allclose(numerical,e.cp_j_kg_k.at(t),
                                   atol=2e-5,rtol=3e-9)

    def test_invalid_gas_properties_and_pressures_are_rejected(self):
        with self.assertRaises(ValueError):
            IdealHydrogenProperties(
                cp_j_kg_k=TemperatureTable([200.,500.],[1000.,1000.]),
                viscosity_pa_s=self.eos.viscosity_pa_s)
        with self.assertRaises(ValueError):
            self.eos.density_kg_m3(-1.,300.)
        with self.assertRaises(ValueError):
            self.eos.density_kg_m3(1.e6,1190.)
        with self.assertRaises(ValueError):
            IdealHydrogenProperties(
                cp_j_kg_k=self.eos.cp_j_kg_k,
                viscosity_pa_s=self.eos.viscosity_pa_s,
                molar_mass_kg_mol=-1.)


class CompressibleHydraulicMomentumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eos=synthetic_h2()

    def test_zero_flow_retains_constant_pressure_even_with_hotter_gas(self):
        t=np.linspace(300.,650.,9)[:,None]+np.zeros((9,19))
        r=simulate(nz=9,temperature=t,flow=0.,inlet=300.)
        np.testing.assert_allclose(r.pressure_faces_pa,3e6,rtol=0,atol=0)
        np.testing.assert_allclose(r.velocity_faces_m_s,0,rtol=0,atol=0)
        np.testing.assert_allclose(r.friction_loss_pa,0,rtol=0,atol=0)
        np.testing.assert_allclose(r.momentum_residual_pa,0,rtol=0,atol=0)
        self.assertGreater(np.max(r.density_faces_kg_m3),
                           np.min(r.density_faces_kg_m3))

    def test_laminar_darcy_friction_is_exact_64_over_re(self):
        re=np.array([1.,100.,1000.,2300.])
        f=darcy_friction_factor(re,0.)
        np.testing.assert_allclose(f,64/re,atol=1e-14,rtol=0.)

    def test_turbulent_friction_positive_and_roughness_increases_drag(self):
        re=np.array([5e3,2e4,1e5])
        f0=darcy_friction_factor(re,0.)
        fr=darcy_friction_factor(re,1e-3)
        self.assertTrue(np.all(fr>f0))
        self.assertTrue(np.all(f0>0.))

    def test_pressure_velocity_mass_flux_and_momentum_closure(self):
        r=simulate()
        self.assertEqual(r.pressure_faces_pa.shape,(17,19))
        self.assertEqual(r.channel_gas_mass_kg.shape,(16,19))
        self.assertTrue(np.all(np.diff(r.pressure_faces_pa,axis=0)<0))
        self.assertGreater(float(r.pressure_drop_pa[0]),0.)
        self.assertTrue(np.all(r.pressure_faces_pa>0))
        area=math.pi*(NervaConfig().coolant_bore_diameter_cm*.01)**2/4
        np.testing.assert_allclose(
            r.density_faces_kg_m3*r.velocity_faces_m_s*area,
            np.broadcast_to(r.mass_flow_kg_s,(17,19)),
            rtol=4e-14,atol=1e-13)
        self.assertLess(float(np.max(np.abs(r.momentum_residual_pa))),1e-5)
        self.assertLess(r.maximum_mach,.75)
        self.assertGreater(r.total_fluid_inventory_kg,0.)

    def test_isothermal_friction_and_compressibility_integrated_analytic(self):
        # Exact isothermal momentum integral with constant Darcy f:
        # 0.5*(pin²-pout²)-G²*RT*ln(pin/pout)
        #      = f*G²*RT*L/(2D).
        r=simulate(nz=96,temperature=420.,flow=.001,pressure=3e6)
        c=NervaConfig()
        D=c.coolant_bore_diameter_cm*.01
        area=math.pi*D*D/4
        G=.001/area
        R=self.eos.gas_constant
        T=420.
        f=float(r.darcy_friction_segments[0,0])
        length=c.active_length_cm*.01
        pin=3e6
        exact=brentq(
            lambda pout: .5*(pin*pin-pout*pout)
              -G*G*R*T*math.log(pin/pout)
              -f*G*G*R*T*length/(2*D),
            1e6,pin)
        self.assertAlmostEqual(float(r.pressure_faces_pa[-1,0]),exact,
                               delta=8.0)

    def test_refinement_converges_to_isothermal_integral(self):
        kwargs=dict(temperature=520.,flow=.0011,pressure=3e6)
        coarse=simulate(nz=4,**kwargs)
        medium=simulate(nz=16,**kwargs)
        fine=simulate(nz=64,**kwargs)
        c=NervaConfig()
        D=c.coolant_bore_diameter_cm*.01
        area=math.pi*D*D/4
        G=.0011/area
        f=float(fine.darcy_friction_segments[0,0])
        exact=brentq(
            lambda p: .5*(3e6**2-p**2)
                      -G**2*self.eos.gas_constant*520.*math.log(3e6/p)
                      -f*G**2*self.eos.gas_constant*520.*
                       (c.active_length_cm*.01)/(2*D),
            1e6,3e6)
        err=np.array([abs(float(r.pressure_faces_pa[-1,0])-exact)
                      for r in (coarse,medium,fine)])
        self.assertLess(err[2],err[1])
        self.assertLess(err[1],err[0])

    def test_hotter_outlet_hydrogen_changes_compressible_pressure_profile(self):
        cold=simulate(temperature=330.,inlet=330.,flow=.0006)
        hot_t=np.linspace(330.,720.,16)[:,None]+np.zeros((16,19))
        hot=simulate(temperature=hot_t,inlet=330.,flow=.0006)
        self.assertGreater(hot.pressure_drop_pa[0],cold.pressure_drop_pa[0])
        self.assertGreater(hot.density_faces_kg_m3[0,0],
                           hot.density_faces_kg_m3[-1,0])
        self.assertLess(float(np.max(np.abs(hot.momentum_residual_pa))),1e-5)

    def test_independent_channels_preserve_different_mass_flow_rates(self):
        flow=np.linspace(.0002,.001,19)
        pin=np.linspace(3.e6,3.4e6,19)
        r=simulate(flow=flow,pressure=pin)
        np.testing.assert_allclose(r.mass_flow_kg_s,flow,rtol=0,atol=0)
        self.assertTrue(np.all(r.pressure_drop_pa>0))
        self.assertGreater(np.std(r.pressure_drop_pa),100.)

    def test_flux_enthalpy_uses_exact_cp_integral(self):
        r=simulate(temperature=430.,inlet=300.)
        expected=r.mass_flow_kg_s[None,:]*self.eos.enthalpy_j_kg(
            r.temperature_faces_k)
        np.testing.assert_allclose(r.flow_enthalpy_faces_w,expected,rtol=0,atol=0)
        self.assertGreater(
            r.flow_enthalpy_faces_w[-1,0],r.flow_enthalpy_faces_w[0,0])

    def test_subsonic_flow_guard_fails_closed_for_unqualified_mass_flux(self):
        with self.assertRaisesRegex(ValueError,"Mach|subsonic|choke"):
            simulate(flow=.05,pressure=3e6)
        with self.assertRaises(ValueError):
            simulate(temperature=1170.)
        with self.assertRaises(ValueError):
            simulate(pressure=-1.)
        with self.assertRaises(ValueError):
            simulate(roughness=.001)

    def test_validated_pressure_requires_same_geometric_length(self):
        c=replace(NervaConfig(),active_length_cm=132.)
        with self.assertRaises(ValueError):
            solve_ideal_hydrogen_hydraulics(
                c,self.eos,axial_edges_m=[0.,1.2],
                hydrogen_temperature_k=np.full((1,19),450.),
                inlet_temperature_k=400.,inlet_pressure_pa=3e6,
                mass_flow_kg_s=.0005)


if __name__=="__main__":
    unittest.main(verbosity=2)
