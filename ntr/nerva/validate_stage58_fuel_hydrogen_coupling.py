"""Stage58 numerical verification: one fuel/H2 conserved energy state.

All forcing and property values are SYNTHETIC. Tests establish discrete
coupling/limiting cases, not NASA NERVA hot operation or neutronic power.
"""
from __future__ import annotations

import unittest

import numpy as np

from .config import NervaConfig
from .fuel_cutcell_conduction_2d import build_cutcell_fuel_2d
from .fuel_cutcell_axial_transport import build_axial_cutcell_model
from .fuel_cutcell_nonlinear_enthalpy import (
    TemperatureTable, NonlinearThermalProperties,
)
from .hydrogen_channel_hydraulics import IdealHydrogenProperties
from .hydrogen_transient_fvm import (
    HydrogenBoundary, build_transient_hydrogen_model,
    make_state, primitive,
)
from .fuel_hydrogen_coupled_transient import (
    CoupledCoreState, advance_coupled_core, build_coupled_core_model,
)


def _temperature_laws(film=1400.):
    t=[240.,300.,500.,800.,1100.]
    eos=IdealHydrogenProperties(
        cp_j_kg_k=TemperatureTable(
            t,[14000.,14200.,14600.,15100.,15700.]),
        viscosity_pa_s=TemperatureTable(
            t,[8e-6,9e-6,12e-6,15e-6,19e-6]))
    p=NonlinearThermalProperties(
        fuel_cp_j_kg_k=TemperatureTable(t,[950.,1000.,1150.,1300.,1600.]),
        fuel_k_w_m_k=TemperatureTable(t,[16.,12.,10.,9.,8.]),
        coating_k_w_m_k=TemperatureTable(t,[22.,20.,18.,16.,15.]),
        hydrogen_cp_j_kg_k=eos.cp_j_kg_k,
        hydrogen_film_w_m2_k=TemperatureTable(
            t,[film]*len(t),allow_zero=True))
    return eos,p


def coupled(*, film=1400.,nz=2, nx=24,ny=30):
    c=NervaConfig()
    z=np.linspace(0.,c.active_length_cm/100,nz+1)
    cross=build_cutcell_fuel_2d(
        c,nx=nx,ny=ny,conductivity_w_m_k=12.,
        density_kg_m3=1000.,heat_capacity_j_kg_k=1000.,
        zrc_conductivity_w_m_k=20.,
        hydrogen_film_w_m2_k=film)
    solid=build_axial_cutcell_model(
        cross,axial_edges_m=z,
        solid_axial_conductivity_w_m_k=12.,
        hydrogen_mass_kg=.01,
        hydrogen_cp_j_kg_k=14200.,
        hydrogen_mass_flow_kg_s=0.,
        hydrogen_flow_cp_j_kg_k=14200.)
    eos,p=_temperature_laws(film)
    fluid=build_transient_hydrogen_model(c,eos,axial_edges_m=z)
    model=build_coupled_core_model(
        c,solid,fluid,p,solid_density_kg_m3=1000.)
    return model


def initial(m, tf=400., th=400., vel=0.):
    return CoupledCoreState(
        np.full((m.nzone,m.solid.transverse.n_fuel),tf),
        make_state(m.fluid,pressure_pa=3e6,
                   temperature_k=th,velocity_m_s=vel))


def solve(m,s,dt=5e-6,q=None,bc=HydrogenBoundary(),**kwargs):
    if q is None:
        q=np.zeros_like(s.fuel_temperature_k)
    return advance_coupled_core(
        m,s,dt_s=dt,fuel_deposition_w=q,
        boundary=bc,**kwargs)


class CoupledCorePhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=coupled()

    def test_zero_forcing_uniform_thermal_rest_stays_uniform(self):
        m=self.model
        old=initial(m,tf=400.,th=400.)
        r=solve(m,old)
        np.testing.assert_allclose(r.state.fuel_temperature_k,
                                   400.,rtol=0,atol=3e-10)
        t=primitive(m.fluid,r.state.hydrogen)
        np.testing.assert_allclose(t.temperature_k,400.,rtol=0,atol=3e-10)
        np.testing.assert_allclose(t.pressure_pa,3e6,rtol=0,atol=3e-7)
        self.assertLess(abs(r.first_law_residual_j),1e-6)
        self.assertEqual(r.substeps,1)
        self.assertLess(np.max(np.abs(r.wall_heat_to_hydrogen_j)),1e-9)

    def test_sealed_hot_solid_cools_and_fluid_heats_with_first_law(self):
        m=self.model
        old=initial(m,tf=550.,th=400.)
        r=solve(m,old,dt=2e-5)
        now=primitive(m.fluid,r.state.hydrogen)
        self.assertGreater(float(np.max(now.temperature_k)),400.)
        self.assertLess(float(np.min(r.state.fuel_temperature_k)),550.)
        self.assertGreater(float(np.sum(r.wall_heat_to_hydrogen_j)),0.)
        self.assertGreater(r.hydrogen_total_energy_change_j,0.)
        self.assertLess(r.fuel_storage_change_j,0.)
        self.assertAlmostEqual(r.hydrogen_net_outlet_energy_j,0.,delta=1e-10)
        self.assertLess(r.relative_first_law_residual,1e-8)

    def test_zero_contact_is_exact_independent_solid_and_fluid_limit(self):
        m=coupled(film=0.)
        old=initial(m,tf=400.,th=400.)
        q=1e7*m.solid.solid_volume_m3
        r=solve(m,old,dt=1e-5,q=q)
        self.assertEqual(float(np.sum(r.wall_heat_to_hydrogen_j)),0.)
        self.assertAlmostEqual(r.hydrogen_total_energy_change_j,0.,
                               delta=1e-8)
        self.assertAlmostEqual(r.fuel_storage_change_j,
                               r.fuel_deposition_j,delta=3e-7)
        self.assertGreater(float(np.mean(r.state.fuel_temperature_k)),400.)

    def test_fluid_pressure_feedback_is_from_conserved_state(self):
        m=self.model
        old=initial(m,tf=650.,th=400.)
        r=solve(m,old,dt=1e-5)
        p0=primitive(m.fluid,old.hydrogen)
        p1=primitive(m.fluid,r.state.hydrogen)
        self.assertGreater(float(np.mean(p1.pressure_pa)),
                           float(np.mean(p0.pressure_pa)))
        self.assertGreater(float(np.mean(p1.temperature_k)),
                           float(np.mean(p0.temperature_k)))
        np.testing.assert_allclose(
            p1.density_kg_m3,p0.density_kg_m3,
            atol=1e-12,rtol=0)
        self.assertLess(r.relative_first_law_residual,1e-8)

    def test_two_identical_channels_have_identical_conserved_response(self):
        m=self.model
        old=initial(m,tf=500.,th=400.)
        r=solve(m,old)
        h=r.wall_heat_to_hydrogen_j
        self.assertEqual(h.shape,(m.nzone,19))
        self.assertTrue(np.all(np.isfinite(h)))
        self.assertGreater(float(np.sum(h)),0.)
        # Thermal geometry is not identical among channels: edge channels
        # are clipped by the hex. Do not assert their power equality.
        self.assertGreater(float(np.max(h)-np.min(h)),0.)

    def test_open_flow_total_energy_accounts_for_actual_boundary_flux(self):
        m=self.model
        old=initial(m,tf=400.,th=400.,vel=20.)
        r=solve(m,old,dt=1e-5,bc=HydrogenBoundary("transmissive"))
        self.assertLess(abs(r.first_law_residual_j),1e-6)
        self.assertLess(abs(r.hydrogen_mass_change_kg),1e-10)
        self.assertLess(abs(r.hydrogen_net_outlet_energy_j),1e-8)

    def test_subcycling_and_energy_accounting(self):
        m=coupled(film=0.,nz=3)
        old=initial(m,tf=400.,th=400.)
        r=solve(m,old,dt=.00025)
        self.assertGreater(r.substeps,1)
        self.assertLess(r.relative_first_law_residual,1e-8)
        self.assertEqual(r.maximum_hydrogen_mach,0.)

    def test_property_and_mesh_mismatch_fails_closed(self):
        m=self.model
        c=m.config
        with self.assertRaises(ValueError):
            build_coupled_core_model(
                c,m.solid,m.fluid,
                NonlinearThermalProperties(
                    m.properties.fuel_cp_j_kg_k,
                    m.properties.fuel_k_w_m_k,
                    m.properties.coating_k_w_m_k,
                    TemperatureTable([240.,1100.],[14000.,14000.]),
                    m.properties.hydrogen_film_w_m2_k),
                solid_density_kg_m3=1000.)
        with self.assertRaises(ValueError):
            build_coupled_core_model(
                c,m.solid,m.fluid,m.properties,
                solid_density_kg_m3=-1.)
        s=initial(m)
        with self.assertRaises(ValueError):
            solve(m,s,q=np.ones((2,3)))
        with self.assertRaises(ValueError):
            solve(m,s,dt=-1.)

    def test_wall_mapping_keeps_19_distinct_channels_and_exact_fuel_volume(self):
        m=self.model
        self.assertEqual(m.solid.fuel_hydrogen_exchange_w_k.shape[1],
                         19*m.nzone)
        self.assertEqual(m.fluid.volume_m3.shape,(m.nzone,19))
        self.assertAlmostEqual(float(np.sum(m.solid.solid_volume_m3)),
                               float(np.sum(m.solid.transverse.fuel_volumes_m3)),
                               delta=1e-15)


if __name__=="__main__":
    unittest.main(verbosity=2)
