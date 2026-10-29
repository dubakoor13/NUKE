"""Stage70 numerical verification of resolved fuel–coating–hydrogen coupling.

Synthetic cp(T), coating mass/density, contact coefficients and source watts
demonstrate *numerical* conservation; not NASA NERVA thermal validation.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .fuel_hydrogen_coupled_transient import (
    with_resolved_coating, advance_coupled_core,
)
from .resolved_coating_thermal import (
    build_resolved_coating, coating_interface_power,
    advance_coating_enthalpy,
)
from .validate_stage58_fuel_hydrogen_coupling import coupled,initial,solve


def coating_model(*, film=1250.,nz=2,nx=20,ny=24):
    base=coupled(film=film,nz=nz,nx=nx,ny=ny)
    cp=TemperatureTable(
        [240.,300.,500.,800.,1100.],
        [690.,725.,800.,890.,1000.])
    return with_resolved_coating(
        base,density_kg_m3=5000.,
        heat_capacity=cp,contact_resistance_m2_k_w=1e-5)


def coating_initial(m, tf=400., tc=400., th=400.):
    s=initial(m,tf=tf,th=th)
    return replace(
        s,coating_temperature_k=np.full((m.nzone,19),tc))


class ResolvedCoatingThermalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=coating_model()

    def test_exact_annular_geometry_positive_and_contact_closure(self):
        m=self.m
        coat=m.coating
        self.assertEqual(coat.volume_m3.shape,(2,19))
        self.assertEqual(coat.inner_bore_area_m2.shape,(2,19))
        self.assertTrue(np.all(coat.mass_kg>0))
        self.assertTrue(np.all(coat.inner_bore_area_m2>0))
        self.assertEqual(coat.fuel_arc_m2.shape,
                         (m.solid.transverse.n_fuel,19))
        self.assertAlmostEqual(
            coat.volume_m3.sum(),
            sum(coat.volume_m3[0,:])*2,delta=1e-14)
        full_annulus=19*math.pi*(
            m.config.coated_channel_radius_cm**2
            -m.config.channel_radius_cm**2)*1e-4*1.32
        self.assertLessEqual(float(coat.volume_m3.sum()),full_annulus+1e-14)
        # Peripheral clipping preserves the bore material identity:
        self.assertLess(float(coat.volume_m3.sum()),full_annulus)

    def test_uniform_equilibrium_zero_sources_is_stationary(self):
        m=self.m
        s=coating_initial(m)
        r=solve(m,s,dt=5e-6)
        np.testing.assert_allclose(
            r.state.fuel_temperature_k,s.fuel_temperature_k,atol=1e-10)
        np.testing.assert_allclose(
            r.state.coating_temperature_k,s.coating_temperature_k,atol=1e-10)
        self.assertLess(abs(r.coating_storage_change_j),2e-9)
        self.assertEqual(r.coating_deposition_j,0.)
        self.assertLess(abs(r.first_law_residual_j),3e-8)

    def test_selective_coating_heating_changes_only_resolved_coating(self):
        m=coating_model(film=0.)
        old=coating_initial(m)
        q=np.zeros((m.nzone,19))
        q[1,8]=120.
        r=solve(m,old,dt=5e-6,coating_deposition_w=q)
        self.assertAlmostEqual(r.coating_deposition_j,120*5e-6,delta=2e-15)
        self.assertGreater(r.state.coating_temperature_k[1,8],400.)
        self.assertEqual(float(np.sum(r.wall_heat_to_hydrogen_j)),0.)
        self.assertLess(abs(r.hydrogen_total_energy_change_j),2e-8)
        self.assertLess(abs(r.coating_first_law_residual_j),4e-8)
        self.assertLess(r.relative_first_law_residual,2e-8)

    def test_hot_coating_transfers_to_both_fuel_and_hydrogen(self):
        m=self.m
        old=coating_initial(m,tf=400.,tc=510.,th=400.)
        r=solve(m,old,dt=5e-6)
        self.assertLess(r.state.coating_temperature_k.mean(),510.)
        self.assertLess(r.fuel_to_coating_j,0.)
        self.assertGreater(r.coating_to_hydrogen_j,0.)
        self.assertGreater(r.fuel_storage_change_j,0.)
        self.assertGreater(r.hydrogen_total_energy_change_j,0.)
        self.assertAlmostEqual(
            r.coating_to_hydrogen_j,
            float(np.sum(r.wall_heat_to_hydrogen_j)),delta=1e-13)
        self.assertLess(r.relative_first_law_residual,2e-8)

    def test_three_separate_first_law_ledgers_close_together(self):
        m=self.m
        old=coating_initial(m,tf=495.,tc=445.,th=395.)
        fuel_q=np.full_like(old.fuel_temperature_k,2.)
        coat_q=np.full((m.nzone,19),8.)
        gas_q=np.full((m.nzone,19),5.)
        dt=5e-6
        r=solve(m,old,dt=dt,q=fuel_q,
                coating_deposition_w=coat_q,hydrogen_deposition_w=gas_q)
        self.assertAlmostEqual(
            r.fuel_storage_change_j+r.fuel_to_coating_j,
            r.fuel_deposition_j,delta=8e-7)
        self.assertAlmostEqual(
            r.coating_storage_change_j+r.coating_to_hydrogen_j-
            r.fuel_to_coating_j,r.coating_deposition_j,delta=8e-7)
        self.assertAlmostEqual(
            r.hydrogen_total_energy_change_j+
            r.hydrogen_net_outlet_energy_j-r.coating_to_hydrogen_j,
            r.hydrogen_deposition_j,delta=1e-7)
        self.assertLess(r.relative_first_law_residual,3e-8)

    def test_coating_enthalpy_exact_integral_with_no_contact(self):
        m=self.m
        coat=m.coating
        old=np.full((m.nzone,19),400.)
        q=np.zeros_like(old)
        q[1,10]=60.
        dt=.01
        updated=advance_coating_enthalpy(
            coat,old,dt_s=dt,net_w=q)
        h=coat.heat_capacity.integral(updated)
        h0=coat.heat_capacity.integral(old)
        np.testing.assert_allclose(
            coat.mass_kg*(h-h0),q*dt,
            atol=3e-10,rtol=0)
        self.assertEqual(np.count_nonzero(np.abs(updated-old)>1e-10),1)

    def test_contact_interface_sums_cancel_for_every_axial_zone(self):
        m=self.m
        tf=np.full((m.nzone,m.solid.transverse.n_fuel),490.)
        tc=np.full((m.nzone,19),450.)
        th=np.full((m.nzone,19),395.)
        sf,cin,cout=coating_interface_power(
            m.coating,m.solid,m.properties,tf,tc,th)
        np.testing.assert_allclose(sf.sum(axis=1),cin.sum(axis=1),
                                   rtol=1e-13,atol=1e-11)
        self.assertGreater(float(cin.sum()),0.)
        self.assertGreater(float(cout.sum()),0.)

    def test_zero_coating_film_insulates_hydrogen_but_not_fuel_contact(self):
        m=coating_model(film=0.)
        tf=np.full((m.nzone,m.solid.transverse.n_fuel),480.)
        tc=np.full((m.nzone,19),430.)
        th=np.full((m.nzone,19),380.)
        _,cin,cout=coating_interface_power(
            m.coating,m.solid,m.properties,tf,tc,th)
        self.assertGreater(float(cin.sum()),0.)
        self.assertEqual(float(cout.sum()),0.)

    def test_nonphysical_masses_and_missing_state_rejected(self):
        m=self.m
        cp=m.coating.heat_capacity
        with self.assertRaisesRegex(ValueError,"positive density"):
            build_resolved_coating(
                m.config,m.solid,density_kg_m3=0.,heat_capacity=cp)
        s=initial(m)
        with self.assertRaisesRegex(ValueError,"coating temperature"):
            solve(m,s,dt=5e-6)
        bad=replace(s,coating_temperature_k=np.ones((1,19))*400.)
        with self.assertRaisesRegex(ValueError,"coating state"):
            solve(m,bad,dt=5e-6)

    def test_legacy_stage58_is_bitwise_identical_without_coating(self):
        m=coupled(film=1250.,nz=2,nx=20,ny=24)
        state=initial(m,tf=470.,th=400.)
        a=solve(m,state,dt=5e-6)
        b=solve(m,state,dt=5e-6,coating_deposition_w=0.)
        np.testing.assert_array_equal(
            a.state.fuel_temperature_k,b.state.fuel_temperature_k)
        np.testing.assert_array_equal(
            a.state.hydrogen.conserved,b.state.hydrogen.conserved)
        self.assertIsNone(a.state.coating_temperature_k)
        self.assertEqual(a.coating_storage_change_j,0.)
        self.assertEqual(a.coating_to_hydrogen_j,0.)

    def test_unmodeled_coating_deposition_fails_closed(self):
        m=coupled(film=1250.,nz=2,nx=20,ny=24)
        s=initial(m)
        with self.assertRaisesRegex(ValueError,"without resolved coating"):
            solve(m,s,coating_deposition_w=10.)
        bad=coating_initial(self.m)
        with self.assertRaisesRegex(ValueError,"coating nuclear deposition"):
            solve(self.m,bad,coating_deposition_w=-1.)
        with self.assertRaisesRegex(ValueError,"coating nuclear deposition"):
            solve(self.m,bad,coating_deposition_w=np.ones((2,18)))

    def test_source_outside_coating_table_domain_fails_closed(self):
        m=self.m
        s=coating_initial(m)
        with self.assertRaisesRegex(ValueError,"qualified temperature table"):
            solve(m,s,dt=1e-3,
                  coating_deposition_w=np.full((m.nzone,19),1e10))


if __name__=="__main__":
    unittest.main(verbosity=2)
