"""Stage75: physical fuel/outer-ZrC-inner-ZrC/H2 first-law regressions.

This suite uses SYNTHETIC heat sources and material tables; it verifies
radial boundary attachment, no duplicated wall energy, per-channel scope
and conservation. It does not claim measured NERVA heat or OpenMC runs.
"""
from __future__ import annotations
from dataclasses import replace
import unittest

import numpy as np

from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .fuel_hydrogen_coupled_transient import with_resolved_coating
from .coating_radial_interfaces import radial_interface_power
from .coating_radial_fvm import (
    distribute_angular_power_to_radial_layers,
)
from .coating_angular_interfaces import distribute_channel_coating_power
from .validate_stage58_fuel_hydrogen_coupling import coupled,initial,solve
from .validate_stage73_angular_interfaces import (
    integrated_angular_model,integrated_angular_initial,
)


def radial_model(*,nz=2,na=8,nr=2,film=1200.):
    base=coupled(film=film,nz=nz,nx=20,ny=24)
    cp=TemperatureTable(
        [240.,300.,500.,800.,1100.],
        [690.,725.,800.,890.,1000.])
    return with_resolved_coating(
        base,density_kg_m3=5000.,heat_capacity=cp,
        contact_resistance_m2_k_w=1e-5,
        axial_conduction_enabled=True,angular_sectors=na,
        radial_layers=nr,
    )


def radial_initial(model,*,tf=450.,tc=430.,th=400.):
    base=initial(model,tf=tf,th=th)
    return replace(base,coating_temperature_k=np.full(
        model.radial_coating.sector_mass_kg.shape,tc))


def boundaries(model,state):
    return radial_interface_power(
        model.coating,model.solid,model.properties,
        model.angular_contacts,model.radial_coating,
        state.fuel_temperature_k,state.coating_temperature_k,
        np.full((model.nzone,19),400.))


class RadialCoatingCoupledConservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=radial_model()

    def test_outer_contact_enters_outer_shell_and_inner_bore_only_inner_shell(self):
        m=self.model
        old=radial_initial(m,tf=510.,tc=455.,th=390.)
        p=boundaries(m,old)
        self.assertEqual(
            p.fuel_to_coating_by_radial_w.shape,
            m.radial_coating.sector_mass_kg.shape)
        self.assertTrue(np.all(p.fuel_to_coating_by_radial_w[:,:,:,:-1]==0))
        self.assertTrue(np.all(p.coating_to_hydrogen_by_radial_w[:,:,:,1:]==0))
        self.assertGreater(float(p.fuel_to_coating_by_radial_w.sum()),0.)
        self.assertGreater(float(p.coating_to_hydrogen_by_radial_w.sum()),0.)
        self.assertLess(abs(p.fuel_antisymmetric_residual_w),1e-8)
        self.assertLess(abs(p.hydrogen_antisymmetric_residual_w),1e-8)
        np.testing.assert_allclose(
            p.coating_to_hydrogen_by_radial_w.sum(axis=(2,3)),
            p.coating_to_hydrogen_by_channel_w,atol=2e-10,rtol=0)

    def test_hot_fuel_coating_and_hydrogen_source_close_three_ledgers(self):
        m=self.model
        s=radial_initial(m,tf=505.,tc=455.,th=395.)
        fsrc=np.full_like(s.fuel_temperature_k,3.)
        csrc=np.full((m.nzone,19),2.)
        hsrc=np.full((m.nzone,19),1.)
        dt=5e-6
        result=solve(
            m,s,dt=dt,q=fsrc,
            coating_deposition_w=csrc,hydrogen_deposition_w=hsrc)
        self.assertEqual(
            result.state.coating_temperature_k.shape,
            m.radial_coating.sector_mass_kg.shape)
        self.assertEqual(len(result.coating_radial_face_transfer_j),
                         len(m.radial_coating.face_left))
        self.assertGreater(result.fuel_to_coating_j,0.)
        self.assertGreater(result.coating_to_hydrogen_j,0.)
        self.assertLess(result.relative_first_law_residual,4e-8)
        self.assertLess(abs(result.coating_first_law_residual_j),5e-7)
        self.assertAlmostEqual(result.coating_deposition_j,
                               csrc.sum()*dt,delta=2e-13)
        self.assertAlmostEqual(
            result.fuel_storage_change_j+result.fuel_to_coating_j,
            result.fuel_deposition_j,delta=5e-7)
        self.assertAlmostEqual(
            result.coating_storage_change_j+
            result.coating_to_hydrogen_j-result.fuel_to_coating_j,
            result.coating_deposition_j,delta=5e-7)
        self.assertAlmostEqual(
            result.hydrogen_total_energy_change_j+
            result.hydrogen_net_outlet_energy_j-
            result.coating_to_hydrogen_j,
            result.hydrogen_deposition_j,delta=2e-7)

    def test_hot_inner_shell_to_hydrogen_and_outer_to_fuel_are_distinct(self):
        m=self.model
        s=radial_initial(m,tf=400.,tc=400.,th=400.)
        hot=s.coating_temperature_k.copy()
        hot[0,5,3,0]=540.
        hot[1,4,2,-1]=510.
        s=replace(s,coating_temperature_k=hot)
        p=boundaries(m,s)
        self.assertGreater(p.coating_to_hydrogen_by_channel_w[0,5],0.)
        self.assertLess(p.fuel_to_coating_by_fuel_w.sum(),0.)
        self.assertEqual(float(np.sum(
            p.coating_to_hydrogen_by_radial_w[:,:,:,1:])),0.)
        self.assertEqual(float(np.sum(
            p.fuel_to_coating_by_radial_w[:,:,:,:-1])),0.)

    def test_insulated_hydrogen_heats_only_coating_and_fuel(self):
        m=radial_model(film=0.)
        s=radial_initial(m,tf=480.,tc=430.,th=400.)
        p=boundaries(m,s)
        self.assertEqual(float(p.coating_to_hydrogen_by_channel_w.sum()),0.)
        result=solve(m,s,dt=5e-6)
        self.assertEqual(float(result.coating_to_hydrogen_j),0.)
        self.assertLess(abs(result.coating_first_law_residual_j),4e-7)
        self.assertLess(result.relative_first_law_residual,4e-8)

    def test_one_radial_coating_source_bin_is_not_duplicated(self):
        m=radial_model(film=0.)
        s=radial_initial(m,tf=400.,tc=400.,th=400.)
        q=np.zeros(m.radial_coating.sector_mass_kg.shape)
        q[0,8,3,1]=40.
        r=solve(m,s,dt=5e-6,coating_deposition_w=q)
        self.assertAlmostEqual(r.coating_deposition_j,2e-4,delta=2e-13)
        self.assertGreater(r.state.coating_temperature_k[0,8,3,1],400.)
        self.assertEqual(float(r.coating_to_hydrogen_j),0.)
        self.assertLess(r.relative_first_law_residual,4e-8)

    def test_2d_angular_and_radial_explicit_sources_are_equivalent(self):
        m=self.model
        s=radial_initial(m)
        channel=np.zeros((m.nzone,19))
        channel[0,3]=55.
        angular=distribute_channel_coating_power(channel,m.angular_coating)
        radial=distribute_angular_power_to_radial_layers(
            angular,m.radial_coating)
        a=solve(m,s,dt=5e-6,coating_deposition_w=channel)
        b=solve(m,s,dt=5e-6,coating_deposition_w=angular)
        c=solve(m,s,dt=5e-6,coating_deposition_w=radial)
        np.testing.assert_allclose(
            a.state.coating_temperature_k,b.state.coating_temperature_k,
            atol=2e-9,rtol=0)
        np.testing.assert_allclose(
            b.state.coating_temperature_k,c.state.coating_temperature_k,
            atol=2e-9,rtol=0)
        self.assertAlmostEqual(a.coating_deposition_j,55*5e-6,delta=2e-13)
        self.assertAlmostEqual(a.coating_deposition_j,c.coating_deposition_j,
                               delta=1e-13)

    def test_zero_temperature_gradient_and_zero_heat_is_stationary(self):
        m=self.model
        s=radial_initial(m,tf=400.,tc=400.,th=400.)
        r=solve(m,s,dt=5e-6)
        np.testing.assert_allclose(
            r.state.coating_temperature_k,s.coating_temperature_k,
            atol=2e-9,rtol=0)
        self.assertEqual(r.coating_deposition_j,0.)
        self.assertLess(abs(r.coating_first_law_residual_j),6e-8)

    def test_invalid_radial_state_source_and_model_requirements(self):
        m=self.model
        old=radial_initial(m)
        with self.assertRaisesRegex(ValueError,"coating state"):
            solve(m,replace(old,coating_temperature_k=
                  old.coating_temperature_k[:,:,:,:-1]),dt=5e-6)
        with self.assertRaisesRegex(ValueError,"coating nuclear deposition"):
            solve(m,old,coating_deposition_w=
                  np.ones((m.nzone,19,8,3)))
        with self.assertRaisesRegex(ValueError,"resolved radial coating"):
            with_resolved_coating(
                coupled(nz=2,nx=20,ny=24),
                density_kg_m3=5000.,heat_capacity=m.coating.heat_capacity,
                radial_layers=2,axial_conduction_enabled=True)

    def test_stage73_no_radial_resolution_remains_independent(self):
        old_model=integrated_angular_model(nz=2,na=8)
        old_state=integrated_angular_initial(old_model)
        self.assertIsNone(old_model.radial_coating)
        r=solve(old_model,old_state,dt=5e-6)
        self.assertIsNone(r.coating_radial_face_transfer_j)
        self.assertLess(r.relative_first_law_residual,4e-8)


if __name__=="__main__":
    unittest.main(verbosity=2)
