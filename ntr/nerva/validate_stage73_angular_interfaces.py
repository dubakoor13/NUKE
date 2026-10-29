"""Stage73 independent physical geometry and conservative two-sided wall flux tests.

The boundary arcs are split at Stage53 Cartesian cut-cell edges and at
Stage72 angular sector boundaries, then compared independently with
Stage53 total fuel contact and Stage70 bore area. Material laws, nuclear
heat rates and all temperatures here are synthetic verification inputs.
"""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from .coating_angular_fvm import build_angular_coating_mesh
from .coating_angular_interfaces import (
    build_angular_coating_interfaces,angular_interface_power,
    distribute_channel_coating_power,
)
from .validate_stage71_axial_coating import axial_model


def interface_fixture(*,nz=2,na=16,film=1200.):
    m=axial_model(nz=nz,film=film)
    grid=build_angular_coating_mesh(
        m.config,m.coating,angular_sectors=na,density_kg_m3=5000.)
    contact=build_angular_coating_interfaces(
        m.config,m.solid,m.coating,grid)
    return m,grid,contact


def flux_case(m,contact,tf=500.,tc=450.,th=400.):
    fuel=np.full((m.nzone,m.solid.transverse.n_fuel),tf)
    coat=np.full((m.nzone,19,contact.sector_count),tc)
    h2=np.full((m.nzone,19),th)
    return angular_interface_power(
        m.coating,m.solid,m.properties,contact,fuel,coat,h2)


class AngularCoatingContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.angular,cls.contact=interface_fixture()

    def test_exact_outer_arcs_reconstruct_every_stage53_fuel_cell_channel(self):
        m=self.model
        na=self.contact.sector_count
        nf=m.solid.transverse.n_fuel
        arc=self.contact.fuel_outer_full_m2.toarray().reshape((nf,19,na))
        np.testing.assert_allclose(
            arc.sum(axis=2),m.coating.fuel_arc_m2.toarray(),
            atol=4e-11,rtol=0)
        self.assertLess(self.contact.fuel_arc_residual_m2,4e-11)
        self.assertTrue(np.all(arc>=0))
        self.assertGreater(int(np.count_nonzero(arc)),19*na)

    def test_inner_coolant_arcs_reconstruct_stage70_bore_areas(self):
        inner=self.contact.h2_inner_full_m2
        self.assertEqual(inner.shape,(19,16))
        np.testing.assert_allclose(
            inner.sum(axis=1),
            self.model.coating.inner_bore_area_m2.sum(axis=0),
            atol=4e-11,rtol=0)
        self.assertLess(self.contact.hydrogen_arc_residual_m2,4e-11)

    def test_partitioned_arcs_are_zero_on_geometrically_absent_sleeves(self):
        inner=self.contact.h2_inner_full_m2
        self.assertTrue(np.all(inner[~self.angular.sector_active]<=1e-13))
        self.assertTrue(np.all(self.angular.sector_mass_kg>=0))

    def test_uniform_hot_fuel_flux_is_exactly_antisymmetric(self):
        result=flux_case(self.model,self.contact,tf=520.,tc=450.,th=390.)
        self.assertGreater(float(result.fuel_to_coating_by_fuel_w.sum()),0.)
        self.assertGreater(float(result.coating_to_hydrogen_by_channel_w.sum()),0.)
        self.assertAlmostEqual(
            float(result.fuel_to_coating_by_fuel_w.sum()),
            float(result.fuel_to_coating_by_sector_w.sum()),
            delta=3e-11)
        self.assertAlmostEqual(
            float(result.coating_to_hydrogen_by_sector_w.sum()),
            float(result.coating_to_hydrogen_by_channel_w.sum()),
            delta=3e-11)
        self.assertLess(abs(result.equal_opposite_fuel_interface_residual_w),2e-11)

    def test_uniform_coating_temperature_keeps_angular_sector_fluxes_distinct(self):
        m=self.model
        r=flux_case(m,self.contact,tf=510.,tc=460.,th=410.)
        p=r.fuel_to_coating_by_sector_w
        self.assertEqual(p.shape,(2,19,16))
        self.assertTrue(np.all(p>=0))
        # Peripheral ZrC interfaces are hex-clipped and not uniform around
        # a channel, even with all three material temperatures uniform.
        self.assertTrue(np.any(np.ptp(p[0,1:,:],axis=1)>1e-4))

    def test_one_hot_angular_temperature_changes_only_that_sleeve_boundary(self):
        m=self.model
        tf=np.full((m.nzone,m.solid.transverse.n_fuel),440.)
        tc=np.full((m.nzone,19,16),450.)
        th=np.full((m.nzone,19),440.)
        baseline=angular_interface_power(m.coating,m.solid,m.properties,
            self.contact,tf,tc,th)
        tc[0,4,5]=500.
        changed=angular_interface_power(m.coating,m.solid,m.properties,
            self.contact,tf,tc,th)
        df=changed.fuel_to_coating_by_sector_w-baseline.fuel_to_coating_by_sector_w
        dh=(changed.coating_to_hydrogen_by_sector_w-
            baseline.coating_to_hydrogen_by_sector_w)
        self.assertLess(df[0,4,5],0.)
        self.assertGreater(dh[0,4,5],0.)
        mask=np.ones(df.shape,dtype=bool)
        mask[0,4,5]=False
        np.testing.assert_allclose(df[mask],0.,atol=1e-13,rtol=0)
        np.testing.assert_allclose(dh[mask],0.,atol=1e-13,rtol=0)

    def test_zero_h2_film_blocks_coolant_but_not_fuel_contact(self):
        m,grid,contact=interface_fixture(film=0.)
        r=flux_case(m,contact,tf=510.,tc=450.,th=380.)
        self.assertGreater(float(r.fuel_to_coating_by_fuel_w.sum()),0.)
        self.assertEqual(float(r.coating_to_hydrogen_by_sector_w.sum()),0.)
        self.assertEqual(float(r.coating_to_hydrogen_by_channel_w.sum()),0.)

    def test_angular_source_partition_conserves_channel_and_total_watts(self):
        m=self.model
        p=np.zeros((m.nzone,19))
        p[0,5]=17.
        p[1,8]=80.
        result=distribute_channel_coating_power(p,self.angular)
        self.assertEqual(result.shape,(2,19,16))
        np.testing.assert_allclose(result.sum(axis=2),p,atol=1e-13,rtol=0)
        self.assertAlmostEqual(float(result.sum()),97.,delta=1e-13)
        self.assertTrue(np.all(result>=0))
        self.assertTrue(np.all(result[:,~np.any(
            self.angular.sector_active,axis=1),:]==0))

    def test_geometric_rebin_four_and_twenty_four_sectors_keep_total_area(self):
        a,b,_=interface_fixture(na=8)
        c,d,_=interface_fixture(na=24)
        np.testing.assert_allclose(
            b.sector_mass_kg.sum(axis=2),a.coating.mass_kg,
            atol=2e-9,rtol=2e-7)
        np.testing.assert_allclose(
            d.sector_mass_kg.sum(axis=2),c.coating.mass_kg,
            atol=2e-9,rtol=2e-7)
        self.assertEqual(self.contact.sector_count,16)

    def test_fail_closed_on_bad_interface_state_and_nuclear_source(self):
        m=self.model
        fuel=np.full((2,m.solid.transverse.n_fuel),450.)
        coating=np.full((2,19,16),450.)
        h2=np.full((2,19),450.)
        with self.assertRaisesRegex(ValueError,"interface shape"):
            angular_interface_power(
                m.coating,m.solid,m.properties,self.contact,
                fuel,coating[:,:,:-1],h2)
        with self.assertRaisesRegex(ValueError,"source"):
            distribute_channel_coating_power(
                np.full((2,19),-1.),self.angular)
        with self.assertRaisesRegex(ValueError,"source"):
            distribute_channel_coating_power(
                np.ones((2,18)),self.angular)


from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .fuel_hydrogen_coupled_transient import with_resolved_coating
from .validate_stage58_fuel_hydrogen_coupling import coupled,initial,solve


def integrated_angular_model(*,nz=2,na=8,film=1200.):
    base=coupled(film=film,nz=nz,nx=20,ny=24)
    cp=TemperatureTable(
        [240.,300.,500.,800.,1100.],
        [690.,725.,800.,890.,1000.])
    return with_resolved_coating(
        base,density_kg_m3=5000.,heat_capacity=cp,
        contact_resistance_m2_k_w=1e-5,
        axial_conduction_enabled=True,angular_sectors=na)


def integrated_angular_initial(m, *, tf=450.,tc=430.,th=400.):
    a=initial(m,tf=tf,th=th)
    return replace(a,coating_temperature_k=np.full(
        m.angular_coating.sector_mass_kg.shape,tc))


class AngularCoatingFullCoreEnergyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=integrated_angular_model()

    def test_three_temperature_angular_state_and_full_energy(self):
        m=self.m
        old=integrated_angular_initial(m)
        fuel=np.full_like(old.fuel_temperature_k,3.)
        coat=np.full((m.nzone,19),2.)
        gas=np.full((m.nzone,19),1.)
        r=solve(m,old,dt=5e-6,q=fuel,
                coating_deposition_w=coat,hydrogen_deposition_w=gas)
        self.assertEqual(
            r.state.coating_temperature_k.shape,
            m.angular_coating.sector_mass_kg.shape)
        self.assertEqual(len(r.coating_angular_face_transfer_j),
                         len(m.angular_coating.edge_left))
        self.assertGreater(r.fuel_to_coating_j,0.)
        self.assertGreater(r.coating_to_hydrogen_j,0.)
        self.assertLess(r.relative_first_law_residual,3e-8)
        self.assertLess(abs(r.coating_first_law_residual_j),3e-7)
        self.assertAlmostEqual(
            r.coating_deposition_j,float(coat.sum())*5e-6,delta=1e-13)
        self.assertAlmostEqual(
            r.fuel_storage_change_j+r.fuel_to_coating_j,
            r.fuel_deposition_j,delta=4e-7)
        self.assertAlmostEqual(
            r.coating_storage_change_j+r.coating_to_hydrogen_j-
            r.fuel_to_coating_j,r.coating_deposition_j,delta=4e-7)

    def test_one_sector_coating_heating_uses_only_one_angular_source_bin(self):
        m=integrated_angular_model(film=0.)
        initial_state=integrated_angular_initial(
            m,tf=400.,tc=400.,th=400.)
        p=np.zeros(m.angular_coating.sector_mass_kg.shape)
        p[0,0,0]=20.
        r=solve(m,initial_state,dt=5e-6,
                coating_deposition_w=p)
        self.assertAlmostEqual(r.coating_deposition_j,1e-4,delta=1e-14)
        self.assertGreater(r.state.coating_temperature_k[0,0,0],400.)
        self.assertEqual(float(r.coating_to_hydrogen_j),0.)
        self.assertLess(abs(r.coating_first_law_residual_j),5e-7)
        self.assertLess(r.relative_first_law_residual,3e-8)

    def test_zero_source_equilibrium_keeps_all_three_states(self):
        m=self.m
        state=integrated_angular_initial(m,tf=400.,tc=400.,th=400.)
        r=solve(m,state,dt=5e-6)
        np.testing.assert_allclose(
            r.state.coating_temperature_k,state.coating_temperature_k,
            atol=1e-9,rtol=0)
        self.assertEqual(r.coating_deposition_j,0.)
        self.assertLess(abs(r.first_law_residual_j),4e-8)
        self.assertLess(abs(r.coating_first_law_residual_j),4e-8)

    def test_angular_source_per_channel_uniform_volumetric_projection(self):
        m=self.m
        state=integrated_angular_initial(m)
        w=np.zeros((m.nzone,19))
        w[0,2]=50.
        a=solve(m,state,dt=5e-6,coating_deposition_w=w)
        q=distribute_channel_coating_power(w,m.angular_coating)
        b=solve(m,state,dt=5e-6,coating_deposition_w=q)
        np.testing.assert_allclose(
            a.state.coating_temperature_k,b.state.coating_temperature_k,
            atol=3e-11,rtol=0)
        np.testing.assert_allclose(
            a.state.fuel_temperature_k,b.state.fuel_temperature_k,
            atol=3e-11,rtol=0)
        self.assertAlmostEqual(a.coating_deposition_j,b.coating_deposition_j,
                               delta=1e-13)

    def test_incompatible_angular_dimensions_and_property_scope_fail_closed(self):
        m=self.m
        base=initial(m)
        with self.assertRaisesRegex(ValueError,"coating state"):
            solve(m,base,dt=5e-6)
        bad=replace(base,coating_temperature_k=np.ones((2,19))*400.)
        with self.assertRaisesRegex(ValueError,"coating state"):
            solve(m,bad,dt=5e-6)
        state=integrated_angular_initial(m)
        with self.assertRaisesRegex(ValueError,"coating nuclear deposition"):
            solve(m,state,coating_deposition_w=np.ones((2,19,7)))
        with self.assertRaisesRegex(ValueError,"angular coating transport"):
            with_resolved_coating(
                coupled(nz=2,nx=20,ny=24),density_kg_m3=5000.,
                heat_capacity=m.coating.heat_capacity,
                angular_sectors=8,axial_conduction_enabled=False)


if __name__=="__main__":
    unittest.main(verbosity=2)
