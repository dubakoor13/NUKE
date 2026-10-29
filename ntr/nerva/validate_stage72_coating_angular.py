"""Stage72 analytical angular conduction and conservative clipped-ring tests.

All density, k(T), cp(T) and nuclear watts are synthetic. This validates
internal ZrC physics only, NOT Stage58's 3-temperature angular boundary
source distribution or genuine OpenMC reactor heating.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .coating_angular_fvm import (
    build_angular_coating_mesh, advance_angular_coating_enthalpy,
)
from .resolved_coating_thermal import advance_coating_axial_enthalpy
from .validate_stage71_axial_coating import axial_model


CP=800.
K=20.
DENSITY=5000.
NT=16


def make_mesh(nz=2, angular_sectors=NT):
    base=axial_model(nz=nz,film=0.)
    mesh=build_angular_coating_mesh(
        base.config,base.coating,angular_sectors=angular_sectors,
        density_kg_m3=DENSITY,
    )
    cp=TemperatureTable([250.,1000.],[CP,CP])
    conductivity=TemperatureTable([250.,1000.],[K,K])
    return base,replace(base.coating,heat_capacity=cp),mesh,conductivity


def advance(model,coating,grid,temps,powers,dt):
    return advance_angular_coating_enthalpy(
        coating,grid,temps,conductivity_w_m_k=model,
        dt_s=dt,deposition_w=powers,
    )


class AngularCoatingHeatTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base,cls.coat,cls.grid,cls.k=make_mesh()

    def test_each_annular_sector_sums_to_stage70_exact_clipped_mass(self):
        mass=self.grid.sector_mass_kg
        self.assertEqual(mass.shape,(2,19,NT))
        np.testing.assert_allclose(
            mass.sum(axis=2),self.base.coating.mass_kg,
            rtol=2e-7,atol=1e-10)
        self.assertTrue(np.all(mass>=0))
        self.assertTrue(np.all(self.grid.sector_area_m2>=0))
        self.assertEqual(len(self.grid.edge_left),len(self.grid.edge_right))
        self.assertEqual(len(self.grid.edge_left),
                         len(self.grid.face_kind))

    def test_central_full_annulus_sectors_have_equal_geometric_area(self):
        area=self.grid.sector_area_m2[0]
        np.testing.assert_allclose(
            area,np.full(NT,area.mean()),rtol=1e-11,atol=1e-15)
        radius=self.base.config.coated_channel_radius_cm
        bore=self.base.config.channel_radius_cm
        annulus=math.pi*(radius*radius-bore*bore)*1e-4
        self.assertAlmostEqual(float(area.sum()),annulus,delta=1e-14)

    def test_peripheral_channel_is_geometrically_clipped_not_full_annulus(self):
        areas=self.grid.sector_area_m2
        self.assertTrue(np.all(areas.sum(axis=1)>0))
        full=areas[0].sum()
        self.assertTrue(np.any(areas.sum(axis=1)<full-1e-14))
        self.assertTrue(np.any(np.ptp(areas[1:,:],axis=1)>1e-14))

    def test_periodic_angular_graph_never_crosses_distinct_channel_id(self):
        ntheta=NT
        left=self.grid.edge_left
        right=self.grid.edge_right
        period=19*ntheta
        lc=(left%period)//ntheta
        rc=(right%period)//ntheta
        np.testing.assert_array_equal(lc,rc)
        is_angular=self.grid.face_kind==1
        la=left[is_angular]%ntheta
        ra=right[is_angular]%ntheta
        np.testing.assert_array_equal(ra,(la+1)%ntheta)
        self.assertGreater(int(np.count_nonzero(
            (la==ntheta-1)&(ra==0))),0)
        axial=self.grid.face_kind==0
        np.testing.assert_array_equal(
            right[axial]-left[axial],np.full(
                np.count_nonzero(axial),19*ntheta))

    def test_periodic_fourier_eigenmode_matches_analytic_implicit_decay(self):
        base,coating,geom,k=make_mesh(nz=1)
        theta=2*math.pi*np.arange(NT)/NT
        wavenum=2
        t=np.full((1,19,NT),450.)
        t[0,0,:]+=20*np.cos(wavenum*theta)
        q=np.zeros_like(t)
        dt=0.04
        r=advance(k,coating,geom,t,q,dt)
        sector_mass=geom.sector_mass_kg[0,0,0]
        g=1./(2*geom.face_resistance_left_geom[0]/K)
        eig=2*g*(1-math.cos(wavenum*2*math.pi/NT))
        amplitude=20/(1+dt*eig/(sector_mass*CP))
        expected=450+amplitude*np.cos(wavenum*theta)
        np.testing.assert_allclose(r.temperature_k[0,0,:],
                                   expected,atol=2e-8,rtol=0)
        self.assertLess(abs(r.total_first_law_residual_j),1e-8)

    def test_single_hot_angular_sector_spreads_around_same_sleeve_only(self):
        t=np.full((2,19,NT),450.)
        t[0,0,0]=620.
        q=np.zeros_like(t)
        r=advance(self.k,self.coat,self.grid,t,q,dt=0.05)
        self.assertLess(r.temperature_k[0,0,0],620.)
        self.assertGreater(r.temperature_k[0,0,1],450.)
        self.assertGreater(r.temperature_k[0,0,-1],450.)
        self.assertGreater(r.temperature_k[1,0,0],450.)
        np.testing.assert_allclose(
            r.temperature_k[:,1:,:],450.,atol=2e-10,rtol=0)
        self.assertEqual(r.deposited_energy_j,0.)
        self.assertLess(abs(r.total_first_law_residual_j),2e-8)

    def test_angular_uniform_mode_reduces_to_previous_axial_model(self):
        nz=2
        t_ax=np.ones((nz,19))*450.
        t_ax[0,0]=525.
        t_ax[1,0]=410.
        t=np.repeat(t_ax[:,:,None],NT,axis=2)
        power_ax=np.zeros((nz,19))
        power_ax[1,0]=8.
        ratio=self.grid.sector_mass_kg/self.grid.sector_mass_kg.sum(
            axis=2,keepdims=True)
        q=power_ax[:,:,None]*ratio
        dt=2.
        angular=advance(self.k,self.coat,self.grid,t,q,dt)
        old=advance_coating_axial_enthalpy(
            self.coat,t_ax,conductivity_w_m_k=self.k,
            dt_s=dt,net_w=power_ax)
        np.testing.assert_allclose(
            angular.temperature_k[:,:,0],old.temperature_k,
            atol=2e-6,rtol=0)
        for a in range(NT):
            np.testing.assert_allclose(
                angular.temperature_k[:,:,a],old.temperature_k,
                atol=2e-6,rtol=0)
        self.assertAlmostEqual(
            angular.enthalpy_storage_change_j,
            old.global_first_law_residual_j+dt*power_ax.sum(),
            delta=2e-7)

    def test_nonconstant_heat_capacity_and_k_still_conserve_energy(self):
        base=axial_model(nz=2)
        cp=TemperatureTable(
            [250.,400.,600.,800.,1000.],[700.,790.,880.,1010.,1150.])
        kval=TemperatureTable(
            [250.,400.,600.,800.,1000.],[11.,17.,24.,29.,36.])
        coating=replace(self.coat,heat_capacity=cp)
        t=np.full((2,19,NT),445.)
        t[0,3,4]=610.
        power=np.zeros_like(t)
        power[1,3,4]=3.
        r=advance(kval,coating,self.grid,t,power,dt=0.03)
        self.assertLess(r.maximum_node_residual_j,2e-8)
        self.assertLess(abs(r.total_first_law_residual_j),2e-8)
        self.assertAlmostEqual(
            r.enthalpy_storage_change_j,
            r.deposited_energy_j,delta=2e-8)

    def test_independent_azimuthal_heat_pulse_has_near_zero_net_face_energy(self):
        t=np.full((2,19,NT),440.)
        t[0,4,6]=520.
        q=np.zeros_like(t)
        r=advance(self.k,self.coat,self.grid,t,q,dt=0.1)
        self.assertLess(abs(r.internal_flux_cancellation_j),1e-10)
        self.assertTrue(np.any(np.abs(r.face_transfer_j[
            r.face_kind==1])>0))
        self.assertTrue(np.any(np.abs(r.face_transfer_j[
            r.face_kind==0])>0))
        self.assertLess(abs(r.total_first_law_residual_j),2e-8)

    def test_zero_source_uniform_mode_exactly_stationary(self):
        t=np.full((2,19,NT),475.)
        r=advance(self.k,self.coat,self.grid,t,
                  np.zeros_like(t),dt=2.)
        np.testing.assert_array_equal(r.temperature_k,t)
        self.assertAlmostEqual(r.enthalpy_storage_change_j,0.,delta=1e-11)
        self.assertAlmostEqual(r.total_first_law_residual_j,0.,delta=1e-11)

    def test_nonphysical_property_or_shape_or_time_is_rejected(self):
        t=np.full((2,19,NT),470.)
        q=np.zeros_like(t)
        with self.assertRaisesRegex(ValueError,"angular coating state/watts"):
            advance(self.k,self.coat,self.grid,t,q[:,:,:-1],dt=0.1)
        with self.assertRaisesRegex(ValueError,"time step"):
            advance(self.k,self.coat,self.grid,t,q,dt=0.)
        with self.assertRaises(ValueError):
            advance(self.k,self.coat,self.grid,np.full_like(t,100.),
                    q,dt=.1)
        with self.assertRaisesRegex(ValueError,"density"):
            build_angular_coating_mesh(
                self.base.config,self.base.coating,
                density_kg_m3=0.)
        with self.assertRaisesRegex(ValueError,"sectors"):
            build_angular_coating_mesh(
                self.base.config,self.base.coating,
                angular_sectors=1,density_kg_m3=DENSITY)

    def test_internal_operator_does_not_claim_fuel_or_hydrogen_boundary_flux(self):
        # Stage72 is an isolated conduction component pending physical
        # angular arc-to-fuel cutcell and inner bore-to-fluid projections.
        t=np.full((2,19,NT),470.)
        t[:,0,0]=600.
        q=np.zeros_like(t)
        r=advance(self.k,self.coat,self.grid,t,q,dt=.05)
        self.assertAlmostEqual(
            r.deposited_energy_j,0.,delta=1e-12)
        self.assertAlmostEqual(
            r.enthalpy_storage_change_j,0.,delta=2e-8)
        self.assertNotEqual(
            r.temperature_k[0,0,0],t[0,0,0])


if __name__=="__main__":
    unittest.main(verbosity=2)
