"""Stage74 clipped radial ZrC heat-transport manufactured/conservation cases.

All cp(T), k(T), density and imposed heat sources are synthetic. These
tests verify spatial conservation, not NERVA temperatures or OpenMC heat.
The Stage73 outer-fuel/inner-H2 boundary attachment is NOT enabled.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from .coating_radial_fvm import (
    build_radial_coating_mesh,advance_radial_coating_enthalpy,
    distribute_angular_power_to_radial_layers,
)
from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .validate_stage73_angular_interfaces import integrated_angular_model


DENSITY=5000.
CP=800.
K=20.


def fixture(*,nz=2,na=8,nrad=3):
    model=integrated_angular_model(nz=nz,na=na,film=0.)
    mesh=build_radial_coating_mesh(
        model.config,model.coating,model.angular_coating,
        radial_layers=nrad,density_kg_m3=DENSITY)
    cp=TemperatureTable([240.,1100.],[CP,CP])
    conductivity=TemperatureTable([240.,1100.],[K,K])
    return model,replace(model.coating,heat_capacity=cp),mesh,conductivity


def advance(model,coating,mesh,temperature,power,dt):
    return advance_radial_coating_enthalpy(
        coating,mesh,temperature,conductivity_w_m_k=model,
        dt_s=dt,deposition_w=power,
    )


class RadialCoatingFiniteVolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.coat,cls.mesh,cls.k=fixture()

    def test_true_hex_clipped_radial_cells_recover_stage70_mass(self):
        m=self.model
        grid=self.mesh
        self.assertEqual(grid.sector_mass_kg.shape,(2,19,8,3))
        np.testing.assert_allclose(
            grid.sector_mass_kg.sum(axis=(2,3)),
            m.coating.mass_kg,rtol=2e-7,atol=1e-10)
        np.testing.assert_allclose(
            grid.sector_mass_kg.sum(axis=3),
            m.angular_coating.sector_mass_kg,rtol=2e-7,atol=1e-10)
        self.assertAlmostEqual(
            grid.physical_coating_mass_kg,
            float(m.coating.mass_kg.sum()),delta=1e-8)
        self.assertTrue(np.all(grid.sector_volume_m3>=0))

    def test_middle_radial_cylinder_face_uses_exact_hex_clipped_arc(self):
        m,coat,grid,k=fixture(nz=1,na=16,nrad=2)
        radius=grid.radius_edges_m[1]
        # The central channel has no hex clipping, so all angular face
        # areas equal radius*delta-theta*the physical axial length.
        target=radius*(2*math.pi/16)*m.config.active_length_cm*.01
        np.testing.assert_allclose(
            grid.radial_area_full_m2[0,:,0],
            np.full(16,target),rtol=0,atol=1e-12)
        self.assertTrue(np.all(grid.radial_area_full_m2>=0))

    def test_each_internal_radial_face_is_same_channel_sector_only(self):
        grid=self.mesh
        nc,na,nrad=19,8,3
        edges=grid.face_kind==2
        left=grid.face_left[edges]
        right=grid.face_right[edges]
        self.assertGreater(len(left),0)
        np.testing.assert_array_equal(right-left,np.ones(len(left)))
        np.testing.assert_array_equal(
            (left//nrad)//na%nc,(right//nrad)//na%nc)
        np.testing.assert_array_equal(
            (left//nrad)%na,(right//nrad)%na)
        self.assertEqual(set(grid.face_kind.tolist()),{0,1,2})

    def test_two_radial_shell_constant_property_implicit_analytic(self):
        model,coat,grid,k=fixture(nz=1,na=8,nrad=2)
        temp=np.full(grid.sector_mass_kg.shape,440.)
        temp[0,0,:,0]=570.
        temp[0,0,:,1]=390.
        dt=20.
        r=advance(k,coat,grid,temp,np.zeros_like(temp),dt)
        index=np.where(grid.face_kind==2)[0][0]
        self.assertEqual(grid.face_left[index],0)
        self.assertEqual(grid.face_right[index],1)
        conductance=1./((grid.face_left_geom[index]+
                         grid.face_right_geom[index])/K)
        cap=grid.sector_mass_kg[0,0,0,:]*CP
        analytic=np.linalg.solve(
            [[cap[0]/dt+conductance,-conductance],
             [-conductance,cap[1]/dt+conductance]],
            cap*temp[0,0,0,:]/dt)
        np.testing.assert_allclose(r.temperature_k[0,0,0,:],
                                   analytic,rtol=0,atol=1e-8)
        np.testing.assert_allclose(r.temperature_k[0,0,:,:],
                                   np.broadcast_to(analytic,(8,2)),
                                   rtol=0,atol=1e-8)
        self.assertLess(abs(r.global_first_law_residual_j),1e-8)

    def test_single_radial_pulse_diffuses_inward_outward_and_axially(self):
        grid=self.mesh
        temp=np.full(grid.sector_mass_kg.shape,450.)
        temp[0,0,0,1]=600.
        r=advance(self.k,self.coat,grid,temp,np.zeros_like(temp),dt=1.)
        self.assertLess(r.temperature_k[0,0,0,1],600.)
        self.assertGreater(r.temperature_k[0,0,0,0],450.)
        self.assertGreater(r.temperature_k[0,0,0,2],450.)
        self.assertGreater(r.temperature_k[0,0,1,1],450.)
        self.assertGreater(r.temperature_k[1,0,0,1],450.)
        np.testing.assert_allclose(r.temperature_k[:,1:,:,:],450.,
                                   rtol=0,atol=2e-9)
        self.assertLess(abs(r.global_first_law_residual_j),2e-8)
        self.assertLess(abs(r.assembled_face_cancellation_j),1e-10)

    def test_zero_sources_uniform_temperature_is_exact_equilibrium(self):
        temp=np.full(self.mesh.sector_mass_kg.shape,450.)
        r=advance(self.k,self.coat,self.mesh,temp,
                  np.zeros_like(temp),dt=5.)
        np.testing.assert_array_equal(r.temperature_k,temp)
        self.assertEqual(r.total_deposited_energy_j,0.)
        self.assertEqual(r.global_first_law_residual_j,0.)

    def test_angular_source_to_radial_uniform_volume_conserves_power(self):
        rng=np.arange(2*19*8,dtype=float).reshape((2,19,8))
        angular_power=.01*(1+rng%17)
        q=distribute_angular_power_to_radial_layers(
            angular_power,self.mesh)
        np.testing.assert_allclose(q.sum(axis=-1),angular_power,
                                   atol=1e-13,rtol=0)
        self.assertEqual(q.shape,self.mesh.sector_mass_kg.shape)
        self.assertTrue(np.all(q>=0))

    def test_nonconstant_cp_and_conductivity_preserve_global_first_law(self):
        cp=TemperatureTable(
            [240.,300.,500.,800.,1100.],[690.,725.,800.,890.,1000.])
        k=TemperatureTable(
            [240.,300.,500.,800.,1100.],[22.,20.,18.,16.,15.])
        coat=replace(self.coat,heat_capacity=cp)
        temp=np.full(self.mesh.sector_mass_kg.shape,450.)
        temp[0,4,3,0]=535.
        power=np.zeros_like(temp)
        power[1,4,3,2]=5.
        r=advance(k,coat,self.mesh,temp,power,dt=.1)
        self.assertLess(abs(r.total_storage_change_j-
                            r.total_deposited_energy_j),2e-8)
        self.assertLess(r.maximum_node_residual_j,2e-8)
        self.assertLess(abs(r.assembled_face_cancellation_j),2e-10)

    def test_number_of_radial_layers_refinement_preserves_exact_total_mass(self):
        for nrad in (2,4):
            with self.subTest(radial_layers=nrad):
                m,coat,grid,k=fixture(nz=1,na=8,nrad=nrad)
                self.assertAlmostEqual(
                    grid.physical_coating_mass_kg,
                    float(m.coating.mass_kg.sum()),delta=2e-8)

    def test_nonphysical_time_sources_and_radial_edges_rejected(self):
        temp=np.full(self.mesh.sector_mass_kg.shape,450.)
        with self.assertRaisesRegex(ValueError,"time step"):
            advance(self.k,self.coat,self.mesh,temp,
                    np.zeros_like(temp),dt=0.)
        with self.assertRaisesRegex(ValueError,"state and power"):
            advance(self.k,self.coat,self.mesh,temp,
                    np.ones((2,19,8)),dt=1.)
        with self.assertRaises(ValueError):
            advance(self.k,self.coat,self.mesh,np.full_like(temp,200.),
                    np.zeros_like(temp),dt=1.)
        with self.assertRaisesRegex(ValueError,"radial coating edges"):
            build_radial_coating_mesh(
                self.model.config,self.model.coating,self.model.angular_coating,
                radial_layers=3,density_kg_m3=DENSITY,
                radial_edges_m=np.array([.01,.02,.03,.04]))
        with self.assertRaisesRegex(ValueError,"radial coating density"):
            build_radial_coating_mesh(
                self.model.config,self.model.coating,self.model.angular_coating,
                radial_layers=3,density_kg_m3=-10.)


if __name__=="__main__":
    unittest.main(verbosity=2)
