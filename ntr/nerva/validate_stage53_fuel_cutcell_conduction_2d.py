"""Stage53 geometrically resolved 2D fuel cut-cell numerical acceptance tests.

All temperatures, conductivities, heat sources and hydrogen transport inputs
here are synthetic numerical verification parameters, not engine performance.
"""
from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.sparse.csgraph import connected_components

from .config import NervaConfig
from .fuel_cutcell_conduction_2d import (
    advance_cutcell_fuel_hydrogen,
    build_cutcell_fuel_2d,
)
from .fuel_sector_geometry import fuel_sector_fuel_areas_cm2


def model(*, nx=32, ny=40, film=0., k=12.):
    return build_cutcell_fuel_2d(
        NervaConfig(), nx=nx, ny=ny,
        conductivity_w_m_k=k,
        density_kg_m3=1000.,
        heat_capacity_j_kg_k=1000.,
        zrc_conductivity_w_m_k=20.,
        hydrogen_film_w_m2_k=film)


def evolve(op, *, f=None,h=None,qf=None,qh=None,dt=1., flow=0.,inlet=300.):
    n=op.n_fuel
    return advance_cutcell_fuel_hydrogen(
        op, np.full(n,300.) if f is None else f,
        np.full(19,300.) if h is None else h,
        dt_s=dt,
        fuel_source_w=np.zeros(n) if qf is None else qf,
        hydrogen_source_w=np.zeros(19) if qh is None else qh,
        hydrogen_heat_capacity_j_k=50.,
        hydrogen_flow_heat_capacity_w_k=flow,
        hydrogen_inlet_temperature_k=inlet)


class ExactEmbeddedGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=NervaConfig()
        cls.op=model(nx=32,ny=40)

    def test_true_fuel_area_volume_mass(self):
        op=self.op
        reference=float(np.sum(fuel_sector_fuel_areas_cm2(self.c)))
        self.assertEqual(op.cell_ids.shape,(40,32))
        self.assertGreater(op.n_fuel,250)
        self.assertAlmostEqual(float(op.fuel_areas_cm2.sum()),reference,delta=4e-11)
        self.assertAlmostEqual(float(op.fuel_volumes_m3.sum()),
                               0.000270729921870941,delta=5e-15)
        self.assertAlmostEqual(float(op.fuel_heat_capacity_j_k.sum()),
                               270.729921870941,delta=1e-8)

    def test_19_exact_channel_interfaces_include_six_clipped(self):
        op=self.op
        c=self.c
        self.assertEqual(op.fuel_zrc_contact_m2.shape,(op.n_fuel,19))
        full=2*math.pi*c.coated_channel_radius_cm*.01*op.axial_length_m
        self.assertEqual(int(np.count_nonzero(op.channel_fuel_interface_m2 <
                                               full-1e-11)),6)
        self.assertTrue(np.all(op.channel_fuel_interface_m2>0))
        self.assertLess(float(op.channel_fuel_interface_m2.max())-full,1e-12)
        self.assertGreater(float(op.fuel_zrc_contact_m2.nnz),19)

    def test_lateral_heat_matrix_is_symmetric_and_neutral_to_uniform_T(self):
        op=self.op
        L=op.lateral_laplacian_w_k
        self.assertEqual(L.shape,(op.n_fuel,op.n_fuel))
        self.assertGreater(len(op.lateral_face_pairs),op.n_fuel//2)
        np.testing.assert_allclose((L-L.T).data,0.,atol=1e-11,rtol=0.)
        np.testing.assert_allclose(L@np.ones(op.n_fuel),0.,atol=2e-10,rtol=0.)
        self.assertTrue(np.all(op.lateral_face_conductance_w_k>0))
        self.assertEqual(connected_components(L,directed=False)[0],1)

    def test_grid_refinement_preserves_exact_true_fuel_area_and_wall_arc(self):
        fine=model(nx=48,ny=56)
        coarse=self.op
        self.assertGreater(fine.n_fuel,coarse.n_fuel)
        self.assertAlmostEqual(float(fine.fuel_volumes_m3.sum()),
                               float(coarse.fuel_volumes_m3.sum()),delta=4e-14)
        np.testing.assert_allclose(fine.channel_fuel_interface_m2,
                                   coarse.channel_fuel_interface_m2,
                                   atol=2e-11,rtol=0.)


class True2DImplicitThermalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.op=model(nx=32,ny=40,film=2000.)

    def test_uniform_one_k_analytic_uniform_synthetic_volumetric_source(self):
        op=model(nx=32,ny=40,film=0.)
        # C/V = 1e6 J/(m3 K), Q/V = 1e6 W/m3, dt = 1 s => +1 K.
        r=evolve(op,qf=1e6*op.fuel_volumes_m3)
        np.testing.assert_allclose(r.fuel_temperature_k,301.,
                                   atol=5e-10,rtol=0.)
        np.testing.assert_allclose(r.hydrogen_temperature_k,300.,
                                   atol=1e-10,rtol=0.)
        self.assertLess(abs(r.first_law_residual_j),1e-7)

    def test_spatial_heat_diffusion_between_actual_cut_cells(self):
        op=model(nx=32,ny=40,film=0.)
        q=np.zeros(op.n_fuel)
        q[np.argmin(np.sum(op.cell_centers_cm**2,axis=1))]=150.
        r=evolve(op,qf=q,dt=.2)
        self.assertGreater(r.fuel_temperature_k.max(),300.)
        self.assertGreater(np.count_nonzero(r.fuel_temperature_k>300.+1e-8),3)
        self.assertAlmostEqual(r.fuel_storage_j,30.,delta=1e-6)
        self.assertLess(abs(r.first_law_residual_j),1e-6)

    def test_uniform_hot_fuel_cools_hydrogen_and_preserves_global_first_law(self):
        op=self.op
        r=evolve(op,f=np.full(op.n_fuel,420.),dt=.15)
        self.assertLess(r.fuel_temperature_k.min(),420.)
        self.assertGreater(r.hydrogen_temperature_k.max(),300.)
        self.assertGreater(r.wall_exchange_j,0.)
        self.assertAlmostEqual(r.fuel_storage_j+r.hydrogen_storage_j,
                               0.,delta=2e-6)

    def test_hydrogen_outlet_enthalpy_is_counted_in_balance(self):
        op=self.op
        r=evolve(op,f=np.full(op.n_fuel,480.),
                 flow=4.,inlet=300.,dt=.7)
        self.assertGreater(r.coolant_export_j,0.)
        self.assertLess(r.fuel_temperature_k.mean(),480.)
        self.assertLess(abs(r.first_law_residual_j),1e-5)

    def test_unforced_adiabatic_uniform_mode(self):
        op=self.op
        r=evolve(op,f=np.full(op.n_fuel,365.),h=np.full(19,365.),dt=10.)
        np.testing.assert_allclose(r.fuel_temperature_k,365.,
                                   atol=2e-9,rtol=0.)
        np.testing.assert_allclose(r.hydrogen_temperature_k,365.,
                                   atol=2e-9,rtol=0.)

    def test_large_implicit_timestep_positive_and_energy_conservative(self):
        op=self.op
        r=evolve(op,f=np.full(op.n_fuel,520.),
                 flow=0.,h=np.full(19,290.),dt=50.)
        self.assertGreater(r.fuel_temperature_k.min(),289.9999)
        self.assertLess(r.fuel_temperature_k.max(),520.0001)
        self.assertGreater(r.hydrogen_temperature_k.min(),289.9999)
        self.assertLess(r.hydrogen_temperature_k.max(),520.0001)
        self.assertLess(abs(r.first_law_residual_j),1e-5)

    def test_inputs_checked_and_unsupported_meshing_rejected(self):
        with self.assertRaises(ValueError):
            model(nx=12)
        with self.assertRaises(ValueError):
            model(nx=32,k=-5.)
        with self.assertRaises(ValueError):
            evolve(self.op,qf=np.zeros(19))
        with self.assertRaises(ValueError):
            evolve(self.op,dt=0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
