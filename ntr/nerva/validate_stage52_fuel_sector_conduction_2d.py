"""Regression tests for conservative planar NERVA fuel-sector heat transport.

These are geometry, matrix, manufactured-solution, and synthetic-energy
tests, NOT reactor-power validation. No design or neutronic inputs change.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np
from scipy.sparse.csgraph import connected_components

from .config import NervaConfig
from .fuel_sector_conduction_2d import (
    advance_fuel_sector_2d,
    build_fuel_sector_conduction_2d,
)
from .fuel_sector_geometry import fuel_sector_fuel_areas_cm2
from .layout import coolant_channel_positions


def make_operator(*, film: float = 0.0, conductivity: object = 12.0):
    # All thermophysical numbers are SYNTHETIC solver verification inputs.
    return build_fuel_sector_conduction_2d(
        NervaConfig(),
        conductivity_w_m_k=conductivity,
        density_kg_m3=1800.0,
        heat_capacity_j_kg_k=1000.0,
        zrc_conductivity_w_m_k=20.0,
        hydrogen_film_w_m2_k=film,
    )


def step(op, *, dt=1., fuel=None, hydrogen=None, qf=None,
         qh=None, flow=0., inlet=300., h2_capacity=80.):
    return advance_fuel_sector_2d(
        op, np.full(19, 300.) if fuel is None else fuel,
        np.full(19, 300.) if hydrogen is None else hydrogen,
        dt_s=dt,
        fuel_heating_w=np.zeros(19) if qf is None else qf,
        hydrogen_heating_w=np.zeros(19) if qh is None else qh,
        hydrogen_heat_capacity_j_k=h2_capacity,
        hydrogen_flow_heat_capacity_w_k=flow,
        inlet_hydrogen_temperature_k=inlet,
    )


class ActualPlanarFuelGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = NervaConfig()
        cls.op = make_operator()

    def test_true_19_sector_solid_mass_exactly_preserved(self):
        op=self.op
        self.assertEqual(op.fuel_area_cm2.shape,(19,))
        np.testing.assert_allclose(op.fuel_area_cm2,
                                   fuel_sector_fuel_areas_cm2(self.c),rtol=0,atol=0)
        self.assertAlmostEqual(op.fuel_volume_m3.sum(),
                               0.000270729921870941,delta=3e-16)
        np.testing.assert_allclose(op.fuel_heat_capacity_j_k,
                                   1800e3*op.fuel_volume_m3,rtol=2e-14)

    def test_lateral_2d_face_graph_is_connected_and_symmetric(self):
        op=self.op
        self.assertGreaterEqual(len(op.face_pairs),30)
        self.assertTrue(np.all(op.face_conductance_w_k>0))
        self.assertTrue(np.all(op.face_lengths_m>0))
        for i,j in op.face_pairs:
            self.assertLess(i,j)
        lap=op.laplacian_w_k
        self.assertAlmostEqual(float(np.max(np.abs(lap.toarray()-lap.toarray().T))),
                               0.0,delta=1e-11)
        np.testing.assert_allclose(lap@np.ones(19),0,rtol=0,atol=1e-11)
        self.assertEqual(connected_components(lap, directed=False)[0],1)
        eigen=np.linalg.eigvalsh(lap.toarray())
        self.assertGreater(eigen[1],1e-8)
        self.assertGreaterEqual(eigen[0],-1e-11)

    def test_internal_face_area_and_flux_formula(self):
        op=self.op
        sites=coolant_channel_positions(self.c)
        for (i,j), length, g in zip(op.face_pairs,op.face_lengths_m,
                                    op.face_conductance_w_k):
            dist=math.dist(sites[i],sites[j])*1e-2
            self.assertAlmostEqual(g,12.*length*1.32/dist,
                                   delta=1e-11*max(1.,g))

    def test_exact_channel_arc_and_outer_zrc_wall_area(self):
        op=self.op
        c=self.c
        self.assertEqual(op.fuel_zrc_contact_area_m2.shape,(19,))
        self.assertEqual(op.h2_bore_area_m2.shape,(19,))
        self.assertTrue(np.all(op.fuel_zrc_contact_area_m2>0))
        full=2*math.pi*c.coated_channel_radius_cm*1e-2*1.32
        self.assertEqual(int(np.count_nonzero(op.fuel_zrc_contact_area_m2 <
                                               full-1e-12)),6)
        np.testing.assert_allclose(op.h2_bore_area_m2,
                                   2*math.pi*c.channel_radius_cm*1e-2*1.32,
                                   rtol=0,atol=1e-15)
        # Numerically independent angular quadrature of actual fuel-to-ZrC
        # circle arcs (checks the analytic clipping geometry in the solver).
        theta=(np.arange(16384,dtype=float)+.5)*2*math.pi/16384
        xdir=np.cos(theta)
        ydir=np.sin(theta)
        ap=.5*c.inner_fuel_flat_to_flat_cm
        t=math.sqrt(3)/2
        approx=[]
        for x,y in coolant_channel_positions(c):
            xx=x+c.coated_channel_radius_cm*xdir
            yy=y+c.coated_channel_radius_cm*ydir
            inside=(np.abs(xx)<ap)&(np.abs(.5*xx+t*yy)<ap)&(np.abs(.5*xx-t*yy)<ap)
            approx.append(full*np.mean(inside))
        np.testing.assert_allclose(op.fuel_zrc_contact_area_m2,
                                   np.asarray(approx),rtol=0,atol=3e-7)
        self.assertGreater(op.exterior_zrc_contact_area_m2,0.)
        self.assertLess(op.exterior_zrc_contact_area_m2,
                        6*c.inner_fuel_edge_length_cm*1e-2*1.32)

    def test_adiabatic_outer_wall_and_uniform_temperature_nullspace(self):
        result=step(self.op, fuel=np.full(19,472.), hydrogen=np.full(19,301.))
        np.testing.assert_allclose(result.fuel_temperature_k,472.,rtol=0,atol=2e-12)
        np.testing.assert_allclose(result.hydrogen_temperature_k,301.,
                                   rtol=0,atol=2e-12)
        self.assertLess(abs(result.conservation_residual_j),1e-8)

    def test_inhomogeneous_conductivity_harmonic_faces(self):
        k=np.linspace(7.0,31.0,19)
        op=make_operator(conductivity=k)
        sites=coolant_channel_positions(self.c)
        for (i,j), length,g in zip(op.face_pairs,op.face_lengths_m,
                                    op.face_conductance_w_k):
            harmonic=2*k[i]*k[j]/(k[i]+k[j])
            expected=harmonic*length*1.32/(math.dist(sites[i],sites[j])*1e-2)
            self.assertAlmostEqual(g,expected,delta=1e-11*max(1.,g))


class ImplicitConductionConservationTests(unittest.TestCase):
    def test_uniform_volumetric_source_analytic_one_step(self):
        op=make_operator(film=0)
        source=1e6*op.fuel_volume_m3  # SYNTHETIC 1 MW/m3, not real NERVA watts
        r=step(op,qf=source,dt=1.)
        np.testing.assert_allclose(r.fuel_temperature_k,
                                   300+1e6/1.8e6,rtol=0,atol=2e-11)
        np.testing.assert_allclose(r.hydrogen_temperature_k,300,
                                   rtol=0,atol=2e-11)
        self.assertLess(abs(r.conservation_residual_j),1e-8)

    def test_lateral_nonuniform_source_spreads_heat_and_preserves_energy(self):
        op=make_operator(film=0)
        q=np.zeros(19)
        q[0]=200.  # central channel sector by radius-sorted axial indices
        r=step(op,dt=.5,qf=q)
        self.assertGreater(r.fuel_temperature_k[0],300)
        self.assertGreater(np.count_nonzero(r.fuel_temperature_k>300+1e-6),1)
        self.assertAlmostEqual(r.fuel_storage_j,100.,delta=2e-8)
        self.assertLess(abs(r.conservation_residual_j),2e-8)

    def test_coupled_wall_transfer_is_equal_and_opposite(self):
        op=make_operator(film=2000.)
        q=np.full(19,10.)
        r=step(op,dt=1.2,qf=q)
        self.assertGreater(np.min(r.fuel_temperature_k),300)
        self.assertGreater(np.min(r.hydrogen_temperature_k),300)
        self.assertGreater(r.wall_exchange_j,0.)
        self.assertAlmostEqual(r.fuel_storage_j+r.hydrogen_storage_j,
                               r.source_energy_j,delta=2e-8)

    def test_coolant_flow_exports_enthalpy_consistently(self):
        op=make_operator(film=3500.)
        r=step(op,dt=.4,fuel=np.full(19,550.),
               hydrogen=np.full(19,300.),flow=8.,inlet=300.)
        self.assertGreater(r.coolant_export_j,0.)
        self.assertLess(np.max(r.fuel_temperature_k),550.)
        self.assertGreater(np.min(r.hydrogen_temperature_k),300.)
        self.assertAlmostEqual(r.fuel_storage_j+r.hydrogen_storage_j+
                               r.coolant_export_j,0.,delta=2e-7)

    def test_implicit_solver_is_stable_at_large_dt(self):
        op=make_operator(film=1500.)
        r=step(op,dt=100.,fuel=np.full(19,600.),
               hydrogen=np.full(19,300.),flow=0.,inlet=300.)
        self.assertGreater(min(r.fuel_temperature_k.min(),
                               r.hydrogen_temperature_k.min()),299.9999)
        self.assertLess(max(r.fuel_temperature_k.max(),
                            r.hydrogen_temperature_k.max()),600.0001)
        self.assertLess(abs(r.conservation_residual_j),1e-7)

    def test_wall_resistance_uses_series_zrc_and_hydrogen_film(self):
        op=make_operator(film=2000.)
        c=NervaConfig()
        length=c.active_length_cm/100
        b=c.channel_radius_cm/100
        outer=c.coated_channel_radius_cm/100
        geometric_fraction=op.fuel_zrc_contact_area_m2 / (
            2*math.pi*outer*length)
        expected=geometric_fraction*2*math.pi*length / (
            math.log(outer/b)/20.+1/(b*2000.))
        np.testing.assert_allclose(op.channel_wall_conductance_w_k,
                                   expected,rtol=1e-14,atol=1e-13)

    def test_invalid_physical_or_numerical_inputs_rejected(self):
        c=NervaConfig()
        params=dict(conductivity_w_m_k=12.,density_kg_m3=1800.,
                    heat_capacity_j_kg_k=1000.,zrc_conductivity_w_m_k=20.,
                    hydrogen_film_w_m2_k=1000.)
        for key,bad in (("conductivity_w_m_k",0.),
                        ("density_kg_m3",-1.),
                        ("heat_capacity_j_kg_k",float("nan")),
                        ("zrc_conductivity_w_m_k",0.),
                        ("hydrogen_film_w_m2_k",-2.)):
            altered=dict(params)
            altered[key]=bad
            with self.subTest(parameter=key),self.assertRaises(ValueError):
                build_fuel_sector_conduction_2d(c,**altered)
        op=make_operator()
        with self.assertRaises(ValueError):
            step(op,dt=-1.)
        with self.assertRaises(ValueError):
            step(op,qf=np.zeros(18))
        with self.assertRaises(ValueError):
            step(op,flow=-1.)
        # The solver never silently adjusts the historical geometry.
        with self.assertRaises(ValueError):
            build_fuel_sector_conduction_2d(
                replace(c,coolant_bore_pitch_cm=.1),**params)


if __name__ == "__main__":
    unittest.main(verbosity=2)
