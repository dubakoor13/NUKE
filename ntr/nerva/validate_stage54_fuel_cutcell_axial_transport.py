"""Stage54 true axial solid/H2 transport numerical regressions.

The test values are synthetic material/flow data, never reactor operating
conditions or source-normalized nuclear heating predictions.
"""
from __future__ import annotations

import unittest

import numpy as np

from .config import NervaConfig
from .fuel_cutcell_conduction_2d import (
    advance_cutcell_fuel_hydrogen,
    build_cutcell_fuel_2d,
)
from .fuel_cutcell_axial_transport import (
    advance_axial_cutcell,
    build_axial_cutcell_model,
)


def base(*, film=1500., nx=28, ny=34):
    return build_cutcell_fuel_2d(
        NervaConfig(), nx=nx, ny=ny,
        conductivity_w_m_k=12.,
        density_kg_m3=1000.,
        heat_capacity_j_kg_k=1000.,
        zrc_conductivity_w_m_k=20.,
        hydrogen_film_w_m2_k=film,
    )


def axial(b, *, z=None, flow=0.003, axial_k=12.,
          h2mass=0.01, h2cp=1000., flowcp=1000.):
    if z is None:
        z=np.linspace(0,b.axial_length_m,4)
    return build_axial_cutcell_model(
        b, axial_edges_m=z,
        solid_axial_conductivity_w_m_k=axial_k,
        hydrogen_mass_kg=h2mass,
        hydrogen_cp_j_kg_k=h2cp,
        hydrogen_mass_flow_kg_s=flow,
        hydrogen_flow_cp_j_kg_k=flowcp,
    )


def advance(m, *, f=None, h=None, qf=None, qh=None, dt=1., inlet=300.):
    nf=m.transverse.n_fuel
    nz=m.axial_zones
    return advance_axial_cutcell(
        m,
        np.full((nz,nf),300.) if f is None else f,
        np.full((nz,19),300.) if h is None else h,
        dt_s=dt,
        fuel_source_w=np.zeros((nz,nf)) if qf is None else qf,
        hydrogen_source_w=np.zeros((nz,19)) if qh is None else qh,
        inlet_hydrogen_temperature_k=inlet,
    )


class ThreeDimensionalSolidGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b=base(film=0.)
        cls.m=axial(cls.b,z=[0.,0.10,0.55,1.32],flow=0.)

    def test_exact_extruded_volume_and_fuel_thermal_mass(self):
        m=self.m
        self.assertEqual(m.axial_zones,3)
        self.assertEqual(m.solid_volume_m3.shape,(3,self.b.n_fuel))
        self.assertAlmostEqual(float(m.solid_volume_m3.sum()),
                               float(self.b.fuel_volumes_m3.sum()),delta=2e-15)
        self.assertAlmostEqual(float(m.solid_storage_j_k.sum()),
                               float(self.b.fuel_heat_capacity_j_k.sum()),delta=2e-10)
        self.assertEqual(m.hydrogen_storage_j_k.shape,(3,19))
        self.assertEqual(m.axial_solid_interface_w_k.shape,(2,self.b.n_fuel))

    def test_axial_diffusion_symmetric_conservative_positive(self):
        m=self.m
        L=m.solid_diffusion_w_k
        self.assertEqual(L.shape,(m.fuel_unknowns,m.fuel_unknowns))
        np.testing.assert_allclose((L-L.T).data,0.,rtol=0,atol=1e-10)
        np.testing.assert_allclose(L@np.ones(m.fuel_unknowns),0.,
                                   rtol=0,atol=5e-10)
        self.assertTrue(np.all(m.axial_solid_interface_w_k>0.))
        self.assertEqual(m.fuel_hydrogen_exchange_w_k.shape,
                         (m.fuel_unknowns,m.hydrogen_unknowns))

    def test_exact_axial_series_resistance_for_unequal_slabs(self):
        m=self.m
        dz=np.diff(m.axial_edges_m)
        area=self.b.fuel_areas_cm2*1e-4
        expected=area/(dz[0]/(2*12.)+dz[1]/(2*12.))
        np.testing.assert_allclose(m.axial_solid_interface_w_k[0],
                                   expected,rtol=1e-14,atol=1e-13)

    def test_heterogeneous_axial_conductivity_series_resistance(self):
        b=self.b
        n=b.n_fuel
        k=np.stack([np.full(n,8.),np.full(n,20.)])
        m=axial(b,z=[0.,.3,1.32],flow=0.,axial_k=k)
        expected=(b.fuel_areas_cm2*1e-4)/(.3/(2*8.)+1.02/(2*20.))
        np.testing.assert_allclose(m.axial_solid_interface_w_k[0],
                                   expected,rtol=2e-14,atol=1e-13)

    def test_upwind_transport_is_lower_triangular_only(self):
        m=axial(self.b,flow=.012)
        A=m.hydrogen_upwind_w_k.toarray()
        self.assertGreater(A[19,19],0.)
        self.assertLess(A[19,0],0.)
        self.assertEqual(A[0,19],0.)
        self.assertEqual(A[38,19],-12.)
        self.assertEqual(A[38,0],0.)

    def test_invalid_flow_mesh_and_properties_are_rejected(self):
        b=self.b
        for kwargs in (
            {"z":[.1,1.32]}, {"z":[0.,.8,.7,1.32]},
            {"z":[0.,1.30]}, {"flow":-.01}, {"axial_k":0.},
            {"h2mass":-1.}, {"h2cp":0.}, {"flowcp":float("nan")},
        ):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                axial(b,**kwargs)


class CoupledAxialNumericsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b=base(film=1800.)
        cls.m=axial(cls.b,flow=.005)

    def test_one_axial_slab_reduces_exactly_to_stage53(self):
        b=self.b
        m=axial(b,z=[0.,b.axial_length_m],flow=.003,
                h2mass=.05,h2cp=1000.,flowcp=1000.)
        qf=1.e6*b.fuel_volumes_m3
        qh=np.full(19,5.)
        oldf=np.full(b.n_fuel,340.)
        oldh=np.full(19,310.)
        new=advance(m,f=oldf[None,:],h=oldh[None,:],
                    qf=qf[None,:],qh=qh[None,:],dt=.4,inlet=300.)
        reference=advance_cutcell_fuel_hydrogen(
            b,oldf,oldh,dt_s=.4,fuel_source_w=qf,
            hydrogen_source_w=qh,hydrogen_heat_capacity_j_k=50.,
            hydrogen_flow_heat_capacity_w_k=3.,
            hydrogen_inlet_temperature_k=300.)
        np.testing.assert_allclose(new.fuel_temperature_k[0],
                                   reference.fuel_temperature_k,rtol=0,atol=2e-10)
        np.testing.assert_allclose(new.hydrogen_temperature_k[0],
                                   reference.hydrogen_temperature_k,rtol=0,atol=2e-10)
        self.assertLess(abs(new.conservation_residual_j),3e-7)

    def test_two_slab_axial_diffusion_matches_analytic_eigenmode(self):
        b=base(film=0.)
        m=axial(b,z=[0.,.66,1.32],flow=0.,axial_k=12.)
        old=np.vstack((np.full(b.n_fuel,320.),
                       np.full(b.n_fuel,380.)))
        dt=1000.
        result=advance(m,f=old,dt=dt)
        cp_density=1e6
        frac=cp_density*.66**2/(cp_density*.66**2+2*dt*12.)
        expected_low=350.-30.*frac
        expected_high=350.+30.*frac
        np.testing.assert_allclose(result.fuel_temperature_k[0],
                                   expected_low,atol=4e-8,rtol=0.)
        np.testing.assert_allclose(result.fuel_temperature_k[1],
                                   expected_high,atol=4e-8,rtol=0.)
        self.assertLess(abs(result.conservation_residual_j),2e-6)

    def test_3zone_upwind_plugflow_matches_recurrence(self):
        b=base(film=0.)
        m=axial(b,flow=.02,h2mass=.01,h2cp=1000.,flowcp=1000.)
        result=advance(m,dt=1.,inlet=400.)
        expected=[]
        prior=400.
        for z in range(3):
            t=(10.*300.+20.*prior)/30.
            expected.append(t)
            prior=t
        np.testing.assert_allclose(
            result.hydrogen_temperature_k[:,0],expected,
            atol=3e-10,rtol=0.)
        np.testing.assert_allclose(result.hydrogen_temperature_k,
                                   np.asarray(expected)[:,None]+np.zeros((3,19)),
                                   atol=3e-10,rtol=0.)
        self.assertLess(abs(result.conservation_residual_j),4e-7)

    def test_midsection_fuel_heating_advects_downstream(self):
        m=self.m
        f=np.full((3,m.transverse.n_fuel),300.)
        f[1,:]=400.
        result=advance(m,f=f,dt=2.,inlet=300.)
        self.assertGreater(result.hydrogen_temperature_k[1,:].min(),300.)
        self.assertGreater(result.hydrogen_temperature_k[2,:].min(),300.)
        self.assertGreater(result.coolant_outlet_net_enthalpy_j,0.)
        self.assertLess(result.relative_conservation_residual,5e-9)

    def test_advection_and_axial_diffusion_joint_first_law(self):
        m=self.m
        n=m.transverse.n_fuel
        qf=np.zeros((3,n))
        qf[0,:]=1e6*m.solid_volume_m3[0,:]
        qf[1,:]=2e6*m.solid_volume_m3[1,:]
        qf[2,:]=.5e6*m.solid_volume_m3[2,:]
        result=advance(m,qf=qf,dt=.2,inlet=300.)
        self.assertGreater(result.coolant_outlet_net_enthalpy_j,0.)
        self.assertGreater(result.deposited_energy_j,0.)
        self.assertAlmostEqual(
            result.fuel_stored_energy_change_j+
            result.hydrogen_stored_energy_change_j+
            result.coolant_outlet_net_enthalpy_j,
            result.deposited_energy_j,delta=2e-6)
        self.assertLess(result.relative_conservation_residual,5e-9)

    def test_unforced_uniform_temperature_stationary(self):
        m=self.m
        result=advance(
            m,f=np.full((3,m.transverse.n_fuel),320.),
            h=np.full((3,19),320.),dt=50.,inlet=320.)
        np.testing.assert_allclose(result.fuel_temperature_k,320.,
                                   atol=2e-8,rtol=0.)
        np.testing.assert_allclose(result.hydrogen_temperature_k,320.,
                                   atol=2e-8,rtol=0.)
        self.assertLess(abs(result.conservation_residual_j),2e-5)

    def test_rejects_invalid_shapes_sources_and_steps(self):
        m=self.m
        with self.assertRaises(ValueError):
            advance(m,dt=0.)
        with self.assertRaises(ValueError):
            advance(m,qf=np.zeros((2,m.transverse.n_fuel)))
        with self.assertRaises(ValueError):
            advance(m,inlet=-20.)
        with self.assertRaises(ValueError):
            advance(m,h=np.zeros((2,19)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
