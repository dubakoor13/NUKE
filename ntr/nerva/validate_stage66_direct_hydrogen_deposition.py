"""Stage66: separately conserved direct coolant deposition and wall heat.

All deposited nuclear watts and ideal-H2 material properties used here are
synthetic. Stage66 numerically tests correct energy accounting in the
coupled thermal–hydraulic state, NOT a real OpenMC coolant tally, selected
fuel-element instance, nuclear energy budget or NASA NERVA hot operation.
"""
from __future__ import annotations

import unittest
import numpy as np

from .hydrogen_transient_fvm import HydrogenBoundary, primitive
from .validate_stage58_fuel_hydrogen_coupling import coupled, initial, solve


class DirectHydrogenNuclearDepositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.insulated=coupled(film=0.,nz=2,nx=20,ny=24)

    def test_one_channel_one_axial_zone_direct_heat_increases_gas_energy(self):
        m=self.insulated
        old=initial(m,tf=400.,th=400.)
        q=np.zeros((m.nzone,19))
        q[1,8]=5000.
        dt=1e-5
        r=solve(m,old,dt=dt,hydrogen_deposition_w=q)
        gas=primitive(m.fluid,r.state.hydrogen)
        self.assertAlmostEqual(r.hydrogen_deposition_j,.05,delta=1e-15)
        self.assertAlmostEqual(float(r.hydrogen_deposition_per_channel_j[1,8]),
                               .05,delta=1e-15)
        self.assertEqual(np.count_nonzero(r.hydrogen_deposition_per_channel_j),1)
        self.assertAlmostEqual(r.hydrogen_total_energy_change_j,.05,delta=2e-10)
        self.assertLess(abs(r.fuel_storage_change_j),2e-8)
        self.assertAlmostEqual(float(np.sum(r.wall_heat_to_hydrogen_j)),
                               0.,delta=1e-15)
        self.assertGreater(gas.temperature_k[1,8],400.)
        self.assertGreater(gas.pressure_pa[1,8],3e6)
        self.assertLess(r.relative_first_law_residual,2e-9)

    def test_direct_nuclear_heat_cannot_be_counted_as_fuel_wall_transfer(self):
        m=coupled(film=1550.,nz=2,nx=20,ny=24)
        old=initial(m,tf=525.,th=400.)
        fuel_q=4e5*m.solid.solid_volume_m3
        gas_q=np.full((2,19),800.)
        dt=6e-6
        r=solve(m,old,dt=dt,q=fuel_q,hydrogen_deposition_w=gas_q)
        wall=float(np.sum(r.wall_heat_to_hydrogen_j))
        self.assertGreater(wall,0.)
        self.assertAlmostEqual(r.hydrogen_deposition_j,800*38*dt,delta=1e-13)
        self.assertAlmostEqual(
            r.fuel_storage_change_j+wall,r.fuel_deposition_j,delta=5e-7)
        self.assertAlmostEqual(
            r.hydrogen_total_energy_change_j+
            r.hydrogen_net_outlet_energy_j-wall,
            r.hydrogen_deposition_j,delta=5e-8)
        self.assertLess(r.relative_first_law_residual,1e-8)

    def test_nonzero_wall_convection_and_direct_gas_heating_are_distinct(self):
        m=coupled(film=1400.,nz=2,nx=20,ny=24)
        old=initial(m,tf=530.,th=410.)
        gas_q=np.zeros((2,19))
        gas_q[0,4]=4000.
        r=solve(m,old,dt=5e-6,hydrogen_deposition_w=gas_q)
        self.assertGreater(float(np.sum(r.wall_heat_to_hydrogen_j)),0.)
        self.assertGreater(r.hydrogen_deposition_j,0.)
        self.assertGreater(r.hydrogen_total_energy_change_j,0.)
        self.assertLess(r.fuel_storage_change_j,0.)
        self.assertLess(abs(r.first_law_residual_j),5e-7)

    def test_zero_direct_deposition_is_legacy_equivalent(self):
        m=coupled(film=1300.,nz=2,nx=20,ny=24)
        old=initial(m,tf=460.,th=400.)
        a=solve(m,old,dt=5e-6)
        b=solve(m,old,dt=5e-6,
                hydrogen_deposition_w=np.zeros((m.nzone,19)))
        np.testing.assert_array_equal(a.state.fuel_temperature_k,
                                      b.state.fuel_temperature_k)
        np.testing.assert_array_equal(a.state.hydrogen.conserved,
                                      b.state.hydrogen.conserved)
        np.testing.assert_array_equal(a.wall_heat_to_hydrogen_j,
                                      b.wall_heat_to_hydrogen_j)
        self.assertEqual(a.hydrogen_deposition_j,0.)
        self.assertEqual(b.hydrogen_deposition_j,0.)

    def test_open_flow_mass_and_energy_plus_independent_nuclear_heat(self):
        m=coupled(film=1000.,nz=2,nx=20,ny=24)
        old=initial(m,tf=480.,th=395.,vel=12.)
        fuel_q=3e5*m.solid.solid_volume_m3
        gas_q=np.full((m.nzone,19),1700.)
        r=solve(m,old,dt=1e-5,q=fuel_q,
                hydrogen_deposition_w=gas_q,
                bc=HydrogenBoundary("transmissive"))
        wall=float(np.sum(r.wall_heat_to_hydrogen_j))
        self.assertAlmostEqual(
            r.fuel_storage_change_j+wall,r.fuel_deposition_j,delta=5e-7)
        self.assertAlmostEqual(
            r.hydrogen_total_energy_change_j+
            r.hydrogen_net_outlet_energy_j-wall,
            r.hydrogen_deposition_j,delta=5e-8)
        self.assertLess(r.relative_first_law_residual,2e-8)
        self.assertLess(abs(r.hydrogen_mass_change_kg),1e-10)

    def test_hydrogen_direct_power_consistent_across_acoustic_subcycling(self):
        m=self.insulated
        old=initial(m,tf=400.,th=400.)
        q=np.full((m.nzone,19),40.)
        dt=2.5e-4
        r=solve(m,old,dt=dt,hydrogen_deposition_w=q)
        self.assertGreater(r.substeps,1)
        self.assertAlmostEqual(
            r.hydrogen_deposition_j,40*38*dt,delta=1e-12)
        self.assertAlmostEqual(
            r.hydrogen_total_energy_change_j,
            r.hydrogen_deposition_j,delta=3e-9)
        self.assertLess(r.relative_first_law_residual,2e-9)

    def test_scalar_hydrogen_power_broadcast_over_19_channels(self):
        m=self.insulated
        old=initial(m,tf=400.,th=400.)
        dt=5e-6
        r=solve(m,old,dt=dt,hydrogen_deposition_w=21.)
        np.testing.assert_allclose(r.hydrogen_deposition_per_channel_j,
                                   21*dt,rtol=0,atol=1e-15)
        self.assertAlmostEqual(r.hydrogen_deposition_j,
                               m.nzone*19*21*dt,delta=1e-14)

    def test_invalid_negative_nuclear_sources_fail_before_time_advance(self):
        m=self.insulated
        old=initial(m,tf=400.,th=400.)
        gas_negative=np.zeros((m.nzone,19))
        gas_negative[0,0]=-1.
        for q in (gas_negative,np.full((m.nzone,19),np.nan),
                  np.ones((m.nzone,18)),np.array([1.,2.]),
                  -1.):
            with self.subTest(q_shape=np.shape(q)):
                with self.assertRaisesRegex(ValueError,"hydrogen nuclear deposition"):
                    solve(m,old,hydrogen_deposition_w=q)
        fuel_negative=np.zeros_like(old.fuel_temperature_k)
        fuel_negative[0,0]=-1.
        with self.assertRaisesRegex(ValueError,"fuel deposition"):
            solve(m,old,q=fuel_negative)
        np.testing.assert_array_equal(
            old.hydrogen.conserved,
            initial(m,tf=400.,th=400.).hydrogen.conserved)

    def test_zero_source_joint_uniform_rest_is_unchanged(self):
        m=self.insulated
        old=initial(m,tf=400.,th=400.)
        r=solve(m,old,dt=5e-6,q=np.zeros_like(old.fuel_temperature_k),
                hydrogen_deposition_w=0.)
        self.assertEqual(r.fuel_deposition_j,0.)
        self.assertEqual(r.hydrogen_deposition_j,0.)
        self.assertEqual(float(np.sum(r.wall_heat_to_hydrogen_j)),0.)
        self.assertLess(abs(r.hydrogen_total_energy_change_j),1e-8)
        self.assertLess(abs(r.fuel_storage_change_j),1e-8)


if __name__=="__main__":
    unittest.main(verbosity=2)
