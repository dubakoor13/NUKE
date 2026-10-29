"""Stage71 axial conduction inside the 19 individual ZrC coating sleeves.

These are manufactured/numerical verification cases using synthetic
cp(T), k(T), contact conductances and nuclear heating. Physical
circumferential conduction would require independently resolved angular
coating DOFs: neighboring channels are disjoint and MUST NOT be linked.
"""
from __future__ import annotations

from dataclasses import replace
import unittest
import numpy as np

from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .fuel_hydrogen_coupled_transient import with_resolved_coating
from .resolved_coating_thermal import (
    advance_coating_axial_enthalpy, advance_coating_enthalpy,
)
from .validate_stage58_fuel_hydrogen_coupling import (
    coupled, initial, solve,
)
from .validate_stage70_resolved_coating import (
    coating_model, coating_initial,
)


def axial_model(nz=3,film=0.):
    base=coupled(film=film,nz=nz,nx=20,ny=24)
    cp=TemperatureTable(
        [240.,300.,500.,800.,1100.],[690.,725.,800.,890.,1000.])
    return with_resolved_coating(
        base,density_kg_m3=5000.,heat_capacity=cp,
        contact_resistance_m2_k_w=1e-5,
        axial_conduction_enabled=True,
    )


def constant_tables(model):
    cp=TemperatureTable([250.,1000.],[800.,800.])
    k=TemperatureTable([250.,1000.],[20.,20.])
    return replace(model.coating,heat_capacity=cp),k


class AxialCoatingFiniteVolumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=axial_model(nz=3,film=0.)

    def test_exact_area_matches_each_original_annular_axial_volume(self):
        m=self.m
        c=m.coating
        dz=np.diff(c.axial_edges_m)
        area=c.volume_m3/dz[:,None]
        self.assertTrue(c.axial_conduction_enabled)
        np.testing.assert_allclose(area,np.broadcast_to(area[:1,:],area.shape),rtol=5e-11,atol=1e-15)
        self.assertGreater(float(np.min(area)),0.)
        self.assertLess(float(np.sum(area)),1e-4)

    def test_two_zone_constant_property_implicit_matches_analytic_matrix(self):
        m=axial_model(nz=2)
        c,k=constant_tables(m)
        t=np.full((2,19),450.)
        t[0,8]=600.
        t[1,8]=370.
        dt=1000.
        result=advance_coating_axial_enthalpy(
            c,t,conductivity_w_m_k=k,dt_s=dt,
            net_w=np.zeros_like(t))
        dz=np.diff(c.axial_edges_m)
        A=c.volume_m3[0,8]/dz[0]
        G=A/(dz[0]/(2*20.)+dz[1]/(2*20.))
        capacities=c.mass_kg[:,8]*800.
        matrix=np.array([
            [capacities[0]/dt+G,-G],
            [-G,capacities[1]/dt+G],
        ])
        expected=np.linalg.solve(matrix,capacities*t[:,8]/dt)
        np.testing.assert_allclose(result.temperature_k[:,8],
                                   expected,atol=5e-10,rtol=0)
        np.testing.assert_allclose(result.temperature_k[:,0],
                                   t[:,0],atol=1e-11,rtol=0)
        self.assertAlmostEqual(
            result.axial_interface_j[0,8],
            G*(expected[0]-expected[1])*dt,delta=5e-10)
        self.assertLess(abs(result.global_first_law_residual_j),1e-8)

    def test_discrete_constant_mode_and_zero_source_is_unchanged(self):
        m=self.m
        t=np.full((m.nzone,19),470.)
        q=np.zeros_like(t)
        step=advance_coating_axial_enthalpy(
            m.coating,t,conductivity_w_m_k=m.properties.coating_k_w_m_k,
            dt_s=0.2,net_w=q)
        np.testing.assert_array_equal(step.temperature_k,t)
        np.testing.assert_array_equal(step.axial_interface_j,
                                      np.zeros((2,19)))
        self.assertAlmostEqual(step.net_axial_energy_j,0.,delta=1e-12)
        self.assertEqual(step.global_first_law_residual_j,0.)

    def test_axial_heating_spreads_without_heating_other_channels(self):
        m=self.m
        c,k=constant_tables(m)
        t=np.full((3,19),450.)
        source=np.zeros_like(t)
        source[0,8]=1.
        step=advance_coating_axial_enthalpy(
            c,t,conductivity_w_m_k=k,
            dt_s=100.,net_w=source)
        self.assertGreater(step.temperature_k[0,8],450.)
        self.assertGreater(step.temperature_k[1,8],450.)
        self.assertGreater(step.temperature_k[2,8],450.)
        np.testing.assert_allclose(
            step.temperature_k[:,[i for i in range(19) if i!=8]],
            450.,atol=1e-11,rtol=0)
        self.assertAlmostEqual(
            float(np.sum(c.mass_kg*(c.heat_capacity.integral(
                step.temperature_k)-c.heat_capacity.integral(t)))),
            100.,delta=1e-7)
        self.assertAlmostEqual(step.net_axial_energy_j,0.,delta=1e-12)

    def test_variable_cp_and_k_recomputed_conservatively(self):
        m=self.m
        t=np.full((3,19),430.)
        t[0,6]=570.
        t[1,6]=350.
        power=np.zeros_like(t)
        power[1,6]=4.
        dt=25.
        r=advance_coating_axial_enthalpy(
            m.coating,t,conductivity_w_m_k=m.properties.coating_k_w_m_k,
            dt_s=dt,net_w=power)
        self.assertLess(r.temperature_k[0,6],570.)
        self.assertGreater(r.temperature_k[1,6],350.)
        self.assertAlmostEqual(
            float(np.sum(m.coating.mass_kg*
                   (m.coating.heat_capacity.integral(r.temperature_k)-
                    m.coating.heat_capacity.integral(t)))),
            dt*4.,delta=3e-8)
        self.assertLess(
            float(np.max(np.abs(r.per_node_first_law_residual_j))),1e-7)

    def test_flux_is_equal_opposite_for_adjacent_axial_nodes(self):
        m=self.m
        c,k=constant_tables(m)
        t=np.full((3,19),450.)
        t[:,4]=[600.,400.,450.]
        r=advance_coating_axial_enthalpy(
            c,t,conductivity_w_m_k=k,dt_s=30.,
            net_w=np.zeros_like(t))
        face=r.axial_interface_j[:,4]
        axial_net=np.zeros(3)
        axial_net[:-1]+=face
        axial_net[1:]-=face
        self.assertAlmostEqual(float(np.sum(axial_net)),0.,delta=1e-12)
        actual=c.mass_kg[:,4]*(
            c.heat_capacity.integral(r.temperature_k[:,4])-
            c.heat_capacity.integral(t[:,4]))
        np.testing.assert_allclose(actual+axial_net,0.,
                                   rtol=0,atol=2e-8)

    def test_single_axial_slab_has_no_conduction_and_matches_stage70(self):
        m=axial_model(nz=1)
        t=np.full((1,19),450.)
        power=np.zeros_like(t)
        power[0,2]=3.
        dt=3e-3
        a=advance_coating_axial_enthalpy(
            m.coating,t,
            conductivity_w_m_k=m.properties.coating_k_w_m_k,
            dt_s=dt,net_w=power)
        b=advance_coating_enthalpy(
            m.coating,t,dt_s=dt,net_w=power)
        np.testing.assert_allclose(a.temperature_k,b,
                                   atol=2e-10,rtol=0)
        self.assertEqual(a.axial_interface_j.shape,(0,19))

    def test_small_conductivity_limit_approaches_uncoupled_local_enthalpy(self):
        m=self.m
        slow=TemperatureTable([240.,1100.],[1e-9,1e-9])
        t=np.full((3,19),450.)
        t[0,1]=480.
        q=np.zeros_like(t)
        q[2,1]=2.
        dt=.01
        a=advance_coating_axial_enthalpy(
            m.coating,t,conductivity_w_m_k=slow,
            dt_s=dt,net_w=q)
        b=advance_coating_enthalpy(
            m.coating,t,dt_s=dt,net_w=q)
        np.testing.assert_allclose(a.temperature_k,b,atol=2e-9,rtol=0)

    def test_stage58_three_temperature_plus_axial_coating_diffusion(self):
        m=axial_model(nz=3,film=1400.)
        s=coating_initial(m,tf=485.,tc=440.,th=395.)
        t=np.asarray(s.coating_temperature_k).copy()
        t[0,8]=475.
        s=replace(s,coating_temperature_k=t)
        result=solve(m,s,dt=5e-6,
                     q=np.full_like(s.fuel_temperature_k,3.),
                     coating_deposition_w=np.full((3,19),2.),
                     hydrogen_deposition_w=np.full((3,19),1.))
        self.assertEqual(result.coating_axial_interface_j.shape,(2,19))
        self.assertGreater(
            abs(float(result.coating_axial_interface_j[0,8])),0.)
        self.assertLess(result.relative_first_law_residual,3e-8)
        self.assertLess(abs(result.coating_first_law_residual_j),8e-7)
        self.assertLess(result.coating_axial_max_node_residual_j,1e-7)

    def test_stage70_no_axial_diffusion_still_uses_original_solver(self):
        m=coating_model(nz=2,nx=20,ny=24)
        s=coating_initial(m,tf=480.,tc=445.,th=400.)
        r=solve(m,s,dt=5e-6)
        self.assertIsNone(r.coating_axial_interface_j)
        self.assertEqual(r.coating_axial_max_node_residual_j,0.)
        self.assertLess(r.relative_first_law_residual,3e-8)

    def test_source_and_temperature_domain_fail_closed(self):
        m=self.m
        good=np.full((3,19),450.)
        for invalid in (np.ones((2,19)),np.full((3,19),np.nan)):
            with self.assertRaisesRegex(ValueError,"shape"):
                advance_coating_axial_enthalpy(
                    m.coating,good,
                    conductivity_w_m_k=m.properties.coating_k_w_m_k,
                    dt_s=1.,net_w=invalid)
        with self.assertRaisesRegex(ValueError,"time step"):
            advance_coating_axial_enthalpy(
                m.coating,good,
                conductivity_w_m_k=m.properties.coating_k_w_m_k,
                dt_s=0.,net_w=np.zeros_like(good))
        with self.assertRaises(ValueError):
            advance_coating_axial_enthalpy(
                m.coating,np.full((3,19),200.),
                conductivity_w_m_k=m.properties.coating_k_w_m_k,
                dt_s=1.,net_w=np.zeros((3,19)))

    def test_nonphysical_cross_channel_conduction_is_not_created(self):
        # A channel-hot perturbation must NOT heat adjacent distinct bores
        # when coupling to the fuel and H2 is not part of this operator.
        m=self.m
        c,k=constant_tables(m)
        initial_temp=np.full((3,19),430.)
        initial_temp[0,5]=600.
        r=advance_coating_axial_enthalpy(
            c,initial_temp,conductivity_w_m_k=k,dt_s=400.,
            net_w=np.zeros((3,19)))
        self.assertGreater(r.temperature_k[1,5],430.)
        np.testing.assert_array_equal(
            r.temperature_k[:,[i for i in range(19) if i!=5]],
            initial_temp[:,[i for i in range(19) if i!=5]])


if __name__=="__main__":
    unittest.main(verbosity=2)
