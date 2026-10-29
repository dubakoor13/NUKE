"""Stage55 conservative nonlinear enthalpy transport acceptance tests.

Every temperature-dependent table here is SYNTHETIC and is used only to
verify numerical thermodynamics, not to claim real NERVA fuel, ZrC, or
high-pressure hydrogen physical performance.
"""
from __future__ import annotations

import unittest

import numpy as np

from .config import NervaConfig
from .fuel_cutcell_conduction_2d import build_cutcell_fuel_2d
from .fuel_cutcell_axial_transport import (
    advance_axial_cutcell,build_axial_cutcell_model,
)
from .fuel_cutcell_nonlinear_enthalpy import (
    TemperatureTable,NonlinearThermalProperties,advance_nonlinear_axial_cutcell,
)


def table(values, *, allow_zero=False):
    return TemperatureTable([200.,300.,400.,700.],values,allow_zero=allow_zero)


def laws(*, nonlinear=False, film=2500.):
    return NonlinearThermalProperties(
        fuel_cp_j_kg_k=table([900.,1000.,1130.,1600.] if nonlinear
                             else [1000.]*4),
        fuel_k_w_m_k=table([16.,12.,10.,7.] if nonlinear
                           else [12.]*4),
        coating_k_w_m_k=table([25.,20.,16.,11.] if nonlinear
                              else [20.]*4),
        hydrogen_cp_j_kg_k=table([850.,1000.,1250.,1650.] if nonlinear
                                 else [1000.]*4),
        hydrogen_film_w_m2_k=table(
            [film*.65,film,film*1.2,film*1.5] if nonlinear
            else [film]*4,allow_zero=True),
    )


def setup(*, film=2500., nz=2):
    c=NervaConfig()
    b=build_cutcell_fuel_2d(
        c,nx=24,ny=30,
        conductivity_w_m_k=12.,
        density_kg_m3=1000.,
        heat_capacity_j_kg_k=1000.,
        zrc_conductivity_w_m_k=20.,
        hydrogen_film_w_m2_k=film)
    m=build_axial_cutcell_model(
        b,axial_edges_m=np.linspace(0.,1.32,nz+1),
        solid_axial_conductivity_w_m_k=12.,
        hydrogen_mass_kg=.02,
        hydrogen_cp_j_kg_k=1000.,
        hydrogen_mass_flow_kg_s=.005,
        hydrogen_flow_cp_j_kg_k=1000.)
    return c,m


def run(c,m,p,*,old_f=300.,old_h=300.,inlet=300.,
        source_f=None,source_h=None,dt=.2,max_iterations=24):
    nz=m.axial_zones
    nf=m.transverse.n_fuel
    if source_f is None:
        source_f=np.zeros((nz,nf))
    if source_h is None:
        source_h=np.zeros((nz,19))
    if np.isscalar(old_f):
        old_f=np.full((nz,nf),float(old_f))
    if np.isscalar(old_h):
        old_h=np.full((nz,19),float(old_h))
    return advance_nonlinear_axial_cutcell(
        m,c,p,old_f,old_h,
        dt_s=dt,fuel_source_w=source_f,hydrogen_source_w=source_h,
        solid_density_kg_m3=1000.,hydrogen_mass_kg=.02,
        inlet_hydrogen_temperature_k=inlet,
        max_iterations=max_iterations,
    )


class ExactThermophysicalTableTests(unittest.TestCase):
    def test_exact_piecewise_linear_enthalpy_integral(self):
        cp=TemperatureTable([200.,300.,500.],[1000.,1200.,2000.])
        self.assertAlmostEqual(float(cp.integral(200.)),0.)
        self.assertAlmostEqual(float(cp.integral(300.)),110000.)
        self.assertAlmostEqual(float(cp.integral(400.)),250000.)
        self.assertAlmostEqual(float(cp.integral(500.)),430000.)
        values=np.array([220.,301.,480.])
        self.assertTrue(np.all(cp.at(values)>0))
        self.assertTrue(np.all(np.diff(cp.integral(values))>0))

    def test_piecewise_enthalpy_derivative_equals_tabulated_cp(self):
        cp=table([800.,1000.,1200.,2000.])
        t=np.array([250.,320.,550.])
        eps=1e-4
        numerical=(cp.integral(t+eps)-cp.integral(t-eps))/(2*eps)
        np.testing.assert_allclose(numerical,cp.at(t),
                                   rtol=2e-10,atol=2e-6)

    def test_property_temperature_limits_and_bad_data_fail_closed(self):
        with self.assertRaises(ValueError):
            TemperatureTable([300.,300.],[1.,2.])
        with self.assertRaises(ValueError):
            TemperatureTable([300.,500.],[1.,-2.])
        with self.assertRaises(ValueError):
            TemperatureTable([300.,500.],[1.,float("nan")])
        with self.assertRaises(ValueError):
            table([1.,2.,3.,4.]).at(199.)
        with self.assertRaises(ValueError):
            table([1.,2.,3.,4.]).integral(701.)

    def test_zero_film_allowed_only_when_explicit(self):
        values=TemperatureTable([200.,700.],[0.,0.],allow_zero=True)
        self.assertEqual(float(values.at(300.)),0.)
        with self.assertRaises(ValueError):
            TemperatureTable([200.,700.],[0.,0.])


class CoupledNonlinearTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c,cls.m=setup()

    def test_constant_properties_matches_stage54_full_coupled_field(self):
        c,m=self.c,self.m
        p=laws()
        oldf=np.full((m.axial_zones,m.transverse.n_fuel),343.)
        oldh=np.full((m.axial_zones,19),310.)
        qf=1e6*m.solid_volume_m3
        qh=np.full((m.axial_zones,19),2.)
        original=advance_axial_cutcell(
            m,oldf,oldh,dt_s=.25,fuel_source_w=qf,
            hydrogen_source_w=qh,inlet_hydrogen_temperature_k=301.)
        new=run(c,m,p,old_f=oldf,old_h=oldh,inlet=301.,
                source_f=qf,source_h=qh,dt=.25)
        np.testing.assert_allclose(new.fuel_temperature_k,
                                   original.fuel_temperature_k,
                                   atol=4e-9,rtol=0.)
        np.testing.assert_allclose(new.hydrogen_temperature_k,
                                   original.hydrogen_temperature_k,
                                   atol=4e-9,rtol=0.)
        self.assertLess(abs(new.first_law_residual_j),1e-7)

    def test_nonlinear_enthalpy_uniform_source_exact_one_degree(self):
        c,m=setup(film=0.)
        # Q per kg = integral cp from 300 to 301, not cp(300)*1.
        prop=NonlinearThermalProperties(
            fuel_cp_j_kg_k=table([800.,1000.,1200.,1800.]),
            fuel_k_w_m_k=table([15.,12.,10.,5.]),
            coating_k_w_m_k=table([25.,20.,16.,11.]),
            hydrogen_cp_j_kg_k=table([900.,1000.,1100.,1400.]),
            hydrogen_film_w_m2_k=table([0.]*4,allow_zero=True),
        )
        ent=prop.fuel_cp_j_kg_k.integral
        delta=float(ent(301.)-ent(300.))
        power=1000.*m.solid_volume_m3*delta
        r=run(c,m,prop,source_f=power,dt=1.)
        np.testing.assert_allclose(r.fuel_temperature_k,301.,
                                   atol=3e-8,rtol=0.)
        np.testing.assert_allclose(r.hydrogen_temperature_k,300.,
                                   atol=1e-9,rtol=0.)
        self.assertLess(abs(r.first_law_residual_j),2e-7)

    def test_nonlinear_upwind_transport_uses_specific_enthalpy(self):
        c,m=setup(film=0.,nz=3)
        prop=laws(nonlinear=True,film=0.)
        r=run(c,m,prop,inlet=405.,dt=.75)
        ent=prop.hydrogen_cp_j_kg_k.integral
        upstream=float(ent(405.))
        old=float(ent(300.))
        expected=[]
        for iz in range(3):
            h=(.02/.75*old+.005*upstream)/(.02/.75+.005)
            expected.append(h)
            upstream=h
        np.testing.assert_allclose(
            ent(r.hydrogen_temperature_k[:,0]),expected,
            rtol=0,atol=4e-7)
        np.testing.assert_allclose(
            r.hydrogen_temperature_k[:,1:],
            r.hydrogen_temperature_k[:,0,None],atol=5e-9,rtol=0.)
        self.assertLess(abs(r.first_law_residual_j),2e-6)

    def test_spatially_nonuniform_heating_and_temperature_properties(self):
        c,m=self.c,self.m
        q=1e6*m.solid_volume_m3
        q[0,:]*=1.8
        q[1,:]*=.3
        result=run(c,m,laws(nonlinear=True),
                   old_f=345.,old_h=310.,inlet=295.,
                   source_f=q,dt=.4)
        self.assertGreater(result.iterations,0)
        self.assertLess(result.final_max_residual_w,2e-8)
        self.assertGreater(result.hydrogen_net_export_j,0.)
        self.assertGreater(np.max(result.fuel_temperature_k),
                           np.min(result.fuel_temperature_k))
        self.assertLess(result.relative_first_law_residual,2e-9)

    def test_stronger_temperature_dependent_film_changes_heat_split(self):
        c,m=self.c,self.m
        p1=laws(nonlinear=True)
        p2=NonlinearThermalProperties(
            fuel_cp_j_kg_k=p1.fuel_cp_j_kg_k,
            fuel_k_w_m_k=p1.fuel_k_w_m_k,
            coating_k_w_m_k=p1.coating_k_w_m_k,
            hydrogen_cp_j_kg_k=p1.hydrogen_cp_j_kg_k,
            hydrogen_film_w_m2_k=table([500.]*4,allow_zero=True),
        )
        hot=run(c,m,p1,old_f=390.,old_h=300.,inlet=300.,dt=.35)
        cold=run(c,m,p2,old_f=390.,old_h=300.,inlet=300.,dt=.35)
        self.assertGreater(hot.fuel_to_hydrogen_transfer_j,
                           cold.fuel_to_hydrogen_transfer_j)
        self.assertLess(abs(hot.first_law_residual_j),1e-5)

    def test_exact_stationary_state_conserves_enthalpy(self):
        c,m=self.c,self.m
        result=run(c,m,laws(nonlinear=True),old_f=333.,
                   old_h=333.,inlet=333.,dt=.8)
        np.testing.assert_allclose(result.fuel_temperature_k,333.,
                                   atol=3e-8,rtol=0.)
        np.testing.assert_allclose(result.hydrogen_temperature_k,333.,
                                   atol=3e-8,rtol=0.)
        self.assertLess(abs(result.first_law_residual_j),2e-6)

    def test_insufficient_property_range_does_not_extrapolate(self):
        c,m=self.c,self.m
        with self.assertRaises(ValueError):
            run(c,m,laws(nonlinear=True),old_f=800.)
        with self.assertRaises(ValueError):
            run(c,m,laws(nonlinear=True),inlet=150.)
        with self.assertRaises(ValueError):
            run(c,m,laws(nonlinear=True),dt=0.)
        with self.assertRaises(ValueError):
            run(c,m,laws(nonlinear=True),source_f=np.zeros((3,2)))


if __name__=="__main__":
    unittest.main(verbosity=2)
