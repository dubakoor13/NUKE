"""Analytic and independent-matrix checks for Stage 19 conjugate fuel/H2 FV.

All properties, dimensions and heat depositions below are SYNTHETIC.
No NASA NERVA dimensions or evaluated thermophysical data are implied.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from .fx2th_conjugate_fuel_transport import (
    ConjugateCoreGeometry, ConjugateCoreState,
    ConjugateCoreDeposition, assemble_conjugate_core,
    advance_conjugate_core, effective_solid_temperature_for_xs,
)


def fixture(*, npath=2, nz=3, nr=3, flow=.002, wall=20.):
    z=np.linspace(0,2,nz+1)
    r=np.stack([np.linspace(.01+.001*i,.031+.001*i,nr+1)
                for i in range(npath)])
    shape=(npath,nz,nr)
    geo=ConjugateCoreGeometry(
        axial_edges_m=z,
        fuel_radial_edges_m=r,
        channels_per_radial_zone=np.arange(npath)+1,
        solid_density_kg_m3=np.full(shape,2000.),
        solid_heat_capacity_j_kg_k=np.full(shape,500.),
        solid_conductivity_w_m_k=np.full(shape,12.),
        hydrogen_mass_kg=np.full((npath,nz),.002),
        hydrogen_cp_j_kg_k=np.full(npath,1000.),
        hydrogen_mass_flow_kg_s=np.full(npath,flow),
        inlet_temperature_k=np.full(npath,300.),
        inner_wall_conductance_w_k=np.full((npath,nz),wall),
    )
    Ts=np.full(shape,400.)
    Th=np.full((npath,nz),330.)
    state=ConjugateCoreState(0.,Ts,Th)
    power=ConjugateCoreDeposition(
        source_id="synthetic-resolved-thermal-test",
        solid_w=np.full(shape,4.),
        direct_hydrogen_w=np.full((npath,nz),3.),
        other_w=np.full((npath,nz),2.),
        declared_total_w=float(4.*np.prod(shape)+5.*npath*nz),
        synthetic=True,
    )
    return geo,state,power


def heat(geo,solid=0.,h2=0.,other=0.):
    p,z,r=geo.shape
    return ConjugateCoreDeposition(
        "synthetic-fixed-heat",np.full((p,z,r),solid),
        np.full((p,z),h2),np.full((p,z),other),
        float((p*z*r)*solid+(p*z)*(h2+other)),True
    )


class ConjugateFuelTransportTests(unittest.TestCase):
    def test_exact_annular_volume_and_parallel_count(self):
        geo,_,_=fixture()
        edges=geo.fuel_radial_edges_m
        for p in range(geo.shape[0]):
            expected=(geo.channels_per_radial_zone[p]*math.pi*
                      (edges[p,-1]**2-edges[p,0]**2)*2.)
            self.assertAlmostEqual(geo.solid_volumes_m3[p].sum(),expected,places=14)
        self.assertGreater(geo.solid_volumes_m3.min(),0)

    def test_radial_face_conductance_is_exact_log_resistance(self):
        geo,s,power=fixture(npath=1,nz=1,nr=2,flow=0,wall=0)
        matrix=assemble_conjugate_core(geo,s,heat(geo),.1).matrix.toarray()
        r=geo.fuel_radial_edges_m[0]
        rc=np.sqrt(.5*(r[:-1]**2+r[1:]**2))
        R=np.log(r[1]/rc[0])/(2*math.pi*geo.solid_conductivity_w_m_k[0,0,0]*2)
        R+=np.log(rc[1]/r[1])/(2*math.pi*geo.solid_conductivity_w_m_k[0,0,1]*2)
        self.assertAlmostEqual(matrix[0,1],-1./R,places=10)
        self.assertAlmostEqual(matrix[1,0],-1./R,places=10)

    def test_axial_harmonic_conductance_nonuniform(self):
        geo,s,_=fixture(npath=1,nz=2,nr=1,flow=0,wall=0)
        geo=replace(geo,axial_edges_m=np.array([0.,.25,2.]),
                    solid_conductivity_w_m_k=np.array([[[7.],[30.]]]))
        A=assemble_conjugate_core(geo,s,heat(geo),.1).matrix.toarray()
        cross=math.pi*(geo.fuel_radial_edges_m[0,1]**2-
                       geo.fuel_radial_edges_m[0,0]**2)
        conductance=cross/(.25/(2*7)+1.75/(2*30))
        self.assertAlmostEqual(A[0,1],-conductance,places=11)
        self.assertAlmostEqual(A[1,0],-conductance,places=11)

    def test_axial_coolant_upwind_matrix_uses_new_time_upstream(self):
        geo,s,_=fixture(npath=1,nz=3,nr=1,flow=.003,wall=0)
        A=assemble_conjugate_core(geo,s,heat(geo),.2).matrix.toarray()
        # First 3 unknowns solid, last 3 coolant. No spurious downwind.
        self.assertAlmostEqual(A[4,3],-3.,places=12)
        self.assertAlmostEqual(A[5,4],-3.,places=12)
        self.assertEqual(A[3,4],0.)
        self.assertEqual(A[4,5],0.)

    def test_radial_conduction_steady_analytic_cylinder(self):
        geo,s,_=fixture(npath=1,nz=1,nr=2,flow=.002,wall=20.)
        solid=np.array([[[0.,100.]]])
        power=ConjugateCoreDeposition(
            "synthetic-outer-fuel-heating",solid,np.zeros((1,1)),
            np.zeros((1,1)),100.,True)
        step=advance_conjugate_core(geo,s,power,1.e10)
        gas=step.state.hydrogen_temperature_k[0,0]
        inner,outer=step.state.fuel_temperature_k[0,0]
        self.assertAlmostEqual(gas,300.+100./2.,delta=.0003)
        self.assertAlmostEqual(inner,gas+100./20.,delta=.0003)
        r=geo.fuel_radial_edges_m[0]
        rc=np.sqrt(.5*(r[:-1]**2+r[1:]**2))
        k=geo.solid_conductivity_w_m_k[0,0]
        resistance=(np.log(r[1]/rc[0])/k[0]+
                    np.log(rc[1]/r[1])/k[1])/(2*math.pi*2)
        self.assertAlmostEqual(outer-inner,100*resistance,delta=.0003)

    def test_axial_direct_coolant_upwind_steady_reference(self):
        geo,s,_=fixture(npath=1,nz=3,nr=1,flow=.004,wall=0)
        power=ConjugateCoreDeposition(
            "synthetic-axial-direct",np.zeros((1,3,1)),
            np.array([[8.,12.,20.]]),np.zeros((1,3)),40.,True)
        new=advance_conjugate_core(geo,s,power,1e9)
        np.testing.assert_allclose(new.state.hydrogen_temperature_k[0],
                                   [302.,305.,310.],atol=1e-5,rtol=0)
        self.assertAlmostEqual(new.outlet_net_enthalpy_j/1e9,40.,places=4)

    def test_zero_power_closed_system_preserves_uniform_temperature(self):
        geo,s,_=fixture(npath=1,nz=2,nr=3,flow=0,wall=60.)
        s=ConjugateCoreState(0.,np.full(geo.shape,345.),
                               np.full((1,2),345.))
        result=advance_conjugate_core(geo,s,heat(geo),.7)
        np.testing.assert_allclose(result.state.fuel_temperature_k,
                                   345.,atol=1e-10)
        np.testing.assert_allclose(result.state.hydrogen_temperature_k,
                                   345.,atol=1e-10)
        self.assertAlmostEqual(result.thermal_balance_residual_j,0,places=8)

    def test_global_energy_balance_solid_hydrogen_and_unmodeled(self):
        geo,s,p=fixture()
        st=advance_conjugate_core(geo,s,p,.35)
        self.assertLess(st.thermal_relative_residual,1e-11)
        self.assertAlmostEqual(st.other_deposition_j,.35*geo.shape[0]*geo.shape[1]*2)
        self.assertAlmostEqual(st.deposited_total_j,.35*p.declared_total_w)
        self.assertTrue(st.synthetic)
        self.assertTrue(np.isfinite(st.internal_radial_w).all())
        self.assertTrue(np.isfinite(st.internal_axial_w).all())

    def test_signed_reverse_wall_transfer_permitted(self):
        geo,s,_=fixture(npath=1,nz=1,nr=2,flow=0,wall=100.)
        s=ConjugateCoreState(0.,np.full(geo.shape,300.),
                              np.full((1,1),450.))
        step=advance_conjugate_core(geo,s,heat(geo),.01)
        self.assertLess(step.wall_transfer_j,0.)
        self.assertGreater(step.state.fuel_temperature_k[0,0,0],300.)
        self.assertLess(step.state.hydrogen_temperature_k[0,0],450.)
        self.assertLess(step.thermal_relative_residual,1e-10)

    def test_sparse_solve_matches_independent_dense_linear_solution(self):
        geo,s,p=fixture(npath=2,nz=2,nr=3)
        sys=assemble_conjugate_core(geo,s,p,.04)
        expected=np.linalg.solve(sys.matrix.toarray(),sys.rhs)
        step=advance_conjugate_core(geo,s,p,.04)
        actual=np.r_[step.state.fuel_temperature_k.flatten(),
                     step.state.hydrogen_temperature_k.flatten()]
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(sys.matrix@actual,sys.rhs,rtol=1e-12)

    def test_branches_have_no_unphysical_cross_zone_transport(self):
        geo,s,p=fixture(npath=2,nz=3,nr=2)
        mat=assemble_conjugate_core(geo,s,p,.05).matrix.toarray()
        ns=2*3*2
        branch0=list(range(0,6))+list(range(ns,ns+3))
        branch1=list(range(6,12))+list(range(ns+3,ns+6))
        self.assertEqual(np.max(np.abs(mat[np.ix_(branch0,branch1)])),0.)
        self.assertEqual(np.max(np.abs(mat[np.ix_(branch1,branch0)])),0.)

    def test_constant_temperature_radial_effective_XS_is_volume_weighted(self):
        geo,s,_=fixture(npath=1,nz=1,nr=2)
        T=np.array([[[300.,900.]]])
        state=replace(s,fuel_temperature_k=T)
        volume=geo.solid_volumes_m3[0,0]
        expected=float(np.dot(volume,[300.,900.])/volume.sum())
        self.assertAlmostEqual(
           effective_solid_temperature_for_xs(geo,state)[0,0],expected,places=12)

    def test_internal_flux_integrals_cancel_not_counted_as_sources(self):
        geo,s,p=fixture(npath=1,nz=2,nr=3)
        out=advance_conjugate_core(geo,s,p,.05)
        self.assertGreater(np.max(np.abs(out.internal_radial_w)),0.)
        self.assertGreater(np.max(np.abs(out.internal_axial_w)),0.)
        # Only external boundary enthalpy and deposition appear in GLOBAL first law.
        self.assertAlmostEqual(
            out.fuel_storage_j+out.hydrogen_storage_j+
            out.outlet_net_enthalpy_j+out.other_deposition_j,
            out.deposited_total_j,
            delta=1e-8*max(1.,out.deposited_total_j))

    def test_nonuniform_properties_positive_solution(self):
        geo,s,p=fixture(npath=2,nz=3,nr=3)
        rho=np.asarray(geo.solid_density_kg_m3).copy()
        k=np.asarray(geo.solid_conductivity_w_m_k).copy()
        k[0,2,0]=.8;k[1,0,2]=60.;rho[0,1,2]=3100.
        geo=replace(geo,solid_density_kg_m3=rho,
                    solid_conductivity_w_m_k=k)
        out=advance_conjugate_core(geo,s,p,.2)
        self.assertTrue(np.all(out.state.fuel_temperature_k>0))
        self.assertTrue(np.all(out.state.hydrogen_temperature_k>0))
        self.assertLess(out.thermal_relative_residual,1e-10)

    def test_invalid_geometry_and_transport_coefficients_rejected(self):
        geo,s,p=fixture()
        with self.assertRaisesRegex(ValueError,"annulus"):
            replace(geo,fuel_radial_edges_m=np.array([[0.,.02,.03,.04],
                                                      [.01,.02,.03,.04]]))
        with self.assertRaisesRegex(ValueError,"channels"):
            replace(geo,channels_per_radial_zone=np.array([1.,1.5]))
        with self.assertRaisesRegex(ValueError,"axial"):
            replace(geo,axial_edges_m=np.array([0.,2.,1.,3.]))
        with self.assertRaises(ValueError):
            replace(geo,solid_conductivity_w_m_k=np.full(geo.shape,-2.))
        with self.assertRaises(ValueError):
            replace(geo,hydrogen_cp_j_kg_k=np.zeros(2))

    def test_invalid_heat_partition_and_time_rejected(self):
        geo,s,p=fixture()
        with self.assertRaisesRegex(ValueError,"partitions"):
            advance_conjugate_core(geo,s,replace(p,declared_total_w=1.),.1)
        with self.assertRaises(ValueError):
            advance_conjugate_core(geo,s,p,0.)
        with self.assertRaises(ValueError):
            advance_conjugate_core(geo,s,p,-.3)
        with self.assertRaises(ValueError):
            advance_conjugate_core(geo,s,
                 replace(p,solid_w=np.full(geo.shape,-1.)),.1)

    def test_independent_fixed_temperature_state_does_not_mutate_inputs(self):
        geo,s,p=fixture()
        old=s.fuel_temperature_k.copy()
        a=advance_conjugate_core(geo,s,p,.1)
        b=advance_conjugate_core(geo,s,p,.1)
        np.testing.assert_array_equal(a.state.fuel_temperature_k,b.state.fuel_temperature_k)
        np.testing.assert_array_equal(a.state.hydrogen_temperature_k,b.state.hydrogen_temperature_k)
        np.testing.assert_array_equal(s.fuel_temperature_k,old)


if __name__=="__main__":
    unittest.main(verbosity=2)
