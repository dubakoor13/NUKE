"""Independent synthetic verification of spatial temperature-table core inputs.

All temperatures/cross sections are made-up regression fixtures. This suite
does not certify evaluated neutron data, reactor hot-core temperatures,
Doppler worth, pressure/moderation feedback, or a NERVA prediction.
"""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
from scipy.linalg import eigvals

from .fx2th_rz_temperature_tables import (
    RZTemperatureTables, sample_rz_temperature_tables,
    evaluate_rz_temperature_direction,
)
from .fx2th_full_family_fission_eigen import (
    assemble_family_fission, solve_family_eigen,
)
from .validate_fx2th_full_family_fission_eigen import unequal_births
from .fx2th_rz_adjoint_sensitivity import evaluate_rz_material_sensitivity


def synthetic_tables() -> RZTemperatureTables:
    g,d=unequal_births(nr=2,nz=3,boundary="zero_flux")
    nr,nz,ng=g.diffusion_m.shape
    nf=d.families
    temperatures=np.array([300.,550.,850.,1150.])
    nt=len(temperatures)
    shape=(nt,nr,nz,ng)
    D=np.empty(shape)
    Sa=np.empty(shape)
    S=np.empty(shape+(ng,))
    Nu=np.empty(shape)
    Chi=np.empty(shape)
    Beta=np.empty(shape+(nf,))
    Chid=np.empty((nt,nr,nz,nf,ng))
    for it,t in enumerate(temperatures):
        f=(t-temperatures[0])/(temperatures[-1]-temperatures[0])
        for r in range(nr):
            for z in range(nz):
                w=1.+.1*r+.025*z
                D[it,r,z]=g.diffusion_m[r,z]*(1.+.13*f*w)
                Sa[it,r,z]=g.absorption_inv_m[r,z]*(1.+.07*f*w+
                                                    .018*f*f)
                S[it,r,z]=g.scattering_inv_m[r,z]*(1.-.09*f*w)
                Nu[it,r,z]=g.nu_fission_inv_m[r,z]*(1.-.05*f*w+
                                                     .013*f*f)
                Chi[it,r,z]=[.76+.045*f+.009*r-.006*z,
                              .24-.045*f-.009*r+.006*z]
                Beta[it,r,z]=d.beta_by_incident_group_and_family[
                    r*nz+z]*(1.+.08*f*w)
                Chid[it,r,z,0]=[.12+.03*f+.01*r,.88-.03*f-.01*r]
                Chid[it,r,z,1]=[.91-.05*f-.01*z,.09+.05*f+.01*z]
    return RZTemperatureTables(
        source_id="synthetic-rz-2group-temperature-table",
        source_kind="synthetic",
        nuclear_manifest_sha256="a"*64,
        reference_grid=g,
        temperature_knots_k=temperatures,
        energy_group_edges_ev=np.array([0.,1.,20.]),
        diffusion_m=D,
        absorption_inv_m=Sa,
        scattering_inv_m=S,
        nu_fission_inv_m=Nu,
        prompt_spectrum=Chi,
        beta_by_incident_group_and_family=Beta,
        delayed_spectra_by_family=Chid,
        group_speeds_m_s=d.group_speeds_m_s,
        decay_constants_s_inv=d.decay_constants_s_inv,
    )


def heterogeneous_temperature():
    return np.array([[415.,660.,970.],[377.,770.,1015.]])


def temperature_direction():
    return np.array([[1.,-.4,.6],[-.15,.5,1.3]])


def dense_keff(material):
    f=assemble_family_fission(material).total
    eigenvalues=eigvals(f.toarray(),material.integrated_loss.toarray())
    physical=[float(e.real) for e in eigenvalues if
              abs(e.imag)<1e-9 and e.real>0]
    return max(physical)


class TemperatureTableTests(unittest.TestCase):
    def setUp(self):
        self.tables=synthetic_tables()
        self.temperatures=heterogeneous_temperature()

    def test_sample_correct_shapes_and_cylindrical_cell_volumes(self):
        t=self.tables
        result=sample_rz_temperature_tables(t,self.temperatures)
        self.assertEqual(result.grid.diffusion_m.shape,(2,3,2))
        self.assertEqual(result.grid.scattering_inv_m.shape,(2,3,2,2))
        self.assertEqual(result.material.beta_by_incident_group_and_family.shape,(6,2,2))
        self.assertEqual(result.material.delayed_spectra_by_family.shape,(6,2,2))
        self.assertAlmostEqual(result.material.volumes_m3.sum(),2*np.pi,places=12)
        np.testing.assert_array_equal(result.temperature_k,self.temperatures)
        self.assertFalse(result.at_any_knot.any())
        self.assertEqual(result.source_kind,"synthetic")

    def test_cell_temperature_lookup_uses_its_own_interval(self):
        table=self.tables
        out=sample_rz_temperature_tables(table,self.temperatures)
        indexes=np.array([[0,1,2],[0,1,2]])
        np.testing.assert_array_equal(out.lower_knot_indices,indexes)
        for ir,iz in ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2)):
            j=indexes[ir,iz]
            Ti=self.temperatures[ir,iz]
            w=(Ti-table.temperature_knots_k[j])/(
                table.temperature_knots_k[j+1]-table.temperature_knots_k[j])
            expected=(1-w)*table.absorption_inv_m[j,ir,iz]+(
                w*table.absorption_inv_m[j+1,ir,iz])
            np.testing.assert_allclose(
                out.grid.absorption_inv_m[ir,iz],expected,rtol=0,atol=1e-14)

    def test_piecewise_linear_derivative_equals_independent_sample_fd(self):
        table=self.tables
        T=self.temperatures
        sampled=sample_rz_temperature_tables(table,T)
        plus=sample_rz_temperature_tables(table,T+.001)
        minus=sample_rz_temperature_tables(table,T-.001)
        props=[
           (sampled.slopes.diffusion_m_per_k,"diffusion_m"),
           (sampled.slopes.absorption_inv_m_per_k,"absorption_inv_m"),
           (sampled.slopes.scattering_inv_m_per_k,"scattering_inv_m"),
           (sampled.slopes.nu_fission_inv_m_per_k,"nu_fission_inv_m"),
           (sampled.slopes.prompt_spectrum_per_k,"fission_spectrum"),
        ]
        for slope,attr in props:
            with self.subTest(attr=attr):
                fd=(getattr(plus.grid,attr)-getattr(minus.grid,attr))/.002
                np.testing.assert_allclose(slope,fd,rtol=1e-8,atol=1e-11)
        for slope,attr in (
            (sampled.slopes.beta_by_incident_group_and_family_per_k,
             "beta_by_incident_group_and_family"),
            (sampled.slopes.delayed_spectra_by_family_per_k,
             "delayed_spectra_by_family"),
        ):
            fd=(getattr(plus.material,attr)-getattr(minus.material,attr))/.002
            np.testing.assert_allclose(
                slope.reshape(fd.shape),fd,rtol=1e-8,atol=1e-11)

    def test_all_interpolated_source_partitions_preserve_neutron_births(self):
        sampled=sample_rz_temperature_tables(self.tables,self.temperatures)
        d=sampled.material
        phi=np.arange(1,13,dtype=float).reshape(6,2)*.3
        matrices=assemble_family_fission(d)
        total=float((matrices.total@phi.flatten()).sum())
        reference=float((d.volumes_m3[:,None]*d.nu_sigma_f_inv_m*phi).sum())
        self.assertAlmostEqual(total,reference,places=11)
        self.assertAlmostEqual(
            total,float((matrices.prompt@phi.flatten()).sum())+
            sum(float((mat@phi.flatten()).sum())
                for mat in matrices.delayed_by_family),places=11)

    def test_all_normalized_spectra_and_beta_hold_at_mixed_temperatures(self):
        d=sample_rz_temperature_tables(self.tables,self.temperatures).material
        np.testing.assert_allclose(d.prompt_spectrum.sum(axis=1),1.,atol=1e-12)
        np.testing.assert_allclose(d.delayed_spectra_by_family.sum(axis=2),1.,atol=1e-12)
        self.assertTrue(np.all(d.beta_by_incident_group_and_family.sum(axis=2)<1))
        self.assertTrue(np.all(d.nu_sigma_f_inv_m>=0))
        self.assertTrue(np.all(d.group_speeds_m_s>0))

    def test_tables_interpolate_exactly_at_all_knots(self):
        table=self.tables
        for it,t in enumerate(table.temperature_knots_k):
            with self.subTest(t=t):
                sample=sample_rz_temperature_tables(table,
                             np.full(table.spatial_shape,t))
                np.testing.assert_allclose(sample.grid.diffusion_m,
                                           table.diffusion_m[it],rtol=0,atol=0)
                np.testing.assert_allclose(sample.grid.absorption_inv_m,
                                           table.absorption_inv_m[it],rtol=0,atol=0)
                np.testing.assert_allclose(sample.grid.fission_spectrum,
                                           table.prompt_spectrum[it],rtol=0,atol=0)
                self.assertTrue(sample.at_any_knot.all())

    def test_no_temperature_extrapolation(self):
        for x in (200.,1300.):
            with self.subTest(temperature=x),self.assertRaisesRegex(ValueError,"extrapolation"):
                sample_rz_temperature_tables(self.tables,
                                np.full(self.tables.spatial_shape,x))

    def test_temperature_knot_derivative_is_rejected_not_fabricated(self):
        t=self.temperatures.copy()
        t[0,0]=self.tables.temperature_knots_k[1]
        with self.assertRaisesRegex(ValueError,"knot"):
            evaluate_rz_temperature_direction(self.tables,t,
                                              np.ones_like(t))
        direction=np.ones_like(t);direction[0,0]=0.
        result=evaluate_rz_temperature_direction(self.tables,t,direction)
        self.assertTrue(np.isfinite(result.adjoint.d_keff_d_parameter))

    def test_zero_temperature_direction_is_exact_zero(self):
        result=evaluate_rz_temperature_direction(
            self.tables,self.temperatures,np.zeros_like(self.temperatures))
        self.assertEqual(result.adjoint.d_keff_d_parameter,0.)
        self.assertEqual(result.adjoint.d_reactivity_d_parameter,0.)

    def test_spatial_temperature_map_changes_discrete_nuclear_state(self):
        a=sample_rz_temperature_tables(self.tables,self.temperatures)
        colder=self.temperatures.copy();colder[1,1]-=60.
        b=sample_rz_temperature_tables(self.tables,colder)
        self.assertNotEqual(a.material.source_id,b.material.source_id)
        self.assertNotEqual(a.material.nu_sigma_f_inv_m[4,0],
                            b.material.nu_sigma_f_inv_m[4,0])
        self.assertEqual(a.material.nu_sigma_f_inv_m[0,0],
                         b.material.nu_sigma_f_inv_m[0,0])

    def test_temperature_direction_keff_matches_dense_recomputed_reference(self):
        table=self.tables;T=self.temperatures;u=temperature_direction()
        analytic=evaluate_rz_temperature_direction(table,T,u)
        h=.01
        plus=dense_keff(sample_rz_temperature_tables(table,T+h*u).material)
        minus=dense_keff(sample_rz_temperature_tables(table,T-h*u).material)
        fd=(plus-minus)/(2*h)
        np.testing.assert_allclose(analytic.adjoint.d_keff_d_parameter,fd,
                                   rtol=2e-5,atol=2e-8)

    def test_uniform_temperature_shift_derivative_matches_dense_reference(self):
        table=self.tables;T=self.temperatures;u=np.ones_like(T)
        analytic=evaluate_rz_temperature_direction(table,T,u)
        h=.01
        plus=dense_keff(sample_rz_temperature_tables(table,T+h).material)
        minus=dense_keff(sample_rz_temperature_tables(table,T-h).material)
        np.testing.assert_allclose(analytic.adjoint.d_keff_d_parameter,
                                  (plus-minus)/(2*h),rtol=2e-5,atol=2e-8)

    def test_reactivity_temperature_derivative_matches_dense_reference(self):
        table=self.tables;T=self.temperatures;u=temperature_direction()
        analytic=evaluate_rz_temperature_direction(table,T,u)
        h=.01
        kp=dense_keff(sample_rz_temperature_tables(table,T+h*u).material)
        km=dense_keff(sample_rz_temperature_tables(table,T-h*u).material)
        fd=((1.-1./kp)-(1.-1./km))/(2*h)
        np.testing.assert_allclose(analytic.adjoint.d_reactivity_d_parameter,
                                  fd,rtol=2e-5,atol=2e-8)

    def test_family_beta_and_birth_spectra_respond_to_temperature(self):
        table=self.tables
        sample=sample_rz_temperature_tables(table,self.temperatures)
        for derivative in (
            sample.slopes.beta_by_incident_group_and_family_per_k,
            sample.slopes.prompt_spectrum_per_k,
            sample.slopes.delayed_spectra_by_family_per_k,
        ):
            self.assertGreater(float(np.max(np.abs(derivative))),0.)
        np.testing.assert_allclose(
            sample.slopes.prompt_spectrum_per_k.sum(axis=-1),0.,
            rtol=0,atol=1e-12)
        np.testing.assert_allclose(
            sample.slopes.delayed_spectra_by_family_per_k.sum(axis=-1),0.,
            rtol=0,atol=1e-12)

    def test_only_one_cell_temperature_derivative_matches_dense(self):
        T=self.temperatures
        direction=np.zeros_like(T);direction[1,2]=1.
        analytic=evaluate_rz_temperature_direction(self.tables,T,direction)
        h=.1
        kp=dense_keff(sample_rz_temperature_tables(self.tables,T+h*direction).material)
        km=dense_keff(sample_rz_temperature_tables(self.tables,T-h*direction).material)
        np.testing.assert_allclose(analytic.adjoint.d_keff_d_parameter,
                                  (kp-km)/(2*h),rtol=4e-5,atol=2e-9)

    def test_tight_bracket_continuity_but_nonunique_derivative(self):
        table=self.tables
        node=table.temperature_knots_k[1]
        tmin=np.full(table.spatial_shape,node-1e-6)
        tmax=np.full(table.spatial_shape,node+1e-6)
        a=sample_rz_temperature_tables(table,tmin).grid
        b=sample_rz_temperature_tables(table,tmax).grid
        self.assertLess(np.max(np.abs(a.absorption_inv_m-b.absorption_inv_m)),1e-7)
        with self.assertRaisesRegex(ValueError,"knot"):
            evaluate_rz_temperature_direction(table,np.full(table.spatial_shape,node),
                                             np.ones(table.spatial_shape))

    def test_change_of_table_manifest_changes_state_provenance(self):
        table=self.tables
        a=sample_rz_temperature_tables(table,self.temperatures)
        alt=replace(table,nuclear_manifest_sha256="b"*64)
        b=sample_rz_temperature_tables(alt,self.temperatures)
        self.assertNotEqual(a.material.source_id,b.material.source_id)
        self.assertEqual(a.nuclear_manifest_sha256,"a"*64)
        self.assertEqual(b.nuclear_manifest_sha256,"b"*64)

    def test_source_kind_externally_supplied_does_not_certify_data(self):
        table=replace(self.tables,source_kind="externally-tabulated")
        a=sample_rz_temperature_tables(table,self.temperatures)
        self.assertEqual(a.source_kind,"externally-tabulated")
        self.assertIsInstance(a.material.source_id,str)

    def test_fail_closed_invalid_temperature_mesh_and_direction(self):
        table=self.tables
        for temps in (np.ones((3,2))*500.,
                      np.array([[500.,np.nan,700.],[600.,800.,900.]]),
                      np.zeros((2,3))):
            with self.subTest(temps=temps),self.assertRaises(ValueError):
                sample_rz_temperature_tables(table,temps)
        for u in (np.ones((1,3)),np.ones((2,3))*np.inf):
            with self.subTest(direction=u),self.assertRaises(ValueError):
                evaluate_rz_temperature_direction(table,self.temperatures,u)

    def test_reject_nonmonotonic_knots_and_invalid_e_group_edges(self):
        table=self.tables
        with self.assertRaisesRegex(ValueError,"knots"):
            replace(table,temperature_knots_k=np.array([300.,650.,550.,1000.]))
        with self.assertRaisesRegex(ValueError,"energy group edges"):
            replace(table,energy_group_edges_ev=np.array([0.,1.,.1]))

    def test_reject_invalid_scattering_and_diffusion(self):
        table=self.tables
        D=np.asarray(table.diffusion_m).copy();D[0,0,0,0]=-1.
        with self.assertRaises(ValueError):
            replace(table,diffusion_m=D)
        S=np.asarray(table.scattering_inv_m).copy();S[0,0,0,0,0]=.2
        with self.assertRaisesRegex(ValueError,"self-scattering"):
            replace(table,scattering_inv_m=S)

    def test_reject_unphysical_beta_and_spectrum_normalization(self):
        table=self.tables
        beta=np.asarray(table.beta_by_incident_group_and_family).copy()
        beta[0,0,0,0,:]=[.8,.4]
        with self.assertRaisesRegex(ValueError,"beta"):
            replace(table,beta_by_incident_group_and_family=beta)
        chip=np.asarray(table.prompt_spectrum).copy()
        chip[0,0,0,0]+=.2
        with self.assertRaisesRegex(ValueError,"normalized"):
            replace(table,prompt_spectrum=chip)

    def test_declared_manifest_hash_requires_64_hex_digits(self):
        with self.assertRaisesRegex(ValueError,"SHA256"):
            replace(self.tables,nuclear_manifest_sha256="not-a-hash")

    def test_real_family_decay_rates_stay_separate_from_temperature(self):
        result=sample_rz_temperature_tables(self.tables,self.temperatures)
        np.testing.assert_array_equal(result.material.decay_constants_s_inv,
                                      self.tables.decay_constants_s_inv)
        self.assertEqual(len(result.material.decay_constants_s_inv),2)


if __name__=="__main__":
    unittest.main(verbosity=2)
