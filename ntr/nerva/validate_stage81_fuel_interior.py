"""Stage81: independent affine Fourier checks and full coupled XY regressions.

The internal fuel flux must reproduce exact Fourier heat through EACH
physical open face for a manufactured T(x,y)=T0+a*x+b*y with constant k.
Only then compare the same physical synthetic Stage75 nuclear W/m3 and
H2 state on 3 real cutcell meshes.

NOTE: Exact affine face flux is a consistency check, NOT proof of
asymptotic convergence of a curved-hole manufactured heat PDE.
"""
from __future__ import annotations

from dataclasses import replace
import unittest
import numpy as np

from .fuel_interior_nonorthogonal import (
    build_interior_fuel_face_correction,
    nonorthogonal_face_heat_w,
    conservative_face_divergence_w,
)
from .fuel_wall_nonorthogonal import reconstruct_fuel_gradient
from .fuel_cutcell_xy_convergence import run_fuel_xy_convergence
from .coating_mesh_convergence import CoupledMeshResolution
from .validate_stage80_nonorthogonal_fuel import corrected_factory
from .validate_stage78_embedded_wall_distance import GRIDS
from .validate_stage76_coupled_mesh_convergence import BASE,HEAT
from .fuel_hydrogen_coupled_transient import (
    with_resolved_coating,advance_coupled_core,
)
from .validate_stage58_fuel_hydrogen_coupling import coupled


def full_factory(mesh:CoupledMeshResolution):
    m,state=corrected_factory(mesh)
    operator=build_interior_fuel_face_correction(
        m.solid,m.angular_contacts.fuel_gradient_reconstruction)
    m=replace(
        m,
        coating=replace(
            m.coating,fuel_wall_distance_model="moment_full_nonorthogonal"),
        interior_fuel_faces=operator)
    return m,state


class ConservativeInteriorFuelNonorthogonalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.state=full_factory(BASE)
        cls.study=run_fuel_xy_convergence(
            full_factory,base_mesh=BASE,heating=HEAT,xy_sizes=GRIDS)

    def test_every_accepted_face_has_real_centroid_normal_gap_and_positive_area(self):
        m=self.model
        f=m.interior_fuel_faces
        self.assertGreater(f.face_count,100)
        self.assertTrue(np.all(f.normal_distance_m>0.))
        self.assertTrue(np.all(f.physical_face_full_area_m2>0.))
        self.assertTrue(np.all(f.modified_full_length_geometry_m>0.))
        self.assertTrue(np.all(np.isfinite(f.tangential_offset_m)))
        self.assertGreater(float(np.max(np.abs(f.tangential_offset_m))),1e-6)
        self.assertEqual(
            len(f.physical_face_full_area_m2),
            len(m.solid.transverse.lateral_face_conductance_w_k))
        self.assertTrue(np.all((f.normal_x+f.normal_y)==1.))

    def test_reference_stage53_open_face_area_is_preserved_exactly(self):
        m=self.model
        f=m.interior_fuel_faces
        base=m.solid.transverse
        nx=f.normal_x.astype(bool)
        ny=f.normal_y.astype(bool)
        spacing=np.where(
            nx,np.diff(base.x_edges_cm)[0],
            np.diff(base.y_edges_cm)[0])*.01
        np.testing.assert_allclose(
            f.physical_face_full_area_m2,
            f.reference_full_length_geometry_m*spacing,
            atol=2e-16,rtol=3e-15)
        np.testing.assert_allclose(
            f.modified_full_length_geometry_m*f.normal_distance_m,
            f.physical_face_full_area_m2,
            atol=2e-16,rtol=3e-15)

    def test_manufactured_affine_fourier_flux_is_exact_at_every_skew_face(self):
        m=self.model
        f=m.interior_fuel_faces
        centroid=f.gradient.centroids_cm*.01
        grad_x=1800.
        grad_y=-1320.
        tf=(470.+grad_x*centroid[:,0]+grad_y*centroid[:,1])
        temperatures=np.vstack((tf,tf+9.))
        k=np.full_like(temperatures,19.)
        thickness=np.array([1.,1.])
        q,skew,g=nonorthogonal_face_heat_w(
            f,temperatures,k,thickness)
        analytic=-19.*f.physical_face_full_area_m2*(
            grad_x*f.normal_x+grad_y*f.normal_y)
        np.testing.assert_allclose(
            q,np.broadcast_to(analytic,q.shape),atol=1e-8,rtol=1e-10)
        self.assertGreater(float(np.max(np.abs(skew))),1e-5)
        self.assertTrue(np.all(g>0.))

    def test_affine_two_point_only_has_real_nonorthogonal_error(self):
        m=self.model
        f=m.interior_fuel_faces
        centroid=f.gradient.centroids_cm*.01
        k=np.full((1,m.solid.transverse.n_fuel),20.)
        temp=(450.+2500.*centroid[:,0]-1300.*centroid[:,1])[None,:]
        corrected,skew,g=nonorthogonal_face_heat_w(
            f,temp,k,np.array([1.]))
        uncorrected=g*(temp[:,f.face_cell_i]-temp[:,f.face_cell_j])
        self.assertGreater(
            float(np.max(np.abs(corrected-uncorrected))),1e-5)
        np.testing.assert_allclose(corrected-uncorrected,skew,
                                   atol=4e-14,rtol=1e-11)

    def test_deferred_correction_is_exactly_energy_antisymmetric(self):
        m=self.model
        f=m.interior_fuel_faces
        xy=f.gradient.centroids_cm*.01
        temp=(450.+600.*xy[:,0]-300.*xy[:,1])[None,:]
        t=np.vstack((temp[0],temp[0]+10.))
        k=np.full_like(t,19.)
        q,correction,g=nonorthogonal_face_heat_w(
            f,t,k,np.array([.5,.5]))
        rhs=conservative_face_divergence_w(
            f,correction,m.solid.transverse.n_fuel)
        self.assertLess(float(np.max(np.abs(rhs.sum(axis=1)))),1e-10)
        total=conservative_face_divergence_w(
            f,q,m.solid.transverse.n_fuel)
        self.assertLess(float(np.max(np.abs(total.sum(axis=1)))),1e-10)

    def test_constant_temperature_has_zero_internal_heat_and_skew(self):
        m=self.model
        f=m.interior_fuel_faces
        t=np.full((2,m.solid.transverse.n_fuel),450.)
        k=np.full_like(t,19.)
        q,c,g=nonorthogonal_face_heat_w(
            f,t,k,np.array([.5,.5]))
        np.testing.assert_allclose(q,0.,atol=3e-12,rtol=0)
        np.testing.assert_allclose(c,0.,atol=3e-12,rtol=0)

    def test_valid_real_geometry_on_three_true_embedded_xy_meshes(self):
        for nx,ny in GRIDS:
            m,_=full_factory(CoupledMeshResolution(
                nx,ny,BASE.axial_zones,
                BASE.angular_sectors,BASE.radial_layers))
            faces=m.interior_fuel_faces
            self.assertEqual(
                faces.face_count,
                len(m.solid.transverse.lateral_face_pairs))
            self.assertTrue(np.all(np.isfinite(
                faces.modified_full_length_geometry_m)))
            self.assertGreater(faces.face_count,100)
            self.assertLess(faces.gradient.maximum_gradient_condition,1e9)

    def test_full_coupled_three_material_energy_and_source_ledger_closes(self):
        s=self.study
        self.assertLess(s.max_relative_energy_residual,5e-8)
        self.assertLess(s.max_wall_contact_discrepancy_m2,4e-11)
        self.assertLess(s.max_fuel_inventory_discrepancy_m3,5e-12)
        self.assertLess(s.max_nuclear_source_rebin_difference_j,2e-11)
        for case in s.cases:
            self.assertLess(
                abs(case.observation.coating_first_law_residual_j),4e-7)
            self.assertGreater(
                case.observation.measurements["fuel_to_coating_energy_j"],0.)

    def test_same_W_per_volume_on_all_three_fuel_xy_discretizations(self):
        s=self.study
        energy=[a.observation.total_nuclear_energy_j for a in s.cases]
        self.assertLess(max(energy)-min(energy),1e-8)
        self.assertEqual(s.sizes,GRIDS)
        for metric in ("fuel_peak_k","fuel_outer_wall_peak_abs_w_m2",
                       "fuel_to_coating_energy_j"):
            diff=s.metrics[metric]
            self.assertGreaterEqual(diff.relative_mid_to_fine,0.)
            self.assertTrue(diff.interpretation)

    def test_public_opt_in_builds_full_interior_and_wall_contact_operator(self):
        base=coupled(film=1200.,nz=2,nx=20,ny=24)
        m=self.model
        new=with_resolved_coating(
            base,density_kg_m3=5000.,
            heat_capacity=m.coating.heat_capacity,
            contact_resistance_m2_k_w=1e-5,
            axial_conduction_enabled=True,
            angular_sectors=8,radial_layers=2,
            fuel_wall_distance_model="moment_full_nonorthogonal")
        self.assertEqual(
            new.coating.fuel_wall_distance_model,
            "moment_full_nonorthogonal")
        self.assertIsNotNone(new.interior_fuel_faces)
        self.assertIsNotNone(
            new.angular_contacts.fuel_gradient_reconstruction)

    def test_older_wall_only_mode_does_not_change_internal_fuel_operator(self):
        old,st=corrected_factory(BASE)
        self.assertIsNone(old.interior_fuel_faces)
        new=self.model
        self.assertIsNotNone(new.interior_fuel_faces)
        self.assertEqual(
            old.solid.transverse.lateral_face_pairs.shape,
            new.solid.transverse.lateral_face_pairs.shape)

    def test_nonphysical_interior_gradient_face_input_fails_closed(self):
        f=self.model.interior_fuel_faces
        n=self.model.solid.transverse.n_fuel
        with self.assertRaisesRegex(ValueError,"nonorthogonal fuel face"):
            nonorthogonal_face_heat_w(
                f,np.ones((2,n)),np.ones((2,n-1)),np.ones(2))
        with self.assertRaisesRegex(ValueError,"nonorthogonal fuel face"):
            nonorthogonal_face_heat_w(
                f,np.ones((2,n)),np.ones((2,n)),np.array([-1.,1.]))
        with self.assertRaisesRegex(ValueError,"invalid face heat"):
            conservative_face_divergence_w(f,np.ones((2,8)),n)


if __name__=="__main__":
    unittest.main(verbosity=2)
