"""Stage80: analytic affine-field and fully coupled curved-wall checks.

Stage80 is a DEFERRED nonorthogonal heat-flux correction, not proof of
second-order accuracy or validated NERVA reactor thermal performance.
All physical heating/property inputs in mesh runs are synthetic.
"""
from __future__ import annotations

from dataclasses import replace
import unittest
import numpy as np

from .fuel_wall_nonorthogonal import (
    build_fuel_gradient_reconstruction,reconstruct_fuel_gradient,
    tangential_wall_temperature_correction_k,
)
from .coating_angular_interfaces import nonorthogonal_fuel_wall_correction_k
from .coating_radial_interfaces import radial_interface_power
from .fuel_cutcell_xy_convergence import run_fuel_xy_convergence
from .coating_mesh_convergence import CoupledMeshResolution
from .validate_stage76_coupled_mesh_convergence import (
    BASE,HEAT,physical_fixture,
)
from .validate_stage79_fuel_wall_moments import moment_factory
from .validate_stage78_embedded_wall_distance import GRIDS


def corrected_factory(mesh:CoupledMeshResolution):
    m,state=physical_fixture(mesh)
    reconstruction=build_fuel_gradient_reconstruction(m.config,m.solid)
    m=replace(m,
        coating=replace(m.coating,
            fuel_wall_distance_model="moment_nonorthogonal_gradient"),
        angular_contacts=replace(
            m.angular_contacts,
            fuel_gradient_reconstruction=reconstruction))
    return m,state


def static_wall_flux(model,gradient_x=0.,gradient_y=0.):
    xy=model.angular_contacts.fuel_gradient_reconstruction.centroids_cm
    tf=(500.+gradient_x*.01*xy[:,0]+gradient_y*.01*xy[:,1])
    fuel=np.broadcast_to(
        tf[None,:],(model.nzone,model.solid.transverse.n_fuel)).copy()
    coating=np.full(model.radial_coating.sector_mass_kg.shape,460.)
    hydrogen=np.full((model.nzone,19),400.)
    return radial_interface_power(
        model.coating,model.solid,model.properties,
        model.angular_contacts,model.radial_coating,
        fuel,coating,hydrogen)


class NonorthogonalFuelWallCorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model,cls.initial=corrected_factory(BASE)
        cls.full=run_fuel_xy_convergence(
            corrected_factory,base_mesh=BASE,heating=HEAT,xy_sizes=GRIDS)

    def test_actual_material_centroids_and_sparse_linear_gradient(self):
        m=self.model
        op=m.angular_contacts.fuel_gradient_reconstruction
        xy=op.centroids_cm
        self.assertEqual(xy.shape,(m.solid.transverse.n_fuel,2))
        self.assertLess(op.exact_material_area_residual_cm2,1e-10)
        self.assertGreaterEqual(op.minimum_neighbours,2)
        self.assertLess(op.maximum_gradient_condition,1e9)
        self.assertTrue(np.all(np.isfinite(xy)))
        self.assertEqual(op.gradient_x_m_inv.shape,(len(xy),len(xy)))

    def test_manufactured_two_dimensional_affine_temperature_gradient_exact(self):
        op=self.model.angular_contacts.fuel_gradient_reconstruction
        xy=op.centroids_cm*.01
        true=(17.3,-9.4)
        t=430.+true[0]*xy[:,0]+true[1]*xy[:,1]
        state=np.vstack((t,t+31.))
        x,y=reconstruct_fuel_gradient(op,state)
        np.testing.assert_allclose(x,np.full_like(x,true[0]),
                                   atol=3e-8,rtol=0)
        np.testing.assert_allclose(y,np.full_like(y,true[1]),
                                   atol=3e-8,rtol=0)

    def test_manufactured_constant_field_yields_no_spurious_wall_correction(self):
        m=self.model
        t=np.full((m.nzone,m.solid.transverse.n_fuel),455.)
        rim=m.angular_contacts.fuel_outer_full_m2.tocoo()
        c=nonorthogonal_fuel_wall_correction_k(
            m.angular_contacts,t,rim.row,rim.col)
        np.testing.assert_allclose(c,0.,atol=3e-11,rtol=0)

    def test_affine_nonorthogonal_jump_exactly_removes_tangential_component(self):
        m=self.model
        op=m.angular_contacts.fuel_gradient_reconstruction
        xy=op.centroids_cm*.01
        a,b=1200.,-700.
        t=np.tile((450.+a*xy[:,0]+b*xy[:,1]),(m.nzone,1))
        rim=m.angular_contacts.fuel_outer_full_m2.tocoo()
        corr=nonorthogonal_fuel_wall_correction_k(
            m.angular_contacts,t,rim.row,rim.col)
        contact=m.angular_contacts
        nx=np.asarray(contact.fuel_wall_mean_normal_x[
            rim.row,rim.col]).ravel()
        ny=np.asarray(contact.fuel_wall_mean_normal_y[
            rim.row,rim.col]).ravel()
        tang=np.asarray(contact.fuel_wall_tangential_offset_m[
            rim.row,rim.col]).ravel()
        normalization=np.hypot(nx,ny)
        analytic=(-a*ny+b*nx)/normalization*tang
        np.testing.assert_allclose(
            corr,np.broadcast_to(analytic[None,:],corr.shape),
            atol=3e-10,rtol=0)
        self.assertGreater(float(np.max(np.abs(corr))),1e-5)

    def test_normal_and_tangential_projection_reconstruct_affine_cell_wall_jump(self):
        contact=self.model.angular_contacts
        rim=contact.fuel_outer_full_m2.tocoo()
        n_x=np.asarray(contact.fuel_wall_mean_normal_x[rim.row,rim.col]).ravel()
        n_y=np.asarray(contact.fuel_wall_mean_normal_y[rim.row,rim.col]).ravel()
        scale=np.hypot(n_x,n_y)
        ux=n_x/scale
        uy=n_y/scale
        normal=np.asarray(contact.fuel_wall_signed_normal_depth_m[
            rim.row,rim.col]).ravel()
        tangent=np.asarray(contact.fuel_wall_tangential_offset_m[
            rim.row,rim.col]).ravel()
        ax,ay=2000.,-1200.
        linear=(ax*ux+ay*uy)*normal+(-ax*uy+ay*ux)*tangent
        residual=linear-(-ax*uy+ay*ux)*tangent
        np.testing.assert_allclose(residual,
            (ax*ux+ay*uy)*normal,atol=2e-12,rtol=0)

    def test_nonorthogonal_model_conserves_signed_fuel_to_coating_heat(self):
        p=static_wall_flux(self.model,gradient_x=700.,gradient_y=-500.)
        self.assertLess(abs(p.fuel_antisymmetric_residual_w),1e-8)
        self.assertLess(abs(p.hydrogen_antisymmetric_residual_w),1e-8)
        np.testing.assert_allclose(
            p.fuel_to_coating_by_fuel_w.sum(axis=1),
            p.fuel_to_coating_by_radial_w.sum(axis=(1,2,3)),
            atol=1e-8,rtol=0)

    def test_three_actual_full_coupled_fuel_coating_h2_meshes_close_first_law(self):
        s=self.full
        self.assertEqual(len(s.cases),3)
        self.assertLess(s.max_relative_energy_residual,5e-8)
        self.assertLess(s.max_nuclear_source_rebin_difference_j,2e-11)
        self.assertLess(s.max_wall_contact_discrepancy_m2,4e-11)
        for case in s.cases:
            self.assertLess(abs(
                case.observation.coating_first_law_residual_j),4e-7)
            self.assertGreater(
                case.observation.measurements[
                    "fuel_outer_wall_peak_abs_w_m2"],0.)

    def test_fixed_physical_nuclear_source_and_geometry_across_all_meshes(self):
        s=self.full
        vols=[c.fuel_volume_m3 for c in s.cases]
        self.assertLess(max(vols)-min(vols),5e-12)
        sources=[c.observation.total_nuclear_energy_j for c in s.cases]
        self.assertLess(max(sources)-min(sources),1e-8)
        self.assertIn("fuel_outer_wall_peak_abs_w_m2",s.metrics)

    def test_missing_gradient_geometry_fails_closed(self):
        m,state=self.model,self.initial
        without=replace(m.angular_contacts,
                        fuel_gradient_reconstruction=None)
        broken=replace(m,angular_contacts=without)
        from .fuel_hydrogen_coupled_transient import advance_coupled_core
        with self.assertRaisesRegex(ValueError,"actual-centroid fuel LS"):
            advance_coupled_core(
                broken,state,dt_s=1e-6,
                fuel_deposition_w=np.zeros_like(state.fuel_temperature_k))

    def test_previous_centroid_normal_tpfa_remains_available(self):
        from .validate_stage79_fuel_wall_moments import moment_factory
        model,state=moment_factory(BASE)
        self.assertEqual(
            model.coating.fuel_wall_distance_model,"moment_normal_depth")
        self.assertIsNone(
            model.angular_contacts.fuel_gradient_reconstruction)
        from .validate_stage58_fuel_hydrogen_coupling import solve
        result=solve(model,state,dt=5e-6)
        self.assertLess(result.relative_first_law_residual,5e-8)

    def test_invalid_gradient_state_and_normal_rejected(self):
        op=self.model.angular_contacts.fuel_gradient_reconstruction
        with self.assertRaisesRegex(ValueError,"fuel gradient state"):
            reconstruct_fuel_gradient(op,np.ones((2,8)))
        gx=np.ones((2,4))
        with self.assertRaisesRegex(ValueError,"zero average"):
            tangential_wall_temperature_correction_k(
                gx,gx,np.array([0]),np.array([.02]),
                np.array([0.]),np.array([0.]))
        with self.assertRaisesRegex(ValueError,"fuel nonorthogonal"):
            tangential_wall_temperature_correction_k(
                gx,gx,np.array([0]),np.array([.02,.03]),
                np.array([1.]),np.array([0.]))


if __name__=="__main__":
    unittest.main(verbosity=2)
