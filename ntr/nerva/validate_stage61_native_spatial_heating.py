"""Stage61 exact spatial source-to-thermal mapping acceptance tests.

Native-format HDF5 is SYNTHETIC; source scores do NOT assert actual OpenMC
transport, selected reactor fuel-element identity, enrichment, power or a
NASA-benchmarked NERVA operating point.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from .validate_stage58_fuel_hydrogen_coupling import coupled,initial,solve
from .validate_stage60_native_hdf5_heating import (
    synthetic_statepoint,FUEL_ID,DATA_SHA,
)
from .openmc_native_fuel_statepoint import (
    read_openmc_neutron_fuel_statepoint,project_native_fuel_heating,
)
from .openmc_native_spatial_fuel_heating import (
    project_native_3d_fuel_heating,
)
from .openmc_fuel_axial_heat_handoff import check_heating_for_coupled_core


def local_native(model, *, dim=(2,3,4), rate=1e18):
    """Construct native-shaped HDF5 with mesh bounding ONE thermal fuel hex."""
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/"statepoint.8.h5"
        synthetic_statepoint(path,dimension=dim)
        t=model.solid.transverse
        with h5py.File(path,"a") as f:
            mesh=f["tallies/meshes/mesh 19"]
            mesh["lower_left"][...]=[t.x_edges_cm[0],
                                    t.y_edges_cm[0],-66.]
            mesh["upper_right"][...]=[t.x_edges_cm[-1],
                                      t.y_edges_cm[-1],66.]
        return read_openmc_neutron_fuel_statepoint(
            path,expected_fuel_material_id=FUEL_ID,
            nuclear_data_sha256=DATA_SHA,
            source_rate_particles_per_s=rate,
            source_rate_basis="SYNTHETIC stage61 physical-volume numerical test",
            photon_transport_confirmed_disabled=True,
            expected_active_length_m=1.32)


class ExactSpatialFuelHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=coupled(film=0.,nz=4,nx=20,ny=24)
        cls.native=local_native(cls.m)

    def test_native_voxel_to_fuel_cutcell_energy_conservation(self):
        m=self.m
        native=self.native
        r=project_native_3d_fuel_heating(
            m.config,native,m.solid,
            isolated_element_geometry_confirmed=True)
        source=float(np.sum(native.source_mesh_mean_ev_per_source))
        expected=source*native.score.source_rate_particles_per_s*1.602176634e-19
        self.assertAlmostEqual(r.resolved.mapped_fuel_power_w,expected,delta=1e-11)
        self.assertLess(abs(r.resolved.fuel_power_balance_residual_w),1e-11)
        np.testing.assert_allclose(
            r.resolved.fuel_deposition_w.sum(axis=1),
            native.score.fuel_neutron_ev_per_source*
              native.score.source_rate_particles_per_s*1.602176634e-19,
            atol=1e-10,rtol=0)
        self.assertEqual(r.resolved.fuel_deposition_w.shape,
                         (4,m.solid.transverse.n_fuel))
        self.assertIn("XY distribution retained",
                      r.resolved.transverse_approximation)

    def test_xy_area_weights_form_exact_partition_of_all_source_fuel(self):
        r=project_native_3d_fuel_heating(
            self.m.config,self.native,self.m.solid,
            isolated_element_geometry_confirmed=True)
        weights=r.source_xy_to_fuel_weights
        mat_area=self.m.solid.transverse.fuel_areas_cm2
        source_area=r.source_xy_fuel_area_cm2.ravel()
        active=source_area>1e-14
        np.testing.assert_allclose(
            np.asarray(weights.sum(axis=0)).ravel()[active],
            1.,rtol=0,atol=2e-10)
        np.testing.assert_allclose(
            weights@source_area,mat_area,
            rtol=2e-10,atol=2e-10)
        self.assertAlmostEqual(float(np.sum(source_area)),
                               float(np.sum(mat_area)),delta=2e-10)

    def test_native_3d_preserves_genuine_transverse_profile_vs_old_uniform_map(self):
        native=self.native
        new=project_native_3d_fuel_heating(
            self.m.config,native,self.m.solid,
            isolated_element_geometry_confirmed=True)
        old=project_native_fuel_heating(native,self.m.solid)
        self.assertAlmostEqual(
            new.resolved.mapped_fuel_power_w,
            old.resolved.mapped_fuel_power_w,delta=1e-11)
        # Stage60/59 always imposed a single uniform fuel-volume heating
        # density regardless of the source x/y profile. Stage61 retains it.
        self.assertGreater(
            float(np.max(np.abs(
                new.resolved.fuel_deposition_w-old.resolved.fuel_deposition_w))),
            1e-5)
        cx=self.m.solid.transverse.cell_centers_cm[:,0]
        left=new.resolved.fuel_deposition_w[:,cx<0].sum()
        right=new.resolved.fuel_deposition_w[:,cx>0].sum()
        self.assertGreater(right,left)

    def test_axial_rebin_same_3d_energy_with_alternate_target_slabs(self):
        m=coupled(film=0.,nz=3,nx=20,ny=24)
        native=local_native(m,dim=(2,3,4))
        projection=project_native_3d_fuel_heating(
            m.config,native,m.solid,
            isolated_element_geometry_confirmed=True)
        self.assertEqual(projection.resolved.fuel_deposition_w.shape[0],3)
        self.assertAlmostEqual(
            projection.resolved.mapped_fuel_power_w,
            float(native.source_mesh_mean_ev_per_source.sum())*1e18*
            1.602176634e-19,delta=1e-11)
        self.assertLess(abs(projection.resolved.fuel_power_balance_residual_w),
                        1e-11)

    def test_3d_mesh_marginal_uncertainty_bound_uses_same_intersections(self):
        result=project_native_3d_fuel_heating(
            self.m.config,self.native,self.m.solid,
            isolated_element_geometry_confirmed=True)
        expected=(float(np.sum(
            self.native.source_mesh_stderr_ev_per_source))*
            1e18*1.602176634e-19)
        self.assertAlmostEqual(result.total_uncertainty_bound_w,
                               expected,delta=1e-11)
        self.assertAlmostEqual(
            float(np.sum(result.fuel_cell_uncertainty_bound_w)),
            expected,delta=1e-11)
        self.assertTrue(np.all(result.fuel_cell_uncertainty_bound_w>=0))
        self.assertIn("source voxel",result.local_heating_approximation)

    def test_reject_entire_reactor_mesh_remapped_into_one_fuel_hex(self):
        # Source has the same (nx,ny,nz) array SHAPE, but deliberately
        # spans the whole reactor and cannot belong to just one hex.
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"statepoint.8.h5"
            synthetic_statepoint(path)
            native=read_openmc_neutron_fuel_statepoint(
                path,expected_fuel_material_id=FUEL_ID,
                nuclear_data_sha256=DATA_SHA,
                source_rate_particles_per_s=1e18,
                source_rate_basis="SYNTHETIC whole-core mismatch fixture",
                photon_transport_confirmed_disabled=True)
        with self.assertRaisesRegex(ValueError,"whole-core"):
            project_native_3d_fuel_heating(
                self.m.config,native,self.m.solid,
                isolated_element_geometry_confirmed=True)

    def test_caller_must_confirm_one_element_source_scope(self):
        with self.assertRaisesRegex(ValueError,"independently confirmed"):
            project_native_3d_fuel_heating(
                self.m.config,self.native,self.m.solid,
                isolated_element_geometry_confirmed=False)

    def test_3d_source_metadata_must_match_native_axial_tally(self):
        wrong=np.asarray(self.native.source_mesh_mean_ev_per_source).copy()
        wrong[0,0,0]*=1.25
        bad=replace(self.native,source_mesh_mean_ev_per_source=wrong)
        with self.assertRaisesRegex(ValueError,"disagree"):
            project_native_3d_fuel_heating(
                self.m.config,bad,self.m.solid,
                isolated_element_geometry_confirmed=True)

    def test_outside_fuel_voxel_must_not_carry_nonzero_fuel_heating(self):
        native=local_native(self.m,dim=(12,12,4))
        # Source corners have no fuel where the inner hex is truncated.
        with self.assertRaisesRegex(ValueError,"no fuel"):
            project_native_3d_fuel_heating(
                self.m.config,native,self.m.solid,
                isolated_element_geometry_confirmed=True)

    def test_new_source_watts_feed_stage58_same_correct_geometry(self):
        m=self.m
        heat=project_native_3d_fuel_heating(
            m.config,self.native,m.solid,
            isolated_element_geometry_confirmed=True)
        fuel_watts=check_heating_for_coupled_core(heat.resolved,m)
        r=solve(m,initial(m,tf=400.,th=400.),dt=1e-5,q=fuel_watts)
        self.assertAlmostEqual(
            r.fuel_deposition_j,
            heat.resolved.mapped_fuel_power_w*1e-5,delta=2e-11)
        self.assertLess(r.relative_first_law_residual,2e-9)
        self.assertEqual(float(np.sum(r.wall_heat_to_hydrogen_j)),0.)

    def test_true_3d_shapes_and_stderr_sign_fail_closed(self):
        native=self.native
        bad=replace(native,source_mesh_stderr_ev_per_source=
                    np.full((2,3,4),-1.))
        with self.assertRaisesRegex(ValueError,"native 3D mean"):
            project_native_3d_fuel_heating(
                self.m.config,bad,self.m.solid,
                isolated_element_geometry_confirmed=True)
        bad=replace(native,source_mesh_mean_ev_per_source=np.zeros((1,3,4)))
        with self.assertRaisesRegex(ValueError,"native 3D mean"):
            project_native_3d_fuel_heating(
                self.m.config,bad,self.m.solid,
                isolated_element_geometry_confirmed=True)


if __name__=="__main__":
    unittest.main(verbosity=2)
