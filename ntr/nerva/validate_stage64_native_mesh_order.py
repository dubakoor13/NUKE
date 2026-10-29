"""Stage64 independent source-level OpenMC mesh bin indexing acceptance.

Exact upstream OpenMC src/mesh.cpp:
  StructuredMesh::get_bin_from_indices({i+1,j+1,k+1})
    = ((k*ny)+j)*nx+i
so X is FASTEST. The core HDF5 parser and 3D fuel cut-cell projection must
use this indexing, or a tally can conserve total nuclear heat while
depositing it into the WRONG axial slice and transverse fuel location.

HDF5 inputs here use SYNTHETIC scores but encode actual C++ mesh bin
indexing explicitly, not a shared helper with the production parser.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from .validate_stage60_native_hdf5_heating import (
    synthetic_statepoint,FUEL_ID,DATA_SHA,
)
from .validate_stage58_fuel_hydrogen_coupling import coupled
from .openmc_native_fuel_statepoint import read_openmc_neutron_fuel_statepoint
from .openmc_native_spatial_fuel_heating import project_native_3d_fuel_heating
from .openmc_fuel_axial_heat_handoff import EV_TO_J


def native_bin(i: int,j: int,k: int,nx: int,ny: int) -> int:
    """Independent literal transcription of OpenMC C++ get_bin_from_indices."""
    return ((k*ny)+j)*nx+i


def read_hot_voxel(*,i: int=1,j: int=1,k: int=2,
                   nx: int=2,ny: int=3,nz: int=4,
                   one_element=None):
    """Create a native-layout file with one nonzero, named mesh heating bin."""
    with tempfile.TemporaryDirectory() as td:
        path=Path(td)/"statepoint.10.h5"
        synthetic_statepoint(path,dimension=(nx,ny,nz),n=8)
        with h5py.File(path,"a") as h:
            if one_element is not None:
                xy=one_element.solid.transverse
                mesh=h["tallies/meshes/mesh 19"]
                mesh["lower_left"][...]=[xy.x_edges_cm[0],
                                        xy.y_edges_cm[0],-66.]
                mesh["upper_right"][...]=[xy.x_edges_cm[-1],
                                          xy.y_edges_cm[-1],66.]
            results=h["tallies/tally 3/results"]
            array=results[...]
            array[:,0,:]=0.
            flat=native_bin(i,j,k,nx,ny)
            array[flat,0,0]=8*7.
            array[flat,0,1]=8*7.**2
            results[...]=array
        return read_openmc_neutron_fuel_statepoint(
            path,expected_fuel_material_id=FUEL_ID,
            nuclear_data_sha256=DATA_SHA,
            source_rate_particles_per_s=1e18,
            source_rate_basis="SYNTHETIC, C++ bin index one-hot test only",
            photon_transport_confirmed_disabled=True,
            expected_active_length_m=1.32)


class LiteralCxxMeshIndexTests(unittest.TestCase):
    def test_independent_cpp_bin_numbering(self):
        nx,ny,nz=3,4,5
        all_bins=[native_bin(i,j,k,nx,ny)
                  for k in range(nz) for j in range(ny) for i in range(nx)]
        self.assertEqual(all_bins,list(range(nx*ny*nz)))
        self.assertEqual(native_bin(1,0,0,nx,ny),1)
        self.assertEqual(native_bin(0,1,0,nx,ny),3)
        self.assertEqual(native_bin(0,0,1,nx,ny),12)
        self.assertEqual(native_bin(2,3,4,nx,ny),59)

    def test_true_native_single_voxel_yields_only_correct_xyz(self):
        native=read_hot_voxel(i=1,j=1,k=2)
        self.assertEqual(native.mesh_dimensions,(2,3,4))
        mean=native.source_mesh_mean_ev_per_source
        self.assertEqual(mean.shape,(2,3,4))
        self.assertAlmostEqual(float(mean[1,1,2]),7.,delta=1e-13)
        self.assertEqual(int(np.count_nonzero(mean)),1)
        np.testing.assert_allclose(native.score.fuel_neutron_ev_per_source,
                                   [0.,0.,7.,0.],rtol=0,atol=1e-13)
        np.testing.assert_allclose(native.source_mesh_stderr_ev_per_source,
                                   np.zeros((2,3,4)),rtol=0,atol=5e-8)

    def test_distinct_axial_and_x_mesh_bins_never_alias(self):
        a=read_hot_voxel(i=1,j=1,k=0)
        b=read_hot_voxel(i=1,j=1,k=3)
        self.assertAlmostEqual(a.source_mesh_mean_ev_per_source[1,1,0],7.)
        self.assertAlmostEqual(b.source_mesh_mean_ev_per_source[1,1,3],7.)
        np.testing.assert_allclose(a.score.fuel_neutron_ev_per_source,
                                   [7.,0.,0.,0.],atol=1e-13,rtol=0)
        np.testing.assert_allclose(b.score.fuel_neutron_ev_per_source,
                                   [0.,0.,0.,7.],atol=1e-13,rtol=0)

    def test_fuel_cutcell_mapping_preserves_one_hot_transverse_and_axial_location(self):
        m=coupled(film=0.,nz=4,nx=20,ny=24)
        native=read_hot_voxel(one_element=m)
        r=project_native_3d_fuel_heating(
            m.config,native,m.solid,isolated_element_geometry_confirmed=True)
        q=r.resolved.fuel_deposition_w
        self.assertEqual(q.shape,(4,m.solid.transverse.n_fuel))
        expected=7e18*EV_TO_J
        self.assertAlmostEqual(float(q.sum()),expected,delta=1e-11)
        self.assertAlmostEqual(float(q[2].sum()),expected,delta=1e-11)
        np.testing.assert_array_equal(q[[0,1,3]],np.zeros_like(q[[0,1,3]]))
        # Rightmost source x bin (i=1) has x>=0 for a centered hex.
        x=m.solid.transverse.cell_centers_cm[:,0]
        self.assertAlmostEqual(float(q[2,x<0].sum()),0.,delta=1e-13)
        self.assertAlmostEqual(float(q[2,x>0].sum()),expected,delta=1e-11)
        column=native_bin(1,1,0,2,3)  # XY source bin i+nx*j
        self.assertAlmostEqual(
            float(r.source_xy_to_fuel_weights[:,column].sum()),
            1.,delta=2e-10)
        self.assertLess(abs(r.resolved.fuel_power_balance_residual_w),1e-11)

    def test_other_transverse_voxel_changes_spatial_heat_map(self):
        m=coupled(film=0.,nz=4,nx=20,ny=24)
        a=read_hot_voxel(i=0,j=1,k=2,one_element=m)
        b=read_hot_voxel(i=1,j=1,k=2,one_element=m)
        qa=project_native_3d_fuel_heating(
            m.config,a,m.solid,isolated_element_geometry_confirmed=True)
        qb=project_native_3d_fuel_heating(
            m.config,b,m.solid,isolated_element_geometry_confirmed=True)
        x=m.solid.transverse.cell_centers_cm[:,0]
        np.testing.assert_allclose(qa.resolved.fuel_deposition_w[2,x>0],
                                   0.,atol=1e-13,rtol=0)
        np.testing.assert_allclose(qb.resolved.fuel_deposition_w[2,x<0],
                                   0.,atol=1e-13,rtol=0)
        self.assertGreater(float(np.max(np.abs(
            qa.resolved.fuel_deposition_w-qb.resolved.fuel_deposition_w))),1e-5)
        self.assertAlmostEqual(qa.resolved.mapped_fuel_power_w,
                               qb.resolved.mapped_fuel_power_w,delta=1e-11)


if __name__=="__main__":
    unittest.main(verbosity=2)
