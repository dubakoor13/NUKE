"""Stage65 native coupled neutron/photon HEATING tally verification.

SYNTHETIC native-layout HDF5 data emulate the real OpenMC mesh/material/
particle scoring structure. They establish parser and conservative mapping
correctness only, not an actual evaluated-data reactor transport result.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from .validate_stage60_native_hdf5_heating import (
    synthetic_statepoint,FUEL_ID,DATA_SHA,
)
from .validate_stage58_fuel_hydrogen_coupling import coupled,initial,solve
from .openmc_native_coupled_fuel_heating import read_openmc_coupled_fuel_statepoint
from .openmc_native_spatial_fuel_heating import project_native_3d_fuel_heating
from .openmc_fuel_axial_heat_handoff import EV_TO_J,check_heating_for_coupled_core

RATE=1e18


def coupled_native_fixture(path: Path, *, model=None,
                           particle_order=("neutron","photon"),
                           heating_score="heating"):
    dims=(2,3,4)
    nx,ny,nz=dims
    synthetic_statepoint(path,dimension=dims,n=8,photon=True)
    x=np.arange(nx)[:,None,None]
    y=np.arange(ny)[None,:,None]
    z=np.arange(nz)[None,None,:]
    neutron=np.broadcast_to(1.+.35*x+.07*y+.05*z,dims).copy()
    photon=np.broadcast_to(.2+.07*x+.12*y+.09*z,dims).copy()
    with h5py.File(path,"a") as h:
        tally=h["tallies/tally 3"]
        del tally["filters"]
        tally.create_dataset("filters",data=np.array([11,12,13],dtype=int))
        del tally["score_bins"]
        tally.create_dataset(
            "score_bins",data=np.array(
                [heating_score,"fission","nu-fission"],dtype="S"))
        new=np.zeros((nx*ny*nz*2,3,2))
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    flat=((k*ny)+j)*nx+i
                    for m,name in enumerate(particle_order):
                        val=neutron[i,j,k] if name=="neutron" else photon[i,j,k]
                        row=flat*2+m
                        new[row,0,0]=8*val
                        new[row,0,1]=8*(val**2+7*(0.01+0.002*m)**2)
        del tally["results"]
        tally.create_dataset("results",data=new)
        fg=h.create_group("tallies/filters/filter 13")
        fg.create_dataset("type",data=np.bytes_("particle"))
        fg.create_dataset("bins",data=np.asarray(particle_order,dtype="S"))
        fg.create_dataset("n_bins",data=2)
        if model is not None:
            m=h["tallies/meshes/mesh 19"]
            t=model.solid.transverse
            m["lower_left"][...]=[t.x_edges_cm[0],t.y_edges_cm[0],-66.]
            m["upper_right"][...]=[t.x_edges_cm[-1],t.y_edges_cm[-1],66.]
    return neutron,photon


def load(path: Path):
    return read_openmc_coupled_fuel_statepoint(
        path,expected_fuel_material_id=FUEL_ID,
        nuclear_data_sha256=DATA_SHA,
        source_rate_particles_per_s=RATE,
        source_rate_basis="SYNTHETIC HDF5 source-rate test, NOT NERVA",
        expected_active_length_m=1.32)


class NativeCoupledNuclearHeatTests(unittest.TestCase):
    def setUp(self):
        self.td=tempfile.TemporaryDirectory()
        self.path=Path(self.td.name)/"statepoint.10.h5"
        self.neutron,self.photon=coupled_native_fixture(self.path)

    def tearDown(self):
        self.td.cleanup()

    def test_native_photon_transport_heating_scored_separately(self):
        data=load(self.path)
        np.testing.assert_allclose(
            data.source_mesh_neutron_mean_ev_per_source,self.neutron,
            rtol=0,atol=1e-13)
        np.testing.assert_allclose(
            data.source_mesh_photon_mean_ev_per_source,self.photon,
            rtol=0,atol=1e-13)
        np.testing.assert_allclose(
            data.source_mesh_mean_ev_per_source,self.neutron+self.photon,
            rtol=0,atol=1e-13)
        np.testing.assert_allclose(
            data.score.fuel_neutron_ev_per_source,
            self.neutron.sum(axis=(0,1)),rtol=0,atol=1e-13)
        np.testing.assert_allclose(
            data.score.fuel_photon_ev_per_source,
            self.photon.sum(axis=(0,1)),rtol=0,atol=1e-13)
        self.assertFalse(data.photon_transport_verified_disabled)
        self.assertIn("physical deposition",data.photon_status)

    def test_particle_filter_reversed_order_preserves_both_heat_parts(self):
        coupled_native_fixture(self.path,particle_order=("photon","neutron"))
        data=load(self.path)
        np.testing.assert_allclose(
            data.source_mesh_neutron_mean_ev_per_source,self.neutron,
            atol=1e-13,rtol=0)
        np.testing.assert_allclose(
            data.source_mesh_photon_mean_ev_per_source,self.photon,
            atol=1e-13,rtol=0)

    def test_zero_photon_heating_retains_neutron_component(self):
        with h5py.File(self.path,"a") as h:
            data=h["tallies/tally 3/results"]
            array=data[...]
            array[1::2,0,:]=0.
            data[...]=array
        parsed=load(self.path)
        self.assertEqual(float(np.sum(parsed.score.fuel_photon_ev_per_source)),0.)
        self.assertGreater(float(np.sum(parsed.score.fuel_neutron_ev_per_source)),0.)

    def test_spatial_3d_neutron_photon_maps_conserve_each_component(self):
        m=coupled(film=0.,nz=4,nx=20,ny=24)
        coupled_native_fixture(self.path,model=m)
        native=load(self.path)
        q=project_native_3d_fuel_heating(
            m.config,native,m.solid,isolated_element_geometry_confirmed=True)
        xn=float(self.neutron.sum())*RATE*EV_TO_J
        xp=float(self.photon.sum())*RATE*EV_TO_J
        self.assertAlmostEqual(q.resolved.fuel_neutron_power_w,xn,delta=1e-12)
        self.assertAlmostEqual(q.resolved.fuel_photon_power_w,xp,delta=1e-12)
        self.assertAlmostEqual(q.resolved.mapped_fuel_power_w,xn+xp,delta=1e-12)
        np.testing.assert_allclose(
            q.resolved.fuel_deposition_w,
            q.neutron_fuel_deposition_w+q.photon_fuel_deposition_w,
            rtol=0,atol=2e-14)
        self.assertLess(abs(q.resolved.fuel_power_balance_residual_w),1e-11)
        self.assertGreater(q.total_uncertainty_bound_w,0)
        self.assertTrue(np.all(q.fuel_cell_uncertainty_bound_w>=0))

    def test_coupled_physical_heat_into_stage58_once(self):
        m=coupled(film=0.,nz=4,nx=20,ny=24)
        coupled_native_fixture(self.path,model=m)
        native=load(self.path)
        mapped=project_native_3d_fuel_heating(
            m.config,native,m.solid,isolated_element_geometry_confirmed=True)
        watts=check_heating_for_coupled_core(mapped.resolved,m)
        r=solve(m,initial(m,tf=400.,th=400.),dt=1e-5,q=watts)
        self.assertAlmostEqual(
            r.fuel_deposition_j,mapped.resolved.mapped_fuel_power_w*1e-5,
            delta=3e-12)
        self.assertLess(r.relative_first_law_residual,2e-9)
        self.assertEqual(float(np.sum(r.wall_heat_to_hydrogen_j)),0.)

    def test_photon_enabled_heating_local_fails_closed(self):
        coupled_native_fixture(self.path,heating_score="heating-local")
        with self.assertRaisesRegex(ValueError,"heating.*not heating-local"):
            load(self.path)

    def test_neutron_only_hdf5_cannot_masquerade_as_coupled(self):
        with h5py.File(self.path,"a") as h:
            h.attrs["photon_transport"]=0
        with self.assertRaisesRegex(ValueError,"photon_transport=1"):
            load(self.path)

    def test_particle_filter_missing_wrong_and_duplicate_rejected(self):
        with h5py.File(self.path,"a") as h:
            del h["tallies/filters/filter 13"]
        with self.assertRaisesRegex(ValueError,"filter missing"):
            load(self.path)
        coupled_native_fixture(self.path)
        with h5py.File(self.path,"a") as h:
            bins=h["tallies/filters/filter 13/bins"]
            bins[:]=np.array(["neutron","neutron"],dtype="S")
        with self.assertRaisesRegex(ValueError,"neutron and photon"):
            load(self.path)

    def test_wrong_particle_score_stride_rejected(self):
        with h5py.File(self.path,"a") as h:
            t=h["tallies/tally 3"]
            del t["results"]
            t.create_dataset("results",data=np.zeros((24,3,2)))
        with self.assertRaisesRegex(ValueError,"moment dimensions"):
            load(self.path)
        coupled_native_fixture(self.path)
        with h5py.File(self.path,"a") as h:
            h["tallies/tally 3/filters"][:]=[13,11,12]
        with self.assertRaisesRegex(ValueError,"filter stride"):
            load(self.path)

    def test_reject_physically_inconsistent_combined_component_map(self):
        m=coupled(film=0.,nz=4,nx=20,ny=24)
        coupled_native_fixture(self.path,model=m)
        native=load(self.path)
        bad=replace(native,source_mesh_photon_mean_ev_per_source=
                    native.source_mesh_photon_mean_ev_per_source*2)
        with self.assertRaisesRegex(ValueError,"contributions inconsistent"):
            project_native_3d_fuel_heating(
                m.config,bad,m.solid,isolated_element_geometry_confirmed=True)


if __name__=="__main__":
    unittest.main(verbosity=2)
