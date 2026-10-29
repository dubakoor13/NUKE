"""Stage60 synthetic HDF5 fixtures mimicking native OpenMC statepoint storage.

The parser reads HDF5 bytes with the same tally sum/sum_sq, mesh ID,
filter and score organization as OpenMC; the fixture is NOT a genuine
OpenMC transport calculation or qualified neutron cross-section result.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from .openmc_native_fuel_statepoint import (
    read_openmc_neutron_fuel_statepoint, project_native_fuel_heating
)
from .openmc_fuel_axial_heat_handoff import (
    project_openmc_axial_fuel_heating, check_heating_for_coupled_core,
)
from .validate_stage58_fuel_hydrogen_coupling import coupled, initial, solve

FUEL_ID=531
DATA_SHA="c"*64


def synthetic_statepoint(path: Path, *,
                         dimension=(2,3,4),n=8,
                         photon=False,material_id=FUEL_ID,
                         tally_name="nerva_3d_fuel_heating",
                         score_name="heating-local",
                         incorrect_shape=False,
                         negative_variance=False):
    """Write a test fixture with the real native HDF5 group hierarchy."""
    nx,ny,nz=dimension
    original=np.arange(nx*ny*nz,dtype=float).reshape((nx,ny,nz))
    mean=1+original*0.2
    err=0.1+original*0.001
    if negative_variance:
        sumsq=np.zeros_like(mean)
    else:
        sumsq=n*(mean**2+(n-1)*err**2)
    tally_scores=np.zeros((nx*ny*nz,3,2))
    # Real OpenMC structured-mesh tally: bin=i + nx*(j + ny*k).
    # Native data are flat with X fastest (Fortran array order).
    tally_scores[:,0,0]=n*mean.reshape(-1,order="F")
    tally_scores[:,0,1]=sumsq.reshape(-1,order="F")
    tally_scores[:,1,0]=n*.25
    tally_scores[:,1,1]=n*(.25**2+(n-1)*.01**2)
    tally_scores[:,2,0]=n*.5
    tally_scores[:,2,1]=n*(.5**2+(n-1)*.03**2)
    with h5py.File(path,"w") as h:
        h.attrs["filetype"]=np.bytes_("statepoint")
        h.attrs["version"]=np.array([18,2],dtype=np.int32)
        h.attrs["tallies_present"]=1
        h.attrs["photon_transport"]=int(photon)
        h.attrs["openmc_version"]=np.array([0,15,3],dtype=np.int64)
        h.create_dataset("run_mode",data=np.bytes_("eigenvalue"))
        h.create_dataset("energy_mode",data=np.bytes_("continuous-energy"))
        h.create_dataset("n_batches",data=n+2)
        h.create_dataset("current_batch",data=n+2)
        h.create_dataset("n_inactive",data=2)
        h.create_dataset("n_particles",data=2000)
        manifest=h.create_group("tallies")
        manifest.attrs["n_tallies"]=1
        manifest.attrs["ids"]=np.array([3],dtype=np.int64)
        tally=manifest.create_group("tally 3")
        tally.create_dataset("name",data=np.bytes_(tally_name))
        tally.create_dataset("score_bins",
                             data=np.array([score_name,"fission","nu-fission"],dtype="S"))
        tally.create_dataset("nuclides",data=np.array(["total"],dtype="S"))
        tally.create_dataset("filters",data=np.array([11,12]))
        tally.create_dataset("n_realizations",data=n)
        if incorrect_shape:
            tally_scores=tally_scores[:-1]
        tally.create_dataset("results",data=tally_scores)
        meshfilter=h.create_group("tallies/filters/filter 11")
        meshfilter.create_dataset("type",data=np.bytes_("mesh"))
        meshfilter.create_dataset("bins",data=[19])
        meshfilter.create_dataset("n_bins",data=nx*ny*nz)
        matfilter=h.create_group("tallies/filters/filter 12")
        matfilter.create_dataset("type",data=np.bytes_("material"))
        matfilter.create_dataset("bins",data=[material_id])
        matfilter.create_dataset("n_bins",data=1)
        mesh=h.create_group("tallies/meshes/mesh 19")
        mesh.create_dataset("type",data=np.bytes_("regular"))
        mesh.create_dataset("dimension",data=dimension)
        mesh.create_dataset("lower_left",data=[-2.,-3.,-66.])
        mesh.create_dataset("upper_right",data=[2.,3.,66.])
    return mean,err


class NativeStatepointMomentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/"statepoint.8.h5"
        self.mean,self.stderr=synthetic_statepoint(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self,**replacements):
        kw={
            "expected_fuel_material_id":FUEL_ID,
            "nuclear_data_sha256":DATA_SHA,
            "source_rate_particles_per_s":1e18,
            "source_rate_basis":"SYNTHETIC test rate, not physical NERVA power",
            "photon_transport_confirmed_disabled":True,
            "expected_active_length_m":1.32,
        }
        kw.update(replacements)
        return read_openmc_neutron_fuel_statepoint(self.path,**kw)

    def test_true_native_hdf5_moments_to_axial_energy(self):
        result=self.read()
        expected=np.sum(self.mean,axis=(0,1))
        np.testing.assert_allclose(
            result.score.fuel_neutron_ev_per_source,expected,rtol=0,atol=1e-14)
        np.testing.assert_allclose(
            result.axial_uncertainty_upper_bound_ev_per_source,
            np.sum(self.stderr,axis=(0,1)),atol=1e-12,rtol=1e-12)
        np.testing.assert_array_equal(result.score.fuel_photon_ev_per_source,
                                      np.zeros(4))
        np.testing.assert_allclose(
            result.score.axial_edges_m,[0.,.33,.66,.99,1.32],
            rtol=0,atol=2e-16)
        self.assertEqual(result.mesh_dimensions,(2,3,4))
        self.assertEqual(result.active_realizations,8)
        self.assertEqual(result.tally_id,3)
        self.assertIn("covariance unavailable",result.uncertainty_status)

    def test_statepoint_sha_matches_actual_file_bytes(self):
        raw=self.path.read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        result=self.read(expected_statepoint_sha256=digest)
        self.assertEqual(result.statepoint_sha256,digest)
        with self.assertRaisesRegex(ValueError,"SHA256"):
            self.read(expected_statepoint_sha256="f"*64)

    def test_neutron_only_guard_rejects_coupled_photon_modes(self):
        with self.assertRaisesRegex(ValueError,"photon"):
            self.read(photon_transport_confirmed_disabled=False)
        with self.assertRaisesRegex(ValueError,"neutron-only"):
            self.read(photon_transport_confirmed_disabled="yes")
        self.assertEqual(self.read().photon_status.startswith("neutron-only"),True)

    def test_mesh_xyz_axis_order_is_openmc_x_fastest(self):
        # Each (x,y,z) carries a distinct native HDF5 mean, with X varying
        # fastest. A wrong C-order reshape changes the axial heating.
        result=self.read()
        np.testing.assert_allclose(
            result.source_mesh_mean_ev_per_source,self.mean,rtol=0,atol=1e-14)
        self.assertGreater(
            result.score.fuel_neutron_ev_per_source[-1],
            result.score.fuel_neutron_ev_per_source[0])

    def test_one_realization_cannot_supply_statistical_uncertainty(self):
        synthetic_statepoint(self.path,n=1)
        with self.assertRaisesRegex(ValueError,"fewer than two"):
            self.read()

    def test_invalid_tally_type_material_and_score_fail_closed(self):
        synthetic_statepoint(self.path,material_id=700)
        with self.assertRaisesRegex(ValueError,"material ID"):
            self.read()
        synthetic_statepoint(self.path,tally_name="unrelated_tally")
        with self.assertRaisesRegex(ValueError,"named NERVA"):
            self.read()
        synthetic_statepoint(self.path,score_name="heating")
        with self.assertRaisesRegex(ValueError,"heating-local"):
            self.read()

    def test_unphysical_raw_moments_are_rejected(self):
        synthetic_statepoint(self.path,negative_variance=True)
        with self.assertRaisesRegex(ValueError,"negative variance"):
            self.read()
        synthetic_statepoint(self.path,incorrect_shape=True)
        with self.assertRaisesRegex(ValueError,"moment shape"):
            self.read()

    def test_missing_source_rate_or_wrong_fuel_length_refused(self):
        with self.assertRaisesRegex(ValueError,"source rate"):
            self.read(source_rate_particles_per_s=0.)
        with self.assertRaisesRegex(ValueError,"source rate"):
            self.read(source_rate_basis="")
        with self.assertRaisesRegex(ValueError,"length"):
            self.read(expected_active_length_m=1.5)
        with self.assertRaisesRegex(ValueError,"material ID"):
            self.read(expected_fuel_material_id=0)

    def test_native_hdf5_scores_drive_stage59_and_stage58_unchanged(self):
        native=self.read()
        model=coupled(film=0.,nz=4,nx=20,ny=24)
        mapped=project_openmc_axial_fuel_heating(model.solid,native.score)
        watts=check_heating_for_coupled_core(mapped,model)
        self.assertEqual(watts.shape,
                         (4,model.solid.transverse.n_fuel))
        self.assertAlmostEqual(
            mapped.mapped_fuel_power_w,
            float(self.mean.sum())*1e18*1.602176634e-19,
            delta=2e-12)
        old=initial(model,tf=400.,th=400.)
        output=solve(model,old,dt=1e-5,q=watts)
        self.assertAlmostEqual(
            output.fuel_deposition_j,
            mapped.mapped_fuel_power_w*1e-5,delta=2e-12)
        self.assertLess(output.relative_first_law_residual,2e-9)
        self.assertIn("SYNTHETIC",native.score.source_rate_basis)

    def test_native_uncertainty_bound_propagates_with_identical_geometry_weights(self):
        native=self.read()
        m=coupled(film=0.,nz=4,nx=20,ny=24)
        projected=project_native_fuel_heating(native,m.solid)
        upper=projected.axial_uncertainty_upper_bound_w
        expected=(np.sum(self.stderr,axis=(0,1))
                  *1e18*1.602176634e-19)
        np.testing.assert_allclose(upper,expected,rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(
            projected.fuel_cell_uncertainty_upper_bound_w.sum(axis=1),
            expected,rtol=1e-12,atol=1e-13)
        self.assertAlmostEqual(
            projected.total_fuel_uncertainty_upper_bound_w,
            float(expected.sum()),delta=1e-12)
        self.assertGreater(projected.relative_total_uncertainty_upper_bound,0.)
        self.assertLess(projected.relative_total_uncertainty_upper_bound,1.)
        self.assertIn("covariance",projected.uncertainty_status)

    def test_uncertainty_bound_rejects_negative_or_invalid_axial_input(self):
        native=self.read()
        m=coupled(film=0.,nz=4,nx=20,ny=24)
        with self.assertRaisesRegex(ValueError,"uncertainty"):
            project_native_fuel_heating(
                replace(native,axial_uncertainty_upper_bound_ev_per_source=
                        np.array([-1.,0.,0.,0.])),m.solid)
        with self.assertRaisesRegex(ValueError,"uncertainty"):
            project_native_fuel_heating(
                replace(native,axial_uncertainty_upper_bound_ev_per_source=
                        np.zeros(3)),m.solid)

    def test_nonstatepoint_bytes_rejected(self):
        with h5py.File(self.path,"a") as f:
            f.attrs["filetype"]=np.bytes_("summary")
        with self.assertRaisesRegex(ValueError,"statepoint"):
            self.read()


if __name__=="__main__":
    unittest.main(verbosity=2)
