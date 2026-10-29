"""Stage63 native OpenMC statepoint writer/parser format compatibility.

Uses HDF5 fixtures conforming to the source-level structure in OpenMC
src/state_point.cpp, include/openmc/constants.h, and openmc/statepoint.py.
Fixture tests do NOT establish genuinely simulated neutron physics.
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
from .openmc_native_fuel_statepoint import read_openmc_neutron_fuel_statepoint


class NativeOpenMCVersionAndBatchTests(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.TemporaryDirectory()
        self.f=Path(self.d.name)/"statepoint.10.h5"
        synthetic_statepoint(self.f,n=8)

    def tearDown(self):
        self.d.cleanup()

    def read(self):
        return read_openmc_neutron_fuel_statepoint(
            self.f,expected_fuel_material_id=FUEL_ID,
            nuclear_data_sha256=DATA_SHA,
            source_rate_particles_per_s=1e18,
            source_rate_basis="SYNTHETIC test input only",
            photon_transport_confirmed_disabled=True,
            expected_active_length_m=1.32)

    def test_real_native_cpp_schema_major_minor_array_recognized(self):
        result=self.read()
        self.assertEqual(result.statepoint_format_version,(18,2))
        self.assertEqual(result.current_batch,10)
        self.assertEqual(result.energy_mode,"continuous-energy")
        self.assertEqual(result.active_realizations,8)

    def test_previous_scalar_version_fake_is_rejected(self):
        with h5py.File(self.f,"a") as f:
            f.attrs["version"]=18
        with self.assertRaisesRegex(ValueError,"major,minor"):
            self.read()

    def test_native_major_version_is_checked_independent_of_minor(self):
        with h5py.File(self.f,"a") as f:
            f.attrs["version"]=np.array([18,0])
        self.assertEqual(self.read().statepoint_format_version,(18,0))
        with h5py.File(self.f,"a") as f:
            f.attrs["version"]=np.array([19,0])
        with self.assertRaisesRegex(ValueError,"major,minor"):
            self.read()

    def test_invalid_minor_and_length_are_rejected(self):
        with h5py.File(self.f,"a") as f:
            f.attrs["version"]=np.array([18,-1])
        with self.assertRaisesRegex(ValueError,"major,minor"):
            self.read()
        with h5py.File(self.f,"a") as f:
            f.attrs["version"]=np.array([18,2,0])
        with self.assertRaisesRegex(ValueError,"major,minor"):
            self.read()

    def test_multigroup_mode_cannot_be_treated_as_continuous_energy(self):
        with h5py.File(self.f,"a") as f:
            f["energy_mode"][...]=np.bytes_("multi-group")
        with self.assertRaisesRegex(ValueError,"continuous-energy"):
            self.read()

    def test_missing_energy_mode_rejected(self):
        with h5py.File(self.f,"a") as f:
            del f["energy_mode"]
        with self.assertRaisesRegex(ValueError,"continuous-energy"):
            self.read()

    def test_missing_current_batch_rejected(self):
        with h5py.File(self.f,"a") as f:
            del f["current_batch"]
        with self.assertRaisesRegex(ValueError,"current_batch"):
            self.read()

    def test_intermediate_statepoint_counts_completed_not_final_batches(self):
        # A configured 50-batch run saved at batch 10 has only 8 active
        # realizations after two inactive batches, not 48.
        with h5py.File(self.f,"a") as f:
            f["n_batches"][...]=50
            f["current_batch"][...]=10
        result=self.read()
        self.assertEqual(result.n_batches,50)
        self.assertEqual(result.current_batch,10)
        self.assertEqual(result.active_realizations,8)

    def test_intermediate_statepoint_with_too_many_realizations_rejected(self):
        with h5py.File(self.f,"a") as f:
            f["n_batches"][...]=50
            f["current_batch"][...]=10
            f["tallies/tally 3/n_realizations"][...]=9
        with self.assertRaisesRegex(ValueError,"exceed active"):
            self.read()

    def test_current_batch_beyond_planned_or_during_inactive_rejected(self):
        with h5py.File(self.f,"a") as f:
            f["current_batch"][...]=11
        with self.assertRaisesRegex(ValueError,"invalid source batches"):
            self.read()
        with h5py.File(self.f,"a") as f:
            f["current_batch"][...]=2
        with self.assertRaisesRegex(ValueError,"invalid source batches"):
            self.read()


if __name__=="__main__":
    unittest.main(verbosity=2)
