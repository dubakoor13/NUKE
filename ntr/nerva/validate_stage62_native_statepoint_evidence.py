"""Stage62 OpenMC-native HDF5 evidence/physics mode acceptance tests.

Test files are synthetic HDF5 approximating native OpenMC metadata.
No true transport history, evaluated cross-section library, or historical
NERVA geometry/source normalization is asserted by any passing test.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import h5py
import numpy as np

from .validate_stage60_native_hdf5_heating import (
    DATA_SHA,FUEL_ID,synthetic_statepoint,
)
from .openmc_native_fuel_statepoint import read_openmc_neutron_fuel_statepoint


class Stage62NativeSourceEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/"statepoint.10.h5"
        synthetic_statepoint(self.path,n=8)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def reader(self):
        return read_openmc_neutron_fuel_statepoint(
            self.path,
            expected_fuel_material_id=FUEL_ID,
            nuclear_data_sha256=DATA_SHA,
            source_rate_particles_per_s=1e18,
            source_rate_basis="SYNTHETIC source rate for numerical tests, not a reactor",
            photon_transport_confirmed_disabled=True,
            expected_active_length_m=1.32,
        )

    def test_native_source_metadata_is_retained(self):
        data=self.reader()
        self.assertTrue(data.photon_transport_verified_disabled)
        self.assertEqual(data.openmc_version,(0,15,3))
        self.assertEqual(data.n_batches,10)
        self.assertEqual(data.n_inactive,2)
        self.assertEqual(data.n_particles,2000)
        self.assertEqual(data.active_realizations,8)

    def test_photon_enabled_in_file_rejected_despite_external_claim(self):
        with h5py.File(self.path,"a") as h:
            h.attrs["photon_transport"]=1
        with self.assertRaisesRegex(ValueError,"photon transport enabled"):
            self.reader()

    def test_photon_mode_missing_is_never_assumed_disabled(self):
        with h5py.File(self.path,"a") as h:
            del h.attrs["photon_transport"]
        with self.assertRaisesRegex(ValueError,"photon_transport.*missing"):
            self.reader()

    def test_photon_mode_noninteger_and_invalid_values_rejected(self):
        with h5py.File(self.path,"a") as h:
            h.attrs["photon_transport"]=2
        with self.assertRaisesRegex(ValueError,"mode flag invalid"):
            self.reader()
        with h5py.File(self.path,"a") as h:
            h.attrs["photon_transport"]=np.bytes_("0")
        with self.assertRaisesRegex(ValueError,"integer flag"):
            self.reader()

    def test_wrong_realization_count_rejected(self):
        with h5py.File(self.path,"a") as h:
            h["tallies/tally 3/n_realizations"][...]=9
        with self.assertRaisesRegex(ValueError,"exceed active"):
            self.reader()

    def test_missing_physical_nparticles_and_invalid_batch_count_rejected(self):
        with h5py.File(self.path,"a") as h:
            del h["n_particles"]
        with self.assertRaisesRegex(ValueError,"particle metadata missing"):
            self.reader()
        synthetic_statepoint(self.path,n=8)
        with h5py.File(self.path,"a") as h:
            h["n_batches"][...]=2
            h["n_inactive"][...]=2
        with self.assertRaisesRegex(ValueError,"invalid source batches"):
            self.reader()

    def test_missing_and_duplicate_tally_manifest_rejected(self):
        with h5py.File(self.path,"a") as h:
            del h["tallies"].attrs["ids"]
        with self.assertRaisesRegex(ValueError,"tally ID manifest unavailable"):
            self.reader()
        synthetic_statepoint(self.path,n=8)
        with h5py.File(self.path,"a") as h:
            h["tallies"].attrs["n_tallies"]=2
            h["tallies"].attrs["ids"]=np.array([3,3])
        with self.assertRaisesRegex(ValueError,"invalid identities"):
            self.reader()

    def test_tally_not_in_openmc_manifest_rejected(self):
        with h5py.File(self.path,"a") as h:
            h["tallies"].attrs["ids"]=np.array([99])
        with self.assertRaisesRegex(ValueError,"not in native"):
            self.reader()

    def test_missing_openmc_version_rejected(self):
        with h5py.File(self.path,"a") as h:
            del h.attrs["openmc_version"]
        with self.assertRaisesRegex(ValueError,"version metadata missing"):
            self.reader()

    def test_unqualified_photon_mode_cannot_be_overridden(self):
        synthetic_statepoint(self.path,n=8,photon=True)
        with self.assertRaisesRegex(ValueError,"photon transport enabled"):
            self.reader()


if __name__=="__main__":
    unittest.main(verbosity=2)
