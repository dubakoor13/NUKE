"""Stage 29: fail-closed photon HDF5 coverage and provenance regressions.

Fixtures use fake bytes and mocked OpenMC readers ONLY to prove negative
handling. Actual evaluated nuclear/photon binaries are independently required
by production inspect_nuclear_data; no mock can physically qualify a core.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openmc.data import ATOMIC_NUMBER, IncidentPhoton

from .config import NervaConfig
from .nuclear_data import (
    inspect_nuclear_data, required_neutron_nuclides, required_photon_elements,
)


class CoupledPhotonDataPreflightTests(unittest.TestCase):
    def setUp(self):
        self._tmp=TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root=Path(self._tmp.name)
        self.nuclides=required_neutron_nuclides(NervaConfig())
        self.elements=required_photon_elements(NervaConfig())
        self.root.joinpath("photon").mkdir()
        for nuclide in self.nuclides:
            (self.root / f"{nuclide}.h5").write_bytes(b"synthetic-neutron-file")
        for element in self.elements:
            (self.root / "photon" / f"{element}.h5").write_bytes(
                b"synthetic-photon-file")
        self.xml=self.root/"cross_sections.xml"

    def manifest(self, *, photons=None, duplicates=(), multi=False):
        if photons is None:
            photons=self.elements
        rows=[
            f'<library type="neutron" materials="{nuclide}" path="{nuclide}.h5"/>'
            for nuclide in self.nuclides
        ]
        for element in (*photons,*duplicates):
            mats=element+(" H" if multi and element=="U" else "")
            rows.append(
                f'<library type="photon" materials="{mats}" path="photon/{element}.h5"/>')
        self.xml.write_text("<cross_sections>"+"".join(rows)+"</cross_sections>",
                            encoding="utf-8")

    def reader(self, *, identity=None, missing_reaction=False, no_relaxation=False):
        def fake(path):
            name=Path(path).stem
            reported=identity if identity is not None and name=="H" else name
            return SimpleNamespace(
                name=reported, atomic_number=ATOMIC_NUMBER[reported],
                reactions={} if missing_reaction and name=="H" else
                          {502:object(),504:object(),522:object()},
                atomic_relaxation=None if no_relaxation and name=="H" else object(),
            )
        return fake

    def test_photon_request_requires_every_element_not_one_arbitrary_file(self):
        self.manifest(photons=("H",))
        r=inspect_nuclear_data(self.xml,photon_transport=True)
        self.assertFalse(r.ready)
        self.assertTrue(r.photon_library_present)
        self.assertEqual(r.required_photon_elements,self.elements)
        self.assertEqual(r.missing_photon_elements,
                         tuple(sorted(set(self.elements)-{"H"})))
        self.assertNotEqual(r.missing_photon_elements,())

    def test_xml_entry_and_existing_fake_binary_do_not_qualify_photon_physics(self):
        self.manifest()
        r=inspect_nuclear_data(self.xml,photon_transport=True)
        self.assertFalse(r.ready)
        self.assertEqual(r.missing_photon_elements,())
        self.assertEqual(r.verified_photon_elements,())
        self.assertTrue(r.photon_binary_audit_errors)

    def test_missing_required_photon_HDF5_rejected(self):
        self.manifest()
        (self.root/"photon"/"H.h5").unlink()
        with patch.object(IncidentPhoton,"from_hdf5",side_effect=self.reader()):
            r=inspect_nuclear_data(self.xml,photon_transport=True)
        self.assertFalse(r.ready)
        self.assertIn("H",r.required_photon_elements)
        self.assertTrue(any("H: referenced" in x for x in r.photon_binary_audit_errors))
        self.assertTrue(any(x.endswith("H.h5") for x in r.referenced_files_missing))

    def test_swapped_photon_element_identity_fails_even_with_manifest_match(self):
        self.manifest()
        with patch.object(IncidentPhoton,"from_hdf5",side_effect=self.reader(identity="U")):
            r=inspect_nuclear_data(self.xml,photon_transport=True)
        self.assertFalse(r.ready)
        self.assertTrue(any("H: rejected" in x and "identity" in x
                            for x in r.photon_binary_audit_errors))

    def test_required_photon_interactions_not_fake_implicitly(self):
        self.manifest()
        with patch.object(IncidentPhoton,"from_hdf5",
                          side_effect=self.reader(missing_reaction=True)):
            r=inspect_nuclear_data(self.xml,photon_transport=True)
        self.assertFalse(r.ready)
        self.assertTrue(any("H: rejected" in x and "required" in x
                            for x in r.photon_binary_audit_errors))

    def test_atomic_relaxation_data_required_for_accepted_binary(self):
        self.manifest()
        with patch.object(IncidentPhoton,"from_hdf5",
                          side_effect=self.reader(no_relaxation=True)):
            r=inspect_nuclear_data(self.xml,photon_transport=True)
        self.assertFalse(r.ready)
        self.assertTrue(any("H: rejected" in x and "relaxation" in x
                            for x in r.photon_binary_audit_errors))

    def test_duplicate_element_entry_never_silently_overwrites_photon_file(self):
        self.manifest(duplicates=("H",))
        with patch.object(IncidentPhoton,"from_hdf5",side_effect=self.reader()):
            r=inspect_nuclear_data(self.xml,photon_transport=True)
        self.assertFalse(r.ready)
        self.assertTrue(any("H: ambiguous" in x for x in r.photon_binary_audit_errors))

    def test_multi_element_photon_HDF5_file_refused_as_identity_ambiguous(self):
        self.manifest(multi=True)
        with patch.object(IncidentPhoton,"from_hdf5",side_effect=self.reader()):
            r=inspect_nuclear_data(self.xml,photon_transport=True)
        self.assertFalse(r.ready)
        self.assertTrue(any("U: multi-element" in x for x in r.photon_binary_audit_errors))

    def test_neutron_only_does_not_require_photon_files(self):
        self.manifest(photons=())
        r=inspect_nuclear_data(self.xml,photon_transport=False)
        self.assertTrue(r.ready)
        self.assertEqual(r.required_photon_elements,())
        self.assertEqual(r.verified_photon_elements,())

    def test_mocked_hdf5_success_tests_mapping_not_evaluated_release(self):
        self.manifest()
        with patch.object(IncidentPhoton,"from_hdf5",side_effect=self.reader()) as mock:
            r=inspect_nuclear_data(self.xml,photon_transport=True)
        self.assertEqual(mock.call_count,len(self.elements))
        self.assertTrue(r.ready)
        self.assertEqual(r.verified_photon_elements,self.elements)
        self.assertEqual(len(r.photon_binary_sha256),len(self.elements))
        self.assertFalse(r.photon_evaluated_release_verified)
        # This mocks only the reader; NO genuine physical photon data was
        # loaded. The passing test never asserts hot reactor qualification.

    def test_skip_file_existence_flag_cannot_bypass_photon_binary_gate(self):
        self.manifest()
        r=inspect_nuclear_data(self.xml,photon_transport=True,
                               require_referenced_files=False)
        self.assertFalse(r.ready)
        self.assertTrue(r.photon_binary_audit_errors)


if __name__=="__main__":
    unittest.main(verbosity=2)
