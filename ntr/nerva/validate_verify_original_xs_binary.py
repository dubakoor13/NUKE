"""Unit tests for actual binary isotope-identity comparison (not fake physics).

A separate CI job executes the real isotope-header and original-archive
checks inside exactly the OpenMC transport Docker environment.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from .verify_original_xs_binary import verify_original_hash_index


class RealIsotopeManifestNumerics(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.file = self.folder / "H1.h5"
        self.file.write_bytes(b"illustrative unit test only: not actual nuclear data")
        self.record = [{
            "path": str(self.file.resolve()),
            "sha256": hashlib.sha256(self.file.read_bytes()).hexdigest(),
            "bytes": self.file.stat().st_size,
        }]

    def test_unmodified_candidate_byte_digest_matches(self):
        result = verify_original_hash_index(
            self.record, ["H1"], {"H1": self.file})
        self.assertEqual(result["H1"]["verified_bytes"],self.file.stat().st_size)
        self.assertEqual(result["H1"]["sha256"],self.record[0]["sha256"])

    def test_mutated_candidate_bytes_fail_even_if_same_length(self):
        self.file.write_bytes(b"X" * self.file.stat().st_size)
        with self.assertRaisesRegex(ValueError,"SHA256/size mismatch"):
            verify_original_hash_index(self.record,["H1"],{"H1":self.file})

    def test_missing_actual_bytes_fail(self):
        self.file.unlink()
        with self.assertRaisesRegex(ValueError,"actual candidate.*missing"):
            verify_original_hash_index(self.record,["H1"],{"H1":self.file})

    def test_original_manifest_invalid_sha_fails(self):
        bad = [{**self.record[0], "sha256":"a"*63}]
        with self.assertRaisesRegex(ValueError,"invalid/duplicate"):
            verify_original_hash_index(bad,["H1"],{"H1":self.file})

    def test_wrong_candidate_index_rejected(self):
        with self.assertRaisesRegex(ValueError,"candidate evaluated isotope library"):
            verify_original_hash_index(self.record,["H1"],{"H2":self.file})

    def test_duplicate_manifest_records_rejected(self):
        with self.assertRaisesRegex(ValueError,"invalid/duplicate"):
            verify_original_hash_index(self.record*2,["H1"],{"H1":self.file})

    def test_source_hash_valid_but_candidate_path_different_fails(self):
        duplicate = self.folder / "alternative" / "H1.h5"
        duplicate.parent.mkdir()
        duplicate.write_bytes(self.file.read_bytes())
        with self.assertRaisesRegex(ValueError,"candidate isotope path differs"):
            verify_original_hash_index(self.record,["H1"],{"H1":duplicate})


if __name__ == "__main__":
    unittest.main(verbosity=2)
