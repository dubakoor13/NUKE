"""Stage 32 math tests plus mandatory separate genuine-HDF5 evidence tests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .real_coupled_source_bank_audit import (
    audit_stage31_final_fission_banks,
    final_bank_spatial_statistics,
)


class SourceBankSpatialMathTests(unittest.TestCase):
    def test_identical_final_banks_zero_spatial_distance(self):
        h = np.full((4, 2, 2, 2), 1/8.)
        r = final_bank_spatial_statistics(h)
        self.assertEqual(r["max_pairwise_final_bank_total_variation"], 0.)
        self.assertEqual(len(r["pairwise_final_bank_total_variation"]), 6)
        self.assertFalse(r["source_stationarity_established"])

    def test_disjoint_banks_have_unit_total_variation(self):
        h = np.zeros((4, 2, 2, 2))
        h[0, 0, 0, 0] = h[1, 1, 1, 1] = 1.
        h[2, 0, 0, 0] = h[3, 1, 1, 1] = 1.
        r = final_bank_spatial_statistics(h)
        self.assertEqual(r["max_pairwise_final_bank_total_variation"], 1.)
        self.assertAlmostEqual(r["mean_pairwise_final_bank_total_variation"], 4/6.)

    def test_axial_fractions_are_normalized(self):
        h = np.zeros((4, 2, 2, 2))
        h[:, 0, 0, 0] = [.25, .5, .75, 1.]
        h[:, 1, 1, 1] = [.75, .5, .25, 0.]
        r = final_bank_spatial_statistics(h)
        for x in r["per_seed_final_bank_axial_fractions"]:
            self.assertAlmostEqual(sum(x), 1.)
        self.assertAlmostEqual(sum(r["axial_mean_source_fraction"]), 1.)

    def test_bad_histogram_fails_closed(self):
        for h in (np.zeros((3, 2, 2, 2)), np.full((4, 2, 2, 2), -1.),
                  np.full((4, 2, 2, 2), np.nan), np.zeros((4, 2, 2, 2)),
                  np.zeros((4, 1, 2, 2))):
            with self.subTest(h=h.shape), self.assertRaises(ValueError):
                final_bank_spatial_statistics(h)

    def test_stage31_missing_original_binaries_never_fakes_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                audit_stage31_final_fission_banks(tmp)

    def test_reject_forged_stage31_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"actual_four_seed_coupled_heating_covariance.json"
            path.write_text(json.dumps({"schema":"forged", "real_native_coupled_hdf5_count":4}))
            with self.assertRaisesRegex(ValueError, "unqualified"):
                audit_stage31_final_fission_banks(tmp)

    def test_reject_wrong_native_statepoint_hash_even_when_histogram_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seeds = (13007, 23011, 33013, 43019)
            ledger = {
                "schema": "nuke-executed-coupled-neutron-photon-four-seed-stage31-v1",
                "real_native_coupled_hdf5_count": 4,
                "original_independent_openmc_rng_seeds": list(seeds),
                "all_electron_positron_neutron_photon_energy_deposition_partitions_conserved": True,
                "statistically_calibrated_fission_source_convergence": False,
                "real_neutron_photon_heating_hot_NERVA_physically_qualified": False,
                "mesh_lower_left_cm": [-5., -5., -5.],
                "mesh_upper_right_cm": [5., 5., 5.],
                "mesh_dimensions_xyz": [12, 12, 10],
                "per_actual_seed": [
                    {"seed":s, "native_statepoint_sha256":f"real_{s}",
                     "native_total_heating_ev_per_source":100.} for s in seeds
                ],
            }
            (root/"actual_four_seed_coupled_heating_covariance.json").write_text(json.dumps(ledger))
            for s in seeds:
                (root/f"seed_{s}").mkdir()
                (root/f"seed_{s}"/"statepoint.65.h5").write_bytes(b"mock fixture, not nuclear data")
            stub = (np.full((4, 4, 10), 1/160.),
                    {"seed":seeds[0], "completed_batch":65,
                     "statepoint_sha256":"bad_hash"})
            with patch("ntr.nerva.real_coupled_source_bank_audit.read_source_bank", return_value=stub):
                with self.assertRaisesRegex(ValueError, "not from pinned"):
                    audit_stage31_final_fission_banks(root)


class ActualOriginalCoupledFissionBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("NUKE_STAGE32_ORIGINAL_FOUR_SEED_ROOT")
        if not path:
            raise RuntimeError("four actual Stage31 native neutron/photon HDF5 statepoints required")
        cls.result = audit_stage31_final_fission_banks(path)

    def test_four_real_final_native_source_banks_match_original_heating_shas(self):
        r = self.result
        self.assertEqual(r["actual_source_bank_count"], 4)
        self.assertTrue(r["native_statepoint_hashes_match_original_heating_ledger"])
        self.assertEqual(r["seed_order"], [13007, 23011, 33013, 43019])
        self.assertTrue(all(x["source_bank"]["source_sites"] > 1 for x in r["source_bank_observations"]))
        self.assertTrue(all(x["source_bank"]["source_weights"] > 0 for x in r["source_bank_observations"]))

    def test_actual_source_diversity_does_not_infer_stationarity(self):
        r = self.result
        s = r["spatial_diagnostics"]
        self.assertEqual(s["bank_count"], 4)
        self.assertEqual(len(s["pairwise_final_bank_total_variation"]), 6)
        self.assertFalse(s["source_stationarity_established"])
        self.assertFalse(r["fission_source_stationarity_qualified"])
        self.assertFalse(r["historical_hot_nerva_or_reactor_watts_qualified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
