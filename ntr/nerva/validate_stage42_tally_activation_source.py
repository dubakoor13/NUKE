"""Stage42 adversarial mathematics and mandatory original OpenMC neutron HDF5."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .real_coupled_tally_activation_source_control import (
    validate_native_tally_control,
    native_inactive_sensitivity,
    audit_native_source_tally_activation,
)


class TallyActivationControlMathTests(unittest.TestCase):
    def test_invalid_initial_source_or_inactive_threshold_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            with self.assertRaisesRegex(ValueError,"unknown initial source"):
                validate_native_tally_control(p,p,mode="unphysical",inactive=20)
            with self.assertRaisesRegex(ValueError,"unknown initial source"):
                validate_native_tally_control(p,p,mode="legacy-broad",inactive=90)

    def test_fake_provenance_without_real_openmc_xml_or_hdf5_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            a=p/"a";b=p/"b"
            a.mkdir();b.mkdir()
            fake={"config":{"seed":13007,"inactive":20,"batches":100,"particles":3000,
                   "photon_transport":True,"initial_fission_source":"legacy-broad",
                   "statepoint_batches":[60,100]},
                  "openmc_version":"0.15.3",
                  "xml_sha256":{"geometry.xml":"x","materials.xml":"x",
                                "settings.xml":"x","tallies.xml":"x"},
                  "nuclear_data":{"cross_sections_sha256":"fake"}}
            (a/"transport_provenance.json").write_text(json.dumps(fake))
            fake["config"]["diagnostic_no_user_tallies"]=True
            (b/"transport_provenance.json").write_text(json.dumps(fake))
            with self.assertRaisesRegex(ValueError,"XML SHA"):
                validate_native_tally_control(a,b,mode="legacy-broad",inactive=20)

    def test_missing_real_transport_cannot_be_replaced_with_empty_folder(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                audit_native_source_tally_activation(Path(root)/"a",
                                                     Path(root)/"b")

    def test_source_only_20_vs_40_missing_real_hdf5_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                native_inactive_sensitivity(Path(root)/"a",Path(root)/"b",
                                            mode="legacy-broad")

    def test_source_only_bad_mode_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                native_inactive_sensitivity(Path(root)/"a",Path(root)/"b",
                                            mode="fabricated")

    def test_source_hdf5_must_keep_evaluated_photon_transport(self):
        with tempfile.TemporaryDirectory() as root:
            a=Path(root)/"a";b=Path(root)/"b"
            a.mkdir();b.mkdir()
            for p,inactive in ((a,20),(b,40)):
                (p/"transport_provenance.json").write_text(json.dumps({
                    "config":{"seed":13007,"inactive":inactive,
                              "initial_fission_source":"legacy-broad",
                              "diagnostic_no_user_tallies":True}
                }))
            with self.assertRaises(FileNotFoundError):
                native_inactive_sensitivity(a,b,mode="legacy-broad")


class GenuineOpenMCTallyFreeControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original=os.getenv("NUKE_STAGE42_REAL_SCORED_STAGE41")
        source_only=os.getenv("NUKE_STAGE42_REAL_NO_USER_TALLIES")
        if not original or not source_only:
            raise RuntimeError("Stage42 requires genuine archived 4 baseline and 4 source-only C++ OpenMC runs")
        cls.original,cls.source_only=Path(original),Path(source_only)
        cls.r=audit_native_source_tally_activation(cls.original,cls.source_only)

    def test_sixteen_real_original_neutron_banks_and_zero_scoring_controls(self):
        r=self.r
        self.assertEqual(r["genuine_actual_openmc_hdf5_checkpoint_files_inspected"],16)
        self.assertEqual(r["new_source_only_OpenMC_Cplusplus_source_runs"],4)
        self.assertTrue(r["photon_transport_retained_in_all_eight_original_executions"])
        self.assertTrue(r["all_four_original_source_only_runs_have_no_user_scoring_tallies"])
        for mode in r["per_source_initialization"]:
            self.assertEqual(len(mode["scored_vs_no_user_tallies_at_fixed_inactive"]),2)
            for pair in mode["scored_vs_no_user_tallies_at_fixed_inactive"]:
                self.assertTrue(pair[
                    "identical_physical_geometry_material_nuclear_data_source_and_settings_proven"])
                self.assertTrue(pair["native_source_only_hdf5_has_no_user_scoring_tallies"])

    def test_real_fission_genealogy_diagnostics_not_reactor_power(self):
        r=self.r
        self.assertFalse(r["hot_historical_NERVA_or_physical_watts_qualified"])
        self.assertFalse(r["source_stationarity_statistically_qualified"])
        self.assertFalse(r["user_tally_activation_CAUSALLY_changes_neutron_genealogy_PROVEN"])
        for mode in r["per_source_initialization"]:
            x=mode["20_vs_40_inactive_without_any_user_tallies"]
            self.assertTrue(x["user_scoring_tallies_removed_in_both_original_runs"])
            self.assertEqual(len(x["native_exact_full_3000_site_banks_match_at_60_and_100"]),2)
            for pair in mode["scored_vs_no_user_tallies_at_fixed_inactive"]:
                self.assertFalse(pair["source_only_candidate_physical_heating_scored"])


if __name__=="__main__":
    unittest.main(verbosity=2)
