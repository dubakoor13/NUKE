"""Actual executed OpenMC neutronic source, reaction and energy data tests.

NO mocked HDF5; require original saved four-seed 2026-10-23 archive
from Actions 39933618638 artifact 12492499186.

Unit tests of real evidence intentionally do NOT certify a hot reactor,
a stationary fission source, or the physical use of nuclear score watts.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import h5py
import numpy as np

from .real_native_tally_physics_ledger import (
    audit_native_neutron_heating,
)


class ExecutedNeutronPhysicsTallyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original=os.environ.get("NUKE_REAL_CORE_EVIDENCE_PATH")
        if not original or not Path(original).exists():
            raise RuntimeError(
                "Original OpenMC four-seed HDF5 evidence required; "
                "no synthetic statepoint substitution is allowed"
            )
        cls.source=Path(original)

    def test_four_original_statepoints_pass_native_cross_representation(self):
        result=audit_native_neutron_heating(self.source)
        self.assertEqual(result["independent_rng_seed_count"],4)
        self.assertEqual([r["seed"] for r in result["seed_reaction_scores"]],
                         [13007,23011,33013,43019])
        self.assertTrue(result["spatially_partitioned_neutron_scores_close"])
        self.assertTrue(result["native_neutron_population_identity_closes"])
        self.assertTrue(all(r["same_domain_mesh_material_energy_cross_scores_verified"]
                            for r in result["seed_reaction_scores"]))

    def test_native_neutron_budget_and_leakage_are_per_source(self):
        result=audit_native_neutron_heating(self.source)
        means=result["native_mean_scores_per_source"]
        np.testing.assert_allclose(
            means["global_analog_absorption_per_start"],
            .039666666666666656,atol=5e-13,rtol=0)
        np.testing.assert_allclose(
            means["global_tracklength_absorption_per_start"],
            .03991528763875892,atol=5e-13,rtol=0)
        np.testing.assert_allclose(
            means["native_neutron_vacuum_leakage_per_start"],
            1.0238555555555557,atol=5e-13,rtol=0)
        np.testing.assert_allclose(
            means["net_multiplicative_scattering_per_start"],
            .06352222222222137,atol=5e-13,rtol=0)
        for item in result["seed_reaction_scores"]:
            self.assertAlmostEqual(
                item["neutron_transport_population_accounting"],1.,places=12)

    def test_banked_fission_generation_not_added_to_current_balance(self):
        result=audit_native_neutron_heating(self.source)
        m=result["native_mean_scores_per_source"]
        self.assertAlmostEqual(m["global_tracklength_nu_fission_per_start"],
                               .011688432934274583,places=13)
        self.assertAlmostEqual(m["global_analog_nu_fission_per_start"],
                               .011669789908648396,places=13)
        self.assertGreater(m["native_neutron_vacuum_leakage_per_start"],1.)
        self.assertFalse(result["source_stationarity_demonstrated"])

    def test_nonzero_heating_and_fission_energy_are_not_local_deposition(self):
        result=audit_native_neutron_heating(self.source)
        scores=result["native_mean_scores_per_source"]
        self.assertGreater(scores["mesh_heating_score_ev_per_source"],0.)
        self.assertGreater(scores["mesh_kappa_fission_ev_per_source"],0.)
        self.assertGreater(scores["mesh_kappa_fission_ev_per_source"],
                           scores["mesh_heating_score_ev_per_source"])
        self.assertFalse(result["physical_local_heating_qualified"])
        self.assertFalse(result["hot_core_reactor_power_normalization_allowed"])
        self.assertFalse(result["historical_nerva_operating_state_validated"])
        for seed in result["seed_reaction_scores"]:
            self.assertTrue(seed["local_heating_all_first_and_second_moments_zero"])
            self.assertFalse(seed["material_separated_local_heating_usable_for_watts"])

    def test_actual_independent_seeds_have_distinct_hashes(self):
        report=audit_native_neutron_heating(self.source)
        sha=[r["native_statepoint_sha256"] for r in report["seed_reaction_scores"]]
        self.assertEqual(len(set(sha)),4)
        self.assertEqual(len(sha),4)

    def test_missing_real_evidence_fails_without_fabrication(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError,"missing REAL OpenMC evidence"):
                audit_native_neutron_heating(td)

    def test_native_hdf5_tally_corruption_is_rejected(self):
        if not self.source.is_file():
            self.skipTest("Destructive source mutation requires repackaged native ZIP")
        with zipfile.ZipFile(self.source) as src:
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"changed-tally.zip"
                with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as dst:
                    for name in src.namelist():
                        content=src.read(name)
                        if name=="seed_23011/statepoint.120.h5":
                            with h5py.File(io.BytesIO(content),"r") as sp:
                                self.assertIn("tallies",sp)
                            content=content[:-1]+bytes([content[-1]^1])
                        dst.writestr(name,content)
                with self.assertRaisesRegex(ValueError,"SHA256"):
                    audit_native_neutron_heating(path)

    def test_forged_heating_qualification_is_rejected(self):
        if not self.source.is_file():
            self.skipTest("Report tampering requires ZIP, not extracted directory")
        with zipfile.ZipFile(self.source) as src:
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/"changed-energy-provenance.zip"
                with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_DEFLATED) as dst:
                    for name in src.namelist():
                        content=src.read(name)
                        if name=="heating_data_qualification.json":
                            changed=json.loads(content)
                            changed["thermal_power_normalization_allowed"]=True
                            content=json.dumps(changed).encode()
                        dst.writestr(name,content)
                with self.assertRaisesRegex(ValueError,"qualification"):
                    audit_native_neutron_heating(path)


if __name__=="__main__":
    unittest.main(verbosity=2)
