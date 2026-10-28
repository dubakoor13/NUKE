"""Stage36 population sensitivity: deterministic mathematics + mandatory real HDF5."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .real_coupled_population_sensitivity import (
    _population_statistics,
    _verify_cross_population_identity,
    audit_real_population_sensitivity,
)
from .real_coupled_source_temporal import _hdf5_identity


def _fixtures():
    s = np.ones((2, 4, 3, 4, 4, 10), dtype=float) / 160.
    h = np.ones((2, 4, 3, 12, 12, 10), dtype=float) * 200.
    return s, h


class PopulationSensitivityMathTests(unittest.TestCase):
    def test_equal_spatial_shapes_different_heating_amplitudes_have_zero_tv(self):
        s, h = _fixtures()
        h[1] *= 3.
        out = _population_statistics(s, h)
        self.assertEqual(out["source_final_seed_mean_4x4x10_TV_cross_population"], 0.)
        for t in out["heating_spatial_resolution_sensitivity"]:
            self.assertEqual(t["cross_population_final_seed_average_heating_TV"], 0.)
        self.assertFalse(out["higher_particle_count_guarantees_reduced_bias"])

    def test_disjoint_early_final_sources_and_heat_expose_full_drift(self):
        s, h = _fixtures()
        s[:] = 0.
        h[:] = 0.
        s[:, :, 0, 0, 0, 0] = 1.
        s[:, :, 1, 2, 2, 4] = 1.
        s[:, :, 2, 3, 3, 9] = 1.
        h[:, :, 0, 0, 0, 0] = 5.
        h[:, :, 1, 6, 6, 4] = 5.
        h[:, :, 2, 11, 11, 9] = 5.
        result = _population_statistics(s, h)
        for p in result["per_population_diagnostics"]:
            self.assertEqual(p["observed_mean_source_TV_batch30_to_65"], 1.)
            self.assertEqual(p["observed_mean_fine_3D_heating_TV_first_to_last"], 1.)

    def test_between_population_disjoint_final_source_means_are_maximally_different(self):
        s, h = _fixtures()
        s[0] = 0.
        s[1] = 0.
        s[0, :, :, 0, 0, 0] = 1.
        s[1, :, :, 3, 3, 9] = 1.
        r = _population_statistics(s, h)
        self.assertEqual(r["source_final_seed_mean_4x4x10_TV_cross_population"], 1.)

    def test_population_hotspots_stable_under_valid_coarsening(self):
        s, h = _fixtures()
        h[0] = 0.
        h[1] = 0.
        h[0, :, :, 0, 0, 5] = 1.
        h[1, :, :, 1, 1, 5] = 1.
        out = _population_statistics(s, h)
        levels = out["heating_spatial_resolution_sensitivity"]
        self.assertEqual(levels[0]["cross_population_final_seed_average_heating_TV"], 1.)
        for coarser in levels[1:]:
            self.assertEqual(coarser["cross_population_final_seed_average_heating_TV"], 0.)

    def test_source_or_heating_nonfinite_or_invalid_shapes_rejected(self):
        s, h = _fixtures()
        for invalid_s in (s[:1], np.full_like(s, np.nan), np.zeros_like(s),
                          np.full_like(s, -1.)):
            with self.subTest(shape=invalid_s.shape), self.assertRaises(ValueError):
                _population_statistics(invalid_s, h)
        for invalid_h in (h[:1], np.full_like(h, np.nan), np.zeros_like(h),
                          np.full_like(h, -1.)):
            with self.subTest(shape=invalid_h.shape), self.assertRaises(ValueError):
                _population_statistics(s, invalid_h)

    def test_unequal_geometry_xml_rejected_before_comparison(self):
        config = {"seed":13007, "particles":3000,
                  "statepoint_batches":[30,45,65], "batches":65,
                  "inactive":20, "photon_transport":True}
        left = {"config":config, "xml_sha256":{"geometry.xml":"a","materials.xml":"b","tallies.xml":"c"},
                "nuclear_data":{"cross_sections_sha256":"same"}}
        right = json.loads(json.dumps(left))
        right["config"]["particles"] = 6000
        _verify_cross_population_identity(
            left,right,{"native_total_heating_ev_per_source":1.},
            {"native_total_heating_ev_per_source":1.},13007)
        right["xml_sha256"]["materials.xml"] = "different"
        with self.assertRaisesRegex(ValueError,"geometry, material or tally"):
            _verify_cross_population_identity(
                left,right,{"native_total_heating_ev_per_source":1.},
                {"native_total_heating_ev_per_source":1.},13007)

    def test_unmatched_nuclear_data_and_control_or_temperature_rejected(self):
        cfg = {"seed":13007,"particles":3000,"statepoint_batches":[30,45,65],
               "batches":65,"inactive":20,"photon_transport":True,
               "fuel_temperature_k":300., "control_drum_angle_deg":0.}
        left = {"config":cfg,"xml_sha256":{x:"same" for x in
                   ("geometry.xml","materials.xml","tallies.xml")},
                "nuclear_data":{"cross_sections_sha256":"xs"}}
        right = json.loads(json.dumps(left))
        right["config"]["particles"] = 6000
        right["config"]["fuel_temperature_k"] = 900.
        with self.assertRaisesRegex(ValueError,"differs beyond particle count"):
            _verify_cross_population_identity(left,right,
                {"native_total_heating_ev_per_source":1.},
                {"native_total_heating_ev_per_source":1.},13007)
        right["config"]["fuel_temperature_k"] = 300.
        right["nuclear_data"]["cross_sections_sha256"] = "other"
        with self.assertRaisesRegex(ValueError,"different evaluated"):
            _verify_cross_population_identity(left,right,
                {"native_total_heating_ev_per_source":1.},
                {"native_total_heating_ev_per_source":1.},13007)

    def test_zero_native_heating_not_accepted(self):
        cfg = {"seed":13007,"particles":3000,"statepoint_batches":[30,45,65],
               "batches":65,"inactive":20,"photon_transport":True}
        left = {"config":cfg,
                "xml_sha256":{x:"same" for x in
                               ("geometry.xml","materials.xml","tallies.xml")},
                "nuclear_data":{"cross_sections_sha256":"xs"}}
        right = json.loads(json.dumps(left))
        right["config"]["particles"] = 6000
        with self.assertRaisesRegex(ValueError,"invalid original native"):
            _verify_cross_population_identity(left,right,
                 {"native_total_heating_ev_per_source":0.},
                 {"native_total_heating_ev_per_source":1.},13007)

    def test_missing_original_executed_hdf5_must_fail(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                audit_real_population_sensitivity(Path(d)/"a",Path(d)/"b")

    def test_forged_original_heating_ledger_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ("a","b"):
                folder = root/name
                folder.mkdir()
                (folder/"actual_four_seed_coupled_heating_covariance.json").write_text(
                    json.dumps({"schema":"fabricated","real_native_coupled_hdf5_count":4}))
            with self.assertRaisesRegex(ValueError,"unverified"):
                audit_real_population_sensitivity(root/"a",root/"b")

    def test_parameterized_population_reader_has_backwards_compatible_default(self):
        # Never synthesize accepted reactor HDF5; absent real statepoints must fail.
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "missing.h5"
            with self.assertRaises(ValueError):
                _hdf5_identity(path)
            with self.assertRaises(ValueError):
                _hdf5_identity(path,expected_particles=6000)


class ActualOriginalPopulationOpenMCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old = os.environ.get("NUKE_STAGE36_REAL_BASELINE_3000")
        new = os.environ.get("NUKE_STAGE36_REAL_CANDIDATE_6000")
        if not old or not new:
            raise RuntimeError("24 original native coupled OpenMC HDF5 checkpoints REQUIRED")
        cls.audit,cls.fields = audit_real_population_sensitivity(old,new)

    def test_24_native_statepoints_and_original_energy_and_source_history(self):
        r=self.audit
        self.assertEqual(r["physical_source_HDF5_count"],24)
        self.assertTrue(r["identical_geometry_material_tally_xml_and_evaluated_data_across_populations"])
        self.assertTrue(r["all_disjoint_material_particle_mesh_energy_partitions_close"])
        self.assertEqual(self.fields["source_probability"].shape,(2,4,3,4,4,10))
        self.assertEqual(self.fields["heating_ev_per_mc_start"].shape,(2,4,3,12,12,10))
        self.assertEqual(len(r["per_original_simulation"]),2)
        self.assertTrue(all(x["first_final_exact_generation_history_prefix_verified"]
                            for group in r["per_original_simulation"] for x in group))

    def test_reproduces_archived_original_3000_particle_physics_observations(self):
        p=self.audit["population_and_spatial_heating_diagnostics"]["per_population_diagnostics"][0]
        self.assertAlmostEqual(p["observed_mean_source_TV_batch30_to_65"],
                               0.6015,places=9)
        self.assertAlmostEqual(p["observed_mean_fine_3D_heating_TV_first_to_last"],
                               0.2248836445788241,places=9)

    def test_no_false_convergence_or_operating_reactor_claims(self):
        d=self.audit["population_and_spatial_heating_diagnostics"]
        self.assertFalse(d["higher_particle_count_guarantees_reduced_bias"])
        self.assertFalse(d["source_stationarity_or_uncertainty_qualified"])
        self.assertFalse(d["nasa_hot_nerva_reactor_power_qualified"])
        self.assertFalse(self.audit["reactor_hot_core_criticality_or_watts_qualified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
