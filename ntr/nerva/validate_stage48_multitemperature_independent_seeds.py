"""Stage48 numerical and mandatory real multi-kT independent-seed OpenMC QA."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .real_multitemperature_independent_seed_neutronics import (
    SEEDS, TEMPS, CHECKPOINTS, BINS,
    independent_seed_summary,
    total_variation,
    original_native_history,
    audit_original_ensemble,
)


class IndependentSeedMultiTemperatureMathTests(unittest.TestCase):
    def test_exact_four_seed_descriptive_sem_not_per_neutron(self):
        m=independent_seed_summary([0.009,0.01,0.011,0.012])
        self.assertEqual(m["independent_seed_count"],4)
        self.assertAlmostEqual(m["mean_across_four_original_independent_seeds"],0.0105)
        self.assertAlmostEqual(m["empirical_seed_mean_sem"],np.std(
            [0.009,0.01,0.011,0.012],ddof=1)/2)
        self.assertTrue(m["interval_does_not_account_for_unresolved_source_bias"])

    def test_four_unique_seeds_required_and_nonphysical_values_rejected(self):
        for data in ([0.01,0.011,0.012],[0.01,0.011,0.012,-0.01],
                     [0.01,0.011,0.012,float("nan")],
                     [0.01,0.011,0.012,math.inf]):
            with self.subTest(data=data),self.assertRaises(ValueError):
                independent_seed_summary(data)

    def test_spatial_total_variation_is_nonnegative_symmetric_and_bounded(self):
        a=np.array([[0.6,0.4],[0.,0.]])
        b=np.array([[0.1,0.9],[0.,0.]])
        self.assertAlmostEqual(total_variation(a,b),0.5)
        self.assertAlmostEqual(total_variation(a,b),total_variation(b,a))
        self.assertAlmostEqual(total_variation(a,a),0.)

    def test_impossible_source_mass_and_different_mesh_rejected(self):
        a=np.array([0.7,0.3])
        for other in (np.array([0.1,0.2]),np.array([1.0]),
                      np.array([float("nan"),0.]),
                      np.array([-0.1,1.1])):
            with self.assertRaises(ValueError):
                total_variation(a,other)

    def test_original_hdf5_cannot_be_replaced_by_json_or_missing_data(self):
        with tempfile.TemporaryDirectory() as root:
            f=Path(root)/"fake.h5"
            f.write_text(json.dumps({"seed":SEEDS[0],"k":0.01}))
            with self.assertRaises((OSError,ValueError,KeyError)):
                original_native_history(f,seed=SEEDS[0],batch=60)
            with self.assertRaises(FileNotFoundError):
                original_native_history(Path(root)/"nonexistent",seed=SEEDS[0],batch=60)

    def test_wrong_seed_or_checkpoint_not_accepted(self):
        with self.assertRaises(ValueError):
            original_native_history(Path("irrelevant"),seed=999,batch=60)
        with self.assertRaises(ValueError):
            original_native_history(Path("irrelevant"),seed=SEEDS[0],batch=59)

    def test_no_real_evaluated_XS_or_statepoints_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                audit_original_ensemble(Path(root)/"missing",
                                        nuclear_index=Path(root)/"cold_only_missing.xml")

    def test_declared_temperatures_and_four_different_seed_labels(self):
        self.assertEqual(TEMPS,(750,900))
        self.assertEqual(len(SEEDS),len(set(SEEDS)))
        self.assertEqual(CHECKPOINTS,(60,100))
        self.assertEqual(BINS,(4,4,10))


class ActualEvaluatedMultiTemperatureIndependentSeedOpenMC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=os.environ.get("NUKE_STAGE48_ACTUAL_CPP_HDF5_ROOT")
        evaluated=os.environ.get("NUKE_STAGE48_ACTUAL_MULTI_KT_XS")
        if not source or not evaluated:
            raise RuntimeError(
                "Stage48 physical acceptance requires original 8-run C++ neutron "
                "HDF5 and official evaluated multi-temperature isotope library"
            )
        cls.results=audit_original_ensemble(Path(source),nuclear_index=Path(evaluated))

    def test_actual_16_native_hdf5_independent_seed_and_temperature_controls(self):
        r=self.results
        self.assertEqual(r["original_OpenMC_CPP_neutron_eigenvalue_transport_runs"],8)
        self.assertEqual(r["actual_original_HDF5_source_statepoints_inspected"],16)
        self.assertTrue(r["same_original_geometry_all_eight"])
        self.assertEqual(r["independent_seed_labels"],list(SEEDS))
        self.assertEqual(r["temperature_K"],list(TEMPS))
        for case in r["original_nuclear_source_observations"]:
            self.assertEqual(len(case["four_original_Cplusplus_runs"]),4)
            self.assertTrue(all(x["source_total_variation_60_to_100"]>=0
                                for x in case["four_original_Cplusplus_runs"]))
            self.assertEqual(case["seed_level_descriptive_k_uncertainty"][
                "independent_seed_count"],4)

    def test_real_source_diagnostic_never_becomes_unvalidated_reactor_claim(self):
        r=self.results
        self.assertFalse(r["physical_eigenvalue_source_stationarity_demonstrated"])
        self.assertFalse(r["temperature_coefficient_or_worth_statistically_qualified"])
        self.assertFalse(r["historical_NASA_NERVA_operating_core_qualified"])
        self.assertFalse(r["coupled_thermal_feedback_power_or_watts_qualified"])
        self.assertTrue(r[
            "independent_seed_measurements_do_NOT_equal_independent_neutron_histories"])


if __name__=="__main__":
    unittest.main(verbosity=2)
