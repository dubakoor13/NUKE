"""Stage49: genuine neutron source birth position ≠ complete phase-space state."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .config import NervaConfig
from .physical_core_reference_validation import assess_core_fidelity
from .real_source_phase_space_and_reference_fidelity import (
    EXPECTED_SOURCE_FIELDS,
    characterize_original_neutron_source,
    native_phase_space,
    audit_actual_full_source_phase_space,
)


def synthetic_neutron_bank() -> np.ndarray:
    """Tiny mathematical fixture only; NEVER evidence of reactor transport."""
    dtype = np.dtype([
        ("r", [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]),
        ("u", [("x", "<f8"), ("y", "<f8"), ("z", "<f8")]),
        ("E", "<f8"), ("time", "<f8"), ("wgt", "<f8"),
        ("delayed_group", "<i4"), ("surf_id", "<i4"), ("particle", "<i4"),
    ])
    a=np.zeros(4,dtype=dtype)
    a["u"]["z"]=1.0
    a["E"]=[1e6,2e6,2e6,2e6]
    a["wgt"]=1.
    a["r"]["x"]=[0,0,0,1]
    return a


class NeutronBirthPhaseSpaceMathTests(unittest.TestCase):
    def test_identical_position_does_not_imply_identical_source_energy(self):
        r=characterize_original_neutron_source(synthetic_neutron_bank())
        self.assertEqual(r["exact_unique_birth_positions"],2)
        self.assertEqual(r["exact_unique_position_and_energy"],3)
        self.assertEqual(r["exact_unique_complete_source_particle_records"],3)
        self.assertEqual(r["exact_repeated_complete_source_records"],1)
        self.assertEqual(r["source_sites_sharing_birth_position"],2)

    def test_identical_position_energy_can_still_have_distinct_directions(self):
        a=synthetic_neutron_bank()
        a["u"]["z"][2]=0.
        a["u"]["x"][2]=1.
        r=characterize_original_neutron_source(a)
        self.assertEqual(r["exact_unique_birth_positions"],2)
        self.assertEqual(r["exact_unique_position_and_energy"],3)
        self.assertEqual(r["exact_unique_position_direction_energy"],4)
        self.assertEqual(r["exact_unique_complete_source_particle_records"],4)

    def test_even_full_duplicate_source_records_are_not_independence_measure(self):
        r=characterize_original_neutron_source(synthetic_neutron_bank())
        self.assertGreater(r["fraction_unique_complete_particle_records"],
                           r["fraction_unique_birth_positions"])
        self.assertAlmostEqual(r["fraction_unique_complete_particle_records"],0.75)
        self.assertAlmostEqual(r["fraction_unique_birth_positions"],0.50)

    def test_weighted_source_concentration_and_energy_are_actual_metrics(self):
        a=synthetic_neutron_bank()
        a["wgt"]=[1.,3.,2.,4.]
        r=characterize_original_neutron_source(a)
        self.assertAlmostEqual(r["largest_weighted_single_birth_position_fraction"],0.6)
        self.assertAlmostEqual(r["inverse_simpson_birth_position_count"],
                               1./(0.6**2+0.4**2))
        self.assertEqual(r["source_birth_energy_fraction_above_100keV"],1.0)

    def test_fake_neutron_direction_cannot_pass_as_real_phase_space(self):
        a=synthetic_neutron_bank()
        a["u"]["z"][0]=2.
        with self.assertRaisesRegex(ValueError,"nonphysical"):
            characterize_original_neutron_source(a)

    def test_nonphysical_energy_or_source_weight_is_rejected(self):
        for key,value in (("E",-0.1),("wgt",0.0),("E",float("nan"))):
            with self.subTest(key=key,value=value):
                a=synthetic_neutron_bank()
                a[key][0]=value
                with self.assertRaisesRegex(ValueError,"nonphysical"):
                    characterize_original_neutron_source(a)

    def test_missing_native_hdf5_never_replaced_with_synthetic_particle_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                native_phase_space(Path(d)/"not_original.h5",seed=13007,batch=100)
            with self.assertRaises(FileNotFoundError):
                audit_actual_full_source_phase_space(Path(d)/"not_original")

    def test_historical_reference_is_distinct_heritage_element_vs_proposed_snre(self):
        p=assess_core_fidelity(NervaConfig(core_rings=3),assembly="core")
        self.assertTrue(p["heritage_element_geometry_matches"])
        self.assertFalse(p["proposed_snre_bulk_geometry_matches"])
        self.assertEqual(p["overall_historical_core_validation"],"NOT_VALIDATED")
        self.assertFalse(p["material_fuel_matrix_is_verified_historical_nerva_composition"])


class GenuineOriginalOpenMCNeutronSourceQA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.getenv("NUKE_STAGE49_ORIGINAL_STAGE48_HDF5_ROOT")
        if not root or not Path(root).is_dir():
            raise RuntimeError("Stage49 requires ALL original Stage48 neutron HDF5 source banks")
        cls.report=audit_actual_full_source_phase_space(Path(root))

    def test_actual_16_original_neutron_hdf5_full_source_records(self):
        r=self.report
        self.assertEqual(r["original_openmc_v0153_hdf5_inspected"],16)
        self.assertTrue(r["positions_not_equivalent_to_unique_neutron_source_phase_states"])
        self.assertEqual(len(r["per_temperature"]),2)
        for temp in r["per_temperature"]:
            self.assertEqual(temp["independent_rng_seed_count"],4)
            for case in temp["original_cpp_source_observations"]:
                self.assertEqual(len(case["native_checkpoints"]),2)
                for state in case["native_checkpoints"]:
                    self.assertEqual(state["source_sites"],3000)
                    self.assertGreater(state["exact_unique_complete_source_particle_records"],
                                       state["exact_unique_birth_positions"])
                    self.assertGreater(state["exact_unique_position_and_energy"],
                                       state["exact_unique_birth_positions"])
                    self.assertLess(state["source_direction_max_unit_norm_error"],1e-9)
                    self.assertGreater(state["native_combined_k"],0)
                    self.assertFalse(state["individual_neutron_histories_proven_independent"])

    def test_original_19_element_core_is_not_reconstructed_operating_nerva(self):
        r=self.report
        self.assertTrue(r["same_conceptual_19_element_geometry_all_cases"])
        self.assertFalse(r["historical_NASA_NERVA_reactor_validated"])
        self.assertFalse(r["source_stationarity_demonstrated"])
        self.assertFalse(r["physical_reactor_watts_or_thermal_feedback_validated"])
        historic=r["historical_nasa_geometry_fidelity_diagnostic"]
        self.assertFalse(historic["proposed_snre_bulk_geometry_matches"])
        self.assertFalse(historic["material_fuel_matrix_is_verified_historical_nerva_composition"])


if __name__=="__main__":
    unittest.main(verbosity=2)
