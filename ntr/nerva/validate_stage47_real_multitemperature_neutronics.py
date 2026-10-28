"""Stage47 exact C++ stochastic interpolation and ORIGINAL neutron HDF5 QA.

Mathematical fixtures below are explicitly NOT actual isotope evaluations.
The acceptance tests require separately downloaded, evaluated nuclear HDF5
and original executed OpenMC C++ statepoints.
"""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

from .config import NervaConfig
from .audit_nuclear_data_temperatures import audit_temperature_coverage
from .real_evaluated_multitemperature_transport import (
    OFFICIAL_CPP_BLOB_SHA,
    INTERMEDIATE_K,
    audit_real_multitemperature_cpp,
    read_original_openmc_transport,
    strict_two_table_coverage,
    verify_upstream_cpp_temperature_sampling,
)


def fixture_audit(hdf5_path:Path, *, grid=(293.6,600.,900.,1200.),
                  method="interpolation", target=750.)->dict:
    return {
        "neutron_temperature_method":method,
        "model_explicit_temperature_coverage_ok":True,
        "nuclides":[{
            "nuclide":"dummy_mathematical_fixture_not_evaluated",
            "hdf5_path":str(hdf5_path),
            "load_error":None,
            "available_temperatures_K":list(grid),
            "requested_by_material":[{
                "material_key":"fuel",
                "requested_temperature_K":target,
                "actual_transport_temperature_selection":{
                    "covered":True,
                    "lower_evaluated_K":600.,
                    "upper_evaluated_K":900.,
                    "upper_table_weight":0.5,
                },
            }],
        }],
    }


class MultitemperatureInterpolationMathTests(unittest.TestCase):
    def test_mathematical_600_to_900_k_sampling_weight_at_750(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp=Path(tmp)/"synthetic_unit_fixture_not_evaluated.h5"
            fp.write_bytes(b"NOT A REAL EVALUATED NEUTRON HDF5 FILE")
            rows=strict_two_table_coverage(fixture_audit(fp),expected_K=750.)
            self.assertEqual(rows["verified_unique_evaluated_isotopes"],1)
            self.assertEqual(rows["material_isotope_table_details"][0][
                "upper_table_selection_probability"],0.5)
            self.assertEqual(len(rows["material_isotope_table_details"][0][
                "evaluated_isotope_HDF5_sha256"]),64)

    def test_single_temperature_never_treated_as_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp=Path(tmp)/"unit.h5"
            fp.write_bytes(b"NOT EVALUATED")
            with self.assertRaisesRegex(ValueError,"multi-temperature"):
                strict_two_table_coverage(
                    fixture_audit(fp,grid=(293.6,)),expected_K=750.)

    def test_exact_temperature_endpoint_not_a_two_table_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp=Path(tmp)/"unit.h5"
            fp.write_bytes(b"NOT EVALUATED")
            audit=fixture_audit(fp)
            audit["nuclides"][0]["requested_by_material"][0][
                "requested_temperature_K"]=600.
            with self.assertRaisesRegex(ValueError,"material neutron temperature"):
                strict_two_table_coverage(audit,expected_K=750.)

    def test_uncalibrated_temperature_weight_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp=Path(tmp)/"unit.h5"
            fp.write_bytes(b"NOT EVALUATED")
            audit=fixture_audit(fp)
            audit["nuclides"][0]["requested_by_material"][0][
                "actual_transport_temperature_selection"]["upper_table_weight"]=0.51
            with self.assertRaisesRegex(ValueError,"no two distinct"):
                strict_two_table_coverage(audit,expected_K=750.)

    def test_nearest_selector_is_not_real_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp=Path(tmp)/"unit.h5"
            fp.write_bytes(b"NOT EVALUATED")
            with self.assertRaisesRegex(ValueError,"wrong selector"):
                strict_two_table_coverage(
                    fixture_audit(fp,method="nearest"),expected_K=750.)

    def test_unqualified_hot_nuclear_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp=Path(tmp)/"unit.h5"
            fp.write_bytes(b"NOT EVALUATED")
            data=fixture_audit(fp)
            data["model_explicit_temperature_coverage_ok"]=False
            with self.assertRaisesRegex(ValueError,"unqualified"):
                strict_two_table_coverage(data,expected_K=750.)

    def test_upstream_code_without_official_blob_sha_cannot_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"nuclide.cpp"
            p.write_text("if (f > prn(p.current_seed()))",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Git blob SHA"):
                verify_upstream_cpp_temperature_sampling(p)

    def test_no_executed_openmc_native_hdf5_cannot_be_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                read_original_openmc_transport(
                    Path(tmp)/"fake_original",Path(tmp)/"fake_xs.xml",temp=750.)


class ActualEvaluatedMultitemperatureOpenMCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=os.environ.get("NUKE_STAGE47_ORIGINAL_MULTI_KT_RUNS")
        xs=os.environ.get("NUKE_STAGE47_GENUINE_MULTI_KT_XS_INDEX")
        cpp=os.environ.get("NUKE_STAGE47_PINNED_CPP_SOURCE")
        if not root or not xs or not cpp:
            raise RuntimeError(
                "Stage47 requires real official multi-kT neutron HDF5, "
                "original OpenMC v0.15.3 neutron statepoints, pinned C++."
            )
        cls.report=audit_real_multitemperature_cpp(
            Path(root),Path(xs),Path(cpp))

    def test_all_original_hdf5_and_evaluated_two_temperature_interpolation(self):
        r=self.report
        self.assertEqual(r["original_evaluated_neutron_source_runs"],2)
        self.assertTrue(r["temperature_only_fixed_composition_and_geometry"])
        self.assertTrue(r["original_openmc_v0153_temperature_sampling_cpp_provenance"][
            "stochastic_evaluated_table_sampling_confirmed"])
        actual=r["real_multi_temperature_evaluated_neutron_brackets"]
        self.assertGreater(actual["verified_unique_evaluated_isotopes"],0)
        self.assertTrue(actual[
            "real_evaluated_tables_on_both_sides_of_each_explicit_material_temperature"])
        for row in actual["material_isotope_table_details"]:
            self.assertTrue(row["genuine_lower_table_K"]<750.<row["genuine_upper_table_K"])
            self.assertTrue(0<row["upper_table_selection_probability"]<1)
            self.assertEqual(len(row["evaluated_isotope_HDF5_sha256"]),64)

    def test_real_neutron_temperature_only_runs_not_qualified_hot_reactor(self):
        r=self.report
        self.assertFalse(r["historical_NERVA_reactor_temperature_coefficient_qualified"])
        self.assertFalse(r["statistical_source_stationarity_or_uncertainty_qualified"])
        self.assertFalse(r["coolant_EOS_fixed_density_temperature_feedback_qualified"])
        self.assertFalse(r["reactor_watts_or_hot_NASA_NERVA_model_qualified"])
        self.assertEqual(len(r["source_temperature_cases"]),2)


if __name__=="__main__":
    unittest.main(verbosity=2)
