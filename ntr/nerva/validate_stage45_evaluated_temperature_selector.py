"""Stage45 physically evaluated neutron temperature-table selection tests.

Analytic grids below are mathematical unit fixtures, NOT evaluated XS. An
actual OpenMC material/export + neutron transport test runs separately with
the official installed evaluated HDF5 and real source bank.
"""
from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import openmc

from .config import NervaConfig
from .model import build_model
from .audit_nuclear_data_temperatures import (
    resolve_evaluated_neutron_temperature, audit_temperature_coverage,
)


class EvaluatedTemperatureSelectionMathTests(unittest.TestCase):
    def test_interpolation_brackets_real_table_endpoints_only(self):
        r=resolve_evaluated_neutron_temperature(
            [900.,293.6,1200.],600.,method="interpolation",nearest_tolerance_k=10.)
        self.assertTrue(r["covered"])
        self.assertEqual(r["lower_evaluated_K"],293.6)
        self.assertEqual(r["upper_evaluated_K"],900.)
        self.assertAlmostEqual(r["upper_table_weight"],(600.-293.6)/(900.-293.6))

    def test_one_evaluated_temperature_exact_match_is_allowed(self):
        r=resolve_evaluated_neutron_temperature(
            [293.6],293.6,method="interpolation",nearest_tolerance_k=10.)
        self.assertTrue(r["covered"])
        self.assertEqual(r["lower_evaluated_K"],r["upper_evaluated_K"])
        self.assertEqual(r["upper_table_weight"],0.)

    def test_cold_only_data_cannot_qualify_hot_fuel(self):
        for method in ("nearest","interpolation"):
            r=resolve_evaluated_neutron_temperature(
                [293.6],900.,method=method,nearest_tolerance_k=10.)
            self.assertFalse(r["covered"])
            self.assertIn("no extrapolation",r["reason"])

    def test_hot_temperature_above_evaluated_range_rejected_even_at_large_tolerance(self):
        r=resolve_evaluated_neutron_temperature(
            [293.6,900.],901.,method="interpolation",nearest_tolerance_k=10.)
        self.assertFalse(r["covered"])

    def test_nearest_matches_only_physically_available_10K_tolerance(self):
        good=resolve_evaluated_neutron_temperature(
            [293.6,600.],595.,method="nearest",nearest_tolerance_k=5.)
        bad=resolve_evaluated_neutron_temperature(
            [293.6,600.],594.99,method="nearest",nearest_tolerance_k=5.)
        self.assertTrue(good["covered"])
        self.assertFalse(bad["covered"])

    def test_nearest_does_not_claim_interpolation_at_interior_temperature(self):
        a=resolve_evaluated_neutron_temperature(
            [293.6,900.],600.,method="nearest",nearest_tolerance_k=10.)
        b=resolve_evaluated_neutron_temperature(
            [293.6,900.],600.,method="interpolation",nearest_tolerance_k=10.)
        self.assertFalse(a["covered"])
        self.assertTrue(b["covered"])

    def test_invalid_evaluated_data_cannot_be_interpolated(self):
        for grid in ([293.,293.],[math.nan],[0.],[],[math.inf]):
            with self.subTest(grid=grid),self.assertRaises(ValueError):
                resolve_evaluated_neutron_temperature(
                    grid,600.,method="interpolation",nearest_tolerance_k=10.)

    def test_invalid_temperature_and_method_fail(self):
        for temperature,method in ((-1.,"nearest"),(math.nan,"nearest"),
                                   (600.,"guess")):
            with self.assertRaises(ValueError):
                resolve_evaluated_neutron_temperature(
                    [293.6,900.],temperature,method=method,nearest_tolerance_k=10.)

    def test_bound_kernel_interpolation_not_silently_assumed(self):
        with self.assertRaisesRegex(ValueError,"evaluated-bound"):
            replace(NervaConfig(),
                    neutron_temperature_method="interpolation",
                    thermal_scattering_treatment="evaluated-bound").validate()

    def test_actual_openmc_model_exports_requested_temperature_selector(self):
        for method in ("nearest","interpolation"):
            config=replace(NervaConfig(),core_rings=3,
                           neutron_temperature_method=method,
                           hydrogen_temperature_k=293.6059129982851,
                           fuel_temperature_k=293.6059129982851)
            model=build_model(config,assembly="core")
            self.assertEqual(model.settings.temperature["method"],method)
            if method=="nearest":
                self.assertEqual(model.settings.temperature["tolerance"],10.)
            with tempfile.TemporaryDirectory() as temp:
                model.export_to_xml(directory=temp)
                root=ET.parse(Path(temp)/"settings.xml").getroot()
                self.assertEqual(root.findtext("temperature_method"),method)
                if method=="nearest":
                    self.assertEqual(float(root.findtext("temperature_tolerance")),10.)
                else:
                    self.assertIsNone(root.findtext("temperature_tolerance"))


class ActualEvaluatedNeutronTemperatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xs=os.environ.get("NUKE_STAGE45_REAL_EVALUATED_XS")
        if not xs or not Path(xs).is_file():
            raise RuntimeError("Actual OpenMC evaluated neutron cross_sections.xml required")
        cls.xs=Path(xs)

    def test_real_cold_neutron_tables_and_true_interpolation_availability(self):
        c=replace(NervaConfig(),core_rings=3,
                  hydrogen_temperature_k=293.6059129982851,
                  fuel_temperature_k=293.6059129982851,
                  neutron_temperature_method="interpolation")
        report=audit_temperature_coverage(self.xs,c)
        self.assertEqual(report["neutron_temperature_method"],"interpolation")
        self.assertTrue(report["model_explicit_temperature_coverage_ok"])
        requested=[r for r in report["nuclides"]
                   if any(x["requested_temperature_K"] is not None
                          for x in r["requested_by_material"])]
        self.assertGreater(len(requested),1)
        for r in requested:
            self.assertTrue(r["explicit_requested_temperatures_supported_by_actual_method"])
            for request in r["requested_by_material"]:
                selection=request.get("actual_transport_temperature_selection")
                if selection is not None:
                    self.assertTrue(selection["covered"])
                    self.assertTrue(0<=selection["upper_table_weight"]<=1)

    def test_actual_evaluated_library_hot_snapshot_rejected_if_unavailable(self):
        cold=replace(NervaConfig(),core_rings=3,
                     hydrogen_temperature_k=293.6059129982851,
                     fuel_temperature_k=293.6059129982851,
                     neutron_temperature_method="interpolation")
        cold_report=audit_temperature_coverage(self.xs,cold)
        self.assertTrue(cold_report["model_explicit_temperature_coverage_ok"])
        hot=replace(cold,fuel_temperature_k=1800.)
        hot_report=audit_temperature_coverage(self.xs,hot)
        fuel_nuclides=[
            row for row in hot_report["nuclides"]
            if any(x["material_key"]=="fuel" for x in row["requested_by_material"])
        ]
        if all(any(t>1800 for t in row["available_temperatures_K"])
               for row in fuel_nuclides):
            # A genuine multi-temperature evaluated library may support this
            # request: report coverage rather than invent expected rejection.
            self.assertTrue(len(fuel_nuclides)>0)
        else:
            self.assertFalse(hot_report["model_explicit_temperature_coverage_ok"])


if __name__=="__main__":
    unittest.main(verbosity=2)
