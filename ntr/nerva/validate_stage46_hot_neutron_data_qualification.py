"""Stage46: physical neutron HDF5 kT coverage tests; never fabricated hot data."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import unittest

from .config import NervaConfig
from .audit_nuclear_data_temperatures import (
    audit_temperature_coverage,
    resolve_evaluated_neutron_temperature,
)
from .hot_neutron_temperature_data_qualification import (
    qualify_explicit_hot_neutron_data, REQUIRED_DECLARED_MATERIALS,
)


def _config(**kwargs):
    base=NervaConfig(
        core_rings=3,
        neutron_temperature_method="interpolation",
        fuel_temperature_k=900.,
        hydrogen_temperature_k=600.,
        bound_solid_temperature_k=600.,
        structural_solid_temperature_k=600.,
    )
    return replace(base,**kwargs)


def _fixture(config:NervaConfig,grid:list[float]=(293.6,600.,900.,1200.)):
    requests={
        "fuel":config.fuel_temperature_k,
        "hydrogen":config.hydrogen_temperature_k,
        "zrc":config.structural_solid_temperature_k,
        "reflector":config.bound_solid_temperature_k,
        "graphite":config.bound_solid_temperature_k,
        "zrh":config.bound_solid_temperature_k,
        "inconel":config.structural_solid_temperature_k,
        "b4c":config.structural_solid_temperature_k,
        "aluminum":config.structural_solid_temperature_k,
    }
    rows=[]
    for idx,(key,t) in enumerate(requests.items()):
        if t is None:
            selection=None
        else:
            selection=resolve_evaluated_neutron_temperature(
                list(grid),t,method=config.neutron_temperature_method,
                nearest_tolerance_k=10.)
        rows.append({
            "nuclide":f"dummy_{idx}",
            "available_temperatures_K":list(grid),
            "requested_by_material":[{
                "material_key":key,
                "requested_temperature_K":t,
                "actual_transport_temperature_selection":selection,
            }],
            "load_error":None,
        })
    return {
        "nuclides":rows,
        "neutron_temperature_method":config.neutron_temperature_method,
        "model_explicit_temperature_coverage_ok":all(
            v is not None and
            resolve_evaluated_neutron_temperature(
                list(grid),v,method=config.neutron_temperature_method,
                nearest_tolerance_k=10.)["covered"]
            for v in requests.values()
        ),
    }


class HotTemperatureQualificationNumericsTests(unittest.TestCase):
    def test_multiple_evaluated_tables_allow_explicit_snapshot_preflight_only(self):
        cfg=_config()
        report=qualify_explicit_hot_neutron_data(cfg,_fixture(cfg))
        self.assertTrue(report["qualified_hot_incident_neutron_data_support"])
        self.assertFalse(report["historical_NASA_NERVA_hot_reactor_validated"])
        self.assertFalse(report["physical_hydrogen_EOS_pressure_density_thermal_feedback_qualified"])
        self.assertFalse(report["hot_bound_thermal_scattering_qualified"])

    def test_single_cold_table_does_not_qualify_any_hot_isotope(self):
        cfg=_config()
        r=qualify_explicit_hot_neutron_data(cfg,_fixture(cfg,[293.6]))
        self.assertFalse(r["qualified_hot_incident_neutron_data_support"])
        self.assertEqual(len(r["neutron_isotopes_with_missing_hot_coverage"]),
                         len(REQUIRED_DECLARED_MATERIALS))

    def test_missing_explicit_structural_temperature_rejected(self):
        cfg=_config(structural_solid_temperature_k=None)
        r=qualify_explicit_hot_neutron_data(cfg,_fixture(cfg))
        self.assertFalse(r["qualified_hot_incident_neutron_data_support"])
        self.assertIn("zrc",r["missing_explicit_material_temperatures"])
        self.assertIn("aluminum",r["missing_explicit_material_temperatures"])

    def test_missing_explicit_bound_moderator_temperature_rejected(self):
        cfg=_config(bound_solid_temperature_k=None)
        r=qualify_explicit_hot_neutron_data(cfg,_fixture(cfg))
        self.assertFalse(r["qualified_hot_incident_neutron_data_support"])
        self.assertIn("zrh",r["missing_explicit_material_temperatures"])

    def test_nearest_method_is_not_mistaken_for_true_interpolation(self):
        cfg=_config(neutron_temperature_method="nearest")
        r=qualify_explicit_hot_neutron_data(cfg,_fixture(cfg))
        self.assertFalse(r["qualified_hot_incident_neutron_data_support"])

    def test_cold_fuel_is_not_mislabeled_hot(self):
        cfg=_config(fuel_temperature_k=293.6)
        r=qualify_explicit_hot_neutron_data(cfg,_fixture(cfg))
        self.assertFalse(r["qualified_hot_incident_neutron_data_support"])
        self.assertFalse(r["fuel_above_provisional_hot_test_threshold"])

    def test_duplicate_isotope_name_fails_closed(self):
        cfg=_config()
        a=_fixture(cfg)
        a["nuclides"].append(a["nuclides"][0].copy())
        with self.assertRaisesRegex(ValueError,"duplicate"):
            qualify_explicit_hot_neutron_data(cfg,a)

    def test_false_report_method_is_rejected(self):
        cfg=_config()
        a=_fixture(cfg)
        a["neutron_temperature_method"]="nearest"
        with self.assertRaisesRegex(ValueError,"disagree"):
            qualify_explicit_hot_neutron_data(cfg,a)

    def test_stale_requested_material_temperature_is_rejected(self):
        cfg=_config()
        a=_fixture(cfg)
        a["nuclides"][0]["requested_by_material"][0]["requested_temperature_K"]=950.
        with self.assertRaisesRegex(ValueError,"does not match"):
            qualify_explicit_hot_neutron_data(cfg,a)

    def test_structural_temperature_invalid_nonfinite(self):
        for t in (float("nan"),0.,-20.,math.inf):
            with self.subTest(t=t),self.assertRaises(ValueError):
                _config(structural_solid_temperature_k=t).validate()

    def test_actual_config_material_map_sets_all_explicit_solid_temperatures(self):
        from .materials import build_materials
        cfg=_config()
        materials=build_materials(cfg)
        self.assertEqual(set(materials),set(REQUIRED_DECLARED_MATERIALS))
        for key,material in materials.items():
            expected=(cfg.fuel_temperature_k if key=="fuel" else
                      cfg.hydrogen_temperature_k if key=="hydrogen" else
                      cfg.bound_solid_temperature_k if key in ("reflector","graphite","zrh")
                      else cfg.structural_solid_temperature_k)
            self.assertEqual(float(material.temperature),expected)


class ActualInstalledNeutronXSDoesNotQualifyHotCore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        xs=os.environ.get("NUKE_STAGE46_ACTUAL_EVALUATED_NEUTRON_CROSS_SECTIONS")
        if not xs or not Path(xs).is_file():
            raise RuntimeError("REAL evaluated neutron cross_sections.xml required")
        cls.xs=Path(xs)

    def test_native_cold_nndc_hdf5_does_not_fabricate_hot_evaluations(self):
        cfg=_config()
        real_audit=audit_temperature_coverage(self.xs,cfg)
        result=qualify_explicit_hot_neutron_data(cfg,real_audit)
        self.assertTrue(result["real_nuclear_data_evaluation_required"])
        # This GitHub OpenMC v0.15.3 Docker NNDC library only has cold
        # isotope evaluations; a future installed true multitemperature
        # library would need a different assertion for its actual data.
        self.assertFalse(result["qualified_hot_incident_neutron_data_support"])
        self.assertGreater(len(result["neutron_isotopes_with_missing_hot_coverage"]),0)
        self.assertFalse(result["historical_NASA_NERVA_hot_reactor_validated"])

    def test_native_cold_evaluated_isotopes_are_not_sufficient_at_cold_material_state(self):
        cfg=_config(fuel_temperature_k=293.6059129982851,
                    hydrogen_temperature_k=293.6059129982851,
                    bound_solid_temperature_k=293.6059129982851,
                    structural_solid_temperature_k=293.6059129982851)
        audit=audit_temperature_coverage(self.xs,cfg)
        result=qualify_explicit_hot_neutron_data(cfg,audit)
        self.assertTrue(result["all_declared_solid_and_coolant_temperatures_explicit"])
        self.assertFalse(result["fuel_above_provisional_hot_test_threshold"])
        self.assertFalse(result["qualified_hot_incident_neutron_data_support"])


if __name__=="__main__":
    unittest.main(verbosity=2)
