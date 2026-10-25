"""Read actual executed OpenMC summary.h5, XML and nuclear data source audit.

No fake statepoint, no substituted isotope library; original 2026-10-23
GitHub Actions artifact 12492499186 MUST be accessible. Mathematical
verification and tamper detection do not qualify hot NERVA materials.
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

from .real_native_material_xs_audit import (
    audit_native_material_xs, _xml_materials, _summary_materials_and_cells,
)


class ExecutedOpenMCMaterialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src=os.environ.get("NUKE_REAL_CORE_EVIDENCE_PATH")
        if not src or not Path(src).exists():
            raise RuntimeError("Original OpenMC HDF5 evidence required; synthetic replacements forbidden.")
        cls.source=Path(src)

    def test_native_HDF5_and_XML_geometry_materials_reconcile(self):
        r=audit_native_material_xs(self.source)
        self.assertEqual(r["actual_openmc_seeds"],4)
        self.assertEqual(r["declared_material_count"],9)
        self.assertEqual(r["executed_reachable_material_count"],4)
        self.assertEqual(r["unused_material_ids"],[23,24,25,26,27])
        self.assertEqual([x["seed"] for x in r["per_seed_native_evidence"]],
                         [13007,23011,33013,43019])
        self.assertTrue(all(x["reachable_material_ids"]==[19,20,21,22]
                            and x["material_filled_cells"]==60
                            for x in r["per_seed_native_evidence"]))

    def test_real_executed_nuclear_isotopes_are_a_subset_of_indexed_library(self):
        r=audit_native_material_xs(self.source)
        self.assertEqual(r["executed_reachable_neutron_isotope_count"],13)
        self.assertEqual(r["declared_neutron_isotope_file_count"],29)
        self.assertIn("U235",r["executed_reachable_neutron_isotopes"])
        self.assertIn("H1",r["executed_reachable_neutron_isotopes"])
        self.assertIn("Be9",r["executed_reachable_neutron_isotopes"])
        self.assertNotIn("B10",r["executed_reachable_neutron_isotopes"])

    def test_actual_single_cold_temperature_and_implicit_material_temperatures(self):
        r=audit_native_material_xs(self.source)
        np.testing.assert_allclose(
            r["available_executed_neutron_temperatures_K"],
            [293.6059129982851],rtol=0,atol=2e-10)
        self.assertEqual(r["executed_materials_with_explicit_temperature"],1)
        self.assertEqual(r["executed_materials_using_implicit_temperature"],3)
        temp=r["per_seed_native_evidence"][0]["explicit_temperature_by_material_K"]
        self.assertIsNone(temp["19"])
        self.assertAlmostEqual(temp["20"],293.6059129982851)
        self.assertIsNone(temp["21"])
        self.assertIsNone(temp["22"])

    def test_genuine_isotope_bytes_release_and_bound_scattering_remain_unqualified(self):
        r=audit_native_material_xs(self.source)
        self.assertFalse(r["actual_isotope_HDF5_binary_bytes_in_source_archive"])
        self.assertFalse(r["claimed_evaluated_XS_file_hashes_independently_reverified"])
        self.assertFalse(r["nuclear_data_release_edition_verified"])
        self.assertFalse(r["qualified_bound_scattering_and_hot_materials"])
        self.assertFalse(r["historical_hot_nerva_qualification"])

    def test_native_summary_material_inventory_is_not_self_report_only(self):
        if not self.source.is_file():
            self.skipTest("Mutation regression needs ZIP constructed from actual source")
        with zipfile.ZipFile(self.source) as z:
            xml=_xml_materials(z.read("seed_13007/materials.xml"))
            orig=z.read("seed_13007/summary.h5")
        reachable,cells=_summary_materials_and_cells(orig,xml)
        self.assertEqual(reachable,[19,20,21,22])
        self.assertEqual(len(cells),60)

    def _mutate(self,entry,change):
        if not self.source.is_file():
            self.skipTest("Tamper detection runs against ZIP repackaged from genuine original files")
        with zipfile.ZipFile(self.source) as z:
            with tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp)/"corrupted-physical-evidence.zip"
                with zipfile.ZipFile(output,"w",compression=zipfile.ZIP_DEFLATED) as new:
                    for name in z.namelist():
                        data=z.read(name)
                        if name==entry:
                            data=change(data)
                        new.writestr(name,data)
                with self.assertRaises(ValueError):
                    audit_native_material_xs(output)

    def test_forged_material_temperature_in_report_is_detected(self):
        def forge(raw):
            report=json.loads(raw)
            report["physical_materials"][0]["explicit_material_temperature_K"]=800.
            return json.dumps(report).encode()
        self._mutate("core_material_temperature_qualification.json",forge)

    def test_modified_native_hdf5_summary_material_assignment_is_detected(self):
        def forge(raw):
            data=io.BytesIO(raw)
            with h5py.File(data,"r+") as h5:
                h5["materials/material 19/name"][...]=b"INCONSISTENT FUEL MATERIAL"
            return data.getvalue()
        self._mutate("seed_13007/summary.h5",forge)

    def test_forged_evaluated_temperature_grid_is_detected(self):
        def forge(raw):
            item=json.loads(raw)
            for row in item["nuclides"]:
                row["available_temperatures_K"]=[293.6059129982851,800.]
            return json.dumps(item).encode()
        self._mutate("neutron_temperature_coverage_audit.json",forge)

    def test_executed_xml_material_mutation_is_detected(self):
        def forge(raw):
            return raw.replace(b'nuclide name="U235"',b'nuclide name="U233"')
        self._mutate("seed_13007/materials.xml",forge)

    def test_missing_archive_refuses_fake_physics(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,"missing REAL OpenMC evidence"):
                audit_native_material_xs(tmp)


if __name__=="__main__":
    unittest.main(verbosity=2)
