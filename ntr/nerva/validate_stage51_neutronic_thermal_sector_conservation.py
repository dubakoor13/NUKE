"""Stage51: transport CSG/19-sector thermal mass and energy QA tests.

Only a synthetic unit heat-density test is performed: 1e6 W/m³ is NOT
estimated reactor thermal power or nuclear energy deposition.
"""
from __future__ import annotations

from dataclasses import replace
import math
import os
import unittest

import numpy as np

from .config import NervaConfig
from .fuel_sector_geometry import (
    fuel_sector_coated_channel_areas_cm2,
    fuel_sector_equivalent_outer_radii_m,
    fuel_sector_fuel_areas_cm2,
    fuel_sector_total_areas_cm2,
)
from .real_nasa2015_fuel_tie_csg_volume_audit import (
    fuel_analytical_areas,
)
from .real_neutronic_thermal_sector_conservation import (
    fuel_thermal_volume_balance, verify_true_openmc_csg_ownership,
)


class FuelThermalCellVolumeNumericalTests(unittest.TestCase):
    def setUp(self):
        self.c=NervaConfig(core_rings=3)

    def test_exact_19_sector_total_area_equals_true_inner_hex(self):
        c=self.c
        t=fuel_sector_total_areas_cm2(c)
        self.assertEqual(t.shape,(19,))
        self.assertAlmostEqual(float(t.sum()),
                               math.sqrt(3)*c.inner_fuel_flat_to_flat_cm**2/2,
                               places=12)

    def test_coated_footprints_are_exactly_six_clipped_and_13_full_disks(self):
        c=self.c
        footprints=fuel_sector_coated_channel_areas_cm2(c)
        full=math.pi*c.coated_channel_radius_cm**2
        self.assertEqual(len(footprints),19)
        self.assertEqual(int(np.sum(footprints<full-1e-13)),6)
        self.assertEqual(int(np.sum(np.abs(footprints-full)<1e-13)),13)
        self.assertAlmostEqual(19*full-float(footprints.sum()),
                               0.0008995304697577,places=12)

    def test_sectors_sum_to_independent_openmc_exact_CSG_fuel_area(self):
        c=self.c
        sectors=fuel_sector_fuel_areas_cm2(c)
        actual=fuel_analytical_areas(c)["exact_2d_partition_areas_cm2"]["fuel_matrix"]
        self.assertAlmostEqual(float(sectors.sum()),actual,places=12)
        self.assertAlmostEqual(actual,2.0509842565980376,places=12)

    def test_old_sector_fuel_mass_undercount_is_quantified_not_suppressed(self):
        p=fuel_thermal_volume_balance(self.c)
        self.assertAlmostEqual(p["actual_missing_legacy_thermal_fuel_area_cm2"],
                               0.0008995304697577,places=12)
        self.assertAlmostEqual(p["actual_missing_legacy_thermal_volume_cm3"],
                               0.0008995304697577*132.,places=10)
        self.assertGreater(p["corrected_vs_original_fuel_mass_relative_change"],0)

    def test_each_equivalent_annulus_preserves_its_own_actual_CSG_fuel_mass(self):
        c=self.c
        radii=fuel_sector_equivalent_outer_radii_m(c)
        sectors=fuel_sector_fuel_areas_cm2(c)
        inner=c.coated_channel_radius_cm/100.
        rebuilt=np.pi*(radii*radii-inner*inner)*1e4
        self.assertTrue(np.allclose(rebuilt,sectors,atol=2e-13,rtol=0))
        self.assertTrue(np.all(radii>inner))

    def test_synthetic_uniform_source_keeps_all_sector_delta_T_identical(self):
        p=fuel_thermal_volume_balance(self.c)
        self.assertTrue(p["synthetic_thermal_energy_conservation_test"])
        self.assertAlmostEqual(p["synthetic_one_second_thermal_temperature_rise_K"],
                               1.,places=9)
        self.assertLess(abs(p["synthetic_thermal_energy_residual_j"]),1e-7)
        self.assertFalse(p["native_NERVA_reactor_thermal_power_or_temperature_verified"])

    def test_neutronic_thermal_equivalence_does_not_claim_real_hex_conduction(self):
        p=fuel_thermal_volume_balance(self.c)
        self.assertTrue(p["equivalent_annulus_preserves_mass_not_2D_hex_conduction_operator"])
        self.assertAlmostEqual(p["corrected_fuel_volume_m3"],
                               p["corrected_material_fuel_volume_cm3"]*1e-6,
                               places=14)

    def test_coating_escape_from_inner_fuel_bore_is_rejected(self):
        c=self.c
        with self.assertRaises(ValueError):
            fuel_sector_coated_channel_areas_cm2(
                replace(c,zrc_coating_thickness_cm=0.18))

    def test_coating_passages_overlap_or_outer_escape_are_rejected(self):
        c=self.c
        with self.assertRaises(ValueError):
            fuel_sector_coated_channel_areas_cm2(
                replace(c,coolant_bore_pitch_cm=0.26))

    def test_random_geometry_CSG_sampler_rejects_unbounded_point_request(self):
        with self.assertRaises(ValueError):
            verify_true_openmc_csg_ownership(self.c,random_points=0)
        with self.assertRaises(ValueError):
            verify_true_openmc_csg_ownership(self.c,random_points=50000)


class ActualOpenMCFuelCSGCellAndThermalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("NUKE_STAGE51_REAL_OPENMC_CSG")!="1":
            raise RuntimeError("Actual OpenMC Stage51 geometry acceptance must run")
        cls.cfg=NervaConfig(core_rings=3)
        cls.actual=verify_true_openmc_csg_ownership(cls.cfg,random_points=3500)
        cls.thermal=fuel_thermal_volume_balance(cls.cfg)

    def test_actual_OpenMC_fuel_cells_do_not_overlap_at_clipped_ZrC_rings(self):
        a=self.actual
        self.assertTrue(a[
            "all_actual_cells_unique_for_sampled_real_fuel_and_coating_points"])
        self.assertGreater(a["genuine_openmc_csg_cell_ownership_points_checked"],3500)
        self.assertEqual(len(a["six_analytically_proved_outer_channel_cap_sites"]),6)
        self.assertGreater(a["true_OpenMC_material_hit_counts"]["hydrogen"],0)
        self.assertGreater(a["true_OpenMC_material_hit_counts"]["zrc"],0)
        self.assertGreater(a["true_OpenMC_material_hit_counts"]["fuel"],0)

    def test_thermal_masses_equal_true_CSG_and_synthetic_energy_conserves(self):
        a=self.thermal
        self.assertAlmostEqual(a["exact_sum_of_19_thermal_sector_fuel_areas_cm2"],
                               a["actual_openmc_fuel_material_area_cm2"],
                               places=12)
        self.assertTrue(a["synthetic_thermal_energy_conservation_test"])
        self.assertFalse(a["native_NERVA_reactor_thermal_power_or_temperature_verified"])


if __name__=="__main__":
    unittest.main(verbosity=2)
