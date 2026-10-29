"""Stage50 exact NASA-derived fuel/tie geometry, original OpenMC CSG checks."""
from __future__ import annotations

from dataclasses import replace
import math
import os
import unittest

from .config import NervaConfig
from .layout import outer_channel_margin_cm
from .real_nasa2015_fuel_tie_csg_volume_audit import (
    audit_one_historical_element_reference,
    fuel_analytical_areas,
    tie_analytical_areas,
    actual_openmc_csg_classification,
    _fuel_expected,
    _tie_expected,
    hex_area,
)


class HistoricalFuelTieAnalyticMathTests(unittest.TestCase):
    def setUp(self):
        self.c=NervaConfig(core_rings=3)

    def test_2015_reference_19_hydrogen_boress_50micron_coating(self):
        p=fuel_analytical_areas(self.c)
        self.assertEqual(p["fuel_coolant_channel_count"],19)
        self.assertAlmostEqual(p["channel_bore_diameter_cm"],0.2565)
        self.assertAlmostEqual(p["channel_center_pitch_cm"],0.40894)
        self.assertAlmostEqual(p["ZrC_outer_nominal_thickness_cm"],0.005)
        self.assertAlmostEqual(p["actual_config_fuel_active_length_cm"],132.)

    def test_exact_fuel_hydrogen_and_both_coating_area_closure(self):
        p=fuel_analytical_areas(self.c)
        a=p["exact_2d_partition_areas_cm2"]
        self.assertAlmostEqual(sum(a.values()),p["hex_external_total_cross_section_cm2"],places=12)
        # The REAL OpenMC y-oriented inner hex clips 6 channel coating
        # rings by 0.00363 cm at the flats; do not double count ZrC.
        cap=p["area_of_ring_outside_inner_hex_but_inside_exterior_ZrC_cm2"]
        self.assertEqual(p[
            "number_outer_channel_coating_rings_clipped_by_actual_inner_hex"],6)
        self.assertGreater(cap,0)
        self.assertAlmostEqual(p["negative_channel_coating_to_inner_hex_margin_cm"],
                               -0.00363,places=12)
        self.assertAlmostEqual(a["fuel_matrix"],2.0500847261282797+cap,places=11)
        self.assertAlmostEqual(a["hydrogen_coolant_bores"],0.9817891339999295,places=12)
        self.assertAlmostEqual(a["ZrC_channel_coatings"],0.07804501549680462-cap,places=11)
        self.assertAlmostEqual(a["ZrC_external_hex_coating"],0.03290896534380883,places=12)

    def test_2d_to_3d_hydrogen_volume_and_wetted_surface_consistent(self):
        p=fuel_analytical_areas(self.c)
        self.assertAlmostEqual(p["exact_active_length_volumes_cm3"]["hydrogen_coolant_bores"],
                               math.pi*self.c.channel_radius_cm**2*19*132.)
        self.assertAlmostEqual(p["hydrogen_channel_wetted_surface_cm2"],
                               19*math.pi*self.c.coolant_bore_diameter_cm*132.)
        self.assertAlmostEqual(sum(p["exact_active_length_volumes_cm3"].values()),
                               p["total_element_volume_cm3"],places=9)

    def test_existing_fast_layout_matches_actual_vertical_flat_openmc_orientation(self):
        # This former helper incorrectly returned a +0.1059 cm clearance
        # by using the flat-horizontal normals. Actual OpenMC y hex
        # clips six channel coatings across its INNER hex by 0.00363cm.
        p=fuel_analytical_areas(self.c)
        self.assertAlmostEqual(outer_channel_margin_cm(self.c),-0.00363,places=12)
        self.assertAlmostEqual(outer_channel_margin_cm(self.c),
                               p["negative_channel_coating_to_inner_hex_margin_cm"],
                               places=12)
        self.assertGreater(
            outer_channel_margin_cm(self.c)+self.c.zrc_coating_thickness_cm,0.)

    def test_figure5_and_figure6_nominal_width_ambiguity_not_suppressed(self):
        p=fuel_analytical_areas(self.c)
        self.assertEqual(p["NASA_fig5_drawn_element_width_cm"],1.913)
        self.assertEqual(p["NASA_fig6_element_width_cm"],1.905)
        self.assertAlmostEqual(p["NASA_two_figure_nominal_width_discrepancy_cm"],0.008)
        self.assertTrue(p["model_uses_fig6_nominal_width_not_fabrication_tolerance"])

    def test_tie_cold_gap_supply_return_layers_positive_and_conserved(self):
        p=tie_analytical_areas(self.c)
        a=p["actual_tie_shell_areas_cm2"]
        self.assertEqual(len(a),11)
        self.assertAlmostEqual(sum(a.values()),hex_area(self.c.fuel_flat_to_flat_cm),places=12)
        self.assertGreater(p["hydrogen_supply_plus_return_flow_area_cm2"],0)
        self.assertGreater(p["cold_condition_void_gap_area_cm2"],0)
        self.assertTrue(p["original_nasa_figure_cold_gaps_are_not_hot_thermal_expansion"])

    def test_unphysical_channel_overlap_or_escape_rejected(self):
        for kw in ({"coolant_bore_diameter_cm":0.405},
                   {"zrc_coating_thickness_cm":0.18}):
            with self.subTest(kw=kw),self.assertRaises(ValueError):
                fuel_analytical_areas(replace(self.c,**kw))

    def test_inverted_tie_radial_gap_rejected_not_silently_clipped(self):
        c=replace(self.c,tie_zrc_inner_diameter_cm=1.3)
        with self.assertRaisesRegex(ValueError,"overlap|negative|strictly ordered"):
            tie_analytical_areas(c)

    def test_analytical_hex_inside_channel_and_coating_membership(self):
        c=self.c
        self.assertEqual(_fuel_expected(0.,0.,c),"hydrogen")
        self.assertEqual(_fuel_expected(c.channel_radius_cm+0.0025,0.,c),"zrc")
        self.assertEqual(_fuel_expected(0.2,0.,c),"fuel")
        self.assertEqual(_fuel_expected(c.fuel_flat_to_flat_cm/2-0.0025,0.,c),"zrc")
        with self.assertRaisesRegex(ValueError,"outside"):
            _fuel_expected(0.,self.c.fuel_flat_to_flat_cm,c)

    def test_analytical_tie_supply_moderator_void_and_graphite_classification(self):
        c=self.c
        self.assertEqual(_tie_expected(0.,0.,c),"hydrogen")
        self.assertEqual(_tie_expected(0.25,0.,c),"inconel")
        self.assertEqual(_tie_expected(0.263,0.,c),"void")
        self.assertEqual(_tie_expected(0.4,0.,c),"zrh")
        self.assertEqual(_tie_expected(0.60,0.,c),"hydrogen")
        self.assertEqual(_tie_expected(0.81,0.,c),"void")
        self.assertEqual(_tie_expected(0.85,0.,c),"graphite")
        self.assertEqual(_tie_expected(c.fuel_flat_to_flat_cm/2-0.0025,0.,c),"zrc")

    def test_single_geometry_reference_does_not_claim_historical_k_watts(self):
        result=audit_one_historical_element_reference(c=self.c,sample_count=200)
        self.assertTrue(result["original_three_ring_19_fuel_element_surrogate"])
        self.assertFalse(result["material_atom_number_fractions_are_historical_NASA_verified"])
        self.assertFalse(result["physical_coupled_hot_NERVA_reactor_validated"])
        self.assertFalse(result["fuel_element_geometry_entails_reactor_k_or_watts"])


class ActualTransportOpenMCCSGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("NUKE_STAGE50_ORIGINAL_OPENMC_CSG")!="1":
            raise RuntimeError("Stage50 native actual OpenMC CSG acceptance flag required")
        cls.result=audit_one_historical_element_reference(sample_count=1800)

    def test_original_fuel_element_and_tie_material_CSG_cells(self):
        r=self.result
        csg=r["original_OpenMC_CSG_analytic_material_membership"]
        self.assertTrue(csg[
            "original_openmc_csg_matches_independent_analytic_membership_at_all_points"])
        self.assertGreater(csg["actual_openmc_fuel_csg_samples_passed"],1900)
        self.assertGreater(csg["actual_openmc_tie_csg_samples_passed"],1800)
        self.assertEqual(set(csg["openmc_fuel_material_cell_hit_counts"]),
                         {"fuel","hydrogen","zrc"})
        self.assertEqual(set(csg["openmc_tie_material_and_void_cell_hit_counts"]),
                         {"hydrogen","inconel","zrh","zrc","void","graphite"})

    def test_actual_CSG_conservation_does_not_validate_historical_reactor_inventory(self):
        r=self.result
        self.assertTrue(r["element_level_nasa_2015_dimensions"]["fuel"][
            "analytical_area_conservation"])
        self.assertTrue(r["element_level_nasa_2015_dimensions"]["tie"][
            "coaxial_shell_area_conservation"])
        historical=r["separate_heritage_element_vs_proposed_SNRE_reference"]
        self.assertTrue(historical["heritage_element_geometry_matches"])
        self.assertFalse(historical["proposed_snre_bulk_geometry_matches"])
        self.assertEqual(historical["overall_historical_core_validation"],"NOT_VALIDATED")


if __name__=="__main__":
    unittest.main(verbosity=2)
