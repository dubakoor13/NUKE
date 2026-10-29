"""Stage77 numerical verification of XY fuel embedded-boundary convergence.

The reactor model, physical volume and W/m3 loading are identical while
only the fuel transverse grid changes. This is a 3-level numerical
sensitivity exercise, not qualified historical NERVA thermal data.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from .coating_mesh_convergence import (
    CoupledMeshResolution, METRICS, _compare,
)
from .fuel_cutcell_xy_convergence import (
    run_fuel_xy_convergence,_continuous_fuel_temperature,
)
from .validate_stage76_coupled_mesh_convergence import (
    BASE, HEAT, physical_fixture,
)

GRID_PAIRS=((20,24),(28,32),(36,40))


class FullCoupledFuelXyRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study=run_fuel_xy_convergence(
            physical_fixture,base_mesh=BASE,
            heating=HEAT,xy_sizes=GRID_PAIRS)

    def test_three_physical_xy_refinements_keep_all_other_dimensions_fixed(self):
        study=self.study
        self.assertEqual(study.sizes,GRID_PAIRS)
        self.assertEqual(len(study.cases),3)
        for case,pair in zip(study.cases,GRID_PAIRS):
            self.assertEqual(
                (case.mesh.fuel_nx,case.mesh.fuel_ny),pair)
            self.assertEqual(case.mesh.axial_zones,BASE.axial_zones)
            self.assertEqual(case.mesh.angular_sectors,BASE.angular_sectors)
            self.assertEqual(case.mesh.radial_layers,BASE.radial_layers)
        self.assertEqual(set(study.metrics),set(METRICS))

    def test_real_fuel_cutcell_node_count_grows_with_refinement(self):
        counts=[case.physical_fuel_cutcell_count
                for case in self.study.cases]
        self.assertTrue(counts[0]<counts[1]<counts[2])
        for case in self.study.cases:
            self.assertGreater(case.physical_fuel_cutcell_count,25)
            self.assertGreater(case.minimum_fuel_cutcell_volume_m3,0.)
            self.assertGreater(case.fuel_volume_m3,0.)

    def test_exact_physical_fuel_inventory_independent_of_xy_cell_count(self):
        study=self.study
        volume=study.cases[0].fuel_volume_m3
        self.assertLess(
            study.max_fuel_inventory_discrepancy_m3,
            max(5e-12,5e-7*volume))
        for case in study.cases:
            self.assertAlmostEqual(
                case.fuel_volume_m3,volume,
                delta=max(5e-12,5e-7*volume))
            self.assertAlmostEqual(
                case.observation.fuel_volume_m3,
                case.fuel_volume_m3,delta=1e-12)

    def test_all_nineteen_physical_wall_contact_areas_preserved(self):
        original=self.study.cases[0].fuel_coating_contact_area_by_channel_m2
        self.assertEqual(original.shape,(19,))
        self.assertTrue(np.all(original>0.))
        self.assertLess(self.study.max_wall_contact_discrepancy_m2,4e-11)
        for case in self.study.cases:
            np.testing.assert_allclose(
                case.fuel_coating_contact_area_by_channel_m2,
                original,rtol=0,atol=4e-11)

    def test_identical_physical_nuclear_heating_with_different_cell_counts(self):
        first=self.study.cases[0].observation
        for case in self.study.cases:
            obs=case.observation
            self.assertAlmostEqual(
                obs.total_nuclear_energy_j,first.total_nuclear_energy_j,
                delta=2e-8)
            self.assertAlmostEqual(
                obs.total_nuclear_energy_j,
                obs.source_energy_from_ledger_j,delta=2e-11)
        self.assertLess(
            self.study.max_nuclear_source_rebin_difference_j,2e-11)

    def test_each_xy_mesh_closes_real_fuel_coating_gas_first_law(self):
        for case in self.study.cases:
            o=case.observation
            self.assertLess(o.relative_first_law_residual,5e-8)
            self.assertLess(o.absolute_first_law_residual_j,3e-7)
            self.assertLess(abs(o.coating_first_law_residual_j),3e-7)
            self.assertLess(o.radial_node_max_residual_j,3e-8)
        self.assertLess(self.study.max_relative_energy_residual,5e-8)

    def test_common_nonuniform_continuous_initial_temperature_is_real_xy(self):
        for n_x,n_y in GRID_PAIRS:
            mesh=CoupledMeshResolution(
                n_x,n_y,BASE.axial_zones,
                BASE.angular_sectors,BASE.radial_layers)
            model,old=physical_fixture(mesh)
            s=_continuous_fuel_temperature(model,old)
            arr=s.fuel_temperature_k
            self.assertEqual(arr.shape,
                             (BASE.axial_zones,model.solid.transverse.n_fuel))
            self.assertGreater(float(np.ptp(arr[0])),12.)
            xy=np.asarray(model.solid.transverse.cell_centers_cm)
            # The gradient is intentionally nonaxisymmetric: x and y
            # both influence the same physical smooth field.
            self.assertGreater(
                abs(float(np.corrcoef(arr[0],xy[:,0])[0,1])),.1)
            self.assertGreater(
                abs(float(np.corrcoef(arr[0],xy[:,1])[0,1])),.1)

    def test_actual_peak_temperature_and_local_wall_flux_respond_to_xy_grid(self):
        for metric in ("fuel_peak_k","fuel_volume_mean_k",
                       "fuel_outer_wall_peak_abs_w_m2",
                       "fuel_to_coating_energy_j"):
            values=[case.observation.measurements[metric]
                    for case in self.study.cases]
            self.assertTrue(all(math.isfinite(x) for x in values))
            self.assertGreater(values[-1],0.)
            delta=self.study.metrics[metric]
            self.assertAlmostEqual(
                delta.coarse_to_mid_abs,abs(values[1]-values[0]),
                delta=1e-10*max(1.,abs(values[1])))
            self.assertAlmostEqual(
                delta.mid_to_fine_abs,abs(values[2]-values[1]),
                delta=1e-10*max(1.,abs(values[2])))
            self.assertTrue("not proof" in delta.interpretation or
                            "unresolved" in delta.interpretation or
                            "reporting floor" in delta.interpretation)

    def test_refinement_flags_nonmonotone_not_false_grid_independence(self):
        self.assertIn("NOT imply",self.study.interpretation)
        for metric in METRICS:
            result=self.study.metrics[metric]
            self.assertGreaterEqual(result.coarse_to_mid_abs,0.)
            self.assertGreaterEqual(result.mid_to_fine_abs,0.)
        # A nonmonotonic embedded-boundary metric cannot be asserted to
        # have an extrapolated physical continuum limit.
        sample=_compare("fuel_outer_wall_peak_abs_w_m2",
                        (1e6,1.1e6,1.05e6))
        self.assertFalse(sample.decreasing_successive_differences)
        self.assertIn("unresolved",sample.interpretation)

    def test_bad_xy_sequences_fail_before_creating_solver_model(self):
        errors=[
            (((20,24),(20,32),(36,40)),"increase both"),
            (((20,24),(28,32),(24,40)),"increase both"),
            (((21,24),(28,32),(36,40)),"start from"),
            (((20,24),(28,32)),"three fuel XY"),
            (((20,24),(28,32),(36,-1)),"three fuel XY"),
        ]
        def no_factory(_mesh):
            raise AssertionError("incorrect grid request reached model builder")
        for sizes,pattern in errors:
            with self.subTest(sizes=sizes):
                with self.assertRaisesRegex(ValueError,pattern):
                    run_fuel_xy_convergence(
                        no_factory,base_mesh=BASE,
                        heating=HEAT,xy_sizes=sizes)


if __name__=="__main__":
    unittest.main(verbosity=2)
