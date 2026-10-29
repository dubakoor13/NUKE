"""Stage76: physically consistent full-core thermal grid refinement tests.

All material tables, flow conditions and volumetric source densities are
synthetic numerical checks, NOT NERVA empirical thermal qualification.
These tests compare the SAME physical domain/power with changes only in
radial ZrC shells, angular sectors, and common H2/fuel axial cells.
"""
from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from .coating_mesh_convergence import (
    CoupledMeshResolution,UniformVolumetricHeating,
    METRICS,_compare,run_coupled_mesh_observation,
    run_coupled_mesh_convergence,
)
from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .fuel_hydrogen_coupled_transient import with_resolved_coating
from .validate_stage58_fuel_hydrogen_coupling import coupled,initial


BASE=CoupledMeshResolution(
    fuel_nx=20,fuel_ny=24,axial_zones=2,
    angular_sectors=8,radial_layers=2)
HEAT=UniformVolumetricHeating(
    fuel_w_m3=1.8e7,
    coating_w_m3=7.5e7,
    hydrogen_w_m3=2.0e6,
    dt_s=5e-6,
)


def physical_fixture(mesh:CoupledMeshResolution):
    """Build SAME physical single-fuel-element control volumes on each mesh."""
    model=coupled(
        film=1200.,nz=mesh.axial_zones,
        nx=mesh.fuel_nx,ny=mesh.fuel_ny)
    cp=TemperatureTable(
        [240.,300.,500.,800.,1100.],
        [690.,725.,800.,890.,1000.])
    model=with_resolved_coating(
        model,density_kg_m3=5000.,
        heat_capacity=cp,
        contact_resistance_m2_k_w=1e-5,
        axial_conduction_enabled=True,
        angular_sectors=mesh.angular_sectors,
        radial_layers=mesh.radial_layers)
    state=initial(model,tf=505.,th=395.)
    state=replace(state,coating_temperature_k=np.full(
        model.radial_coating.sector_mass_kg.shape,455.))
    return model,state


class CoupledThermalMeshRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study=run_coupled_mesh_convergence(
            physical_fixture,base_mesh=BASE,heating=HEAT,
            radial_counts=(2,3,4),
            angular_counts=(8,12,16),
            axial_counts=(2,3,4))

    def test_all_three_independent_refinement_axes_have_three_levels(self):
        s=self.study
        self.assertEqual(set(s.axes),{"radial","angular","axial"})
        self.assertEqual(s.axes["radial"].counts,(2,3,4))
        self.assertEqual(s.axes["angular"].counts,(8,12,16))
        self.assertEqual(s.axes["axial"].counts,(2,3,4))
        self.assertEqual(s.unique_cases_run,7)
        for axis in s.axes.values():
            self.assertEqual(len(axis.observations),3)
            self.assertEqual(set(axis.differences),set(METRICS))

    def test_exact_geometric_volume_and_source_normalization_in_all_meshes(self):
        s=self.study
        base=s.axes["radial"].observations[0]
        volumes=np.array([
            base.fuel_volume_m3,base.coating_volume_m3,base.hydrogen_volume_m3])
        for axis in s.axes.values():
            for obs in axis.observations:
                candidate=np.array([
                    obs.fuel_volume_m3,
                    obs.coating_volume_m3,
                    obs.hydrogen_volume_m3])
                np.testing.assert_allclose(candidate,volumes,rtol=5e-7,
                                           atol=5e-13)
                expected=(HEAT.fuel_w_m3*obs.fuel_volume_m3+
                          HEAT.coating_w_m3*obs.coating_volume_m3+
                          HEAT.hydrogen_w_m3*obs.hydrogen_volume_m3)*HEAT.dt_s
                self.assertAlmostEqual(
                    obs.total_nuclear_energy_j,expected,delta=2e-12)
                self.assertAlmostEqual(
                    obs.source_energy_from_ledger_j,
                    obs.total_nuclear_energy_j,delta=2e-11)

    def test_every_refinement_level_closes_three_temperature_first_law(self):
        s=self.study
        self.assertLess(s.max_relative_first_law_residual,5e-8)
        self.assertLess(s.maximum_nuclear_source_imbalance_j,2e-11)
        for axis in s.axes.values():
            for obs in axis.observations:
                self.assertLess(obs.absolute_first_law_residual_j,2e-7)
                self.assertLess(abs(obs.coating_first_law_residual_j),2e-7)
                self.assertLess(obs.radial_node_max_residual_j,3e-8)

    def test_peak_fuel_coating_and_local_wall_flux_are_measured_not_inferred(self):
        for axis in self.study.axes.values():
            for obs in axis.observations:
                self.assertGreater(obs.measurements["fuel_peak_k"],495.)
                self.assertGreater(obs.measurements["coating_peak_k"],430.)
                self.assertGreater(obs.measurements["hydrogen_peak_k"],390.)
                self.assertGreater(
                    obs.measurements["fuel_outer_wall_peak_abs_w_m2"],0.)
                self.assertGreater(
                    obs.measurements["hydrogen_inner_wall_peak_abs_w_m2"],0.)
                self.assertGreater(obs.wall_flux_face_count,0)
                self.assertTrue(all(math.isfinite(v)
                                    for v in obs.measurements.values()))

    def test_fixed_power_and_area_do_not_use_cell_count_as_heating_rate(self):
        baseline=self.study.axes["radial"].observations[0]
        for axis in self.study.axes.values():
            for obs in axis.observations:
                self.assertAlmostEqual(obs.total_nuclear_energy_j,
                                       baseline.total_nuclear_energy_j,
                                       delta=2e-8)
                self.assertGreater(obs.fuel_volume_m3,0.)
                self.assertGreater(obs.coating_volume_m3,0.)
                self.assertGreater(obs.hydrogen_volume_m3,0.)

    def test_observed_difference_ratios_match_actual_three_solutions(self):
        for axis in self.study.axes.values():
            for name in METRICS:
                outputs=[obs.measurements[name] for obs in axis.observations]
                delta=axis.differences[name]
                self.assertAlmostEqual(delta.coarse_to_mid_abs,
                                       abs(outputs[1]-outputs[0]),
                                       delta=1e-12*max(1.,abs(outputs[1])))
                self.assertAlmostEqual(delta.mid_to_fine_abs,
                                       abs(outputs[2]-outputs[1]),
                                       delta=1e-12*max(1.,abs(outputs[2])))
                self.assertTrue(delta.interpretation)
                self.assertFalse("proven convergence" in delta.interpretation)

    def test_refinement_isolate_each_geometry_axis(self):
        for name,axis in self.study.axes.items():
            a,b,c=(x.mesh for x in axis.observations)
            for field in ("fuel_nx","fuel_ny","axial_zones",
                          "angular_sectors","radial_layers"):
                values=[getattr(mesh,field) for mesh in (a,b,c)]
                varied={"radial":"radial_layers",
                        "angular":"angular_sectors",
                        "axial":"axial_zones"}[name]
                if field==varied:
                    self.assertTrue(values[0]<values[1]<values[2])
                else:
                    self.assertEqual(values,[values[0]]*3)

    def test_monotone_and_nonmonotone_classification_does_not_claim_richardson(self):
        convergent=_compare("fuel_peak_k",(500.,501.,501.4))
        self.assertTrue(convergent.decreasing_successive_differences)
        self.assertAlmostEqual(convergent.adjacent_difference_ratio,.4)
        self.assertIn("not proof",convergent.interpretation)
        nonmonotone=_compare("fuel_peak_k",(500.,501.,500.5))
        self.assertFalse(nonmonotone.decreasing_successive_differences)
        self.assertIn("unresolved",nonmonotone.interpretation)
        floor=_compare("fuel_peak_k",(500.,500.,500.))
        self.assertIsNone(floor.adjacent_difference_ratio)
        self.assertIsNone(floor.decreasing_successive_differences)

    def test_invalid_refinement_and_unphysical_source_rejected(self):
        with self.assertRaisesRegex(ValueError,"resolution"):
            CoupledMeshResolution(20,24,2,7,2)
        with self.assertRaisesRegex(ValueError,"W/m3"):
            UniformVolumetricHeating(1.,-1.,1.,5e-6)
        with self.assertRaisesRegex(ValueError,"dt"):
            UniformVolumetricHeating(1.,1.,1.,0.)
        with self.assertRaisesRegex(ValueError,"increasing"):
            run_coupled_mesh_convergence(
                physical_fixture,base_mesh=BASE,heating=HEAT,
                angular_counts=(8,8,16))
        with self.assertRaisesRegex(ValueError,"same physical base"):
            run_coupled_mesh_convergence(
                physical_fixture,base_mesh=BASE,heating=HEAT,
                radial_counts=(3,4,5))

    def test_actual_solver_mesh_mismatch_rejected_before_numerics(self):
        model,state=physical_fixture(BASE)
        wrong=CoupledMeshResolution(20,24,2,8,3)
        with self.assertRaisesRegex(ValueError,"actual solver geometry"):
            run_coupled_mesh_observation(model,state,wrong,HEAT)
        badstate=replace(
            state,coating_temperature_k=np.ones((2,19,8)))
        with self.assertRaisesRegex(ValueError,"initial coating"):
            run_coupled_mesh_observation(model,badstate,BASE,HEAT)


if __name__=="__main__":
    unittest.main(verbosity=2)
