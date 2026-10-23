"""Dependency-free physical/numerical checks for NESS boundary exchange."""
from __future__ import annotations

import unittest

from .ness_system_balance import (
    HydrogenPort, audit_control_volume, btu_lbm_to_j_kg,
    horsepower_to_w, lbm_s_to_kg_s, psi_to_pa, rankine_to_kelvin,
)


class NESSBoundaryTests(unittest.TestCase):
    def test_nozzle_coolant_heat_addition(self):
        # Synthetic 10 kg/s coolant leg, 100 kW heat absorbed.
        result = audit_control_volume(
            [HydrogenPort("cold", 10.0, 1e6)],
            [HydrogenPort("warm", 10.0, 1.01e6)],
            heat_to_fluid_w=100_000.0,
        )
        self.assertTrue(result.passed)
        self.assertAlmostEqual(result.energy_residual_w, 0.0, places=6)

    def test_two_branch_enthalpy_mixing(self):
        # Synthetic reflector/tie junction with independently specified
        # outlet enthalpy; a correct mix conserves total mass AND energy.
        result = audit_control_volume(
            [HydrogenPort("tie", 2.0, 2e6), HydrogenPort("reflector", 8.0, 1e6)],
            [HydrogenPort("mixed", 10.0, 1.2e6)],
        )
        self.assertTrue(result.passed)

    def test_detect_inconsistent_turbine_power(self):
        broken = audit_control_volume(
            [HydrogenPort("turbine_in", 1.0, 2e6)],
            [HydrogenPort("turbine_out", 1.0, 1.5e6)],
            shaft_to_fluid_w=-400_000.0,  # should be -500 kW
        )
        self.assertFalse(broken.energy_ok)
        self.assertAlmostEqual(broken.energy_residual_w, 100_000.0)

    def test_dynamic_energy_storage(self):
        result = audit_control_volume(
            [HydrogenPort("in", 5.0, 1e6)],
            [HydrogenPort("out", 5.0, 1e6)],
            heat_to_fluid_w=200_000.0, storage_rate_w=200_000.0,
        )
        self.assertTrue(result.passed)

    def test_mass_error_even_with_enthalpy_zero(self):
        result = audit_control_volume(
            [HydrogenPort("in", 2.0, 0.0)],
            [HydrogenPort("out", 2.01, 0.0)],
        )
        self.assertFalse(result.mass_ok)
        self.assertTrue(result.energy_ok)
        self.assertFalse(result.passed)

    def test_enthalpy_zero_invariance_when_mass_closes(self):
        a = audit_control_volume(
            [HydrogenPort("in", 1., -1e6)],
            [HydrogenPort("out", 1., -0.9e6)],
            heat_to_fluid_w=100_000.,
        )
        b = audit_control_volume(
            [HydrogenPort("in", 1., 9e6)],
            [HydrogenPort("out", 1., 9.1e6)],
            heat_to_fluid_w=100_000.,
        )
        self.assertTrue(a.passed)
        self.assertTrue(b.passed)
        self.assertAlmostEqual(a.energy_residual_w, b.energy_residual_w, places=6)

    def test_invalid_ports_and_tolerances(self):
        with self.assertRaises(ValueError):
            HydrogenPort("negative", -1., 1.)
        with self.assertRaises(ValueError):
            HydrogenPort("invalid", 1., float("nan"))
        with self.assertRaises(ValueError):
            audit_control_volume(
                [HydrogenPort("in", 1., 1.)],
                [HydrogenPort("out", 1., 1.)],
                power_rtol=-1.,
            )

    def test_explicit_unit_conversions(self):
        self.assertAlmostEqual(lbm_s_to_kg_s(1.), 0.45359237)
        self.assertAlmostEqual(psi_to_pa(1.), 6894.757293168)
        self.assertAlmostEqual(rankine_to_kelvin(540.), 300.)
        self.assertAlmostEqual(btu_lbm_to_j_kg(1.), 2326.0, delta=0.1)
        self.assertAlmostEqual(horsepower_to_w(1.), 745.69987158)


if __name__ == "__main__":
    unittest.main()
