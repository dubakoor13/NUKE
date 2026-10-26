"""Stage 28 validates strict real-data neutron-only heating rejection.

Requires original archived four-seed OpenMC binaries; test doubles are only
used by the separate Stage 10 mathematical reader regression suite.
"""
from __future__ import annotations

import os
from pathlib import Path
import unittest

from .audit_original_neutron_only_thermal_gate import audit_original_neutron_only_thermal_gate


class OriginalColdNeutronOnlyHeatingRejectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = os.environ.get("NUKE_REAL_CORE_EVIDENCE_PATH")
        if not raw or not Path(raw).exists():
            raise RuntimeError("Original OpenMC four-seed HDF5 evidence is mandatory")
        cls.path = Path(raw)

    def test_native_original_positive_heating_does_not_cross_coupled_power_gate(self):
        report = audit_original_neutron_only_thermal_gate(self.path)
        self.assertEqual(report["seeds_checked"], 4)
        self.assertEqual([x["seed"] for x in report["original_hdf5_observations"]],
                         [13007,23011,33013,43019])
        for item in report["original_hdf5_observations"]:
            self.assertGreater(item["ordinary_heating_ev_per_source"], 0)
            self.assertFalse(item["photon_transport"])
            self.assertTrue(item["all_heating_local_raw_moments_zero"])
            self.assertTrue(item["total_core_thermal_power_bridge_rejected"])

    def test_physically_unqualified_reactor_power_does_not_promote(self):
        report = audit_original_neutron_only_thermal_gate(self.path)
        self.assertFalse(report["real_coupled_neutron_photon_transport_performed"])
        self.assertFalse(report["positive_ordinary_heating_is_complete_deposited_power"])
        self.assertFalse(report["qualified_local_heating_available"])
        self.assertFalse(report["hot_core_watt_normalization_allowed"])
        self.assertFalse(report["historical_NERVA_qualified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
