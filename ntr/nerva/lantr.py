"""Public NASA LANTR / bimodal-NTR performance reference model.

This module contains system-level performance data from NASA LANTR and BNTR
publications. It is intended for dashboard/system studies and deliberately does
not encode reactor fuel loading or criticality details.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


LBF_TO_N = 4.4482216152605
LBM_TO_KG = 0.45359237
PSI_TO_PA = 6894.757293168


@dataclass(frozen=True)
class LantrPoint:
    oxygen_hydrogen_ratio: float
    delivered_isp_s: float
    thrust_augmentation_factor: float
    thrust_n: float
    engine_mass_kg: float
    engine_thrust_to_weight: float


@dataclass(frozen=True)
class BimodalPowerMode:
    electric_power_kwe_per_engine: float
    reference_stage_power_kwe: float
    idle_reactor_thermal_power_kwt: float
    conversion_efficiency: float


# NASA/GRC "LO2-Augmented NTR (LANTR) Concept" public performance table.
_MR = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
_ISP_S = np.array([900.0, 725.0, 637.0, 588.0, 552.0, 516.0], dtype=float)
_AUG = np.array([1.000, 1.611, 2.123, 2.616, 3.066, 3.441], dtype=float)
_THRUST_LBF = np.array([16500.0, 26587.0, 35026.0, 43165.0, 50587.0, 56779.0], dtype=float)
_ENGINE_MASS_LBM = np.array([5462.0, 5677.0, 5834.0, 5987.0, 6139.0, 6295.0], dtype=float)
_TW = np.array([3.02, 4.68, 6.00, 7.21, 8.24, 9.02], dtype=float)

BASE_REACTOR_POWER_MW = 365.0
BASE_REACTOR_EXIT_TEMPERATURE_K = 2734.0
BASE_CHAMBER_PRESSURE_PA = 1000.0 * PSI_TO_PA
BASE_NOZZLE_AREA_RATIO = 300.0
BASE_THRUST_N = 16500.0 * LBF_TO_N
BASE_ISP_S = 900.0

# Public BNTR reference power-generation values.
BIMODAL_POWER = BimodalPowerMode(
    electric_power_kwe_per_engine=25.0,
    reference_stage_power_kwe=50.0,
    idle_reactor_thermal_power_kwt=125.0,
    conversion_efficiency=25.0 / 125.0,
)

NASA_LANTR_SOURCE = (
    "NASA/GRC LO2-Augmented NTR (LANTR) Concept: Operational Features and "
    "Performance Characteristics, NTRS 20200002036"
)
NASA_LANTR_URL = "https://ntrs.nasa.gov/citations/20200002036"

NASA_BIMODAL_SOURCE = (
    "NASA/TM-2016-219393 and NASA/CP-2004-212963, bimodal NTR power references"
)
NASA_BIMODAL_URL = "https://ntrs.nasa.gov/citations/20160014801"


def lantr_point(oxygen_hydrogen_ratio: float) -> LantrPoint:
    """Interpolate the public NASA LANTR table for O/H MR between 0 and 5."""
    mr = float(oxygen_hydrogen_ratio)
    if not 0.0 <= mr <= 5.0:
        raise ValueError("oxygen_hydrogen_ratio must lie between 0 and 5")

    return LantrPoint(
        oxygen_hydrogen_ratio=mr,
        delivered_isp_s=float(np.interp(mr, _MR, _ISP_S)),
        thrust_augmentation_factor=float(np.interp(mr, _MR, _AUG)),
        thrust_n=float(np.interp(mr, _MR, _THRUST_LBF) * LBF_TO_N),
        engine_mass_kg=float(np.interp(mr, _MR, _ENGINE_MASS_LBM) * LBM_TO_KG),
        engine_thrust_to_weight=float(np.interp(mr, _MR, _TW)),
    )


def lantr_table() -> list[LantrPoint]:
    """Return the exact tabulated NASA LANTR points."""
    return [lantr_point(float(mr)) for mr in _MR]
