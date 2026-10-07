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


# Higher-pressure trimodal LANTR reference published by NASA. This is a
# separate architecture from the 1000-psia small-engine table above.
HIGH_PRESSURE_LANTR_CHAMBER_PRESSURE_PA = 2000.0 * PSI_TO_PA
HIGH_PRESSURE_LANTR_NOZZLE_AREA_RATIO = 500.0
HIGH_PRESSURE_LANTR_BASE_THRUST_N = 15000.0 * LBF_TO_N
HIGH_PRESSURE_LANTR_TEMPERATURE_RANGE_K = (2500.0, 2900.0)

# Published 5-hour / 2900 K column for the 15-klbf trimodal LANTR.
_HIGH_MR = np.array([0.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
_HIGH_ISP_5H_2900K = np.array([941.0, 772.0, 647.0, 576.0, 514.0], dtype=float)
_HIGH_TW = np.array([3.0, 4.8, 8.2, 11.0, 13.1], dtype=float)

NASA_HIGH_PRESSURE_LANTR_SOURCE = (
    "NASA TM-106726 / LANTR trimodal concept: 15-klbf LANTR, "
    "2000-psia nozzle inlet pressure, 500:1 expansion ratio"
)
NASA_HIGH_PRESSURE_LANTR_URL = "https://ntrs.nasa.gov/citations/19950005290"


def high_pressure_lantr_isp(
    oxygen_hydrogen_ratio: float,
) -> tuple[float, float]:
    """Return 5-hour/2900 K Isp and engine T/W for the 2000-psia LANTR."""
    mr = float(oxygen_hydrogen_ratio)
    if not 0.0 <= mr <= 7.0:
        raise ValueError("oxygen_hydrogen_ratio must lie between 0 and 7")

    isp = float(np.interp(mr, _HIGH_MR, _HIGH_ISP_5H_2900K))
    thrust_to_weight = float(np.interp(mr, _HIGH_MR, _HIGH_TW))
    return isp, thrust_to_weight


G0_M_S2 = 9.80665
BASE_HYDROGEN_MASS_FLOW_KG_S = BASE_THRUST_N / (G0_M_S2 * BASE_ISP_S)


@dataclass(frozen=True)
class CoupledLantrPoint:
    oxygen_hydrogen_ratio: float
    chamber_pressure_pa: float
    reactor_exit_temperature_k: float
    hydrogen_mass_flow_kg_s: float
    oxygen_mass_flow_kg_s: float
    total_mass_flow_kg_s: float
    delivered_isp_s: float
    thrust_n: float
    thrust_augmentation_factor: float


def coupled_lantr_point(
    oxygen_hydrogen_ratio: float,
    chamber_pressure_pa: float,
    reactor_exit_temperature_k: float = BASE_REACTOR_EXIT_TEMPERATURE_K,
) -> CoupledLantrPoint:
    """Scale the NASA LANTR reference with pressure for fixed throat geometry.

    The reference engine is anchored to the published MR/Isp table at
    BASE_CHAMBER_PRESSURE_PA and BASE_REACTOR_EXIT_TEMPERATURE_K.

    For a fixed throat and approximately fixed gas properties, choked hydrogen
    flow scales as Pc/sqrt(Tc). Oxygen flow is MR times hydrogen flow, and
    thrust is total propellant flow times g0 times the published delivered Isp.
    """
    if chamber_pressure_pa <= 0.0:
        raise ValueError("chamber_pressure_pa must be positive")
    if reactor_exit_temperature_k <= 0.0:
        raise ValueError("reactor_exit_temperature_k must be positive")

    reference = lantr_point(oxygen_hydrogen_ratio)
    pressure_scale = chamber_pressure_pa / BASE_CHAMBER_PRESSURE_PA
    temperature_scale = (
        BASE_REACTOR_EXIT_TEMPERATURE_K / reactor_exit_temperature_k
    ) ** 0.5

    hydrogen_flow = (
        BASE_HYDROGEN_MASS_FLOW_KG_S
        * pressure_scale
        * temperature_scale
    )
    oxygen_flow = oxygen_hydrogen_ratio * hydrogen_flow
    total_flow = hydrogen_flow + oxygen_flow
    thrust = total_flow * G0_M_S2 * reference.delivered_isp_s

    return CoupledLantrPoint(
        oxygen_hydrogen_ratio=float(oxygen_hydrogen_ratio),
        chamber_pressure_pa=float(chamber_pressure_pa),
        reactor_exit_temperature_k=float(reactor_exit_temperature_k),
        hydrogen_mass_flow_kg_s=float(hydrogen_flow),
        oxygen_mass_flow_kg_s=float(oxygen_flow),
        total_mass_flow_kg_s=float(total_flow),
        delivered_isp_s=float(reference.delivered_isp_s),
        thrust_n=float(thrust),
        thrust_augmentation_factor=float(thrust / (
            BASE_HYDROGEN_MASS_FLOW_KG_S
            * pressure_scale
            * temperature_scale
            * G0_M_S2
            * BASE_ISP_S
        )),
    )


HIGH_PRESSURE_BASE_HYDROGEN_MASS_FLOW_KG_S = (
    HIGH_PRESSURE_LANTR_BASE_THRUST_N
    / (G0_M_S2 * high_pressure_lantr_isp(0.0)[0])
)


def coupled_high_pressure_lantr_point(
    oxygen_hydrogen_ratio: float,
    chamber_pressure_pa: float,
    reactor_exit_temperature_k: float = 2900.0,
) -> CoupledLantrPoint:
    """Scale the 2000-psia trimodal LANTR for fixed throat geometry.

    The anchor is the published 15-klbf, 2000-psia, 2900 K / 5-hour reference.
    Hydrogen choked flow scales as Pc/sqrt(Tc). Oxygen flow is MR times the
    hydrogen flow. Delivered Isp is interpolated from the published high-
    pressure trimodal table.
    """
    if chamber_pressure_pa <= 0.0:
        raise ValueError("chamber_pressure_pa must be positive")
    if reactor_exit_temperature_k <= 0.0:
        raise ValueError("reactor_exit_temperature_k must be positive")

    isp, _ = high_pressure_lantr_isp(oxygen_hydrogen_ratio)
    pressure_scale = (
        chamber_pressure_pa / HIGH_PRESSURE_LANTR_CHAMBER_PRESSURE_PA
    )
    temperature_scale = (2900.0 / reactor_exit_temperature_k) ** 0.5

    hydrogen_flow = (
        HIGH_PRESSURE_BASE_HYDROGEN_MASS_FLOW_KG_S
        * pressure_scale
        * temperature_scale
    )
    oxygen_flow = oxygen_hydrogen_ratio * hydrogen_flow
    total_flow = hydrogen_flow + oxygen_flow
    thrust = total_flow * G0_M_S2 * isp

    base_thrust_at_same_pressure_temperature = (
        HIGH_PRESSURE_BASE_HYDROGEN_MASS_FLOW_KG_S
        * pressure_scale
        * temperature_scale
        * G0_M_S2
        * high_pressure_lantr_isp(0.0)[0]
    )

    return CoupledLantrPoint(
        oxygen_hydrogen_ratio=float(oxygen_hydrogen_ratio),
        chamber_pressure_pa=float(chamber_pressure_pa),
        reactor_exit_temperature_k=float(reactor_exit_temperature_k),
        hydrogen_mass_flow_kg_s=float(hydrogen_flow),
        oxygen_mass_flow_kg_s=float(oxygen_flow),
        total_mass_flow_kg_s=float(total_flow),
        delivered_isp_s=float(isp),
        thrust_n=float(thrust),
        thrust_augmentation_factor=float(
            thrust / base_thrust_at_same_pressure_temperature
        ),
    )
