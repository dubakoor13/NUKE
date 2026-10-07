"""Public historical NERVA/NTP reference presets for comparison dashboards.

These presets intentionally contain system-level and geometric values published
in NASA reports. They do not encode fissile loading, enrichment distribution,
or an exact criticality recipe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


LBF_TO_N = 4.4482216152605
LBM_TO_KG = 0.45359237
PSI_TO_PA = 6894.757293168
IN_TO_CM = 2.54
RANKINE_TO_K = 5.0 / 9.0


@dataclass(frozen=True)
class HistoricalNervaPreset:
    key: str
    name: str
    category: str
    description: str

    thermal_power_mw: Optional[float] = None
    thrust_n: Optional[float] = None
    isp_s: Optional[float] = None
    isp_range_s: Optional[tuple[float, float]] = None

    chamber_temperature_k: Optional[float] = None
    chamber_temperature_range_k: Optional[tuple[float, float]] = None
    chamber_pressure_pa: Optional[float] = None
    hydrogen_mass_flow_kg_s: Optional[float] = None
    nozzle_expansion_ratio: Optional[float] = None

    fuel_element_flat_to_flat_cm: Optional[float] = None
    coolant_channels_per_element: Optional[int] = None
    coolant_channel_diameter_cm: Optional[float] = None
    fuel_element_count: Optional[int] = None
    core_diameter_cm: Optional[float] = None
    active_length_cm: Optional[float] = None

    tie_heating_fraction_range: Optional[tuple[float, float]] = None
    reflector_heating_fraction_range: Optional[tuple[float, float]] = None

    source_title: str = ""
    source_url: str = ""
    source_note: str = ""


XE_PRIME = HistoricalNervaPreset(
    key="xe_prime",
    name="XE-Prime",
    category="Tested prototype",
    description=(
        "Integrated NERVA prototype engine tested in 1969. Values below are "
        "from NASA's comparison table for tested NTP systems."
    ),
    thermal_power_mw=1140.0,
    thrust_n=55.4e3 * LBF_TO_N,
    isp_s=710.0,
    chamber_temperature_k=4105.0 * RANKINE_TO_K,
    chamber_pressure_pa=565.8 * PSI_TO_PA,
    hydrogen_mass_flow_kg_s=70.2 * LBM_TO_KG,
    nozzle_expansion_ratio=10.0,
    fuel_element_flat_to_flat_cm=0.752 * IN_TO_CM,
    coolant_channels_per_element=19,
    coolant_channel_diameter_cm=0.10 * IN_TO_CM,
    fuel_element_count=1584,
    core_diameter_cm=35.0 * IN_TO_CM,
    active_length_cm=52.0 * IN_TO_CM,
    source_title="NASA TM-105252, An Overview of Tested and Analyzed NTP Concepts",
    source_url="https://ntrs.nasa.gov/citations/19920001919",
    source_note="Table 3, XE-PRIME column.",
)


NERVA_75K_1972 = HistoricalNervaPreset(
    key="nerva_75k_1972",
    name="1972 NERVA 75 klbf",
    category="Flight-engine design reference",
    description=(
        "75,000 lbf-class NERVA design reference. NASA reports chamber "
        "temperature and Isp as ranges; they are preserved as ranges here."
    ),
    thrust_n=75.0e3 * LBF_TO_N,
    isp_range_s=(825.0, 850.0),
    chamber_temperature_range_k=(2350.0, 2500.0),
    chamber_pressure_pa=450.0 * PSI_TO_PA,
    nozzle_expansion_ratio=100.0,
    source_title="NASA TM-106739, NERVA-Type Nuclear Thermal Rocket Engines",
    source_url="https://ntrs.nasa.gov/citations/19950009268",
    source_note="Table 2, 72 NERVA column.",
)


NERVA_GRAPHITE_DERIVATIVE = HistoricalNervaPreset(
    key="graphite_derivative_75k",
    name="75 klbf graphite NERVA derivative",
    category="NASA derivative study",
    description=(
        "State-of-the-art graphite-fuel NERVA derivative from the same NASA "
        "75-klbf comparison table."
    ),
    thrust_n=75.0e3 * LBF_TO_N,
    isp_s=875.0,
    chamber_temperature_k=2500.0,
    chamber_pressure_pa=500.0 * PSI_TO_PA,
    nozzle_expansion_ratio=200.0,
    source_title="NASA TM-106739, NERVA-Type Nuclear Thermal Rocket Engines",
    source_url="https://ntrs.nasa.gov/citations/19950009268",
    source_note="Table 2, state-of-the-art graphite derivative column.",
)


NESS_R1 = HistoricalNervaPreset(
    key="ness_r1",
    name="NESS / R-1 baseline reactor",
    category="Reactor analysis baseline",
    description=(
        "NASA NESS baseline reactor derived from Rover/NERVA data. This is a "
        "reactor-system reference, not a single tested engine operating point."
    ),
    thermal_power_mw=1500.0,
    fuel_element_flat_to_flat_cm=1.9,
    coolant_channels_per_element=19,
    core_diameter_cm=38.0 * IN_TO_CM,
    active_length_cm=52.0 * IN_TO_CM,
    tie_heating_fraction_range=(0.03, 0.07),
    reflector_heating_fraction_range=(0.01, 0.02),
    source_title="NASA CR-191080, Nuclear Engine System Simulation (NESS)",
    source_url="https://ntrs.nasa.gov/citations/19930014687",
    source_note=(
        "Baseline reactor design: approximately 1500 MW core; R-1 nominal "
        "core 38 in diameter by 52 in long; tie-tube heating 3-7%; reflector "
        "heating 1-2%."
    ),
)


PRESETS = {
    preset.key: preset
    for preset in (
        XE_PRIME,
        NERVA_75K_1972,
        NERVA_GRAPHITE_DERIVATIVE,
        NESS_R1,
    )
}


def get_preset(key: str) -> HistoricalNervaPreset:
    try:
        return PRESETS[key]
    except KeyError as exc:
        raise KeyError(
            f"unknown historical preset {key!r}; choose from {sorted(PRESETS)}"
        ) from exc
