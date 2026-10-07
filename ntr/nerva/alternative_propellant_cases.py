"""Source-backed alternative-propellant NTP cases from the uploaded NASA/AMA deck.

The deck provides:
* qualitative/reference data for H-NTP (LH2) and A-NTP (LNH3);
* complete steady-state PBM state tables for two frozen-chemistry M-NTP
  (LCH4) sizing cases.

No unsupported H-NTP/A-NTP state table is invented here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ReferencePropellantCase:
    key: str
    name: str
    propellant: str
    reference_temperature_k: float
    reference_isp_s: float
    storage_temperature_range_k: tuple[float, float]
    density_kg_m3: float
    cycle_summary: str
    mission_summary: str
    risk_summary: str
    boost_pump: bool
    serial_preheat: bool


@dataclass(frozen=True)
class FlowState:
    state: int
    mass_flow_kg_s: float
    temperature_k: float
    pressure_mpa: float


@dataclass(frozen=True)
class MethanePbmCase:
    key: str
    name: str
    sizing_constraint: str
    states: tuple[FlowState, ...]
    reactor_power_mw: float
    isp_s: float
    total_mass_flow_kg_s: float
    maximum_fuel_temperature_k: float
    chamber_temperature_k: float
    maximum_moderator_temperature_k: float
    maximum_control_drum_reflector_temperature_k: float
    pump_power_kw: float
    pump_efficiency_percent: float
    maximum_system_pressure_mpa: float
    turbopump_speed_rpm: float
    turbine_power_kw: float
    turbine_efficiency_percent: float
    turbine_pressure_ratio: float
    turbine_mass_flow_kg_s: float
    turbine_inlet_temperature_k: float
    turbine_delta_temperature_k: float

    @property
    def derived_thrust_n(self) -> float:
        """Thrust implied by the source mdot and Isp values."""
        return self.total_mass_flow_kg_s * 9.80665 * self.isp_s

    @property
    def chamber_pressure_mpa(self) -> float:
        return self.state(21).pressure_mpa

    @property
    def pump_pressure_rise_mpa(self) -> float:
        return self.state(3).pressure_mpa - self.state(1).pressure_mpa

    def state(self, number: int) -> FlowState:
        for item in self.states:
            if item.state == number:
                return item
        raise KeyError(f"state {number} is not present in {self.name}")


H_NTP = ReferencePropellantCase(
    key="h_ntp",
    name="H-NTP baseline",
    propellant="LH2",
    reference_temperature_k=2700.0,
    reference_isp_s=900.0,
    storage_temperature_range_k=(14.0, 33.0),
    density_kg_m3=71.0,
    cycle_summary=(
        "Baseline expander architecture; LH2 requires a boost pump and exits "
        "the pumps already supercritical."
    ),
    mission_summary="High-delta-v crewed Mars transfers",
    risk_summary="Highly flammable H2; hydrogen embrittlement / hot-H2 attack",
    boost_pump=True,
    serial_preheat=False,
)


A_NTP = ReferencePropellantCase(
    key="a_ntp",
    name="A-NTP ammonia",
    propellant="LNH3",
    reference_temperature_k=2700.0,
    reference_isp_s=370.0,
    storage_temperature_range_k=(195.0, 405.0),
    density_kg_m3=730.0,
    cycle_summary=(
        "Serial pre-heat loop with extra pumping; no boost pump. NH3 reaches "
        "supercritical conditions only near state 12 in the deck architecture."
    ),
    mission_summary="Long-dwell depots and low-Isp cargo where simplicity matters",
    risk_summary="Toxic vapor; pressure-drop and pre-heater thermal-stress focus",
    boost_pump=False,
    serial_preheat=True,
)


METHANE_CASE_1 = MethanePbmCase(
    key="m_ntp_case_1",
    name="M-NTP Case 1",
    sizing_constraint="Chamber temperature limited to approximately 2100 K",
    states=(
        FlowState(1, 19.55, 100.00, 0.13),
        FlowState(3, 19.55, 103.71, 11.03),
        FlowState(4, 19.55, 103.72, 10.98),
        FlowState(5, 9.77, 103.72, 10.97),
        FlowState(6, 9.77, 104.68, 8.93),
        FlowState(7, 9.77, 266.19, 8.69),
        FlowState(8, 9.77, 345.55, 7.29),
        FlowState(9, 9.77, 103.72, 10.98),
        FlowState(10, 9.77, 105.03, 8.19),
        FlowState(11, 9.77, 291.39, 7.86),
        FlowState(12, 19.55, 317.83, 7.29),
        FlowState(15, 14.65, 318.28, 7.46),
        FlowState(16, 14.57, 295.88, 5.48),
        FlowState(17, 4.90, 318.28, 7.46),
        FlowState(18, 4.90, 311.54, 5.48),
        FlowState(19, 19.55, 299.79, 5.48),
        FlowState(20, 19.55, 297.45, 4.89),
        FlowState(21, 19.55, 2100.79, 4.60),
        FlowState(22, 19.55, 824.27, 0.00),
    ),
    reactor_power_mw=177.65,
    isp_s=353.71,
    total_mass_flow_kg_s=19.55,
    maximum_fuel_temperature_k=2270.30,
    chamber_temperature_k=2100.79,
    maximum_moderator_temperature_k=435.30,
    maximum_control_drum_reflector_temperature_k=419.44,
    pump_power_kw=482.47,
    pump_efficiency_percent=83.10,
    maximum_system_pressure_mpa=11.03,
    turbopump_speed_rpm=36602.72,
    turbine_power_kw=580.61,
    turbine_efficiency_percent=82.99,
    turbine_pressure_ratio=1.40,
    turbine_mass_flow_kg_s=14.65,
    turbine_inlet_temperature_k=318.28,
    turbine_delta_temperature_k=22.41,
)


METHANE_CASE_2 = MethanePbmCase(
    key="m_ntp_case_2",
    name="M-NTP Case 2",
    sizing_constraint=(
        "Fuel temperature limited to approximately 2850 K, allowing chamber "
        "temperature to rise to approximately 2630 K"
    ),
    states=(
        FlowState(1, 17.43, 100.00, 0.14),
        FlowState(3, 17.43, 103.80, 11.28),
        FlowState(4, 17.43, 103.80, 11.24),
        FlowState(5, 8.72, 103.80, 11.23),
        FlowState(6, 8.72, 104.82, 9.07),
        FlowState(7, 8.72, 382.98, 8.77),
        FlowState(8, 8.72, 494.38, 6.93),
        FlowState(9, 8.72, 103.80, 11.24),
        FlowState(10, 8.72, 105.32, 8.01),
        FlowState(11, 8.72, 397.69, 7.59),
        FlowState(12, 17.43, 448.27, 6.93),
        FlowState(15, 8.70, 448.54, 6.98),
        FlowState(16, 8.66, 423.71, 5.11),
        FlowState(17, 8.73, 448.54, 6.98),
        FlowState(18, 8.73, 445.82, 5.10),
        FlowState(19, 17.43, 434.94, 5.10),
        FlowState(20, 17.43, 434.71, 4.96),
        FlowState(21, 17.43, 2630.28, 4.60),
        FlowState(22, 17.43, 1194.76, 0.00),
    ),
    reactor_power_mw=214.66,
    isp_s=400.16,
    total_mass_flow_kg_s=17.43,
    maximum_fuel_temperature_k=2850.83,
    chamber_temperature_k=2630.28,
    maximum_moderator_temperature_k=550.42,
    maximum_control_drum_reflector_temperature_k=595.01,
    pump_power_kw=440.09,
    pump_efficiency_percent=83.08,
    maximum_system_pressure_mpa=11.28,
    turbopump_speed_rpm=39451.63,
    turbine_power_kw=529.72,
    turbine_efficiency_percent=82.15,
    turbine_pressure_ratio=1.40,
    turbine_mass_flow_kg_s=8.70,
    turbine_inlet_temperature_k=448.54,
    turbine_delta_temperature_k=24.83,
)


FOUR_CASES = {
    H_NTP.key: H_NTP,
    A_NTP.key: A_NTP,
    METHANE_CASE_1.key: METHANE_CASE_1,
    METHANE_CASE_2.key: METHANE_CASE_2,
}
