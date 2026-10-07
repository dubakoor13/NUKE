"""Generic nuclear-airbreathing ramjet thermodynamic demonstrator.

This module intentionally models only public, high-level cycle physics:
freestream -> diffuser -> reactor heat addition -> nozzle -> thrust.

It does NOT model reactor criticality, fuel loading, shielding, inlet geometry,
vehicle sizing, guidance, range, or weapon-specific performance.

Air mass flow is supplied directly by the caller instead of being inferred
from an inlet capture area.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


G0_M_S2 = 9.80665


@dataclass(frozen=True)
class AirModel:
    gamma: float = 1.33
    gas_constant_j_kg_k: float = 287.05
    cp_j_kg_k: float = 1150.0

    def validate(self) -> None:
        if self.gamma <= 1.0:
            raise ValueError("gamma must exceed 1")
        if self.gas_constant_j_kg_k <= 0.0:
            raise ValueError("gas constant must be positive")
        if self.cp_j_kg_k <= 0.0:
            raise ValueError("cp must be positive")


@dataclass(frozen=True)
class NuclearRamjetResult:
    mach: float
    ambient_temperature_k: float
    ambient_pressure_pa: float
    air_mass_flow_kg_s: float
    flight_velocity_m_s: float

    diffuser_total_temperature_k: float
    diffuser_total_pressure_pa: float
    reactor_outlet_total_temperature_k: float
    reactor_outlet_total_pressure_pa: float
    reactor_thermal_power_w: float

    nozzle_exit_temperature_k: float
    nozzle_exit_pressure_pa: float
    nozzle_exit_velocity_m_s: float
    nozzle_exit_mach: float

    gross_thrust_n: float
    ram_drag_n: float
    net_thrust_n: float
    specific_thrust_n_s_kg: float


def _stagnation_temperature(
    static_temperature_k: float,
    mach: float,
    gamma: float,
) -> float:
    return static_temperature_k * (
        1.0 + 0.5 * (gamma - 1.0) * mach * mach
    )


def _ideal_stagnation_pressure(
    static_pressure_pa: float,
    mach: float,
    gamma: float,
) -> float:
    return static_pressure_pa * (
        1.0 + 0.5 * (gamma - 1.0) * mach * mach
    ) ** (gamma / (gamma - 1.0))


def evaluate_nuclear_ramjet(
    mach: float,
    ambient_temperature_k: float,
    ambient_pressure_pa: float,
    air_mass_flow_kg_s: float,
    reactor_outlet_total_temperature_k: float,
    diffuser_pressure_recovery: float = 0.90,
    reactor_total_pressure_ratio: float = 0.95,
    reactor_heat_transfer_efficiency: float = 0.95,
    nozzle_efficiency: float = 0.95,
    air: AirModel | None = None,
) -> NuclearRamjetResult:
    """Evaluate a generic nuclear-heated airbreathing ramjet cycle.

    The nozzle is assumed ideally expanded to ambient pressure when sufficient
    total pressure is available. No inlet/nozzle dimensions are inferred.
    """
    if air is None:
        air = AirModel()
    air.validate()

    if mach <= 0.0:
        raise ValueError("mach must be positive")
    if ambient_temperature_k <= 0.0:
        raise ValueError("ambient temperature must be positive")
    if ambient_pressure_pa <= 0.0:
        raise ValueError("ambient pressure must be positive")
    if air_mass_flow_kg_s <= 0.0:
        raise ValueError("air mass flow must be positive")
    if reactor_outlet_total_temperature_k <= ambient_temperature_k:
        raise ValueError(
            "reactor outlet total temperature must exceed ambient temperature"
        )
    for name, value in (
        ("diffuser_pressure_recovery", diffuser_pressure_recovery),
        ("reactor_total_pressure_ratio", reactor_total_pressure_ratio),
        ("reactor_heat_transfer_efficiency", reactor_heat_transfer_efficiency),
        ("nozzle_efficiency", nozzle_efficiency),
    ):
        if not (0.0 < value <= 1.0):
            raise ValueError(f"{name} must lie in (0, 1]")

    gamma = air.gamma
    gas_constant = air.gas_constant_j_kg_k
    cp = air.cp_j_kg_k

    speed_of_sound = math.sqrt(
        gamma * gas_constant * ambient_temperature_k
    )
    flight_velocity = mach * speed_of_sound

    diffuser_total_temperature = _stagnation_temperature(
        ambient_temperature_k,
        mach,
        gamma,
    )
    ideal_total_pressure = _ideal_stagnation_pressure(
        ambient_pressure_pa,
        mach,
        gamma,
    )
    diffuser_total_pressure = (
        ambient_pressure_pa
        + diffuser_pressure_recovery
        * (ideal_total_pressure - ambient_pressure_pa)
    )

    reactor_total_pressure = (
        diffuser_total_pressure * reactor_total_pressure_ratio
    )

    if reactor_outlet_total_temperature_k <= diffuser_total_temperature:
        raise ValueError(
            "reactor outlet temperature must exceed diffuser total temperature"
        )

    heat_to_air = (
        air_mass_flow_kg_s
        * cp
        * (
            reactor_outlet_total_temperature_k
            - diffuser_total_temperature
        )
    )
    reactor_thermal_power = (
        heat_to_air / reactor_heat_transfer_efficiency
    )

    if reactor_total_pressure <= ambient_pressure_pa:
        raise ValueError(
            "insufficient reactor-outlet total pressure for expansion"
        )

    pressure_ratio = ambient_pressure_pa / reactor_total_pressure
    exit_temperature_ideal = (
        reactor_outlet_total_temperature_k
        * pressure_ratio ** ((gamma - 1.0) / gamma)
    )

    ideal_kinetic_energy = (
        cp
        * (
            reactor_outlet_total_temperature_k
            - exit_temperature_ideal
        )
    )
    exit_velocity = math.sqrt(
        2.0 * nozzle_efficiency * ideal_kinetic_energy
    )

    exit_temperature = (
        reactor_outlet_total_temperature_k
        - exit_velocity * exit_velocity / (2.0 * cp)
    )
    exit_speed_of_sound = math.sqrt(
        gamma * gas_constant * exit_temperature
    )
    exit_mach = exit_velocity / exit_speed_of_sound

    gross_thrust = air_mass_flow_kg_s * exit_velocity
    ram_drag = air_mass_flow_kg_s * flight_velocity
    net_thrust = gross_thrust - ram_drag
    specific_thrust = net_thrust / air_mass_flow_kg_s

    return NuclearRamjetResult(
        mach=float(mach),
        ambient_temperature_k=float(ambient_temperature_k),
        ambient_pressure_pa=float(ambient_pressure_pa),
        air_mass_flow_kg_s=float(air_mass_flow_kg_s),
        flight_velocity_m_s=float(flight_velocity),
        diffuser_total_temperature_k=float(
            diffuser_total_temperature
        ),
        diffuser_total_pressure_pa=float(
            diffuser_total_pressure
        ),
        reactor_outlet_total_temperature_k=float(
            reactor_outlet_total_temperature_k
        ),
        reactor_outlet_total_pressure_pa=float(
            reactor_total_pressure
        ),
        reactor_thermal_power_w=float(
            reactor_thermal_power
        ),
        nozzle_exit_temperature_k=float(
            exit_temperature
        ),
        nozzle_exit_pressure_pa=float(
            ambient_pressure_pa
        ),
        nozzle_exit_velocity_m_s=float(
            exit_velocity
        ),
        nozzle_exit_mach=float(exit_mach),
        gross_thrust_n=float(gross_thrust),
        ram_drag_n=float(ram_drag),
        net_thrust_n=float(net_thrust),
        specific_thrust_n_s_kg=float(
            specific_thrust
        ),
    )


def evaluate_nuclear_ramjet_from_thermal_power(
    mach: float,
    ambient_temperature_k: float,
    ambient_pressure_pa: float,
    air_mass_flow_kg_s: float,
    deposited_thermal_power_w: float,
    heat_transfer_efficiency: float = 0.85,
    diffuser_pressure_recovery: float = 0.90,
    reactor_total_pressure_ratio: float = 0.95,
    nozzle_efficiency: float = 0.95,
    air: AirModel | None = None,
) -> NuclearRamjetResult:
    """Drive the generic ramjet cycle from deposited OpenMC thermal power.

    The OpenMC model provides nuclear deposited power. A separate engineering
    heat-transfer efficiency determines how much of that deposited power reaches
    the working air. No channel/inlet geometry is inferred here.
    """
    if air is None:
        air = AirModel()
    air.validate()

    if deposited_thermal_power_w <= 0.0:
        raise ValueError("deposited_thermal_power_w must be positive")
    if not (0.0 < heat_transfer_efficiency <= 1.0):
        raise ValueError("heat_transfer_efficiency must lie in (0, 1]")

    diffuser_total_temperature = _stagnation_temperature(
        ambient_temperature_k,
        mach,
        air.gamma,
    )

    heat_to_air_w = (
        deposited_thermal_power_w * heat_transfer_efficiency
    )
    reactor_outlet_total_temperature_k = (
        diffuser_total_temperature
        + heat_to_air_w / (air_mass_flow_kg_s * air.cp_j_kg_k)
    )

    return evaluate_nuclear_ramjet(
        mach=mach,
        ambient_temperature_k=ambient_temperature_k,
        ambient_pressure_pa=ambient_pressure_pa,
        air_mass_flow_kg_s=air_mass_flow_kg_s,
        reactor_outlet_total_temperature_k=(
            reactor_outlet_total_temperature_k
        ),
        diffuser_pressure_recovery=diffuser_pressure_recovery,
        reactor_total_pressure_ratio=reactor_total_pressure_ratio,
        reactor_heat_transfer_efficiency=heat_transfer_efficiency,
        nozzle_efficiency=nozzle_efficiency,
        air=air,
    )
