"""Legacy dashboard nozzle helper.

For NERVA engineering calculations use :mod:`ntr.nerva.nerva_engine`.
That module solves exit pressure from fixed nozzle expansion ratio, exposes
c*/Cf, and includes reactor-power/mass-flow/temperature coupling.

This file remains for backward compatibility with the original synthetic web
dashboard and should not be used as the primary NERVA performance model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


G0_M_S2 = 9.80665
HYDROGEN_GAS_CONSTANT_J_KG_K = 4124.0


@dataclass(frozen=True)
class MixedHydrogenState:
    mass_flow_kg_s: float
    temperature_k: float
    pressure_pa: float


@dataclass(frozen=True)
class NozzlePerformance:
    thrust_n: float
    isp_s: float
    effective_exhaust_velocity_m_s: float
    exit_velocity_m_s: float
    exit_mach: float
    expansion_ratio: float
    throat_area_m2: float
    exit_area_m2: float
    momentum_thrust_n: float
    pressure_thrust_n: float


def combine_outlet_streams(
    fuel_mass_flow_kg_s: float,
    fuel_temperature_k: float,
    fuel_pressure_pa: float,
    tie_mass_flow_kg_s: float,
    tie_temperature_k: float,
    tie_pressure_pa: float,
) -> MixedHydrogenState:
    """Combine fuel and tie-tube hydrogen streams with equal-cp mixing."""
    values = (
        fuel_mass_flow_kg_s,
        fuel_temperature_k,
        fuel_pressure_pa,
        tie_mass_flow_kg_s,
        tie_temperature_k,
        tie_pressure_pa,
    )
    if any(value <= 0.0 for value in values):
        raise ValueError("all stream inputs must be positive")

    mass_flow = fuel_mass_flow_kg_s + tie_mass_flow_kg_s
    temperature = (
        fuel_mass_flow_kg_s * fuel_temperature_k
        + tie_mass_flow_kg_s * tie_temperature_k
    ) / mass_flow
    pressure = min(fuel_pressure_pa, tie_pressure_pa)

    return MixedHydrogenState(
        mass_flow_kg_s=float(mass_flow),
        temperature_k=float(temperature),
        pressure_pa=float(pressure),
    )


def ideal_nozzle_performance(
    mass_flow_kg_s: float,
    chamber_temperature_k: float,
    chamber_pressure_pa: float,
    exit_pressure_pa: float = 1000.0,
    ambient_pressure_pa: float = 0.0,
    gamma: float = 1.35,
    gas_constant_j_kg_k: float = HYDROGEN_GAS_CONSTANT_J_KG_K,
    nozzle_efficiency: float = 0.95,
) -> NozzlePerformance:
    """Estimate choked ideal-gas nozzle performance.

    This is a system-level demonstration model. It assumes a calorically
    perfect gas with constant gamma and applies a scalar nozzle efficiency to
    ideal kinetic energy. It is not a substitute for CEA/Cantera equilibrium
    nozzle analysis.
    """
    if mass_flow_kg_s <= 0.0:
        raise ValueError("mass_flow_kg_s must be positive")
    if chamber_temperature_k <= 0.0:
        raise ValueError("chamber_temperature_k must be positive")
    if chamber_pressure_pa <= 0.0:
        raise ValueError("chamber_pressure_pa must be positive")
    if exit_pressure_pa <= 0.0:
        raise ValueError("exit_pressure_pa must be positive")
    if ambient_pressure_pa < 0.0:
        raise ValueError("ambient_pressure_pa must be non-negative")
    if exit_pressure_pa >= chamber_pressure_pa:
        raise ValueError("exit pressure must be lower than chamber pressure")
    if gamma <= 1.0:
        raise ValueError("gamma must exceed 1")
    if gas_constant_j_kg_k <= 0.0:
        raise ValueError("gas constant must be positive")
    if not (0.0 < nozzle_efficiency <= 1.0):
        raise ValueError("nozzle_efficiency must lie in (0, 1]")

    pressure_ratio = chamber_pressure_pa / exit_pressure_pa
    exponent = (gamma - 1.0) / gamma
    exit_mach = math.sqrt(
        2.0
        / (gamma - 1.0)
        * (pressure_ratio**exponent - 1.0)
    )

    area_ratio = (
        1.0
        / exit_mach
        * (
            2.0
            / (gamma + 1.0)
            * (1.0 + 0.5 * (gamma - 1.0) * exit_mach**2)
        )
        ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    )

    choked_mass_flux = (
        chamber_pressure_pa
        * math.sqrt(gamma / (gas_constant_j_kg_k * chamber_temperature_k))
        * (2.0 / (gamma + 1.0))
        ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
    )
    throat_area = mass_flow_kg_s / choked_mass_flux
    exit_area = throat_area * area_ratio

    ideal_exit_velocity = math.sqrt(
        2.0
        * gamma
        / (gamma - 1.0)
        * gas_constant_j_kg_k
        * chamber_temperature_k
        * (
            1.0
            - (exit_pressure_pa / chamber_pressure_pa)
            ** ((gamma - 1.0) / gamma)
        )
    )
    exit_velocity = math.sqrt(nozzle_efficiency) * ideal_exit_velocity

    momentum_thrust = mass_flow_kg_s * exit_velocity
    pressure_thrust = (
        exit_pressure_pa - ambient_pressure_pa
    ) * exit_area
    thrust = momentum_thrust + pressure_thrust
    effective_velocity = thrust / mass_flow_kg_s
    isp = effective_velocity / G0_M_S2

    return NozzlePerformance(
        thrust_n=float(thrust),
        isp_s=float(isp),
        effective_exhaust_velocity_m_s=float(effective_velocity),
        exit_velocity_m_s=float(exit_velocity),
        exit_mach=float(exit_mach),
        expansion_ratio=float(area_ratio),
        throat_area_m2=float(throat_area),
        exit_area_m2=float(exit_area),
        momentum_thrust_n=float(momentum_thrust),
        pressure_thrust_n=float(pressure_thrust),
    )
