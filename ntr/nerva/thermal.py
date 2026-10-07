"""First-pass 1-D hydrogen fuel-channel thermal model for NERVA studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HydrogenProperties:
    """Constant-property hydrogen surrogate for the first coupling stage."""

    cp_j_kg_k: float = 14300.0
    dynamic_viscosity_pa_s: float = 1.20e-5
    thermal_conductivity_w_m_k: float = 0.18
    prandtl: float = 0.95
    gas_constant_j_kg_k: float = 4124.0


@dataclass(frozen=True)
class ChannelSolution:
    """Axial solution for one representative fuel coolant channel."""

    z_center_m: np.ndarray
    bulk_temperature_k: np.ndarray
    wall_temperature_k: np.ndarray
    pressure_pa: np.ndarray
    reynolds: np.ndarray
    heat_transfer_coefficient_w_m2_k: np.ndarray
    heat_flux_w_m2: np.ndarray
    channel_power_w: np.ndarray
    mass_flow_kg_s: float
    channel_diameter_m: float

    @property
    def outlet_temperature_k(self) -> float:
        return float(self.bulk_temperature_k[-1])

    @property
    def outlet_pressure_pa(self) -> float:
        return float(self.pressure_pa[-1])

    @property
    def absorbed_power_w(self) -> float:
        return float(np.sum(self.channel_power_w))


def friction_factor(
    reynolds: float,
    relative_roughness: float = 0.0,
) -> float:
    """Darcy friction factor using laminar or Haaland turbulent correlation."""
    if reynolds <= 0.0:
        raise ValueError("reynolds must be positive")
    if relative_roughness < 0.0:
        raise ValueError("relative_roughness must be non-negative")

    if reynolds < 2300.0:
        return 64.0 / reynolds

    term = (relative_roughness / 3.7) ** 1.11 + 6.9 / reynolds
    return 1.0 / (-1.8 * np.log10(term)) ** 2


def nusselt_number(reynolds: float, prandtl: float) -> float:
    """Return a simple fully-developed/Dittus-Boelter Nusselt estimate."""
    if reynolds <= 0.0:
        raise ValueError("reynolds must be positive")
    if prandtl <= 0.0:
        raise ValueError("prandtl must be positive")

    if reynolds < 2300.0:
        return 3.66
    return 0.023 * reynolds**0.8 * prandtl**0.4


def solve_fuel_channel(
    axial_total_fuel_power_w: np.ndarray,
    z_edges_m: np.ndarray,
    fuel_channel_count: int,
    mass_flow_per_channel_kg_s: float,
    inlet_temperature_k: float,
    inlet_pressure_pa: float,
    channel_diameter_m: float,
    properties: HydrogenProperties | None = None,
    roughness_m: float = 1.0e-6,
) -> ChannelSolution:
    """Solve one representative fuel channel using equal power sharing.

    The model performs a steady 1-D energy balance, ideal-gas density estimate,
    Dittus-Boelter heat-transfer estimate, and Darcy-Weisbach pressure loss.
    Axial conduction, acceleration pressure loss, real-gas effects, radiation,
    and fuel-matrix conduction are intentionally deferred to later fidelity
    stages.
    """
    if properties is None:
        properties = HydrogenProperties()

    power = np.asarray(axial_total_fuel_power_w, dtype=float)
    z_edges = np.asarray(z_edges_m, dtype=float)

    if power.ndim != 1:
        raise ValueError("axial_total_fuel_power_w must be one-dimensional")
    if z_edges.ndim != 1 or z_edges.size != power.size + 1:
        raise ValueError("z_edges_m must contain one more value than power bins")
    if np.any(np.diff(z_edges) <= 0.0):
        raise ValueError("z_edges_m must be strictly increasing")
    if np.any(power < 0.0):
        raise ValueError("axial power must be non-negative")
    if fuel_channel_count < 1:
        raise ValueError("fuel_channel_count must be positive")
    if mass_flow_per_channel_kg_s <= 0.0:
        raise ValueError("mass flow must be positive")
    if inlet_temperature_k <= 0.0 or inlet_pressure_pa <= 0.0:
        raise ValueError("inlet state must be positive")
    if channel_diameter_m <= 0.0:
        raise ValueError("channel diameter must be positive")
    if roughness_m < 0.0:
        raise ValueError("roughness must be non-negative")

    props = properties
    for name, value in (
        ("cp_j_kg_k", props.cp_j_kg_k),
        ("dynamic_viscosity_pa_s", props.dynamic_viscosity_pa_s),
        ("thermal_conductivity_w_m_k", props.thermal_conductivity_w_m_k),
        ("prandtl", props.prandtl),
        ("gas_constant_j_kg_k", props.gas_constant_j_kg_k),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive")

    n = power.size
    channel_power = power / float(fuel_channel_count)
    dz = np.diff(z_edges)
    z_center = 0.5 * (z_edges[:-1] + z_edges[1:])

    bulk_temperature = np.empty(n, dtype=float)
    wall_temperature = np.empty(n, dtype=float)
    pressure = np.empty(n, dtype=float)
    reynolds = np.empty(n, dtype=float)
    htc = np.empty(n, dtype=float)
    heat_flux = np.empty(n, dtype=float)

    area = 0.25 * np.pi * channel_diameter_m**2
    wetted_per_length = np.pi * channel_diameter_m
    relative_roughness = roughness_m / channel_diameter_m

    temperature_in = float(inlet_temperature_k)
    pressure_in = float(inlet_pressure_pa)

    for i in range(n):
        q = float(channel_power[i])
        delta_t = q / (mass_flow_per_channel_kg_s * props.cp_j_kg_k)
        temperature_mean = temperature_in + 0.5 * delta_t

        density = pressure_in / (props.gas_constant_j_kg_k * temperature_mean)
        velocity = mass_flow_per_channel_kg_s / (density * area)
        re = (
            density
            * velocity
            * channel_diameter_m
            / props.dynamic_viscosity_pa_s
        )
        nu = nusselt_number(re, props.prandtl)
        h = nu * props.thermal_conductivity_w_m_k / channel_diameter_m

        wetted_area = wetted_per_length * dz[i]
        q_flux = 0.0 if q == 0.0 else q / wetted_area
        wall_t = temperature_mean + q_flux / h

        f = friction_factor(re, relative_roughness)
        delta_p = (
            f
            * dz[i]
            / channel_diameter_m
            * 0.5
            * density
            * velocity**2
        )
        pressure_out = pressure_in - delta_p
        if pressure_out <= 0.0:
            raise ValueError(
                f"pressure became non-positive in axial bin {i}; "
                "reduce mass flow/length or increase inlet pressure"
            )

        temperature_out = temperature_in + delta_t

        bulk_temperature[i] = temperature_out
        wall_temperature[i] = wall_t
        pressure[i] = pressure_out
        reynolds[i] = re
        htc[i] = h
        heat_flux[i] = q_flux

        temperature_in = temperature_out
        pressure_in = pressure_out

    return ChannelSolution(
        z_center_m=z_center,
        bulk_temperature_k=bulk_temperature,
        wall_temperature_k=wall_temperature,
        pressure_pa=pressure,
        reynolds=reynolds,
        heat_transfer_coefficient_w_m2_k=htc,
        heat_flux_w_m2=heat_flux,
        channel_power_w=channel_power,
        mass_flow_kg_s=float(mass_flow_per_channel_kg_s),
        channel_diameter_m=float(channel_diameter_m),
    )
