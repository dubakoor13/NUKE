"""Counterflow thermal-hydraulic model for a NERVA-derived tie tube."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NervaConfig
from .thermal import HydrogenProperties, friction_factor, nusselt_number


@dataclass(frozen=True)
class DuctPathSolution:
    """One flow path, stored in flow order."""

    z_center_m: np.ndarray
    bulk_temperature_k: np.ndarray
    wall_temperature_k: np.ndarray
    pressure_pa: np.ndarray
    reynolds: np.ndarray
    heat_transfer_coefficient_w_m2_k: np.ndarray
    heat_flux_w_m2: np.ndarray
    path_power_w: np.ndarray

    @property
    def outlet_temperature_k(self) -> float:
        return float(self.bulk_temperature_k[-1])

    @property
    def outlet_pressure_pa(self) -> float:
        return float(self.pressure_pa[-1])

    @property
    def absorbed_power_w(self) -> float:
        return float(np.sum(self.path_power_w))


@dataclass(frozen=True)
class TieTubeSolution:
    """Supply-down / return-up solution for one representative tie tube."""

    supply: DuctPathSolution
    return_path: DuctPathSolution
    mass_flow_per_tie_kg_s: float
    supply_heat_fraction: float
    tie_power_w: float

    @property
    def outlet_temperature_k(self) -> float:
        return self.return_path.outlet_temperature_k

    @property
    def outlet_pressure_pa(self) -> float:
        return self.return_path.outlet_pressure_pa


def default_supply_heat_fraction(config: NervaConfig) -> float:
    """Perimeter-weighted first estimate of heat split between tie flow paths."""
    supply_d = 2.0 * config.tie_inner_tube_inner_radius_cm
    return_inner_d = 2.0 * config.tie_zrh_outer_radius_cm
    return_outer_d = 2.0 * config.tie_outer_tube_inner_radius_cm

    supply_perimeter = np.pi * supply_d
    return_perimeter = np.pi * (return_inner_d + return_outer_d)
    return float(supply_perimeter / (supply_perimeter + return_perimeter))


def _solve_duct_path(
    path_power_w: np.ndarray,
    z_center_m: np.ndarray,
    dz_m: np.ndarray,
    mass_flow_kg_s: float,
    inlet_temperature_k: float,
    inlet_pressure_pa: float,
    flow_area_m2: float,
    hydraulic_diameter_m: float,
    heated_perimeter_m: float,
    properties: HydrogenProperties,
    roughness_m: float,
) -> DuctPathSolution:
    power = np.asarray(path_power_w, dtype=float)
    z_center = np.asarray(z_center_m, dtype=float)
    dz = np.asarray(dz_m, dtype=float)

    if power.ndim != 1 or z_center.shape != power.shape or dz.shape != power.shape:
        raise ValueError("path power, z centers, and dz must be one-dimensional and equal length")
    if np.any(power < 0.0) or np.any(dz <= 0.0):
        raise ValueError("path power must be non-negative and dz must be positive")
    if mass_flow_kg_s <= 0.0:
        raise ValueError("mass_flow_kg_s must be positive")
    if inlet_temperature_k <= 0.0 or inlet_pressure_pa <= 0.0:
        raise ValueError("inlet state must be positive")
    if flow_area_m2 <= 0.0 or hydraulic_diameter_m <= 0.0 or heated_perimeter_m <= 0.0:
        raise ValueError("duct geometry must be positive")
    if roughness_m < 0.0:
        raise ValueError("roughness_m must be non-negative")

    n = power.size
    bulk_temperature = np.empty(n)
    wall_temperature = np.empty(n)
    pressure = np.empty(n)
    reynolds = np.empty(n)
    htc = np.empty(n)
    heat_flux = np.empty(n)

    temperature_in = float(inlet_temperature_k)
    pressure_in = float(inlet_pressure_pa)
    rel_roughness = roughness_m / hydraulic_diameter_m

    for i in range(n):
        q = float(power[i])
        delta_t = q / (mass_flow_kg_s * properties.cp_j_kg_k)
        temperature_mean = temperature_in + 0.5 * delta_t

        density = pressure_in / (
            properties.gas_constant_j_kg_k * temperature_mean
        )
        velocity = mass_flow_kg_s / (density * flow_area_m2)
        re = (
            density
            * velocity
            * hydraulic_diameter_m
            / properties.dynamic_viscosity_pa_s
        )
        nu = nusselt_number(re, properties.prandtl)
        h = (
            nu
            * properties.thermal_conductivity_w_m_k
            / hydraulic_diameter_m
        )

        heated_area = heated_perimeter_m * dz[i]
        q_flux = 0.0 if q == 0.0 else q / heated_area
        wall_t = temperature_mean + q_flux / h

        f = friction_factor(re, rel_roughness)
        delta_p = (
            f
            * dz[i]
            / hydraulic_diameter_m
            * 0.5
            * density
            * velocity**2
        )
        pressure_out = pressure_in - delta_p
        if pressure_out <= 0.0:
            raise ValueError(
                f"pressure became non-positive in path bin {i}; "
                "reduce mass flow or increase inlet pressure"
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

    return DuctPathSolution(
        z_center_m=z_center,
        bulk_temperature_k=bulk_temperature,
        wall_temperature_k=wall_temperature,
        pressure_pa=pressure,
        reynolds=reynolds,
        heat_transfer_coefficient_w_m2_k=htc,
        heat_flux_w_m2=heat_flux,
        path_power_w=power,
    )


def solve_tie_tube_counterflow(
    axial_total_tie_power_w: np.ndarray,
    z_edges_m: np.ndarray,
    tie_tube_count: int,
    mass_flow_per_tie_kg_s: float,
    inlet_temperature_k: float,
    inlet_pressure_pa: float,
    config: NervaConfig | None = None,
    properties: HydrogenProperties | None = None,
    supply_heat_fraction: float | None = None,
    roughness_m: float = 1.0e-6,
) -> TieTubeSolution:
    """Solve one representative tie tube with supply and annular return flow.

    The central supply path flows from low-z to high-z. The annular return path
    then flows back from high-z to low-z. Heat deposited in all tie-tube solids
    is divided between the two paths. If no explicit split is supplied, the
    first-pass estimate is proportional to their wetted perimeters.
    """
    if config is None:
        config = NervaConfig()
    if properties is None:
        properties = HydrogenProperties()
    config.validate()

    power = np.asarray(axial_total_tie_power_w, dtype=float)
    z_edges = np.asarray(z_edges_m, dtype=float)

    if power.ndim != 1:
        raise ValueError("axial_total_tie_power_w must be one-dimensional")
    if z_edges.ndim != 1 or z_edges.size != power.size + 1:
        raise ValueError("z_edges_m must contain one more entry than power bins")
    if np.any(np.diff(z_edges) <= 0.0):
        raise ValueError("z_edges_m must be strictly increasing")
    if np.any(power < 0.0):
        raise ValueError("tie power must be non-negative")
    if tie_tube_count < 1:
        raise ValueError("tie_tube_count must be positive")
    if mass_flow_per_tie_kg_s <= 0.0:
        raise ValueError("mass_flow_per_tie_kg_s must be positive")

    if supply_heat_fraction is None:
        supply_heat_fraction = default_supply_heat_fraction(config)
    if not (0.0 <= supply_heat_fraction <= 1.0):
        raise ValueError("supply_heat_fraction must lie between 0 and 1")

    per_tie_power = power / float(tie_tube_count)
    dz = np.diff(z_edges)
    z_center = 0.5 * (z_edges[:-1] + z_edges[1:])

    supply_power = per_tie_power * supply_heat_fraction
    return_power_physical = per_tie_power * (1.0 - supply_heat_fraction)

    supply_diameter_m = (
        2.0 * config.tie_inner_tube_inner_radius_cm / 100.0
    )
    supply_area_m2 = 0.25 * np.pi * supply_diameter_m**2
    supply_perimeter_m = np.pi * supply_diameter_m

    return_inner_diameter_m = config.tie_zrh_outer_diameter_cm / 100.0
    return_outer_diameter_m = (
        2.0 * config.tie_outer_tube_inner_radius_cm / 100.0
    )
    return_area_m2 = 0.25 * np.pi * (
        return_outer_diameter_m**2 - return_inner_diameter_m**2
    )
    return_hydraulic_diameter_m = (
        return_outer_diameter_m - return_inner_diameter_m
    )
    return_perimeter_m = np.pi * (
        return_outer_diameter_m + return_inner_diameter_m
    )

    supply = _solve_duct_path(
        path_power_w=supply_power,
        z_center_m=z_center,
        dz_m=dz,
        mass_flow_kg_s=mass_flow_per_tie_kg_s,
        inlet_temperature_k=inlet_temperature_k,
        inlet_pressure_pa=inlet_pressure_pa,
        flow_area_m2=supply_area_m2,
        hydraulic_diameter_m=supply_diameter_m,
        heated_perimeter_m=supply_perimeter_m,
        properties=properties,
        roughness_m=roughness_m,
    )

    return_path = _solve_duct_path(
        path_power_w=return_power_physical[::-1],
        z_center_m=z_center[::-1],
        dz_m=dz[::-1],
        mass_flow_kg_s=mass_flow_per_tie_kg_s,
        inlet_temperature_k=supply.outlet_temperature_k,
        inlet_pressure_pa=supply.outlet_pressure_pa,
        flow_area_m2=return_area_m2,
        hydraulic_diameter_m=return_hydraulic_diameter_m,
        heated_perimeter_m=return_perimeter_m,
        properties=properties,
        roughness_m=roughness_m,
    )

    return TieTubeSolution(
        supply=supply,
        return_path=return_path,
        mass_flow_per_tie_kg_s=float(mass_flow_per_tie_kg_s),
        supply_heat_fraction=float(supply_heat_fraction),
        tie_power_w=float(np.sum(per_tie_power)),
    )
