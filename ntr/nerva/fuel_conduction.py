"""First-pass solid conduction estimates for a NERVA-like fuel element."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import NervaConfig
from .layout import minimum_channel_ligament_cm
from .thermal import ChannelSolution


@dataclass(frozen=True)
class FuelSolidProperties:
    """Constant solid conductivities for the first thermal-fidelity stage."""

    fuel_matrix_conductivity_w_m_k: float = 25.0
    zrc_conductivity_w_m_k: float = 20.0


@dataclass(frozen=True)
class FuelSolidTemperatureEstimate:
    """Temperatures from coolant wall through coating into the fuel matrix."""

    coolant_wall_temperature_k: np.ndarray
    fuel_surface_temperature_k: np.ndarray
    peak_fuel_temperature_k: np.ndarray
    effective_half_ligament_m: float
    coating_thickness_m: float


def estimate_fuel_solid_temperatures(
    channel: ChannelSolution,
    config: NervaConfig | None = None,
    properties: FuelSolidProperties | None = None,
) -> FuelSolidTemperatureEstimate:
    """Estimate coating and local peak fuel temperatures.

    The coolant-side wall temperature comes from the convective channel solve.
    The ZrC temperature rise is modeled as 1-D conduction through the channel
    coating. The fuel-matrix rise uses a symmetry-slab estimate over half of the
    minimum ligament between neighboring coated channels:

        delta_T_fuel = q_wall * L_half / (2 k_fuel)

    This is intentionally a screening estimate, not a 2-D/3-D fuel-element
    conduction solution.
    """
    if config is None:
        config = NervaConfig()
    if properties is None:
        properties = FuelSolidProperties()

    config.validate()

    if properties.fuel_matrix_conductivity_w_m_k <= 0.0:
        raise ValueError("fuel matrix conductivity must be positive")
    if properties.zrc_conductivity_w_m_k <= 0.0:
        raise ValueError("ZrC conductivity must be positive")

    coating_thickness_m = config.zrc_coating_thickness_cm / 100.0
    half_ligament_m = 0.5 * minimum_channel_ligament_cm(config) / 100.0

    heat_flux = np.asarray(channel.heat_flux_w_m2, dtype=float)
    wall = np.asarray(channel.wall_temperature_k, dtype=float)

    fuel_surface = (
        wall
        + heat_flux
        * coating_thickness_m
        / properties.zrc_conductivity_w_m_k
    )
    peak_fuel = (
        fuel_surface
        + heat_flux
        * half_ligament_m
        / (2.0 * properties.fuel_matrix_conductivity_w_m_k)
    )

    return FuelSolidTemperatureEstimate(
        coolant_wall_temperature_k=wall,
        fuel_surface_temperature_k=fuel_surface,
        peak_fuel_temperature_k=peak_fuel,
        effective_half_ligament_m=float(half_ligament_m),
        coating_thickness_m=float(coating_thickness_m),
    )


@dataclass(frozen=True)
class FuelSectorTemperatureEstimate:
    """Equivalent-annulus temperatures for one channel-associated fuel sector."""

    coolant_wall_temperature_k: np.ndarray
    fuel_surface_temperature_k: np.ndarray
    peak_fuel_temperature_k: np.ndarray
    sector_fuel_area_m2: float
    coating_inner_radius_m: float
    coating_outer_radius_m: float
    equivalent_outer_radius_m: float
    volumetric_heating_w_m3: np.ndarray


def estimate_fuel_sector_temperatures(
    channel: ChannelSolution,
    z_edges_m: np.ndarray,
    sector_fuel_area_m2: float,
    config: NervaConfig | None = None,
    properties: FuelSolidProperties | None = None,
) -> FuelSectorTemperatureEstimate:
    """Estimate local solid temperatures with an equivalent heated annulus.

    The actual Voronoi fuel sector around one coolant channel is replaced by an
    annulus with the same fuel-material cross-sectional area. Uniform
    volumetric heat generation is assumed within each axial fuel-sector bin,
    with an adiabatic outer annulus boundary and heat removal through the
    coated coolant bore.

    The coating drop uses the exact cylindrical resistance:

        DeltaT_ZrC = Q'/(2*pi*k_ZrC) * ln(r_o/r_i)

    The uniformly heated equivalent fuel annulus has:

        DeltaT_fuel =
            q'''/(2*k_fuel) *
            [R^2 ln(R/r_o) - (R^2-r_o^2)/2]

    where r_o is the outer coating radius and R is chosen so that
    pi*(R^2-r_o^2) equals the actual Voronoi-sector fuel area.

    This remains a reduced-order conduction model; it is not a 2-D finite
    element solution of the real hexagonal sector.
    """
    if config is None:
        config = NervaConfig()
    if properties is None:
        properties = FuelSolidProperties()

    config.validate()

    if properties.fuel_matrix_conductivity_w_m_k <= 0.0:
        raise ValueError("fuel matrix conductivity must be positive")
    if properties.zrc_conductivity_w_m_k <= 0.0:
        raise ValueError("ZrC conductivity must be positive")
    if sector_fuel_area_m2 <= 0.0:
        raise ValueError("sector_fuel_area_m2 must be positive")

    z_edges = np.asarray(z_edges_m, dtype=float)
    wall_power = np.asarray(
        channel.wall_heat_power_w,
        dtype=float,
    )
    wall = np.asarray(
        channel.wall_temperature_k,
        dtype=float,
    )

    if z_edges.ndim != 1 or z_edges.size != wall_power.size + 1:
        raise ValueError(
            "z_edges_m must contain one more value than channel power bins"
        )
    dz = np.diff(z_edges)
    if np.any(dz <= 0.0):
        raise ValueError("z_edges_m must be strictly increasing")
    if np.any(wall_power < 0.0):
        raise ValueError("wall heat power must be non-negative")

    coating_inner_radius_m = (
        config.channel_radius_cm / 100.0
    )
    coating_outer_radius_m = (
        config.coated_channel_radius_cm / 100.0
    )
    if coating_outer_radius_m <= coating_inner_radius_m:
        raise ValueError(
            "coating outer radius must exceed coolant-bore radius"
        )

    equivalent_outer_radius_m = float(
        np.sqrt(
            coating_outer_radius_m**2
            + sector_fuel_area_m2 / np.pi
        )
    )

    heat_per_length_w_m = wall_power / dz
    coating_delta_t = (
        heat_per_length_w_m
        / (
            2.0
            * np.pi
            * properties.zrc_conductivity_w_m_k
        )
        * np.log(
            coating_outer_radius_m
            / coating_inner_radius_m
        )
    )

    volumetric_heating_w_m3 = (
        wall_power
        / (sector_fuel_area_m2 * dz)
    )

    r_outer = equivalent_outer_radius_m
    r_inner = coating_outer_radius_m
    geometry_factor_m2 = (
        r_outer**2 * np.log(r_outer / r_inner)
        - 0.5 * (r_outer**2 - r_inner**2)
    )
    if geometry_factor_m2 < 0.0:
        raise ValueError(
            "equivalent-annulus geometry factor is negative"
        )

    fuel_delta_t = (
        volumetric_heating_w_m3
        * geometry_factor_m2
        / (
            2.0
            * properties.fuel_matrix_conductivity_w_m_k
        )
    )

    fuel_surface = wall + coating_delta_t
    peak_fuel = fuel_surface + fuel_delta_t

    return FuelSectorTemperatureEstimate(
        coolant_wall_temperature_k=wall,
        fuel_surface_temperature_k=fuel_surface,
        peak_fuel_temperature_k=peak_fuel,
        sector_fuel_area_m2=float(sector_fuel_area_m2),
        coating_inner_radius_m=float(
            coating_inner_radius_m
        ),
        coating_outer_radius_m=float(
            coating_outer_radius_m
        ),
        equivalent_outer_radius_m=(
            equivalent_outer_radius_m
        ),
        volumetric_heating_w_m3=(
            volumetric_heating_w_m3
        ),
    )
