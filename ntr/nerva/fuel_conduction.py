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
