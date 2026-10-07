"""Normalization helpers for OpenMC NERVA mesh tallies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EV_TO_J = 1.602176634e-19


@dataclass(frozen=True)
class NormalizedMeshFields:
    """Normalized fields on a regular Cartesian mesh."""

    power_density_w_cm3: np.ndarray
    fission_rate_cm3_s: np.ndarray
    flux_cm2_s: np.ndarray
    source_rate_s: float
    total_power_w: float
    cell_volume_cm3: float


def source_rate_for_power(
    reactor_power_w: float,
    total_heating_ev_per_source: float,
) -> float:
    """Return source particles/s required to normalize heating to reactor power."""
    if reactor_power_w <= 0.0:
        raise ValueError("reactor_power_w must be positive")
    if total_heating_ev_per_source <= 0.0:
        raise ValueError("total heating tally must be positive")

    return reactor_power_w / (total_heating_ev_per_source * EV_TO_J)


def normalize_regular_mesh(
    heating_ev_per_source: np.ndarray,
    fission_per_source: np.ndarray,
    flux_tracklength_per_source: np.ndarray,
    cell_volume_cm3: float,
    reactor_power_w: float,
) -> NormalizedMeshFields:
    """Normalize raw regular-mesh tallies to engineering-rate fields."""
    heating = np.asarray(heating_ev_per_source, dtype=float)
    fission = np.asarray(fission_per_source, dtype=float)
    flux = np.asarray(flux_tracklength_per_source, dtype=float)

    if heating.shape != fission.shape or heating.shape != flux.shape:
        raise ValueError("heating, fission, and flux fields must have identical shapes")
    if cell_volume_cm3 <= 0.0:
        raise ValueError("cell_volume_cm3 must be positive")

    total_heating = float(np.sum(heating))
    source_rate = source_rate_for_power(reactor_power_w, total_heating)

    voxel_power_w = heating * EV_TO_J * source_rate
    power_density = voxel_power_w / cell_volume_cm3
    fission_rate = fission * source_rate / cell_volume_cm3
    flux_rate = flux * source_rate / cell_volume_cm3

    return NormalizedMeshFields(
        power_density_w_cm3=power_density,
        fission_rate_cm3_s=fission_rate,
        flux_cm2_s=flux_rate,
        source_rate_s=source_rate,
        total_power_w=float(np.sum(voxel_power_w)),
        cell_volume_cm3=cell_volume_cm3,
    )
