"""Normalization helpers for OpenMC NERVA mesh tallies."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


EV_TO_J = 1.602176634e-19

def heating_score_for_transport(photon_transport: bool) -> str:
    """Physically consistent OpenMC deposition estimator for the run mode.

    neutron-only: heating-local, including secondary-gamma energy locally.
    neutron-photon: heating, tallying energy where photons deposit it.
    Never normalize a transported-photon spatial power map with heating-local.
    """
    if not isinstance(photon_transport, (bool, np.bool_)):
        raise TypeError("photon_transport must be boolean")
    return "heating" if photon_transport else "heating-local"



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
    if not math.isfinite(reactor_power_w) or reactor_power_w <= 0:
        raise ValueError("reactor_power_w must be positive finite")
    if (not math.isfinite(total_heating_ev_per_source)
            or total_heating_ev_per_source <= 0):
        raise ValueError("total heating tally must be positive finite")
    rate = reactor_power_w / (total_heating_ev_per_source * EV_TO_J)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("normalized OpenMC source rate is nonfinite")
    return rate


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

    if heating.size == 0 or heating.shape != fission.shape or heating.shape != flux.shape:
        raise ValueError("heating, fission, and flux fields must have identical nonempty shapes")
    if not math.isfinite(cell_volume_cm3) or cell_volume_cm3 <= 0.0:
        raise ValueError("cell_volume_cm3 must be positive finite")
    if (not np.isfinite(heating).all()
            or not np.isfinite(fission).all()
            or not np.isfinite(flux).all()):
        raise ValueError("nonfinite OpenMC heating, fission, or flux tallies")
    if np.any(fission < 0) or np.any(flux < 0):
        raise ValueError("negative OpenMC reaction rate or tracklength")
    scale = float(np.max(np.abs(heating)))
    if scale <= 0 or float(np.min(heating)) < -1e-6 * scale:
        raise ValueError("nonpositive or materially negative OpenMC deposited heating")

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
