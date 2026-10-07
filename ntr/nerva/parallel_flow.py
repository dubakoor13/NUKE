"""Utilities for balancing flow among parallel hydraulic branches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParallelFlowUpdate:
    mass_flow_kg_s: np.ndarray
    target_mass_flow_kg_s: np.ndarray
    max_relative_change: float


def pressure_spread_fraction(delta_pressure_pa: np.ndarray) -> float:
    values = np.asarray(delta_pressure_pa, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    mean = float(np.mean(values))
    if mean <= 0.0:
        return 0.0
    return float((np.max(values) - np.min(values)) / mean)


def update_parallel_mass_flow(
    current_mass_flow_kg_s: np.ndarray,
    delta_pressure_pa: np.ndarray,
    total_mass_flow_kg_s: float,
    relaxation: float = 0.5,
) -> ParallelFlowUpdate:
    """Update branch flows from an effective quadratic resistance.

    For each branch, define an iteration-local resistance

        R_i = DeltaP_i / mdot_i^2.

    A common pressure drop then implies mdot_i proportional to 1/sqrt(R_i).
    The resulting target distribution is relaxed against the current
    distribution. This does not assume R_i remains constant between iterations;
    the thermal-hydraulic branch solver should be rerun after every update.
    """
    current = np.asarray(
        current_mass_flow_kg_s,
        dtype=float,
    ).reshape(-1)
    delta_p = np.asarray(
        delta_pressure_pa,
        dtype=float,
    ).reshape(-1)

    if current.shape != delta_p.shape:
        raise ValueError("mass-flow and pressure-drop arrays must match")
    if current.size == 0:
        raise ValueError("at least one hydraulic branch is required")
    if total_mass_flow_kg_s <= 0.0:
        raise ValueError("total mass flow must be positive")
    if not (0.0 < relaxation <= 1.0):
        raise ValueError("relaxation must lie in (0, 1]")
    if np.any(current <= 0.0):
        raise ValueError("all branch mass flows must be positive")
    if np.any(delta_p <= 0.0):
        raise ValueError("all branch pressure drops must be positive")

    resistance = delta_p / (current**2)
    conductance = 1.0 / np.sqrt(resistance)

    target = (
        total_mass_flow_kg_s
        * conductance
        / float(np.sum(conductance))
    )

    updated = (
        (1.0 - relaxation) * current
        + relaxation * target
    )
    updated *= (
        total_mass_flow_kg_s / float(np.sum(updated))
    )

    relative = np.abs(updated - current) / np.maximum(
        current,
        1.0e-30,
    )

    return ParallelFlowUpdate(
        mass_flow_kg_s=updated,
        target_mass_flow_kg_s=target,
        max_relative_change=float(np.max(relative)),
    )
