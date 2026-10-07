"""Pure-Python geometric helpers for NERVA-like fuel elements and lattices."""

from __future__ import annotations

import math
from typing import List, Tuple

from .config import NervaConfig


Point2D = Tuple[float, float]
HexIndex = Tuple[int, int]

_HEX_NEIGHBORS: tuple[HexIndex, ...] = (
    (1, 0),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (0, -1),
    (1, -1),
)


def axial_hex_coordinates(radius: int) -> List[HexIndex]:
    if radius < 0:
        raise ValueError("radius must be non-negative")

    result: List[HexIndex] = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if max(abs(q), abs(r), abs(q + r)) <= radius:
                result.append((q, r))

    def key(item: HexIndex) -> Tuple[float, float]:
        q, r = item
        x = q + 0.5 * r
        y = 0.5 * math.sqrt(3.0) * r
        return (math.hypot(x, y), math.atan2(y, x))

    result.sort(key=key)
    return result


def coolant_channel_positions(config: NervaConfig) -> List[Point2D]:
    positions: List[Point2D] = []
    for q, r in axial_hex_coordinates(2):
        x = config.coolant_bore_pitch_cm * (q + 0.5 * r)
        y = config.coolant_bore_pitch_cm * (0.5 * math.sqrt(3.0) * r)
        positions.append((x, y))
    return positions


def minimum_channel_ligament_cm(config: NervaConfig) -> float:
    return config.coolant_bore_pitch_cm - 2.0 * config.coated_channel_radius_cm


def outer_channel_margin_cm(config: NervaConfig) -> float:
    positions = coolant_channel_positions(config)
    apothem = 0.5 * config.inner_fuel_flat_to_flat_cm
    normals = (
        (0.0, 1.0),
        (math.sqrt(3.0) / 2.0, 0.5),
        (math.sqrt(3.0) / 2.0, -0.5),
    )
    margins = []
    for x, y in positions:
        support = max(abs(nx * x + ny * y) for nx, ny in normals)
        margins.append(apothem - support - config.coated_channel_radius_cm)
    return min(margins)


def is_tie_tube_site(index: HexIndex) -> bool:
    """Return True for the tie-tube sublattice of a three-color hex pattern.

    On the triangular lattice this coloring gives each tie-tube site six fuel
    neighbors. Each interior fuel site then has three tie-tube and three fuel
    neighbors, matching the public SNRE-style connectivity.
    """
    q, r = index
    return (q - r) % 3 == 0


def mixed_core_counts(rings: int) -> tuple[int, int]:
    """Return fuel-element and tie-tube counts for the mixed-core pattern."""
    if rings < 1:
        raise ValueError("rings must be >= 1")
    coordinates = axial_hex_coordinates(rings - 1)
    tie_tubes = sum(1 for index in coordinates if is_tie_tube_site(index))
    return len(coordinates) - tie_tubes, tie_tubes


def mixed_core_neighbor_check(rings: int) -> tuple[int, int]:
    """Validate interior SNRE-style neighbor connectivity."""
    if rings < 2:
        return (0, 0)

    radius = rings - 1
    coordinates = set(axial_hex_coordinates(radius))
    interior_ties = 0
    interior_fuels = 0

    for q, r in coordinates:
        g = max(abs(q), abs(r), abs(q + r))
        if g >= radius:
            continue

        neighbor_types = []
        for dq, dr in _HEX_NEIGHBORS:
            neighbor = (q + dq, r + dr)
            if neighbor not in coordinates:
                raise AssertionError("interior site has a missing lattice neighbor")
            neighbor_types.append(is_tie_tube_site(neighbor))

        if is_tie_tube_site((q, r)):
            if any(neighbor_types):
                raise AssertionError("interior tie tube has a tie-tube neighbor")
            interior_ties += 1
        else:
            if sum(neighbor_types) != 3:
                raise AssertionError("interior fuel element does not have three tie-tube neighbors")
            interior_fuels += 1

    return interior_ties, interior_fuels
