"""Pure-Python geometric helpers for NERVA-like fuel elements."""

from __future__ import annotations

import math
from typing import List, Tuple

from .config import NervaConfig


Point2D = Tuple[float, float]


def axial_hex_coordinates(radius: int) -> List[Tuple[int, int]]:
    if radius < 0:
        raise ValueError("radius must be non-negative")

    result: List[Tuple[int, int]] = []
    for q in range(-radius, radius + 1):
        for r in range(-radius, radius + 1):
            if max(abs(q), abs(r), abs(q + r)) <= radius:
                result.append((q, r))

    def key(item: Tuple[int, int]) -> Tuple[float, float]:
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
