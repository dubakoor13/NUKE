"""Deterministic geometry helpers for the 19 fuel-channel scoring sectors."""

from __future__ import annotations

import math

import numpy as np

from .config import NervaConfig
from .layout import coolant_channel_positions


Point = tuple[float, float]


def inner_fuel_hex_vertices_cm(
    config: NervaConfig,
) -> list[Point]:
    """Return the orientation='y' inner-fuel hex vertices in CCW order."""
    side = config.inner_fuel_edge_length_cm
    x = 0.5 * math.sqrt(3.0) * side
    return [
        (0.0, side),
        (-x, 0.5 * side),
        (-x, -0.5 * side),
        (0.0, -side),
        (x, -0.5 * side),
        (x, 0.5 * side),
    ]


def _clip_halfplane(
    polygon: list[Point],
    a: float,
    b: float,
    d: float,
    tolerance: float = 1.0e-12,
) -> list[Point]:
    """Clip a polygon to a*x + b*y <= d."""
    if not polygon:
        return []

    result: list[Point] = []

    def value(point: Point) -> float:
        return a * point[0] + b * point[1] - d

    previous = polygon[-1]
    f_previous = value(previous)
    previous_inside = f_previous <= tolerance

    for current in polygon:
        f_current = value(current)
        current_inside = f_current <= tolerance

        if current_inside != previous_inside:
            denominator = f_previous - f_current
            if abs(denominator) > 1.0e-30:
                t = f_previous / denominator
                intersection = (
                    previous[0] + t * (current[0] - previous[0]),
                    previous[1] + t * (current[1] - previous[1]),
                )
                result.append(intersection)

        if current_inside:
            result.append(current)

        previous = current
        f_previous = f_current
        previous_inside = current_inside

    return result


def polygon_area_cm2(polygon: list[Point]) -> float:
    if len(polygon) < 3:
        return 0.0

    total = 0.0
    for i, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(i + 1) % len(polygon)]
        total += x1 * y2 - x2 * y1
    return 0.5 * abs(total)


def fuel_sector_polygons_cm(
    config: NervaConfig,
) -> list[list[Point]]:
    """Return the 19 nearest-channel Voronoi cells clipped by the inner hex."""
    positions = coolant_channel_positions(config)
    base = inner_fuel_hex_vertices_cm(config)
    sectors: list[list[Point]] = []

    for i, (xi, yi) in enumerate(positions):
        polygon = list(base)

        for j, (xj, yj) in enumerate(positions):
            if i == j:
                continue

            # Points closer to i than j satisfy:
            # (p_j - p_i).x <= (|p_j|^2 - |p_i|^2)/2
            a = xj - xi
            b = yj - yi
            d = 0.5 * (
                xj * xj
                + yj * yj
                - xi * xi
                - yi * yi
            )
            polygon = _clip_halfplane(
                polygon,
                a,
                b,
                d,
            )
            if not polygon:
                raise RuntimeError(
                    f"fuel sector {i + 1} became empty"
                )

        sectors.append(polygon)

    return sectors


def fuel_sector_total_areas_cm2(
    config: NervaConfig,
) -> np.ndarray:
    """Area inside each Voronoi sector before subtracting its coated bore."""
    return np.asarray(
        [
            polygon_area_cm2(polygon)
            for polygon in fuel_sector_polygons_cm(config)
        ],
        dtype=float,
    )


def fuel_sector_fuel_areas_cm2(
    config: NervaConfig,
) -> np.ndarray:
    """Fuel-material area in each sector, excluding the local coated channel."""
    total = fuel_sector_total_areas_cm2(config)
    coated_hole = math.pi * config.coated_channel_radius_cm**2
    fuel = total - coated_hole

    if np.any(fuel <= 0.0):
        raise ValueError(
            "fuel-sector area is non-positive after coated-channel subtraction"
        )
    return fuel


def fuel_sector_equivalent_outer_radii_m(
    config: NervaConfig,
) -> np.ndarray:
    """Equivalent annulus outer radius preserving each sector fuel area."""
    fuel_area_m2 = fuel_sector_fuel_areas_cm2(config) * 1.0e-4
    inner_radius_m = config.coated_channel_radius_cm / 100.0
    return np.sqrt(
        inner_radius_m**2
        + fuel_area_m2 / math.pi
    )
