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


def fuel_sector_coated_channel_areas_cm2(
    config: NervaConfig,
) -> np.ndarray:
    """Exact *inner-hex footprint* of each coated coolant circle [cm²].

    OpenMC uses orientation="y" (vertical flats), not the older
    horizontal-flat layout assumption. Six outer channels cross a single
    inner-hex flat by ~0.00363 cm at nominal dimensions. Only the part
    INSIDE the inner hex displaces the fuel matrix; the outer part is
    already in the external ZrC region. Coating material identity on
    both sides is unchanged. This is an exact circle/half-plane cap
    calculation, not a polygon sampling or fabricated volume correction.

    The geometry formula supports clipping against exactly one flat per
    channel. Fail closed if a proposed geometry reaches two flats,
    touches the outer hex, or sends a hydrogen bore outside the inner hex.
    """
    config.validate()
    radius = config.coated_channel_radius_cm
    bore = config.channel_radius_cm
    apothem = config.inner_fuel_flat_to_flat_cm / 2.0
    external_apothem = config.fuel_flat_to_flat_cm / 2.0
    normals = (
        (1.0, 0.0),
        (0.5, math.sqrt(3.0) / 2.0),
        (0.5, -math.sqrt(3.0) / 2.0),
    )
    faces = normals + tuple((-nx, -ny) for nx, ny in normals)
    effective = []
    for x, y in coolant_channel_positions(config):
        distances = [apothem - nx * x - ny * y for nx, ny in faces]
        if min(distances) <= bore:
            raise ValueError("hydrogen bore extends beyond actual inner fuel hex")
        if min(distances) <= 0:
            raise ValueError("coolant-channel centre lies outside inner fuel hex")
        outside_outer = [
            external_apothem - nx * x - ny * y
            for nx, ny in faces
        ]
        if min(outside_outer) <= radius:
            raise ValueError("coated coolant channel exits actual outer hex")
        cross = [d for d in distances if d < radius]
        if len(cross) > 1:
            raise ValueError("two-flat coated-circle clipping not qualified")
        cap = 0.0
        if cross:
            d = cross[0]
            cap = (radius**2 * math.acos(d / radius)
                   - d * math.sqrt(radius**2 - d**2))
        area = math.pi * radius**2 - cap
        if area <= math.pi * bore**2:
            raise ValueError("nonphysical clipped coated-channel area")
        effective.append(area)
    if len(effective) != 19:
        raise ValueError("19 actual fuel-element channels required")
    return np.asarray(effective, dtype=float)


def fuel_sector_fuel_areas_cm2(
    config: NervaConfig,
) -> np.ndarray:
    """Actual 19 Voronoi fuel volumes per axial length in OpenMC CSG.

    The former implementation removed a COMPLETE circle from each
    inner-hex sector and lost the fuel area of the six clipped rings.
    This uses only the true inner-hex footprints of those circles.
    """
    total = fuel_sector_total_areas_cm2(config)
    coated_in_fuel = fuel_sector_coated_channel_areas_cm2(config)
    fuel = total - coated_in_fuel
    if np.any(fuel <= 0.0) or not np.all(np.isfinite(fuel)):
        raise ValueError(
            "fuel-sector area is non-positive after true CSG channel subtraction"
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
