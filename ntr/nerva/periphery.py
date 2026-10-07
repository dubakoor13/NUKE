"""Reactor-periphery geometry for the NERVA-derived demonstrator."""

from __future__ import annotations

import math

import openmc

from .config import NervaConfig
from .geometry import build_fuel_element_universe, build_tie_tube_universe
from .layout import axial_hex_coordinates, is_tie_tube_site, mixed_core_counts


def _build_mixed_lattice(
    config: NervaConfig,
    materials: dict[str, openmc.Material],
) -> openmc.HexLattice:
    """Create the mixed fuel/tie lattice used inside the full reactor model."""
    fuel_universe = build_fuel_element_universe(config, materials)
    tie_universe = build_tie_tube_universe(config, materials)

    filler_universe = openmc.Universe(
        name="homogenized peripheral graphite filler",
        cells=[
            openmc.Cell(
                name="graphite partial-filler surrogate",
                fill=materials["graphite"],
            )
        ],
    )

    lattice = openmc.HexLattice(name="reactor mixed fuel tie-tube lattice")
    lattice.center = (0.0, 0.0)
    lattice.pitch = (config.fuel_flat_to_flat_cm,)
    lattice.orientation = "y"

    rings = []
    for ring_index in range(config.core_rings - 1, 0, -1):
        rings.append([fuel_universe] * (6 * ring_index))
    rings.append([fuel_universe])
    lattice.universes = rings

    for index in axial_hex_coordinates(config.core_rings - 1):
        ring_index, within_index = lattice.get_universe_index(index)
        if is_tie_tube_site(index):
            rings[ring_index][within_index] = tie_universe

    lattice.universes = rings
    lattice.outer = filler_universe
    return lattice


def build_reactor_geometry(
    config: NervaConfig,
    materials: dict[str, openmc.Material],
) -> openmc.Geometry:
    """Build mixed core, filler, Be reflector, control drums, and Al vessel.

    The public NASA cross section establishes this topology. Drum radius,
    absorber thickness, reflector thickness, and vessel thickness remain
    parameterized engineering surrogates until calibrated to a specific test
    article.
    """
    config.validate()
    lattice = _build_mixed_lattice(config, materials)

    half_length = 0.5 * config.active_length_cm
    zmin = openmc.ZPlane(z0=-half_length, boundary_type="vacuum")
    zmax = openmc.ZPlane(z0=half_length, boundary_type="vacuum")
    axial_region = +zmin & -zmax

    core_cylinder = openmc.ZCylinder(r=config.core_radius_cm)
    reflector_outer = openmc.ZCylinder(r=config.reflector_outer_radius_cm)
    vessel_outer = openmc.ZCylinder(
        r=config.pressure_vessel_outer_radius_cm,
        boundary_type="vacuum",
    )

    cells: list[openmc.Cell] = [
        openmc.Cell(
            name="mixed fuel tie core plus homogenized peripheral filler",
            fill=lattice,
            region=-core_cylinder & axial_region,
        )
    ]

    reflector_region = +core_cylinder & -reflector_outer & axial_region

    drum_surfaces: list[openmc.ZCylinder] = []
    drum_angle = math.radians(config.control_drum_angle_deg)

    for index in range(config.control_drum_count):
        theta = 2.0 * math.pi * index / config.control_drum_count
        center_radius = config.control_drum_center_radius_cm
        cx = center_radius * math.cos(theta)
        cy = center_radius * math.sin(theta)

        drum_outer = openmc.ZCylinder(
            x0=cx,
            y0=cy,
            r=config.control_drum_radius_cm,
            name=f"control drum {index + 1} outer",
        )
        absorber_inner = openmc.ZCylinder(
            x0=cx,
            y0=cy,
            r=config.control_absorber_inner_radius_cm,
            name=f"control drum {index + 1} absorber inner",
        )
        drum_surfaces.append(drum_outer)

        # Angle 0 deg places the absorber-bearing half-shell toward the core.
        base_x = -math.cos(theta)
        base_y = -math.sin(theta)
        nx = math.cos(drum_angle) * base_x - math.sin(drum_angle) * base_y
        ny = math.sin(drum_angle) * base_x + math.cos(drum_angle) * base_y

        split = openmc.Plane(
            a=nx,
            b=ny,
            d=nx * cx + ny * cy,
            name=f"control drum {index + 1} absorber split",
        )

        absorber_region = (
            +absorber_inner
            & -drum_outer
            & +split
            & axial_region
        )
        beryllium_region = -drum_outer & ~absorber_region & axial_region

        cells.append(
            openmc.Cell(
                name=f"control drum {index + 1} B4C absorber",
                fill=materials["b4c"],
                region=absorber_region,
            )
        )
        cells.append(
            openmc.Cell(
                name=f"control drum {index + 1} beryllium body",
                fill=materials["reflector"],
                region=beryllium_region,
            )
        )

    for drum_outer in drum_surfaces:
        reflector_region &= +drum_outer

    cells.append(
        openmc.Cell(
            name="beryllium radial reflector with drum bores",
            fill=materials["reflector"],
            region=reflector_region,
        )
    )

    cells.append(
        openmc.Cell(
            name="aluminum-alloy pressure-vessel surrogate",
            fill=materials["aluminum"],
            region=+reflector_outer & -vessel_outer & axial_region,
        )
    )

    root = openmc.Universe(name="NERVA-derived reactor assembly", cells=cells)
    return openmc.Geometry(root)


def reactor_summary(config: NervaConfig) -> dict[str, float | int | str]:
    fuel_elements, tie_tubes = mixed_core_counts(config.core_rings)
    return {
        "fuel_elements": fuel_elements,
        "tie_tubes": tie_tubes,
        "fuel_coolant_channels": 19 * fuel_elements,
        "tie_tube_hydrogen_passages": 2 * tie_tubes,
        "peripheral_filler_model": "homogenized graphite partial-filler surrogate",
        "control_drums": config.control_drum_count,
        "control_drum_angle_deg": config.control_drum_angle_deg,
        "control_absorber": "natural B4C half-shell surrogate",
        "reflector_outer_radius_cm": config.reflector_outer_radius_cm,
        "pressure_vessel_outer_radius_cm": config.pressure_vessel_outer_radius_cm,
    }
