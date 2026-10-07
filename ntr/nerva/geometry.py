"""OpenMC geometry construction for the NERVA-derived model."""

from __future__ import annotations

import openmc

from .config import NervaConfig
from .layout import coolant_channel_positions


def build_fuel_element_universe(
    config: NervaConfig,
    materials: dict[str, openmc.Material],
) -> openmc.Universe:
    inner_hex = openmc.model.HexagonalPrism(
        edge_length=config.inner_fuel_edge_length_cm,
        orientation="y",
        boundary_type="transmission",
    )
    outer_hex = openmc.model.HexagonalPrism(
        edge_length=config.fuel_edge_length_cm,
        orientation="y",
        boundary_type="transmission",
    )

    outer_cylinders = []
    cells = []
    for index, (x, y) in enumerate(coolant_channel_positions(config), start=1):
        inner = openmc.ZCylinder(
            x0=x,
            y0=y,
            r=config.channel_radius_cm,
            name=f"coolant channel {index} inner",
        )
        outer = openmc.ZCylinder(
            x0=x,
            y0=y,
            r=config.coated_channel_radius_cm,
            name=f"coolant channel {index} coating outer",
        )
        outer_cylinders.append(outer)
        cells.extend((
            openmc.Cell(name=f"hydrogen channel {index}", fill=materials["hydrogen"], region=-inner),
            openmc.Cell(name=f"ZrC channel coating {index}", fill=materials["zrc"], region=+inner & -outer),
        ))

    fuel_region = -inner_hex
    for outer in outer_cylinders:
        fuel_region &= +outer

    cells.append(openmc.Cell(name="fuel matrix", fill=materials["fuel"], region=fuel_region))
    cells.append(openmc.Cell(name="outer ZrC coating", fill=materials["zrc"], region=+inner_hex & -outer_hex))
    return openmc.Universe(name="NERVA-like 19-channel fuel element", cells=cells)


def build_core_geometry(
    config: NervaConfig,
    materials: dict[str, openmc.Material],
) -> openmc.Geometry:
    fuel_universe = build_fuel_element_universe(config, materials)

    filler_cell = openmc.Cell(name="outer lattice filler", fill=materials["reflector"])
    filler_universe = openmc.Universe(name="outer lattice filler universe", cells=[filler_cell])

    lattice = openmc.HexLattice(name="NERVA-like fuel lattice")
    lattice.center = (0.0, 0.0)
    lattice.pitch = (config.fuel_flat_to_flat_cm,)
    lattice.orientation = "y"

    rings = []
    for ring_index in range(config.core_rings - 1, 0, -1):
        rings.append([fuel_universe] * (6 * ring_index))
    rings.append([fuel_universe])
    lattice.universes = rings
    lattice.outer = filler_universe

    half_length = 0.5 * config.active_length_cm
    zmin = openmc.ZPlane(z0=-half_length, boundary_type="vacuum")
    zmax = openmc.ZPlane(z0=half_length, boundary_type="vacuum")
    core_cylinder = openmc.ZCylinder(r=config.core_radius_cm)
    outer_cylinder = openmc.ZCylinder(r=config.reflector_outer_radius_cm, boundary_type="vacuum")
    axial_region = +zmin & -zmax

    core_cell = openmc.Cell(
        name="finite NERVA-like core",
        fill=lattice,
        region=-core_cylinder & axial_region,
    )
    reflector_cell = openmc.Cell(
        name="beryllium radial reflector",
        fill=materials["reflector"],
        region=+core_cylinder & -outer_cylinder & axial_region,
    )

    root = openmc.Universe(cells=[core_cell, reflector_cell])
    return openmc.Geometry(root)


def geometry_summary(config: NervaConfig) -> dict[str, float | int]:
    n_elements = 1 + 3 * config.core_rings * (config.core_rings - 1)
    return {
        "fuel_elements": n_elements,
        "coolant_channels_per_element": 19,
        "total_coolant_channels": 19 * n_elements,
        "active_length_cm": config.active_length_cm,
        "core_radius_cm": config.core_radius_cm,
        "reflector_outer_radius_cm": config.reflector_outer_radius_cm,
    }
