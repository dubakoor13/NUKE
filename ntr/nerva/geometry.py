"""OpenMC geometry construction for the NERVA-derived model."""

from __future__ import annotations

import openmc

from .config import NervaConfig
from .layout import coolant_channel_positions


def build_fuel_element_universe(
    config: NervaConfig,
    materials: dict[str, openmc.Material],
) -> openmc.Universe:
    """Build one hexagonal fuel element with 19 coated coolant channels."""
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
            openmc.Cell(
                name=f"hydrogen channel {index}",
                fill=materials["hydrogen"],
                region=-inner,
            ),
            openmc.Cell(
                name=f"ZrC channel coating {index}",
                fill=materials["zrc"],
                region=+inner & -outer,
            ),
        ))

    fuel_region = -inner_hex
    for outer in outer_cylinders:
        fuel_region &= +outer

    cells.append(
        openmc.Cell(
            name="fuel matrix",
            fill=materials["fuel"],
            region=fuel_region,
        )
    )
    cells.append(
        openmc.Cell(
            name="outer ZrC coating",
            fill=materials["zrc"],
            region=+inner_hex & -outer_hex,
        )
    )
    return openmc.Universe(
        name="NERVA-like 19-channel fuel element",
        cells=cells,
    )


def build_tie_tube_universe(
    config: NervaConfig,
    materials: dict[str, openmc.Material],
) -> openmc.Universe:
    """Build a NERVA-derived coaxial tie/support element.

    The radial stack follows the public NASA tie-tube cross section. Small
    manufacturing/cold-condition gaps are represented as void regions.
    """
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

    supply = openmc.ZCylinder(r=config.tie_inner_tube_inner_radius_cm)
    inner_tube_outer = openmc.ZCylinder(r=config.tie_inner_tube_outer_radius_cm)
    zrh_inner = openmc.ZCylinder(r=config.tie_zrh_inner_radius_cm)
    zrh_outer = openmc.ZCylinder(r=config.tie_zrh_outer_radius_cm)
    outer_tube_inner = openmc.ZCylinder(r=config.tie_outer_tube_inner_radius_cm)
    outer_tube_outer = openmc.ZCylinder(r=config.tie_outer_tube_outer_radius_cm)
    zrc_inner = openmc.ZCylinder(r=config.tie_zrc_inner_radius_cm)
    zrc_outer = openmc.ZCylinder(r=config.tie_zrc_outer_radius_cm)
    graphite_inner = openmc.ZCylinder(r=config.tie_graphite_inner_radius_cm)

    cells = [
        openmc.Cell(
            name="tie-tube hydrogen supply",
            fill=materials["hydrogen"],
            region=-supply,
        ),
        openmc.Cell(
            name="tie-tube inner Inconel tube",
            fill=materials["inconel"],
            region=+supply & -inner_tube_outer,
        ),
        openmc.Cell(
            name="tie-tube inner cold gap",
            region=+inner_tube_outer & -zrh_inner,
        ),
        openmc.Cell(
            name="tie-tube ZrH moderator",
            fill=materials["zrh"],
            region=+zrh_inner & -zrh_outer,
        ),
        openmc.Cell(
            name="tie-tube hydrogen return",
            fill=materials["hydrogen"],
            region=+zrh_outer & -outer_tube_inner,
        ),
        openmc.Cell(
            name="tie-tube outer Inconel tube",
            fill=materials["inconel"],
            region=+outer_tube_inner & -outer_tube_outer,
        ),
        openmc.Cell(
            name="tie-tube outer cold gap",
            region=+outer_tube_outer & -zrc_inner,
        ),
        openmc.Cell(
            name="tie-tube ZrC sleeve",
            fill=materials["zrc"],
            region=+zrc_inner & -zrc_outer,
        ),
        openmc.Cell(
            name="tie-tube liner-to-graphite gap",
            region=+zrc_outer & -graphite_inner,
        ),
        openmc.Cell(
            name="tie-tube graphite filler",
            fill=materials["graphite"],
            region=+graphite_inner & -inner_hex,
        ),
        openmc.Cell(
            name="tie-tube outer ZrC coating",
            fill=materials["zrc"],
            region=+inner_hex & -outer_hex,
        ),
    ]

    return openmc.Universe(
        name="NERVA-like coaxial tie/support element",
        cells=cells,
    )


def _finite_lattice_geometry(
    config: NervaConfig,
    lattice: openmc.HexLattice,
    core_radius_cm: float,
    outer_radius_cm: float,
    materials: dict[str, openmc.Material],
    name: str,
) -> openmc.Geometry:
    half_length = 0.5 * config.active_length_cm
    zmin = openmc.ZPlane(z0=-half_length, boundary_type="vacuum")
    zmax = openmc.ZPlane(z0=half_length, boundary_type="vacuum")
    core_cylinder = openmc.ZCylinder(r=core_radius_cm)
    outer_cylinder = openmc.ZCylinder(
        r=outer_radius_cm,
        boundary_type="vacuum",
    )
    axial_region = +zmin & -zmax

    lattice_cell = openmc.Cell(
        name=name,
        fill=lattice,
        region=-core_cylinder & axial_region,
    )
    reflector_cell = openmc.Cell(
        name="beryllium radial reflector",
        fill=materials["reflector"],
        region=+core_cylinder & -outer_cylinder & axial_region,
    )

    root = openmc.Universe(cells=[lattice_cell, reflector_cell])
    return openmc.Geometry(root)


def build_cluster_geometry(
    config: NervaConfig,
    materials: dict[str, openmc.Material],
) -> openmc.Geometry:
    """Build the seven-position cluster: six fuel elements around one tie tube."""
    fuel_universe = build_fuel_element_universe(config, materials)
    tie_universe = build_tie_tube_universe(config, materials)

    filler_universe = openmc.Universe(
        name="cluster outer filler universe",
        cells=[
            openmc.Cell(
                name="cluster outer filler",
                fill=materials["reflector"],
            )
        ],
    )

    lattice = openmc.HexLattice(name="six-fuel one-tie cluster")
    lattice.center = (0.0, 0.0)
    lattice.pitch = (config.fuel_flat_to_flat_cm,)
    lattice.orientation = "y"
    lattice.universes = [
        [fuel_universe] * 6,
        [tie_universe],
    ]
    lattice.outer = filler_universe

    return _finite_lattice_geometry(
        config=config,
        lattice=lattice,
        core_radius_cm=config.cluster_radius_cm,
        outer_radius_cm=config.cluster_reflector_outer_radius_cm,
        materials=materials,
        name="finite six-fuel one-tie cluster",
    )


def build_core_geometry(
    config: NervaConfig,
    materials: dict[str, openmc.Material],
) -> openmc.Geometry:
    """Build the original small all-fuel demonstrator surrounded by Be."""
    fuel_universe = build_fuel_element_universe(config, materials)

    filler_cell = openmc.Cell(
        name="outer lattice filler",
        fill=materials["reflector"],
    )
    filler_universe = openmc.Universe(
        name="outer lattice filler universe",
        cells=[filler_cell],
    )

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

    return _finite_lattice_geometry(
        config=config,
        lattice=lattice,
        core_radius_cm=config.core_radius_cm,
        outer_radius_cm=config.reflector_outer_radius_cm,
        materials=materials,
        name="finite NERVA-like all-fuel core",
    )


def cluster_summary(config: NervaConfig) -> dict[str, float | int]:
    return {
        "fuel_elements": 6,
        "tie_tubes": 1,
        "fuel_coolant_channels": 6 * 19,
        "tie_tube_hydrogen_passages": 2,
        "active_length_cm": config.active_length_cm,
        "cluster_radius_cm": config.cluster_radius_cm,
        "reflector_outer_radius_cm": config.cluster_reflector_outer_radius_cm,
    }


def geometry_summary(config: NervaConfig) -> dict[str, float | int]:
    n_elements = 1 + 3 * config.core_rings * (config.core_rings - 1)
    return {
        "fuel_elements": n_elements,
        "tie_tubes": 0,
        "coolant_channels_per_element": 19,
        "total_coolant_channels": 19 * n_elements,
        "active_length_cm": config.active_length_cm,
        "core_radius_cm": config.core_radius_cm,
        "reflector_outer_radius_cm": config.reflector_outer_radius_cm,
    }
