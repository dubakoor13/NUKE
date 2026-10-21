"""Validate the 19-sector fuel scoring partition."""

from __future__ import annotations

import numpy as np
import openmc

from .config import NervaConfig
from .geometry import build_fuel_element_universe
from .layout import coolant_channel_positions
from .materials import build_materials


def main() -> int:
    config = NervaConfig()
    materials = build_materials(config)
    universe = build_fuel_element_universe(
        config,
        materials,
    )

    sector_cells = sorted(
        [
            cell
            for cell in universe.cells.values()
            if cell.name.startswith(
                "fuel matrix channel sector "
            )
        ],
        key=lambda cell: cell.name,
    )
    coating = [
        cell for cell in universe.cells.values()
        if cell.name == "outer ZrC coating"
    ]
    assert len(coating) == 1
    # The HexLattice, not a second coincident hexagonal prism, provides
    # the outer fuel-element boundary. Verify that rounding-scale points
    # immediately around the nominal outer-prism face remain covered by
    # the ZrC cell: the previous double clip lost real neutrons here.
    edge = config.fuel_flat_to_flat_cm / 2.
    for eps in (-1e-8, 1e-8):
        for side in (-1, 1):
            point = (side * (edge + eps), 0., 0.)
            assert point in coating[0].region, (
                f"uncovered outer fuel-element lattice edge: {point}"
            )

    assert len(sector_cells) == 19
    assert all(
        cell.fill is materials["fuel"]
        for cell in sector_cells
    )

    # Reconstruct the pre-partition fuel region independently and verify
    # sampled fuel points are covered by exactly one sector.
    inner_hex = openmc.model.HexagonalPrism(
        edge_length=config.inner_fuel_edge_length_cm,
        orientation="y",
    )
    original_fuel_region = -inner_hex
    for x, y in coolant_channel_positions(config):
        outer = openmc.ZCylinder(
            x0=x,
            y0=y,
            r=config.coated_channel_radius_cm,
        )
        original_fuel_region &= +outer

    extent = config.inner_fuel_edge_length_cm
    samples = np.linspace(-extent, extent, 81)
    tested = 0
    sector_hits = np.zeros(19, dtype=int)

    for x in samples:
        for y in samples:
            point = (float(x), float(y), 0.0)
            if point not in original_fuel_region:
                continue

            owners = [
                i
                for i, cell in enumerate(sector_cells)
                if point in cell.region
            ]
            assert len(owners) == 1, (
                f"fuel point {point} belongs to "
                f"{len(owners)} sectors"
            )
            sector_hits[owners[0]] += 1
            tested += 1

    assert tested > 100
    assert np.all(sector_hits > 0)

    surfaces = {}
    for cell in sector_cells:
        surfaces.update(cell.region.get_surfaces())
    voronoi = [
        surface
        for surface in surfaces.values()
        if surface.name.startswith(
            "fuel channel Voronoi boundary "
        )
    ]
    assert len(voronoi) == 171
    assert all(
        surface.boundary_type == "transmission"
        for surface in voronoi
    )

    print("NERVA fuel-sector geometry validation: PASS")
    print(f"  sectors: {len(sector_cells)}")
    print(f"  shared Voronoi planes: {len(voronoi)}")
    print(f"  sampled fuel points: {tested}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
