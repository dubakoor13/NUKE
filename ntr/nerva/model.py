"""Assemble the NERVA-derived OpenMC model."""

from __future__ import annotations

import openmc

from .config import NervaConfig
from .geometry import build_cluster_geometry, build_core_geometry, build_mixed_core_geometry
from .materials import build_materials
from .periphery import build_reactor_geometry
from .tallies import build_tallies


def build_model(
    config: NervaConfig | None = None,
    assembly: str = "core",
) -> openmc.Model:
    """Construct a NERVA-derived core, cluster, mixed core, or full reactor assembly."""
    if config is None:
        config = NervaConfig()
    config.validate()

    material_map = build_materials(config)
    materials = openmc.Materials(list(material_map.values()))

    if assembly == "core":
        geometry = build_core_geometry(config, material_map)
        source_region_radius = config.core_radius_cm
        radial_extent = config.reflector_outer_radius_cm
    elif assembly == "cluster":
        geometry = build_cluster_geometry(config, material_map)
        source_region_radius = config.cluster_radius_cm
        radial_extent = config.cluster_reflector_outer_radius_cm
    elif assembly == "mixed-core":
        geometry = build_mixed_core_geometry(config, material_map)
        source_region_radius = config.core_radius_cm
        radial_extent = config.reflector_outer_radius_cm
    elif assembly == "reactor":
        geometry = build_reactor_geometry(config, material_map)
        source_region_radius = config.core_radius_cm
        radial_extent = config.pressure_vessel_outer_radius_cm
    else:
        raise ValueError("assembly must be 'core', 'cluster', 'mixed-core', or 'reactor'")

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.batches = config.batches
    settings.inactive = config.inactive
    settings.particles = config.particles

    half_length = 0.5 * config.active_length_cm
    source_radius = max(0.5, 0.8 * source_region_radius)
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(
            (-source_radius, -source_radius, -0.8 * half_length),
            (source_radius, source_radius, 0.8 * half_length),
        ),
        constraints={"fissionable": True},
    )

    tallies = build_tallies(config, radial_extent_cm=radial_extent)

    plot = openmc.SlicePlot(name=f"nerva_{assembly}_xy")
    plot.basis = "xy"
    plot.origin = (0.0, 0.0, 0.0)
    width = 2.05 * radial_extent
    plot.width = (width, width)
    plot.pixels = (1000, 1000)
    plot.color_by = "material"

    return openmc.Model(
        geometry=geometry,
        materials=materials,
        settings=settings,
        tallies=tallies,
        plots=openmc.Plots([plot]),
    )
