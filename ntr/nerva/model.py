"""Assemble the NERVA-derived OpenMC model."""

from __future__ import annotations

import openmc

from .config import NervaConfig
from .geometry import build_core_geometry
from .materials import build_materials
from .tallies import build_tallies


def build_model(config: NervaConfig | None = None) -> openmc.Model:
    if config is None:
        config = NervaConfig()
    config.validate()

    material_map = build_materials(config)
    materials = openmc.Materials(list(material_map.values()))
    geometry = build_core_geometry(config, material_map)

    settings = openmc.Settings()
    settings.run_mode = "eigenvalue"
    settings.batches = config.batches
    settings.inactive = config.inactive
    settings.particles = config.particles

    half_length = 0.5 * config.active_length_cm
    source_radius = max(0.5, 0.8 * config.core_radius_cm)
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(
            (-source_radius, -source_radius, -0.8 * half_length),
            (source_radius, source_radius, 0.8 * half_length),
            only_fissionable=True,
        )
    )

    tallies = build_tallies(config)

    plot = openmc.Plot(name="nerva_xy")
    plot.basis = "xy"
    plot.origin = (0.0, 0.0, 0.0)
    width = 2.05 * config.reflector_outer_radius_cm
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
