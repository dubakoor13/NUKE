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
    *,
    eigenvalue_entropy_mesh: bool = True,
    initial_fission_source: str = "legacy-broad",
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
    settings.photon_transport = config.photon_transport
    # Bind audited nearest-library tolerance to the actual transport.
    # Never rely on the default 10 K tolerance for evaluated bound physics.
    if (config.thermal_scattering_treatment == "evaluated-bound"
            or config.bound_nearest_temperature_tolerance_k > 0):
        settings.temperature = {
            "method": "nearest",
            "tolerance": max(1.0e-6, config.bound_nearest_temperature_tolerance_k),
        }

    half_length = 0.5 * config.active_length_cm
    # Changing the *initial guess* for the fission source is a numerical
    # source-relaxation experiment, NOT a reactor geometry or enrichment change.
    if initial_fission_source not in ("legacy-broad", "central-compact"):
        raise ValueError("initial_fission_source must be legacy-broad or central-compact")
    extent_fraction = 0.8 if initial_fission_source == "legacy-broad" else 0.35
    source_radius = max(0.5, extent_fraction * source_region_radius)
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(
            (-source_radius, -source_radius, -extent_fraction * half_length),
            (source_radius, source_radius, extent_fraction * half_length),
        ),
        constraints={"fissionable": True},
    )

    # Shannon source-entropy mesh for eigenvalue source-convergence
    # diagnostics. This changes observability, not reactor design.
    if eigenvalue_entropy_mesh:
        entropy_mesh = openmc.RegularMesh(name="nerva_entropy_mesh")
        entropy_mesh.dimension = (
            config.entropy_mesh_xy,
            config.entropy_mesh_xy,
            config.entropy_mesh_z,
        )
        entropy_mesh.lower_left = (
            -source_region_radius,
            -source_region_radius,
            -half_length,
        )
        entropy_mesh.upper_right = (
            source_region_radius,
            source_region_radius,
            half_length,
        )
        settings.entropy_mesh = entropy_mesh

    if config.tally_rel_err_trigger is not None:
        settings.trigger_active = True
        settings.trigger_max_batches = config.trigger_max_batches
        settings.trigger_batch_interval = config.trigger_batch_interval

    tallies = build_tallies(
        config,
        radial_extent_cm=radial_extent,
        fuel_material=material_map["fuel"],
        geometry=geometry,
        materials=material_map,
    )

    # OpenMC 0.16 introduced SlicePlot; the official transport Docker image
    # still offers the v0.15 Plot class. The slice is for visualization only.
    # This compatibility branch never changes neutronics or source physics.
    plot_type = getattr(openmc, "SlicePlot", openmc.Plot)
    plot = plot_type(name=f"nerva_{assembly}_xy")
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
