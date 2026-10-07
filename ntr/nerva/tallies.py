"""Tallies for the NERVA-derived OpenMC model."""

from __future__ import annotations

import openmc

from .config import NervaConfig


def build_tallies(
    config: NervaConfig,
    radial_extent_cm: float | None = None,
    fuel_material: openmc.Material | None = None,
) -> openmc.Tallies:
    """Create whole-reactor and fuel-only 3-D neutronics tallies."""
    half_length = 0.5 * config.active_length_cm
    r = config.reflector_outer_radius_cm if radial_extent_cm is None else radial_extent_cm

    mesh = openmc.RegularMesh(name="nerva_core_mesh")
    mesh.dimension = (48, 48, 24)
    mesh.lower_left = (-r, -r, -half_length)
    mesh.upper_right = (r, r, half_length)

    mesh_filter = openmc.MeshFilter(mesh)

    neutronics = openmc.Tally(name="nerva_3d_neutronics")
    neutronics.filters = [mesh_filter]
    neutronics.scores = ["flux", "fission", "heating-local"]

    tallies = [neutronics]

    if fuel_material is not None:
        fuel_heating = openmc.Tally(name="nerva_3d_fuel_heating")
        fuel_heating.filters = [
            mesh_filter,
            openmc.MaterialFilter([fuel_material]),
        ]
        fuel_heating.scores = ["heating-local"]
        tallies.append(fuel_heating)

    return openmc.Tallies(tallies)
