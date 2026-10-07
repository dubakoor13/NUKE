"""Tallies for the NERVA-derived OpenMC model."""

from __future__ import annotations

import openmc

from .config import NervaConfig


def build_tallies(
    config: NervaConfig,
    radial_extent_cm: float | None = None,
) -> openmc.Tallies:
    """Create 3-D flux, fission-rate, and local-heating mesh tallies."""
    half_length = 0.5 * config.active_length_cm
    r = config.reflector_outer_radius_cm if radial_extent_cm is None else radial_extent_cm

    mesh = openmc.RegularMesh(name="nerva_core_mesh")
    mesh.dimension = (48, 48, 24)
    mesh.lower_left = (-r, -r, -half_length)
    mesh.upper_right = (r, r, half_length)

    tally = openmc.Tally(name="nerva_3d_neutronics")
    tally.filters = [openmc.MeshFilter(mesh)]
    tally.scores = ["flux", "fission", "heating-local"]

    return openmc.Tallies([tally])
