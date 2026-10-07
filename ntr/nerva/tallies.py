"""Tallies for the NERVA-derived OpenMC model."""

from __future__ import annotations

import numpy as np
import openmc

from .config import NervaConfig


_TIE_SOLID_CELL_NAMES = {
    "tie-tube inner Inconel tube",
    "tie-tube ZrH moderator",
    "tie-tube outer Inconel tube",
    "tie-tube ZrC sleeve",
    "tie-tube graphite filler",
    "tie-tube outer ZrC coating",
}

_MATERIAL_ORDER = (
    "fuel",
    "hydrogen",
    "zrc",
    "reflector",
    "graphite",
    "zrh",
    "inconel",
    "b4c",
    "aluminum",
)

def _safe_slug(value: str) -> str:
    return (
        value.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _cells_named(
    geometry: openmc.Geometry,
    name: str,
) -> list[openmc.Cell]:
    return [
        cell
        for cell in geometry.get_all_cells().values()
        if cell.name == name
    ]


def _attach_heating_trigger(
    tally: openmc.Tally,
    config: NervaConfig,
) -> None:
    if config.tally_rel_err_trigger is None:
        return
    trigger = openmc.Trigger(
        "rel_err",
        config.tally_rel_err_trigger,
        ignore_zeros=True,
    )
    trigger.scores = ["heating-local"]
    tally.triggers = [trigger]


def _energy_edges(config: NervaConfig) -> np.ndarray:
    """Log-spaced neutron-energy bins from thermal to 20 MeV."""
    return np.logspace(
        -5.0,
        np.log10(2.0e7),
        config.diagnostic_energy_groups + 1,
    )


def _vacuum_surfaces(
    geometry: openmc.Geometry,
) -> list[openmc.Surface]:
    return [
        surface
        for surface in geometry.get_all_surfaces().values()
        if surface.boundary_type == "vacuum"
    ]


def build_tallies(
    config: NervaConfig,
    radial_extent_cm: float | None = None,
    fuel_material: openmc.Material | None = None,
    geometry: openmc.Geometry | None = None,
    materials: dict[str, openmc.Material] | None = None,
) -> openmc.Tallies:
    """Create engineering and diagnostics tallies.

    The original mesh/fuel/tie tally names are preserved for backward
    compatibility. Additional tallies expose reaction rates, material-wise
    energy deposition, spectra, axial profiles, and vacuum-boundary currents.
    """
    half_length = 0.5 * config.active_length_cm
    r = (
        config.reflector_outer_radius_cm
        if radial_extent_cm is None
        else radial_extent_cm
    )

    mesh = openmc.RegularMesh(name="nerva_core_mesh")
    mesh.dimension = (48, 48, 24)
    mesh.lower_left = (-r, -r, -half_length)
    mesh.upper_right = (r, r, half_length)

    mesh_filter = openmc.MeshFilter(mesh)

    neutronics = openmc.Tally(name="nerva_3d_neutronics")
    neutronics.filters = [mesh_filter]
    neutronics.scores = [
        "flux",
        "absorption",
        "fission",
        "nu-fission",
        "heating",
        "heating-local",
    ]

    _attach_heating_trigger(neutronics, config)
    tallies: list[openmc.Tally] = [neutronics]

    if fuel_material is None and materials is not None:
        fuel_material = materials.get("fuel")

    if fuel_material is not None:
        fuel_heating = openmc.Tally(name="nerva_3d_fuel_heating")
        fuel_heating.filters = [
            mesh_filter,
            openmc.MaterialFilter([fuel_material]),
        ]
        fuel_heating.scores = [
            "heating-local",
            "fission",
            "nu-fission",
        ]
        tallies.append(fuel_heating)

    if geometry is not None:
        tie_cells = [
            cell
            for cell in geometry.get_all_cells().values()
            if cell.name in _TIE_SOLID_CELL_NAMES
        ]
        if tie_cells:
            tie_heating = openmc.Tally(name="nerva_3d_tie_heating")
            tie_heating.filters = [
                mesh_filter,
                openmc.CellFilter(tie_cells),
            ]
            tie_heating.scores = [
                "heating-local",
                "absorption",
            ]
            tallies.append(tie_heating)


    # Repeated-cell instance diagnostics. These preserve the actual Monte Carlo
    # power distribution instead of averaging all repeated lattice placements.
    if geometry is not None and config.element_instance_tallies:
        fuel_cells = _cells_named(geometry, "fuel matrix")
        if fuel_cells:
            fuel_instance = openmc.Tally(
                name="nerva_fuel_element_instances"
            )
            fuel_instance.filters = [
                openmc.DistribcellFilter(fuel_cells[0])
            ]
            fuel_instance.scores = [
                "flux",
                "fission",
                "nu-fission",
                "heating-local",
            ]
            tallies.append(fuel_instance)

        for tie_name in sorted(_TIE_SOLID_CELL_NAMES):
            cells = _cells_named(geometry, tie_name)
            if not cells:
                continue
            tie_instance = openmc.Tally(
                name=(
                    "nerva_tie_instance_"
                    + _safe_slug(tie_name)
                )
            )
            tie_instance.filters = [
                openmc.DistribcellFilter(cells[0])
            ]
            tie_instance.scores = [
                "absorption",
                "heating-local",
            ]
            tallies.append(tie_instance)

    if geometry is not None and config.channel_instance_tallies:
        for channel_index in range(1, 20):
            channel_name = f"hydrogen channel {channel_index}"
            channel_cells = _cells_named(geometry, channel_name)
            if not channel_cells:
                continue
            channel_tally = openmc.Tally(
                name=(
                    f"nerva_hydrogen_channel_"
                    f"{channel_index:02d}_instances"
                )
            )
            channel_tally.filters = [
                openmc.DistribcellFilter(channel_cells[0])
            ]
            channel_tally.scores = [
                "flux",
                "absorption",
                "heating-local",
            ]
            tallies.append(channel_tally)

    # Material-resolved transport and energy deposition.
    if materials is not None:
        material_list = [
            materials[key]
            for key in _MATERIAL_ORDER
            if key in materials
        ]
        if material_list:
            material_transport = openmc.Tally(
                name="nerva_material_transport"
            )
            material_transport.filters = [
                openmc.MaterialFilter(material_list)
            ]
            material_transport.scores = [
                "flux",
                "absorption",
                "fission",
                "nu-fission",
                "heating",
                "heating-local",
            ]
            tallies.append(material_transport)

        energy_filter = openmc.EnergyFilter(_energy_edges(config))

        fuel = materials.get("fuel")
        if fuel is not None:
            fuel_spectrum = openmc.Tally(name="nerva_fuel_spectrum")
            fuel_spectrum.filters = [
                openmc.MaterialFilter([fuel]),
                energy_filter,
            ]
            fuel_spectrum.scores = [
                "flux",
                "absorption",
                "fission",
                "nu-fission",
            ]
            tallies.append(fuel_spectrum)

        hydrogen = materials.get("hydrogen")
        if hydrogen is not None:
            hydrogen_spectrum = openmc.Tally(
                name="nerva_hydrogen_spectrum"
            )
            hydrogen_spectrum.filters = [
                openmc.MaterialFilter([hydrogen]),
                energy_filter,
            ]
            hydrogen_spectrum.scores = [
                "flux",
                "absorption",
                "heating-local",
            ]
            tallies.append(hydrogen_spectrum)

    # Axial profile with much finer z resolution than the 3-D mesh.
    axial_mesh = openmc.RegularMesh(name="nerva_axial_mesh")
    axial_mesh.dimension = (1, 1, config.axial_mesh_bins)
    axial_mesh.lower_left = (-r, -r, -half_length)
    axial_mesh.upper_right = (r, r, half_length)

    axial = openmc.Tally(name="nerva_axial_transport")
    axial.filters = [openmc.MeshFilter(axial_mesh)]
    axial.scores = [
        "flux",
        "fission",
        "nu-fission",
        "heating-local",
    ]
    tallies.append(axial)

    # Current across all vacuum boundaries provides a leakage diagnostic.
    if geometry is not None:
        vacuum_surfaces = _vacuum_surfaces(geometry)
        if vacuum_surfaces:
            leakage = openmc.Tally(name="nerva_vacuum_boundary_current")
            leakage.filters = [
                openmc.SurfaceFilter(vacuum_surfaces)
            ]
            leakage.scores = ["current"]
            tallies.append(leakage)

    return openmc.Tallies(tallies)
