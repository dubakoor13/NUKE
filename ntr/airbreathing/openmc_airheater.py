"""Detailed OpenMC fixed-source model for a generic nuclear air heater.

Safety/fidelity boundary
------------------------
This is deliberately a non-calibrated, non-optimized research demonstrator.
It uses a fixed arbitrary cylindrical test geometry, a low-enrichment surrogate
material, and OpenMC *fixed-source* transport. It does not calculate k-effective,
does not include control elements, and does not perform geometry/enrichment
optimization.

The purpose is to exercise OpenMC transport/tally capability and produce
volumetric nuclear-heating fields that can be coupled to the generic
airbreathing thermodynamic model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import openmc


@dataclass(frozen=True)
class GenericAirHeaterConfig:
    """Fixed didactic geometry and transport settings."""

    active_radius_cm: float = 12.0
    active_length_cm: float = 40.0
    air_duct_radius_cm: float = 3.0
    vessel_thickness_cm: float = 0.5

    fuel_temperature_k: float = 1200.0
    air_temperature_k: float = 700.0
    air_pressure_mpa: float = 1.0
    vessel_temperature_k: float = 700.0

    batches: int = 40
    particles: int = 5000
    photon_transport: bool = False

    mesh_dimension: tuple[int, int, int] = (32, 32, 32)
    spectrum_groups: int = 80

    def validate(self) -> None:
        if self.active_radius_cm <= 0.0:
            raise ValueError("active_radius_cm must be positive")
        if self.active_length_cm <= 0.0:
            raise ValueError("active_length_cm must be positive")
        if not 0.0 < self.air_duct_radius_cm < self.active_radius_cm:
            raise ValueError("air duct must lie inside active radius")
        if self.vessel_thickness_cm <= 0.0:
            raise ValueError("vessel thickness must be positive")
        if self.fuel_temperature_k <= 0.0:
            raise ValueError("fuel temperature must be positive")
        if self.air_temperature_k <= 0.0:
            raise ValueError("air temperature must be positive")
        if self.air_pressure_mpa <= 0.0:
            raise ValueError("air pressure must be positive")
        if self.batches <= 0 or self.particles <= 0:
            raise ValueError("batches and particles must be positive")
        if any(value <= 0 for value in self.mesh_dimension):
            raise ValueError("mesh dimensions must be positive")
        if self.spectrum_groups < 8:
            raise ValueError("spectrum_groups must be >= 8")

    @property
    def half_length_cm(self) -> float:
        return 0.5 * self.active_length_cm

    @property
    def vessel_outer_radius_cm(self) -> float:
        return self.active_radius_cm + self.vessel_thickness_cm


@dataclass(frozen=True)
class AirHeaterBuild:
    model: openmc.Model
    materials: dict[str, openmc.Material]
    cells: dict[str, openmc.Cell]
    surfaces: dict[str, openmc.Surface]
    config: GenericAirHeaterConfig


def _air_density_g_cm3(
    temperature_k: float,
    pressure_mpa: float,
) -> float:
    """Ideal dry-air density for OpenMC material bookkeeping."""
    pressure_pa = pressure_mpa * 1.0e6
    rho_kg_m3 = pressure_pa / (287.05 * temperature_k)
    return rho_kg_m3 / 1000.0


def build_materials(
    config: GenericAirHeaterConfig,
) -> dict[str, openmc.Material]:
    """Build deliberately generic LEU-surrogate, air, and vessel materials."""
    config.validate()

    fuel = openmc.Material(name="generic LEU carbide surrogate")
    fuel.temperature = config.fuel_temperature_k
    fuel.set_density("g/cm3", 4.0)
    # Deliberately low uranium loading and low enrichment. This is not
    # calibrated to any historical or contemporary propulsion reactor.
    fuel.add_element("U", 0.005, percent_type="ao", enrichment=5.0)
    fuel.add_element("Zr", 0.495, percent_type="ao")
    fuel.add_element("C", 0.500, percent_type="ao")

    air = openmc.Material(name="generic dry air working fluid")
    air.temperature = config.air_temperature_k
    air.set_density(
        "g/cm3",
        _air_density_g_cm3(
            config.air_temperature_k,
            config.air_pressure_mpa,
        ),
    )
    air.add_element("N", 0.79, percent_type="ao")
    air.add_element("O", 0.21, percent_type="ao")

    vessel = openmc.Material(name="generic Fe-Cr-Ni vessel surrogate")
    vessel.temperature = config.vessel_temperature_k
    vessel.set_density("g/cm3", 7.9)
    vessel.add_element("Fe", 0.70, percent_type="wo")
    vessel.add_element("Cr", 0.20, percent_type="wo")
    vessel.add_element("Ni", 0.10, percent_type="wo")

    return {
        "fuel": fuel,
        "air": air,
        "vessel": vessel,
    }


def build_geometry(
    config: GenericAirHeaterConfig,
    materials: dict[str, openmc.Material],
) -> tuple[
    openmc.Geometry,
    dict[str, openmc.Cell],
    dict[str, openmc.Surface],
]:
    """Build a fixed cylindrical irradiation/heating test geometry."""
    config.validate()

    air_radius = openmc.ZCylinder(
        r=config.air_duct_radius_cm,
        name="generic air duct boundary",
    )
    active_outer = openmc.ZCylinder(
        r=config.active_radius_cm,
        name="generic active-material outer boundary",
    )
    vessel_outer = openmc.ZCylinder(
        r=config.vessel_outer_radius_cm,
        boundary_type="vacuum",
        name="generic vessel outer vacuum boundary",
    )
    z_min = openmc.ZPlane(
        z0=-config.half_length_cm,
        boundary_type="vacuum",
        name="generic lower vacuum boundary",
    )
    z_max = openmc.ZPlane(
        z0=config.half_length_cm,
        boundary_type="vacuum",
        name="generic upper vacuum boundary",
    )

    axial = +z_min & -z_max

    air_cell = openmc.Cell(
        name="generic air duct",
        fill=materials["air"],
        region=-air_radius & axial,
    )
    fuel_cell = openmc.Cell(
        name="generic LEU surrogate annulus",
        fill=materials["fuel"],
        region=+air_radius & -active_outer & axial,
    )
    vessel_cell = openmc.Cell(
        name="generic vessel shell",
        fill=materials["vessel"],
        region=+active_outer & -vessel_outer & axial,
    )

    root = openmc.Universe(
        name="generic nuclear air-heater root",
        cells=[air_cell, fuel_cell, vessel_cell],
    )

    return (
        openmc.Geometry(root),
        {
            "air": air_cell,
            "fuel": fuel_cell,
            "vessel": vessel_cell,
        },
        {
            "air_radius": air_radius,
            "active_outer": active_outer,
            "vessel_outer": vessel_outer,
            "z_min": z_min,
            "z_max": z_max,
        },
    )


def _energy_edges(config: GenericAirHeaterConfig) -> np.ndarray:
    return np.logspace(-5.0, 7.30103, config.spectrum_groups + 1)


def build_tallies(
    config: GenericAirHeaterConfig,
    cells: dict[str, openmc.Cell],
    surfaces: dict[str, openmc.Surface],
) -> openmc.Tallies:
    """Build spatial, regional, spectral, and leakage diagnostics."""
    config.validate()

    r = config.vessel_outer_radius_cm
    half = config.half_length_cm

    mesh = openmc.RegularMesh(name="generic_airheater_3d_mesh")
    mesh.dimension = config.mesh_dimension
    mesh.lower_left = (-r, -r, -half)
    mesh.upper_right = (r, r, half)

    mesh_tally = openmc.Tally(name="airheater_3d_transport")
    mesh_tally.filters = [openmc.MeshFilter(mesh)]
    mesh_tally.scores = [
        "flux",
        "heating",
        "heating-local",
        "absorption",
        "fission",
        "nu-fission",
    ]

    region_tally = openmc.Tally(name="airheater_region_transport")
    region_tally.filters = [
        openmc.CellFilter(
            [cells["fuel"], cells["air"], cells["vessel"]]
        )
    ]
    region_tally.scores = [
        "flux",
        "heating",
        "heating-local",
        "absorption",
        "fission",
        "nu-fission",
    ]

    energy_filter = openmc.EnergyFilter(_energy_edges(config))

    fuel_spectrum = openmc.Tally(name="airheater_fuel_spectrum")
    fuel_spectrum.filters = [
        openmc.CellFilter([cells["fuel"]]),
        energy_filter,
    ]
    fuel_spectrum.scores = [
        "flux",
        "absorption",
        "fission",
        "nu-fission",
    ]

    air_spectrum = openmc.Tally(name="airheater_air_spectrum")
    air_spectrum.filters = [
        openmc.CellFilter([cells["air"]]),
        energy_filter,
    ]
    air_spectrum.scores = [
        "flux",
        "absorption",
        "heating-local",
    ]

    leakage = openmc.Tally(name="airheater_boundary_current")
    leakage.filters = [
        openmc.SurfaceFilter(
            [
                surfaces["vessel_outer"],
                surfaces["z_min"],
                surfaces["z_max"],
            ]
        )
    ]
    leakage.scores = ["current"]

    return openmc.Tallies(
        [
            mesh_tally,
            region_tally,
            fuel_spectrum,
            air_spectrum,
            leakage,
        ]
    )


def build_openmc_airheater(
    config: GenericAirHeaterConfig | None = None,
) -> AirHeaterBuild:
    """Construct the generic OpenMC fixed-source air-heater model."""
    if config is None:
        config = GenericAirHeaterConfig()
    config.validate()

    material_map = build_materials(config)
    geometry, cells, surfaces = build_geometry(
        config,
        material_map,
    )

    settings = openmc.Settings()
    settings.run_mode = "fixed source"
    settings.batches = config.batches
    settings.particles = config.particles
    settings.photon_transport = config.photon_transport

    source_radius = 0.85 * config.active_radius_cm
    settings.source = openmc.IndependentSource(
        space=openmc.stats.Box(
            (-source_radius, -source_radius, -0.8 * config.half_length_cm),
            (source_radius, source_radius, 0.8 * config.half_length_cm),
        ),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.Watt(
            a=0.988e6,
            b=2.249e-6,
        ),
        constraints={"fissionable": True},
    )

    tallies = build_tallies(config, cells, surfaces)

    plot = openmc.SlicePlot(name="generic_airheater_xy")
    plot.basis = "xy"
    plot.origin = (0.0, 0.0, 0.0)
    width = 2.05 * config.vessel_outer_radius_cm
    plot.width = (width, width)
    plot.pixels = (900, 900)
    plot.color_by = "material"

    model = openmc.Model(
        geometry=geometry,
        materials=openmc.Materials(list(material_map.values())),
        settings=settings,
        tallies=tallies,
        plots=openmc.Plots([plot]),
    )

    return AirHeaterBuild(
        model=model,
        materials=material_map,
        cells=cells,
        surfaces=surfaces,
        config=config,
    )
