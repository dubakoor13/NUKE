"""Conceptual OpenMC materials for the NERVA-derived model."""

from __future__ import annotations

import openmc

from .config import NervaConfig


def build_materials(config: NervaConfig) -> dict[str, openmc.Material]:
    """Create materials used by the demonstration model."""
    config.validate()

    fuel = openmc.Material(name="conceptual (U,Zr)C-graphite surrogate")
    fuel.set_density("g/cm3", config.fuel_matrix_density_g_cm3)
    fuel.add_element("C", 0.90, percent_type="ao")
    fuel.add_element("Zr", 0.08, percent_type="ao")
    fuel.add_element(
        "U",
        0.02,
        percent_type="ao",
        enrichment=config.uranium_enrichment_wt_percent,
    )

    hydrogen = openmc.Material(name="hydrogen coolant")
    hydrogen.set_density("g/cm3", config.hydrogen_density_g_cm3)
    hydrogen.add_element("H", 1.0, percent_type="ao")
    hydrogen.temperature = config.hydrogen_temperature_k

    zrc = openmc.Material(name="ZrC protective coating")
    zrc.set_density("g/cm3", config.zrc_density_g_cm3)
    zrc.add_element("Zr", 1.0, percent_type="ao")
    zrc.add_element("C", 1.0, percent_type="ao")

    reflector = openmc.Material(name="beryllium radial reflector")
    reflector.set_density("g/cm3", config.reflector_density_g_cm3)
    reflector.add_element("Be", 1.0, percent_type="ao")

    return {
        "fuel": fuel,
        "hydrogen": hydrogen,
        "zrc": zrc,
        "reflector": reflector,
    }
