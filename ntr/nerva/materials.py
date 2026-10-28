"""Conceptual OpenMC materials for the NERVA-derived model."""

from __future__ import annotations

import openmc

from .config import NervaConfig


# GNDS/OpenMC thermal-scattering identifiers, not interchangeable with
# free-atom scattering kernels. Do NOT apply graphite S(a,b) to every
# carbon atom of mixed UC/ZrC/graphite fuel: its phase fractions are not
# specified by the current conceptual fuel chemistry.
EVALUATED_BOUND_THERMAL_TABLES: dict[str, tuple[str, ...]] = {
    "reflector": ("c_Be",),
    "graphite": ("c_Graphite",),
    "zrh": ("c_H_in_ZrH", "c_Zr_in_ZrH"),
}


def required_bound_scattering_tables(config: NervaConfig) -> tuple[str, ...]:
    """Required S(a,b) kernels for the specified *surrogate* materials."""
    if config.thermal_scattering_treatment == "free-gas":
        return ()
    if config.thermal_scattering_treatment != "evaluated-bound":
        raise ValueError("invalid thermal-scattering model")
    return tuple(sorted({
        table
        for names in EVALUATED_BOUND_THERMAL_TABLES.values()
        for table in names
    }))


def build_materials(config: NervaConfig) -> dict[str, openmc.Material]:
    """Create materials used by the demonstration model."""
    config.validate()

    fuel = openmc.Material(name="conceptual (U,Zr)C-graphite surrogate")
    fuel.set_density("g/cm3", config.fuel_matrix_density_g_cm3)
    if config.fuel_temperature_k is not None:
        fuel.temperature = config.fuel_temperature_k
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

    graphite = openmc.Material(name="tie-tube graphite filler")
    graphite.set_density("g/cm3", config.graphite_density_g_cm3)
    graphite.add_element("C", 1.0, percent_type="ao")

    zrh = openmc.Material(name="conceptual zirconium-hydride moderator")
    zrh.set_density("g/cm3", config.zrh_density_g_cm3)
    zrh.add_element("Zr", 1.0, percent_type="ao")
    zrh.add_element("H", 1.7, percent_type="ao")

    inconel = openmc.Material(name="simplified Inconel tube surrogate")
    inconel.set_density("g/cm3", config.inconel_density_g_cm3)
    inconel.add_element("Ni", 0.70, percent_type="wo")
    inconel.add_element("Cr", 0.18, percent_type="wo")
    inconel.add_element("Fe", 0.12, percent_type="wo")

    b4c = openmc.Material(name="natural boron-carbide control absorber")
    b4c.set_density("g/cm3", config.b4c_density_g_cm3)
    b4c.add_element("B", 4.0, percent_type="ao")
    b4c.add_element("C", 1.0, percent_type="ao")

    aluminum = openmc.Material(name="aluminum-alloy pressure-vessel surrogate")
    aluminum.set_density("g/cm3", config.aluminum_density_g_cm3)
    aluminum.add_element("Al", 1.0, percent_type="ao")

    # Preserve composition/density/geometry. Apply the explicitly modeled
    # coating, metallic and absorber temperatures to the actual C++ physics.
    if config.structural_solid_temperature_k is not None:
        for structural in (zrc, inconel, b4c, aluminum):
            structural.temperature = config.structural_solid_temperature_k

    # Comparison control: same Be/graphite/ZrH solid temperatures in
    # both free-gas and bound-scattering models; only S(a,b) differs.
    if config.bound_solid_temperature_k is not None:
        for solid in (reflector, graphite, zrh):
            solid.temperature = config.bound_solid_temperature_k

    if config.thermal_scattering_treatment == "evaluated-bound":
        for key, tables in EVALUATED_BOUND_THERMAL_TABLES.items():
            target = {
                "reflector": reflector,
                "graphite": graphite,
                "zrh": zrh,
            }[key]
            for table in tables:
                target.add_s_alpha_beta(table)
        # Carbon bound within heterogeneous fuel phases is intentionally
        # untreated until graphite/UC/ZrC phase fractions are documented.

    return {
        "fuel": fuel,
        "hydrogen": hydrogen,
        "zrc": zrc,
        "reflector": reflector,
        "graphite": graphite,
        "zrh": zrh,
        "inconel": inconel,
        "b4c": b4c,
        "aluminum": aluminum,
    }
