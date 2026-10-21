"""Physics validation tests: NASA reactor-variant separation and bound scattering.

The "pass" here means *correctly rejects* incomplete physical models/data.
No artificial neutron transport or surrogate k-effective is certified.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

import openmc

from .config import NervaConfig
from .materials import build_materials, required_bound_scattering_tables
from .nuclear_data import inspect_nuclear_data
from .audit_bound_thermal_scattering import audit_bound_thermal_scattering
from .physical_core_reference_validation import assess_core_fidelity


def reject(action, word):
    try:
        action()
    except (ValueError, RuntimeError) as error:
        assert word in str(error), str(error)
    else:
        raise AssertionError("invalid neutron physics accepted: "+word)


def main():
    cold = NervaConfig(core_rings=3)
    free = build_materials(cold)
    for key in ("graphite", "reflector", "zrh"):
        assert not free[key]._sab, (key, free[key]._sab)
    assert required_bound_scattering_tables(cold) == ()

    physics = replace(
        cold, thermal_scattering_treatment="evaluated-bound",
        bound_solid_temperature_k=293.605913,
    )
    materials = build_materials(physics)
    expected = {
        "graphite": {"c_Graphite"},
        "reflector": {"c_Be"},
        "zrh": {"c_H_in_ZrH", "c_Zr_in_ZrH"},
    }
    for material_key, wanted in expected.items():
        material = materials[material_key]
        actual = {name for name, fraction in material._sab}
        assert actual == wanted, (material_key, actual)
        assert material.temperature == physics.bound_solid_temperature_k

    # The graphite binder fractions in UC/ZrC/C composite fuel are unknown.
    # Do not silently assign graphite bound scattering to *all* fuel C atoms.
    assert not materials["fuel"]._sab
    assert not materials["hydrogen"]._sab
    assert len(required_bound_scattering_tables(physics)) == 4
    # Correct OpenMC v0.15 nearest-temperature physics is explicitly
    # auditable for both the free-gas control and the bound-scattering run.
    paired_free=replace(
        cold, bound_solid_temperature_k=295.99644373136465,
        bound_nearest_temperature_tolerance_k=3.0,
    )
    paired_bound=replace(
        physics, bound_solid_temperature_k=295.99644373136465,
        bound_nearest_temperature_tolerance_k=3.0,
    )
    free_solids=build_materials(paired_free)
    bound_solids=build_materials(paired_bound)
    for key in ("reflector","graphite","zrh"):
        assert free_solids[key].temperature == bound_solids[key].temperature
        assert not free_solids[key]._sab
        assert bound_solids[key]._sab
    from .model import build_model
    free_model=build_model(paired_free,assembly="core")
    bound_model=build_model(paired_bound,assembly="core")
    assert free_model.settings.temperature == bound_model.settings.temperature
    assert free_model.settings.temperature["method"] == "nearest"
    assert free_model.settings.temperature["tolerance"] == 3.
    for bad in (float("nan"),-1.,5.01):
        reject(
            lambda v=bad: replace(
                physics,bound_nearest_temperature_tolerance_k=v
            ).validate(),"tolerance",
        )


    reject(
        lambda: replace(cold, thermal_scattering_treatment="invented").validate(),
        "thermal_scattering_treatment",
    )
    reject(
        lambda: replace(cold, thermal_scattering_treatment="evaluated-bound").validate(),
        "bound_solid_temperature_k",
    )
    reject(
        lambda: replace(cold, bound_solid_temperature_k=-1).validate(),
        "solid temperature",
    )

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "U235.h5").write_bytes(b"synthetic-invalid-neutron")
        (root / "c_Be.h5").write_bytes(b"synthetic-invalid-sab")
        xml = root / "cross_sections.xml"
        xml.write_text(
            '<cross_sections>'
            '<library type="neutron" materials="U235" path="U235.h5"/>'
            '<library type="thermal" materials="c_Be" path="c_Be.h5"/>'
            '</cross_sections>'
        )
        audit = audit_bound_thermal_scattering(xml, physics)
        assert not audit["ready"]
        assert len(audit["tables"]) == 4
        assert not audit["physically_complete_bound_thermal_scattering"]
        assert any("absent" in (r["error"] or "") for r in audit["tables"])
        assert audit_bound_thermal_scattering(xml, cold)["treatment"] == "free-gas"
        report = inspect_nuclear_data(xml, config=physics)
        assert not report.ready
        assert "c_Zr_in_ZrH" in report.missing_thermal_scattering_tables

    ref = assess_core_fidelity(cold, assembly="core")
    assert ref["heritage_element_geometry_matches"]
    assert not ref["proposed_snre_bulk_geometry_matches"]
    assert ref["proposed_snre_core_reference"]["snre_fuel_element_count"]["model"] == 19
    assert ref["proposed_snre_core_reference"]["snre_tie_tube_count"]["model"] == 0
    assert ref["overall_historical_core_validation"] == "NOT_VALIDATED"
    assert ref["reactor_criticality_validated_against_nasa_measurements"] is False

    mixed = assess_core_fidelity(cold, assembly="mixed-core")
    assert 0 < mixed["proposed_snre_core_reference"]["snre_tie_tube_count"]["model"] < 241
    assert not mixed["proposed_snre_bulk_geometry_matches"]
    assert not assess_core_fidelity(
        replace(cold, fuel_flat_to_flat_cm=2.2)
    )["heritage_element_geometry_matches"]

    print("NERVA physical core/material-data validation: PASS")
    print("  physically historically NASA validated: NO")
    print("  real hot-neutronics neutron transport executed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
