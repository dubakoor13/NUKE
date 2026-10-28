"""Fail-closed reactor-core hot neutron nuclear-data qualification.

This is a MATERIAL/XS-physics preflight for the actual uniform-material
OpenMC transport inputs. It does not change reactor geometry, fuel loading,
densities, controls, cross sections, or make unvalidated hot reactor power.

A reported supported temperature requires REAL evaluated neutron HDF5
IncidentNeutron.kTs evidence from audit_temperature_coverage (not a synthetic
temperature-dependent formula). An implicit material temperature is not a
physically specified core thermal state. A sole cold table does not support a
hot material condition. Bound neutron scattering and thermal-hydraulic
feedback remain separate physical qualification requirements.
"""
from __future__ import annotations

import math

from .config import NervaConfig

REQUIRED_DECLARED_MATERIALS = (
    "fuel", "hydrogen", "zrc", "reflector", "graphite", "zrh",
    "inconel", "b4c", "aluminum",
)

# This is a conservative preflight designation of a high-T TEST, not an
# empirical operating threshold for a historical NERVA core.
HOT_TEST_FUEL_TEMPERATURE_K = 600.0


def qualify_explicit_hot_neutron_data(config: NervaConfig,
                                      audit: dict) -> dict:
    """Qualify actual requested material kT support, NOT reactor physics.

    Called *after* audit_temperature_coverage loads real evaluated HDF5 and
    verifies isotope identity. Uses those HDF5 kT tables, not guessed grids.
    """
    config.validate()
    if not isinstance(audit,dict) or not isinstance(audit.get("nuclides"),list):
        raise ValueError("real per-nuclide evaluated neutron kT audit is mandatory")
    if audit.get("neutron_temperature_method") != config.neutron_temperature_method:
        raise ValueError("audited and executed neutron temperature operators disagree")

    requested = {
        "fuel":config.fuel_temperature_k,
        "hydrogen":config.hydrogen_temperature_k,
        "zrc":config.structural_solid_temperature_k,
        "reflector":config.bound_solid_temperature_k,
        "graphite":config.bound_solid_temperature_k,
        "zrh":config.bound_solid_temperature_k,
        "inconel":config.structural_solid_temperature_k,
        "b4c":config.structural_solid_temperature_k,
        "aluminum":config.structural_solid_temperature_k,
    }

    missing_explicit = [
        key for key in REQUIRED_DECLARED_MATERIALS
        if requested[key] is None
    ]
    fuel_hot_test = bool(
        config.fuel_temperature_k is not None
        and config.fuel_temperature_k >= HOT_TEST_FUEL_TEMPERATURE_K
    )
    isotope_rows=[]
    seen=set()
    for row in audit["nuclides"]:
        name=row.get("nuclide")
        if not isinstance(name,str) or not name or name in seen:
            raise ValueError("duplicate or invalid real evaluated isotope evidence")
        seen.add(name)
        temperatures=row.get("available_temperatures_K",[])
        actual=row.get("requested_by_material",[])
        if not isinstance(actual,list) or not actual:
            raise ValueError("isotope audit has no real material references")
        grid_valid=(
            isinstance(temperatures,list)
            and len(temperatures)>=2
            and all(isinstance(t,(float,int)) and math.isfinite(t) and t>0
                    for t in temperatures)
            and all(b>a for a,b in zip(temperatures,temperatures[1:]))
        )
        failures=[]
        if row.get("load_error") is not None:
            failures.append("missing or unloadable evaluated neutron isotope HDF5")
        if not grid_valid:
            failures.append("fewer than two distinct evaluated neutron kT tables")
        for material in actual:
            key=material.get("material_key")
            if key not in requested:
                raise ValueError("evaluated nuclear temperature audit contains unknown material")
            wanted=requested[key]
            measured=material.get("requested_temperature_K")
            if wanted is None:
                failures.append(f"{key}: explicit material temperature missing")
            elif (not isinstance(measured,(int,float))
                  or not math.isclose(float(measured),float(wanted),
                                      rel_tol=0,abs_tol=1e-8)):
                raise ValueError("audited isotope/material temperature does not match physical config")
            selection=material.get("actual_transport_temperature_selection")
            if (not isinstance(selection,dict)
                    or selection.get("covered") is not True):
                failures.append(f"{key}: requested temperature not covered by actual evaluated kT")
            elif (selection.get("lower_evaluated_K") is None
                  or selection.get("upper_evaluated_K") is None):
                raise ValueError("accepted neutron kT selection lacks evaluated tables")
        isotope_rows.append({
            "nuclide":name,
            "evaluated_neutron_temperature_count":len(temperatures),
            "evaluated_min_K":min(temperatures) if temperatures else None,
            "evaluated_max_K":max(temperatures) if temperatures else None,
            "materials_that_use_this_nuclide":sorted({
                p["material_key"] for p in actual
            }),
            "supported_at_requested_explicit_temperatures":len(failures)==0,
            "physical_evidence_failures":sorted(set(failures)),
        })

    physically_supported=(
        config.neutron_temperature_method=="interpolation"
        and fuel_hot_test and not missing_explicit
        and bool(isotope_rows)
        and all(r["supported_at_requested_explicit_temperatures"]
                for r in isotope_rows)
        and audit.get("model_explicit_temperature_coverage_ok") is True
    )
    return {
        "schema":"nuke-stage46-real-evaluated-hot-neutron-temperature-preflight-v1",
        "real_nuclear_data_evaluation_required":True,
        "core_material_temperature_inputs_K":requested,
        "declared_materials_checked":list(REQUIRED_DECLARED_MATERIALS),
        "all_declared_solid_and_coolant_temperatures_explicit":
            not missing_explicit,
        "missing_explicit_material_temperatures":missing_explicit,
        "selected_incident_neutron_temperature_method":
            config.neutron_temperature_method,
        "hot_test_fuel_temperature_threshold_K":
            HOT_TEST_FUEL_TEMPERATURE_K,
        "fuel_above_provisional_hot_test_threshold":fuel_hot_test,
        "isotope_evaluation_evidence":isotope_rows,
        "neutron_isotopes_with_missing_hot_coverage":[
            r["nuclide"] for r in isotope_rows
            if not r["supported_at_requested_explicit_temperatures"]],
        "qualified_hot_incident_neutron_data_support":bool(physically_supported),
        "thermal_scattering_is_explicit_free_gas_surrogate":
            config.thermal_scattering_treatment=="free-gas",
        "hot_bound_thermal_scattering_qualified":False,
        "physical_hydrogen_EOS_pressure_density_thermal_feedback_qualified":False,
        "source_stationarity_verified":False,
        "historical_NASA_NERVA_hot_reactor_validated":False,
        "original_reactor_watts_calculated":False,
        "interpretation":[
            "The gate qualifies actual evaluated isotope HDF5 kT table coverage only.",
            "All declared materials checked even if not root-reachable: conservative fail-closed policy.",
            "A true multi-temperature HDF5 library is necessary but not sufficient for nuclear reactor fidelity.",
            "Fuel/hydrogen density and material geometry remain user-specified fixed values.",
            "No coupled solid/coolant thermal feedback, historical source convergence or reactor thermal W is implied.",
        ],
    }
