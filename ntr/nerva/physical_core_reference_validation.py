"""Source-specific core *physical-fidelity* assessment, not criticality tuning.

Evidence: NASA NTRS 20180000405, Borowski et al., p. 10, Fig. 8.
- Rover/NERVA heritage fuel element: 0.75 inch across flats, 19 bores,
  length 1.32 m; element-level only, not an identical reactor design.
- Proposed (not built) SNRE: 35-inch fuel element; 564 fuel / 241 tie
  elements; 14.7 cm radial Be reflector. This is a different reactor
  design and must NOT be mixed with the 132 cm heritage element.

A dimension match of individual elements is NOT reactor-core validation.
The NUKE three-ring cold fuel surrogate is not a historical NASA geometry
or certified material configuration. No enrichment or reactivity tuning.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from .config import NervaConfig
from .geometry import geometry_summary, mixed_core_summary

NASA_NTRS = "https://ntrs.nasa.gov/api/citations/20180000405/downloads/20180000405.pdf"


def assess_core_fidelity(config: NervaConfig, assembly: str = "core") -> dict:
    config.validate()
    if assembly not in ("core", "mixed-core"):
        raise ValueError("NASA core physical-fidelity assessment requires 'core' or 'mixed-core'")

    summary = geometry_summary(config) if assembly == "core" else mixed_core_summary(config)
    heritage = {
        "element_flat_to_flat_cm": {
            "reference": 1.905,
            "model": config.fuel_flat_to_flat_cm,
            "tolerance": 0.0254,
        },
        "fuel_coolant_channels_per_element": {
            "reference": 19,
            "model": (
                summary["coolant_channels_per_element"] if assembly == "core"
                else 19
            ),
            "tolerance": 0,
        },
        "heritage_element_length_cm": {
            "reference": 132.,
            "model": config.active_length_cm,
            "tolerance": 1.,
        },
    }
    for value in heritage.values():
        value["within_tolerance"] = abs(value["model"] - value["reference"]) <= value["tolerance"]

    proposed_snre = {
        "snre_fuel_element_count": {
            "reference": 564, "model": summary["fuel_elements"], "tolerance": 0,
        },
        "snre_tie_tube_count": {
            "reference": 241, "model": summary["tie_tubes"], "tolerance": 0,
        },
        "snre_active_element_length_cm": {
            "reference": 88.9,
            "model": config.active_length_cm,
            "tolerance": 0.1,
        },
        "snre_reflector_radial_thickness_cm": {
            "reference": 14.7, "model": config.reflector_thickness_cm, "tolerance": 0.1,
        },
    }
    for value in proposed_snre.values():
        value["within_tolerance"] = abs(value["model"] - value["reference"]) <= value["tolerance"]

    heritage_element_geometry_match = all(
        value["within_tolerance"] for value in heritage.values()
    )
    snre_bulk_geometry_match = all(
        value["within_tolerance"] for value in proposed_snre.values()
    )
    return {
        "reference": NASA_NTRS,
        "page_number_printed": 10,
        "model_assembly": assembly,
        "model_type": "conceptual reduced NUKE reactor core",
        "heritage_element_reference": heritage,
        "proposed_snre_core_reference": proposed_snre,
        "heritage_element_geometry_matches": heritage_element_geometry_match,
        "proposed_snre_bulk_geometry_matches": snre_bulk_geometry_match,
        "material_fuel_matrix_is_verified_historical_nerva_composition": False,
        "fuel_uranium_loading_is_historically_verified": False,
        "bound_thermal_scattering_mode": config.thermal_scattering_treatment,
        "bound_fuel_carbon_phase_fractions_verified": False,
        "high_temperature_neutron_xs_verified": False,
        "reactor_criticality_validated_against_nasa_measurements": False,
        "overall_historical_core_validation": "NOT_VALIDATED",
        "interpretation": (
            "NASA describes DISTINCT heritage NERVA element and unbuilt "
            "SNRE reactor configurations; the element length must not be "
            "taken from heritage NERVA while comparing SNRE element counts. "
            "Element-level dimensional agreement cannot validate fuel "
            "chemistry, uranium inventory, reflector material conditions, "
            "thermal scattering, neutron leakage, flux, k-effective or "
            "fission-source convergence. This diagnostic does not recommend "
            "design modifications or optimize reactor criticality."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assembly", choices=("core", "mixed-core"), default="core")
    p.add_argument("--rings", type=int, default=3)
    p.add_argument("--output", type=Path, default=Path("build/nerva_core_reference_audit.json"))
    a = p.parse_args()
    report = assess_core_fidelity(NervaConfig(core_rings=a.rings), a.assembly)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("NASA 2018 heritage element geometry match:",report["heritage_element_geometry_matches"])
    print("NASA 2018 unbuilt SNRE core geometry match:",report["proposed_snre_bulk_geometry_matches"])
    print("Historical NERVA core physical validation:",report["overall_historical_core_validation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
