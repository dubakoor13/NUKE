"""Independent evaluated-data feasibility gate for a real OpenMC scattering pair.

An OpenMC nearest-temperature tolerance can make a simulation executable
even when no evaluated neutron cross sections cover the physical solid
temperature requested for its S(alpha,beta) evaluation. Such a run is NOT
physically matched and must not be compared as validated bound physics.

This file is observation and workflow gating only. It neither constructs
nor tunes the reactor model or creates nuclear data.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .audit_bound_thermal_scattering import audit_bound_thermal_scattering
from .audit_nuclear_data_temperatures import audit_temperature_coverage
from .config import NervaConfig
from .nuclear_data import resolve_cross_sections


def assess_scattering_pair(free: dict, bound: dict, kernels: dict) -> dict:
    """Require physical overlap of *actual* evaluated neutron and S(a,b) grids."""
    reasons = []
    if free.get("model_explicit_temperature_coverage_ok") is not True:
        reasons.append("free-gas control lacks evaluated neutron XS at requested material temperatures")
    if bound.get("model_explicit_temperature_coverage_ok") is not True:
        reasons.append("bound-scattering case lacks evaluated neutron XS at requested material temperatures")
    if kernels.get("ready") is not True or not all(
        row.get("actual_hdf5_kernel_loaded") is True
        and row.get("strict_tabulated_temperature_covered") is True
        for row in kernels.get("tables", [])
    ) or not kernels.get("tables"):
        reasons.append("bound thermal-scattering S(a,b) kernel temperature domain not physically verified")
    # Report each exact evaluated neutron nuclide/material conflict, rather
    # than silently selecting a conveniently close out-of-range library node.
    incompatible = []
    for label, audit in (("free", free), ("bound", bound)):
        for row in audit.get("nuclides", []):
            if row.get("load_error") is not None:
                incompatible.append({
                    "case": label, "nuclide": row.get("nuclide"),
                    "error": row["load_error"]
                })
            for material in row.get("requested_by_material", []):
                temp = material.get("requested_temperature_K")
                if temp is not None and (
                    row.get("explicit_requested_temperatures_in_range") is not True
                    or row.get("explicit_requested_temperatures_within_nearest_tolerance") is not True
                ):
                    incompatible.append({
                        "case": label,
                        "nuclide": row.get("nuclide"),
                        "material": material.get("material_name"),
                        "requested_temperature_K": temp,
                        "evaluated_temperature_range_K": [
                            row.get("available_min_K"), row.get("available_max_K")
                        ],
                    })
    return {
        "audit": "ACTUAL_EVALUATED_NEUTRON_AND_BOUND_KERNEL_TEMPERATURE_COMPATIBILITY",
        "matched_physical_temperature_data_available": not reasons,
        "real_bound_vs_free_transport_comparison_permitted": not reasons,
        "actual_transport_executed_by_this_preflight": False,
        "blockers": reasons,
        "uncovered_executed_material_nuclides": incompatible,
        "bound_sab_table_temperatures_K": {
            row["table"]: row.get("evaluated_temperatures_K", [])
            for row in kernels.get("tables", [])
        },
        "nasa_nerva_hot_core_physically_validated": False,
        "fission_source_stationarity_validated": False,
        "interpretation": (
            "Matching evaluated neutron and bound S(a,b) temperatures is a "
            "necessary precondition for a meaningful transport comparison, "
            "not proof of physically correct heterogeneous fuel scattering, "
            "source convergence or a historical NASA reactor benchmark. "
            "When no overlap exists the experiment MUST remain unexecuted."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cross-sections", type=Path, default=None)
    ap.add_argument("--hydrogen-temperature-k", type=float, required=True)
    ap.add_argument("--solid-temperature-k", type=float, required=True)
    ap.add_argument("--nearest-tolerance-k", type=float, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    data = resolve_cross_sections(args.cross_sections)
    cfg = NervaConfig(
        core_rings=3,
        hydrogen_temperature_k=args.hydrogen_temperature_k,
        bound_solid_temperature_k=args.solid_temperature_k,
        bound_nearest_temperature_tolerance_k=args.nearest_tolerance_k,
    )
    cfg.validate()
    bound_cfg = replace(cfg, thermal_scattering_treatment="evaluated-bound")
    bound_cfg.validate()
    free = audit_temperature_coverage(data, cfg)
    bound = audit_temperature_coverage(data, bound_cfg)
    kernels = audit_bound_thermal_scattering(data, bound_cfg)
    report = assess_scattering_pair(free, bound, kernels)
    report["cross_sections_xml"] = str(data)
    report["requested_hydrogen_temperature_K"] = args.hydrogen_temperature_k
    report["requested_bound_solid_temperature_K"] = args.solid_temperature_k
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("Matched evaluated neutron + bound scattering data:",
          report["matched_physical_temperature_data_available"])
    for blocker in report["blockers"]:
        print("PHYSICAL BLOCKER:", blocker)
    print("Report:", args.output)
    # Incompatible physics is an *expected, documented* blocking result,
    # not permission to run an unqualified transport experiment.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
