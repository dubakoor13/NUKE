"""Audit neutron-data temperature coverage for the current NERVA materials."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import openmc.data

from .config import NervaConfig
from .materials import build_materials
from .nuclear_data import _library_entries, resolve_cross_sections


# HDF5 kT -> Kelvin conversion and serialized requested material
# temperatures can differ by a few floating-point ulps. This is not a
# physical nearest-table tolerance and must never license extrapolation.
ROUNDING_ONLY_TEMPERATURE_EPSILON_K = 1e-8

def resolve_evaluated_neutron_temperature(
    grid: list[float], requested_k: float, *, method: str,
    nearest_tolerance_k: float,
) -> dict:
    """Select genuine neutron kT tables without extrapolating hot XS."""
    values = sorted(float(x) for x in grid)
    if (not values or any(not math.isfinite(x) or x<=0 for x in values)
            or len(values)!=len(set(values))
            or not math.isfinite(requested_k) or requested_k<=0
            or method not in ("nearest","interpolation")
            or not math.isfinite(nearest_tolerance_k)
            or nearest_tolerance_k<0):
        raise ValueError("invalid requested neutron temperature or evaluated kT grid")
    lower=values[0]
    upper=values[-1]
    epsilon=ROUNDING_ONLY_TEMPERATURE_EPSILON_K
    if requested_k<lower-epsilon or requested_k>upper+epsilon:
        return {
            "covered":False,
            "reason":"temperature outside evaluated neutron kT support; no extrapolation",
            "lower_evaluated_K":None,"upper_evaluated_K":None,
            "upper_table_weight":None,
        }
    nearest=min(values,key=lambda x:abs(x-requested_k))
    if method=="nearest":
        covered=abs(nearest-requested_k)<=nearest_tolerance_k+epsilon
        return {
            "covered":bool(covered),
            "reason":("evaluated nearest temperature accepted" if covered
                      else "nearest evaluated kT offset exceeds actual OpenMC tolerance"),
            "lower_evaluated_K":nearest,
            "upper_evaluated_K":nearest,
            "upper_table_weight":0.0,
        }
    if abs(nearest-requested_k)<=epsilon:
        return {
            "covered":True,
            "reason":"exact evaluated temperature; no interpolation needed",
            "lower_evaluated_K":nearest,
            "upper_evaluated_K":nearest,
            "upper_table_weight":0.0,
        }
    for lo,hi in zip(values,values[1:]):
        if lo<requested_k<hi:
            return {
                "covered":True,
                "reason":"bracketed evaluated incident-neutron temperature interpolation",
                "lower_evaluated_K":lo,
                "upper_evaluated_K":hi,
                "upper_table_weight":float((requested_k-lo)/(hi-lo)),
            }
    return {
        "covered":False,
        "reason":"no two neighboring evaluated temperatures bracket neutron state",
        "lower_evaluated_K":None,"upper_evaluated_K":None,
        "upper_table_weight":None,
    }



def _requested_temperatures(
    config: NervaConfig,
) -> dict[str, list[dict]]:
    materials = build_materials(config)
    requested: dict[str, list[dict]] = {}
    for key, material in materials.items():
        temperature = material.temperature
        for nuclide in material.get_nuclides():
            requested.setdefault(nuclide, []).append(
                {
                    "material_key": key,
                    "material_name": material.name,
                    "requested_temperature_K": (
                        None
                        if temperature is None
                        else float(temperature)
                    ),
                }
            )
    return requested


def audit_temperature_coverage(
    cross_sections: str | Path | None = None,
    config: NervaConfig | None = None,
) -> dict:
    if config is None:
        config = NervaConfig()

    path = resolve_cross_sections(cross_sections)
    entries = _library_entries(path)
    neutron_paths: dict[str, Path] = {}
    for entry in entries:
        if entry.data_type != "neutron":
            continue
        file_path = Path(entry.resolved_path)
        for material in entry.materials:
            if material in neutron_paths:
                raise ValueError(
                    "ambiguous duplicate evaluated neutron-data index for " + material
                )
            neutron_paths[material] = file_path

    requested = _requested_temperatures(config)
    rows = []

    for nuclide in sorted(requested):
        file_path = neutron_paths.get(nuclide)
        row = {
            "nuclide": nuclide,
            "hdf5_path": None if file_path is None else str(file_path),
            "requested_by_material": requested[nuclide],
            "available_temperatures_K": [],
            "available_min_K": None,
            "available_max_K": None,
            "explicit_requested_temperatures_in_range": None,
            "explicit_requested_temperatures_within_nearest_tolerance": None,
            "explicit_requested_temperatures_supported_by_actual_method": None,
            "load_error": None,
        }

        if file_path is None:
            row["load_error"] = "nuclide not present in neutron library inventory"
            rows.append(row)
            continue
        if not file_path.is_file():
            row["load_error"] = "referenced HDF5 file does not exist"
            rows.append(row)
            continue

        try:
            data = openmc.data.IncidentNeutron.from_hdf5(
                file_path
            )
            if data.name != nuclide:
                raise ValueError(
                    f"evaluated neutron HDF5 identity mismatch: "
                    f"index={nuclide}, file={data.name}"
                )
            temperatures = [
                float(kT / openmc.data.K_BOLTZMANN)
                for kT in data.kTs
            ]
            if (not temperatures
                    or any(not math.isfinite(t) or t <= 0 for t in temperatures)
                    or len(temperatures) != len(set(temperatures))):
                raise ValueError("invalid evaluated neutron temperature grid")
            temperatures.sort()
        except Exception as exc:
            row["load_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue
        row["available_temperatures_K"] = temperatures
        if temperatures:
            row["available_min_K"] = temperatures[0]
            row["available_max_K"] = temperatures[-1]

            explicit = [
                item["requested_temperature_K"]
                for item in requested[nuclide]
                if item["requested_temperature_K"] is not None
            ]
            if explicit:
                row["explicit_requested_temperatures_in_range"] = all(
                    math.isfinite(value)
                    and temperatures[0] - ROUNDING_ONLY_TEMPERATURE_EPSILON_K
                    <= value <=
                    temperatures[-1] + ROUNDING_ONLY_TEMPERATURE_EPSILON_K
                    for value in explicit
                )
                tol = (
                    max(1e-6, config.bound_nearest_temperature_tolerance_k)
                    if (config.thermal_scattering_treatment == "evaluated-bound"
                        or config.bound_nearest_temperature_tolerance_k > 0)
                    else 10.0
                )
                row["explicit_requested_temperatures_within_nearest_tolerance"] = all(
                    math.isfinite(value)
                    and temperatures[0] - ROUNDING_ONLY_TEMPERATURE_EPSILON_K
                    <= value <=
                    temperatures[-1] + ROUNDING_ONLY_TEMPERATURE_EPSILON_K
                    and min(abs(x-value) for x in temperatures) <= tol + 1e-9
                    for value in explicit
                )

                selections=[]
                for item in row["requested_by_material"]:
                    if item["requested_temperature_K"] is None:
                        continue
                    decision=resolve_evaluated_neutron_temperature(
                        temperatures,item["requested_temperature_K"],
                        method=config.neutron_temperature_method,
                        nearest_tolerance_k=tol,
                    )
                    item["actual_transport_temperature_selection"]=decision
                    selections.append(decision["covered"])
                row["explicit_requested_temperatures_supported_by_actual_method"]=all(selections)
                for item in row["requested_by_material"]:
                    value = item["requested_temperature_K"]
                    if value is None:
                        item["nearest_available_temperature_K"] = None
                        item["nearest_temperature_delta_K"] = None
                        continue
                    nearest = min(
                        temperatures,
                        key=lambda candidate: abs(candidate - value),
                    )
                    item["nearest_available_temperature_K"] = nearest
                    item["nearest_temperature_delta_K"] = nearest - value

        rows.append(row)

    explicit_rows = [
        row
        for row in rows
        if any(
            item["requested_temperature_K"] is not None
            for item in row["requested_by_material"]
        )
    ]
    explicit_coverage_ok = all(
        row["load_error"] is None
        and row["explicit_requested_temperatures_in_range"] is True
        and row["explicit_requested_temperatures_supported_by_actual_method"] is True
        for row in explicit_rows
    )

    return {
        "cross_sections_xml": str(path),
        "neutron_temperature_method": config.neutron_temperature_method,
        "neutron_temperature_tolerance_K": (
            (max(1e-6, config.bound_nearest_temperature_tolerance_k)
             if (config.thermal_scattering_treatment == "evaluated-bound"
                 or config.bound_nearest_temperature_tolerance_k > 0)
             else 10.0)
            if config.neutron_temperature_method == "nearest" else None
        ),
        "interpolation_never_extrapolates_beyond_evaluated_data": True,
        "model_explicit_temperature_coverage_ok": explicit_coverage_ok,
        "evaluated_temperature_extrapolation_permitted": False,
        "nonphysical_extrapolation_rounding_slack_K":
            ROUNDING_ONLY_TEMPERATURE_EPSILON_K,
        "all_indexed_required_nuclides_have_verified_hdf5_identity": all(
            row["load_error"] is None for row in rows
        ),
        "nuclides": rows,
        "interpretation": (
            "Read-only audit of available neutron-data temperatures. "
            "This report does not change OpenMC temperature treatment, "
            "material temperatures, thermal scattering data, composition, "
            "or reactor configuration."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit neutron HDF5 temperature coverage for the current "
            "NERVA material definitions."
        )
    )
    parser.add_argument(
        "--cross-sections",
        type=Path,
        default=None,
        help="cross_sections.xml; defaults to OPENMC_CROSS_SECTIONS.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "build/nerva_nuclear_data_temperature_audit.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_temperature_coverage(
        cross_sections=args.cross_sections,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    failed = [
        row
        for row in report["nuclides"]
        if row["load_error"] is not None
    ]
    print("NERVA neutron-data temperature audit:")
    print(f"  nuclides inspected: {len(report['nuclides'])}")
    print(f"  load/inventory errors: {len(failed)}")
    print(
        f"  explicit model temperatures covered by library range: "
        f"{report['model_explicit_temperature_coverage_ok']}"
    )
    print(f"  output: {args.output}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
