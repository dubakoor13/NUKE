"""Actual OpenMC core material-temperature and nuclear-data qualification.

Read physical temperatures, nuclides and S(alpha,beta) assignments from
EXECUTED per-seed OpenMC materials.xml and settings.xml, not desired
NervaConfig defaults. Independently load all 29 evaluated-neutron HDF5
temperature lists via the existing OpenMC data coverage audit and require
the actual cross_sections.xml SHA256 in transport provenance.

An unset material temperature is an implicit solver default, NOT an
explicitly validated solid operating temperature. A cold NNDC index with
one ~293.606 K evaluation cannot qualify hot fuel, Be or ZrH physics.

This is a fail-closed, observational nuclear-data evidence gate.
It never alters neutron temperatures, materials, concentrations,
cross-section libraries, enrichment, core geometry or reactor controls.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import openmc.data

from .nuclear_data import _library_entries
from .materials import EVALUATED_BOUND_THERMAL_TABLES
from .audit_nuclear_data_temperatures import audit_temperature_coverage
from .config import NervaConfig


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def _load_executed_bound_kernel_evidence(index: Path, materials: list[dict]) -> dict:
    """Load real evaluated S(alpha,beta,T) HDF5 for actually exported materials.

    Kernel *names* in materials.xml are not evidence that the referenced
    thermal evaluations exist or cover the physical material temperature.
    This is a read-only audit: no synthetic kernels or data interpolation.
    """
    requested = {
        name for material in materials
        for name in material["bound_thermal_scattering_tables"]
    }
    if not requested:
        return {}
    found = {}
    for entry in _library_entries(index):
        if entry.data_type != "thermal":
            continue
        for name in entry.materials:
            if name in requested:
                if name in found:
                    raise ValueError("duplicate evaluated bound thermal kernel: " + name)
                found[name] = entry
    result = {}
    for name in sorted(requested):
        entry = found.get(name)
        if entry is None or not entry.file_exists:
            raise ValueError("missing evaluated bound thermal HDF5 kernel: " + name)
        file = Path(entry.resolved_path)
        try:
            data = openmc.data.ThermalScattering.from_hdf5(file)
            if data.name != name:
                raise ValueError("thermal HDF5 kernel identity mismatch: " + name)
            values = data.kTs.values() if isinstance(data.kTs, dict) else data.kTs
            temperatures = sorted(
                float(kT / openmc.data.K_BOLTZMANN) for kT in values
            )
            if (not temperatures or any(
                    not math.isfinite(t) or t <= 0 for t in temperatures
            ) or len(temperatures) != len(set(temperatures))):
                raise ValueError("invalid evaluated bound kernel temperatures: " + name)
        except (OSError, KeyError, TypeError, AttributeError) as exc:
            raise ValueError("unloadable evaluated bound thermal HDF5: " + name) from exc
        result[name] = {
            "temperatures_K": temperatures,
            "hdf5_sha256": _sha(file),
        }
    return result


def _executed_material_inventory(path: Path) -> list[dict]:
    root=ET.parse(path).getroot()
    if root.tag != "materials":
        raise ValueError("executed OpenMC materials.xml root is invalid")
    output=[]
    ids=set()
    names=set()
    for elem in root.findall("material"):
        mid=int(elem.attrib["id"])
        name=elem.attrib.get("name","")
        if mid<=0 or not name or mid in ids or name in names:
            raise ValueError("duplicate/unnamed physical material")
        ids.add(mid)
        names.add(name)
        values=elem.findall("nuclide")
        nuclides=[x.attrib.get("name") for x in values]
        if not nuclides or any(not x for x in nuclides) or len(set(nuclides))!=len(nuclides):
            raise ValueError("missing or duplicate evaluated nuclide in OpenMC material")
        explicit=elem.attrib.get("temperature")
        temp=None if explicit is None else float(explicit)
        if temp is not None and (not math.isfinite(temp) or temp<=0):
            raise ValueError("nonphysical exported OpenMC material temperature")
        scatter=[x.attrib.get("name") for x in elem.findall("sab")]
        if any(not x for x in scatter) or len(set(scatter))!=len(scatter):
            raise ValueError("invalid evaluated bound thermal-scattering assignment")
        output.append({
            "material_id":mid,
            "material_name":name,
            "nuclides":nuclides,
            "explicit_material_temperature_K":temp,
            "temperature_provenance":(
                "exported_materials_xml_explicit" if temp is not None
                else "NOT_SPECIFIED_IMPLICIT_OPENMC_SOLVER_DEFAULT"
            ),
            "bound_thermal_scattering_tables":scatter,
        })
    if len(output)<2:
        raise ValueError("missing physical reactor materials")
    return sorted(output,key=lambda row:row["material_id"])


def _settings_temperature(path: Path) -> dict:
    root=ET.parse(path).getroot()
    if root.tag != "settings":
        raise ValueError("executed OpenMC settings.xml root invalid")
    if (root.findtext("run_mode")!="eigenvalue"
            or root.findtext("photon_transport","false").lower()=="true"):
        raise ValueError("temperature audit applies only to actual neutron-only eigenvalue core")
    temp=root.find("temperature")
    method=None if temp is None else temp.findtext("method")
    default=None if temp is None else temp.findtext("default")
    tol=None if temp is None else temp.findtext("tolerance")
    numeric=None if default is None else float(default)
    tolerance=None if tol is None else float(tol)
    if numeric is not None and (not math.isfinite(numeric) or numeric<=0):
        raise ValueError("nonphysical neutron transport default temperature")
    if tolerance is not None and (
        not math.isfinite(tolerance) or tolerance<=0
    ):
        raise ValueError("nonphysical neutron transport temperature tolerance")
    if method not in (None,"nearest","interpolation"):
        raise ValueError("unsupported executed OpenMC neutron temperature method")
    return {
        "explicit_openmc_settings_default_temperature_K":numeric,
        "temperature_selection_method_from_xml":method,
        "explicit_nearest_tolerance_K":tolerance,
        "solver_temperature_default_is_implicit":default is None,
    }


def evaluate_executed_neutron_temperatures(
    material_list: list[dict],
    nuclear_temperatures: dict[str,list[float]],
    setting: dict,
    *,
    nearest_tolerance_K: float=10.,
    evaluated_bound_kernels: dict[str, dict] | None = None,
) -> dict:
    """Independent scored physical-material vs evaluated HDF5 data reduction.

    Implicit fuel/solid temperatures cannot be declared hot-transport
    qualified even if the cold library happens to offer a nearby kT.
    """
    if (not material_list or not math.isfinite(nearest_tolerance_K)
            or nearest_tolerance_K<=0 or nearest_tolerance_K>20):
        raise ValueError("invalid physical neutron temperature audit tolerance")
    method=setting.get("temperature_selection_method_from_xml")
    runtime_tolerance=setting.get("explicit_nearest_tolerance_K")
    valid_nearest=(
        method=="nearest" and isinstance(runtime_tolerance,(int,float))
        and not isinstance(runtime_tolerance,bool)
        and math.isfinite(runtime_tolerance) and 0<runtime_tolerance<=20
    )
    # A desired tolerance in a Python configuration is NOT proof of the
    # selector actually executed by the neutron-transport executable.
    effective_tolerance=min(nearest_tolerance_K,float(runtime_tolerance)) if valid_nearest else nearest_tolerance_K
    rows=[]
    evaluated_bound_kernels = ({} if evaluated_bound_kernels is None
                               else evaluated_bound_kernels)
    total_explicit=0
    missing_nuclide=set()
    all_within=True
    distinct=set()
    for material in material_list:
        temp=material["explicit_material_temperature_K"]
        if temp is not None:
            if temp<=0 or not math.isfinite(temp):
                raise ValueError("invalid exported physical neutron material temperature")
            total_explicit+=1
        isotopes=[]
        for name in material["nuclides"]:
            available=nuclear_temperatures.get(name,[])
            if (not available or any(not math.isfinite(t) or t<=0 for t in available)
                    or available!=sorted(set(available))):
                missing_nuclide.add(name)
            for t in available:
                distinct.add(float(t))
            nearest=(min(available,key=lambda t:abs(t-temp))
                     if available and temp is not None else None)
            within_range=(available[0]<=temp<=available[-1]
                          if available and temp is not None else None)
            close=(within_range and abs(nearest-temp)<=effective_tolerance+1e-9
                   if nearest is not None else None)
            if temp is not None and close is not True:
                all_within=False
            isotopes.append({
                "nuclide":name,
                "evaluated_temperatures_K":available,
                "nearest_evaluated_temperature_K":nearest,
                "explicit_temperature_within_evaluated_range":within_range,
                "explicit_temperature_nearest_table_covered":close,
            })
        name=material["material_name"].lower()
        is_moderator=(
            "graphite" in name or "reflector" in name
            or "zirconium-hydride" in name
        )
        has_bound=bool(material["bound_thermal_scattering_tables"])
        # Exact phase-specific assignments are essential: any nonempty
        # S(alpha,beta) list is NOT proof of the correct chemistry.
        phase_tables={
            "beryllium radial reflector":EVALUATED_BOUND_THERMAL_TABLES["reflector"],
            "tie-tube graphite filler":EVALUATED_BOUND_THERMAL_TABLES["graphite"],
            "conceptual zirconium-hydride moderator":EVALUATED_BOUND_THERMAL_TABLES["zrh"],
        }
        expected_phase_tables=phase_tables.get(material["material_name"])
        correct_phase_assignment=(
            set(material["bound_thermal_scattering_tables"])==set(expected_phase_tables)
            if expected_phase_tables is not None else None
        )
        mixed_fuel_unqualified=(
            material["material_name"]=="conceptual (U,Zr)C-graphite surrogate"
        )
        # Mixed carbide/graphite fuel has unresolved C-phase partition.
        # Applying graphite S(a,b) to every fuel carbon atom is unphysical.
        # No model or composition mutation is performed here.
        bound_evidence=[]
        for table in material["bound_thermal_scattering_tables"]:
            kernel=evaluated_bound_kernels.get(table)
            grid=kernel.get("temperatures_K",[]) if isinstance(kernel,dict) else []
            valid_grid=(
                isinstance(grid,list) and bool(grid)
                and all(isinstance(t,(int,float)) and math.isfinite(t) and t>0
                        for t in grid)
                and grid==sorted(set(grid))
            )
            nearest=(min(grid,key=lambda t:abs(t-temp))
                     if valid_grid and temp is not None else None)
            covered=(
                nearest is not None
                and grid[0] <= temp <= grid[-1]
                and abs(nearest-temp) <= effective_tolerance+1e-9
            )
            bound_evidence.append({
                "table":table,
                "evaluated_hdf5_sha256": (
                    kernel.get("hdf5_sha256") if isinstance(kernel,dict) else None
                ),
                "evaluated_temperatures_K":grid if valid_grid else [],
                "nearest_evaluated_temperature_K":nearest,
                "executed_material_temperature_covered":bool(covered),
            })
        missing_bound=(
            (is_moderator and not has_bound)
            or mixed_fuel_unqualified
            or (correct_phase_assignment is False)
            or (has_bound and not all(
                x["executed_material_temperature_covered"] for x in bound_evidence
            ))
        )
        row={**material,
            "evaluated_isotope_temperature_evidence":isotopes,
            "all_explicit_isotope_temperatures_covered":(
                temp is not None
                and all(r["explicit_temperature_nearest_table_covered"] is True
                        for r in isotopes)
            ),
            "moderator_or_reflector_solid_declared":is_moderator,
            "expected_bound_phase_tables":(
                list(expected_phase_tables) if expected_phase_tables is not None else None
            ),
            "bound_phase_table_assignment_matches":correct_phase_assignment,
            "mixed_fuel_graphite_phase_fractions_unverified":mixed_fuel_unqualified,
            "evaluated_bound_kernel_temperature_evidence":bound_evidence,
            "bound_scattering_qualification_missing":missing_bound,
        }
        rows.append(row)
    if missing_nuclide:
        raise ValueError("evaluated nuclear data missing/invalid: "+
                         ",".join(sorted(missing_nuclide)))
    all_explicit=total_explicit==len(rows)
    multi_temperature_any=any(len(v)>1 for v in nuclear_temperatures.values())
    all_solid_bound=not any(r["bound_scattering_qualification_missing"] for r in rows)
    # An exploratory cold run can be numerically genuine while being
    # physically unqualified as a NASA nuclear thermal rocket at temperature.
    qualified=(
        all_explicit and all_within
        and multi_temperature_any and all_solid_bound and valid_nearest
        and setting.get("solver_temperature_default_is_implicit") is False
    )
    return {
        "number_of_physical_materials":len(rows),
        "number_with_explicit_exported_temperature":total_explicit,
        "number_relying_on_implicit_temperature":len(rows)-total_explicit,
        "evaluated_neutron_nuclide_count":len(nuclear_temperatures),
        "evaluated_neutron_temperature_values_K":sorted(distinct),
        "evaluated_neutron_temperature_grid_has_multiple_temperatures":
            multi_temperature_any,
        "all_material_temperatures_explicit":all_explicit,
        "all_explicit_material_temperatures_inside_nearest_evaluated_tolerance":
            all_within,
        "thermal_solid_bound_scattering_qualifications_present":all_solid_bound,
        "physical_materials":rows,
        "openmc_executed_settings_temperature":setting,
        "executed_nearest_temperature_selector_qualified":bool(valid_nearest),
        "effective_nearest_temperature_tolerance_K":effective_tolerance,
        "mixed_fuel_graphite_phase_fractions_verified":not any(
            row["mixed_fuel_graphite_phase_fractions_unverified"] for row in rows
        ),
        "hot_operating_temperature_neutron_physics_qualified":bool(qualified),
        "qualifier_reasons":[
            x for flag,x in (
                (not all_explicit, "physical material temperatures missing from executed XML"),
                (not valid_nearest, "executed OpenMC nearest-temperature method/tolerance not explicitly qualified"),
                (any(row["mixed_fuel_graphite_phase_fractions_unverified"] for row in rows),
                 "heterogeneous fuel graphite/carbon phase fractions are not documented"),
                (any(row["bound_phase_table_assignment_matches"] is False for row in rows),
                 "actual bound thermal-scattering tables do not match material phase"),
                (not multi_temperature_any, "only one evaluated nuclear-data temperature per isotope"),
                (not all_solid_bound,
                 "bound solid scattering HDF5 data or executed material temperature not qualified"),
                (not all_within, "requested explicit material temperatures not covered by evaluated data"),
                (setting.get("solver_temperature_default_is_implicit") is True,
                 "implicit OpenMC temperature default not physically certified"),
            ) if flag
        ],
        "nasa_nerva_historical_criticality_or_hot_core_benchmark_validated":False,
        "interpretation":(
            "This reads actually EXECUTED reactor materials and settings, and "
            "the genuine evaluated HDF5 neutron temperature inventory. "
            "Missing material temperatures mean solver defaults, NOT "
            "a qualified fuel/reflector temperature field. Cold neutron "
            "bookkeeping success does not demonstrate hot NERVA physics, "
            "independent source convergence, adequate graphite-phase S(a,b), "
            "local nuclear heating or measured NASA benchmark agreement. "
            "No reactor parameter is changed by the audit."
        ),
    }


def audit_actual_core_neutron_temperature(summary_path: Path,
                                          cross_sections: Path|None=None)->dict:
    path=Path(summary_path)
    report=json.loads(path.read_text(encoding="utf-8"))
    if (report.get("execution")!="ACTUAL_OPENMC_TRANSPORT"
            or report.get("assembly")!="core"
            or len(report.get("runs",[]))<4):
        raise ValueError("actual four-seed OpenMC core neutron transport required")
    index=Path(cross_sections or os.environ.get("OPENMC_CROSS_SECTIONS",""))
    if not index.is_file() or _sha(index)!=report.get("cross_sections_xml_sha256"):
        raise ValueError("evaluated neutron cross-sections index missing or SHA256 mismatched")
    seeds=set()
    model_sha={key:set() for key in ("geometry.xml","materials.xml","tallies.xml")}
    per_seed_settings_sha={}
    normalized_settings_sha=set()
    primary=None
    settings=None
    for run in report["runs"]:
        seed=run["seed"]
        statepoint=Path(run["statepoint_path"])
        if (seed in seeds or not statepoint.is_file()
                or _sha(statepoint)!=run["statepoint_sha256"]):
            raise ValueError("genuine neutron eigenvalue seed/statepoint mismatch")
        seeds.add(seed)
        provenance=json.loads((statepoint.parent/"transport_provenance.json").read_text())
        if (provenance.get("config",{}).get("seed")!=seed
                or provenance.get("nuclear_data",{}).get("cross_sections_sha256")!=_sha(index)
                or provenance.get("assembly")!="core"):
            raise ValueError("different evaluated nuclear data or source seed")
        for name in (*model_sha, "settings.xml"):
            candidate=statepoint.parent/name
            if (_sha(candidate)!=provenance.get("xml_sha256",{}).get(name)):
                raise ValueError("actual reactor physical XML/provenance tampered: "+name)
            if name in model_sha:
                model_sha[name].add(_sha(candidate))
        # settings.xml legitimately differs per independent random seed.
        # Require the native OpenMC <seed> to match, and compare all OTHER
        # settings identically, rather than incorrectly expecting a single
        # settings.xml hash across different neutron source seeds.
        settings_xml=statepoint.parent/"settings.xml"
        per_seed_settings_sha[str(seed)]=_sha(settings_xml)
        settings_tree=ET.parse(settings_xml).getroot()
        seed_node=settings_tree.find("seed")
        if seed_node is None or int(seed_node.text)!=seed:
            raise ValueError("executed OpenMC settings random source seed mismatch")
        settings_tree.remove(seed_node)
        normalized_settings_sha.add(hashlib.sha256(
            ET.tostring(settings_tree,encoding="utf-8")
        ).hexdigest())
        material=_executed_material_inventory(statepoint.parent/"materials.xml")
        physical_settings=_settings_temperature(settings_xml)
        if primary is None:
            primary=material
            settings=physical_settings
        elif material!=primary or settings!=physical_settings:
            raise ValueError("four-seed reactor physical materials or temperatures changed")
    if (any(len(x)!=1 for x in model_sha.values())
            or len(normalized_settings_sha)!=1):
        raise ValueError("four-seed reactor geometry/material/tallies/settings except RNG seed inconsistent")

    cfg=replace(NervaConfig(),
        hydrogen_temperature_k=float(report["hydrogen_temperature_K"]),
        thermal_scattering_treatment=report["thermal_scattering_treatment"],
        bound_solid_temperature_k=report.get("bound_solid_temperature_K"),
        bound_nearest_temperature_tolerance_k=float(
            report.get("bound_nearest_temperature_tolerance_K") or 0),
    )
    data=audit_temperature_coverage(index,cfg)
    temperatures={}
    for entry in data["nuclides"]:
        if entry["load_error"] is not None:
            raise ValueError("unloadable evaluated neutron isotope HDF5")
        temperatures[entry["nuclide"]]=entry["available_temperatures_K"]
    nuclides={name for material in primary for name in material["nuclides"]}
    if set(temperatures)!=nuclides:
        raise ValueError("actually exported reactor nuclides differ from evaluated data audit")
    bound_kernels=_load_executed_bound_kernel_evidence(index,primary)
    physical=evaluate_executed_neutron_temperatures(
        primary,temperatures,settings,evaluated_bound_kernels=bound_kernels
    )
    physical.update({
        "evaluated_bound_kernel_hdf5_evidence":bound_kernels,
        "analysis":"ACTUAL_OPENMC_NUCLEAR_DATA_AND_EXPORTED_MATERIAL_TEMPERATURE_QUALIFICATION",
        "real_transport_summary_sha256":_sha(path),
        "cross_sections_xml_sha256":_sha(index),
        "actual_openmc_seed_count":len(seeds),
        "physical_model_xml_sha256":{k:next(iter(v)) for k,v in model_sha.items()},
        "settings_xml_per_seed_sha256":per_seed_settings_sha,
        "settings_xml_except_rng_seed_sha256":next(iter(normalized_settings_sha)),
        "all_real_source_model_provenance_consistent":True,
    })
    return physical


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--cross-sections",type=Path,default=None)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=audit_actual_core_neutron_temperature(a.summary,a.cross_sections)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("Actual OpenMC evaluated nuclides",result["evaluated_neutron_nuclide_count"])
    print("Explicit physical temperatures",result["number_with_explicit_exported_temperature"],
          "/",result["number_of_physical_materials"])
    print("Evaluated neutron temperatures [K]",result["evaluated_neutron_temperature_values_K"])
    print("Hot core nuclear-data qualification:",
          "PASS" if result["hot_operating_temperature_neutron_physics_qualified"]
          else "FAIL_PHYSICAL_QUALIFICATION_EXPECTED")
    print("NASA historical NERVA reactor validation: NO")
    # Return success when this diagnostic truthfully observes unqualified
    # physical input; CI asserts unqualified for the cold NNDC demonstration.
    return 0


if __name__=="__main__":
    raise SystemExit(main())
