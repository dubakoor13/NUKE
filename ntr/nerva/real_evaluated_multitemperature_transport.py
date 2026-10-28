"""Stage47: verify REAL two-table temperature sampling in original OpenMC C++.

No synthetic neutron cross sections, no fuel/geometry/control changes.
This audit accepts only evaluated incident-neutron HDF5 with TWO genuine
temperature tables bracketing the explicitly requested hot material state.

OpenMC v0.15.3 src/nuclide.cpp uses stochastic table sampling:
  f = (kT-kT_lower)/(kT_upper-kT_lower)
  if (f > prn(p.current_seed())) ++i_temp;
Its global interpolation setting can be DOWNGRADED to nearest if an
incident neutron isotope has just a single table. All required nuclides
must therefore pass, not merely the fuel nuclide.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .config import NervaConfig
from .audit_nuclear_data_temperatures import audit_temperature_coverage
from .hot_neutron_temperature_data_qualification import (
    qualify_explicit_hot_neutron_data,
)

OFFICIAL_CPP_BLOB_SHA="5ae6e30ee24e4f16c31fd04dbf9b7c9b4afa5c3e"
INTERMEDIATE_K=750.0
REFERENCE_K=900.0
SOURCE_SEED=13007
BATCHES=24
INACTIVE=8
PARTICLES=3000


def _sha(path:Path)->str:
    if not path.is_file():
        raise FileNotFoundError("original nuclear-evidence file missing: "+str(path))
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def verify_upstream_cpp_temperature_sampling(source:Path)->dict:
    """Pin exact official upstream C++ release blob and sampling equation."""
    raw=source.read_bytes()
    blob=hashlib.sha1(("blob "+str(len(raw))+"\\0").encode()+raw).hexdigest()
    if blob!=OFFICIAL_CPP_BLOB_SHA:
        raise ValueError("OpenMC v0.15.3 src/nuclide.cpp Git blob SHA mismatch")
    text=raw.decode("utf-8")
    required=(
        "temps_available.size() == 1",
        "settings::temperature_method = TemperatureMethod::NEAREST",
        "case TemperatureMethod::INTERPOLATION",
        "f = (kT - kTs_[i_temp]) / (kTs_[i_temp + 1] - kTs_[i_temp]);",
        "if (f > prn(p.current_seed()))",
    )
    if not all(token in text for token in required):
        raise ValueError("original OpenMC stochastic temperature sampling not present")
    return {"v0153_upstream_source_git_blob_sha":blob,
            "source_sha256":_sha(source),
            "stochastic_evaluated_table_sampling_confirmed":True,
            "one_temperature_global_fallback_present":True}


def strict_two_table_coverage(audit:dict, *, expected_K:float)->dict:
    """Fail closed when even ONE required evaluated nuclide lacks brackets."""
    if (audit.get("neutron_temperature_method")!="interpolation"
            or not audit.get("model_explicit_temperature_coverage_ok")
            or not audit.get("nuclides")):
        raise ValueError("unqualified evaluated neutron data or wrong selector")
    rows=[]
    for nuclide in audit["nuclides"]:
        grid=nuclide.get("available_temperatures_K",[])
        if (nuclide.get("load_error") is not None or len(grid)<2):
            raise ValueError("missing real multi-temperature neutron HDF5 for "+str(nuclide.get("nuclide")))
        for request in nuclide["requested_by_material"]:
            if (request["requested_temperature_K"] is None
                    or not math.isclose(request["requested_temperature_K"],expected_K,
                                        abs_tol=1e-8,rel_tol=0)):
                raise ValueError("material neutron temperature not explicit/constant")
            selected=request.get("actual_transport_temperature_selection")
            if not isinstance(selected,dict) or selected.get("covered") is not True:
                raise ValueError("uncovered nuclear HDF5 temperature")
            lo,hi=selected.get("lower_evaluated_K"),selected.get("upper_evaluated_K")
            weight=selected.get("upper_table_weight")
            if (lo is None or hi is None or weight is None
                    or not (lo<expected_K<hi)
                    or not (0.0<weight<1.0)
                    or not math.isclose(weight,(expected_K-lo)/(hi-lo),
                                        rel_tol=1e-12,abs_tol=1e-12)):
                raise ValueError("no two distinct original evaluated HDF5 kT tables bracketing material")
            rows.append({"isotope":nuclide["nuclide"],
                         "material":request["material_key"],
                         "temperature_K":expected_K,
                         "genuine_lower_table_K":lo,
                         "genuine_upper_table_K":hi,
                         "upper_table_selection_probability":weight,
                         "evaluated_neutron_HDF5_path":nuclide["hdf5_path"]})
    return {"verified_real_isotope_material_pairs":len(rows),
            "verified_unique_evaluated_isotopes":len(set(x["isotope"] for x in rows)),
            "real_evaluated_tables_on_both_sides_of_each_explicit_material_temperature":True,
            "material_isotope_table_details":rows}


def _model_for_temperature(temp:float)->NervaConfig:
    return replace(
        NervaConfig(),core_rings=3,
        fuel_temperature_k=temp,hydrogen_temperature_k=temp,
        bound_solid_temperature_k=temp,structural_solid_temperature_k=temp,
        neutron_temperature_method="interpolation",
    )


def read_original_openmc_transport(folder:Path, xs:Path, *, temp:float)->dict:
    source=folder/"transport_provenance.json"
    report=json.loads(source.read_text(encoding="utf-8"))
    cfg=report["config"]
    if (cfg.get("neutron_temperature_method")!="interpolation"
            or cfg.get("batches")!=BATCHES or cfg.get("inactive")!=INACTIVE
            or cfg.get("particles")!=PARTICLES or cfg.get("seed")!=SOURCE_SEED
            or cfg.get("photon_transport") is not False):
        raise ValueError("original C++ eigenvalue settings differ beyond temperature")
    for key in ("fuel_temperature_k","hydrogen_temperature_k",
                "bound_solid_temperature_k","structural_solid_temperature_k"):
        if not math.isclose(float(cfg[key]),temp,rel_tol=0,abs_tol=1e-8):
            raise ValueError("material temperature not exactly declared in original transport")
    root=ET.parse(folder/"settings.xml").getroot()
    if root.findtext("temperature_method")!="interpolation":
        raise ValueError("C++ neutronics temperature selection not exported")
    if report["nuclear_data"]["cross_sections_sha256"]!=_sha(xs):
        raise ValueError("original evaluated library fingerprint changed")
    for name in ("materials.xml","geometry.xml","settings.xml"):
        if report["xml_sha256"][name]!=_sha(folder/name):
            raise ValueError("actual original OpenMC physical input XML SHA changed")
    model_cfg=_model_for_temperature(temp)
    data=audit_temperature_coverage(cross_sections=xs,config=model_cfg)
    gate=qualify_explicit_hot_neutron_data(model_cfg,data)
    if not gate["qualified_hot_incident_neutron_data_support"]:
        raise ValueError("genuine evaluated multi-kT nuclear data did not cover physical material state")
    selections=(strict_two_table_coverage(data,expected_K=temp)
                if temp==INTERMEDIATE_K else None)
    original_hdf5=folder/f"statepoint.{BATCHES}.h5"
    if report["statepoint"]["sha256"]!=_sha(original_hdf5):
        raise ValueError("actual evaluated-neutron C++ HDF5 SHA changed")
    with h5py.File(original_hdf5,"r") as h:
        if (bytes(h.attrs.get("filetype",b"")).rstrip(b"\0")!=b"statepoint"
                or bytes(h["run_mode"][()]).rstrip(b"\0")!=b"eigenvalue"
                or int(h["n_particles"][()])!=PARTICLES
                or int(h["n_inactive"][()])!=INACTIVE
                or int(h["current_batch"][()])!=BATCHES
                or int(h["seed"][()])!=SOURCE_SEED
                or int(h.attrs.get("source_present",0))!=1):
            raise ValueError("not original executed neutron eigenvalue source")
        kg=np.asarray(h["k_generation"][()],dtype=float)
        entropy=np.asarray(h["entropy"][()],dtype=float)
        bank=np.asarray(h["source_bank"][()])
    if (kg.shape!=(BATCHES,) or entropy.shape!=(BATCHES,)
            or bank.shape!=(PARTICLES,) or not np.isfinite(kg).all()
            or not np.isfinite(entropy).all() or np.any(kg<=0)):
        raise ValueError("actual original C++ eigenvalue neutron k or bank invalid")
    return {"temperature_K":temp,"sourcebank":bank,"k":kg,"entropy":entropy,
            "actual_hdf5_sha256":_sha(original_hdf5),
            "nuclear_data_gate":gate,
            "actual_bracketed_evaluated_neutron_tables":selections,
            "geometry_xml_sha256":report["xml_sha256"]["geometry.xml"],
            "evaluated_data_index_sha256":_sha(xs)}


def audit_real_multitemperature_cpp(
    results:Path, xs:Path, official_cpp:Path,
)->dict:
    upstream=verify_upstream_cpp_temperature_sampling(official_cpp)
    intermediate=read_original_openmc_transport(results/"intermediate_750K",xs,temp=INTERMEDIATE_K)
    reference=read_original_openmc_transport(results/"reference_900K",xs,temp=REFERENCE_K)
    if intermediate["geometry_xml_sha256"]!=reference["geometry_xml_sha256"]:
        raise ValueError("original reactor geometry changed in the temperature-only comparison")
    a,b=intermediate,reference
    return {
        "schema":"nuke-stage47-original-cplusplus-real-multitemperature-xs-v1",
        "original_evaluated_neutron_source_runs":2,
        "temperature_only_fixed_composition_and_geometry":True,
        "original_openmc_v0153_temperature_sampling_cpp_provenance":upstream,
        "real_multi_temperature_evaluated_neutron_brackets":a[
            "actual_bracketed_evaluated_neutron_tables"],
        "evaluated_neutron_index_sha256":a["evaluated_data_index_sha256"],
        "source_temperature_cases":[{
            "temperature_K":x["temperature_K"],
            "original_statepoint_sha256":x["actual_hdf5_sha256"],
            "active_k_arithmetic_mean_unqualified":float(np.mean(x["k"][INACTIVE:])),
            "final_source_entropy":float(x["entropy"][-1]),
            "final_native_source_bank_sha256":hashlib.sha256(x["sourcebank"].tobytes()).hexdigest(),
        } for x in (a,b)],
        "max_abs_generation_k_difference":float(np.max(np.abs(a["k"]-b["k"]))),
        "source_banks_bit_identical":bool(np.array_equal(a["sourcebank"],b["sourcebank"])),
        "two_original_cpp_evaluated_data_temperature_cases_executed":True,
        "historical_NERVA_reactor_temperature_coefficient_qualified":False,
        "statistical_source_stationarity_or_uncertainty_qualified":False,
        "coolant_EOS_fixed_density_temperature_feedback_qualified":False,
        "reactor_watts_or_hot_NASA_NERVA_model_qualified":False,
        "limitations":[
            "Actual stochastic interpolation selection is from original official OpenMC C++ v0.15.3.",
            "Only evaluated neutron temperature is changed; no validated thermal-fluid energy balance.",
            "Two single-seed short runs are not a temperature-reactivity coefficient or source-convergence study.",
            "Bound S(alpha,beta) scattering and historical reactor geometry remain unqualified.",
        ],
    }


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--cross-sections",type=Path,required=True)
    p.add_argument("--upstream-cpp",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=audit_real_multitemperature_cpp(a.results,a.cross_sections,a.upstream_cpp)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Actual evaluated nuclear-isotope table pairs:",result[
        "real_multi_temperature_evaluated_neutron_brackets"]["verified_unique_evaluated_isotopes"])
    print("Maximum pairwise original generation k difference:",result["max_abs_generation_k_difference"])
    print("Stationarity/thermal feedback/NASA NERVA hot core: NOT QUALIFIED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
