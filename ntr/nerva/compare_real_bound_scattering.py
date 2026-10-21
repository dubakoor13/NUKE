"""Real OpenMC free-gas vs evaluated S(alpha,beta) comparison.

Two independent groups use identical geometry, nuclide atom densities, model
temperatures, tally meshes, XS index and Monte Carlo settings. The ONLY
material difference allowed is the attachment of evaluated bound-neutron
scattering S(alpha,beta) tables. Real HDF5 statepoints must have valid
recorded file hashes. Population eigenvalues in the 3-ring surrogate do
not demonstrate stationary neutron source or historical reactor fidelity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from statistics import mean
import math


def _load(root: Path, mode: str) -> tuple[dict,list[dict]]:
    s = json.loads((root/"real_core_transport_summary.json").read_text())
    if s.get("execution") != "ACTUAL_OPENMC_TRANSPORT":
        raise ValueError("genuine OpenMC executable evidence required")
    if s.get("assembly") != "core":
        raise ValueError("reactor core assembly must be held fixed")
    if s.get("thermal_scattering_treatment") != mode:
        raise ValueError("thermal treatment mismatch in actual transport summary")
    rows = s.get("runs",[])
    if len(rows)<4 or len({v["seed"] for v in rows}) != len(rows):
        raise ValueError("four distinct real transport seeds per mode required")
    for entry in rows:
        h=entry.get("hdf5_eigenvalue_integrity",{})
        if h.get("seed") != entry["seed"] or h.get("current_batch") != s["batches"]:
            raise ValueError("real OpenMC HDF5 eigenvalue/seed integrity mismatch")
        if len(entry.get("statepoint_sha256","")) != 64:
            raise ValueError("physical HDF5 statepoint SHA256 missing")
    provenances=[]
    for row in rows:
        seed=row["seed"]
        p=json.loads((root/f"seed_{seed}"/"transport_provenance.json").read_text())
        if p.get("config",{}).get("seed") != seed or p.get("assembly") != "core":
            raise ValueError("physical core provenance mismatch")
        if p.get("config",{}).get("thermal_scattering_treatment") != mode:
            raise ValueError("reported mode disagrees with actual transport provenance")
        provenances.append(p)
        a=json.loads((root/f"seed_{seed}"/"thermal_scattering_physics_audit.json").read_text())
        if mode=="evaluated-bound":
            if not a.get("ready") or not all(x.get("sha256") for x in a.get("tables",[])):
                raise ValueError("bound HDF5 evaluated S(alpha,beta) not verified")
        else:
            if a.get("treatment") != "free-gas":
                raise ValueError("free-gas control has unexpected bound scattering")
        b=json.loads((root/f"seed_{seed}"/"nuclear_data_temperature_audit.json").read_text())
        if not b.get("model_explicit_temperature_coverage_ok"):
            raise ValueError("real neutron HDF5 nearest temperatures outside tolerance")
    return s,provenances


def _without_bound_tables(xml: Path) -> list[tuple[str,str]]:
    """Compare every material atom density and temperature, ignoring ONLY S(a,b)."""
    root = ET.parse(xml).getroot()
    records=[]
    for material in root.findall("material"):
        element=ET.fromstring(ET.tostring(material))
        for sab in list(element.findall("sab")):
            element.remove(sab)
        records.append((element.attrib.get("id",""),ET.tostring(element,encoding="unicode")))
    return sorted(records)


def compare(free_root: Path, bound_root: Path) -> dict:
    free_root=Path(free_root)
    bound_root=Path(bound_root)
    f,ps_f=_load(free_root,"free-gas")
    b,ps_b=_load(bound_root,"evaluated-bound")
    if len({v["seed"] for v in f["runs"]+b["runs"]}) != len(f["runs"])+len(b["runs"]):
        raise ValueError("free and bound ensembles reuse RNG seeds")
    for field in ("rings","particles_per_generation","batches","inactive_batches",
                  "hydrogen_temperature_K","bound_solid_temperature_K",
                  "bound_nearest_temperature_tolerance_K","cross_sections_xml_sha256"):
        if f[field]!=b[field]:
            raise ValueError(f"physical/sampling input differs: {field}")
    keys=("assembly","geometry.xml","tallies.xml")
    base=ps_f[0]
    physics_settings={"seed","thermal_scattering_treatment"}
    for group in (ps_f,ps_b):
        for row in group:
            if row.get("assembly") != base.get("assembly"):
                raise ValueError("core geometry assembly differs")
            for name in ("geometry.xml","tallies.xml"):
                if row.get("xml_sha256",{}).get(name) != base.get("xml_sha256",{}).get(name):
                    raise ValueError(f"OpenMC {name} differs across physical control")
            if row.get("nuclear_data",{}).get("cross_sections_sha256") != (
                base.get("nuclear_data",{}).get("cross_sections_sha256")
            ):
                raise ValueError("evaluated neutron/thermal library hash differs")
            for key,value in base["config"].items():
                if key not in physics_settings and row["config"].get(key)!=value:
                    raise ValueError(f"neutron physics control differs: {key}")
    reference=_without_bound_tables(free_root/f"seed_{f['runs'][0]['seed']}"/"materials.xml")
    for label,root,records in (("free",free_root,f["runs"]),("bound",bound_root,b["runs"])):
        for record in records:
            actual=_without_bound_tables(root/f"seed_{record['seed']}"/"materials.xml")
            if actual!=reference:
                raise ValueError(f"{label} materials differ in atom fractions/density/temp")
    ef=f["run_statistics"]
    eb=b["run_statistics"]
    kf=float(ef["inverse_variance_mean_keff"])
    kb=float(eb["inverse_variance_mean_keff"])
    sf=float(ef["dispersion_inflated_mean_keff_error"])
    sb=float(eb["dispersion_inflated_mean_keff_error"])
    if not all(math.isfinite(x) and x>0 for x in (kf,kb,sf,sb)):
        raise ValueError("actual OpenMC eigenvalue statistics are invalid")
    delta=kb-kf
    return {
        "comparison":"ACTUAL_OPENMC_BOUND_VERSUS_FREE_NEUTRON_SCATTERING",
        "evaluated_xs_index_sha256": f["cross_sections_xml_sha256"],
        "identical_material_atom_density_geometry_and_temperature":True,
        "same_evaluated_neutron_and_thermal_library":True,
        "free_gas_keff":kf,
        "bound_thermal_keff":kb,
        "free_gas_dispersion_inflated_se":sf,
        "bound_thermal_dispersion_inflated_se":sb,
        "observed_keff_difference":delta,
        "combined_independent_ensemble_reported_se":math.hypot(sf,sb),
        "observed_difference_over_combined_reported_se":delta/math.hypot(sf,sb),
        "evaluated_bound_solid_temperature_K":b["bound_solid_temperature_K"],
        "nearest_nuclear_temperature_tolerance_K":b["bound_nearest_temperature_tolerance_K"],
        "different_independent_seeds":True,
        "real_openmc_neutron_transport_executed":True,
        "source_convergence_independently_demonstrated":False,
        "historical_nasa_nerva_physical_validation":False,
        "interpretation":(
            "Measured impact of attaching S(alpha,beta) to Be, graphite, ZrH "
            "in a reduced cold conceptual 3-ring core, keeping non-scattering "
            "material compositions/temperatures identical. Near-temperature "
            "OpenMC nearest treatment was explicitly configured/audited; "
            "this neither establishes source stationarity nor high-T reactor "
            "criticality nor NASA historical benchmark agreement."
        ),
    }


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--free",type=Path,required=True)
    p.add_argument("--bound",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=compare(a.free,a.bound)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("ACTUAL OpenMC evaluated S(a,b) scattering delta keff:",
          result["observed_keff_difference"])
    print("  error-normalized:",
          result["observed_difference_over_combined_reported_se"])
    print("  fission-source convergence validated: NO; NASA core validated: NO")
    print("  evidence:",a.output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
