"""Read-only physical neutron-energy partition qualification from actual OpenMC HDF5.

Eight genuine neutron-transport statepoints: fission daughters OFF/ON, four
independent seeds. Requires independent mesh, unfiltered material, energy-
filtered material, and nested fuel/hydrogen reaction integrals to coincide.
No assumed interpolation, fake hot cross sections, power normalization, or
criticality inference. An energy filter extending 1e-5 to 2e7 eV does NOT
automatically cover out-of-range transported particles: prove coverage by
the independent unfiltered scores at the executed source histories.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .neutron_energy_groups import ENERGY_LIMITS_EV, spectral_fractions
from .fixed_source_spatial_response import (
    read_actual_spatial_statepoint, _sha, _label,
)

NEUTRON_SCORES = ("flux", "absorption", "fission", "nu-fission")
CLOSE_RTOL = 1e-8


def _tally(tallies, name: str, active: int):
    hits = [v for k,v in tallies.items()
            if k.startswith("tally ") and isinstance(v,h5py.Group)
            and "name" in v and _label(v["name"][()]) == name]
    if len(hits) != 1:
        raise ValueError("missing or duplicate genuine OpenMC neutron tally: " + name)
    tally = hits[0]
    labels = [_label(x) for x in tally["score_bins"][()]]
    if (len(labels) != len(set(labels)) or
            _label(tally["estimator"][()]) != "tracklength" or
            int(tally["n_realizations"][()]) != active or
            [_label(x) for x in tally["nuclides"][()]] != ["total"]):
        raise ValueError("native neutron tally score, estimator or batch mismatch")
    data = np.asarray(tally["results"][()], dtype=np.float64)
    if (data.ndim != 3 or data.shape[1:] != (len(labels),2) or
            not np.isfinite(data).all()):
        raise ValueError("invalid genuine neutron energy tally moments")
    return tally, labels, data[:,:,0]/active


def _filters(tallies, tally, types):
    ids = np.asarray(tally["filters"][()],dtype=int).reshape(-1)
    if len(ids) != len(types) or int(tally["n_filters"][()]) != len(types):
        raise ValueError("wrong native neutron energy filter count")
    fs = [tallies["filters"]["filter "+str(int(i))] for i in ids]
    if [_label(f["type"][()]) for f in fs] != list(types):
        raise ValueError("wrong physical neutron energy filter order")
    return fs


def _equal(x, y, reason: str, scale: float = 1.0) -> None:
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    if (x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.all(np.abs(x-y) <= CLOSE_RTOL * np.maximum(scale,np.maximum(abs(x),abs(y))))):
        raise ValueError("real neutron energy conservation failure: "+reason)


def inspect_energy_statepoint(path: Path, *, fission_daughters: bool) -> dict:
    path=Path(path)
    spatial=read_actual_spatial_statepoint(path, fission_daughters)
    materials_xml=ET.parse(path.parent/"materials.xml").getroot()
    if materials_xml.tag != "materials":
        raise ValueError("actual source material inventory absent")
    known = {int(x.attrib["id"]):x.attrib.get("name","") for x in
             materials_xml.findall("material")}
    if len(known) < 2 or len(known) != len(set(known)):
        raise ValueError("invalid actual OpenMC material inventory")
    with h5py.File(path,"r") as state:
        tallies=state["tallies"]
        n=spatial["batches"]
        mat,mat_labels,mat_score=_tally(tallies,"nerva_material_transport",n)
        en,en_labels,en_score=_tally(tallies,"nerva_material_energy_transport",n)
        mat_filter=_filters(tallies,mat,("material",))[0]
        mat_id=np.asarray(mat_filter["bins"][()],dtype=int).reshape(-1)
        mat_f,en_f=_filters(tallies,en,("material","energy"))
        energy_id=np.asarray(mat_f["bins"][()],dtype=int).reshape(-1)
        edges=np.asarray(en_f["bins"][()],dtype=np.float64).reshape(-1)
        groups=len(edges)-1
        if (len(mat_id)!=len(known) or set(mat_id)!=set(known)
                or not np.array_equal(energy_id,mat_id)
                or groups < 3 or len(np.unique(edges))!=len(edges)
                or not np.isfinite(edges).all() or np.any(np.diff(edges)<=0)
                or not np.allclose([edges[0],edges[-1]],
                                   [ENERGY_LIMITS_EV[0],ENERGY_LIMITS_EV[-1]],atol=1e-12,rtol=1e-13)
                or any(not np.any(np.isclose(edges,x,rtol=1e-12,atol=0)) for x in ENERGY_LIMITS_EV[1:3])
                or en_score.shape[0]!=len(mat_id)*groups
                or mat_score.shape[0]!=len(mat_id)):
            raise ValueError("wrong executed material or energy-bin provenance")
        if (any(s not in mat_labels for s in NEUTRON_SCORES)
                or any(s not in en_labels for s in (*NEUTRON_SCORES,"heating-local"))):
            raise ValueError("missing actual neutron flux, absorption and fission scores")
        energy = en_score.reshape(len(mat_id),groups,len(en_labels))
        # Distinguish rare genuine out-of-window events from arithmetic
        # roundoff. OpenMC energy filters cover a FINITE 1e-5..2e7 eV
        # domain, whereas the material-only tally has no energy cut.
        # A small closure residual can occur in a real history (observed in
        # the Be reflector for seed 33013), and MUST be exposed.
        truncated = []
        max_fraction_unassigned = 0.0
        for score in (*NEUTRON_SCORES,"heating-local"):
            if score not in mat_labels: raise ValueError("unfiltered score missing: "+score)
            comparison=energy[:,:,en_labels.index(score)].sum(axis=1)
            baseline=mat_score[:,mat_labels.index(score)]
            _equal(comparison, baseline, "incident energy coverage for "+score)
            residual = baseline - comparison
            for k, missing in enumerate(residual):
                if abs(missing) > 1e-11:
                    fraction = abs(float(missing)) / max(abs(float(baseline[k])),1e-15)
                    max_fraction_unassigned = max(max_fraction_unassigned, fraction)
                    truncated.append({
                        "material_id":int(mat_id[k]),
                        "material_name":known[int(mat_id[k])],
                        "reaction_score":score,
                        "unassigned_score_per_external_source":float(missing),
                        "fraction_of_unfiltered_material_score":fraction,
                    })
        grouped = {s:energy[:,:,en_labels.index(s)] for s in NEUTRON_SCORES}
        if any(np.any(x < -1e-10) for x in grouped.values()):
            raise ValueError("nonphysical negative neutron tracklength or reaction intensity")
        for score in ("flux","fission","nu-fission"):
            full=float(grouped[score].sum())
            reference=spatial["scores"][score]["axial_and_3d_integral_per_external_source"]
            _equal(np.array([full]),np.array([reference]),"whole-core mesh vs neutron energy "+score)
        specials = {}
        for name,expected in (("nerva_fuel_spectrum","conceptual (U,Zr)C-graphite surrogate"),
                              ("nerva_hydrogen_spectrum","hydrogen coolant")):
            t,labels,values=_tally(tallies,name,n)
            f0,f1=_filters(tallies,t,("material","energy"))
            mid=np.asarray(f0["bins"][()],dtype=int).reshape(-1)
            other_edges=np.asarray(f1["bins"][()],dtype=float).reshape(-1)
            matches=[i for i,k in enumerate(mat_id) if known[int(k)]==expected]
            if (len(matches)!=1 or mid.size!=1 or int(mid[0])!=int(mat_id[matches[0]])
                    or not np.array_equal(edges,other_edges)
                    or values.shape[0]!=groups):
                raise ValueError("nested physical fuel/hydrogen incident neutron spectrum mismatch")
            for score in labels:
                if score not in en_labels:raise ValueError("unmatched nested neutron score")
                _equal(values[:,labels.index(score)],
                       energy[matches[0],:,en_labels.index(score)],
                       "material energy vs nested "+name+" "+score)
            spectra=grouped["flux"][matches[0]]
            specials["fuel" if name=="nerva_fuel_spectrum" else "hydrogen"]={
                "material_id":int(mid[0]),
                "flux_integral_per_external_source":float(spectra.sum()),
                "incident_energy_spectrum_fractions":
                    spectral_fractions(np.column_stack((edges[:-1],edges[1:])),spectra),
            }
        groups_all=grouped["flux"].sum(axis=0)
        spectrum=spectral_fractions(np.column_stack((edges[:-1],edges[1:])),groups_all)
        if not math.isclose(sum(spectrum.values()),1.0,rel_tol=0,abs_tol=1e-12):
            raise ValueError("neutron spectral fractions fail exact partition")
        return {
            "seed":spatial["seed"],
            "fission_daughters_transported":fission_daughters,
            "statepoint_sha256":spatial["statepoint_sha256"],
            "physical_xml_sha256":{"geometry":spatial["geometry_sha256"],
                "materials":spatial["materials_sha256"],"tallies":spatial["tallies_sha256"]},
            "material_count":len(mat_id),"incident_energy_groups":groups,
            "energy_edges_ev":edges.tolist(),
            "all_materials_energy_integral_closes_unfiltered_scores":True,
            "nested_fuel_hydrogen_spectra_match":True,
            "spatial_mesh_and_spectral_neutron_scores_close":True,
            "incident_energy_window_audited_against_unfiltered_tallies":True,
            "exact_energy_window_coverage_demonstrated":len(truncated)==0,
            "observed_out_of_window_or_nonroundoff_score_residuals":truncated,
            "maximum_material_relative_unassigned_reaction_score":max_fraction_unassigned,
            "unassigned_energy_above_or_below_window_individually_identified":False,
            "whole_core_integral_flux_per_external_source":float(groups_all.sum()),
            "whole_core_incident_energy_flux_fractions":spectrum,
            "fuel_hydrogen":specials,
            "local_neutron_heating_evaluated_and_usable":False,
        }


def audit_eight_statepoint_energy(root: Path, *, batches: int=30,
                        seeds=(13007,23011,33013,43019)) -> dict:
    root=Path(root)
    if len(seeds)!=4 or len(set(seeds))!=4 or any(type(s) is not int or s<=0 for s in seeds):
        raise ValueError("independent physical four-seed spectrum ensemble required")
    rows=[]
    for seed in seeds:
        result={}
        for enabled,folder in [(False,"suppressed_fission_control"),
                               (True,"fission_daughters_transported")]:
            p=root/folder/f"seed_{seed}"/f"statepoint.{batches}.h5"
            v=inspect_energy_statepoint(p,fission_daughters=enabled)
            if v["seed"]!=seed:raise ValueError("real neutron spectrum seed provenance mismatch")
            result["on" if enabled else "off"]=v
        if result["on"]["physical_xml_sha256"]!=result["off"]["physical_xml_sha256"]:
            raise ValueError("material/geometry/tally neutron spectrum physics changed")
        rows.append({"seed":seed,**result})
    measurements={}
    for key in ("thermal_below_0p625_eV","epithermal_0p625eV_to_100keV","fast_above_100keV"):
        off=np.array([r["off"]["whole_core_incident_energy_flux_fractions"][key] for r in rows])
        on=np.array([r["on"]["whole_core_incident_energy_flux_fractions"][key] for r in rows])
        change=on-off
        measurements[key]={
            "fission_off_fraction_mean":float(off.mean()),
            "fission_on_fraction_mean":float(on.mean()),
            "paired_four_seed_fraction_change_mean":float(change.mean()),
            "paired_seed_sem":float(change.std(ddof=1)/2),
        }
    return {
        "audit":"ACTUAL_OPENMC_MATERIAL_ENERGY_FLUX_AND_NEUTRON_REACTION_CONSERVATION",
        "independent_seed_pairs":4,
        "genuine_statepoint_count":8,
        "material_energy_unfiltered_and_3d_spatial_integral_closure":True,
        "spectral_range_not_assumed_to_be_exhaustive":True,
        "maximum_observed_unassigned_material_score_fraction":max(
            r[mode]["maximum_material_relative_unassigned_reaction_score"]
            for r in rows for mode in ("off","on")
        ),
        "real_materials_with_nonroundoff_out_of_window_scores":[
            {"seed":r["seed"],"mode":mode,
             "residuals":r[mode]["observed_out_of_window_or_nonroundoff_score_residuals"]}
            for r in rows for mode in ("off","on")
            if r[mode]["observed_out_of_window_or_nonroundoff_score_residuals"]
        ],
        "incident_neutron_energy_group_boundaries_ev":list(ENERGY_LIMITS_EV),
        "flux_spectrum_fractions_by_mode":measurements,
        "per_seed":rows,
        "nuclear_heating_physically_qualified":False,
        "criticality_or_source_stationarity_validated":False,
        "historical_NASA_NERVA_operating_neutronics_validated":False,
        "interpretation":(
            "These are incident-energy tracklength neutron spectra per injected "
            "external neutron, not normalized reactor flux or temperature. "
            "Independent unfiltered tallies QUANTIFY residuals against "
            "the finite incident-energy window, including tiny nonroundoff "
            "differences. Their cause and energy-side location are not "
            "established by this audit; exhaustive coverage is not claimed. "
            "Spectral changes do not qualify operating-temperature data, "
            "eigenvalue fission source, or local heating."),
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    opts=parser.parse_args()
    result=audit_eight_statepoint_energy(opts.root)
    opts.output.parent.mkdir(parents=True,exist_ok=True)
    opts.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("REAL OpenMC 8-statepoint material energy flux/reaction conservation: PASS")
    print("Incident-energy fractions:",result["flux_spectrum_fractions_by_mode"])
    print("Criticality, hot XS, and local heating: NOT QUALIFIED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
