"""Real OpenMC core fission neutron yield versus incident-neutron energy.

This is NOT a criticality or power-design tool. It independently checks
physics scored by the evaluated neutron library: the ratio of tracklength
nu-fission to fission events (effective neutron yield per fission) for the
declared fuel in each incident-neutron energy bin and each DISJOINT active
generation window of four genuine OpenMC C++ eigenvalue runs.

OpenMC material and material x incident-energy flux, fission, nu-fission,
and absorption are reconciled with unfiltered global nu-fission and the
separate SHA256-authenticated reaction history. The ratio is a tally-derived
flux-weighted yield, not a newly evaluated microscopic cross-section,
measured NERVA performance or independent-fission source stationarity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .core_reaction_history import audit_real_reaction_windows
from .core_reaction_partition import FUEL_MATERIAL_NAME

SCORES = ("flux", "absorption", "fission", "nu-fission")
MIN_PHYSICAL_NU = 1.5
MAX_PHYSICAL_NU = 5.5


def _sha(path: Path) -> str:
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1048576), b""):
            d.update(chunk)
    return d.hexdigest()


def _txt(value) -> str:
    if not isinstance(value, bytes):
        raise ValueError("invalid native OpenMC HDF5 physical tally label")
    return value.decode("utf-8").rstrip("\x00")


def _tally(tallies, name: str, estimator: str, filters: tuple[str,...],
           active: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    items=[o for key,o in tallies.items()
           if key.startswith("tally ") and isinstance(o,h5py.Group)
           and "name" in o and _txt(o["name"][()])==name]
    if len(items)!=1:
        raise ValueError("missing or duplicated genuine neutron reaction tally: "+name)
    t=items[0]
    if (_txt(t["estimator"][()])!=estimator
            or int(t["n_realizations"][()])!=active
            or int(t["n_filters"][()])!=len(filters)):
        raise ValueError("wrong OpenMC neutron estimator or active tally realization count")
    ids=list(map(int,np.atleast_1d(t["filters"][()])))
    if not filters:
        ids=[]
    if len(ids)!=len(filters):
        raise ValueError("unexpected physical material and incident-energy filters")
    bins=[]
    for ident,kind in zip(ids,filters):
        f=tallies["filters"][f"filter {ident}"]
        if _txt(f["type"][()])!=kind:
            raise ValueError("OpenMC source neutron filter type changed")
        bins.append(np.asarray(f["bins"][()]))
    names=tuple(_txt(x) for x in t["score_bins"][()])
    if (len(set(names))!=len(names)
            or any(score not in names for score in
                   (("nu-fission",) if not filters else SCORES))):
        raise ValueError("neutron flux/fission/nu-fission score missing")
    values=np.asarray(t["results"][()],dtype=float)
    if (values.ndim!=3 or values.shape[1:]!=(len(names),2)
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError("invalid physical neutron tally cumulative moments")
    selected = SCORES if filters else ("nu-fission",)
    return values[:,[names.index(s) for s in selected],0],bins,values[:,:,1]


def read_fission_yield_checkpoint(path: Path) -> dict:
    """Actual HDF5 cumulative tally sums and physical material/energy labels."""
    path=Path(path)
    try:
        with h5py.File(path,"r") as h:
            if (_txt(h.attrs["filetype"])!="statepoint"
                    or _txt(h["run_mode"][()])!="eigenvalue"
                    or int(h.attrs["photon_transport"])!=0):
                raise ValueError("fission yield requires real neutron-only OpenMC eigenvalue statepoint")
            seed=int(h["seed"][()])
            batch=int(h["current_batch"][()])
            inactive=int(h["n_inactive"][()])
            particles=int(h["n_particles"][()])
            active=int(h["n_realizations"][()])
            if (seed<=0 or particles<100 or active<2
                    or batch-inactive!=active):
                raise ValueError("invalid OpenMC source generation population/realizations")
            group=h["tallies"]
            global_t,_,_= _tally(
                group,"nerva_global_reaction_tracklength","tracklength",(),active)
            mat,mfilter,_= _tally(
                group,"nerva_material_transport","tracklength",
                ("material",),active)
            mat_e,efilter,_= _tally(
                group,"nerva_material_energy_transport","tracklength",
                ("material","energy"),active)
            mids=np.asarray(mfilter[0],dtype=int)
            emids=np.asarray(efilter[0],dtype=int)
            edges=np.asarray(efilter[1],dtype=float)
            if (len(mids)<2 or len(set(mids))!=len(mids)
                    or not np.array_equal(mids,emids)):
                raise ValueError("material neutron filter IDs do not conserve across scores")
            if (edges.ndim!=1 or len(edges)<4
                    or not np.isfinite(edges).all()
                    or np.any(edges[1:]<=edges[:-1])
                    or edges[0]<0
                    or not np.any(np.isclose(edges,.625,rtol=0,atol=1e-12))
                    or not np.any(np.isclose(edges,1e5,rtol=1e-13,atol=0))):
                raise ValueError("unqualified incident-neutron energy bins")
            ng=len(edges)-1
            if (mat.shape!=(len(mids),len(SCORES))
                    or mat_e.shape!=(len(mids)*ng,len(SCORES))
                    or global_t.shape!=(1,1)):
                raise ValueError("wrong OpenMC material-energy fission scoring dimensions")
            mat_e=mat_e.reshape(len(mids),ng,len(SCORES))
    except (OSError,KeyError,TypeError) as e:
        raise ValueError("incomplete genuine fission-yield neutron HDF5") from e
    return {
        "seed":seed,"batch":batch,"inactive":inactive,"active":active,
        "particles":particles,"sha256":_sha(path),
        "material_ids":mids.tolist(),"energy_edges_ev":edges.tolist(),
        "global_nufission":float(global_t[0,0]),
        "material":mat,"material_energy":mat_e,
    }


def _fuel_material_id(materials_xml: Path) -> int:
    root=ET.parse(materials_xml).getroot()
    if root.tag!="materials":
        raise ValueError("no actual OpenMC physical materials.xml")
    rows=[int(e.attrib["id"]) for e in root.findall("material")
          if e.attrib.get("name")==FUEL_MATERIAL_NAME]
    if len(rows)!=1:
        raise ValueError("declared conceptual fuel material not uniquely identifiable")
    return rows[0]


def calculate_fission_yield_windows(
    checkpoints: list[dict],fuel_id: int, *,
    production_tolerance: float = 1e-9,
) -> dict:
    """Recover separate physical neutron energy-group ν per fission histories."""
    if len(checkpoints)<3:
        raise ValueError("at least three actual active-generation checkpoints")
    ref=checkpoints[0]
    ids=ref["material_ids"]
    edges=ref["energy_edges_ev"]
    if fuel_id not in ids:
        raise ValueError("OpenMC fissionable fuel missing from material filter")
    fuel_idx=ids.index(fuel_id)
    prior=np.zeros_like(ref["material_energy"])
    old_mat=np.zeros_like(ref["material"])
    old_nuf=0.
    old_active=0
    windows=[]
    for row in checkpoints:
        if (row["seed"]!=ref["seed"] or row["inactive"]!=ref["inactive"]
                or row["particles"]!=ref["particles"]
                or row["material_ids"]!=ids or row["energy_edges_ev"]!=edges):
            raise ValueError("source seed, evaluated neutron filter or reactor core changed")
        active=row["active"]
        count=active-old_active
        if count<2 or row["batch"]!=row["inactive"]+active:
            raise ValueError("invalid disjoint neutron generation window")
        energy=(row["material_energy"]-prior)/count
        materials=(row["material"]-old_mat)/count
        global_nuf=(row["global_nufission"]-old_nuf)/count
        if (not np.isfinite(energy).all() or np.any(energy< -1e-10)
                or np.any(materials< -1e-10) or global_nuf<=0):
            raise ValueError("negative or zero disjoint fission/absorption physical score")
        if not np.allclose(energy.sum(axis=1),materials,rtol=0,atol=production_tolerance):
            raise ValueError("energy-filtered fission production does not conserve material")
        if not math.isclose(float(energy[:,:,3].sum()),global_nuf,
                            rel_tol=0,abs_tol=production_tolerance):
            raise ValueError("incident-energy fission production differs from global tally")
        nuf_by_mat=energy[:,:,3].sum(axis=1)
        fission_by_mat=energy[:,:,2].sum(axis=1)
        total_nuf=float(nuf_by_mat.sum())
        total_fissions=float(fission_by_mat.sum())
        if (total_fissions<=0 or
                (total_nuf-nuf_by_mat[fuel_idx])>max(1e-12,1e-6*total_nuf)
                or (total_fissions-fission_by_mat[fuel_idx])>max(1e-12,1e-6*total_fissions)):
            raise ValueError("neutron fission outside the declared fuel material")
        fuel=energy[fuel_idx]
        fission=fuel[:,2]
        nu=fuel[:,3]
        # Require a yield for energy bins with measurable fission weight;
        # low-rate bins are retained without asserting independently precise ν.
        active_bins=fission>max(1e-12,1e-5*total_fissions)
        if np.any((nu > 1e-12)&(fission <= 0)):
            raise ValueError("positive neutron fission yield without fission events")
        yield_ev=np.divide(nu,fission,out=np.full(len(fission),np.nan),
                           where=fission>0)
        if (np.any(~np.isfinite(yield_ev[active_bins]))
                or np.any(yield_ev[active_bins]<MIN_PHYSICAL_NU)
                or np.any(yield_ev[active_bins]>MAX_PHYSICAL_NU)):
            raise ValueError("evaluated neutron yield per fission outside physical guard")
        group_abs=energy[:,:,1].sum(axis=0)
        group_flux=energy[:,:,0].sum(axis=0)
        if np.any(group_abs < -1e-12) or np.any(group_flux < -1e-12):
            raise ValueError("negative incident neutron absorption or flux")
        denom=group_abs
        reproduction=np.divide(energy[:,:,3].sum(axis=0),denom,
                              out=np.full(len(fission),np.nan),where=denom>0)
        group_bounds=np.asarray(edges)
        def category(lo,hi):
            subset=np.where((group_bounds[:-1]>=lo)&(group_bounds[1:]<=hi))[0]
            if len(subset)==0:
                raise ValueError("missing thermal/epithermal/fast neutron energy window")
            group_fission=fission[subset].sum()
            group_nu=nu[subset].sum()
            return {
                "fission_per_source":float(group_fission),
                "nu_fission_per_source":float(group_nu),
                "yield_neutrons_per_fission":(
                    float(group_nu/group_fission) if group_fission>0 else None
                ),
                "absorption_per_source":float(group_abs[subset].sum()),
                "flux_per_source_cm":float(group_flux[subset].sum()),
            }
        windows.append({
            "first_active_batch":row["batch"]-count+1,
            "last_active_batch":row["batch"],
            "active_generations":count,
            "total_fission_events_per_source":total_fissions,
            "total_nu_fission_neutrons_per_source":total_nuf,
            "effective_neutrons_per_fission":total_nuf/total_fissions,
            "fission_event_rates_by_incident_energy_ev":fission.tolist(),
            "nu_fission_rates_by_incident_energy_ev":nu.tolist(),
            "neutron_yield_by_incident_energy_ev":[
                float(x) if math.isfinite(x) else None for x in yield_ev],
            "nu_yield_physically_covered_energy_bin":active_bins.tolist(),
            "whole_core_absorption_by_incident_energy_ev":group_abs.tolist(),
            "whole_core_flux_by_incident_energy_ev_cm":group_flux.tolist(),
            "nu_fission_per_absorbed_neutron_by_incident_energy_ev":[
                float(x) if math.isfinite(x) else None for x in reproduction],
            "thermal":category(0,.625),
            "epithermal":category(.625,1e5),
            "fast":category(1e5,float(edges[-1])),
            "statepoint_sha256":row["sha256"],
            "fuel_origin_and_group_reaction_conservation_passed":True,
        })
        prior=row["material_energy"]
        old_mat=row["material"]
        old_nuf=row["global_nufission"]
        old_active=active
    weights=np.asarray([w["active_generations"] for w in windows])
    total_fission=sum(w["total_fission_events_per_source"]*n for w,n in zip(windows,weights))
    total_nuf=sum(w["total_nu_fission_neutrons_per_source"]*n for w,n in zip(windows,weights))
    if (not math.isclose(total_nuf/old_active,checkpoints[-1]["global_nufission"]/old_active,
                        rel_tol=0,abs_tol=5e-12)):
        raise ValueError("disjoint neutron fission source reconstruction failed")
    return {
        "seed":ref["seed"],"material_ids":ids,"fuel_material_id":fuel_id,
        "incident_neutron_energy_edges_eV":edges,
        "effective_ensemble_nu_per_fission":total_nuf/total_fission,
        "full_run_fissions_per_source":total_fission/old_active,
        "full_run_fission_neutrons_per_source":total_nuf/old_active,
        "disjoint_windows":windows,
        "energy_resolved_fission_yield_physics_checks_passed":True,
    }


def audit_real_energy_fission_yield(summary_path: Path) -> dict:
    path=Path(summary_path)
    primary=audit_real_reaction_windows(path)
    if not primary["all_reaction_window_partition_and_reconstruction_checks_passed"]:
        raise ValueError("native OpenMC tracklength reaction provenance failed")
    summary=json.loads(path.read_text(encoding="utf-8"))
    digits=len(str(summary["batches"]))
    stages=summary["saved_source_bank_batches"]
    runs=[]
    for run in summary["runs"]:
        last=Path(run["statepoint_path"])
        if not last.is_file() or _sha(last)!=run["statepoint_sha256"]:
            raise ValueError("genuine final HDF5 or source SHA256 mismatched")
        fuel_id=_fuel_material_id(last.parent/"materials.xml")
        checkpoints=[
            read_fission_yield_checkpoint(last.parent/f"statepoint.{b:0{digits}d}.h5")
            for b in stages
        ]
        if any(c["seed"]!=run["seed"] or c["batch"]!=b
               for c,b in zip(checkpoints,stages)):
            raise ValueError("OpenMC physical fuel source seed or checkpoint mismatch")
        audited=calculate_fission_yield_windows(checkpoints,fuel_id)
        reference=next((q for q in primary["observed_seed_history"]
                        if q["seed"]==run["seed"]),None)
        if reference is None:
            raise ValueError("fission neutron source is missing from reaction history")
        for actual,expected in zip(audited["disjoint_windows"],reference["windows"]):
            if (actual["last_active_batch"]!=expected["last_active_batch"]
                    or not math.isclose(
                        actual["total_nu_fission_neutrons_per_source"],
                        expected["global_reaction_rates_per_source"]["nu-fission"],
                        abs_tol=3e-12,rel_tol=0)):
                raise ValueError("energy-resolved fission yield differs from genuine neutron production")
        runs.append(audited)
    if len(runs)<4 or len({q["seed"] for q in runs})!=len(runs):
        raise ValueError("requires >=4 genuine independent OpenMC source seeds")
    return {
        "analysis":"ACTUAL_OPENMC_EVALUATED_INCIDENT_NEUTRON_ENERGY_FISSION_YIELD",
        "real_transport_summary_sha256":_sha(path),
        "evaluated_nuclear_data_index_sha256":summary["cross_sections_xml_sha256"],
        "genuine_independent_seed_count":len(runs),
        "saved_source_batches":stages,
        "per_seed_fission_yield_physics":runs,
        "mean_effective_neutrons_per_fission":float(np.mean(
            [q["effective_ensemble_nu_per_fission"] for q in runs])),
        "all_energy_resolved_fission_physics_checks_passed":True,
        "nuclear_evaluation_and_nasa_benchmark_validated":False,
        "fission_source_independence_or_stationarity_proven":False,
        "interpretation":(
            "Genuine OpenMC continuous-energy evaluated-neutron fission and "
            "nu-fission tracklength scores in fuel and material by incident "
            "neutron energy. The yield nu-bar is the flux-weighted ratio "
            "of scored neutron production to scored fission events, and "
            "is NOT an independent library cross-section validation. "
            "Fast-to-thermal yield and absorption describe this cold, strongly "
            "subcritical, source-cloned conceptual core only. No NASA "
            "NERVA criticality, materials benchmark, hot-core heating "
            "qualification or independent fission ancestry is implied."
        ),
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=audit_real_energy_fission_yield(a.summary)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("Genuine OpenMC evaluated fission yield energy accounting: PASS")
    for q in result["per_seed_fission_yield_physics"]:
        win=q["disjoint_windows"]
        print("seed",q["seed"],"mean nu/fission",
              q["effective_ensemble_nu_per_fission"],
              "latest thermal",win[-1]["thermal"]["yield_neutrons_per_fission"],
              "latest fast",win[-1]["fast"]["yield_neutrons_per_fission"])
    print("Historical NASA NERVA reactor fidelity and source convergence: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
