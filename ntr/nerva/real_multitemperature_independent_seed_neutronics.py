"""Stage48 original evaluated-multi-kT OpenMC independent-seed source QA.

Four independent source RNG seeds at each evaluated-XS material temperature,
750 K (stochastic 600/900 K kT tables) and 900 K (evaluated endpoint).
Reads only actual original 60/100 OpenMC C++ HDF5 statepoints. Comparisons
do NOT qualify a historical NERVA core, stationary source, physical watts,
or a temperature-reactivity coefficient. No material/fuel geometry tuning.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .config import NervaConfig
from .model import build_model
from .source_bank_convergence import read_source_bank

TEMPS=(750,900)
SEEDS=(13007,23011,33013,43019)
CHECKPOINTS=(60,100)
PARTICLES=3000
N_INACTIVE=20
N_BATCHES=100
BINS=(4,4,10)
T3_95_TWO_SIDED=3.182446305284263  # exact t_{0.975,3}


def _sha(path:Path)->str:
    if not path.is_file() or not path.stat().st_size:
        raise FileNotFoundError("original evaluated-data OpenMC artifact missing: "+str(path))
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def original_native_history(file:Path, *, seed:int, batch:int)->dict:
    """Verify source-bank and generation history belong to real eigenvalue C++."""
    if seed not in SEEDS or batch not in CHECKPOINTS:
        raise ValueError("invalid independent seed or native generation checkpoint")
    _sha(file)
    with h5py.File(file,"r") as h:
        if (bytes(h.attrs.get("filetype",b"")).rstrip(b"\0")!=b"statepoint"
            or bytes(h["run_mode"][()]).rstrip(b"\0")!=b"eigenvalue"
            or int(h.attrs.get("source_present",0))!=1
            or int(h["seed"][()])!=seed or int(h["current_batch"][()])!=batch
            or int(h["n_batches"][()])!=N_BATCHES
            or int(h["n_inactive"][()])!=N_INACTIVE
            or int(h["n_particles"][()])!=PARTICLES
            or int(h["generations_per_batch"][()])!=1
            or int(h["n_realizations"][()])!=batch-N_INACTIVE):
            raise ValueError("wrong original evaluated neutron HDF5 source identity")
        k=np.asarray(h["k_generation"][()],dtype=np.float64)
        entropy=np.asarray(h["entropy"][()],dtype=np.float64)
        bank=np.asarray(h["source_bank"][()])
    if (k.shape!=(batch,) or entropy.shape!=(batch,)
            or bank.shape!=(PARTICLES,) or bank.dtype.names is None
            or "r" not in bank.dtype.names
            or not np.isfinite(k).all() or np.any(k<=0)
            or not np.isfinite(entropy).all()):
        raise ValueError("invalid actual C++ k, entropy or 3000-site neutron source")
    return {"k":k,"entropy":entropy,"bank":bank,
            "sha256":_sha(file)}


def independent_seed_summary(means:list[float])->dict:
    """Seed-level descriptive 95% Student-t interval, NOT a stationarity test."""
    a=np.asarray(means,dtype=float)
    if a.shape!=(4,) or not np.isfinite(a).all() or np.any(a<=0):
        raise ValueError("four valid independent OpenMC seed estimates required")
    sem=float(np.std(a,ddof=1)/np.sqrt(len(a)))
    mean=float(np.mean(a))
    half=T3_95_TWO_SIDED*sem
    return {"mean_across_four_original_independent_seeds":mean,
            "sample_sd_across_independent_seeds":float(np.std(a,ddof=1)),
            "empirical_seed_mean_sem":sem,
            "illustrative_t3_95pct_interval":[mean-half,mean+half],
            "independent_seed_count":4,
            "interval_does_not_account_for_unresolved_source_bias":True}


def total_variation(a:np.ndarray,b:np.ndarray)->float:
    a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float)
    if (a.shape!=b.shape or a.size<2
            or np.any(a<0) or np.any(b<0)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or not math.isclose(float(np.sum(a)),1.,abs_tol=1e-12)
            or not math.isclose(float(np.sum(b)),1.,abs_tol=1e-12)):
        raise ValueError("source histograms must be same normalized physical mesh")
    return 0.5*float(np.sum(np.abs(a-b)))


def audit_original_ensemble(root:Path|str, *,
                            nuclear_index:Path|str)->dict:
    root=Path(root)
    nuclear_index=Path(nuclear_index)
    nuclear_sha=_sha(nuclear_index)
    report=[]
    all_geometry_sha=set()
    all_material_temp_inputs={}
    all_banks_sha=set()
    for temp in TEMPS:
        temp_name=f"T{temp}K"
        cfg=replace(NervaConfig(),core_rings=3,
                    fuel_temperature_k=float(temp),
                    hydrogen_temperature_k=float(temp),
                    bound_solid_temperature_k=float(temp),
                    structural_solid_temperature_k=float(temp),
                    neutron_temperature_method="interpolation")
        cfg.validate()
        geom_model=build_model(cfg,assembly="core")
        mesh=geom_model.settings.entropy_mesh
        if mesh is None:
            raise ValueError("actual core neutron entropy source mesh absent")
        bounds_low=tuple(float(x) for x in mesh.lower_left)
        bounds_high=tuple(float(x) for x in mesh.upper_right)
        runs=[]
        late=[]
        means=[]
        geom_sha_for_temperature=set()
        material_sha_for_temperature=set()
        for seed in SEEDS:
            folder=root/temp_name/f"seed_{seed}"
            prov=json.loads((folder/"transport_provenance.json").read_text(encoding="utf-8"))
            conf=prov["config"]
            if (int(conf.get("seed",-1))!=seed
                    or int(conf.get("batches",-1))!=N_BATCHES
                    or int(conf.get("inactive",-1))!=N_INACTIVE
                    or int(conf.get("particles",-1))!=PARTICLES
                    or conf.get("neutron_temperature_method")!="interpolation"
                    or conf.get("initial_fission_source") not in (None,"legacy-broad")
                    or conf.get("photon_transport") is not False
                    or conf.get("statepoint_batches")!=list(CHECKPOINTS)):
                raise ValueError("original independent-seed OpenMC settings inconsistent")
            for key in ("fuel_temperature_k","hydrogen_temperature_k",
                        "bound_solid_temperature_k","structural_solid_temperature_k"):
                if not math.isclose(float(conf[key]),float(temp),rel_tol=0,abs_tol=1e-8):
                    raise ValueError("actual neutron model temperature not controlled")
            if prov["nuclear_data"]["cross_sections_sha256"]!=nuclear_sha:
                raise ValueError("cross-section index changed across independent seeds")
            for xml in ("settings.xml","geometry.xml","materials.xml"):
                if prov["xml_sha256"][xml]!=_sha(folder/xml):
                    raise ValueError("original physical OpenMC XML SHA mismatch")
            settings=ET.parse(folder/"settings.xml").getroot()
            if settings.findtext("temperature_method")!="interpolation":
                raise ValueError("native OpenMC temperature interpolation not requested")
            geom_sha_for_temperature.add(prov["xml_sha256"]["geometry.xml"])
            all_geometry_sha.add(prov["xml_sha256"]["geometry.xml"])
            material_sha_for_temperature.add(prov["xml_sha256"]["materials.xml"])
            data=[]
            hist=[]
            metadata=[]
            for batch in CHECKPOINTS:
                file=folder/f"statepoint.{batch:03d}.h5"
                r=original_native_history(file,seed=seed,batch=batch)
                if batch==100 and prov["statepoint"]["sha256"]!=r["sha256"]:
                    raise ValueError("real final OpenMC statepoint SHA mismatch")
                if r["sha256"] in all_banks_sha:
                    raise ValueError("duplicate HDF5 instead of independent seed source")
                all_banks_sha.add(r["sha256"])
                h,meta=read_source_bank(file,lower_cm=bounds_low,
                                        upper_cm=bounds_high,bins=BINS)
                if (meta["seed"]!=seed or meta["completed_batch"]!=batch
                        or meta["source_sites"]!=PARTICLES):
                    raise ValueError("actual source histogram HDF5 seed mismatch")
                data.append(r)
                hist.append(h)
                metadata.append(meta)
            if (not np.array_equal(data[0]["k"],data[1]["k"][:60])
                    or not np.array_equal(data[0]["entropy"],data[1]["entropy"][:60])):
                raise ValueError("within-original-run neutron k/entropy prefix changed")
            active_mean=float(np.mean(data[1]["k"][N_INACTIVE:]))
            if not math.isfinite(active_mean) or active_mean<=0:
                raise ValueError("invalid original C++ neutron multiplication estimator")
            means.append(active_mean)
            late.append(hist[1])
            runs.append({
                "seed":seed,
                "original_hdf5_sha_60":data[0]["sha256"],
                "original_hdf5_sha_100":data[1]["sha256"],
                "active_21_100_mean_k":active_mean,
                "later_41_100_mean_k_same_run":float(np.mean(data[1]["k"][40:])),
                "late_minus_all_active_same_history_k":
                    float(np.mean(data[1]["k"][40:])-active_mean),
                "source_total_variation_60_to_100":total_variation(hist[0],hist[1]),
                "source_entropy_at_60_and_100":[float(data[i]["entropy"][-1]) for i in range(2)],
                "source_bank_unique_positions_60_and_100":[
                    int(x["exact_distinct_site_positions"]) for x in metadata],
                "inverse_simpson_distinct_positions_100":float(
                    metadata[1]["inverse_simpson_distinct_position_count"]),
                "actual_fixed_source_mesh_bounds_cm":{"lower":bounds_low,"upper":bounds_high},
            })
        if len(geom_sha_for_temperature)!=1 or len(material_sha_for_temperature)!=1:
            raise ValueError("core geometry/materials changed among independent seeds")
        all_material_temp_inputs[temp]=next(iter(material_sha_for_temperature))
        between=[total_variation(late[i],late[j]) for i,j in combinations(range(4),2)]
        report.append({
            "actual_evaluated_neutron_material_temperature_K":temp,
            "actual_nuclear_temperature_method":"interpolation",
            "seed_level_descriptive_k_uncertainty":independent_seed_summary(means),
            "mean_source_TV_60_to_100":float(np.mean([r["source_total_variation_60_to_100"] for r in runs])),
            "mean_pairwise_between_seed_source_TV_at_100":float(np.mean(between)),
            "max_pairwise_between_seed_source_TV_at_100":float(np.max(between)),
            "four_original_Cplusplus_runs":runs,
            "source_stationarity_demonstrated":False,
        })
    if len(all_geometry_sha)!=1:
        raise ValueError("core physical geometry changed between temperature-only runs")
    # Independent-seed observation difference only: source not stationary,
    # histories at different temperatures are not valid CRN pairs.
    a=report[0]["seed_level_descriptive_k_uncertainty"]
    b=report[1]["seed_level_descriptive_k_uncertainty"]
    diff=b["mean_across_four_original_independent_seeds"]-a[
        "mean_across_four_original_independent_seeds"]
    half=T3_95_TWO_SIDED*(a["empirical_seed_mean_sem"]+b["empirical_seed_mean_sem"])
    return {
        "schema":"nuke-stage48-original-multitemperature-independent-seed-source-qa-v1",
        "original_OpenMC_CPP_neutron_eigenvalue_transport_runs":8,
        "actual_original_HDF5_source_statepoints_inspected":16,
        "independent_seed_labels":list(SEEDS),
        "temperature_K":list(TEMPS),
        "actual_evaluated_multitemperature_neutron_index_sha256":nuclear_sha,
        "same_original_geometry_all_eight":True,
        "different_material_temperature_states_same_fixed_density":True,
        "original_nuclear_source_observations":report,
        "descriptive_900_minus_750_k_mean_difference":float(diff),
        "conservative_illustrative_seed_t3_95pct_difference_interval":[float(diff-half),float(diff+half)],
        "temperature_coefficient_or_worth_statistically_qualified":False,
        "physical_eigenvalue_source_stationarity_demonstrated":False,
        "independent_seed_measurements_do_NOT_equal_independent_neutron_histories":True,
        "historical_NASA_NERVA_operating_core_qualified":False,
        "coupled_thermal_feedback_power_or_watts_qualified":False,
        "interpretation":[
            "Fixed material density and conceptual fuel chemistry; genuine evaluated 750/900 K neutron cross sections.",
            "Each seed's generation k is autocorrelated; descriptive four-seed uncertainty is not a certified CI.",
            "Within-seed 60/100 source-bank snapshots are correlated and cannot prove source convergence.",
            "Same seed label across two material temperatures is not a common-RNG paired history after source divergence.",
            "No uncertainty-corrected temperature-reactivity coefficient or NASA NERVA physical power claimed.",
        ],
    }


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    parser.add_argument("--cross-sections",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    report=audit_original_ensemble(args.root,nuclear_index=args.cross_sections)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for item in report["original_nuclear_source_observations"]:
        print("Real neutron material K:",item["actual_evaluated_neutron_material_temperature_K"],
              "mean k across 4 seeds:",item[
                  "seed_level_descriptive_k_uncertainty"]["mean_across_four_original_independent_seeds"],
              "source TV (60→100):",item["mean_source_TV_60_to_100"],
              "between seed TV:",item["mean_pairwise_between_seed_source_TV_at_100"])
    print("Historical hot NERVA, stationary neutron source, thermal power: NOT QUALIFIED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
