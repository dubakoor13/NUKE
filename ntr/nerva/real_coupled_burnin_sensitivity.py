"""Stage38: physically identical OpenMC histories with different tally burn-in.

Read actual source-neutron/photon HDF5 from Stage37 (20 inactive) and four
fresh OpenMC runs (40 inactive, same geometry/nuclear data/initial sources).
Changing the inactive threshold should NOT change source transport history;
if it does, refuse a paired burn-in comparison. Unlike comparing unrelated
Monte Carlo seed labels, here *identical* generation trajectories and
source_bank arrays are checked at batches 60 and 100.

Recover exact disjoint active tally moments from native OpenMC statepoints,
checking whether 61-100 heating agrees independent of tally burn-in.
No reactor physical watts, source-stationarity or historical NASA claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .real_coupled_initial_source_relaxation import (
    _native_heating,
    _sha,
)
from .real_coupled_spatial_heating_stability import (
    conservative_block_sum,
    total_variation,
)

MODES = ("legacy-broad", "central-compact")
SEEDS = (13007, 23011)
CHECKPOINTS = (60, 100)
HEATING_MESH = (12, 12, 10)
SPATIAL_SCALES = ((12, 12, 10), (6, 6, 10), (3, 3, 10), (1, 1, 10))
_PARTICLES = ("neutron", "photon", "electron", "positron")


def strict_native_identity(path: Path, *, seed: int, inactive: int, batch: int) -> dict:
    """Read actual native OpenMC generation histories and source bank.

    Do NOT fabricate or accept a JSON stand-in for the genuine HDF5 binary.
    """
    try:
        with h5py.File(path, "r") as h:
            if (bytes(h.attrs["filetype"]).rstrip(b"\\0") != b"statepoint"
                    or bytes(h["run_mode"][()]).rstrip(b"\\0") != b"eigenvalue"
                    or int(h.attrs["photon_transport"]) != 1
                    or int(h.attrs["source_present"]) != 1):
                raise ValueError("genuine coupled OpenMC eigenvalue source statepoint required")
            ident = {
                "seed": int(h["seed"][()]),
                "batch": int(h["current_batch"][()]),
                "total_batches": int(h["n_batches"][()]),
                "inactive": int(h["n_inactive"][()]),
                "particles": int(h["n_particles"][()]),
                "generations_per_batch": int(h["generations_per_batch"][()]),
                "realizations": int(h["n_realizations"][()]),
                "k": np.asarray(h["k_generation"][()], dtype=float),
                "entropy": np.asarray(h["entropy"][()], dtype=float),
                "source_bank": np.asarray(h["source_bank"][()]),
            }
    except (OSError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid original neutron/photon StatePoint: " + str(path)) from exc
    if (ident["seed"] != seed or ident["batch"] != batch
            or ident["total_batches"] != 100 or ident["inactive"] != inactive
            or ident["particles"] != 3000 or ident["generations_per_batch"] != 1
            or ident["realizations"] != batch-inactive
            or ident["k"].shape != (batch,) or ident["entropy"].shape != (batch,)
            or ident["source_bank"].size != 3000
            or not np.isfinite(ident["k"]).all()
            or not np.isfinite(ident["entropy"]).all()
            or np.any(ident["k"] < 0) or np.any(ident["entropy"] < 0)):
        raise ValueError("native OpenMC active/inactive realization or genealogy invalid")
    return ident


def recover_disjoint_sums(cumulative: tuple[np.ndarray, np.ndarray],
                          active_counts: tuple[int,int]) -> tuple[np.ndarray,np.ndarray]:
    """Return means for first recorded active interval and later [61,100].

    Active counts are [40,80] for 20-inactive; [20,60] for 40-inactive.
    No covariance or independent batch SEM can be extracted by subtraction.
    """
    if active_counts not in ((40,80),(20,60)) or len(cumulative)!=2:
        raise ValueError("native recorded active count pair invalid")
    earlier, later = (np.asarray(x,dtype=float) for x in cumulative)
    if (earlier.shape != later.shape or earlier.ndim < 1 or not earlier.size
            or not np.isfinite(earlier).all() or not np.isfinite(later).all()
            or np.any(earlier < 0) or np.any(later < 0)):
        raise ValueError("original neutron photon cumulative moments invalid")
    second = later - earlier
    tolerance = 1e-12*np.maximum(1.,np.maximum(later,earlier))
    if np.any(second < -tolerance):
        raise ValueError("non-monotonic native cumulative deposited heat")
    second = np.maximum(second,0.)
    first_mean=earlier/active_counts[0]
    second_mean=second/(active_counts[1]-active_counts[0])
    recovered=(active_counts[0]*first_mean+
               (active_counts[1]-active_counts[0])*second_mean)/active_counts[1]
    if not np.allclose(recovered,later/active_counts[1],rtol=1e-12,atol=1e-7):
        raise ValueError("disjoint heating failed original native raw moment closure")
    return first_mean,second_mean


def trajectory_identity(original: list[dict], candidate: list[dict]) -> dict:
    """Require identical native sampled source trajectories at 60 and 100.

    Exact equality must be observed, not inferred from using same seed label.
    """
    if (len(original)!=2 or len(candidate)!=2
            or [a["batch"] for a in original]!=list(CHECKPOINTS)
            or [a["batch"] for a in candidate]!=list(CHECKPOINTS)):
        raise ValueError("requires native source banks at 60 and 100")
    for left,right in zip(original,candidate):
        if (left["seed"] != right["seed"]
                or left["inactive"] != 20 or right["inactive"] != 40
                or left["batch"] != right["batch"]
                or not np.array_equal(left["k"],right["k"])
                or not np.array_equal(left["entropy"],right["entropy"])
                or not np.array_equal(left["source_bank"],right["source_bank"])):
            raise ValueError("same seed does not reproduce exact original transport history")
    for records in (original,candidate):
        if (not np.array_equal(records[0]["k"],records[1]["k"][:60])
                or not np.array_equal(records[0]["entropy"],records[1]["entropy"][:60])):
            raise ValueError("native OpenMC same-seed prefix history does not reproduce")
    return {
        "native_k_generation_bit_identical": True,
        "native_shannon_entropy_bit_identical": True,
        "native_3000_fission_source_sites_bit_identical_at_both_checkpoints": True,
        "identical_transport_realizations_verified_NOT_inferred_from_seed": True,
    }


def compare_burnin_heating(
    early20: np.ndarray, early40: np.ndarray,
    late20: np.ndarray, late40: np.ndarray
) -> dict:
    """Measure change in 3D mean by withholding 20 more early batches."""
    a,b,c,d = [np.asarray(x,dtype=float) for x in (early20,early40,late20,late40)]
    if (any(x.shape!=HEATING_MESH or not np.isfinite(x).all()
            or np.any(x < 0) or x.sum()<=0 for x in (a,b,c,d))):
        raise ValueError("four positive original 3D OpenMC heat windows required")
    # The 61-100 window contains the same underlying particle histories,
    # hence every native mesh deposited-energy mean must be bit-reproducible
    # to within floating point summation order, not just shape-close.
    if not np.allclose(c,d,rtol=2e-12,atol=1e-7):
        raise ValueError("identical native 61-100 heating source histories disagree")
    levels=[]
    for shape in SPATIAL_SCALES:
        p=conservative_block_sum(a,shape)
        q=conservative_block_sum(b,shape)
        r=conservative_block_sum(c,shape)
        levels.append({
            "mesh_bins_xyz":list(shape),
            "mean_heating_first_active_window_shape_TV_21_60_vs_41_60":
                total_variation(p/p.sum(),q/q.sum()),
            "mean_heating_21_60_vs_late_61_100_shape_TV":
                total_variation(p/p.sum(),r/r.sum()),
            "mean_heating_41_60_vs_late_61_100_shape_TV":
                total_variation(q/q.sum(),r/r.sum()),
        })
    for fine,coarse in zip(levels,levels[1:]):
        for key in ("mean_heating_first_active_window_shape_TV_21_60_vs_41_60",
                    "mean_heating_21_60_vs_late_61_100_shape_TV",
                    "mean_heating_41_60_vs_late_61_100_shape_TV"):
            if coarse[key]>fine[key]+1e-12:
                raise ValueError("conservative heating spatial TV rose after mesh integration")
    return {
        "four_disjoint_heating_fields_positive_and_native":True,
        "late_61_100_native_heating_reproduced_across_inactive_thresholds":True,
        "deposited_energy_21_60_ev_per_MC_source":float(a.sum()),
        "deposited_energy_41_60_ev_per_MC_source":float(b.sum()),
        "deposited_energy_61_100_ev_per_MC_source":float(c.sum()),
        "heating_shape_sensitivity_to_BurnIn_and_time_at_four_resolutions":levels,
    }


def audit_real_inactive_threshold(reference_root: Path | str,
                                  candidate_root: Path | str) -> dict:
    reference=Path(reference_root)
    candidate=Path(candidate_root)
    runs=[]
    all_pairwise_heating=[]
    for mode in MODES:
        for seed in SEEDS:
            paths=[reference/mode/f"seed_{seed}",candidate/mode/f"seed_{seed}"]
            provenance=[]
            for folder in paths:
                p=folder/"transport_provenance.json"
                if not p.is_file():
                    raise FileNotFoundError("actual original transport provenance required: "+str(p))
                metadata=json.loads(p.read_text())
                for item in ("geometry.xml","materials.xml","tallies.xml","settings.xml"):
                    file=folder/item
                    if not file.is_file() or _sha(file)!=metadata["xml_sha256"][item]:
                        raise ValueError("original transport geometry/material/tally/settings fingerprint mismatch")
                provenance.append(metadata)
            identity0=provenance[0]["config"].copy()
            identity1=provenance[1]["config"].copy()
            for cfg,threshold in ((identity0,20),(identity1,40)):
                if (cfg.pop("inactive")!=threshold
                        or cfg.pop("seed")!=seed
                        or cfg.pop("initial_fission_source")!=mode
                        or cfg.get("particles")!=3000
                        or cfg.get("batches")!=100 or cfg.get("photon_transport") is not True):
                    raise ValueError("source initialization or native inactive threshold does not match")
                batches=cfg.pop("statepoint_batches")
                if batches != ([30,60,100] if threshold==20 else [60,100]):
                    raise ValueError("unexpected native source checkpoint experiment")
            if identity0!=identity1:
                raise ValueError("physical reactor configuration differed across tally burn-in experiment")
            for item in ("geometry.xml","materials.xml","tallies.xml"):
                if provenance[0]["xml_sha256"][item]!=provenance[1]["xml_sha256"][item]:
                    raise ValueError("original material/geometry/tally XML changed")
            if (not provenance[0]["nuclear_data"].get("cross_sections_sha256")
                    or provenance[0]["nuclear_data"]["cross_sections_sha256"] !=
                    provenance[1]["nuclear_data"]["cross_sections_sha256"]):
                raise ValueError("evaluated neutron photon nuclear data differs")
            records=[]
            summed=[]
            native_fingerprints=[]
            signature=None
            for idx,(folder,inactive) in enumerate(zip(paths,(20,40))):
                hists=[]
                heats=[]
                for batch in CHECKPOINTS:
                    file=folder/f"statepoint.{batch:03d}.h5"
                    if not file.is_file():
                        raise FileNotFoundError("original native checkpoint missing: "+str(file))
                    identity=strict_native_identity(file,seed=seed,inactive=inactive,batch=batch)
                    tally=_native_heating(file)
                    if tally["realizations"] != [batch-inactive]*3:
                        raise ValueError("native OpenMC neutron photon energy tally realizations incorrect")
                    current_signature=[tally[x] for x in (
                        "mesh_lower_left_cm","mesh_upper_right_cm",
                        "material_bin_ids","estimators")]
                    if signature is None:
                        signature=current_signature
                    elif signature!=current_signature:
                        raise ValueError("native physical mesh/material/tally estimator changed")
                    hists.append(identity)
                    heats.append(tally)
                    native_fingerprints.append(_sha(file))
                if native_fingerprints[-1] != provenance[idx]["statepoint"]["sha256"]:
                    raise ValueError("final original OpenMC statepoint SHA256 differs from provenance")
                records.append(hists)
                summed.append(heats)
            genealogy=trajectory_identity(records[0],records[1])
            hist={}
            for domain in ("mesh","materials","particles"):
                original_first,original_late=recover_disjoint_sums(
                    (summed[0][0][domain],summed[0][1][domain]),(40,80))
                candidate_first,candidate_late=recover_disjoint_sums(
                    (summed[1][0][domain],summed[1][1][domain]),(20,60))
                for name,heat in (
                    ("original_active21_60",original_first),
                    ("candidate_active41_60",candidate_first),
                    ("native_active61_100_20_inactive",original_late),
                    ("native_active61_100_40_inactive",candidate_late)):
                    hist.setdefault(name,{})[domain]=heat
            for name,data in hist.items():
                sum_mesh=float(data["mesh"].sum())
                if not math.isfinite(sum_mesh) or sum_mesh<=0:
                    raise ValueError("original deposition invalid")
                for domain in ("materials","particles"):
                    if not math.isclose(sum_mesh,float(data[domain].sum()),
                        rel_tol=1e-9,abs_tol=1e-6):
                        raise ValueError("original neutron photon deposited energy partition failed")
            heating=compare_burnin_heating(
                hist["original_active21_60"]["mesh"],
                hist["candidate_active41_60"]["mesh"],
                hist["native_active61_100_20_inactive"]["mesh"],
                hist["native_active61_100_40_inactive"]["mesh"])
            # Important: same HDF5 current generation despite different
            # threshold. Differences in k-generation summaries are entirely
            # selection effects from excluding early batches.
            k=records[0][-1]["k"]
            ent=records[0][-1]["entropy"]
            run={
                "initial_source":mode,"rng_seed":seed,
                "genuine_native_statepoint_sha256_20_inactive_60_and_100":
                    native_fingerprints[:2],
                "genuine_native_statepoint_sha256_40_inactive_60_and_100":
                    native_fingerprints[2:],
                "native_transport_genealogy_reproducibility":genealogy,
                "native_20_vs_40_inactive_deposited_energy":heating,
                "generation_k_mean_after_20_inactive":float(np.mean(k[20:])),
                "generation_k_mean_after_40_inactive":float(np.mean(k[40:])),
                "source_shannon_entropy_mean_after_20_inactive":float(np.mean(ent[20:])),
                "source_shannon_entropy_mean_after_40_inactive":float(np.mean(ent[40:])),
            }
            runs.append(run)
            all_pairwise_heating.append(heating)
    return {
        "schema":"nuke-stage38-original-native-coupled-inactive-batch-sensitivity-v1",
        "original_native_hdf5_count":16,
        "new_reexecuted_openmc_neutron_photon_histories":4,
        "source_modes":list(MODES),
        "seeds_per_mode":list(SEEDS),
        "same_actual_neutron_transport_trajectories_verified":True,
        "same_original_NNDC_nuclear_data_and_physical_reactor_configuration":True,
        "genuine_native_material_mesh_particle_heating_closure":True,
        "per_original_run":runs,
        "descriptive_mean_early_heat_burnin_shape_TV_12x12x10":
            float(np.mean([r["heating_shape_sensitivity_to_BurnIn_and_time_at_four_resolutions"][0][
                "mean_heating_first_active_window_shape_TV_21_60_vs_41_60"] for r in all_pairwise_heating])),
        "burnin_20_vs_40_improves_fission_source_stationarity_PROVEN":False,
        "qualified_original_hot_nerva_reactor_power_watts":False,
        "notes":[
            "Same-seed trajectory is paired ONLY AFTER exact k/entropy/source-bank equality is verified.",
            "Changing inactive batches only changes tally inclusion and reported active statistics.",
            "This is NOT two independent fission-source replicates.",
            "Temporal windows share correlated fission ancestors; no IID SEM or coverage claims.",
            "3D coupled heating in eV/Monte Carlo starting source, not physical W/cm^3.",
            "Cold strongly subcritical surrogate is not a validated historical NERVA reactor.",
        ],
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-stage37-root",type=Path,required=True)
    p.add_argument("--new-40-inactive-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=audit_real_inactive_threshold(args.original_stage37_root,args.new_40_inactive_root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\\n",encoding="utf-8")
    print("Native original neutron+photon statepoints audited:",result["original_native_hdf5_count"])
    print("Bit-identical native transport source histories despite burn-in change: YES")
    print("Mean fine-3D heating shape TV, inactive20 vs inactive40:",result[
        "descriptive_mean_early_heat_burnin_shape_TV_12x12x10"])
    for r in result["per_original_run"]:
        print("seed",r["rng_seed"],"initial source",r["initial_source"],
              "k20",r["generation_k_mean_after_20_inactive"],
              "k40",r["generation_k_mean_after_40_inactive"])
    print("Stationary source and operating NERVA thermal watts qualified: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
