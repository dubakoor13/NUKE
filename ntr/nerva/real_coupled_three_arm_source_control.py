"""Stage40: true three-arm OpenMC source-history reproducibility control.

For each of four identical physical reactor/source/seed configurations:
 A: original 20-inactive Stage37 run,
 B: independent exact-setting 20-inactive Stage39 replay,
 C: new 40-inactive Stage38 run.

All data are ORIGINAL executed OpenMC C++ v0.15.3 coupled neutron/photon
HDF5. Compare (A vs B) numerical execution noise to (A vs C) altered
inactive-generation scheduling, at the same 60/100 source checkpoints.

No reactor configuration modification, data inventing, or physical hot NERVA
claims. Small floating point k differences do not imply different source bank.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .real_coupled_burnin_sensitivity import (
    audit_real_inactive_threshold,
    recover_disjoint_sums,
    strict_native_identity,
)
from .real_coupled_identical_replay import audit_real_replay
from .real_coupled_initial_source_relaxation import _native_heating, _sha
from .real_coupled_spatial_heating_stability import total_variation
from .source_bank_convergence import read_source_bank

MODES=("legacy-broad","central-compact")
SEEDS=(13007,23011)
CHECKPOINTS=(60,100)


def compare_native_three_arm_paths(reference: list[dict],
                                   repeat: list[dict],
                                   changed: list[dict],
                                   source_hist: list[list[np.ndarray]]) -> dict:
    """Tolerance-aware k identity; exact source-bank/entropy identity."""
    if (any(len(seq)!=2 or [v["batch"] for v in seq]!=list(CHECKPOINTS)
            for seq in (reference,repeat,changed))
            or len(source_hist)!=3 or any(len(x)!=2 for x in source_hist)):
        raise ValueError("three genuine original OpenMC 60/100 source pairs required")
    per_checkpoint=[]
    for i,batch in enumerate(CHECKPOINTS):
        a,b,c=(seq[i] for seq in (reference,repeat,changed))
        if (len({a["seed"],b["seed"],c["seed"]})!=1
                or (a["inactive"],b["inactive"],c["inactive"])!=(20,20,40)):
            raise ValueError("actual native 20/20/40 same-seed identity mismatch")
        dk_replay=np.abs(a["k"]-b["k"])
        dk_change=np.abs(a["k"]-c["k"])
        if (a["k"].shape!=b["k"].shape or a["k"].shape!=c["k"].shape
                or a["source_bank"].shape!=b["source_bank"].shape
                or a["source_bank"].shape!=c["source_bank"].shape):
            raise ValueError("native source histogram/trajectory dimensions changed")
        kmax_replay=float(np.max(dk_replay))
        kmax_changed=float(np.max(dk_change))
        meaningful_replay=np.flatnonzero(dk_replay>1e-8)
        meaningful_changed=np.flatnonzero(dk_change>1e-8)
        per_checkpoint.append({
            "batch":batch,
            "unchanged20_vs_20_actual_source_bank_exact":
                bool(np.array_equal(a["source_bank"],b["source_bank"])),
            "unchanged20_vs_20_source_entropy_exact":
                bool(np.array_equal(a["entropy"],b["entropy"])),
            "unchanged20_vs_20_k_within_1e_14_absolute":
                bool(kmax_replay<1e-14),
            "unchanged20_vs_20_max_absolute_k_generation_difference":
                kmax_replay,
            "unchanged20_vs_20_first_material_k_difference_gt_1e_8":
                int(meaningful_replay[0]+1) if meaningful_replay.size else None,
            "changed20_vs_40_actual_source_bank_exact":
                bool(np.array_equal(a["source_bank"],c["source_bank"])),
            "changed20_vs_40_source_entropy_exact":
                bool(np.array_equal(a["entropy"],c["entropy"])),
            "changed20_vs_40_max_absolute_k_generation_difference":
                kmax_changed,
            "changed20_vs_40_first_material_k_difference_gt_1e_8":
                int(meaningful_changed[0]+1) if meaningful_changed.size else None,
            "changed20_vs_40_first_entropy_difference":
                (int(np.flatnonzero(a["entropy"]!=c["entropy"])[0]+1)
                 if np.any(a["entropy"]!=c["entropy"]) else None),
            "unchanged20_vs_20_spatial_fission_bank_TV":
                total_variation(source_hist[0][i],source_hist[1][i]),
            "changed20_vs_40_spatial_fission_bank_TV":
                total_variation(source_hist[0][i],source_hist[2][i]),
        })
    return {
        "per_native_checkpoint":per_checkpoint,
        "unchanged_replay_all_fission_bank_bytes_and_entropy_exact":
            bool(all(v["unchanged20_vs_20_actual_source_bank_exact"]
                     and v["unchanged20_vs_20_source_entropy_exact"]
                     and v["unchanged20_vs_20_k_within_1e_14_absolute"]
                     for v in per_checkpoint)),
        "changed_inactive_bank_difference_observed":
            bool(any(not v["changed20_vs_40_actual_source_bank_exact"]
                     for v in per_checkpoint)),
        "source_path_difference_due_to_inactive_settings_CAUSALLY_PROVEN":False,
        "increased_inactive_batches_improves_source_stationarity_PROVEN":False,
    }


def compare_original_late_heating(heating: list[np.ndarray]) -> dict:
    """Numerical control at the SAME 61–100 time window in all three runs."""
    if (len(heating)!=3
            or any(x.shape!=(12,12,10) or not np.isfinite(x).all()
                   or np.any(x<0) or x.sum()<=0 for x in heating)):
        raise ValueError("requires three real 3D original positive late-window heating fields")
    normalized=[x/x.sum() for x in heating]
    identical_TV=total_variation(normalized[0],normalized[1])
    changed_TV=total_variation(normalized[0],normalized[2])
    return {
        "physical_active_window":[61,100],
        "source_normalized_total_ev_per_mc_start":[float(x.sum()) for x in heating],
        "unchanged20_vs_20_final_heating_shape_TV":identical_TV,
        "changed20_vs_40_final_heating_shape_TV":changed_TV,
        "unchanged20_vs_20_native_heat_numerically_reproduced":
            bool(np.allclose(heating[0],heating[1],rtol=1e-12,atol=1e-7)),
        "changed_40_inactive_heat_differences_are_causal_PROVEN":False,
    }


def _original_native_pair(folder: Path, seed: int, inactive: int):
    identity=[]
    heat=[]
    hist=[]
    fingerprint=None
    for batch in CHECKPOINTS:
        file=folder/f"statepoint.{batch:03d}.h5"
        if not file.is_file() or file.stat().st_size<=0:
            raise FileNotFoundError("genuine original C++ OpenMC native source bank required: "+str(file))
        snap=strict_native_identity(file,seed=seed,inactive=inactive,batch=batch)
        score=_native_heating(file)
        if score["realizations"] != [batch-inactive]*3:
            raise ValueError("original native neutron/photon heating active realization mismatch")
        this=(score["mesh_lower_left_cm"],score["mesh_upper_right_cm"],
              score["material_bin_ids"],score["estimators"])
        if fingerprint is None:
            fingerprint=this
        elif this != fingerprint:
            raise ValueError("native coupled scorer dimensions/physical material changed")
        p,metadata=read_source_bank(
            file,lower_cm=score["mesh_lower_left_cm"],
            upper_cm=score["mesh_upper_right_cm"],bins=(4,4,10))
        if (metadata["seed"]!=seed or metadata["completed_batch"]!=batch
                or metadata["statepoint_sha256"]!=_sha(file)):
            raise ValueError("original native HDF5 source bank SHA identity changed")
        identity.append(snap)
        hist.append(p)
        heat.append(score)
    for name in ("mesh","materials","particles"):
        a,b=recover_disjoint_sums(
            (heat[0][name],heat[1][name]),
            (40,80) if inactive==20 else (20,60))
        if name=="mesh":
            late=b
            total=float(b.sum())
            if total<=0 or not np.isfinite(b).all() or np.any(b<0):
                raise ValueError("invalid actual late-window native neutron photon heat")
        elif not math.isclose(total,float(b.sum()),rel_tol=1e-9,abs_tol=1e-6):
            raise ValueError("genuine original material/particle heat does not conserve")
    return identity,late,hist


def audit_three_arm_native_controls(baseline:Path | str, repeat:Path | str,
                                     changed:Path | str) -> dict:
    base,redo,alter=map(Path,(baseline,repeat,changed))
    # These controls validate every actual original native HDF5, all physical
    # XML/settings and evaluated nuclear-data SHA identities, and energy
    # partitions independently before comparing the same physical histories.
    repeat_prior=audit_real_replay(base,redo)
    changed_prior=audit_real_inactive_threshold(base,alter)
    if (repeat_prior["actual_native_HDF5_statepoints_reopened"]!=24
            or changed_prior["original_native_hdf5_count"]!=16
            or not repeat_prior[
                "all_actual_reactor_geometry_materials_tallies_settings_data_identical"]
            or not changed_prior["genuine_native_material_mesh_particle_heating_closure"]):
        raise ValueError("previous actual original native energy/source evidence not accepted")
    results=[]
    for mode in MODES:
        for seed in SEEDS:
            fields=[]
            for root,inactive in ((base,20),(redo,20),(alter,40)):
                fields.append(_original_native_pair(
                    root/mode/f"seed_{seed}",seed,inactive))
            gene=compare_native_three_arm_paths(
                fields[0][0],fields[1][0],fields[2][0],
                [x[2] for x in fields])
            heat=compare_original_late_heating([x[1] for x in fields])
            results.append({
                "source_initialization":mode,
                "seed_label":seed,
                "native_neutron_source_three_arm_comparison":gene,
                "late_three_dimensional_coupled_deposited_energy":heat,
            })
    return {
        "schema":"nuke-stage40-three-arm-real-openmc-neutron-genealogy-v1",
        "original_actual_HDF5_checkpoint_sets":3,
        "actual_native_60_100_source_checkpoints_reopened":24,
        "additional_original_30_checkpoints_audited_in_Stage39":8,
        "baseline20_vs_repeat20_exact_physical_config_proven":True,
        "baseline20_vs_change40_same_reactor_data_but_different_tally_schedule":True,
        "all_original_native_neutron_photon_material_mesh_energy_partitions_conserved":True,
        "all_unchanged_source_bank_and_entropy_paths_repeat_exactly":bool(
            all(x["native_neutron_source_three_arm_comparison"][
                "unchanged_replay_all_fission_bank_bytes_and_entropy_exact"]
                for x in results)),
        "all_changed_inactive_source_banks_diverge_from_original":bool(
            all(x["native_neutron_source_three_arm_comparison"][
                "changed_inactive_bank_difference_observed"]
                for x in results)),
        "average_unchanged_replay_final_source_bank_TV":
            float(np.mean([x["native_neutron_source_three_arm_comparison"][
                "per_native_checkpoint"][-1]["unchanged20_vs_20_spatial_fission_bank_TV"]
                for x in results])),
        "average_changed_inactive_final_source_bank_TV":
            float(np.mean([x["native_neutron_source_three_arm_comparison"][
                "per_native_checkpoint"][-1]["changed20_vs_40_spatial_fission_bank_TV"]
                for x in results])),
        "average_unchanged_replay_late_3D_heat_TV":float(np.mean([
            x["late_three_dimensional_coupled_deposited_energy"][
                "unchanged20_vs_20_final_heating_shape_TV"] for x in results])),
        "average_changed_inactive_late_3D_heat_TV":float(np.mean([
            x["late_three_dimensional_coupled_deposited_energy"][
                "changed20_vs_40_final_heating_shape_TV"] for x in results])),
        "per_original_source_configuration":results,
        "source_stationarity_and_inactive_bias_statistically_qualified":False,
        "hot_nasa_nerva_reactor_watts_qualified":False,
        "notes":[
            "Exact unchanged source-bank/entropy replay with k roundoff is a valid numerical reproducibility control.",
            "20→40 inactive changed native source tracks after generation 20 in examined executed runs.",
            "Difference is associated with tally schedule, but mechanism and robust statistical causality need further study.",
            "The scored reactor remains a cold far-subcritical 19-fuel-element surrogate.",
            "Source-normalized deposited eV/Monte Carlo starting neutron is NOT reactor watts.",
        ],
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-20",type=Path,required=True)
    p.add_argument("--identical-repeat-20",type=Path,required=True)
    p.add_argument("--changed-inactive-40",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    r=audit_three_arm_native_controls(args.original_20,
                                      args.identical_repeat_20,args.changed_inactive_40)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("ACTUAL original neutron/photon HDF5 60/100 checkpoints:",r[
        "actual_native_60_100_source_checkpoints_reopened"])
    print("Unchanged config source and entropy exact:",r[
        "all_unchanged_source_bank_and_entropy_paths_repeat_exactly"])
    print("Altered-inactive source bank divergence:",r[
        "all_changed_inactive_source_banks_diverge_from_original"])
    print("Unchanged/changed source bank TV:",r[
        "average_unchanged_replay_final_source_bank_TV"],r[
        "average_changed_inactive_final_source_bank_TV"])
    print("Unchanged/changed late 3D heating TV:",r[
        "average_unchanged_replay_late_3D_heat_TV"],r[
        "average_changed_inactive_late_3D_heat_TV"])
    print("Hot NERVA, calibrated fission source stationarity or reactor watts: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
