"""Stage39: replay EXACT original OpenMC settings, test actual C++ RNG reproducibility.

Stage38 proved that equal seed labels did not imply identical source tracks
between different inactive-generation setups. Before attributing divergence
to burn-in, repeat 4 baseline Stage37 settings *without changing anything*.

Read all 24 original native coupled neutron/photon StatePoints; verify actual
original XML SHA, evaluated data, photon mode, source banks, generation k and
entropy, native mesh/material/particle deposition. Never presume common RNG
pairing; report exact equality only when it is observed in the binaries.

Cold highly subcritical demonstrator, NOT an operating NERVA core or watts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .real_coupled_initial_source_relaxation import (
    _identity, _native_heating, _sha, disjoint_windows,
)
from .real_coupled_spatial_heating_stability import (
    conservative_block_sum, total_variation,
)
from .source_bank_convergence import read_source_bank

MODES = ("legacy-broad", "central-compact")
SEEDS = (13007, 23011)
BATCHES = (30, 60, 100)
WINDOWS = ((21, 30), (31, 60), (61, 100))
DIM = (12, 12, 10)
SCALES = ((12, 12, 10), (6, 6, 10), (3, 3, 10), (1, 1, 10))


def source_replication_diagnostics(first: list[dict], second: list[dict],
                                   first_banks: list[np.ndarray],
                                   second_banks: list[np.ndarray],
                                   first_hist: list[np.ndarray],
                                   second_hist: list[np.ndarray]) -> dict:
    """Compare true source histories (no assumed common random numbers)."""
    if (len(first) != 3 or len(second) != 3
            or len(first_banks) != 3 or len(second_banks) != 3
            or len(first_hist) != 3 or len(second_hist) != 3
            or [x["batch"] for x in first] != list(BATCHES)
            or [x["batch"] for x in second] != list(BATCHES)):
        raise ValueError("requires original three native source checkpoints per run")
    for seq in (first,second):
        for small,large in zip(seq,seq[1:]):
            if (not np.array_equal(small["k"],large["k"][:small["batch"]])
                    or not np.array_equal(small["entropy"],
                                          large["entropy"][:small["batch"]])):
                raise ValueError("original same-run native generation history prefix invalid")
    if any(a["seed"]!=b["seed"] for a,b in zip(first,second)):
        raise ValueError("native histories must carry same seed label")
    final_k_delta=first[-1]["k"]-second[-1]["k"]
    first_mismatch=np.flatnonzero(final_k_delta != 0)
    per_batch=[]
    for pos,(a,b,bank_a,bank_b,p,q) in enumerate(zip(
            first,second,first_banks,second_banks,first_hist,second_hist)):
        if a["batch"]!=b["batch"] or bank_a.shape!=bank_b.shape:
            raise ValueError("native source bank dimensions or batch mismatch")
        per_batch.append({
            "batch":BATCHES[pos],
            "native_generation_k_bit_identical":
                bool(np.array_equal(a["k"],b["k"])),
            "native_generation_entropy_bit_identical":
                bool(np.array_equal(a["entropy"],b["entropy"])),
            "native_full_fission_source_bank_bit_identical":
                bool(np.array_equal(bank_a,bank_b)),
            "native_4x4x10_source_bank_shape_TV":total_variation(p,q),
        })
    exact=bool(all(v["native_generation_k_bit_identical"]
                   and v["native_generation_entropy_bit_identical"]
                   and v["native_full_fission_source_bank_bit_identical"]
                   for v in per_batch))
    return {
        "original_executed_same_setting_k_entropy_source_banks_ALL_exact":exact,
        "first_generation_of_k_difference":
            int(first_mismatch[0]+1) if first_mismatch.size else None,
        "maximum_absolute_k_generation_difference":
            float(np.max(np.abs(final_k_delta))),
        "per_original_statepoint":per_batch,
        "same_seed_label_alone_is_NOT_a_statistical_paired_history":True,
    }


def heating_replication_diagnostics(first: np.ndarray,
                                    second: np.ndarray) -> dict:
    """Compare three genuine disjoint active heat fields, no false error bars."""
    a=np.asarray(first,dtype=float)
    b=np.asarray(second,dtype=float)
    if (a.shape!=(3,*DIM) or b.shape!=(3,*DIM)
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a<0) or np.any(b<0)
            or np.any(a.sum(axis=(1,2,3))<=0)
            or np.any(b.sum(axis=(1,2,3))<=0)):
        raise ValueError("two valid native three-window deposited-heating sets required")
    per_window=[]
    for j,window in enumerate(WINDOWS):
        totals=(float(a[j].sum()),float(b[j].sum()))
        levels=[]
        for shape in SCALES:
            x=conservative_block_sum(a[j],shape)
            y=conservative_block_sum(b[j],shape)
            levels.append({
                "mesh_xyz":list(shape),
                "cross_identical_setup_replays_heating_shape_TV":
                    total_variation(x/x.sum(),y/y.sum()),
            })
        if any(coarse["cross_identical_setup_replays_heating_shape_TV"] >
               fine["cross_identical_setup_replays_heating_shape_TV"]+1e-12
               for fine,coarse in zip(levels,levels[1:])):
            raise ValueError("native heating shape TV increased under conservative coarsening")
        per_window.append({
            "real_active_batch_window":list(window),
            "reference_deposited_heating_ev_per_mc_start":totals[0],
            "replay_deposited_heating_ev_per_mc_start":totals[1],
            "relative_difference_replay_minus_reference":
                (totals[1]-totals[0])/totals[0],
            "native_original_3D_disjoint_heat_field_identical":
                bool(np.array_equal(a[j],b[j])),
            "source_normalized_spatial_shape":levels,
        })
    return {
        "per_disjoint_active_window":per_window,
        "both_80_active_batch_energy_means_ev_per_MC_source":[
            float((10*a[0].sum()+30*a[1].sum()+40*a[2].sum())/80),
            float((10*b[0].sum()+30*b[1].sum()+40*b[2].sum())/80),
        ],
        "assumed_particle_genealogy_IID":False,
        "reactor_thermal_power_watts_known":False,
    }


def audit_real_replay(baseline: Path | str, replay: Path | str) -> dict:
    original=Path(baseline)
    candidate=Path(replay)
    cases=[]
    all_digest=set()
    for mode in MODES:
        for seed in SEEDS:
            both=(original/mode/f"seed_{seed}",
                  candidate/mode/f"seed_{seed}")
            cfgs=[]
            hashes=[]
            for folder in both:
                p=folder/"transport_provenance.json"
                if not p.is_file():
                    raise FileNotFoundError("missing original OpenMC C++ provenance: "+str(p))
                cfg=json.loads(p.read_text(encoding="utf-8"))
                args=cfg.get("config",{})
                if (args.get("seed")!=seed
                        or args.get("initial_fission_source")!=mode
                        or args.get("particles")!=3000
                        or args.get("batches")!=100
                        or args.get("inactive")!=20
                        or args.get("statepoint_batches")!=[30,60,100]
                        or args.get("photon_transport") is not True
                        or cfg.get("openmc_version") is None):
                    raise ValueError("actual nuclear transport replay configuration changed")
                checks={}
                for name in ("geometry.xml","materials.xml","tallies.xml","settings.xml"):
                    path=folder/name
                    if (not path.is_file()
                            or not cfg.get("xml_sha256",{}).get(name)
                            or cfg["xml_sha256"][name]!=_sha(path)):
                        raise ValueError("native reactor OpenMC input XML physical fingerprint failed")
                    checks[name]=cfg["xml_sha256"][name]
                nuclear=cfg.get("nuclear_data",{}).get("cross_sections_sha256")
                if not nuclear:
                    raise ValueError("evaluated nuclear data index hash missing")
                hashes.append((checks,nuclear,cfg["openmc_version"]))
                cfgs.append(args)
            if cfgs[0]!=cfgs[1] or hashes[0]!=hashes[1]:
                raise ValueError("replay must have identical evaluated data, all four XML and physics settings")
            k_entropy=[]
            banks=[]
            fractions=[]
            energy=[]
            for run_id,folder in enumerate(both):
                records=[]
                original_banks=[]
                hist=[]
                tal=[]
                signature=None
                for batch in BATCHES:
                    f=folder/f"statepoint.{batch:03d}.h5"
                    if not f.is_file():
                        raise FileNotFoundError("genuine original HDF5 checkpoint absent: "+str(f))
                    digest=_sha(f)
                    if digest in all_digest:
                        raise ValueError("same original native HDF5 statepoint reused as distinct run")
                    all_digest.add(digest)
                    raw=_identity(f,seed=seed,batch=batch)
                    with h5py.File(f,"r") as h:
                        source_bank=np.asarray(h["source_bank"][()])
                    tally=_native_heating(f)
                    bounds=(tally["mesh_lower_left_cm"],tally["mesh_upper_right_cm"],
                            tally["material_bin_ids"],tally["estimators"])
                    if signature is None:
                        signature=bounds
                    elif bounds!=signature:
                        raise ValueError("actual original physics scorer identity changed")
                    if tally["realizations"] != [batch-20]*3:
                        raise ValueError("native active batch realizations incorrect")
                    sites, meta=read_source_bank(
                        f,lower_cm=tally["mesh_lower_left_cm"],
                        upper_cm=tally["mesh_upper_right_cm"],
                        bins=(4,4,10))
                    if (meta["seed"]!=seed or meta["completed_batch"]!=batch
                            or meta["statepoint_sha256"]!=digest):
                        raise ValueError("native original bank identity inconsistent")
                    records.append(raw)
                    original_banks.append(source_bank)
                    hist.append(sites)
                    tal.append(tally)
                if _sha(folder/"statepoint.100.h5") != json.loads(
                    (folder/"transport_provenance.json").read_text())["statepoint"]["sha256"]:
                    raise ValueError("original final StatePoint hash not present in transport provenance")
                sums={}
                for domain in ("mesh","materials","particles"):
                    sums[domain]=disjoint_windows([r[domain] for r in tal])
                for j in range(3):
                    total=float(sums["mesh"][j].sum())
                    if total<=0:
                        raise ValueError("OpenMC native deposited heating non-positive")
                    for domain in ("materials","particles"):
                        if not math.isclose(total,float(sums[domain][j].sum()),
                                            rel_tol=1e-9,abs_tol=1e-6):
                            raise ValueError("actual OpenMC coupled 3D/material/particle heat imbalance")
                k_entropy.append(records)
                banks.append(original_banks)
                fractions.append(hist)
                energy.append(sums["mesh"])
            result=source_replication_diagnostics(
                k_entropy[0],k_entropy[1],banks[0],banks[1],
                fractions[0],fractions[1])
            heat=heating_replication_diagnostics(energy[0],energy[1])
            cases.append({
                "mode":mode,"seed_label":seed,
                "actual_original_and_replay_xml_all_identical":True,
                "same_evaluated_cross_sections_SHA_and_openmc_version":True,
                "native_replayed_fission_source_history":result,
                "native_replayed_coupled_deposited_energy":heat,
            })
    return {
        "schema":"nuke-stage39-genuine-openmc-identical-configuration-replay-v1",
        "actual_native_HDF5_statepoints_reopened":len(all_digest),
        "original_stage37_executions":4,
        "new_identical_configuration_openmc_executions":4,
        "source_modes":list(MODES),
        "two_rng_seed_labels":list(SEEDS),
        "100_batches_20_inactive_3000_source_particles_24_real_hdf5":True,
        "all_actual_reactor_geometry_materials_tallies_settings_data_identical":True,
        "all_actual_native_coupled_heat_energy_partition_closed":True,
        "exact_replay_histories_in_all_four_cases":bool(all(
            r["native_replayed_fission_source_history"][
                "original_executed_same_setting_k_entropy_source_banks_ALL_exact"]
            for r in cases)),
        "per_actual_original_replay":cases,
        "source_stationarity_and_neutron_history_independence_qualified":False,
        "full_hot_nasa_nerva_physical_core_or_reactor_watts_qualified":False,
        "notes":[
            "A same-seed and same-XML replay may diverge due to execution ordering; exact equality is measured, never assumed.",
            "Four repeat pairs are a deterministic replay probe, not independent source convergence evidence.",
            "Hot NASA-reference core model and physically normalized reactor W remain unvalidated.",
            "Only validated native OpenMC neutron/photon HDF5 are used; no synthetic transport results.",
        ],
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--replay",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=audit_real_replay(a.reference,a.replay)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("GENUINE native coupled HDF5 files:",result["actual_native_HDF5_statepoints_reopened"])
    for case in result["per_actual_original_replay"]:
        observed=case["native_replayed_fission_source_history"]
        print("Mode",case["mode"],"seed",case["seed_label"],
              "exact replay:",observed[
                  "original_executed_same_setting_k_entropy_source_banks_ALL_exact"],
              "first k difference:",observed["first_generation_of_k_difference"],
              "final source spatial TV:",observed["per_original_statepoint"][-1][
                  "native_4x4x10_source_bank_shape_TV"])
    print("All identical transport source trajectories observed:",
          result["exact_replay_histories_in_all_four_cases"])
    print("Hot reactor criticality, neutron source stationarity, watts qualified: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
