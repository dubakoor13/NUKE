"""Stage43: localize first ORIGINAL OpenMC C++ fission-source bank divergence.

Four actual source-only neutron+photon OpenMC v0.15.3 histories, 100 batches
and 3000 particles/gen: two identical physical source modes × inactive 20/40.
Save native HDF5 original source banks at generations 19–30,60,100.

A repeated original Stage42 no-user-tally 20/40 source run controls whether
extra checkpoint writes themselves alter a source history. Verify exact
bank/entropy and rounding-level k agreement at 60 and 100 before inferring
the first exact bank divergence. No physical reactor design or heat scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .real_coupled_initial_source_relaxation import _sha

MODES=("legacy-broad","central-compact")
SEED=13007
EARLY=tuple(range(19,31))
BATCHES=EARLY+(60,100)


def read_genuine_source_checkpoint(path: Path, *, batch: int,
                                   inactive: int) -> dict:
    """Refuse fabricated HDF5, noncoupled transport, or missing original bank."""
    if not path.is_file() or path.stat().st_size<1000:
        raise FileNotFoundError("original OpenMC native source HDF5 unavailable: "+str(path))
    with h5py.File(path,"r") as h:
        if (bytes(h.attrs.get("filetype",b"")).rstrip(b"\0")!=b"statepoint"
                or bytes(h["run_mode"][()]).rstrip(b"\0")!=b"eigenvalue"
                or int(h.attrs.get("source_present",0))!=1
                or int(h.attrs.get("photon_transport",0))!=1
                or int(h["current_batch"][()])!=batch
                or int(h["n_batches"][()])!=100
                or int(h["n_inactive"][()])!=inactive
                or int(h["n_particles"][()])!=3000
                or int(h["seed"][()])!=SEED
                or int(h["generations_per_batch"][()])!=1):
            raise ValueError("incorrect real neutron photon source genealogy statepoint")
        source=np.asarray(h["source_bank"][()])
        k=np.asarray(h["k_generation"][()],dtype=np.float64)
        entropy=np.asarray(h["entropy"][()],dtype=np.float64)
        active_count=int(h["n_realizations"][()])
        if "tallies" in h and "ids" in h["tallies"] and len(h["tallies/ids"]):
            raise ValueError("source-only checkpoint unexpectedly scored user tallies")
    if (source.shape!=(3000,) or source.dtype.names is None
            or "r" not in source.dtype.names
            or k.shape!=(batch,) or entropy.shape!=(batch,)
            or active_count!=max(0,batch-inactive)
            or not np.isfinite(k).all() or not np.isfinite(entropy).all()):
        raise ValueError("native eigenvalue source history bank, moment or active count invalid")
    return {
        "source":source,
        "k":k,
        "entropy":entropy,
        "active_realizations":active_count,
        "source_bank_sha256":hashlib.sha256(source.tobytes()).hexdigest(),
        "statepoint_sha256":_sha(path),
    }


def compare_bank_sequences(left: dict[int,dict],right: dict[int,dict]) -> dict:
    """Quantify first exact source-array difference vs first meaningful k drift."""
    if tuple(left)!=BATCHES or tuple(right)!=BATCHES:
        raise ValueError("all native OpenMC 19-30,60,100 banks mandatory")
    observations=[]
    for batch in BATCHES:
        a,b=left[batch],right[batch]
        mismatch=np.abs(a["k"]-b["k"])
        entropy=np.abs(a["entropy"]-b["entropy"])
        observations.append({
            "native_completed_batch":batch,
            "entire_original_source_bank_bytes_equal":bool(np.array_equal(
                a["source"],b["source"])),
            "first_fission_site_array_index_different_1_based":
                int(np.flatnonzero(a["source"]!=b["source"])[0]+1)
                if np.any(a["source"]!=b["source"]) else None,
            "batch_end_shannon_source_entropy_equal":bool(
                np.array_equal(a["entropy"],b["entropy"])),
            "batch_end_k_generation_within_1e_8_absolute":bool(
                np.max(mismatch)<=1e-8),
            "maximum_k_generation_absolute_difference_so_far":
                float(np.max(mismatch)),
            "maximum_entropy_difference_so_far":
                float(np.max(entropy)),
            "inactive20_bank_sha256":a["source_bank_sha256"],
            "inactive40_bank_sha256":b["source_bank_sha256"],
        })
    return {
        "per_original_checkpoint":observations,
        "first_completed_batch_with_different_real_fission_bank":
            next((x["native_completed_batch"] for x in observations
                  if not x["entire_original_source_bank_bytes_equal"]),None),
        "first_completed_batch_with_materially_different_generation_k":
            next((x["native_completed_batch"] for x in observations
                  if not x["batch_end_k_generation_within_1e_8_absolute"]),None),
        "first_completed_batch_with_different_source_entropy":
            next((x["native_completed_batch"] for x in observations
                  if not x["batch_end_shannon_source_entropy_equal"]),None),
        "physical_source_stationarity_established":False,
    }


def audit_per_generation_original_sources(previous:Path|str,
                                          new:Path|str) -> dict:
    prior=Path(previous)
    fresh=Path(new)
    rows=[]
    for mode in MODES:
        histories={}
        for inactive in (20,40):
            folder=fresh/f"inactive{inactive}"/mode/f"seed_{SEED}"
            archived=prior/f"inactive{inactive}"/mode/f"seed_{SEED}"
            p=folder/"transport_provenance.json"
            p0=archived/"transport_provenance.json"
            if not p.is_file() or not p0.is_file():
                raise FileNotFoundError("both original executed source-only transport provenance required")
            now=json.loads(p.read_text(encoding="utf-8"))
            old=json.loads(p0.read_text(encoding="utf-8"))
            config=dict(now["config"])
            if (config.pop("diagnostic_allow_inactive_statepoints",None) is not True
                    or config.pop("diagnostic_no_user_tallies",None) is not True
                    or config.pop("statepoint_batches",None)!=list(BATCHES)
                    or config.get("seed")!=SEED
                    or config.get("inactive")!=inactive
                    or config.get("photon_transport") is not True
                    or config.get("initial_fission_source")!=mode):
                raise ValueError("inactive native checkpoint control settings invalid")
            historic=dict(old["config"])
            if (historic.pop("diagnostic_no_user_tallies",None) is not True
                    or historic.pop("statepoint_batches",None)!=[60,100]
                    or historic!=config):
                raise ValueError("source-checkpoint diagnostic changed physical reactor transport config")
            for key in ("geometry.xml","materials.xml"):
                if (not (folder/key).is_file()
                        or _sha(folder/key)!=now.get("xml_sha256",{}).get(key)
                        or now["xml_sha256"][key]!=old["xml_sha256"][key]):
                    raise ValueError("original neutron core geometry/materials XML changed")
            if (not now["nuclear_data"]["cross_sections_sha256"]
                    or now["nuclear_data"]["cross_sections_sha256"]!=old[
                        "nuclear_data"]["cross_sections_sha256"]):
                raise ValueError("evaluated neutron/photon cross-section index changed")
            new_banks={}
            for batch in BATCHES:
                file=folder/f"statepoint.{batch:03d}.h5"
                new_banks[batch]=read_genuine_source_checkpoint(
                    file,batch=batch,inactive=inactive)
            for batch in (60,100):
                historic_state=read_genuine_source_checkpoint(
                    archived/f"statepoint.{batch:03d}.h5",
                    batch=batch,inactive=inactive)
                recorded=new_banks[batch]
                if (not np.array_equal(historic_state["source"],recorded["source"])
                        or not np.array_equal(historic_state["entropy"],
                                              recorded["entropy"])
                        or np.max(np.abs(historic_state["k"]-recorded["k"]))>=1e-14):
                    raise ValueError("additional inactive sourcepoint writing CHANGED the neutron trajectory")
            for small,large in zip(BATCHES,BATCHES[1:]):
                if (not np.array_equal(new_banks[small]["k"],
                                       new_banks[large]["k"][:small])
                        or not np.array_equal(new_banks[small]["entropy"],
                                              new_banks[large]["entropy"][:small])):
                    raise ValueError("actual original same-run fission source temporal prefixes changed")
            histories[inactive]=new_banks
        observed=compare_bank_sequences(histories[20],histories[40])
        rows.append({
            "initial_fission_source":mode,
            "same_source_history_at_batch_60_and_100_as_prior_stage42_exact":True,
            "per_generation_source_bank_comparison":observed,
        })
    return {
        "schema":"nuke-stage43-native-eigenvalue-source-first-divergence-v1",
        "original_new_openmc_source_only_CPP_histories":4,
        "original_new_openmc_Cplusplus_HDF5_checkpoints":4*len(BATCHES),
        "previous_original_stage42_HDF5_replay_controls":8,
        "full_original_core_material_and_nuclear_data_equivalence":True,
        "all_extra_HDF5_writes_proven_nonintrusive_against_original":True,
        "source_initialization_per_mode":rows,
        "source_stationarity_statistically_established":False,
        "operating_NASA_NERVA_thermal_power_qualified":False,
        "limitations":[
            "Original source-only neutron/photon OpenMC C++ HDF5, not generated approximations.",
            "Native source bank equality, source entropy and k-generation are different numerical observables.",
            "Original source-only model has no user deposited-heating tallies and cannot estimate reactor watts.",
            "This is a cold, highly subcritical small surrogate, not actual hot NASA NERVA.",
        ],
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage42-original-no-tallies",type=Path,required=True)
    p.add_argument("--stage43-new-checkpoints",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=audit_per_generation_original_sources(a.stage42_original_no_tallies,
                                            a.stage43_new_checkpoints)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for row in r["source_initialization_per_mode"]:
        x=row["per_generation_source_bank_comparison"]
        print(row["initial_fission_source"],
              "first native bank divergence batch:",x[
                  "first_completed_batch_with_different_real_fission_bank"],
              "first source entropy divergence:",x[
                  "first_completed_batch_with_different_source_entropy"],
              "first materially different generation k:",x[
                  "first_completed_batch_with_materially_different_generation_k"])
    print("Physical hot NERVA power or qualified stationary source: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
