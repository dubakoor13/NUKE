"""Numerical/adversarial tests for saved OpenMC k-history prefix integrity.

All HDF5 fixtures below are SYNTHETIC format checks, not transport or a
NASA reactor validation. The real workflow separately executes OpenMC.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .real_core_keff_history_qualification import (
    _sha, evaluate_split_chain_k, audit_real_core_keff_histories,
    read_real_keff_checkpoint,
)


def reject(action, word: str) -> None:
    try:
        action()
    except ValueError as exc:
        assert word in str(exc), str(exc)
    else:
        raise AssertionError("invalid OpenMC physics history accepted: " + word)


def fake(file: Path, seed: int, checkpoint: int) -> None:
    index=np.arange(checkpoint,dtype=float)
    k=0.8+0.008*np.sin(index*0.29+seed/1e4)
    entropy=4.1+0.12*np.cos(index*0.17+seed/1e4)
    with h5py.File(file,"w") as h:
        h.attrs["filetype"]=np.bytes_("statepoint")
        h.attrs["source_present"]=1
        h.attrs["photon_transport"]=0
        h.create_dataset("run_mode",data=np.bytes_("eigenvalue"))
        h.create_dataset("seed",data=seed)
        h.create_dataset("current_batch",data=checkpoint)
        h.create_dataset("n_batches",data=80)
        h.create_dataset("n_inactive",data=20)
        h.create_dataset("n_particles",data=100)
        h.create_dataset("generations_per_batch",data=1)
        h.create_dataset("n_realizations",data=checkpoint-20)
        h.create_dataset("k_generation",data=k)
        h.create_dataset("entropy",data=entropy)
        h.create_dataset("k_combined",data=np.array([0.8,0.001]))


def main() -> int:
    rng=np.random.default_rng(124313)
    sequences=[rng.normal(0.8,0.01,100) for _ in range(4)]
    independent=evaluate_split_chain_k(sequences)
    assert independent["independent_seed_count"]==4
    assert independent["split_chains"]==8
    assert independent["generations_in_each_split"]==50
    assert independent["classical_split_chain_rhat_lower_bounded_at_one"]>=1.
    assert not independent["rhat_near_one_proves_spatial_source_convergence"]
    assert not independent["active_source_generation_histories_are_independent"]
    assert len(independent["per_seed"])==4
    shifted=[np.r_[rng.normal(0.8,0.002,50),
                   rng.normal(0.85,0.002,50)] for _ in range(4)]
    changed=evaluate_split_chain_k(shifted)
    assert changed["classical_split_chain_rhat_lower_bounded_at_one"]>1.1
    reject(lambda:evaluate_split_chain_k(sequences[:3]),"at least four")
    reject(lambda:evaluate_split_chain_k(sequences[:3]+[sequences[3][:70]]),"matched")
    reject(lambda:evaluate_split_chain_k([np.ones(100)]*4),"unidentifiable")
    reject(lambda:evaluate_split_chain_k(
        [np.r_[sequences[0][:-1],np.nan]]+sequences[1:]),"matched")

    with TemporaryDirectory() as td:
        root=Path(td)
        seeds=[13007,23011,33013,43019]
        runs=[]
        for seed in seeds:
            folder=root/f"seed_{seed}"
            folder.mkdir()
            for batch in (40,60,80):
                fake(folder/f"statepoint.{batch:02}.h5",seed,batch)
            final=folder/"statepoint.80.h5"
            runs.append({
                "seed":seed,"statepoint_path":str(final),
                "statepoint_sha256":_sha(final),
            })
        summary_path=root/"transport.json"
        spatial_path=root/"spatial.json"
        summary={
            "execution":"ACTUAL_OPENMC_TRANSPORT","assembly":"core",
            "batches":80,"saved_source_bank_batches":[40,60,80],
            "runs":runs,
        }
        def save():
            summary_path.write_text(json.dumps(summary),encoding="utf-8")
            spatial_path.write_text(json.dumps({
                "analysis":"EXACT_SAME_OPENMC_SOURCE_BANKS_ON_NESTED_SPATIAL_GRIDS",
                "real_transport_summary_sha256":_sha(summary_path),
                "resolutions":[{},{}],
                "source_diversity_screen_status":"FAIL_SOURCE_CLONING_SCREEN"
            }),encoding="utf-8")
        def audit():
            save()
            return audit_real_core_keff_histories(summary_path,spatial_path)
        validated=audit()
        assert validated["all_checkpoint_keff_and_entropy_prefixes_verified"]
        assert validated["executed_core_eigenvalue_settings_identical_across_seeds"]
        assert validated["existing_source_diversity_screen_status"]=="FAIL_SOURCE_CLONING_SCREEN"
        assert not validated["spatial_fission_source_stationarity_validated"]
        assert len(validated["real_seed_checkpoints"])==4
        assert all(len(x["checkpoints"])==3 for x in validated["real_seed_checkpoints"])
        early=Path(runs[0]["statepoint_path"]).parent/"statepoint.40.h5"
        assert read_real_keff_checkpoint(early)["generation_keff"].shape==(40,)
        with h5py.File(early,"r+") as h:
            h["k_generation"][1]+=0.1
        reject(audit,"lost exact prefix integrity")
        fake(early,seeds[0],40)
        with h5py.File(early,"r+") as h:
            h["entropy"][2]+=0.1
        reject(audit,"lost exact prefix integrity")
        fake(early,seeds[0],40)
        with h5py.File(early,"r+") as h:
            h["n_particles"][...]=101
        reject(audit,"lost exact prefix integrity")
        fake(early,seeds[0],40)
        with h5py.File(early,"r+") as h:
            h["current_batch"][...]=41
        reject(audit,"invalid real OpenMC eigenvalue")
        fake(early,seeds[0],40)
        with h5py.File(early,"r+") as h:
            h["k_generation"][3]=np.nan
        reject(lambda:read_real_keff_checkpoint(early),"invalid real OpenMC eigenvalue")
        fake(early,seeds[0],40)

        # An altered FINAL statepoint may retain the same valid-format fields,
        # but a different SHA cannot impersonate original executed transport.
        last=Path(runs[0]["statepoint_path"])
        with h5py.File(last,"r+") as h:
            h["k_generation"][31]+=0.12
        save()
        reject(lambda:audit_real_core_keff_histories(summary_path,spatial_path),
               "invalid genuine independent")
        fake(last,seeds[0],80)
        runs[0]["statepoint_sha256"]=_sha(last)
        audit()

        # Source-bank spatial gate hashes must bind to the same summary.
        save()
        spatial=json.loads(spatial_path.read_text())
        spatial["real_transport_summary_sha256"]="0"*64
        spatial_path.write_text(json.dumps(spatial))
        reject(lambda:audit_real_core_keff_histories(summary_path,spatial_path),
               "spatial source-convergence audit provenance")
        save()

        runs[1]["seed"]=runs[0]["seed"]
        reject(audit,"invalid genuine independent")
        runs[1]["seed"]=seeds[1]
        audit()
        summary["saved_source_bank_batches"]=[40,60,70]
        reject(audit,"invalid real OpenMC source checkpoint schedule")

    print("NERVA saved real-format k-generation/entropy prefix and split chain tests: PASS")
    print("  Synthetic HDF5 only: genuine neutron transport belongs to real OpenMC CI")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
