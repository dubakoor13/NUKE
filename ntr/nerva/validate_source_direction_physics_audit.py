"""Synthetic adversarial HDF5 regression for actual-source angular physics.

These manufactured source banks test rejection logic only; they do not
execute OpenMC, establish source convergence, or represent a NASA core.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import h5py
import numpy as np

from .source_direction_physics_audit import (
    _sha256, audit_real_source_angular_physics, read_real_bank_angular_physics,
)


def reject(fn, token: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert token in str(exc), str(exc)
    else:
        raise AssertionError("invalid source physics accepted: " + token)


def fake(path: Path, seed: int, batch: int, *, angle_bad=False, proton_bad=False):
    v = np.array([
        [1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.],
        [0.,0.,1.],[0.,0.,-1.],[1.,0.,0.],[0.,1.,0.],
    ])
    if angle_bad:
        v[0] = [2., 0., 0.]
    r = np.array([[.1,.2,.3]]*4 + [[.4,.5,.6]]*4)
    vector_dtype=np.dtype([("x","f8"),("y","f8"),("z","f8")])
    rec=np.dtype([
        ("r",vector_dtype), ("u",vector_dtype), ("E","f8"),
        ("time","f8"), ("wgt","f8"), ("particle","i4"),
    ])
    x=np.zeros(8,dtype=rec)
    for i,d in enumerate(("x","y","z")):
        x["r"][d] = r[:,i]
        x["u"][d] = v[:,i]
    x["E"] = np.linspace(1e5,4e6,8)
    x["time"] = 1.0
    x["wgt"] = 1.0
    if proton_bad:
        x["particle"][0] = 1
    with h5py.File(path,"w") as h:
        h.attrs["filetype"]=np.bytes_("statepoint")
        h.attrs["source_present"]=1
        h.attrs["photon_transport"]=0
        h.create_dataset("run_mode",data=np.bytes_("eigenvalue"))
        h.create_dataset("seed",data=seed)
        h.create_dataset("current_batch",data=batch)
        h.create_dataset("n_particles",data=8)
        h.create_dataset("source_bank",data=x)


def main() -> int:
    seeds = (13007,23011,33013,43019)
    checkpoints = (40,60,80)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        final_paths=[]
        for seed in seeds:
            folder=root/str(seed)
            folder.mkdir()
            for b in checkpoints:
                fake(folder/f"statepoint.{b:02}.h5",seed,b)
            final_paths.append(folder/"statepoint.80.h5")
        summary=root/"real_core_transport_summary.json"
        payload={
            "execution":"ACTUAL_OPENMC_TRANSPORT",
            "assembly":"core",
            "batches":80,
            "saved_source_bank_batches":list(checkpoints),
            "runs":[],
        }
        def save_summary():
            payload["runs"]=[
                {"seed":seed, "statepoint_path":str(file),
                 "statepoint_sha256":_sha256(file)}
                for seed,file in zip(seeds,final_paths)
            ]
            summary.write_text(json.dumps(payload),encoding="utf-8")
        save_summary()
        report=audit_real_source_angular_physics(summary)
        assert report["actual_independent_seed_count"]==4
        assert report["matched_checkpoint_batches"]==[40,60,80]
        assert report["largest_exact_position_clone_multiplicity"]==4
        assert report["final_bank_effective_position_counts"]==[2.]*4
        assert report["final_bank_mean_emission_direction_magnitudes"][0] > 0
        assert report["nominal_source_population_as_independent_fission_histories"] is False
        assert not report["near_isotropic_directions_prove_spatial_source_convergence"]
        assert not report["fission_source_stationarity_demonstrated"]
        first=report["independent_seed_real_source_records"][0]
        assert first["exact_positions_shared_across_adjacent_saved_checkpoints"] == [2,2]
        _, positions=read_real_bank_angular_physics(final_paths[0])
        assert len(positions)==2
        eigen=first["checkpoints"][0]["direction_second_moment_eigenvalues"]
        assert abs(sum(eigen)-1.)<1e-12
        assert first["checkpoints"][0]["unique_directions_at_most_cloned_position"]==4
        assert first["checkpoints"][0]["unique_energies_at_most_cloned_position"]==4

        f=final_paths[0]
        fake(f,seeds[0],80,angle_bad=True)
        reject(lambda:read_real_bank_angular_physics(f),"unit vector")
        fake(f,seeds[0],80,proton_bad=True)
        reject(lambda:read_real_bank_angular_physics(f),"nonphysical neutron source")
        fake(f,seeds[0],80)
        with h5py.File(f,"r+") as h:
            b=h["source_bank"][()]
            b["E"][0]=np.nan
            h["source_bank"][...]=b
        reject(lambda:read_real_bank_angular_physics(f),"nonphysical neutron source")
        fake(f,seeds[0],80)
        with h5py.File(f,"r+") as h:
            b=h["source_bank"][()]
            b["time"][0]=-1.
            h["source_bank"][...]=b
        reject(lambda:read_real_bank_angular_physics(f),"nonphysical neutron source")
        fake(f,seeds[0],80)
        with h5py.File(f,"r+") as h:
            h["n_particles"][...]=9
        reject(lambda:read_real_bank_angular_physics(f),"source population")
        fake(f,seeds[0],80)
        save_summary()

        with h5py.File(final_paths[1],"r+") as h:
            h["seed"][...]=seeds[0]
        save_summary()
        reject(lambda:audit_real_source_angular_physics(summary),"checkpoint identity/seed")
        fake(final_paths[1],seeds[1],80)
        save_summary()
        payload["runs"][1]["seed"]=seeds[0]
        summary.write_text(json.dumps(payload),encoding="utf-8")
        reject(lambda:audit_real_source_angular_physics(summary),"provenance mismatch")
        save_summary()

        with h5py.File(f,"r+") as h:
            h["current_batch"][...]=99
        reject(lambda:audit_real_source_angular_physics(summary),"checkpoint identity/seed")
        fake(f,seeds[0],80)
        # Physically false, but a valid-format synthetic clone of the old
        # checkpoint cannot be silently claimed independent of its own hash.
        save_summary()
        payload["runs"][0]["statepoint_sha256"]="0"*64
        summary.write_text(json.dumps(payload),encoding="utf-8")
        reject(lambda:audit_real_source_angular_physics(summary),"provenance mismatch")

    print("OpenMC source-bank direction, energy and clone-multiplicity checks: PASS")
    print("  synthetic HDF5 fixtures; real C++ neutron transport required separately")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
