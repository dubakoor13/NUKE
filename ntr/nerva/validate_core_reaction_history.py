"""Numerical checks for OpenMC disjoint active-neutron reaction windows.

Synthetic HDF5 structures only. The real C++ OpenMC CI job must execute
this parser against genuine saved 4-seed statepoints and preserve hashes.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import h5py

from .core_reaction_history import read_checkpoint, calculate_disjoint_windows, audit_real_reaction_windows


def reject(call, text):
    try:
        call()
    except ValueError as exc:
        assert text in str(exc), str(exc)
    else:
        raise AssertionError("invalid active neutron reaction history accepted: "+text)


def emit(path, *, seed, batch, inactive, totals, material=2, groups=3,
         distort_material=False, wrong_realizations=False):
    N=batch-inactive
    with h5py.File(path,"w") as h:
        h.attrs["filetype"]=np.bytes_("statepoint")
        h.attrs["photon_transport"]=0
        h.create_dataset("run_mode", data=np.bytes_("eigenvalue"))
        h.create_dataset("seed", data=seed)
        h.create_dataset("current_batch", data=batch)
        h.create_dataset("n_inactive", data=inactive)
        h.create_dataset("n_particles", data=3000)
        tt=h.create_group("tallies")
        # OpenMC's HDF5 results[...,0] is SUM of per-active-generation
        # tally contributions, NOT the per-generation mean.
        items=[
            ("nerva_global_reaction_tracklength",
             ("absorption","nu-fission"),
             np.asarray(totals,dtype=float).reshape(1,2)),
            ("nerva_material_transport",
             ("flux","absorption","nu-fission"),
             np.column_stack([np.zeros(material),
                              np.array(totals[0]*np.array([.7,.3])),
                              np.array(totals[1]*np.array([1.,0.]))])),
        ]
        incident=np.zeros((material*groups,3),dtype=float)
        for i in range(material):
            incident[i*groups:(i+1)*groups,1]=(
                items[1][2][i,1]*np.array([.2,.3,.5]))
            incident[i*groups:(i+1)*groups,2]=(
                items[1][2][i,2]*np.array([.2,.3,.5]))
        items.append((
            "nerva_material_energy_transport",
            ("flux","absorption","nu-fission"),incident))
        for i,(name,scores,values) in enumerate(items):
            obj=tt.create_group(f"tally {i+1}")
            obj.create_dataset("name", data=np.bytes_(name))
            obj.create_dataset("estimator", data=np.bytes_("tracklength"))
            obj.create_dataset("score_bins",data=np.asarray(scores,dtype="S24"))
            obj.create_dataset("n_realizations",data=N+int(wrong_realizations))
            arr=np.zeros((values.shape[0],values.shape[1],2))
            arr[:,:,0]=values
            arr[:,:,1]=values*values
            if distort_material and i==1:
                arr[0,1,0]*=.8
            obj.create_dataset("results",data=arr)


def main() -> int:
    with TemporaryDirectory() as d:
        root=Path(d)
        seed=13007
        batches=[40,60,80,100,120]
        inactive=30
        reference_rates=[
            (.040,.0117),(.041,.0116),(.039,.0119),
            (.0405,.0115),(.0395,.0118),
        ]
        old=np.zeros(2)
        old_n=0
        paths=[]
        for batch,rate in zip(batches,reference_rates):
            N=batch-inactive
            old=old+(N-old_n)*np.asarray(rate)
            old_n=N
            path=root/f"{batch}.h5"
            emit(path,seed=seed,batch=batch,inactive=inactive,totals=old)
            paths.append(path)
        rows=[read_checkpoint(p) for p in paths]
        valid=calculate_disjoint_windows(rows)
        assert valid["seed"]==seed
        assert [w["number_of_active_generations"] for w in valid["windows"]]==[10,20,20,20,20]
        assert [w["first_active_batch"] for w in valid["windows"]]==[31,41,61,81,101]
        assert valid["final_cumulative_reaction_reconstruction_passed"]
        for w,(a,nu) in zip(valid["windows"],reference_rates):
            assert np.isclose(w["global_reaction_rates_per_source"]["absorption"],a)
            assert np.isclose(w["global_reaction_rates_per_source"]["nu-fission"],nu)
        reject(lambda:calculate_disjoint_windows(rows[::-1]),"not strictly increasing")
        reject(lambda:calculate_disjoint_windows(rows[:2]),"three")
        duplicate=[*rows]
        duplicate[1]={**duplicate[1],"seed":43019}
        reject(lambda:calculate_disjoint_windows(duplicate),"changed physical")
        nonconservative=[*rows]
        changed=dict(nonconservative[-1])
        changed["material"]=changed["material"].copy()
        changed["material"][0,0]*=.8
        nonconservative[-1]=changed
        reject(lambda:calculate_disjoint_windows(nonconservative),"do not conserve")
        corrupted=root/"bad.h5"
        emit(corrupted,seed=seed,batch=120,inactive=inactive,totals=old,
             wrong_realizations=True)
        reject(lambda:read_checkpoint(corrupted),"realization")
        emit(corrupted,seed=seed,batch=120,inactive=inactive,totals=old,
             distort_material=True)
        reject(lambda:calculate_disjoint_windows(rows[:-1]+[
            read_checkpoint(corrupted)]),"do not conserve")
        fake=root/"summary.json"
        fake.write_text(json.dumps({"execution":"ARTIFICIAL","assembly":"core"}))
        reject(lambda:audit_real_reaction_windows(fake),"genuine")

    print("OpenMC cumulative-to-disjoint neutron reaction history mathematics: PASS")
    print("  artificial HDF5 only; real OpenMC C++ statepoints tested separately")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
