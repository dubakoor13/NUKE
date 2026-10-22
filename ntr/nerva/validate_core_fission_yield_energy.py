"""Synthetic HDF5 regression for genuine OpenMC incident-neutron fission yield.

These HDF5 fixtures are synthetic and cannot demonstrate actual reactor
fission yield. The separate real 4-seed OpenMC 0.15.3 C++ workflow reads
genuine evaluated-data statepoints and performs the physical comparison.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import h5py

from .core_fission_yield_energy import (
    read_fission_yield_checkpoint,calculate_fission_yield_windows
)


def reject(fn, phrase):
    try:
        fn()
    except ValueError as exc:
        assert phrase in str(exc), (phrase,str(exc))
    else:
        raise AssertionError("invalid evaluated neutron physics accepted: "+phrase)


def _file(path, seed, batch, rates, *, corrupt_type=False,
          bad_realizations=False, bad_materials=False):
    energy_edges=np.array([
        1e-5,3.96850263e-4,1.57490131e-2,.625,33.9302202,
        1842.01575,1e5,1414213.56,2e7
    ])
    mats=[19,20]
    n=batch-30
    with h5py.File(path,"w") as h:
        h.attrs["filetype"]=np.bytes_("statepoint")
        h.attrs["photon_transport"]=0
        for key,data in (("run_mode",np.bytes_("eigenvalue")),("seed",seed),
                         ("current_batch",batch),("n_inactive",30),
                         ("n_particles",3000),("n_realizations",n)):
            h.create_dataset(key,data=data)
        grp=h.create_group("tallies")
        filt=grp.create_group("filters")
        for ident,name,values in (
            (81,"material",mats),
            (82,"energy",energy_edges),
            *(((83,"material",[19,21]),) if bad_materials else ())
        ):
            f=filt.create_group(f"filter {ident}")
            f.create_dataset("type",data=np.bytes_(name))
            f.create_dataset("bins",data=np.array(values))
            f.create_dataset("n_bins",data=len(values) if ident==81 else len(values)-1)
        for j,(name,filtered,raw) in enumerate((
            ("nerva_global_reaction_tracklength",(),rates.sum((0,1))[[1,3]][np.newaxis,:]),
            ("nerva_material_transport",(81,),rates.sum(axis=1)),
            ("nerva_material_energy_transport",((83 if bad_materials else 81),82),rates.reshape(16,4))
        )):
            t=grp.create_group(f"tally {j+1}")
            t.create_dataset("name",data=np.bytes_(name))
            t.create_dataset("estimator",data=np.bytes_(
                "analog" if corrupt_type and j==2 else "tracklength"))
            t.create_dataset("n_realizations",data=n+int(bad_realizations))
            t.create_dataset("n_filters",data=len(filtered))
            if filtered:
                t.create_dataset("filters",data=np.asarray(filtered,dtype=int))
            names=(("absorption","nu-fission") if not filtered else
                   ("flux","absorption","fission","nu-fission"))
            t.create_dataset("score_bins",data=np.asarray(names,dtype="S24"))
            data=np.zeros((*raw.shape,2))
            data[:,:,0]=n*raw
            data[:,:,1]=n*(raw**2+1e-12)
            t.create_dataset("results",data=data)


def main() -> int:
    fis=np.array([1e-6,.0004,.0018,.00045,.0005,.0002,.00025,.0011])
    nu=np.array([2.436,2.436,2.436,2.436,2.434,2.431,2.509,2.77])
    rates=np.zeros((2,8,4))
    rates[0,:,0]=np.array([0.01,1,10,20,30,40,50,60])
    rates[0,:,1]=np.array([.0001,.0002,.0003,.0004,.0005,.0006,.0007,.0008])
    rates[0,:,2]=fis
    rates[0,:,3]=fis*nu
    rates[1,:,0]=rates[0,:,0]*2
    rates[1,:,1]=rates[0,:,1]*3
    with TemporaryDirectory() as dirname:
        root=Path(dirname)
        for seed in (13007,23011,33013,43019):
            rows=[]
            for batch in (40,60,80,100,120):
                path=root/f"{seed}_{batch}.h5"
                _file(path,seed,batch,rates)
                rows.append(read_fission_yield_checkpoint(path))
            report=calculate_fission_yield_windows(rows,19)
            assert report["energy_resolved_fission_yield_physics_checks_passed"]
            assert len(report["disjoint_windows"])==5
            assert all(w["fuel_origin_and_group_reaction_conservation_passed"]
                       for w in report["disjoint_windows"])
            assert all(abs(w["effective_neutrons_per_fission"]-(
                float(np.dot(fis,nu)/fis.sum())))<1e-12 for w in report["disjoint_windows"])
            assert all(w["thermal"]["yield_neutrons_per_fission"]>2.4 for w in
                       report["disjoint_windows"])
            assert all(w["fast"]["yield_neutrons_per_fission"]>2.5 for w in
                       report["disjoint_windows"])
            assert report["full_run_fission_neutrons_per_source"]>0

        rows=[read_fission_yield_checkpoint(root/f"13007_{batch}.h5")
              for batch in (40,60,80,100,120)]
        fail=deepcopy(rows)
        fail[-1]["seed"]=999
        reject(lambda:calculate_fission_yield_windows(fail,19),"source seed")
        fail=deepcopy(rows)
        fail[-1]["energy_edges_ev"][-1] *=2
        reject(lambda:calculate_fission_yield_windows(fail,19),"energy")
        fail=deepcopy(rows)
        fail[-1]["material_energy"][0,2,3] += 1.
        reject(lambda:calculate_fission_yield_windows(fail,19),"production")
        fail=deepcopy(rows)
        for row in fail:
            # Internally coherent but impossible huge neutron yield.
            row["material_energy"][0,:,3] *= 4.
            row["material"][0,3] *= 4.
            row["global_nufission"] *= 4.
        reject(lambda:calculate_fission_yield_windows(fail,19),"outside physical guard")
        fail=deepcopy(rows)
        for row in fail:
            # Structural material (hydrogen) must not produce fission neutrons
            # while source tally and material partitions remain conservative.
            row["material_energy"][1,3,2] += 1e-3*row["active"]
            row["material_energy"][1,3,3] += 2.5e-3*row["active"]
            row["material"][1,2] += 1e-3*row["active"]
            row["material"][1,3] += 2.5e-3*row["active"]
            row["global_nufission"] += 2.5e-3*row["active"]
        reject(lambda:calculate_fission_yield_windows(fail,19),"outside the declared fuel")
        reject(lambda:calculate_fission_yield_windows(rows,999),"fuel missing")

        bogus=root/"invalid.h5"
        _file(bogus,999,120,rates,corrupt_type=True)
        reject(lambda:read_fission_yield_checkpoint(bogus),"estimator")
        _file(bogus,999,120,rates,bad_realizations=True)
        reject(lambda:read_fission_yield_checkpoint(bogus),"realization")
        _file(bogus,999,120,rates,bad_materials=True)
        reject(lambda:read_fission_yield_checkpoint(bogus),"material neutron filter")
    print("NERVA evaluated incident-energy fuel fission/nu-fission yield physics: PASS")
    print("  synthetic HDF5 fixtures only; genuine OpenMC C++ real-statepoint run required")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
