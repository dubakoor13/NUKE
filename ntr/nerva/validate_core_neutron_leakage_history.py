"""Numerical regressions for genuine-format OpenMC neutron leakage histories.

Constructed HDF5 fixtures ONLY. The physical workflow separately runs the
official OpenMC C++ eigenvalue solver on evaluated neutron cross sections.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .core_neutron_leakage_history import (
    read_native_neutron_checkpoint,
    check_native_neutron_windows,
)


def reject(call, expected):
    try:
        call()
    except ValueError as err:
        assert expected in str(err), (expected,str(err))
    else:
        raise AssertionError("invalid physical neutron balance accepted: "+expected)


def _fixture(path, seed, batch, analog, native, *,
             wrong_row=False, wrong_realizations=False, wrong_filter=False):
    n=batch-30
    with h5py.File(path,"w") as h:
        h.attrs["filetype"]=np.bytes_("statepoint")
        h.attrs["photon_transport"]=0
        for k,v in (
            ("run_mode",np.bytes_("eigenvalue")),
            ("seed",seed),("current_batch",batch),
            ("n_inactive",30),("n_particles",3000),
            ("n_realizations",n),
        ):
            h.create_dataset(k,data=v)
        global_tally=np.zeros((4,3))
        global_tally[2,1:]=native
        global_tally[3,1:]=analog[4]
        if wrong_row:
            global_tally[2,1]*=.7
        h.create_dataset("global_tallies",data=global_tally)
        tallies=h.create_group("tallies")
        for i,(name,method,scores,arrays) in enumerate((
            ("nerva_global_neutron_balance","analog",
             ("absorption","scatter","nu-scatter","nu-fission"),analog[:4]),
            ("nerva_global_reaction_tracklength","tracklength",
             ("absorption","nu-fission"),
             np.stack([analog[0],native])),
        )):
            t=tallies.create_group(f"tally {i+2}")
            t.create_dataset("name",data=np.bytes_(name))
            t.create_dataset("estimator",data=np.bytes_(method))
            t.create_dataset("n_realizations",data=n+int(wrong_realizations))
            t.create_dataset("n_filters",data=int(wrong_filter))
            t.create_dataset("score_bins",data=np.asarray(scores,dtype="S24"))
            t.create_dataset("results",data=arrays[np.newaxis,:,:])


def main() -> int:
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        history=[]
        for seed in (13007,23011,33013,43019):
            folder=root/f"seed_{seed}"
            folder.mkdir()
            prior=0
            raw=np.zeros((5,2))
            nu=np.zeros(2)
            one=[]
            for j,b in enumerate((40,60,80,100,120)):
                n=b-30
                width=n-prior
                absorb=.04+.0001*j
                scatter=14.1
                extra=.064+.0002*j
                leak=1.0-absorb+extra
                rates=np.array([absorb,scatter,scatter+extra,.0117,leak])
                raw[:,0]+=width*rates
                raw[:,1]+=width*(rates*rates+2.0e-7)
                nu[0]+=width*.0117
                nu[1]+=width*(.0117**2+2.0e-7)
                filename=folder/f"statepoint.{b:03d}.h5"
                _fixture(filename,seed,b,raw,nu)
                one.append(read_native_neutron_checkpoint(filename))
                prior=n
            history.append(one)
            report=check_native_neutron_windows(one)
            assert report["all_disjoint_neutron_conservation_windows_passed"]
            assert report["maximum_absolute_disjoint_neutron_conservation_residual"]<1e-11
            assert [x["active_realizations"] for x in report["disjoint_windows"]]==[
                10,20,20,20,20]
            assert all(x["native_vacuum_leakage_per_source"]>1 for x in
                       report["disjoint_windows"])
            assert all(x["net_multiplicative_scattering_per_source"]>0 for x in
                       report["disjoint_windows"])
        faulty=deepcopy(history[0])
        faulty[-1]["cumulative_analog_abs_scatter_nuscatter_nufission_leakage"][
            4,0] += 20.
        result=check_native_neutron_windows(faulty)
        assert not result["all_disjoint_neutron_conservation_windows_passed"]
        assert result["maximum_absolute_disjoint_neutron_conservation_residual"]>.1

        faulty=deepcopy(history[0])
        faulty[-1]["cumulative_analog_abs_scatter_nuscatter_nufission_leakage"][
            4,0] = 0.
        reject(lambda:check_native_neutron_windows(faulty),"negative disjoint")

        faulty=list(reversed(history[0]))
        reject(lambda:check_native_neutron_windows(faulty),"invalid active-generation")

        faulty=deepcopy(history[0])
        faulty[1]["seed"]=32009
        reject(lambda:check_native_neutron_windows(faulty),"mixed nuclear transport seeds")

        filename=root/"broken.h5"
        example=history[0][-1]
        native=example["cumulative_native_k"][2]
        analog=example["cumulative_analog_abs_scatter_nuscatter_nufission_leakage"]
        _fixture(filename,999,120,analog,native,wrong_row=True)
        reject(lambda:read_native_neutron_checkpoint(filename),"k-tracklength row")
        _fixture(filename,999,120,analog,native,wrong_realizations=True)
        reject(lambda:read_native_neutron_checkpoint(filename),"estimator and active generations")
        _fixture(filename,999,120,analog,native,wrong_filter=True)
        reject(lambda:read_native_neutron_checkpoint(filename),"estimator or clipped")
    print("NERVA genuine-format OpenMC neutron analog leakage/history conservation: PASS")
    print("  synthetic fixtures; official OpenMC four-seed reactor run separately required")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
