"""Synthetic HDF5 IO tests of OpenMC fission-source spatial diagnostics.

Constructed source banks do NOT come from neutron transport. Production
runs must provide actual HDF5 statepoints from the OpenMC executable.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import h5py
import numpy as np

from .source_bank_convergence import read_source_bank, compare_independent_sources


def fake(path, *, seed, batch, xyz):
    nested=np.dtype([("x","f8"),("y","f8"),("z","f8")])
    dtype=np.dtype([("r",nested),("wgt","f8")])
    records=np.zeros(len(xyz),dtype=dtype)
    for axis in ("x","y","z"):
        records["r"][axis]=xyz[:,("x","y","z").index(axis)]
    records["wgt"]=1.
    with h5py.File(path,"w") as f:
        f.attrs["filetype"]=np.bytes_("statepoint")
        f.attrs["source_present"]=1
        f.create_dataset("run_mode",data=np.bytes_("eigenvalue"))
        f.create_dataset("seed",data=seed)
        f.create_dataset("current_batch",data=batch)
        f.create_dataset("source_bank",data=records)


def reject(fn,needle):
    try:
        fn()
    except ValueError as exc:
        assert needle in str(exc),str(exc)
    else:
        raise AssertionError("unexpected source-bank acceptance: "+needle)


def main():
    rng=np.random.default_rng(20261021)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        pairs=[]
        for seed in (13007,23011,33013,43019):
            xyz=rng.normal(0,1,size=(1000,3))
            xyz[:,2] *= 7.
            early=root/f"{seed}_60.h5"
            late=root/f"{seed}_120.h5"
            fake(early,seed=seed,batch=60,xyz=xyz)
            fake(late,seed=seed,batch=120,xyz=xyz+0.05*rng.normal(size=xyz.shape))
            pairs.append((early,late))
        bounds=(-5.,-5.,-66.),(5.,5.,66.)
        histogram,details=read_source_bank(
            pairs[0][1],lower_cm=bounds[0],upper_cm=bounds[1]
        )
        assert np.isclose(histogram.sum(),1.)
        assert details["seed"]==13007
        assert details["completed_batch"]==120
        assert details["source_sites"]==1000
        assert details["exact_distinct_site_positions"]==1000
        assert details["distinct_position_fraction"]==1.
        assert 900 < details["inverse_simpson_distinct_position_count"] <= 1000
        assert details["largest_replicated_position_weight_fraction"] == 0.001
        report=compare_independent_sources(
            pairs,lower_cm=bounds[0],upper_cm=bounds[1]
        )
        assert report["independent_seed_count"]==4
        assert report["between_seed_max_source_total_variation"]>0
        assert report["between_seed_mean_source_total_variation"]>0
        assert report["mean_early_to_late_source_total_variation"]>=0
        assert report["source_convergence_independently_demonstrated"] is False
        assert report["historical_nasa_nerva_benchmark_validated"] is False

        # Deliberately cloned spatial source populations: exact position
        # degeneracy is visible without mistaking nominal bank size for
        # independent fission locations.
        clones=np.repeat(np.array([[0.1,0.2,0.3],[0.2,0.3,0.4]]),
                         500, axis=0)
        clone_file=root/"clones.h5"
        fake(clone_file,seed=99017,batch=120,xyz=clones)
        _, cloned=read_source_bank(
            clone_file,lower_cm=bounds[0],upper_cm=bounds[1]
        )
        assert cloned["exact_distinct_site_positions"]==2
        assert cloned["inverse_simpson_distinct_position_count"]==2.
        assert cloned["largest_replicated_position_weight_fraction"]==0.5

        reject(
            lambda:compare_independent_sources(
                pairs[:3],lower_cm=bounds[0],upper_cm=bounds[1]
            ),"at least four",
        )
        reject(
            lambda:compare_independent_sources(
                pairs[:3]+[pairs[0]],
                lower_cm=bounds[0],upper_cm=bounds[1]
            ),"duplicate",
        )
        reject(
            lambda:compare_independent_sources(
                [pairs[0][::-1]]+pairs[1:],
                lower_cm=bounds[0],upper_cm=bounds[1]
            ),"time order",
        )
        reject(
            lambda:read_source_bank(
                pairs[0][1],lower_cm=(-.1,-.1,-.1),
                upper_cm=(.1,.1,.1)
            ),"outside",
        )
        reject(
            lambda:read_source_bank(
                pairs[0][1],lower_cm=bounds[0],upper_cm=bounds[1],
                bins=(1,4,4)
            ),"bins",
        )

    print("NERVA actual-statepoint source-bank diagnostics numerics: PASS")
    print("  HDF5 fixture: synthetic; real neutron transport: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
