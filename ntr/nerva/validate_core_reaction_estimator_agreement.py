"""Synthetic HDF5 regression of REAL-format neutron estimator physics arithmetic.

Fixtures are not neutron transport; the real CI executes OpenMC 0.15.3
and calls the same parser on four actual independent-seed statepoints.
"""
from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path
import json
import numpy as np
import h5py

from .core_reaction_estimator_agreement import (
    _checkpoint, compare_estimator_windows,
    audit_actual_reaction_estimators
)


def emit(path, seed, batch, inactive, cumulative_a, cumulative_t,
         *, wrong_estimator=False, wrong_realizations=False,
         wrong_filters=False):
    with h5py.File(path,"w") as h:
        h.attrs["filetype"]=np.bytes_("statepoint")
        h.attrs["photon_transport"]=0
        for name,value in (("run_mode",np.bytes_("eigenvalue")),
                           ("seed",seed),("current_batch",batch),
                           ("n_inactive",inactive),("n_particles",3000)):
            h.create_dataset(name,data=value)
        tallies=h.create_group("tallies")
        for i,(tally_name,estimator,cumulative,scores) in enumerate((
            ("nerva_global_neutron_balance","analog",cumulative_a,
             ("absorption","scatter","nu-scatter","nu-fission")),
            ("nerva_global_reaction_tracklength","tracklength",cumulative_t,
             ("absorption","nu-fission")),
        )):
            t=tallies.create_group(f"tally {i+2}")
            t.create_dataset("name",data=np.bytes_(tally_name))
            t.create_dataset("estimator",data=np.bytes_(
                "analog" if (wrong_estimator and i==1) else estimator))
            t.create_dataset("n_realizations",data=batch-inactive+
                             int(wrong_realizations))
            t.create_dataset("n_filters",data=int(wrong_filters))
            t.create_dataset("score_bins",data=np.asarray(scores,dtype="S24"))
            data=np.zeros((1,len(scores),2))
            for j,score in enumerate(("absorption","nu-fission")):
                idx=scores.index(score)
                data[0,idx,:]=cumulative[j,:]
            t.create_dataset("results",data=data)


def fail(fn,needle):
    try:
        fn()
    except ValueError as ex:
        assert needle in str(ex), (needle,str(ex))
    else:
        raise AssertionError("nonphysical OpenMC estimator history accepted: "+needle)


def main():
    batches=(40,60,80,100,120)
    seeds=(13007,23011,33013,43019)
    inactive=30
    rates=[(.04,.0117),(.041,.0118),(.039,.0115),
           (.0401,.0116),(.0402,.0119)]
    with TemporaryDirectory() as temp:
        root=Path(temp)
        trajectories=[]
        for seed in seeds:
            prefix=root/f"seed_{seed}"
            prefix.mkdir()
            analog=np.zeros((2,2))
            track=np.zeros((2,2))
            previous=0
            paths=[]
            for j,batch in enumerate(batches):
                n=batch-inactive
                width=n-previous
                # Four test batches with slightly different true per-source
                # estimator samples. Nonzero variance prevents false certainty.
                for k,(rate_a,rate_t) in enumerate(zip(rates[j],rates[j])):
                    analytic=np.array([rate_a,rate_t])
                    analog[k,0]+=width*analytic[k]
                    analog[k,1]+=width*(analytic[k]**2+2.0e-6)
                    track[k,0]+=width*(analytic[k]*1.005)
                    track[k,1]+=width*((analytic[k]*1.005)**2+1.0e-6)
                path=prefix/f"statepoint.{batch:03d}.h5"
                emit(path,seed,batch,inactive,analog,track)
                paths.append(path)
                previous=n
            trajectories.append([_checkpoint(path) for path in paths])
        r=compare_estimator_windows(trajectories)
        assert r["matched_independent_seed_count"]==4
        assert r["gross_neutron_reaction_estimator_agreement_screen_passed"]
        assert r["saved_statepoint_batches"]==list(batches)
        assert r["fission_source_stationarity_validated"] is False
        assert r["nasa_nerva_physical_core_validated"] is False
        for run in r["within_seed_windows"]:
            assert [s["active_generations"] for s in run["window_reactions"]]==[
                10,20,20,20,20]
            for w in run["window_reactions"]:
                for score in ("absorption","nu-fission"):
                    vals=w["reactions"][score]
                    assert vals["analog_per_source"]>0
                    assert vals["tracklength_per_source"]>0
                    assert vals["sum_of_marginal_batch_standard_errors"]>0

        fail(lambda:compare_estimator_windows(trajectories[:3]),"four independent")
        fail(lambda:compare_estimator_windows([trajectories[0]]*4),
             "duplicate OpenMC independent seed")
        fail(lambda:compare_estimator_windows(
            [list(reversed(trajectories[0]))]+trajectories[1:]),"invalid OpenMC")
        altered=[list(t) for t in trajectories]
        changed=dict(altered[0][-1])
        changed["analog"]=changed["analog"].copy()
        changed["analog"][0,0]=-12.0
        altered[0][-1]=changed
        fail(lambda:compare_estimator_windows(altered),"cumulative sums or squares")
        altered=[list(t) for t in trajectories]
        # A whole-seed, mathematically valid constant factor error in the
        # tracklength cross section: every sum doubles, each sum of squares
        # quadruples. A permissive variance alone cannot hide the physics.
        for idx, row in enumerate(altered[0]):
            changed=dict(row)
            changed["tracklength"]=row["tracklength"].copy()
            changed["tracklength"][:,0] *= 2.
            changed["tracklength"][:,1] *= 4.
            altered[0][idx]=changed
        # Catastrophically discordant estimators may exceed the generous
        # statistical screen without tampering with reactor geometry.
        measured=compare_estimator_windows(altered)
        assert not measured["gross_neutron_reaction_estimator_agreement_screen_passed"]

        bad=root/"bad.h5"
        zeros=np.array([[.40,.017],[.12,.006]])
        emit(bad,999,120,inactive,zeros,zeros,wrong_estimator=True)
        fail(lambda:_checkpoint(bad),"estimator mismatch")
        emit(bad,999,120,inactive,zeros,zeros,wrong_realizations=True)
        fail(lambda:_checkpoint(bad),"active generations")
        emit(bad,999,120,inactive,zeros,zeros,wrong_filters=True)
        fail(lambda:_checkpoint(bad),"whole core")
        empty=root/"fake-summary.json"
        empty.write_text(json.dumps({"execution":"SYNTHETIC","assembly":"core"}))
        fail(lambda:audit_actual_reaction_estimators(empty),"genuine OpenMC")

    print("NERVA native analog-vs-tracklength neutron reaction batch SEM/physics: PASS")
    print("  synthetic structured HDF5; actual OpenMC is a separate required CI run")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
