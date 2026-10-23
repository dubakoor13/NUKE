"""Synthetic adversarial tests; compiled OpenMC C++ CI is separate."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .core_prompt_delayed_fission_validation import (
    SCORES, assess_fission_emission_window,read_real_checkpoint,
)


def reject(fn, fragment: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert fragment in str(exc),str(exc)
    else:
        raise AssertionError("invalid neutron production evidence accepted: "+fragment)


def _tally(group, name, index, scores, means, n, *, material: bool):
    item=group.create_group(f"tally {index}")
    item.create_dataset("name",data=np.bytes_(name))
    item.create_dataset("estimator",data=np.bytes_("tracklength"))
    item.create_dataset("n_realizations",data=n)
    item.create_dataset("n_filters",data=int(material))
    if material:
        item.create_dataset("filters",data=[11])
    # Genuine OpenMC 0.15.3 unfiltered tallies omit this HDF5 dataset.
    item.create_dataset("nuclides",data=[np.bytes_("total")])
    item.create_dataset("score_bins",data=[np.bytes_(score) for score in scores])
    values=np.asarray(means,dtype=float)
    raw=np.zeros((*values.shape,2))
    raw[:,:,0]=values*n
    raw[:,:,1]=(values**2 + .001)*n
    item.create_dataset("results",data=raw)


def fake(path: Path, *, batch=40, wrong_prompt=False):
    n=batch-30
    fission=np.array([.03,.0298,.0002])
    mat=np.array([fission,[0.,0.,0.],[0.,0.,0.]])
    if wrong_prompt:
        mat[0,1]=.025
    with h5py.File(path,"w") as sp:
        sp.attrs["filetype"]=np.bytes_("statepoint")
        sp.attrs["openmc_version"]=np.array([0,15,3])
        sp.attrs["photon_transport"]=0
        sp.attrs["source_present"]=1
        sp.create_dataset("run_mode",data=np.bytes_("eigenvalue"))
        sp.create_dataset("seed",data=13007)
        sp.create_dataset("current_batch",data=batch)
        sp.create_dataset("n_inactive",data=30)
        sp.create_dataset("n_realizations",data=n)
        tallies=sp.create_group("tallies")
        filters=tallies.create_group("filters")
        m=filters.create_group("filter 11")
        m.create_dataset("type",data=np.bytes_("material"))
        m.create_dataset("bins",data=[19,20,21])
        _tally(tallies,"nerva_global_fission_emission",1,SCORES,
               fission.reshape(1,3),n,material=False)
        _tally(tallies,"nerva_material_fission_emission",2,SCORES,
               mat,n,material=True)
        _tally(tallies,"nerva_global_reaction_tracklength",3,
               ("nu-fission",),np.array([[.03]]),n,material=False)
        _tally(tallies,"nerva_material_transport",4,
               ("nu-fission",),np.array([[.03],[0.],[0.]]),
               n,material=True)


def main()->int:
    ids=[19,20,21]
    sums=np.array([.3,.298,.002])
    materials=np.array([sums,[0.,0.,0.],[0.,0.,0.]])
    output=assess_fission_emission_window(
        sums,materials,.3,np.array([.3,0.,0.]),ids,{19,20},
        number_of_active_batches=10
    )
    assert abs(output["delayed_fraction_of_evaluated_emission_source"]-1/150)<1e-13
    assert output["maximum_absolute_raw_window_closure_per_score"]<1e-15
    assert not output["effective_delayed_neutron_fraction_beta_eff_computed"]
    assert output["material_source_per_generation_source"][2]["nu-fission"]==0

    reject(lambda:assess_fission_emission_window(
        sums,np.array([[.3,.25,.002],[0.,0.,0.],[0.,0.,0.]]),
        .3,np.array([.3,0.,0.]),ids,{19,20},number_of_active_batches=10),
        "material total=prompt+delayed")
    reject(lambda:assess_fission_emission_window(
        sums,materials,.32,np.array([.3,0.,0.]),ids,{19,20},
        number_of_active_batches=10),
        "independent global nu-fission")
    reject(lambda:assess_fission_emission_window(
        sums,np.array([sums,[0.,0.,0.],[.001,0.,.001]]),
        .3,np.array([.3,0.,0.]),ids,{19,20},
        number_of_active_batches=10),
        "unplaced")
    reject(lambda:assess_fission_emission_window(
        sums,materials,.3,np.array([.3,0.,0.]),ids,{19,20},
        number_of_active_batches=0),
        "invalid actual active-window")
    # A zero scored source is an unobserved value, not evidence that
    # the real reactor has no fission.
    zero=assess_fission_emission_window(
        np.zeros(3),np.zeros((3,3)),0.,np.zeros(3),
        ids,{19,20},number_of_active_batches=10)
    assert zero["delayed_fraction_of_evaluated_emission_source"] is None
    assert zero["zero_observed_fission_source_is_not_proof_of_zero_physical_fission"]

    with TemporaryDirectory() as td:
        p=Path(td)/"statepoint.40.h5"
        fake(p)
        r=read_real_checkpoint(p,set(ids))
        assert r["seed"]==13007 and r["active"]==10
        assert abs(r["global_first"][0]-.3)<1e-13
        fake(p,wrong_prompt=True)
        bad=read_real_checkpoint(p,set(ids))
        reject(lambda:assess_fission_emission_window(
            bad["global_first"],bad["material_first"],
            bad["global_reference_first"],bad["material_reference_first"],
            bad["material_ids"],{19,20},number_of_active_batches=10),
            "material total=prompt+delayed")
        fake(p)
        reject(lambda:read_real_checkpoint(p,{19,20,22}),
               "material bins differ")
        with h5py.File(p,"r+") as h:
            h["tallies/tally 1/estimator"][...]=np.bytes_("analog")
        reject(lambda:read_real_checkpoint(p,set(ids)),
               "estimator/provenance mismatch")
    print("OpenMC prompt/delayed fission-neutron source closure: PASS (synthetic)")
    print("Actual neutron emission requires compiled OpenMC real-statepoint CI")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
