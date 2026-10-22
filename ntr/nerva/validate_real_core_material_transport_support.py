"""Synthetic adversarial HDF5 checks for physical material neutron visitation.

Synthetic tally bytes are numerical tests ONLY. Official four-seed C++
OpenMC transport produces the independent physical evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .real_core_material_transport_support import (
    _sha, real_material_neutron_flux, audit_real_material_transport,
)


def reject(fn, error: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert error in str(exc), (error, str(exc))
    else:
        raise AssertionError("physically inconsistent neutron-material evidence accepted: "+error)


def fixture(file: Path, *, seed: int, batch: int, flux: tuple[float,...]) -> None:
    number=batch-10
    if len(flux)!=5:
        raise ValueError("test fixture requires five defined material bins")
    with h5py.File(file, "w") as h:
        h.attrs["filetype"]=np.bytes_("statepoint")
        h.attrs["source_present"]=1
        h.attrs["photon_transport"]=0
        h.attrs["openmc_version"]=np.asarray([0,15,3])
        h.create_dataset("run_mode",data=np.bytes_("eigenvalue"))
        h.create_dataset("seed",data=seed)
        h.create_dataset("current_batch",data=batch)
        h.create_dataset("n_inactive",data=10)
        tallies=h.create_group("tallies")
        filters=tallies.create_group("filters")
        filtr=filters.create_group("filter 81")
        filtr.create_dataset("type",data=np.bytes_("material"))
        filtr.create_dataset("bins",data=[19,20,21,22,23])
        t=tallies.create_group("tally 82")
        t.create_dataset("name",data=np.bytes_("nerva_material_transport"))
        t.create_dataset("estimator",data=np.bytes_("tracklength"))
        t.create_dataset("n_filters",data=1)
        t.create_dataset("n_realizations",data=number)
        t.create_dataset("n_score_bins",data=2)
        t.create_dataset("filters",data=[81])
        t.create_dataset("score_bins",data=np.asarray([b"flux",b"absorption"]))
        t.create_dataset("nuclides",data=np.asarray([b"total"]))
        scores=np.zeros((5,2,2),dtype=float)
        for i,v in enumerate(flux):
            scores[i,0,0]=v*number
            scores[i,0,1]=v*v*number
        t.create_dataset("results",data=scores)


def main() -> int:
    seeds=(13007,23011,33013,43019)
    batches=(20,30,40)
    expected=(8.,4.,2.,20.,0.)
    with TemporaryDirectory() as tmp:
        root=Path(tmp)
        runs=[]
        for seed in seeds:
            folder=root/str(seed)
            folder.mkdir()
            for batch in batches:
                fixture(folder/f"statepoint.{batch:02}.h5",
                        seed=seed,batch=batch,flux=expected)
            final=folder/"statepoint.40.h5"
            runs.append({"seed":seed,"statepoint_path":str(final),
                         "statepoint_sha256":_sha(final)})
        summary_path=root/"summary.json"
        kerma_path=root/"kerma.json"
        summary={
            "execution":"ACTUAL_OPENMC_TRANSPORT",
            "assembly":"core",
            "batches":40,
            "saved_source_bank_batches":list(batches),
            "runs":runs,
        }
        kerma={
            "real_independent_seed_count":4,
            "physical_material_count":4,
            "declared_material_count":5,
            "executed_reachable_material_ids":[19,20,21,22],
            "physical_materials":[{"material_id":mid} for mid in [19,20,21,22]],
            "uninstantiated_declared_materials":[{"material_id":23}],
        }
        def save():
            summary_path.write_text(json.dumps(summary),encoding="utf-8")
            kerma["transport_summary_sha256"]=_sha(summary_path)
            kerma_path.write_text(json.dumps(kerma),encoding="utf-8")
        def audit():
            save()
            return audit_real_material_transport(summary_path,kerma_path)
        report=audit()
        assert report["four_independent_seed_count"]==4
        assert report["verified_statepoint_checkpoint_count"]==12
        assert report["reachable_material_ids"]==[19,20,21,22]
        assert report["uninstantiated_material_ids"]==[23]
        assert report["observed_reachable_material_ids_with_nonzero_flux"]==[19,20,21,22]
        assert report["all_unplaced_materials_have_exact_zero_neutron_tracklength"]
        assert report["material_transport_support_status"]=="ALL_REACHABLE_SAMPLED_AND_UNPLACED_ZERO"
        assert not report["nasa_nerva_core_physically_validated"]
        first=Path(runs[0]["statepoint_path"])
        row=real_material_neutron_flux(first,{19,20,21,22,23})
        assert row["mean_neutron_tracklength_by_material"]["19"]==8.
        assert row["mean_neutron_tracklength_by_material"]["23"]==0.

        # Previously unused material cannot acquire a positive real tally
        # while the independently proven root-geometry path excludes it.
        fixture(first,seed=seeds[0],batch=40,flux=(8.,4.,2.,20.,1.))
        runs[0]["statepoint_sha256"]=_sha(first)
        reject(audit,"neutron flux scored in unreachable")
        fixture(first,seed=seeds[0],batch=40,flux=expected)
        runs[0]["statepoint_sha256"]=_sha(first)

        # Reachable but unvisited is a coverage failure, not physical void.
        fixture(first,seed=seeds[0],batch=40,flux=(8.,4.,0.,20.,0.))
        runs[0]["statepoint_sha256"]=_sha(first)
        gap=audit()
        assert gap["material_transport_support_status"]=="REACHABLE_MATERIAL_NEUTRON_SAMPLING_INCOMPLETE"
        assert gap["genuine_checkpoint_material_flux"][2][
            "reachable_material_ids_with_no_observed_flux"]==[21]
        fixture(first,seed=seeds[0],batch=40,flux=expected)
        runs[0]["statepoint_sha256"]=_sha(first)

        # Tampered HDF5 cannot pose as previously recorded neutron transport.
        with h5py.File(first,"r+") as h:
            h["seed"][...]=77
        reject(audit,"statepoint provenance mismatch")
        fixture(first,seed=seeds[0],batch=40,flux=expected)
        runs[0]["statepoint_sha256"]=_sha(first)

        with h5py.File(first,"r+") as h:
            h["tallies"]["tally 82"]["results"][4,0,0]=20.
        reject(lambda:real_material_neutron_flux(first,{19,20,21,22,23}),
               "nonphysical")
        fixture(first,seed=seeds[0],batch=40,flux=expected)
        with h5py.File(first,"r+") as h:
            h["tallies"]["filters"]["filter 81"]["bins"][4]=999
        reject(lambda:real_material_neutron_flux(first,{19,20,21,22,23}),
               "material IDs disagree")
        fixture(first,seed=seeds[0],batch=40,flux=expected)
        with h5py.File(first,"r+") as h:
            h["tallies"]["tally 82"]["estimator"][...]=np.bytes_("analog")
        reject(lambda:real_material_neutron_flux(first,{19,20,21,22,23}),
               "wrong OpenMC neutron material flux estimator")
        fixture(first,seed=seeds[0],batch=40,flux=expected)
        runs[0]["statepoint_sha256"]=_sha(first)
        save()
        kerma["transport_summary_sha256"]="a"*64
        kerma_path.write_text(json.dumps(kerma))
        reject(lambda:audit_real_material_transport(summary_path,kerma_path),
               "physical provenance")

    print("Native material-filter neutron tracklength vs geometry reachability: PASS")
    print("  Synthetic HDF5 regression only; real four-seed OpenMC transport checked separately")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
