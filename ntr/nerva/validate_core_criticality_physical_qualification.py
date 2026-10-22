"""Synthetic numerical tests: genuine OpenMC C++ transport is a separate CI job."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .core_criticality_physical_qualification import (
    _digest, read_executed_multiplication, evaluate_criticality_condition,
    audit_real_core_criticality,
)


def reject(fn, message: str) -> None:
    try:
        fn()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("invalid physical multiplication evidence accepted: "+message)


def fake(path: Path, seed: int, *, base: float=0.0117,
         native_bad: bool=False, nan: bool=False) -> None:
    n=90
    means=[base+3e-5,base+1e-4,base-2e-5,1.02]
    if native_bad:
        means[0]=base+0.1
    errors=[0.00017,0.00030,0.00016,0.01]
    moments=np.zeros((4,3),dtype=float)
    for i,(m,se) in enumerate(zip(means,errors)):
        moments[i,1]=n*m
        moments[i,2]=n*m*m + (n*(n-1))*(se*se)
    if nan:
        moments[1,2]=np.nan
    with h5py.File(path,"w") as h:
        h.attrs["filetype"]=np.bytes_("statepoint")
        h.attrs["openmc_version"]=np.array([0,15,3])
        h.attrs["photon_transport"]=0
        h.attrs["source_present"]=1
        h.create_dataset("run_mode",data=np.bytes_("eigenvalue"))
        h.create_dataset("seed",data=seed)
        h.create_dataset("current_batch",data=120)
        h.create_dataset("n_inactive",data=30)
        h.create_dataset("n_particles",data=3000)
        h.create_dataset("n_realizations",data=n)
        h.create_dataset("k_combined",data=np.array([base,0.00015]))
        h.create_dataset("global_tallies",data=moments)


def main() -> int:
    seeds=(13007,23011,33013,43019)
    with TemporaryDirectory() as tmp:
        base=Path(tmp)
        summary_file=base/"summary.json"
        source_file=base/"source_resolution.json"
        files=[base/f"seed_{seed}_statepoint.h5" for seed in seeds]
        for seed,file in zip(seeds,files):
            fake(file,seed)
        measurements=[read_executed_multiplication(p) for p in files]
        assert len(measurements)==4
        assert measurements[0]["active_realizations"]==90
        assert measurements[0]["native_neutron_global_estimators"][
            "k-tracklength"]["mean"]>0
        actual=evaluate_criticality_condition(measurements)
        assert actual["observed_state"]=="FAR_BELOW_CRITICAL"
        assert actual["all_native_and_combined_k_upper_screens_below_one"] is True
        assert actual["largest_native_or_combined_upper_screen_k"] < .1
        assert not actual["nuclear_reactor_physical_criticality_validated"]
        assert not actual["fission_source_stationarity_demonstrated"]

        # Near-unity eigenvalue is NOT sufficient to claim reactor physics.
        for seed,file in zip(seeds,files):
            fake(file,seed,base=.995)
        near=evaluate_criticality_condition([
            read_executed_multiplication(p) for p in files
        ])
        assert near["observed_state"]=="CRITICALITY_NOT_ESTABLISHED"
        assert not near["cold_model_may_be_claimed_a_critical_operating_nerva_core"]
        for seed,file in zip(seeds,files):
            fake(file,seed)
        measurements=[read_executed_multiplication(p) for p in files]

        summary={
            "execution":"ACTUAL_OPENMC_TRANSPORT",
            "assembly":"core",
            "batches":120,
            "inactive_batches":30,
            "particles_per_generation":3000,
            "cross_sections_xml_sha256":"c"*64,
            "runs":[{
                "seed":seed,
                "statepoint_path":str(file),
                "statepoint_sha256":_digest(file),
                "keff_mean":r["combined_k_effective"],
            } for seed,file,r in zip(seeds,files,measurements)],
        }
        def save():
            summary_file.write_text(json.dumps(summary),encoding="utf-8")
            source_file.write_text(json.dumps({
                "analysis":"EXACT_SAME_OPENMC_SOURCE_BANKS_ON_NESTED_SPATIAL_GRIDS",
                "real_transport_summary_sha256":_digest(summary_file),
                "source_stationarity_validated":False,
            }),encoding="utf-8")
        def audit():
            save()
            return audit_real_core_criticality(summary_file,source_file)
        physical=audit()
        assert physical["actual_neutron_transport_execution"] is True
        assert physical["observed_state"]=="FAR_BELOW_CRITICAL"
        assert physical["independent_seed_count"]==4
        assert physical["critical_eigenvalue_necessary_condition"]==1.
        assert not physical["historical_nerva_model_fidelity_validated"]

        changed=list(summary["runs"])
        summary["runs"]=[changed[0],changed[0],changed[2],changed[3]]
        reject(audit,"duplicate or invalid")
        summary["runs"]=changed

        fake(files[0],seeds[0],native_bad=True)
        reject(lambda:read_executed_multiplication(files[0]),
               "disagrees with combined k")
        fake(files[0],seeds[0],nan=True)
        reject(lambda:read_executed_multiplication(files[0]),
               "invalid executed neutron multiplication")
        fake(files[0],seeds[0])
        with h5py.File(files[0],"r+") as h:
            h.attrs["openmc_version"]=[0,16,0]
        reject(lambda:read_executed_multiplication(files[0]),
               "requires actual OpenMC 0.15.3")
        fake(files[0],seeds[0])

        # Actual final HDF5 cannot be swapped while preserving the
        # original SHA256 transport evidence.
        with h5py.File(files[0],"r+") as h:
            h["k_combined"][0]=0.3
        reject(audit,"provenance mismatch")
        fake(files[0],seeds[0])
        save()

        original=json.loads(source_file.read_text())
        original["real_transport_summary_sha256"]="0"*64
        source_file.write_text(json.dumps(original))
        reject(lambda:audit_real_core_criticality(summary_file,source_file),
               "source-convergence qualification is missing")

    print("Physical criticality necessary-condition and native OpenMC k estimators: PASS")
    print("  synthetic statepoints ONLY; real neutron C++ transport separately required")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
