"""Synthetic adversarial native HDF5 tests. These are NOT actual transport."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .core_fission_emission_native_uncertainty import (
    SCORES, native_window_marginal_moments,
    audit_native_fission_emission_uncertainty, _read_checkpoint,
)
from .validate_core_prompt_delayed_fission_validation import fake


def reject(fn, message: str) -> None:
    try:
        fn()
    except ValueError as error:
        assert message in str(error), (message,str(error))
    else:
        raise AssertionError("unphysical evaluated native neutron moments accepted")


def sha(path: Path) -> str:
    h=hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    raw=np.array([8.,7.92,.08])
    second=np.array([20.,19.,.002])
    out=native_window_marginal_moments(raw,second,4)
    assert out["disjoint_active_batch_realizations"]==4
    assert out["native_delayed_fraction_point_estimate"]==.01
    assert out["native_score_moments"]["nu-fission"]["native_batch_sem"]>0
    assert out["native_score_moments"]["delayed-nu-fission"]["native_batch_relative_sem"]>0
    assert (out["first_order_covariance_agnostic_ratio_sem_upper"]>=
            out["first_order_covariance_agnostic_ratio_sem_lower"]>=0)
    assert out["inter_score_covariance_available_from_native_hdf5"] is False
    assert out["ratio_confidence_interval_established"] is False
    assert out["native_sem_qualifies_correlated_fission_genealogy"] is False
    assert not out["adjoint_weighted_beta_eff_estimated"]
    # With zero fission observations the ratio is unobserved, not zero.
    none=native_window_marginal_moments(np.zeros(3),np.zeros(3),4)
    assert none["native_delayed_fraction_point_estimate"] is None
    assert none["first_order_covariance_agnostic_ratio_sem_upper"] is None
    reject(lambda:native_window_marginal_moments(
        raw,np.array([1.,19.,.002]),4),"batch Cauchy identity")
    reject(lambda:native_window_marginal_moments(
        raw,np.array([20.,-1.,.002]),4),"invalid native")
    reject(lambda:native_window_marginal_moments(raw,second,1),
           "invalid native")
    reject(lambda:native_window_marginal_moments(
        raw,np.array([20.,19.,np.nan]),4),"invalid native")
    reject(lambda:native_window_marginal_moments(
        raw,np.array([20.,19.,.002]),0),"invalid native")

    # Full provenance regression over twenty synthetic statepoints.
    with TemporaryDirectory() as temp:
        root=Path(temp)
        checkpoints=[40,60,80,100,120]
        summary={
            "execution":"ACTUAL_OPENMC_TRANSPORT",
            "assembly":"core",
            "saved_source_bank_batches":checkpoints,
            "batches":120,"inactive_batches":30,
            "runs":[],
        }
        emission={
            "audit":"ACTUAL_OPENMC_CORE_PROMPT_DELAYED_FISSION_NEUTRON_SOURCE_CLOSURE",
            "independent_seed_count":4,
            "checked_native_statepoint_checkpoints":20,
            "independently_evaluated_prompt_delayed_total_yield_consistency_screen_passed":True,
            "runs":[],
        }
        for seed in (13007,23011,33013,43019):
            folder=root/f"seed_{seed}"
            folder.mkdir()
            for batch in checkpoints:
                sp=folder/f"statepoint.{batch:03d}.h5"
                fake(sp,batch=batch)
                with h5py.File(sp,"r+") as h:
                    h["seed"][...]=seed
            final=folder/"statepoint.120.h5"
            summary["runs"].append({
                "seed":seed,"statepoint_path":str(final),
                "statepoint_sha256":sha(final)
            })
            emission["runs"].append({
                "seed":seed,
                "final_statepoint_sha256":sha(final),
                "evaluated_delayed_emission_fraction":.0002/.03,
                "windows":[{
                    "neutron_production_per_generation_source":{
                        score:v for score,v in zip(SCORES,(.03,.0298,.0002))
                    }
                } for _ in checkpoints],
            })
        summary_file=root/"real_core_transport_summary.json"
        emission_file=root/"core_prompt_delayed_fission_validation.json"
        summary_file.write_text(json.dumps(summary))
        emission["executed_core_summary_sha256"]=sha(summary_file)
        emission_file.write_text(json.dumps(emission))
        result=audit_native_fission_emission_uncertainty(summary_file,emission_file)
        assert result["verified_hdf5_statepoint_count"]==20
        assert result["independent_seed_count"]==4
        assert result["disjoint_active_windows_per_seed"]==5
        assert result["marginal_native_batch_moments_verified"]
        assert not result["inter_score_covariance_recoverable_from_marginal_hdf5_moments"]
        assert not result["cauchy_ratio_bound_is_confidence_interval"]
        for run in result["real_source_reaction_statistics"]:
            assert len(run["disjoint_active_generation_windows"])==5
            assert sum(w["disjoint_active_batch_realizations"] for w in
                run["disjoint_active_generation_windows"])==90
        # Mutated native second moments are physically inconsistent even
        # if the exported reference means continue to look plausible.
        sp=root/"seed_13007/statepoint.060.h5"
        with h5py.File(sp,"r+") as h:
            t=h["tallies/tally 1/results"]
            t[0,2,1]=0
        reject(lambda:audit_native_fission_emission_uncertainty(
            summary_file,emission_file),"batch Cauchy identity")
        fake(sp,batch=60)
        with h5py.File(sp,"r+") as h:
            h["seed"][...]=13007
        # An inconsistent reference first-moment report must also be rejected.
        emission["runs"][0]["windows"][2]["neutron_production_per_generation_source"][
            "delayed-nu-fission"]=.2
        emission_file.write_text(json.dumps(emission))
        reject(lambda:audit_native_fission_emission_uncertainty(
            summary_file,emission_file),"first/second moment lineage mismatch")
    print("Native OpenMC delayed-fission neutron uncertainty regressions: PASS")
    print("  ALL fixtures synthetic; real four-seed OpenMC HDF5 CI required")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
