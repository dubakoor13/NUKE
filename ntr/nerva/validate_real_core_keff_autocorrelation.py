"""Adversarial checks for descriptive neutron k autocorrelation diagnostics.

All generated arrays/statepoints here are synthetic regression fixtures.
Only CI's separate OpenMC C++ four-seed execution provides real evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .real_core_keff_autocorrelation import (
    assess_generation_k_correlation,
    audit_executed_generation_k_autocorrelation,
)
from .real_core_keff_history_qualification import (
    _sha, audit_real_core_keff_histories,
)
from .validate_real_core_keff_history_qualification import fake


def reject(action, needle: str)->None:
    try:
        action()
    except ValueError as exc:
        assert needle in str(exc),(needle,str(exc))
    else:
        raise AssertionError("corrupted fission-source generation series passed")


def main()->int:
    rng=np.random.default_rng(20261024)
    iid=assess_generation_k_correlation(
        0.8+rng.normal(0,0.01,2000)
    )
    assert iid["active_batches"]==2000
    assert iid["maximum_autocorrelation_lag_inspected"]==30
    assert len(iid["observed_acf_lags_0_to_max"])==31
    assert iid["observed_acf_lags_0_to_max"][0]==1.
    assert iid["initial_positive_sequence_tau_diagnostic"]>=1
    assert not iid["source_stationarity_or_reactor_uncertainty_certified"]
    assert not iid["reactor_keff_combined_sem_replaced_by_batch_k_diagnostic"]

    increments=rng.normal(0,0.01,2000)
    ar=np.empty(len(increments))
    ar[0]=increments[0]
    for k in range(1,len(ar)):
        ar[k]=0.8*ar[k-1]+increments[k]
    correlated=assess_generation_k_correlation(0.8+ar)
    assert correlated["initial_positive_sequence_tau_diagnostic"]>3
    assert correlated["nominal_effective_generation_count_NOT_VALIDATED"]<2000/3
    assert correlated["nominal_sem_inflation_sqrt_tau_NOT_CALIBRATED"]>1.7
    assert len(correlated["blocked_generation_mean_diagnostics"])==5
    assert correlated["blocked_generation_mean_diagnostics"][-1][
        "nonoverlapping_batch_generations_per_block"]==16

    reject(lambda:assess_generation_k_correlation(np.ones(100)),
           "constant k-generation")
    reject(lambda:assess_generation_k_correlation(np.ones(20)),
           "invalid active")
    reject(lambda:assess_generation_k_correlation(
        np.r_[np.ones(99),np.nan]),"invalid active")
    reject(lambda:assess_generation_k_correlation(
        np.r_[np.ones(99),-1.]),"invalid active")
    reject(lambda:assess_generation_k_correlation(
        0.8+rng.normal(0,.01,100),max_lag=0),"invalid active")
    reject(lambda:assess_generation_k_correlation(
        np.ones((10,10))),"invalid active")

    with TemporaryDirectory() as temp:
        root=Path(temp)
        runs=[]
        batches=(40,60,80)
        for seed in (13007,23011,33013,43019):
            folder=root/f"seed_{seed}"
            folder.mkdir()
            for batch in batches:
                fake(folder/f"statepoint.{batch:02}.h5",seed,batch)
            last=folder/"statepoint.80.h5"
            runs.append({
                "seed":seed,"statepoint_path":str(last),
                "statepoint_sha256":_sha(last)
            })
        summary=root/"transport.json"
        spatial=root/"spatial.json"
        history_path=root/"verified_keff_history.json"
        summary.write_text(json.dumps({
            "execution":"ACTUAL_OPENMC_TRANSPORT",
            "assembly":"core",
            "batches":80,
            "saved_source_bank_batches":list(batches),
            "runs":runs
        }))
        spatial.write_text(json.dumps({
            "analysis":"EXACT_SAME_OPENMC_SOURCE_BANKS_ON_NESTED_SPATIAL_GRIDS",
            "real_transport_summary_sha256":_sha(summary),
            "resolutions":[{},{}],
            "source_diversity_screen_status":"FAIL_SOURCE_CLONING_SCREEN",
        }))
        previous=audit_real_core_keff_histories(summary,spatial)
        history_path.write_text(json.dumps(previous))
        output=audit_executed_generation_k_autocorrelation(summary,history_path)
        assert output["independent_rng_seed_count"]==4
        assert output["real_native_terminal_hdf5_count"]==4
        assert output["all_earlier_statepoint_k_entropy_prefix_checks_reconciled"]
        assert output["all_per_seed_multilag_and_block_variance_checks_completed"]
        assert not output["effective_sample_size_is_physical_neutron_count"]
        assert not output["spatial_source_stationarity_demonstrated"]
        assert not output["fully_calibrated_reactor_uncertainty_established"]
        assert len(output["per_real_seed"])==4
        for row in output["per_real_seed"]:
            assert row["active_batches"]==60
            assert row["source_entropy_and_k_prefixes_previously_verified"]
            assert row["naive_iid_generation_k_sem_NOT_CALIBRATED"]>0

        # Hashes tie new statistics to exact native eigenvalue statepoints.
        bad=runs[0]["statepoint_path"]
        with h5py.File(bad,"r+") as f:
            f["k_generation"][-1]=0.123
        reject(lambda:audit_executed_generation_k_autocorrelation(
            summary,history_path),"invalid native four-seed")
        # If a malicious source-history audit disagrees with actual core,
        # refuse to claim a qualified result even with the same summary.
        old=json.loads(history_path.read_text())
        old["real_transport_summary_sha256"]="0"*64
        history_path.write_text(json.dumps(old))
        reject(lambda:audit_executed_generation_k_autocorrelation(
            summary,history_path),"no verified genuine")
    print("OpenMC neutron source k-generation multilag/blocked statistics: PASS")
    print("  all HDF5 here SYNTHETIC, not physically validated transport")
    print("  ESS and confidence levels never certified for fission genealogy")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
