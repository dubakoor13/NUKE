"""Read-only multilag generation-k autocorrelation / block variance audit.

Inspect the *real* 4-seed OpenMC eigenvalue statepoints and the already
SHA256-verified exact-prefix k-generation history audit. Unlike iid mean
uncertainty, autocorrelated fission-source batch series require accounting
for dependence. Report conservative (non-improving) initial-positive-
sequence lag diagnostics and disjoint nonoverlapping block-mean SEMs.

The batch k_generation sequence is NOT OpenMC k_combined; its estimated
autocorrelation is observational, finite-sample and potentially biased by
nonstationary source genealogies. Neither estimated ESS nor these block
diagnostics prove spatial source stationarity, determine reactor criticality,
or provide formally qualified confidence intervals.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .real_core_keff_history_qualification import read_real_keff_checkpoint


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def assess_generation_k_correlation(values: object, *, max_lag: int = 30) -> dict:
    """Estimate descriptive integrated autocorrelation and blocked SEM.

    Geyer-style *initial positive lag-pair* truncation uses the biased
    finite-lag autocovariance estimate and requires adjacent positive
    autocorrelation pairs (rho_1+rho_2, rho_3+rho_4, ...). This is not a
    rigorous effective sample-size claim for a potentially nonstationary
    neutron-fission source; no negative-correlation 'precision bonus'.
    """
    x=np.asarray(values,dtype=np.float64)
    if (x.ndim!=1 or len(x)<40 or not np.isfinite(x).all()
            or np.any(x<0) or type(max_lag) is not int or max_lag<2):
        raise ValueError("invalid active neutron eigenvalue generation-k sequence")
    n=int(x.size)
    mu=float(x.mean())
    centered=x-mu
    variance=float(np.dot(centered,centered)/n)
    if not math.isfinite(variance) or variance<=0:
        raise ValueError("constant k-generation history cannot identify autocorrelation")
    # Max length chosen before observing data; do not choose favorable lags.
    upper=min(max_lag, n//3)
    rho=[1.]
    for lag in range(1,upper+1):
        # Biased (divide by N) ACF is nonnegative-definite for a finite
        # sequence; avoid unstable high-lag divide-by-(N-lag) inflation.
        r=float(np.dot(centered[:-lag],centered[lag:])/(n*variance))
        if not math.isfinite(r) or abs(r)>1+1e-9:
            raise ValueError("nonphysical native generation-k autocorrelation")
        rho.append(r)
    positive=[]
    positive_pairs=0
    for lag in range(1,len(rho)-1,2):
        pair=rho[lag]+rho[lag+1]
        if pair<=0:
            break
        positive.extend((rho[lag],rho[lag+1]))
        positive_pairs+=1
    tau_raw=float(1+2*sum(positive))
    tau=max(1.,tau_raw)
    # Effective sample number is only an optimistic *diagnostic* under
    # stationarity and mixing assumptions unqualified for source banks.
    diagnostic_n_eff=float(n/tau)
    naive_sem=float(np.std(x,ddof=1)/math.sqrt(n))
    blocked=[]
    for size in (1,2,4,8,16):
        m=n//size
        if m<4:
            continue
        b=x[:m*size].reshape(m,size).mean(axis=1)
        var=float(np.var(b,ddof=1))
        if not math.isfinite(var) or var<0:
            raise ValueError("invalid nonoverlapping neutron generation block variance")
        sem=float(math.sqrt(var/m))
        blocked.append({
            "nonoverlapping_batch_generations_per_block":size,
            "number_of_independent_blocks_ASSUMED_NOT_PROVEN":m,
            "discarded_suffix_generations":n-m*size,
            "observed_block_mean_variance":var,
            "block_mean_sem_IF_BLOCKS_INDEPENDENT":sem,
            "sem_to_iid_ratio_diagnostic":sem/naive_sem,
        })
    if not blocked or blocked[0]["nonoverlapping_batch_generations_per_block"]!=1:
        raise ValueError("no valid k-generation blocking series")
    return {
        "active_batches":n,
        "generation_k_mean":mu,
        "observed_generation_k_variance":float(np.var(x,ddof=1)),
        "naive_iid_generation_k_sem_NOT_CALIBRATED":naive_sem,
        "maximum_autocorrelation_lag_inspected":upper,
        "observed_acf_lags_0_to_max":rho,
        "initial_positive_autocorrelation_pairs":positive_pairs,
        "initial_positive_sequence_tau_diagnostic":tau,
        "nominal_effective_generation_count_NOT_VALIDATED":diagnostic_n_eff,
        "nominal_sem_inflation_sqrt_tau_NOT_CALIBRATED":math.sqrt(tau),
        "blocked_generation_mean_diagnostics":blocked,
        "fission_source_generations_independent":False,
        "source_stationarity_or_reactor_uncertainty_certified":False,
        "reactor_keff_combined_sem_replaced_by_batch_k_diagnostic":False,
    }


def audit_executed_generation_k_autocorrelation(
    summary_path: Path, history_path: Path
) -> dict:
    summary_path,history_path=Path(summary_path),Path(history_path)
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    history=json.loads(history_path.read_text(encoding="utf-8"))
    if (summary.get("execution")!="ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly")!="core"
            or history.get("audit")!="ACTUAL_OPENMC_EIGENVALUE_HISTORY_PREFIX_AND_SPLIT_CHAINS"
            or history.get("real_transport_summary_sha256")!=_sha(summary_path)
            or history.get("all_checkpoint_keff_and_entropy_prefixes_verified") is not True
            or history.get("executed_core_eigenvalue_settings_identical_across_seeds") is not True
            or history.get("spatial_fission_source_stationarity_validated") is not False):
        raise ValueError("no verified genuine HDF5 source history provenance")
    runs=summary.get("runs")
    previous=history.get("real_seed_checkpoints")
    expected=history.get("four_seed_split_chain_generation_k")
    if (not isinstance(runs,list) or len(runs)!=4
            or not isinstance(previous,list) or len(previous)!=4
            or not isinstance(expected,dict)
            or expected.get("independent_seed_count")!=4
            or expected.get("active_source_generation_histories_are_independent") is not False
            or not isinstance(summary.get("batches"),int)):
        raise ValueError("four genuine independent source histories required")
    previous_by_seed={p["seed"]:p for p in previous}
    if len(previous_by_seed)!=4:
        raise ValueError("duplicate source history audit RNG seeds")
    unique=set()
    hashes=set()
    rows=[]
    for run in runs:
        seed=run.get("seed")
        file=Path(run.get("statepoint_path",""))
        if (type(seed) is not int or seed<=0 or seed in unique
                or seed not in previous_by_seed
                or not file.is_file() or _sha(file)!=run.get("statepoint_sha256")):
            raise ValueError("invalid native four-seed eigenvalue statepoint provenance")
        unique.add(seed)
        p=previous_by_seed[seed]
        checkpoints=p.get("checkpoints",[])
        if (not isinstance(checkpoints,list) or len(checkpoints)<3
                or checkpoints[-1].get("statepoint_sha256")!=run["statepoint_sha256"]):
            raise ValueError("inconsistent verified native generation checkpoint history")
        state=read_real_keff_checkpoint(file)
        if (state["seed"]!=seed or state["sha256"]!=run["statepoint_sha256"]
                or state["batch"]!=summary["batches"]
                or state["active_realizations"]!=checkpoints[-1]["active_realizations"]):
            raise ValueError("actual executed neutron transport history changed")
        hashes.add(state["sha256"])
        by_batch=state["generation_keff"].reshape(
            state["batch"],state["generations_per_batch"]
        ).mean(axis=1)
        active=by_batch[state["inactive_batches"]:]
        if (len(active)!=expected["active_generations_per_seed"]
                or abs(float(active.mean())-p["mean_per_generation_k"])>2e-13):
            raise ValueError("native k-generation samples disagree with verified split history")
        k=assess_generation_k_correlation(active)
        k.update({
            "seed":seed,
            "genuine_terminal_statepoint_sha256":state["sha256"],
            "underlying_generation_per_batch":state["generations_per_batch"],
            "source_entropy_and_k_prefixes_previously_verified":True,
            "actual_combined_keff_observation_NOT_COMPARABLE_TO_BATCH_SEM":
                [float(t) for t in state["combined_keff"]],
        })
        rows.append(k)
    if len(hashes)!=4:
        raise ValueError("duplicate terminal neutron eigenvalue statepoint")
    return {
        "audit":"REAL_OPENMC_NEUTRON_GENERATION_K_AUTOCORRELATION_BLOCK_VARIANCE_DIAGNOSTIC",
        "real_transport_summary_sha256":_sha(summary_path),
        "verified_generation_history_audit_sha256":_sha(history_path),
        "independent_rng_seed_count":len(unique),
        "real_native_terminal_hdf5_count":len(hashes),
        "all_earlier_statepoint_k_entropy_prefix_checks_reconciled":True,
        "per_real_seed":rows,
        "all_per_seed_multilag_and_block_variance_checks_completed":True,
        "effective_sample_size_is_physical_neutron_count":False,
        "spatial_source_stationarity_demonstrated":False,
        "fully_calibrated_reactor_uncertainty_established":False,
        "critical_or_hot_nerva_validated":False,
        "interpretation":(
            "Genuine executed OpenMC native per-generation k series with "
            "multiple lagged correlation and nonoverlapping block diagnostics. "
            "Finite chain lengths and strongly related fission generations "
            "invalidate naive independent k statistics. Indicative tau/ESS "
            "cannot prove spatial source convergence, reactor k uncertainty, "
            "hot nuclear cross sections or historical NERVA fidelity."
        ),
    }


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--history",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=audit_executed_generation_k_autocorrelation(a.summary,a.history)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("Genuine OpenMC independent source seeds:",
          result["independent_rng_seed_count"])
    print("Per-seed initial-positive multi-lag tau (not certified):",
          [s["initial_positive_sequence_tau_diagnostic"]
           for s in result["per_real_seed"]])
    print("Per-seed observed block mean variance and native genealogy: PASS")
    print("Spatial source stationarity, hot core, historical NERVA: NOT VALIDATED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
