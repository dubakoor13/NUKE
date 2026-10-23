"""Native OpenMC fission-emission active-batch moment uncertainty audit.

Separate marginal OpenMC native SUM/SUMSQ uncertainty for TOTAL, PROMPT and
DELAYED neutron production using actual eigenvalue source-generation tallies.
Subtract cumulative first *and second* moments at source-bank checkpoints
to obtain five disjoint active-generation measurement windows per seed.

The native batch SEM assumes independent generations. Eigenvalue fission
genealogies are correlated, so SEM is an OpenMC NUMERICAL DIAGNOSTIC and
not independently calibrated reactor uncertainty. More importantly, native
separate second moments do not contain the *between-score covariance*
required for the total/delayed fraction. Report only a first-order,
Cauchy-Schwarz covariance-agnostic upper diagnostic bound, never a
confidence interval, beta_eff uncertainty or kinetic reactivity estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .core_prompt_delayed_fission_validation import SCORES, _str, _tally


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as fd:
        for block in iter(lambda: fd.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def native_window_marginal_moments(
    raw_sum: np.ndarray, raw_squared_sum: np.ndarray, n: int
) -> dict:
    """Actual disjoint active-batch SUM/SUMSQ; never infer score covariance."""
    first=np.asarray(raw_sum,dtype=float)
    second=np.asarray(raw_squared_sum,dtype=float)
    if (type(n) is not int or n<2 or first.shape!=(3,)
            or second.shape!=(3,)
            or not np.isfinite(first).all()
            or not np.isfinite(second).all()
            or np.any(first < -1e-12)
            or np.any(second < 0)):
        raise ValueError("invalid native OpenMC fission-emission active-batch moments")
    residual=second-first*first/n
    if np.any(residual < -1e-10*np.maximum(second,1e-24)):
        raise ValueError("native OpenMC neutron-emission batch Cauchy identity violated")
    # On near-equality, negative O(machine epsilon) cancellation is allowed,
    # but no physically significant invalid batch variance can be clipped.
    variance=np.maximum(residual,0.)/(n-1)
    sem=np.sqrt(variance/n)
    means=np.maximum(first,0.)/n
    total=float(means[0])
    delayed=float(means[2])
    fraction=delayed/total if total>0 else None
    if fraction is not None and not 0 <= fraction <= 1.0001:
        raise ValueError("unphysical delayed fission neutron production ratio")
    # delta(r = delayed/total) ~ delta_delayed/total -
    # (delayed/total) delta_total/total.
    # Cauchy bounds the (UNKNOWN) covariance magnitude, but this is only
    # a first-order bound on conventional native-batch marginal variance,
    # NOT coverage guaranteed by correlated eigenvalue generations.
    upper=(float(sem[2])+abs(fraction)*float(sem[0]))/total if fraction is not None else None
    lower=abs(float(sem[2])-abs(fraction)*float(sem[0]))/total if fraction is not None else None
    return {
        "disjoint_active_batch_realizations":n,
        "native_score_moments":{
            score:{
                "raw_batch_sum":float(first[i]),
                "raw_batch_squared_sum":float(second[i]),
                "mean_neutron_production_per_source":float(means[i]),
                "native_batch_sample_variance":float(variance[i]),
                "native_batch_sem":float(sem[i]),
                "native_batch_relative_sem":(
                    float(sem[i]/means[i]) if means[i]>0 else None
                ),
            } for i,score in enumerate(SCORES)
        },
        "native_delayed_fraction_point_estimate":fraction,
        "first_order_covariance_agnostic_ratio_sem_lower":lower,
        "first_order_covariance_agnostic_ratio_sem_upper":upper,
        "inter_score_covariance_available_from_native_hdf5":False,
        "native_sem_qualifies_correlated_fission_genealogy":False,
        "ratio_confidence_interval_established":False,
        "adjoint_weighted_beta_eff_estimated":False,
    }


def _read_checkpoint(path: Path) -> dict:
    with h5py.File(path,"r") as sp:
        if (_str(sp.attrs["filetype"])!="statepoint"
                or _str(sp["run_mode"][()])!="eigenvalue"
                or tuple(map(int,sp.attrs["openmc_version"]))!=(0,15,3)
                or int(sp.attrs["photon_transport"])!=0):
            raise ValueError("real OpenMC 0.15.3 neutron eigenvalue HDF5 required")
        active=int(sp["n_realizations"][()])
        batch=int(sp["current_batch"][()])
        inactive=int(sp["n_inactive"][()])
        seed=int(sp["seed"][()])
        if active<2 or batch-inactive!=active or seed<=0:
            raise ValueError("inconsistent native OpenMC eigenvalue active realizations")
        moments,_=_tally(
            sp["tallies"],"nerva_global_fission_emission",(),active,SCORES
        )
        return {
            "seed":seed,"batch":batch,"inactive":inactive,"active":active,
            "statepoint_sha256":_sha(path),
            "first":moments[0,:,0],
            "second":moments[0,:,1],
        }


def audit_native_fission_emission_uncertainty(
    summary_file: Path, emission_file: Path
) -> dict:
    summary_file=Path(summary_file)
    emission_file=Path(emission_file)
    summary=json.loads(summary_file.read_text(encoding="utf-8"))
    emission=json.loads(emission_file.read_text(encoding="utf-8"))
    if (summary.get("execution")!="ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly")!="core"
            or emission.get("audit")!="ACTUAL_OPENMC_CORE_PROMPT_DELAYED_FISSION_NEUTRON_SOURCE_CLOSURE"
            or emission.get("executed_core_summary_sha256")!=_sha(summary_file)
            or emission.get("independent_seed_count")!=4
            or emission.get("checked_native_statepoint_checkpoints")!=20
            or emission.get("independently_evaluated_prompt_delayed_total_yield_consistency_screen_passed") is not True):
        raise ValueError("missing verified native four-seed core neutron emission provenance")
    checkpoints=summary.get("saved_source_bank_batches")
    runs=summary.get("runs")
    expected_runs=emission.get("runs")
    n_total=summary.get("batches")
    n_inactive=summary.get("inactive_batches")
    if (not isinstance(checkpoints,list) or len(checkpoints)!=5
            or checkpoints[-1]!=n_total
            or checkpoints[0]<=n_inactive
            or any(a>=b for a,b in zip(checkpoints,checkpoints[1:]))
            or not isinstance(runs,list) or len(runs)!=4
            or not isinstance(expected_runs,list) or len(expected_runs)!=4):
        raise ValueError("invalid actual OpenMC active-neutron source checkpoint sequence")
    expected_by_seed={x["seed"]:x for x in expected_runs}
    if len(expected_by_seed)!=4:
        raise ValueError("duplicate neutron-emission physical provenance seed")
    unique_seeds=set()
    hashes=set()
    rows=[]
    for run in runs:
        seed=run.get("seed")
        if (type(seed) is not int or seed<=0 or seed in unique_seeds
                or seed not in expected_by_seed):
            raise ValueError("invalid independently executed OpenMC RNG seed")
        unique_seeds.add(seed)
        final=Path(run["statepoint_path"])
        if not final.is_file() or _sha(final)!=run["statepoint_sha256"]:
            raise ValueError("actual native neutron HDF5 digest inconsistent")
        if run["statepoint_sha256"]!=expected_by_seed[seed]["final_statepoint_sha256"]:
            raise ValueError("neutron production first moments and second moments use different HDF5")
        data=[]
        for checkpoint in checkpoints:
            path=final.parent/f"statepoint.{checkpoint:0{len(str(n_total))}d}.h5"
            if not path.is_file():
                raise ValueError("native neutron emission source checkpoint absent")
            item=_read_checkpoint(path)
            if (item["seed"]!=seed or item["batch"]!=checkpoint
                    or item["inactive"]!=n_inactive
                    or item["statepoint_sha256"] in hashes):
                raise ValueError("duplicate or inconsistent physical eigenvalue checkpoint")
            hashes.add(item["statepoint_sha256"])
            data.append(item)
        if data[-1]["statepoint_sha256"]!=run["statepoint_sha256"]:
            raise ValueError("wrong final OpenMC neutron production statepoint")
        windows=[]
        previous=None
        for index,item in enumerate(data):
            n=item["active"]-(previous["active"] if previous else 0)
            first=item["first"]-(previous["first"] if previous else 0)
            second=item["second"]-(previous["second"] if previous else 0)
            row=native_window_marginal_moments(first,second,int(n))
            row.update({
                "first_active_generation":(n_inactive+1 if previous is None
                                            else previous["batch"]+1),
                "last_active_generation":item["batch"],
                "statepoint_sha256":item["statepoint_sha256"],
            })
            expected=expected_by_seed[seed]["windows"][index]
            for score in SCORES:
                actual=row["native_score_moments"][score]["mean_neutron_production_per_source"]
                earlier=expected["neutron_production_per_generation_source"][score]
                if abs(actual-earlier)>2e-12+2e-10*max(abs(actual),abs(earlier)):
                    raise ValueError("native neutron production first/second moment lineage mismatch")
            windows.append(row)
            previous=item
        last=data[-1]
        result=native_window_marginal_moments(
            last["first"],last["second"],int(last["active"])
        )
        fraction=result["native_delayed_fraction_point_estimate"]
        baseline=expected_by_seed[seed]["evaluated_delayed_emission_fraction"]
        if fraction is None or abs(fraction-baseline)>1e-12:
            raise ValueError("native delayed emission HDF5 ratio differs from verified reference")
        rows.append({
            "seed":seed,
            "actual_statepoint_sha256":last["statepoint_sha256"],
            "all_active_generations":result,
            "disjoint_active_generation_windows":windows,
        })
    if len(hashes)!=20 or len(unique_seeds)!=4:
        raise ValueError("expected twenty independent-seed statepoint evidence files")
    max_relative=max(
        x["native_score_moments"]["delayed-nu-fission"]["native_batch_relative_sem"]
        for row in rows for x in row["disjoint_active_generation_windows"]
        if x["native_score_moments"]["delayed-nu-fission"]["native_batch_relative_sem"] is not None
    )
    return {
        "audit":"GENUINE_OPENMC_CORE_FISSION_EMISSION_NATIVE_SECOND_MOMENT_UNCERTAINTY",
        "executed_core_summary_sha256":_sha(summary_file),
        "real_prompt_delayed_emission_audit_sha256":_sha(emission_file),
        "independent_seed_count":len(unique_seeds),
        "verified_hdf5_statepoint_count":len(hashes),
        "disjoint_active_windows_per_seed":5,
        "largest_native_delayed_emission_batch_relative_sem_in_windows":float(max_relative),
        "real_source_reaction_statistics":rows,
        "marginal_native_batch_moments_verified":True,
        "inter_score_covariance_recoverable_from_marginal_hdf5_moments":False,
        "cauchy_ratio_bound_is_confidence_interval":False,
        "fission_source_stationarity_demonstrated":False,
        "reactor_beta_eff_or_hot_nerva_validated":False,
        "interpretation":(
            "Real OpenMC first AND second moments of evaluated delayed, prompt "
            "and total neutron production, differenced between five cumulative "
            "statepoints to obtain disjoint active-batch moment sets. Native "
            "batch SEMs depend on independence assumptions violated by source "
            "genealogy. The delayed/total source fraction has unknown native "
            "inter-score covariance. First-order Cauchy bounds are only "
            "diagnostic and cannot supply a validated confidence interval "
            "or adjoint-weighted beta_eff."
        ),
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--prompt-emission",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    out=audit_native_fission_emission_uncertainty(args.summary,args.prompt_emission)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print("REAL OpenMC fission neutron production native moments: PASS")
    print("  four seeds / 20 statepoints / disjoint first AND second moment windows")
    print("  native delayed neutron window max relative SEM:",
          out["largest_native_delayed_emission_batch_relative_sem_in_windows"])
    print("  inter-score covariance, beta_eff, formal CI: NOT ESTABLISHED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
