"""REAL OpenMC neutron-reaction estimator agreement over disjoint active batches.

Independently read the genuine ANALOG neutron-balance reaction tally and
TRACKLENGTH neutron-reaction tally from *the same* HDF5 eigenvalue states.
OpenMC stores sums and sums of squared per-batch tallies cumulatively across
active generations. Difference these raw moments (NOT sample means) to
reconstruct five disjoint active-batch windows, with separately computed
batch SEM per estimator. Reconcile the tracklength means with the existing
independently checked material/incident-energy reaction histories.

Analog and tracklength scores use the same correlated neutron histories:
their covariance is unavailable from marginal statepoint moments. The
triangle-inequality sum of reported SEMs is a conservative upper bound on
the difference's SEM *conditional on reliable batch SEMs*, not a true joint
uncertainty or proof that source generations are independent. No reactor
design, geometry, enrichment, control or criticality tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .core_reaction_history import audit_real_reaction_windows


SCORES = ("absorption", "nu-fission")
TALLIES = (
    ("nerva_global_neutron_balance", "analog"),
    ("nerva_global_reaction_tracklength", "tracklength"),
)


def _sha(path: Path) -> str:
    d=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


def _bytestr(v) -> str:
    if not isinstance(v, bytes):
        raise ValueError("missing genuine OpenMC HDF5 string label")
    return v.decode("utf-8").rstrip("\x00")


def _checkpoint(path: Path) -> dict:
    """Read raw neutron reaction per-active-generation sums and second moments."""
    with h5py.File(path,"r") as sp:
        try:
            if (_bytestr(sp.attrs["filetype"]) != "statepoint"
                    or _bytestr(sp["run_mode"][()]) != "eigenvalue"
                    or int(sp.attrs["photon_transport"]) != 0):
                raise ValueError("must be genuine neutron-only eigenvalue statepoint")
            batch = int(sp["current_batch"][()])
            inactive = int(sp["n_inactive"][()])
            particles = int(sp["n_particles"][()])
            seed = int(sp["seed"][()])
            n = batch-inactive
            if seed < 1 or inactive < 0 or particles < 2 or n < 2:
                raise ValueError("invalid OpenMC active generations or seed")
            tallies=sp["tallies"]
            record={}
            for tally_name, estimator in TALLIES:
                matches=[]
                for key, obj in tallies.items():
                    if (key.startswith("tally ") and isinstance(obj,h5py.Group)
                            and "name" in obj and
                            _bytestr(obj["name"][()]) == tally_name):
                        matches.append(obj)
                if len(matches)!=1:
                    raise ValueError("absent/duplicate required OpenMC neutron reaction estimator")
                obj=matches[0]
                if _bytestr(obj["estimator"][()]) != estimator:
                    raise ValueError("OpenMC physical reaction estimator mismatch")
                if int(obj["n_realizations"][()]) != n:
                    raise ValueError("OpenMC reaction tally and active generations disagree")
                if obj["filters"][()].size != 0:
                    raise ValueError("estimator reaction tally must span whole core")
                names=[_bytestr(v) for v in obj["score_bins"][()]]
                if len(set(names)) != len(names) or any(s not in names for s in SCORES):
                    raise ValueError("required absorption/nu-fission HDF5 scores absent")
                data=np.asarray(obj["results"][()],dtype=float)
                if data.shape!=(1,len(names),2) or not np.isfinite(data).all():
                    raise ValueError("invalid cumulative OpenMC global reaction moments")
                scored=data[0, [names.index(s) for s in SCORES], :]
                if np.any(scored < 0):
                    raise ValueError("negative cumulative neutron reaction moments")
                record[estimator]=scored
        except (KeyError,OSError,TypeError) as error:
            raise ValueError("incomplete native OpenMC reaction statepoint HDF5") from error
    return {
        "seed":seed, "batch":batch, "inactive":inactive,
        "active":n, "particles":particles, "sha256":_sha(path),
        **record,
    }


def _disjoint_moments(rows: list[dict], estimator: str) -> list[dict]:
    if len(rows) < 3:
        raise ValueError("three or more real reaction checkpoints required")
    nprev=0
    before=np.zeros((len(SCORES),2),dtype=float)
    output=[]
    for row in rows:
        count=row["active"]-nprev
        if count<2:
            raise ValueError("at least two active realizations per estimator window")
        cumulative=row[estimator]
        if cumulative.shape != before.shape:
            raise ValueError("analog/tracklength physical score dimension changed")
        delta=cumulative-before
        if np.any(delta < -1e-10):
            raise ValueError("neutron reaction cumulative sums or squares decreased")
        mean=delta[:,0]/count
        second=delta[:,1]/count
        variance=(second-mean**2)/(count-1)
        if (not np.isfinite(mean).all() or np.any(mean <= 0)
                or np.any(variance < -1e-12)):
            raise ValueError("negative or zero neutron score or invalid batch variance")
        sem=np.sqrt(np.maximum(variance,0.))
        output.append({
            "last_batch":row["batch"],
            "first_batch":row["batch"]-count+1,
            "active_generations":count,
            "per_source_rate":mean,
            "batch_reported_standard_error":sem,
        })
        before=cumulative
        nprev=row["active"]
    for score_i in range(len(SCORES)):
        final=rows[-1][estimator][score_i,0]/rows[-1]["active"]
        weighted=sum(z["active_generations"]*z["per_source_rate"][score_i]
                     for z in output)/rows[-1]["active"]
        if not math.isclose(float(final),float(weighted),abs_tol=3e-13,rel_tol=0):
            raise ValueError("disjoint estimator windows do not reconstruct final neutron tally")
    return output


def compare_estimator_windows(
    trajectories: list[list[dict]], *,
    sigma_multiplier: float = 5.
) -> dict:
    """Descriptive analog/TL physics consistency conditional on reported batch SEM."""
    if (len(trajectories)<4 or any(len(row)<3 for row in trajectories)
            or not 1<=sigma_multiplier<=10 or not math.isfinite(sigma_multiplier)):
        raise ValueError("require four independent seeds and >=3 active checkpoints")
    seedset=set()
    schedules=set()
    observations=[]
    hashes=set()
    for run in trajectories:
        seeds={r["seed"] for r in run}
        inactive={r["inactive"] for r in run}
        population={r["particles"] for r in run}
        if len(seeds)!=1 or len(inactive)!=1 or len(population)!=1:
            raise ValueError("mixed seed/physical source population in reaction history")
        seed=seeds.pop()
        if seed in seedset:
            raise ValueError("duplicate OpenMC independent seed")
        seedset.add(seed)
        batches=[x["batch"] for x in run]
        active=[x["active"] for x in run]
        if (any(a>=b for a,b in zip(batches,batches[1:]))
                or any(a>=b for a,b in zip(active,active[1:]))
                or any(x["active"] != x["batch"]-x["inactive"] for x in run)):
            raise ValueError("invalid OpenMC neutron active-batch schedule")
        schedules.add(tuple(batches))
        for r in run:
            if r["sha256"] in hashes:
                raise ValueError("duplicate OpenMC statepoint provenance")
            hashes.add(r["sha256"])
        analog=_disjoint_moments(run,"analog")
        tracklength=_disjoint_moments(run,"tracklength")
        windows=[]
        for a,t in zip(analog,tracklength):
            if (a["last_batch"] != t["last_batch"]
                    or a["active_generations"] != t["active_generations"]):
                raise ValueError("different neutron reaction tally time windows")
            detail={}
            for j,score in enumerate(SCORES):
                discrepancy=float(a["per_source_rate"][j]-t["per_source_rate"][j])
                conservative_sem_bound=float(
                    a["batch_reported_standard_error"][j]+
                    t["batch_reported_standard_error"][j]
                )
                if not math.isfinite(conservative_sem_bound) or conservative_sem_bound<=0:
                    raise ValueError("zero/invalid batch estimator uncertainty")
                discrepancy_ratio=abs(discrepancy)/conservative_sem_bound
                detail[score]={
                    "analog_per_source":float(a["per_source_rate"][j]),
                    "tracklength_per_source":float(t["per_source_rate"][j]),
                    "analog_minus_tracklength":discrepancy,
                    "sum_of_marginal_batch_standard_errors":conservative_sem_bound,
                    "discrepancy_over_conservative_error_bound":discrepancy_ratio,
                    "gross_inconsistency_screen_passed":
                        bool(discrepancy_ratio <= sigma_multiplier),
                }
            windows.append({
                "first_active_batch":a["first_batch"],
                "last_active_batch":a["last_batch"],
                "active_generations":a["active_generations"],
                "reactions":detail,
            })
        observations.append({
            "seed":seed,
            "source_particles_per_generation":run[0]["particles"],
            "inactive_generations":run[0]["inactive"],
            "last_statepoint_sha256":run[-1]["sha256"],
            "window_reactions":windows,
        })
    if len(schedules)!=1:
        raise ValueError("four-seed neutron reaction batch schedules changed")
    maxima={}
    allpass=True
    for score in SCORES:
        ratios=[w["reactions"][score]["discrepancy_over_conservative_error_bound"]
                for run in observations for w in run["window_reactions"]]
        failures=sum(not w["reactions"][score]["gross_inconsistency_screen_passed"]
                     for run in observations for w in run["window_reactions"])
        maxima[score]={
            "maximum_discrepancy_over_sum_marginal_batch_SEM":max(ratios),
            "screen_failure_count":failures,
            "window_observation_count":len(ratios),
        }
        allpass &= failures==0
    return {
        "analysis":"GENUINE_OPENMC_ANALOG_VS_TRACKLENGTH_DISJOINT_NEUTRON_REACTION_WINDOWS",
        "native_global_scores":list(SCORES),
        "matched_independent_seed_count":len(seedset),
        "saved_statepoint_batches":list(next(iter(schedules))),
        "estimator_discrepancy_multiplier_screen":sigma_multiplier,
        "within_seed_windows":observations,
        "reaction_consistency":maxima,
        "gross_neutron_reaction_estimator_agreement_screen_passed":bool(allpass),
        "fission_source_stationarity_validated":False,
        "nasa_nerva_physical_core_validated":False,
        "interpretation":(
            "Real neutron-only eigenvalue source tallies, with native analog "
            "reaction and independent tracklength estimator averaged over the "
            "SAME transport histories. Disjoint windows reconstructed from "
            "cumulative OpenMC sum and sum-square moments; unknown covariance "
            "means the sum of marginal SEMs bounds the difference SEM only "
            "conditional on representative batch error estimates. The 5x screen "
            "is a gross consistency guard, not independent-sample significance "
            "or NASA measured validation. Highly subcritical source bank has "
            "correlated/duplicated fission ancestors; source convergence fails."
        ),
    }


def audit_actual_reaction_estimators(summary_path: Path) -> dict:
    path=Path(summary_path)
    # First prove all provenance/energy/material/source reaction identities
    # using the independently validated real neutron reaction-window audit.
    reference=audit_real_reaction_windows(path)
    if not reference["all_reaction_window_partition_and_reconstruction_checks_passed"]:
        raise ValueError("real neutron reaction conservation failed")
    original=json.loads(path.read_text(encoding="utf-8"))
    stages=original["saved_source_bank_batches"]
    digits=len(str(original["batches"]))
    trajectories=[]
    for run in original["runs"]:
        final=Path(run["statepoint_path"])
        if not final.is_file() or _sha(final)!=run["statepoint_sha256"]:
            raise ValueError("real eigenvalue source statepoint hash mismatch")
        paths=[final.parent/f"statepoint.{batch:0{digits}d}.h5" for batch in stages]
        if paths[-1].resolve()!=final.resolve():
            raise ValueError("actual last OpenMC statepoint checkpoint changed")
        rows=[_checkpoint(q) for q in paths]
        if rows[0]["seed"]!=run["seed"]:
            raise ValueError("OpenMC reaction estimators have wrong source seed")
        trajectories.append(rows)
    report=compare_estimator_windows(trajectories)
    by_seed={entry["seed"]:entry for entry in reference["observed_seed_history"]}
    for src in report["within_seed_windows"]:
        prior=by_seed[src["seed"]]
        for pair, expected in zip(src["window_reactions"],prior["windows"]):
            if pair["last_active_batch"]!=expected["last_active_batch"]:
                raise ValueError("disjoint analog/TL window timing disagrees with reaction history")
            for score in SCORES:
                value=pair["reactions"][score]["tracklength_per_source"]
                true=expected["global_reaction_rates_per_source"][score]
                if not math.isclose(value,true,rel_tol=0,abs_tol=3e-12):
                    raise ValueError("real tracklength neutron rates mismatch independently filtered reactions")
    report["real_transport_summary_sha256"]=_sha(path)
    report["independent_reaction_history_identical_statepoints"]=True
    report["reaction_history_source"]="ACTUAL_OPENMC_MATERIAL_AND_INCIDENT_ENERGY_TRACKLENGTH"
    return report


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    report=audit_actual_reaction_estimators(args.summary)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Genuine OpenMC analog vs tracklength neutron reaction windows:",
          "PASS" if report["gross_neutron_reaction_estimator_agreement_screen_passed"]
          else "FAIL")
    for score,data in report["reaction_consistency"].items():
        print(score,"maximum relative discrepancy to conservative batch SEM bound:",
              data["maximum_discrepancy_over_sum_marginal_batch_SEM"])
    print("Spatial fission-source stationarity and NASA core fidelity: NOT VALIDATED")
    return 0 if report["gross_neutron_reaction_estimator_agreement_screen_passed"] else 2


if __name__=="__main__":
    raise SystemExit(main())
