"""Stage 27: genuine OpenMC native global tally moment/uncertainty audit.

Recomputes conditional per-batch standard errors using raw OpenMC HDF5
SUM/SUM_SQ accumulators for *single-bin* global tallies, and reports
independent-seed variability without falsely assuming source stationarity.

CRITICAL: Never sum spatial-bin SUM_SQ to infer variance of an integrated
tally: covariance across mesh/material bins is unavailable from these data.
Likewise analog and tracklength scores share histories; do not combine their
marginal errors as independent. Genuine source nonstationarity precludes
formal hot-NERVA confidence claims even with correct Monte Carlo moments.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .real_native_tally_physics_ledger import audit_native_neutron_heating
from .real_native_root_topology import audit_native_root_topology
from .real_transport_evidence_gate import _Bundle, sha256_bytes

_GLOBAL_SCORES = {
    "nerva_global_neutron_balance": ("absorption", "scatter", "nu-scatter", "nu-fission"),
    "nerva_global_reaction_tracklength": ("absorption", "nu-fission"),
}
_ESTIMATORS = {
    "nerva_global_neutron_balance": "analog",
    "nerva_global_reaction_tracklength": "tracklength",
}
_SCORE_NAMES = {
    "nerva_global_neutron_balance": {
        "absorption": "analog_absorption",
        "scatter": "analog_scatter",
        "nu-scatter": "analog_nu_scatter",
        "nu-fission": "analog_nu_fission",
    },
    "nerva_global_reaction_tracklength": {
        "absorption": "tracklength_absorption",
        "nu-fission": "tracklength_nu_fission",
    },
}


def conditional_raw_moments(total: float, total_sq: float, n: int) -> dict:
    """OpenMC standard error of a mean from N active-batch realizations.

    OpenMC variance convention: SEM²=(SUM_SQ/N-(SUM/N)²)/(N-1).
    The result is a conditional diagnostic; serial source correlations and
    bias are NOT quantified by this per-realization variance.
    """
    if (type(n) is not int or n < 2 or not math.isfinite(float(total)) or
        not math.isfinite(float(total_sq)) or total_sq < 0):
        raise ValueError("invalid raw OpenMC tally moment or realization count")
    mean = total/n
    raw = total_sq/n - mean*mean
    scale = max(abs(total_sq/n), mean*mean)
    # Permit only a tiny floating-point cancellation error, not a wrong sumsq.
    if raw < -5e-12*scale:
        raise ValueError("impossible raw tally second moment")
    variance = max(0., raw)/(n-1)
    if not math.isfinite(variance):
        raise ValueError("nonfinite native tally batch SEM")
    return {
        "mean_per_starting_neutron": float(mean),
        "native_conditional_batch_sem_NOT_SOURCE_CORRECTED": float(math.sqrt(variance)),
        "raw_active_batch_sum": float(total),
        "raw_active_batch_sum_squares": float(total_sq),
        "active_batch_realizations": n,
    }


def _text(value) -> str:
    if not isinstance(value, (bytes, np.bytes_)):
        raise ValueError("invalid original OpenMC tally metadata string")
    return bytes(value).decode("utf-8").rstrip("\x00")


def extract_native_global_moments(statepoint: h5py.File, n: int) -> dict:
    """Only permit *single-score-domain bin* tallies for global SEMs."""
    found = {}
    for name, scores in _GLOBAL_SCORES.items():
        matches = [
            group for group in statepoint["tallies"].values()
            if isinstance(group, h5py.Group) and "name" in group
            and _text(group["name"][()]) == name
        ]
        if len(matches) != 1:
            raise ValueError("missing/duplicated original single-domain global tally: " + name)
        group = matches[0]
        score_bins = tuple(_text(x) for x in group["score_bins"][()])
        if (len(set(score_bins)) != len(score_bins) or
            not set(scores).issubset(score_bins) or
            _text(group["estimator"][()]) != _ESTIMATORS[name] or
            int(group["n_realizations"][()]) != n):
            raise ValueError("invalid native global tally estimator/scores/realizations: " + name)
        data = np.asarray(group["results"][()], dtype=np.float64)
        # Bins >1 would require cross-bin batch covariance, NOT in SUM_SQ.
        if (data.shape != (1, len(score_bins), 2) or
            not np.isfinite(data).all()):
            raise ValueError("global tally cannot be reduced to one bin: " + name)
        for score in scores:
            j = score_bins.index(score)
            value = conditional_raw_moments(
                float(data[0,j,0]), float(data[0,j,1]), n)
            found[_SCORE_NAMES[name][score]] = value
    global_tallies = np.asarray(statepoint["global_tallies"][()], dtype=np.float64)
    if global_tallies.shape != (4,3) or not np.isfinite(global_tallies).all():
        raise ValueError("unexpected native global neutron-leakage format")
    # Source: OpenMC accumulate_global_tallies, global_tallies[:,1:3]
    # retain per-active-batch SUM and SUM_SQ after source-weight scaling.
    found["vacuum_neutron_leakage"] = conditional_raw_moments(
        float(global_tallies[3,1]), float(global_tallies[3,2]), n)
    return found


def between_seed_descriptives(rows: list[dict]) -> dict:
    """Descriptive independent-seed variability; not a calibrated CI."""
    if len(rows) < 3 or len(set(row["seed"] for row in rows)) != len(rows):
        raise ValueError("need distinct independent original OpenMC seed IDs")
    keys = tuple(rows[0]["moments"])
    if any(tuple(row["moments"]) != keys for row in rows):
        raise ValueError("inconsistent native score definitions across seeds")
    report = {}
    for key in keys:
        means = np.asarray([
            r["moments"][key]["mean_per_starting_neutron"] for r in rows
        ], dtype=float)
        sems = np.asarray([
            r["moments"][key]["native_conditional_batch_sem_NOT_SOURCE_CORRECTED"]
            for r in rows
        ], dtype=float)
        if not np.isfinite(means).all() or not np.isfinite(sems).all():
            raise ValueError("nonfinite native seed score")
        variation = float(np.std(means,ddof=1))
        rms_sem = float(np.sqrt(np.mean(sems**2)))
        report[key] = {
            "observed_unweighted_seed_mean": float(np.mean(means)),
            "observed_between_seed_standard_deviation": variation,
            "observed_seed_mean_standard_error_IF_STATIONARY": variation/math.sqrt(len(rows)),
            "native_conditional_batch_sem_root_mean_square": rms_sem,
            "between_seed_variance_to_within_seed_native_sem_variance_diagnostic":
                (variation/rms_sem)**2 if rms_sem>0 else None,
            "individual_seed_means": means.tolist(),
            "individual_seed_native_conditional_batch_SEMs": sems.tolist(),
        }
    a = np.asarray([r["moments"]["analog_absorption"]["mean_per_starting_neutron"]
                    for r in rows])
    t = np.asarray([r["moments"]["tracklength_absorption"]["mean_per_starting_neutron"]
                    for r in rows])
    diff = t-a
    report["paired_tracklength_minus_analog_absorption"] = {
        "observed_paired_seed_mean_difference": float(np.mean(diff)),
        "observed_paired_seed_standard_deviation": float(np.std(diff,ddof=1)),
        "observed_paired_seed_sem_IF_STATIONARY": float(
            np.std(diff,ddof=1)/math.sqrt(len(rows))),
        "score_level_analog_tracklength_covariance_from_raw_marginals_available": False,
    }
    return report


def audit_real_global_uncertainty(artifact: str | Path) -> dict:
    """Audit ORIGINAL 4-seed binary HDF5; never fabricate source/heat."""
    topology = audit_native_root_topology(artifact)
    physics = audit_native_neutron_heating(artifact)
    if (topology["qualified_hot_nerva_core"] is not False or
        physics["source_stationarity_demonstrated"] is not False or
        physics["physical_local_heating_qualified"] is not False):
        raise ValueError("scientific gate improperly promoted")
    prev = {row["seed"]: row for row in physics["seed_reaction_scores"]}
    source = _Bundle(Path(artifact))
    try:
        rows = []
        for row in topology["per_seed_native_geometry"]:
            seed = row["seed"]
            p = prev.get(seed)
            if p is None:
                raise ValueError("native geometry and reaction score seeds differ")
            name = f"seed_{seed}/statepoint.120.h5"
            blob = source.read(name)
            if sha256_bytes(blob) != p["native_statepoint_sha256"]:
                raise ValueError("genuine raw tally statepoint identity mismatch")
            with h5py.File(io.BytesIO(blob), "r") as h5:
                n = int(h5["n_realizations"][()])
                moments = extract_native_global_moments(h5, n)
            for score, prior_name in (
                ("analog_absorption", "global_analog_absorption_per_start"),
                ("analog_nu_fission", "global_analog_nu_fission_per_start"),
                ("tracklength_absorption", "global_tracklength_absorption_per_start"),
                ("tracklength_nu_fission", "global_tracklength_nu_fission_per_start"),
                ("vacuum_neutron_leakage", "native_neutron_vacuum_leakage_per_start"),
            ):
                if not math.isclose(moments[score]["mean_per_starting_neutron"],
                                    p[prior_name], rel_tol=1e-11,abs_tol=1e-12):
                    raise ValueError("native raw first moment differs from Stage 24")
            balance = (moments["analog_absorption"]["mean_per_starting_neutron"]
                      +moments["vacuum_neutron_leakage"]["mean_per_starting_neutron"]
                      -moments["analog_nu_scatter"]["mean_per_starting_neutron"]
                      +moments["analog_scatter"]["mean_per_starting_neutron"])
            if not math.isclose(balance,1.,rel_tol=0,abs_tol=1e-10):
                raise ValueError("native per-seed neutron population identity not conserved")
            rows.append({
                "seed": seed,
                "native_statepoint_sha256": sha256_bytes(blob),
                "active_batch_realizations": n,
                "observed_neutron_population_closure": balance,
                "moments": moments,
            })
        if len(rows)!=4:
            raise ValueError("original four-seed evidence incomplete")
        return {
            "schema": "nuke-native-global-moment-uncertainty-stage27-v1",
            "evidence_kind": "FOUR_ORIGINAL_OPENMC_0153_HDF5_GLOBAL_SUM_SUMSQ",
            "original_neutron_only_cold_surrogate_seed_count": len(rows),
            "raw_sum_sq_native_batch_estimator_recomputed": True,
            "mesh_bin_covariance_available": False,
            "analog_tracklength_covariance_available": False,
            "independent_stationary_fission_source_qualified": False,
            "native_batch_conditional_sem_physically_qualified": False,
            "local_neutron_photon_heating_qualified": False,
            "absolute_reactor_watts_normalization_permitted": False,
            "historical_hot_nerva_validation": False,
            "per_seed_native_global_moments": rows,
            "between_seed_descriptive_statistics": between_seed_descriptives(rows),
            "limitations": [
                "Native standard errors are computed correctly from raw HDF5 SUM/SUM_SQ for single global bins only.",
                "Fission source genealogy/serial dependence can bias nominal per-active-batch uncertainty.",
                "Four independent RNG seeds are not proof of stationary or well-mixed source distributions.",
                "Raw marginal moments cannot reconstruct analog/tracklength score covariance.",
                "Do not add mesh-bin second moments to claim full-core tally uncertainty.",
                "Cold subcritical neutron-only surrogate cannot qualify hot nuclear heating or thermal power.",
            ],
        }
    finally:
        source.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit_real_global_uncertainty(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Original OpenMC seeds audited:", result["original_neutron_only_cold_surrogate_seed_count"])
    for score in ("analog_absorption", "tracklength_absorption",
                  "vacuum_neutron_leakage"):
        record = result["between_seed_descriptive_statistics"][score]
        print(score, "seed mean",record["observed_unweighted_seed_mean"],
              "between-seed sd",record["observed_between_seed_standard_deviation"])
    print("Stationary hot-core neutron transport/heating confidence: NOT QUALIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
