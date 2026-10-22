"""Read-only neutron multiplication / criticality NECESSARY-condition screen.

A numerical eigenvalue calculation with a very subcritical k cannot be a
physically validated operating critical NERVA core. Read four genuine OpenMC
0.15.3 statepoints, compare native collision/absorption/tracklength estimates
to k_combined, and explicitly report the mismatch against k=1.

This is an observed cold-model FALSIFICATION SCREEN, never a calibration,
source-convergence certificate, reactor criticality optimization or a
recommendation to change fuel enrichment, drums, geometry or materials.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def _ascii(value) -> bytes:
    if not isinstance(value, (bytes, np.bytes_)):
        raise ValueError("invalid OpenMC eigenvalue statepoint bytestring")
    return bytes(value).rstrip(b"\0")


# OpenMC 0.15.3 HDF5 global_tallies rows are, in order, collision,
# absorption, tracklength k, and leakage. These are RAW sums and squared
# sums, not precomputed mean/SEM arrays. Reject other OpenMC versions
# rather than pretending the binary format is universal.
NATIVE_GLOBAL_ROWS = ("k-collision", "k-absorption", "k-tracklength", "leakage")


def read_executed_multiplication(path: Path) -> dict:
    path = Path(path)
    try:
        with h5py.File(path, "r") as handle:
            if (_ascii(handle.attrs["filetype"]) != b"statepoint"
                    or _ascii(handle["run_mode"][()]) != b"eigenvalue"
                    or int(handle.attrs["photon_transport"]) != 0
                    or int(handle.attrs.get("source_present", 0)) != 1
                    or tuple(int(x) for x in handle.attrs["openmc_version"]) !=
                    (0, 15, 3)):
                raise ValueError("requires actual OpenMC 0.15.3 neutron eigenvalue statepoint")
            seed = int(handle["seed"][()])
            batch = int(handle["current_batch"][()])
            inactive = int(handle["n_inactive"][()])
            population = int(handle["n_particles"][()])
            count = int(handle["n_realizations"][()])
            combined = np.asarray(handle["k_combined"][()], dtype=float)
            moments = np.asarray(handle["global_tallies"][()], dtype=float)
    except (OSError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError("incomplete actual OpenMC multiplication HDF5 evidence") from exc
    if (seed <= 0 or batch <= inactive or inactive < 0
            or population < 2 or count < 2
            or combined.shape != (2,) or moments.shape != (4, 3)
            or not np.isfinite(combined).all()
            or not np.isfinite(moments).all()
            or combined[0] <= 0 or combined[1] <= 0
            or np.any(moments[:, 1:] < 0)):
        raise ValueError("invalid executed neutron multiplication estimates")
    n = float(count)
    scores = {}
    for name, (_, raw_sum, raw_square) in zip(NATIVE_GLOBAL_ROWS, moments):
        # Empirical SEM of active-batch raw tallies (OpenMC statepoint).
        # Cumulative sum-of-squares must obey Cauchy-Schwarz.
        delta = raw_square - raw_sum * raw_sum / n
        if delta < -1e-9*max(1., raw_square):
            raise ValueError("nonphysical OpenMC raw global-tally moments")
        mean = raw_sum / n
        sem = math.sqrt(max(0., delta) / (n*(n-1)))
        if (not math.isfinite(mean) or not math.isfinite(sem)
                or mean < 0 or sem < 0):
            raise ValueError("invalid native multiplication mean or uncertainty")
        scores[name] = {"mean": float(mean), "reported_sem": float(sem)}
    if any(scores[name]["mean"] <= 0 for name in NATIVE_GLOBAL_ROWS[:3]):
        raise ValueError("nonpositive native OpenMC neutron multiplication estimator")

    k, kerr = float(combined[0]), float(combined[1])
    for name in NATIVE_GLOBAL_ROWS[:3]:
        x = scores[name]
        # Marginal SEM may omit source genealogy correlation, so this is
        # a loose *numerical identity check*, never a joint confidence test.
        if abs(k-x["mean"]) > 10.0*(kerr+x["reported_sem"])+1e-8:
            raise ValueError(
                "native OpenMC neutron multiplication estimate disagrees with combined k"
            )
    return {
        "seed": seed,
        "completed_batch": batch,
        "inactive_batches": inactive,
        "particles_per_generation": population,
        "active_realizations": count,
        "hdf5_sha256": _digest(path),
        "combined_k_effective": k,
        "openmc_reported_combined_k_standard_error": kerr,
        "native_neutron_global_estimators": scores,
        "native_k_estimator_order": list(NATIVE_GLOBAL_ROWS),
    }


def evaluate_criticality_condition(runs: list[dict], *,
                                   sigma_screen: float = 10.) -> dict:
    """Observational necessary condition for critical operation, never design."""
    if (len(runs) < 4 or not math.isfinite(sigma_screen)
            or sigma_screen < 1):
        raise ValueError("requires >=4 independent k runs and finite screening multiplier")
    seeds = [row.get("seed") for row in runs]
    if (any(type(s) is not int or s <= 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("duplicate or invalid independent eigenvalue seeds")
    settings = {(row["completed_batch"],row["inactive_batches"],
                 row["particles_per_generation"],row["active_realizations"])
                for row in runs}
    if len(settings) != 1:
        raise ValueError("different executed core neutron transport settings")
    hashes = [row.get("hdf5_sha256") for row in runs]
    if len(set(hashes)) != len(runs) or any(
        not isinstance(s,str) or len(s)!=64 for s in hashes
    ):
        raise ValueError("duplicate or absent genuine source statepoint hashes")
    all_k = []
    per_seed = []
    for run in runs:
        k = float(run["combined_k_effective"])
        sem = float(run["openmc_reported_combined_k_standard_error"])
        if not math.isfinite(k) or k <= 0 or not math.isfinite(sem) or sem <= 0:
            raise ValueError("invalid physically executed OpenMC k and error")
        estimates = run.get("native_neutron_global_estimators",{})
        if set(estimates) != set(NATIVE_GLOBAL_ROWS):
            raise ValueError("missing native neutron global-tally multiplication estimator")
        variant = {}
        for name in NATIVE_GLOBAL_ROWS[:3]:
            x = estimates[name]
            mean = float(x["mean"])
            err = float(x["reported_sem"])
            if not math.isfinite(mean) or mean <= 0 or not math.isfinite(err) or err < 0:
                raise ValueError("invalid native scored k estimate")
            if abs(k-mean) > 10*(sem+err)+1e-8:
                raise ValueError("native neutron k disagrees with actual combined estimator")
            variant[name] = {
                "mean": mean,
                "upper_screen_value": mean + sigma_screen*err,
            }
            all_k.append(mean + sigma_screen*err)
        upper=k+sigma_screen*sem
        all_k.append(upper)
        per_seed.append({
            "seed":run["seed"],
            "combined_k":k,
            "reported_k_sem":sem,
            "conservative_reported_error_upper_screen_k":upper,
            "distance_from_k_equals_one":1.-k,
            "native_estimator_upper_screen_k":variant,
        })

    # These SEM multiples are only a descriptive contrast: source
    # correlations mean they do NOT define a formal confidence interval.
    far_subcritical = max(all_k) < 1.0
    return {
        "observed_state":"FAR_BELOW_CRITICAL" if far_subcritical
            else "CRITICALITY_NOT_ESTABLISHED",
        "all_native_and_combined_k_upper_screens_below_one":far_subcritical,
        "critical_eigenvalue_necessary_condition":1.0,
        "largest_native_or_combined_upper_screen_k":max(all_k),
        "screen_reported_sem_multiplier":sigma_screen,
        "independent_seed_count":len(seeds),
        "per_seed_physical_neutron_multiplication":per_seed,
        "cold_model_may_be_claimed_a_critical_operating_nerva_core":False,
        "nuclear_reactor_physical_criticality_validated":False,
        "fission_source_stationarity_demonstrated":False,
        "openmc_statistical_sem_is_genealogically_independent":False,
        "interpretation":(
            "Four actually executed cold neutron-only eigenvalue calculations "
            "and the native collision/absorption/tracklength estimators were "
            "compared against the fundamental necessary condition k=1 for a "
            "self-sustaining critical reactor. Reported MC SEM multiples are "
            "descriptive, NOT calibrated confidence limits where fission "
            "source genealogies are correlated. This is a disqualifying "
            "observed mismatch, NOT an enrichment/control/geometry adjustment "
            "and not a replacement for source convergence, hot nuclear data "
            "or independent NASA benchmark validation."
        ),
    }


def audit_real_core_criticality(summary_file: Path,
                               source_audit_file: Path) -> dict:
    summary_file = Path(summary_file)
    source_audit_file = Path(source_audit_file)
    summary=json.loads(summary_file.read_text(encoding="utf-8"))
    source=json.loads(source_audit_file.read_text(encoding="utf-8"))
    if (summary.get("execution")!="ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly")!="core"
            or source.get("analysis")!=
            "EXACT_SAME_OPENMC_SOURCE_BANKS_ON_NESTED_SPATIAL_GRIDS"
            or source.get("real_transport_summary_sha256")!=_digest(summary_file)
            or source.get("source_stationarity_validated") is not False):
        raise ValueError("genuine OpenMC source-convergence qualification is missing")
    runs=summary.get("runs")
    if not isinstance(runs,list) or len(runs)<4:
        raise ValueError("requires four actually executed neutron eigenvalue runs")
    audited=[]
    for entry in runs:
        file=Path(entry.get("statepoint_path",""))
        if not file.is_file():
            raise ValueError("missing real OpenMC core statepoint")
        observed=read_executed_multiplication(file)
        if (observed["seed"] != entry.get("seed")
                or observed["hdf5_sha256"] != entry.get("statepoint_sha256")
                or observed["completed_batch"] != summary.get("batches")
                or observed["inactive_batches"]!=summary.get("inactive_batches")
                or observed["particles_per_generation"]!=summary.get("particles_per_generation")
                or not math.isclose(observed["combined_k_effective"],
                                    float(entry["keff_mean"]),rel_tol=1e-11,
                                    abs_tol=1e-13)):
            raise ValueError("executed criticality evidence provenance mismatch")
        audited.append(observed)
    result=evaluate_criticality_condition(audited)
    return {
        "audit":"GENUINE_OPENMC_CORE_MULTIPLICATION_VERSUS_CRITICALITY_CONDITION",
        "transport_summary_sha256":_digest(summary_file),
        "spatial_source_audit_sha256":_digest(source_audit_file),
        "cross_sections_xml_sha256":summary["cross_sections_xml_sha256"],
        "actual_neutron_transport_execution":True,
        "historical_nerva_model_fidelity_validated":False,
        **result,
    }


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary",type=Path,required=True)
    ap.add_argument("--spatial-audit",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    report=audit_real_core_criticality(args.summary,args.spatial_audit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Physical criticality necessary-condition assessment:",
          report["observed_state"])
    print("Largest native/combined k estimator screen:",
          report["largest_native_or_combined_upper_screen_k"])
    print("NERVA operating criticality: NOT VALIDATED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
