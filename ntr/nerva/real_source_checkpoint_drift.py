"""Read-only HDF5 physics audit of 20 REAL OpenMC source checkpoints.

Checks the executed fission-bank sites, SHA256, k-generation history
prefixes, spatial drift, and source-position replication independently
against the archived original 4-seed, 5-checkpoint numerical observations.

Source sites are genealogically correlated and exact duplicate positions
do NOT estimate independent neutron histories. TV and JS are descriptive,
not source-convergence confidence intervals or a reactor design/tune.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import io
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .real_transport_evidence_gate import (
    _Bundle, audit_real_transport_evidence, sha256_bytes,
)


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return .5 * float(np.sum(np.abs(a - b)))


def _js(a: np.ndarray, b: np.ndarray) -> float:
    m = .5 * (a + b)
    p = a > 0
    q = b > 0
    return .5 * (float(np.sum(a[p] * np.log(a[p]/m[p]))) +
                 float(np.sum(b[q] * np.log(b[q]/m[q]))))


def _approximately(value, expected, label, *, atol=1e-12):
    if not math.isclose(float(value), float(expected),
                        rel_tol=1e-10, abs_tol=atol):
        raise ValueError("HDF5 source bank disagrees with archived " + label)


def audit_real_source_checkpoints(archive: str | Path) -> dict:
    """Verify twenty genuine bank/source histories, without redesign."""
    base = audit_real_transport_evidence(archive)
    bundle = _Bundle(Path(archive))
    try:
        summary = bundle.report("real_core_transport_summary.json")
        traj = summary["multi_checkpoint_source_bank_trajectory"]
        resolution = bundle.report("source_resolution_audit.json")
        batches = [40, 60, 80, 100, 120]
        if (summary.get("saved_source_bank_batches") != batches or
            traj.get("saved_batches") != batches or
            traj.get("spatial_bins") != [4,4,4] or
            traj.get("independent_seed_count") != 4 or
            traj.get("source_stationarity_proven") is not False or
            resolution.get("source_stationarity_validated") is not False or
            resolution.get("source_diversity_screen_status") !=
                "FAIL_SOURCE_CLONING_SCREEN"):
            raise ValueError("source trajectory does not match cold real OpenMC archive")
        spatial = summary["independent_seed_source_bank_shape_diagnostics"]
        low = np.asarray(spatial["spatial_lower_cm"], dtype=float)
        high = np.asarray(spatial["spatial_upper_cm"], dtype=float)
        if (low.shape != (3,) or high.shape != (3,) or
            not np.isfinite(low).all() or not np.isfinite(high).all() or
            np.any(high <= low)):
            raise ValueError("invalid archived source grid bounds")
        runs = traj.get("runs")
        if (not isinstance(runs, list) or len(runs) != 4 or
            [r["seed"] for r in runs] !=
                [r["seed"] for r in base["statepoints"]]):
            raise ValueError("source trajectory seed identities mismatch")

        all_histograms = {b: [] for b in batches}
        observations = []
        group_by_seed = []
        for run in runs:
            seed = int(run["seed"])
            snapshots = run.get("snapshots")
            if (not isinstance(snapshots, list) or
                [s.get("completed_batch") for s in snapshots] != batches):
                raise ValueError("missing source bank checkpoint manifest")
            old_k, old_entropy = None, None
            within_seed = []
            for batch, expected in zip(batches, snapshots):
                name = f"seed_{seed}/statepoint.{batch:03d}.h5"
                blob = bundle.read(name)
                sha = sha256_bytes(blob)
                if sha != expected["statepoint_sha256"]:
                    raise ValueError("source checkpoint HDF5 SHA256 mismatch: " + name)
                with h5py.File(io.BytesIO(blob), "r") as sp:
                    if (bytes(sp.attrs["filetype"]).rstrip(b"\0") != b"statepoint" or
                        bytes(sp["run_mode"][()]).rstrip(b"\0") != b"eigenvalue" or
                        int(sp.attrs["source_present"]) != 1 or
                        int(sp["seed"][()]) != seed or
                        int(sp["current_batch"][()]) != batch or
                        int(sp["n_particles"][()]) != 3000 or
                        int(sp["n_inactive"][()]) != 30 or
                        int(sp["n_realizations"][()]) != batch-30):
                        raise ValueError("executed source bank settings mismatch")
                    bank = sp["source_bank"][()]
                    history = np.asarray(sp["k_generation"][()], float)
                    entropy = np.asarray(sp["entropy"][()], float)
                if (bank.dtype.names is None or
                    not {"r","wgt"}.issubset(bank.dtype.names) or
                    len(bank) != 3000 or history.shape != (batch,) or
                    entropy.shape != (batch,) or
                    not np.isfinite(history).all() or
                    not np.isfinite(entropy).all()):
                    raise ValueError("invalid OpenMC source bank/history data")
                xyz = np.column_stack([bank["r"][k] for k in ("x","y","z")])
                w = np.asarray(bank["wgt"], float)
                if (not np.isfinite(xyz).all() or not np.isfinite(w).all() or
                    np.any(w<=0) or np.any(xyz<low) or np.any(xyz>high)):
                    raise ValueError("nonphysical or out-of-grid source positions")
                if old_k is not None:
                    np.testing.assert_array_equal(
                        old_k, history[:len(old_k)],
                        err_msg="native generation k history prefix differs")
                    np.testing.assert_array_equal(
                        old_entropy, entropy[:len(old_entropy)],
                        err_msg="native entropy history prefix differs")
                old_k, old_entropy = history, entropy
                _,inverse = np.unique(xyz,axis=0,return_inverse=True)
                weights = np.bincount(inverse, weights=w)
                positions = len(weights)
                shares = weights/weights.sum()
                simpson = 1./float(np.sum(shares**2))
                largest = float(np.max(shares))
                centroid = np.average(xyz,axis=0,weights=w)
                hist,_ = np.histogramdd(
                    xyz,bins=[4,4,4],
                    range=list(zip(low,high)),weights=w)
                hist = hist.ravel()/float(np.sum(w))
                if not np.isclose(hist.sum(),1.,rtol=0,atol=1e-12):
                    raise ValueError("source-bank spatial weight incomplete")
                nonzero = hist>0
                hentropy = -float(np.sum(
                    hist[nonzero]*np.log(hist[nonzero])))
                for value,field in (
                    (positions,"exact_distinct_site_positions"),
                    (positions/len(bank),"distinct_position_fraction"),
                    (simpson,"inverse_simpson_distinct_position_count"),
                    (largest,"largest_replicated_position_weight_fraction"),
                    (hentropy,"spatial_entropy_nats"),
                ):
                    _approximately(value,expected[field],f"{seed}/{batch}/{field}")
                np.testing.assert_allclose(
                    centroid,expected["spatial_centroid_cm"],
                    atol=1e-11,rtol=1e-10,
                    err_msg="native source centroid differs from report")
                all_histograms[batch].append(hist)
                within_seed.append(hist)
                observations.append({
                    "seed":seed,"batch":batch,"source_bank_sha256":sha,
                    "nominal_site_records":len(bank),
                    "exact_distinct_position_count":positions,
                    "inverse_simpson_effective_position_count":simpson,
                    "largest_exact_position_weight_share":largest,
                    "source_centroid_cm":centroid.tolist(),
                    "spatial_entropy_nats":hentropy,
                })
            group_by_seed.append(within_seed)

        tv_within=[]
        for i in range(4):
            mean = float(np.mean([
                _tv(seq[i],seq[i+1]) for seq in group_by_seed]))
            _approximately(
                mean,traj["mean_within_seed_adjacent_total_variation_by_interval"][i],
                "within-seed source TV "+str(i))
            tv_within.append(mean)
        tv_between=[]
        js_between=[]
        for i,batch in enumerate(batches):
            pairs=list(combinations(all_histograms[batch],2))
            mean_tv=float(np.mean([_tv(a,b) for a,b in pairs]))
            mean_js=float(np.mean([_js(a,b) for a,b in pairs]))
            _approximately(
                mean_tv,traj["between_seed_mean_total_variation_by_batch"][i],
                "between-seed source TV "+str(batch))
            if not 0<=mean_js<=math.log(2.)+1e-12:
                raise ArithmeticError("invalid Jensen-Shannon information")
            tv_between.append(mean_tv)
            js_between.append(mean_js)
        numbers=[o["exact_distinct_position_count"] for o in observations]
        effective=[o["inverse_simpson_effective_position_count"]
                   for o in observations]
        return {
            "schema":"nuke-real-source-checkpoints-v1",
            "evidence_kind":"ORIGINAL_EXECUTED_OPENMC_20_HDF5_SOURCE_BANK_CHECKPOINTS",
            "transport_summary_sha256":base["transport_summary_sha256"],
            "original_archive_sha256":base["archive_sha256"],
            "rng_seeds":[r["seed"] for r in runs],
            "saved_batches":batches,
            "real_native_statepoint_count":len(observations),
            "source_sites_per_bank":3000,
            "minimum_distinct_positions":min(numbers),
            "maximum_distinct_positions":max(numbers),
            "minimum_inverse_simpson_effective_position_count":min(effective),
            "maximum_inverse_simpson_effective_position_count":max(effective),
            "mean_within_seed_adjacent_total_variation":tv_within,
            "mean_between_seed_total_variation_by_batch":tv_between,
            "mean_between_seed_jensen_shannon_nats_by_batch":js_between,
            "all_checkpoint_sha256_settings_and_prefixes_verified":True,
            "all_independently_recomputed_source_distributions_match_archive":True,
            "source_stationarity_or_independent_genealogies_qualified":False,
            "source_diversity_status":"FAIL_SOURCE_CLONING_SCREEN",
            "physical_hot_nerva_qualification":False,
            "checkpoints":observations,
            "limitations":[
                "Repeated exact source locations describe positional replication, NOT independent particle genealogies.",
                "Within-seed HDF5 snapshots are genealogically correlated.",
                "Spatial total variation and Jensen-Shannon divergence are descriptive, not confidence intervals.",
                "Cold-core source observations do not prove hot critical NERVA operation or usable nuclear heating.",
            ],
        }
    finally:
        bundle.close()


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=audit_real_source_checkpoints(args.archive)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Native executed OpenMC source bank checkpoints:",report["real_native_statepoint_count"])
    print("Distinct positions:",report["minimum_distinct_positions"],"to",
          report["maximum_distinct_positions"])
    print("Effective position count:",
          report["minimum_inverse_simpson_effective_position_count"],"to",
          report["maximum_inverse_simpson_effective_position_count"])
    print("Physical source stationarity: NOT QUALIFIED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
