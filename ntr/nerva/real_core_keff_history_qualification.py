"""Read-only eigenvalue-generation history validation on actual OpenMC HDF5.

Verify exact source-history prefix persistence at each saved eigenvalue
checkpoint, then quantify descriptive cross-seed split-chain k diagnostics.
A stable k history does NOT imply a stationary spatial fission source, since
fission-site genealogies and resampling are strongly correlated.

No core design, enrichment, temperatures, controls, nuclear data, source
population, k-target or neutronic configuration is altered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


def _sha(file: Path) -> str:
    digest = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes(obj) -> bytes:
    if not isinstance(obj, (bytes, np.bytes_)):
        raise ValueError("statepoint HDF5 eigenvalue identity is not a bytestring")
    return bytes(obj).rstrip(b"\x00")


def read_real_keff_checkpoint(path: Path) -> dict:
    """Read exact OpenMC k_generation / entropy arrays from saved HDF5."""
    path = Path(path)
    try:
        with h5py.File(path, "r") as h:
            if (_bytes(h.attrs["filetype"]) != b"statepoint"
                    or _bytes(h["run_mode"][()]) != b"eigenvalue"
                    or int(h.attrs["photon_transport"]) != 0
                    or int(h.attrs.get("source_present", 0)) != 1):
                raise ValueError("requires genuine neutron-only eigenvalue statepoint")
            seed = int(h["seed"][()])
            batch = int(h["current_batch"][()])
            total = int(h["n_batches"][()])
            inactive = int(h["n_inactive"][()])
            particles = int(h["n_particles"][()])
            generations = int(h["generations_per_batch"][()])
            realized = int(h["n_realizations"][()])
            keff = np.asarray(h["k_generation"][()], dtype=np.float64)
            entropy = np.asarray(h["entropy"][()], dtype=np.float64)
            combined = np.asarray(h["k_combined"][()], dtype=np.float64)
    except (KeyError, OSError, TypeError, AttributeError) as exc:
        raise ValueError("missing or invalid genuine OpenMC generation/entropy history") from exc
    if (seed <= 0 or batch < 2 or total < batch or inactive < 0
            or inactive >= batch or particles <= 1 or generations <= 0
            or realized <= 0 or realized > (batch-inactive)*generations
            or keff.shape != (batch*generations,) or entropy.shape != (batch,)
            or combined.shape != (2,)
            or not np.isfinite(keff).all() or np.any(keff < 0)
            or not np.isfinite(entropy).all() or np.any(entropy < 0)
            or not np.isfinite(combined).all()
            or combined[0] <= 0 or combined[1] < 0):
        raise ValueError("invalid real OpenMC eigenvalue/entropy histories or batch settings")
    return {
        "seed": seed,
        "batch": batch,
        "configured_total_batches": total,
        "inactive_batches": inactive,
        "particles_per_generation": particles,
        "generations_per_batch": generations,
        "active_realizations": realized,
        "generation_keff": keff,
        "source_entropy": entropy,
        "combined_keff": combined,
        "sha256": _sha(path),
    }


def evaluate_split_chain_k(active_histories: list[np.ndarray]) -> dict:
    """Descriptive classical split-Rhat and lag-1 correlations, *not* ESS.

    k_generation is not OpenMC's k_combined multivariate estimator. Therefore
    this audit NEVER compares its naive SEM with the k_combined uncertainty
    as though the estimates were statistically identical.
    """
    if len(active_histories) < 4:
        raise ValueError("at least four independent executed k generation histories required")
    arrays = [np.asarray(x, dtype=float) for x in active_histories]
    length = len(arrays[0])
    if (length < 20 or any(x.shape != (length,)
            or not np.isfinite(x).all() or np.any(x < 0) for x in arrays)):
        raise ValueError("matched >=20 real active eigenvalue generations required")
    # Always split each independent source chain into two DISJOINT halves.
    half = length // 2
    chains = np.stack([part for x in arrays
                       for part in (x[:half], x[-half:])])
    n = chains.shape[1]
    means = chains.mean(axis=1)
    within = float(chains.var(axis=1, ddof=1).mean())
    between = float(n*means.var(ddof=1))
    if not math.isfinite(within) or within <= 0 or not math.isfinite(between):
        raise ValueError("unidentifiable split-chain eigenvalue fluctuations")
    estimated = ((n-1)/n)*within + between/n
    raw_rhat = math.sqrt(max(0., estimated/within))
    # Raw finite-sample Rhat can be <1. Do not report spuriously improved
    # convergence on account of a negative fluctuation in between variance.
    lag1=[]
    per_seed=[]
    for values in arrays:
        variance=float(np.var(values,ddof=1))
        if not math.isfinite(variance) or variance <= 0:
            raise ValueError("constant k-generation chain cannot establish uncertainty")
        a=values[:-1]
        b=values[1:]
        va=float(np.var(a))
        vb=float(np.var(b))
        rho=(float(np.mean((a-a.mean())*(b-b.mean()))/
                   math.sqrt(va*vb)) if va > 0 and vb > 0 else None)
        lag1.append(rho)
        per_seed.append({
            "mean_per_generation_k": float(values.mean()),
            "naive_iid_standard_error_NOT_RELIABLE": float(
                math.sqrt(variance/len(values))
            ),
            "active_first_half_mean":float(values[:half].mean()),
            "active_last_half_mean":float(values[-half:].mean()),
            "last_minus_first_half_k_shift":float(
                values[-half:].mean()-values[:half].mean()
            ),
            "lag_1_generation_k_correlation_descriptive":rho,
        })
    return {
        "independent_seed_count":len(arrays),
        "active_generations_per_seed":length,
        "split_chains":int(chains.shape[0]),
        "generations_in_each_split":int(n),
        "within_split_chain_variance_W":within,
        "between_split_chain_variance_B":between,
        "classical_split_chain_rhat_raw":raw_rhat,
        "classical_split_chain_rhat_lower_bounded_at_one":max(1.,raw_rhat),
        "per_seed":per_seed,
        "active_source_generation_histories_are_independent":False,
        "rhat_near_one_proves_spatial_source_convergence":False,
        "physically_validated_reactor_criticality":False,
        "note":(
            "Split Rhat and lag correlations use per-generation k histories "
            "only, not independent per-neutron histories or full spatial source "
            "shapes. Small Rhat is compatible with pathological source cloning. "
            "Do not treat per-generation naive IID SEM as a certified "
            "uncertainty or compare it directly with OpenMC k_combined SEM."
        ),
    }


def audit_real_core_keff_histories(summary_path: Path, spatial_audit_path: Path) -> dict:
    summary_path = Path(summary_path)
    spatial_audit_path = Path(spatial_audit_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    spatial = json.loads(spatial_audit_path.read_text(encoding="utf-8"))
    if (summary.get("execution") != "ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly") != "core"
            or spatial.get("analysis") !=
            "EXACT_SAME_OPENMC_SOURCE_BANKS_ON_NESTED_SPATIAL_GRIDS"
            or spatial.get("real_transport_summary_sha256") != _sha(summary_path)):
        raise ValueError("missing genuine spatial source-convergence audit provenance")
    resolutions = spatial.get("resolutions", [])
    if not isinstance(resolutions, list) or len(resolutions) < 2:
        raise ValueError("missing real source-resolution convergence checks")
    runs=summary.get("runs")
    batches=summary.get("saved_source_bank_batches")
    total=summary.get("batches")
    if (not isinstance(runs,list) or len(runs)<4
            or not isinstance(batches,list) or len(batches)<3
            or any(type(n) is not int or n<1 for n in batches)
            or any(a>=b for a,b in zip(batches,batches[1:]))
            or type(total) is not int or batches[-1]!=total):
        raise ValueError("invalid real OpenMC source checkpoint schedule")
    seen_seeds=set()
    used_hashes=set()
    outputs=[]
    histories=[]
    metadata=None
    for run in runs:
        seed=run.get("seed")
        file=Path(run.get("statepoint_path",""))
        if (type(seed) is not int or seed<=0 or seed in seen_seeds
                or not file.is_file()
                or _sha(file)!=run.get("statepoint_sha256")):
            raise ValueError("invalid genuine independent OpenMC seed/final statepoint hash")
        seen_seeds.add(seed)
        digits=len(str(total))
        checkpoints=[
            file.parent/f"statepoint.{b:0{digits}d}.h5" for b in batches
        ]
        if checkpoints[-1].resolve()!=file.resolve():
            raise ValueError("invalid terminal checkpoint in real OpenMC source history")
        records=[]
        final=None
        for b,path in zip(batches,checkpoints):
            if not path.is_file():
                raise ValueError("missing genuine intermediate eigenvalue statepoint")
            row=read_real_keff_checkpoint(path)
            if (row["seed"]!=seed or row["batch"]!=b
                    or row["sha256"] in used_hashes):
                raise ValueError("real k-generation checkpoint identity mismatch")
            used_hashes.add(row["sha256"])
            if final is None:
                final=row
            else:
                # This is a strong invariant of unchanged execution: saved
                # eigenvalue/entropy histories are cumulative *prefixes*.
                # A changed original checkpoint history is never legitimized
                # by merely recomputing its SHA-256 digest.
                if (row["inactive_batches"]!=final["inactive_batches"]
                        or row["particles_per_generation"]!=final["particles_per_generation"]
                        or row["generations_per_batch"]!=final["generations_per_batch"]
                        or row["configured_total_batches"]!=final["configured_total_batches"]
                        or not np.array_equal(
                            row["generation_keff"][:len(final["generation_keff"])],
                            final["generation_keff"])
                        or not np.array_equal(
                            row["source_entropy"][:len(final["source_entropy"])],
                            final["source_entropy"])):
                    raise ValueError("saved OpenMC k-generation/entropy histories lost exact prefix integrity")
                final=row
            records.append({
                "batch":b,"statepoint_sha256":row["sha256"],
                "active_realizations":row["active_realizations"],
                "generation_keff_count":len(row["generation_keff"]),
                "source_entropy_count":len(row["source_entropy"]),
                "combined_keff_observation":row["combined_keff"].tolist(),
            })
        if final["sha256"]!=run["statepoint_sha256"]:
            raise ValueError("terminal real eigenvalue statepoint provenance disagrees")
        identity=(final["configured_total_batches"],final["inactive_batches"],
                  final["particles_per_generation"],final["generations_per_batch"])
        if metadata is None:
            metadata=identity
        elif identity!=metadata:
            raise ValueError("different real physical core neutron source settings across seeds")
        batched = final["generation_keff"].reshape(
            total,final["generations_per_batch"]
        ).mean(axis=1)
        active=batched[final["inactive_batches"]:]
        if len(active)<20:
            raise ValueError("too few actual active OpenMC batches")
        histories.append(active)
        outputs.append({"seed":seed,"checkpoints":records})
    results=evaluate_split_chain_k(histories)
    for output,item in zip(outputs,results["per_seed"]):
        output.update(item)
    return {
        "audit":"ACTUAL_OPENMC_EIGENVALUE_HISTORY_PREFIX_AND_SPLIT_CHAINS",
        "real_transport_summary_sha256":_sha(summary_path),
        "real_spatial_source_audit_sha256":_sha(spatial_audit_path),
        "all_checkpoint_keff_and_entropy_prefixes_verified":True,
        "executed_core_eigenvalue_settings_identical_across_seeds":True,
        "spatial_fission_source_stationarity_validated":False,
        "existing_source_diversity_screen_status":
            spatial.get("source_diversity_screen_status"),
        "four_seed_split_chain_generation_k":results,
        "real_seed_checkpoints":outputs,
        "historical_nasa_nerva_core_validated":False,
        "interpretation":(
            "All actual saved eigenvalue and Shannon source histories are "
            "proven exact prefixes of the same executed OpenMC run. Classical "
            "split-chain k Rhat is DESCRIPTIVE only and cannot validate "
            "spatial source stationarity, eliminate genealogical correlations, "
            "qualify hot neutron data or validate a historical reactor. "
            "Do not treat k-generation as OpenMC's combined k estimator."
        ),
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--spatial-audit",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=audit_real_core_keff_histories(args.summary,args.spatial_audit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    check=result["four_seed_split_chain_generation_k"]
    print("Exact saved real OpenMC k/entropy history prefixes: PASS")
    print("Classical split-chain k-generation Rhat:",check[
        "classical_split_chain_rhat_lower_bounded_at_one"
    ])
    print("Spatial source stationarity and hot reactor physics: NOT VALIDATED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
