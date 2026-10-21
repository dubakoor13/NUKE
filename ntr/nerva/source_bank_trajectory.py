"""Time-resolved fission-source-bank diagnostics from actual OpenMC HDF5.

No criticality tuning or core design changes. Source-bank snapshots within
a single seed are genealogically correlated and may contain heavily cloned
source sites; their spatial drift is a descriptive diagnostic, NOT an
independent-generation significance test or convergence certification.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from .source_bank_convergence import read_source_bank


def source_bank_trajectory(
    seed_snapshots: list[list[Path]],
    *,
    lower_cm: tuple[float, float, float],
    upper_cm: tuple[float, float, float],
    bins: tuple[int, int, int] = (4, 4, 4),
) -> dict:
    """Compare >=3 checkpoints for >=4 independent seeds.

    Uses the same spatial histogram bins and physical bounds for each
    saved *actual* OpenMC eigenvalue source bank. The output exposes
    time-resolved spatial TV, between-seed disagreement, centroid drift,
    source-site duplication and nonindependence warnings.
    """
    if len(seed_snapshots) < 4:
        raise ValueError("at least four independent eigenvalue seeds required")
    count = len(seed_snapshots[0])
    if count < 3 or any(len(x) != count for x in seed_snapshots):
        raise ValueError("same >=3 saved batch checkpoints required for all seeds")
    visited_hashes: set[str] = set()
    visited_seeds: set[int] = set()
    target_batches: list[int] | None = None
    all_distributions = []
    run_summaries = []
    for paths in seed_snapshots:
        fields = []
        metadata = []
        for path in paths:
            hist, details = read_source_bank(
                Path(path), lower_cm=lower_cm, upper_cm=upper_cm, bins=bins
            )
            if details["statepoint_sha256"] in visited_hashes:
                raise ValueError("duplicate statepoint HDF5 data reused")
            visited_hashes.add(details["statepoint_sha256"])
            fields.append(hist)
            metadata.append(details)
        seeds = {meta["seed"] for meta in metadata}
        if len(seeds) != 1:
            raise ValueError("snapshots from different seeds in same trajectory")
        seed = seeds.pop()
        if seed in visited_seeds:
            raise ValueError("duplicate random seed in source trajectories")
        visited_seeds.add(seed)
        batches = [meta["completed_batch"] for meta in metadata]
        if any(a >= b for a, b in zip(batches, batches[1:])):
            raise ValueError("source-bank batches must be strictly increasing")
        if target_batches is None:
            target_batches = batches
        elif batches != target_batches:
            raise ValueError("source-bank checkpoint batches must match across seeds")

        ordered = np.stack(fields, axis=0)
        step_tv = [
            0.5 * float(np.sum(np.abs(b - a), dtype=np.float64))
            for a, b in zip(ordered, ordered[1:])
        ]
        centroids = np.asarray(
            [meta["spatial_centroid_cm"] for meta in metadata], dtype=float
        )
        centroid_steps = np.linalg.norm(np.diff(centroids, axis=0), axis=1)
        distinct_fraction = [
            meta["distinct_position_fraction"] for meta in metadata
        ]
        inverse_simpson = [
            meta["inverse_simpson_distinct_position_count"]
            for meta in metadata
        ]
        run_summaries.append({
            "seed": seed,
            "snapshots": metadata,
            "adjacent_snapshot_total_variation": step_tv,
            "adjacent_snapshot_centroid_shift_cm": centroid_steps.tolist(),
            "initial_to_final_total_variation": (
                0.5 * float(np.abs(ordered[-1] - ordered[0]).sum())
            ),
            "minimum_distinct_position_fraction": float(min(distinct_fraction)),
            "minimum_inverse_simpson_position_count": float(min(inverse_simpson)),
            "nominal_bank_size_is_independent_sample_count": False,
        })
        all_distributions.append(ordered)

    stack = np.stack(all_distributions, axis=0)
    n_seeds, n_batches = stack.shape[:2]
    between_seed_tv_by_batch = []
    between_seed_max_tv_by_batch = []
    pooled_sem_l2_by_batch = []
    for i in range(n_batches):
        bank = stack[:, i]
        pairwise = [
            0.5 * float(np.abs(bank[a] - bank[b]).sum())
            for a, b in combinations(range(n_seeds), 2)
        ]
        between_seed_tv_by_batch.append(float(np.mean(pairwise)))
        between_seed_max_tv_by_batch.append(float(np.max(pairwise)))
        # Empirical across-SEED spatial variation, not per-neutron iid SEM.
        sem = np.std(bank, axis=0, ddof=1) / np.sqrt(n_seeds)
        pooled_sem_l2_by_batch.append(float(np.linalg.norm(sem.ravel())))

    adjacent = np.array([
        item["adjacent_snapshot_total_variation"] for item in run_summaries
    ], dtype=float)
    aggregate_mean = adjacent.mean(axis=0)
    pooled = stack.mean(axis=0)
    pooled_adjacent_tv = [
        0.5 * float(np.abs(pooled[i + 1] - pooled[i]).sum())
        for i in range(n_batches - 1)
    ]
    return {
        "source_mode": "Actual OpenMC source_bank HDF5; no synthetic substitution",
        "independent_seed_count": n_seeds,
        "saved_batches": target_batches,
        "spatial_bins": list(bins),
        "mean_within_seed_adjacent_total_variation_by_interval":
            aggregate_mean.tolist(),
        "maximum_within_seed_adjacent_total_variation_by_interval":
            adjacent.max(axis=0).tolist(),
        "ensemble_mean_adjacent_total_variation_by_interval":
            pooled_adjacent_tv,
        "between_seed_mean_total_variation_by_batch":
            between_seed_tv_by_batch,
        "between_seed_max_total_variation_by_batch":
            between_seed_max_tv_by_batch,
        "between_seed_spatial_fraction_sem_l2_by_batch":
            pooled_sem_l2_by_batch,
        "runs": run_summaries,
        "source_stationarity_proven": False,
        "historical_nasa_nerva_benchmark_validated": False,
        "interpretation": (
            "These values describe observed drift and cross-seed variability "
            "of resampled fission-source populations. Exact site duplication "
            "reduces nominal source diversity; distinct positions are not "
            "independent fission histories. Time-adjacent banks are strongly "
            "correlated, and even stable summary metrics cannot certify "
            "fundamental-mode source convergence, k-effective accuracy or "
            "NASA NERVA physical fidelity."
        ),
    }
