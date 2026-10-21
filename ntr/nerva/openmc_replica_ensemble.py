"""Empirical spatial-heating uncertainty from >=4 independent OpenMC replicas.

Input is a set of separately seeded, fully normalized OpenMC mesh fields,
not a collection of successive batches from one eigenvalue calculation.
For every active voxel, compute a cross-replica *sample* standard error of
its fraction of prescribed total reactor power. This avoids incorrectly
using the diagonal OpenMC tally std as if normalized bins were independent.

Important: nominal SEM and a two-half split are diagnostics, NOT rigorous
confidence bounds (source bias, mesh bias and shared nuclear-data error).
Reactor power and reactor configuration remain externally prescribed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from itertools import combinations
import hashlib
import math

import numpy as np

from .openmc_replica_convergence import (
    MeshReplica, _check_transport_identity, load_mesh_replica
)


@dataclass(frozen=True)
class ReplicaEnsembleResult:
    source_kind: str
    independent_real_statepoints: bool
    replica_count: int
    reactor_power_w: float
    mesh_dimension: tuple[int, int, int]
    active_voxel_count: int
    active_fraction: float
    p95_active_relative_sem: float
    max_axial_fraction_sem: float
    half_split_total_variation: float
    maximum_balanced_split_total_variation: float | None
    tested_balanced_split_total_variation: float
    balanced_split_method: str
    balanced_splits_tested: int
    balanced_splits_exhaustive: bool
    maximum_leave_one_out_total_variation: float
    mean_pairwise_total_variation: float
    maximum_pairwise_total_variation: float
    tolerance_p95_relative_sem: float
    tolerance_max_axial_fraction_sem: float
    tolerance_half_split_total_variation: float
    tolerance_max_leave_one_out_total_variation: float
    status: str
    interpretation: str


def _reject_bad_threshold(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _worst_balanced_split(stack: np.ndarray, *, exhaustive_limit: int = 12
                         ) -> tuple[float | None, int]:
    """Maximum TV between the two group means, over all balanced splits.

    Exactly enumerate unique partitions for <=12 independently seeded runs.
    For larger ensembles callers use an order-invariant hash-sorted
    split and an independently gated maximum leave-one-out discrepancy.
    Complexity is combinatorial; never silently call a sampled maximum exact.
    """
    n = len(stack)
    if n > exhaustive_limit:
        return None, 0
    k = n // 2
    total = np.sum(stack, axis=0)
    maximum = 0.0
    checked = 0
    for group in combinations(range(n), k):
        # With even n, the complement gives the same total variation.
        if n % 2 == 0 and 0 not in group:
            continue
        first_sum = np.sum(stack[list(group)], axis=0)
        second_sum = total - first_sum
        discrepancy = 0.5 * float(np.sum(np.abs(
            first_sum / k - second_sum / (n - k)
        )))
        maximum = max(maximum, discrepancy)
        checked += 1
    return maximum, checked


def assess_replica_ensemble(
    replicas: list[MeshReplica], *,
    provenance: list[dict | None] | None = None,
    max_p95_relative_sem: float = 0.10,
    max_axial_fraction_sem: float = 0.01,
    max_half_split_total_variation: float = 0.05,
    max_leave_one_out_total_variation: float = 0.05,
    active_threshold_relative: float = 1e-5,
) -> tuple[ReplicaEnsembleResult, dict[str, np.ndarray]]:
    """Calculate empirical (n-1 denominator) cross-replica uncertainties.

    Active bins are chosen from mean normalized power fractions, not by
    selecting only favorable individual run values. For up to twelve
    samples, an EXACT worst balanced split is used for acceptance.
    All ensembles additionally gate the maximum leave-one-out discrepancy;
    input-order split remains a diagnostic but cannot cause acceptance.
    Above twelve samples the split is made deterministic by content hashes
    and is explicitly NOT called an exhaustive maximum.
    """
    n = len(replicas)
    if n < 4:
        raise ValueError("at least four independent replica samples required")
    for name,v in (
        ("max_p95_relative_sem",max_p95_relative_sem),
        ("max_axial_fraction_sem",max_axial_fraction_sem),
        ("max_half_split_total_variation",max_half_split_total_variation),
        ("max_leave_one_out_total_variation",max_leave_one_out_total_variation),
        ("active_threshold_relative",active_threshold_relative),
    ):
        _reject_bad_threshold(v,name)
    if active_threshold_relative >= 1:
        raise ValueError("active threshold must be < 1")
    first=replicas[0]
    real=first.statepoint_sha256 is not None
    if provenance is not None and len(provenance) != n:
        raise ValueError("provenance and replicas have different lengths")
    if real and (provenance is None or any(p is None for p in provenance)):
        raise ValueError("real replica ensemble requires every transport provenance")
    if not real and provenance is not None and any(p is not None for p in provenance):
        raise ValueError("synthetic samples must not be tagged real")

    statepoints: set[str] = set()
    seeds: set[int] = set()
    fractions=[]
    for i,rep in enumerate(replicas):
        if rep.dimension != first.dimension or (
            not np.array_equal(rep.lower_left_cm,first.lower_left_cm)
            or not np.array_equal(rep.upper_right_cm,first.upper_right_cm)
        ):
            raise ValueError("replica mesh topology/bounds changed")
        if rep.source_kind != first.source_kind:
            raise ValueError("real and synthetic replicas cannot be mixed")
        if not math.isclose(rep.reactor_power_w,first.reactor_power_w,rel_tol=1e-8):
            raise ValueError("prescribed thermal power changed across replicas")
        if np.any(rep.power_density_w_cm3 < 0) or not np.all(
            np.isfinite(rep.power_density_w_cm3)
        ):
            raise ValueError("invalid heating mean")
        if real:
            if not rep.statepoint_sha256 or not isinstance(rep.openmc_seed,int):
                raise ValueError("missing OpenMC statepoint or explicit seed")
            if rep.statepoint_sha256 in statepoints or rep.openmc_seed in seeds:
                raise ValueError("duplicate statepoint or random seed")
            statepoints.add(rep.statepoint_sha256)
            seeds.add(rep.openmc_seed)
            if rep.nuclear_data_sha256 != first.nuclear_data_sha256:
                raise ValueError("nuclear data changed between replicas")
            if i > 0:
                _check_transport_identity(provenance[0],provenance[i])
        v=rep.power_density_w_cm3*rep.voxel_volume_cm3/rep.reactor_power_w
        if not np.isclose(v.sum(),1.0,rtol=1e-7,atol=1e-9):
            raise ValueError("normalized heating fractions do not sum to unity")
        fractions.append(v)
    stack=np.stack(fractions,axis=0)
    mean=stack.mean(axis=0)
    std=stack.std(axis=0,ddof=1)
    sem=std/math.sqrt(n)
    active=mean>active_threshold_relative*float(mean.max())
    if not np.any(active):
        raise ValueError("no active heating voxels")
    rel_sem=np.zeros_like(mean)
    rel_sem[active]=sem[active]/mean[active]
    p95=float(np.percentile(rel_sem[active],95))
    axial_means=stack.sum(axis=(1,2))
    axial_sem=axial_means.std(axis=0,ddof=1)/math.sqrt(n)
    max_axial=float(np.max(axial_sem))
    split=n//2
    # Original input-order split is maintained as a descriptive output for
    # legacy consumers. It MUST NOT drive the PASS/FAIL decision.
    first_half=stack[:split].mean(axis=0)
    second_half=stack[split:].mean(axis=0)
    half_tv=0.5*float(np.sum(np.abs(first_half-second_half)))

    # Exhaustive order-INVARIANT balanced partitions for small ensembles.
    worst_balanced, tested_splits = _worst_balanced_split(stack)
    balanced_exhaustive = worst_balanced is not None
    if balanced_exhaustive:
        split_metric = worst_balanced
    else:
        # Large ensembles avoid combinatorial enumeration. Hashing the
        # normalized physics fields makes this fallback independent of
        # CLI input order, but it is NOT a worst-case estimate.
        ordered = sorted(
            stack, key=lambda field: hashlib.sha256(
                np.ascontiguousarray(field).tobytes()
            ).digest()
        )
        split_metric = 0.5 * float(np.abs(
            np.mean(ordered[:split],axis=0) -
            np.mean(ordered[split:],axis=0)
        ).sum())

    # The maximum deviation of ANY replica from all others cannot be
    # concealed by alternating input ordering. It is distinct from SEM.
    total_stack=np.sum(stack,axis=0)
    max_loo=max(
        0.5*float(np.abs(stack[i]-(total_stack-stack[i])/(n-1)).sum())
        for i in range(n)
    )
    pair_values=[
        0.5*float(np.sum(np.abs(stack[i]-stack[j])))
        for i in range(n) for j in range(i+1,n)
    ]
    passing=(
        p95 <= max_p95_relative_sem
        and max_axial <= max_axial_fraction_sem
        and split_metric <= max_half_split_total_variation
        and max_loo <= max_leave_one_out_total_variation
    )
    summary=ReplicaEnsembleResult(
        source_kind=first.source_kind,
        independent_real_statepoints=real,
        replica_count=n,
        reactor_power_w=first.reactor_power_w,
        mesh_dimension=first.dimension,
        active_voxel_count=int(np.count_nonzero(active)),
        active_fraction=float(np.mean(active)),
        p95_active_relative_sem=p95,
        max_axial_fraction_sem=max_axial,
        half_split_total_variation=half_tv,
        maximum_balanced_split_total_variation=worst_balanced,
        tested_balanced_split_total_variation=split_metric,
        balanced_split_method=(
            "exhaustive_worst_balanced_partition"
            if balanced_exhaustive else "deterministic_hash_partition"
        ),
        balanced_splits_tested=tested_splits,
        balanced_splits_exhaustive=balanced_exhaustive,
        maximum_leave_one_out_total_variation=max_loo,
        mean_pairwise_total_variation=float(np.mean(pair_values)),
        maximum_pairwise_total_variation=max(pair_values),
        tolerance_p95_relative_sem=max_p95_relative_sem,
        tolerance_max_axial_fraction_sem=max_axial_fraction_sem,
        tolerance_half_split_total_variation=max_half_split_total_variation,
        tolerance_max_leave_one_out_total_variation=max_leave_one_out_total_variation,
        status="PASS" if passing else "FAIL",
        interpretation=(
            "Empirical spread across independent eigenvalue OpenMC replicas "
            "under fixed geometry and reactor power. SEM assumes independent "
            "replicate estimates; systematic bias and fission-source convergence "
            "are NOT covered. Original two-half split is order-sensitive and "
            "reported for continuity only; acceptance uses the worst balanced "
            "split when at most 12 replicas and the maximum leave-one-out "
            "variation for ALL ensemble sizes. Above 12 replicas, a deterministic "
            "hash-split substitutes for exhaustive enumeration. These are "
            "numerical stress checks, not calibrated hypothesis tests or "
            "historical NERVA validation."
        ),
    )
    outputs={
        "mean_power_fraction":mean,
        "sample_std_power_fraction":std,
        "sem_power_fraction":sem,
        "active_mask":active,
        "relative_sem_active":rel_sem,
        "axial_power_fraction_mean":axial_means.mean(axis=0),
        "axial_power_fraction_sem":axial_sem,
    }
    return summary,outputs


def evaluate_ensemble_directories(
    power_dirs: list[Path], *,
    provenance_paths: list[Path] | None = None,
    allow_synthetic: bool = False,
    **kwargs,
) -> tuple[ReplicaEnsembleResult, dict[str,np.ndarray],list[dict]]:
    if len(power_dirs)<4:
        raise ValueError("at least four power directories required")
    if allow_synthetic and provenance_paths is not None:
        raise ValueError("synthetic mode must not claim real provenance")
    if not allow_synthetic and (
        provenance_paths is None or len(power_dirs)!=len(provenance_paths)
    ):
        raise ValueError("real ensemble requires one transport provenance per replica")
    replicas=[]
    provenance=[]
    for i,p in enumerate(power_dirs):
        r,meta=load_mesh_replica(
            Path(p),
            provenance_path=None if allow_synthetic else Path(provenance_paths[i]),
            allow_synthetic=allow_synthetic,
        )
        replicas.append(r)
        provenance.append(meta)
    summary,fields_out=assess_replica_ensemble(replicas,provenance=provenance,**kwargs)
    identities=[
        {
            "power_directory":str(rep.directory),
            "seed":rep.openmc_seed,
            "statepoint_sha256":rep.statepoint_sha256,
            "nuclear_data_sha256":rep.nuclear_data_sha256,
        }
        for rep in replicas
    ]
    return summary,fields_out,identities
