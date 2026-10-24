"""Strict OpenMC StatePoint -> source-normalized heating score evidence.

OpenMC's *own* Python API reads the statepoint. This module refuses arbitrary
HDF5 layouts, unmatched meshes, unqualified score/nuclide filters and
ambiguous run-mode normalization. No actual OpenMC simulation is performed.

IMPORTANT:
- Tally means and standard errors are eV per source particle for heating.
- Statistical covariance of spatial bins / material tallies is unavailable
  through the standard mean and std_dev arrays. Do not sum bin standard
  errors in quadrature and claim an independently validated total SEM.
- Stage 9 score passes through only if its disjoint material partitions
  satisfy strict per-bin conservation on the *same* mesh.
- Exact source rate or prescribed power remains independently required.
- A SHA of a statepoint and a separate data manifest verifies file identity,
  not that the materials match an independently computed diffusion matrix.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any
import numpy as np

from .fx2th_transport_heat_projection import (
    HeatingMeshScore, HeatingNormalization, ProjectionResult,
    material_fingerprint, project_mesh_heating,
)
from .fx2th_family_spacetime import SpatialFamilyData


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class StatepointHeatingEvidence:
    score: HeatingMeshScore
    run_mode: str
    seed: int
    n_realizations: int
    tally_ids: tuple[int, int, int]
    max_relative_standard_error: float
    statistically_tested_bins: int
    near_zero_bins: int
    photon_transport: bool
    is_actual_statepoint_reader: bool
    statistical_limit: float

    def project(self, normalization: HeatingNormalization,
                thermal_z_edges_cm: np.ndarray) -> ProjectionResult:
        if (self.run_mode == "eigenvalue"
                and normalization.mode != "externally-prescribed-power"):
            raise ValueError("eigenvalue statepoint requires independently prescribed power")
        return project_mesh_heating(self.score, normalization, thermal_z_edges_cm)


def _array_from_tally(tally: Any, mesh_filter_type: type,
                      *, min_realizations: int, relative_error_limit: float,
                      relative_bin_cutoff: float) -> tuple[np.ndarray,np.ndarray,tuple]:
    """Fail-closed extraction of exactly one regular 3-D mesh filter.

    Additional non-mesh material/cell filters are accepted ONLY if they each
    select one bin (e.g., a single material partition); arbitrary multibin
    filters can never be silently aggregated or reshaped here.
    """
    if int(tally.num_realizations) < min_realizations:
        raise ValueError("insufficient independent tally realizations")
    if tuple(tally.scores) != ("heating",):
        raise ValueError("tally must contain exactly the heating score")
    if tuple(str(n) for n in tally.nuclides) != ("total",):
        raise ValueError("tally must contain only the total nuclide bin")
    filters = tuple(tally.filters)
    if not filters or not isinstance(filters[0], mesh_filter_type):
        raise ValueError("first tally filter must be a regular MeshFilter")
    if any(isinstance(f,mesh_filter_type) or int(f.num_bins)!=1
           for f in filters[1:]):
        raise ValueError("extra filters must be non-mesh singleton partitions")
    mesh = filters[0].mesh
    if type(mesh).__name__ != "RegularMesh":
        raise ValueError("only Cartesian RegularMesh is supported")
    dims = tuple(int(x) for x in mesh.dimension)
    lo,hi = np.asarray(mesh.lower_left,dtype=float),np.asarray(mesh.upper_right,dtype=float)
    if (len(dims)!=3 or min(dims)<1 or lo.shape!=(3,) or hi.shape!=(3,)
            or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
            or np.any(hi<=lo)):
        raise ValueError("invalid 3D regular mesh geometry")
    for f in filters:
        if (getattr(f,"translation",None) is not None or
                getattr(f,"rotation",None) is not None):
            raise ValueError("translated/rotated mesh and filters are not supported")
    expected_shape = dims + ((1,)*len(filters[1:])) + (1,1)
    # Standard get_reshaped_data(value='mean', expand_dims=True) is the
    # documented OpenMC Tally API; first three expanded axes are xyz.
    vals = []
    for kind in ("mean","std_dev"):
        a = np.asarray(tally.get_reshaped_data(kind,expand_dims=True),dtype=float)
        if a.shape != expected_shape:
            raise ValueError(f"OpenMC {kind} expanded tally shape {a.shape} not {expected_shape}")
        v = a.reshape(dims)
        if not np.all(np.isfinite(v)) or np.any(v<0):
            raise ValueError(f"nonfinite or negative {kind} heating")
        vals.append(v)
    mean, sem = vals
    maximum = float(mean.max())
    if maximum<=0:
        raise ValueError("heating tally has no positive mean score")
    gate=mean>=maximum*relative_bin_cutoff
    if not np.any(gate):
        raise ValueError("no statistically meaningful heating bins")
    rel = sem[gate]/mean[gate]
    if np.any(rel>relative_error_limit):
        raise ValueError("heating tally uncertainty exceeds configured bin tolerance")
    # If a zero-mean cell has finite std_error, its relative uncertainty is
    # undefined, and the physical significance must be assessed upstream.
    if np.any((mean==0)&(sem>0)):
        raise ValueError("nonzero uncertainty on zero-mean heating bin")
    return mean,sem,(
        int(mesh.id),dims,tuple(float(v) for v in lo),
        tuple(float(v) for v in hi),int(tally.num_realizations),
        float(rel.max()),int(np.count_nonzero(gate)),int(gate.size-np.count_nonzero(gate))
    )


def extract_statepoint_objects(
    statepoint: Any, *, tally_ids: tuple[int,int,int],
    mesh_filter_type: type,
    source_id: str, material_fingerprint_sha256: str,
    statepoint_sha256: str, nuclear_data_sha256: str,
    relative_error_limit: float = .25,
    relative_bin_cutoff: float = .01,
    min_realizations: int = 10,
    source_kind: str = "synthetic",
    actual_statepoint_reader: bool = False,
) -> StatepointHeatingEvidence:
    """Extract three disjoint mesh heating tallies through the OpenMC API.

    This testable object-level helper may receive mocked OpenMC objects.
    Only read_openmc_heating_statepoint() opens a real OpenMC StatePoint.
    A synthetic/mock result must NEVER be promoted to genuine statepoint
    evidence by changing a flag in user-facing interpretation.
    """
    if (not isinstance(tally_ids,tuple) or len(tally_ids)!=3
            or len(set(tally_ids))!=3 or any(not isinstance(t,int) or t<=0 for t in tally_ids)):
        raise ValueError("three distinct positive tally IDs required")
    if (not math.isfinite(relative_error_limit) or not 0<relative_error_limit<=1
            or not math.isfinite(relative_bin_cutoff) or not 0<relative_bin_cutoff<1):
        raise ValueError("invalid relative uncertainty limits")
    if not isinstance(min_realizations,int) or min_realizations<2:
        raise ValueError("minimum tally realizations must be >= 2")
    run_mode=str(statepoint.run_mode)
    if run_mode not in ("fixed source","eigenvalue"):
        raise ValueError("unsupported OpenMC statepoint run mode")
    seed=int(statepoint.seed)
    if seed<0:
        raise ValueError("invalid OpenMC random seed")
    if source_kind not in ("synthetic","extracted-openmc-arrays"):
        raise ValueError("unsupported provenance category")
    if actual_statepoint_reader and source_kind!="extracted-openmc-arrays":
        raise ValueError("actual StatePoint loader needs extracted-openmc-arrays")
    arrays=[]
    meta=[]
    for tally_id in tally_ids:
        try:
            tally=statepoint.tallies[tally_id]
        except KeyError as e:
            raise ValueError(f"missing OpenMC tally ID {tally_id}") from e
        mean,sem,metadata=_array_from_tally(
            tally,mesh_filter_type,min_realizations=min_realizations,
            relative_error_limit=relative_error_limit,
            relative_bin_cutoff=relative_bin_cutoff)
        arrays.append((mean,sem))
        meta.append(metadata)
    common=meta[0][:5]
    if any(m[:5]!=common for m in meta[1:]):
        raise ValueError("heating tallies have different mesh/realizations")
    # Fully specified errors exist on each spatial tally; covariance of
    # scores across distinct material/mesh tallies remains unknown.
    dims=common[1]
    lo=np.asarray(common[2],dtype=float)
    hi=np.asarray(common[3],dtype=float)
    score=HeatingMeshScore(
        source_id=source_id,source_kind=source_kind,
        material_fingerprint_sha256=material_fingerprint_sha256,
        statepoint_sha256=statepoint_sha256,
        nuclear_data_sha256=nuclear_data_sha256,
        tally_score="heating",score_units="eV/source-history/mesh-bin",
        lower_left_cm=lo,upper_right_cm=hi,
        total_ev_per_history_bin=arrays[0][0],
        solid_ev_per_history_bin=arrays[1][0],
        hydrogen_ev_per_history_bin=arrays[2][0]
    )
    score.validate()
    return StatepointHeatingEvidence(
        score=score,run_mode=run_mode,seed=seed,n_realizations=int(common[4]),
        tally_ids=tally_ids,
        max_relative_standard_error=max(m[5] for m in meta),
        statistically_tested_bins=sum(m[6] for m in meta),
        near_zero_bins=sum(m[7] for m in meta),
        photon_transport=bool(statepoint.photon_transport),
        is_actual_statepoint_reader=bool(actual_statepoint_reader),
        statistical_limit=relative_error_limit
    )


def read_openmc_heating_statepoint(
    statepoint_path: str | Path,
    nuclear_data_manifest_path: str | Path,
    nuclear_material: SpatialFamilyData,
    *, tally_ids: tuple[int,int,int],
    source_id: str,
    relative_error_limit: float=.25,
    relative_bin_cutoff: float=.01,
    min_realizations: int=10,
) -> StatepointHeatingEvidence:
    """Read actual statepoint via OpenMC Python API, compute SHA of files.

    The data-manifest SHA proves which manifest was provided, NOT that its
    cross sections were used in OpenMC. Source/model consistency is also
    externally required; the physics compatibility fingerprint is only
    a deterministic declaration of the supplied diffusion inputs.
    """
    try:
        import openmc
    except ImportError as e:
        raise RuntimeError("openmc Python API is needed to read a real StatePoint") from e
    sp_hash=file_sha256(statepoint_path)
    data_hash=file_sha256(nuclear_data_manifest_path)
    # Import uses the official OpenMC high-level object reader; do not
    # replicate unstable raw HDF5 filter/bin formatting.
    with openmc.StatePoint(str(statepoint_path),autolink=False) as sp:
        return extract_statepoint_objects(
            sp,tally_ids=tally_ids,mesh_filter_type=openmc.MeshFilter,
            source_id=source_id,
            material_fingerprint_sha256=material_fingerprint(nuclear_material),
            statepoint_sha256=sp_hash,nuclear_data_sha256=data_hash,
            relative_error_limit=relative_error_limit,
            relative_bin_cutoff=relative_bin_cutoff,
            min_realizations=min_realizations,
            source_kind="extracted-openmc-arrays",
            actual_statepoint_reader=True,
        )


@dataclass(frozen=True)
class IndependentSeedReport:
    source_kind: str
    seeds: tuple[int,...]
    independent_runs: int
    mean_solid_power_w: np.ndarray
    mean_hydrogen_power_w: np.ndarray
    mean_unmodeled_power_w: np.ndarray
    sem_solid_power_w: np.ndarray
    sem_hydrogen_power_w: np.ndarray
    sem_unmodeled_power_w: np.ndarray
    total_mean_w: float
    reference_material_fingerprint_sha256: str
    # No full-covariance within a single statepoint is inferred.


def compare_independent_statepoints(
    evidence: tuple[StatepointHeatingEvidence,...],
    normalization: HeatingNormalization,
    thermal_z_edges_cm: np.ndarray,
    *, minimum_independent_seeds: int=4,
) -> IndependentSeedReport:
    """Calculate between-run SEM, preserving spatial-bin correlation in each run.

    Each run is an independent StatePoint with a different RNG seed,
    compatible tally/mesh and the same material fingerprint. Estimated SEM
    is based on variability of WHOLE projected run totals or components,
    not independent-bin quadrature from one statepoint.
    """
    if len(evidence)<minimum_independent_seeds or minimum_independent_seeds<2:
        raise ValueError("insufficient independently seeded runs")
    seed_ids=tuple(v.seed for v in evidence)
    hashes=[v.score.statepoint_sha256 for v in evidence]
    if len(set(seed_ids))!=len(seed_ids) or len(set(hashes))!=len(hashes):
        raise ValueError("independent OpenMC runs require distinct seeds and statepoint hashes")
    first=evidence[0]
    if any((v.run_mode!=first.run_mode or v.score.source_kind!=first.score.source_kind
            or v.score.material_fingerprint_sha256!=first.score.material_fingerprint_sha256
            or v.score.nuclear_data_sha256!=first.score.nuclear_data_sha256
            or v.score.total_ev_per_history_bin.shape!=first.score.total_ev_per_history_bin.shape
            or not np.array_equal(v.score.lower_left_cm,first.score.lower_left_cm)
            or not np.array_equal(v.score.upper_right_cm,first.score.upper_right_cm)
            or v.tally_ids!=first.tally_ids) for v in evidence[1:]):
        raise ValueError("ensemble statepoints must share mesh, material, data and tally IDs")
    results=[v.project(normalization,thermal_z_edges_cm) for v in evidence]
    def stats(name):
        a=np.asarray([getattr(p.deposition,name) for p in results],dtype=float)
        return a.mean(axis=0),a.std(axis=0,ddof=1)/np.sqrt(len(a))
    sm,ssem=stats("solid_power_w")
    hm,hsem=stats("hydrogen_power_w")
    om,osem=stats("other_power_w")
    return IndependentSeedReport(
        source_kind=first.score.source_kind,seeds=seed_ids,
        independent_runs=len(evidence),mean_solid_power_w=sm,
        mean_hydrogen_power_w=hm,mean_unmodeled_power_w=om,
        sem_solid_power_w=ssem,sem_hydrogen_power_w=hsem,
        sem_unmodeled_power_w=osem,
        total_mean_w=float(np.mean([p.mesh_total_power_w for p in results])),
        reference_material_fingerprint_sha256=first.score.material_fingerprint_sha256,
    )
