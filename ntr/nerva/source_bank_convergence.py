"""Observe REAL OpenMC fission-source-bank spatial stability across seeds.

Uses actual eigenvalue statepoint HDF5 source_bank positions, not Shannon
entropy alone. Within-seed intermediate/final snapshots are correlated.
Cross-seed sample SEM is empirical, not an independent particle uncertainty.
The resulting measurements are diagnostics; NO source convergence is
certified by this helper, regardless of the numerical values.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import hashlib
import math

import h5py
import numpy as np


def _digest(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as fd:
        for chunk in iter(lambda: fd.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def read_source_bank(
    path: Path, *,
    lower_cm: tuple[float,float,float],
    upper_cm: tuple[float,float,float],
    bins: tuple[int,int,int]=(4,4,4),
) -> tuple[np.ndarray, dict]:
    """Read OpenMC HDF5 eigenvalue fission bank and score spatial fractions.

    Never mistake arbitrary particle coordinates or synthetic tally arrays
    for a real source bank. The caller must supply independent provenance.
    """
    bounds_low=np.asarray(lower_cm,dtype=float)
    bounds_high=np.asarray(upper_cm,dtype=float)
    if (bounds_low.shape!=(3,) or bounds_high.shape!=(3,)
            or not np.isfinite(bounds_low).all()
            or not np.isfinite(bounds_high).all()
            or np.any(bounds_high<=bounds_low)):
        raise ValueError("invalid fission-source scoring bounds")
    if (len(bins)!=3 or any(
        isinstance(v,bool) or not isinstance(v,int) or v<2 or v>64 for v in bins
    )):
        raise ValueError("source bank mesh bins must be 3 integers in [2,64]")
    with h5py.File(path,"r") as h:
        typ=h.attrs.get("filetype")
        if typ is None or bytes(typ).rstrip(b"\0") != b"statepoint":
            raise ValueError("source bank must belong to OpenMC statepoint")
        if (h.get("run_mode") is None
                or bytes(h["run_mode"][()]).rstrip(b"\0")!=b"eigenvalue"):
            raise ValueError("source bank must come from eigenvalue transport")
        if int(h.attrs.get("source_present",0)) != 1:
            raise ValueError("OpenMC statepoint source_bank not saved")
        if "source_bank" not in h or "seed" not in h:
            raise ValueError("OpenMC fission source bank or seed missing")
        source=h["source_bank"][()]
        seed=int(h["seed"][()])
        batch=int(h["current_batch"][()])
        if source.ndim !=1 or len(source)<2:
            raise ValueError("too few saved fission source sites")
        if (source.dtype.names is None or "r" not in source.dtype.names
                or "wgt" not in source.dtype.names):
            raise ValueError("not an OpenMC structured fission-source bank")
        r=source["r"]
        if r.dtype.names != ("x","y","z"):
            raise ValueError("invalid OpenMC source-bank coordinates")
        xyz=np.column_stack((r["x"],r["y"],r["z"]))
        weight=np.asarray(source["wgt"],dtype=float)
    if (not np.isfinite(xyz).all() or not np.isfinite(weight).all()
            or np.any(weight<=0)):
        raise ValueError("nonfinite or nonpositive source bank sites")
    # Boundaries are the scored reactor lattice envelope. Missing spatial
    # mass would make sampled source distributions incomparable.
    tolerance=1e-8
    if np.any(xyz<bounds_low-tolerance) or np.any(xyz>bounds_high+tolerance):
        raise ValueError("OpenMC fission source site lies outside scoring bounds")
    hist,_ = np.histogramdd(
        xyz,bins=bins,range=tuple(zip(bounds_low,bounds_high)),
        weights=weight,
    )
    count=float(np.sum(weight))
    total=float(np.sum(hist))
    if not math.isclose(total,count,rel_tol=1e-12,abs_tol=1e-8):
        raise ValueError("OpenMC source bank spatial histogram loses sites")
    frac=hist/total
    # Exact repeated coordinates in OpenMC's resampled source bank expose
    # population cloning. This is NOT an estimator of independent fission
    # histories; descendants can be correlated even at distinct sites.
    distinct_sites, inverse = np.unique(
        xyz, axis=0, return_inverse=True
    )
    site_weight = np.bincount(
        inverse, weights=weight, minlength=len(distinct_sites)
    )
    probabilities = site_weight / total
    largest_site_fraction = float(np.max(probabilities))
    inverse_simpson_sites = float(
        1.0 / np.sum(probabilities * probabilities)
    )
    centroid=np.average(xyz,axis=0,weights=weight)
    nonzero=frac[frac>0]
    entropy=float(-np.sum(nonzero*np.log(nonzero)))
    return frac,{
        "seed":seed,
        "completed_batch":batch,
        "source_sites":int(len(source)),
        "exact_distinct_site_positions":int(len(distinct_sites)),
        "distinct_position_fraction":float(len(distinct_sites)/len(source)),
        "largest_replicated_position_weight_fraction":largest_site_fraction,
        "inverse_simpson_distinct_position_count":inverse_simpson_sites,
        "source_weights":count,
        "spatial_entropy_nats":entropy,
        "spatial_centroid_cm":centroid.tolist(),
        "statepoint_sha256":_digest(Path(path)),
    }


def compare_independent_sources(
    samples: list[tuple[Path,Path]], *,
    lower_cm: tuple[float,float,float],
    upper_cm: tuple[float,float,float],
    bins: tuple[int,int,int]=(4,4,4),
) -> dict:
    """Two saved bank snapshots per seed; >=4 different OpenMC RNG seeds.

    Exposes empirical between-seed source spatial variation and within-seed
    temporal change. NO pass/fail stationarity or criticality decisions.
    """
    if len(samples)<4:
        raise ValueError("at least four independent OpenMC seeds required")
    final_bank=[]
    records=[]
    seeds=set()
    hashes=set()
    for early_file,late_file in samples:
        early,meta_a=read_source_bank(
            Path(early_file),lower_cm=lower_cm,upper_cm=upper_cm,bins=bins
        )
        late,meta_b=read_source_bank(
            Path(late_file),lower_cm=lower_cm,upper_cm=upper_cm,bins=bins
        )
        if meta_a["seed"]!=meta_b["seed"]:
            raise ValueError("source snapshots from different OpenMC seeds")
        if meta_b["completed_batch"]<=meta_a["completed_batch"]:
            raise ValueError("source bank snapshots out of time order")
        if meta_b["seed"] in seeds:
            raise ValueError("duplicate OpenMC seed in source ensemble")
        seeds.add(meta_b["seed"])
        for row in (meta_a,meta_b):
            if row["statepoint_sha256"] in hashes:
                raise ValueError("duplicate OpenMC source statepoint contents")
            hashes.add(row["statepoint_sha256"])
        distance=0.5*float(np.sum(np.abs(early-late)))
        records.append({
            "seed":meta_b["seed"],
            "early_source":meta_a,
            "late_source":meta_b,
            "early_to_late_total_variation":distance,
            "early_to_late_centroid_shift_cm":float(
                np.linalg.norm(
                    np.asarray(meta_a["spatial_centroid_cm"])-
                    np.asarray(meta_b["spatial_centroid_cm"])
                )
            ),
        })
        final_bank.append(late)
    field=np.stack(final_bank)
    sample_mean=np.mean(field,axis=0)
    sem=np.std(field,axis=0,ddof=1)/math.sqrt(len(field))
    pairs=[
        0.5*float(np.sum(np.abs(field[i]-field[j])))
        for i,j in combinations(range(len(field)),2)
    ]
    active=sample_mean>0.01*float(sample_mean.max())
    p95=float(np.percentile(sem[active]/sample_mean[active],95))
    return {
        "source_mode":"OpenMC statepoint HDF5 source_bank",
        "independent_seed_count":len(field),
        "spatial_bins":list(bins),
        "spatial_lower_cm":list(lower_cm),
        "spatial_upper_cm":list(upper_cm),
        "between_seed_mean_source_total_variation":float(np.mean(pairs)),
        "between_seed_max_source_total_variation":float(np.max(pairs)),
        "p95_active_spatial_fraction_sem_relative_across_seeds":p95,
        "mean_early_to_late_source_total_variation":float(
            np.mean([r["early_to_late_total_variation"] for r in records])
        ),
        "runs":records,
        "source_convergence_independently_demonstrated":False,
        "historical_nasa_nerva_benchmark_validated":False,
        "interpretation":(
            "Actual OpenMC saved fission-source-bank locations, binned on "
            "an identical physical spatial mesh. Temporal snapshots from "
            "the same seed are correlated; independent seed means and "
            "their SEM are descriptive statistics only. Source-bank shape "
            "agreement and Shannon entropy alone cannot prove convergence "
            "of high-order fission modes or core-physics fidelity."
        ),
    }
