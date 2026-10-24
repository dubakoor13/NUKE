"""Multiresolution spatial-source diagnostic on authentic OpenMC HDF5 banks.

Extends the Stage 22 native 20-checkpoint HDF5 verification with exact
probability-grid coarsening and reactor-axis radial/axial projections.
Physical interpretation: source-shape reproducibility and coarse/fine
resolution sensitivity. NOT a convergence certificate, independent
fission-family count, core design optimization, or new neutron transport.

Original data: GitHub Actions 39933618638, artifact 12492499186.
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

from .real_transport_evidence_gate import _Bundle, sha256_bytes
from .real_source_checkpoint_drift import audit_real_source_checkpoints

_BINS=(2,4,8,16)
_SEEDS=(13007,23011,33013,43019)
_BATCHES=(40,60,80,100,120)


def coarsen_source_cube(cube: np.ndarray, out_bins: int) -> np.ndarray:
    """Exactly sum 3D probability cells; no interpolation or renormalization."""
    a=np.asarray(cube,dtype=float)
    if (a.ndim!=3 or a.shape[0]<2 or len(set(a.shape))!=1 or
        not isinstance(out_bins,int) or isinstance(out_bins,bool) or
        out_bins<2 or a.shape[0]%out_bins!=0 or
        not np.isfinite(a).all() or np.any(a<0) or
        not np.isclose(a.sum(),1.,atol=1e-10,rtol=0)):
        raise ValueError("normalized cubic source and integral coarsening required")
    r=a.shape[0]//out_bins
    c=a.reshape(out_bins,r,out_bins,r,out_bins,r).sum(axis=(1,3,5))
    if not np.isclose(c.sum(),1.,atol=1e-12,rtol=0):
        raise ArithmeticError("source probability not conserved by coarsening")
    return c


def spatial_tv(a:np.ndarray,b:np.ndarray)->float:
    a,b=np.asarray(a,float),np.asarray(b,float)
    if (a.shape!=b.shape or np.any(~np.isfinite(a)) or
        np.any(~np.isfinite(b)) or np.any(a<0) or np.any(b<0) or
        not np.isclose(a.sum(),1.,atol=1e-10,rtol=0) or
        not np.isclose(b.sum(),1.,atol=1e-10,rtol=0)):
        raise ValueError("comparable normalized source distributions required")
    return .5*float(np.abs(a-b).sum())


def spatial_js(a:np.ndarray,b:np.ndarray)->float:
    a,b=np.asarray(a,float),np.asarray(b,float)
    spatial_tv(a,b)  # strict probability validation
    m=.5*(a+b)
    i=a>0;j=b>0
    result=.5*(float((a[i]*np.log(a[i]/m[i])).sum())+
               float((b[j]*np.log(b[j]/m[j])).sum()))
    if result< -1e-12 or result>math.log(2.)+1e-12:
        raise ArithmeticError("invalid Jensen-Shannon information divergence")
    return float(np.clip(result,0,math.log(2.)))


def _mean_pairs(hists:list[np.ndarray], metric)->float:
    if len(hists)<2:
        raise ValueError("independent replicate distributions required")
    return float(np.mean([metric(x,y) for x,y in combinations(hists,2)]))


def multiresolution_source_report(path:str|Path)->dict:
    """Use native source positions and weights; verify original Stage22 hashes."""
    parent=audit_real_source_checkpoints(path)
    original=_Bundle(Path(path))
    try:
        s=original.report("real_core_transport_summary.json")
        diag=s["independent_seed_source_bank_shape_diagnostics"]
        low=np.asarray(diag["spatial_lower_cm"],float)
        high=np.asarray(diag["spatial_upper_cm"],float)
        if (low.shape!=(3,) or high.shape!=(3,) or not
            np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or
            np.any(high<=low)):
            raise ValueError("invalid actual source scoring physical envelope")
        R=float(np.hypot(max(abs(low[0]),abs(high[0])),
                         max(abs(low[1]),abs(high[1]))))
        # Use the exact same full resolution to form every coarser grid.
        # This explicitly conserves source probability and guarantees
        # total variation / information-processing contraction.
        cube_by_batch={b:[] for b in _BATCHES}
        axial_by_batch={b:[] for b in _BATCHES}
        radius_by_batch={b:[] for b in _BATCHES}
        per_seed={seed:[] for seed in _SEEDS}
        mean_empty={n:[] for n in _BINS}
        native_hashes={(r["seed"],r["batch"]):r["source_bank_sha256"]
                       for r in parent["checkpoints"]}
        for seed in _SEEDS:
            for batch in _BATCHES:
                name=f"seed_{seed}/statepoint.{batch:03d}.h5"
                blob=original.read(name)
                if sha256_bytes(blob)!=native_hashes[(seed,batch)]:
                    raise ValueError("changed native OpenMC source bank HDF5")
                with h5py.File(io.BytesIO(blob),"r") as sp:
                    native=sp["source_bank"][()]
                    actual_seed=int(sp["seed"][()])
                    actual_batch=int(sp["current_batch"][()])
                if (actual_seed!=seed or actual_batch!=batch or
                    len(native)!=3000 or
                    native.dtype.names is None or "r" not in native.dtype.names
                    or "wgt" not in native.dtype.names):
                    raise ValueError("real neutron source bank provenance mismatch")
                xyz=np.column_stack([native["r"][c] for c in ("x","y","z")])
                w=np.asarray(native["wgt"],float)
                if (not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(w))
                    or np.any(w<=0) or np.any(xyz<low) or np.any(xyz>high)):
                    raise ValueError("invalid neutron source positions or physical weights")
                hist=np.histogramdd(xyz,bins=[16]*3,
                                    range=list(zip(low,high)),weights=w)[0]
                cube=hist/hist.sum()
                if not np.isclose(cube.sum(),1.,atol=1e-12):
                    raise ArithmeticError("missing weighted source support")
                # The physical cylindrical coordinates are tied to the
                # declared reactor axis, not to cartesian box corners.
                radius=np.hypot(xyz[:,0],xyz[:,1])
                rh=np.histogram(radius,bins=8,range=(0.,R),weights=w)[0]
                zh=np.histogram(xyz[:,2],bins=8,
                                range=(low[2],high[2]),weights=w)[0]
                if not np.isclose(zh.sum(),w.sum()) or not np.isclose(rh.sum(),w.sum()):
                    raise ArithmeticError("radial/axial source projection lost source weight")
                rdist=rh/w.sum()
                zdist=zh/w.sum()
                cube_by_batch[batch].append(cube)
                radius_by_batch[batch].append(rdist)
                axial_by_batch[batch].append(zdist)
                per_seed[seed].append(cube)
                for n in _BINS:
                    smaller=coarsen_source_cube(cube,n)
                    mean_empty[n].append(
                        float(1-np.count_nonzero(smaller)/smaller.size))
        mean_between_tv={}
        mean_between_js={}
        mean_within_tv={}
        fraction_empty={}
        for n in _BINS:
            between=[]; js=[]
            for batch in _BATCHES:
                cubes=[coarsen_source_cube(x,n)
                       for x in cube_by_batch[batch]]
                between.append(_mean_pairs(cubes,spatial_tv))
                js.append(_mean_pairs(cubes,spatial_js))
            mean_between_tv[str(n)]=between
            mean_between_js[str(n)]=js
            fraction_empty[str(n)]=float(np.mean(mean_empty[n]))
            adjacent=[]
            for i in range(len(_BATCHES)-1):
                adjacent.append(float(np.mean([
                    spatial_tv(coarsen_source_cube(per_seed[seed][i],n),
                               coarsen_source_cube(per_seed[seed][i+1],n))
                    for seed in _SEEDS])))
            mean_within_tv[str(n)]=adjacent
        # Mathematical nonincrease under probability-grid coarsening:
        for batch_ix in range(len(_BATCHES)):
            for smaller,bigger in zip(_BINS[:-1],_BINS[1:]):
                if (mean_between_tv[str(smaller)][batch_ix] >
                    mean_between_tv[str(bigger)][batch_ix]+1e-12 or
                    mean_between_js[str(smaller)][batch_ix] >
                    mean_between_js[str(bigger)][batch_ix]+1e-12):
                    raise ArithmeticError("information increased after spatial coarsening")
        axis={"radial_8_bins":[],"axial_8_bins":[]}
        for batch in _BATCHES:
            axis["radial_8_bins"].append(_mean_pairs(
                radius_by_batch[batch],spatial_tv))
            axis["axial_8_bins"].append(_mean_pairs(
                axial_by_batch[batch],spatial_tv))
        return {
            "schema":"nuke-real-source-multires-v1",
            "evidence_kind":"TWENTY_AUTHENTIC_HDF5_NEUTRON_SOURCE_CHECKPOINTS",
            "source_artifact_sha256_original_zip":parent["original_archive_sha256"],
            "real_statepoints":20,
            "four_independent_rng_seeds":list(_SEEDS),
            "batch_checkpoints":list(_BATCHES),
            "source_bank_records_per_checkpoint":3000,
            "source_positions_are_not_independent_genealogies":True,
            "cartesian_resolution_per_axis":list(_BINS),
            "cartesian_between_seed_mean_total_variation_by_batch":mean_between_tv,
            "cartesian_between_seed_mean_js_nats_by_batch":mean_between_js,
            "cartesian_within_seed_mean_adjacent_tv_by_interval":mean_within_tv,
            "cartesian_mean_empty_bin_fraction":fraction_empty,
            "cylindrical_radial_and_axial_projection_between_seed_mean_tv_by_batch":axis,
            "coarsening_probability_conserved_and_tv_js_contraction":True,
            "fission_source_stationarity_physically_qualified":False,
            "historical_hot_nerva_qualified":False,
            "interpretation":[
                "Source differences are strongly resolution-dependent and persist on coarse physical bins.",
                "Higher fine-grid TV partly reflects sparse source-support and cloning; not a calibrated significance test.",
                "Within-seed time checkpoints are genealogically correlated.",
                "No physical NERVA reactor criticality, evaluated hot-core XS or local nuclear heating is validated here.",
            ],
        }
    finally:
        original.close()


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("evidence",type=Path)
    ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args()
    result=multiresolution_source_report(a.evidence)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    for n,series in result["cartesian_between_seed_mean_total_variation_by_batch"].items():
        print(f"source grid {n}^3: between-seed mean TV at batch 120 = {series[-1]:.6f}")
    print("stationary source: NOT QUALIFIED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
