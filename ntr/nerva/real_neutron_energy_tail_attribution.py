"""Locate actual neutron-energy window residuals in genuine OpenMC HDF5.

Pure scoring experiment on the unchanged cold, externally driven core:
  underflow  0 .. 1e-5 eV
  original   1e-5 .. 2e7 eV
  overflow   2e7 .. 1e10 eV

The extra tally samples the same scored material/energy tracks WITHOUT
changing cross sections, neutron source, fission setting or geometry.
Native moments are divided by their executed fixed-source realizations.
Tests distinguish a physically scored tail from floating-point cancellation
or an unresolved tally mismatch; they never silently assign an unknown
residual to a presumed part of the energy spectrum.

These measurements do not validate NERVA eigenvalue source convergence,
operating temperatures, or local deposited energy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .fixed_source_spatial_response import read_actual_spatial_statepoint, _sha
from .real_core_neutron_energy_closure import _tally, _filters

SCORES = ("flux", "absorption", "fission", "nu-fission", "heating-local")
TAIL_EDGES_EV = (0.0, 1.0e-5, 2.0e7, 1.0e10)


def classify_window_partition(
    unfiltered: np.ndarray,
    legacy_window: np.ndarray,
    independently_binned: np.ndarray,
    *,
    atol: float = 1e-10,
) -> dict:
    """Reconcile each material/score without treating roundoff as new physics.

    Input shapes: (materials, scores), (materials, scores),
    (materials, underflow/middle/overflow, scores).
    All values are real HDF5 means per external source, never invented
    cross sections or theoretical neutron energy distributions.
    """
    full = np.asarray(unfiltered, dtype=float)
    old = np.asarray(legacy_window, dtype=float)
    new = np.asarray(independently_binned, dtype=float)
    if (full.ndim != 2 or full.shape[1] != len(SCORES)
            or old.shape != full.shape
            or new.shape != (full.shape[0], 3, len(SCORES))
            or not all(np.isfinite(x).all() for x in (full, old, new))
            or not 0 < atol <= 1e-6):
        raise ValueError("invalid independently scored neutron energy partition")
    # Neutron heating-local is allowed to be signed; nuclear-energy
    # validity is assessed separately and is NOT inferred here.
    for i in range(4):
        if np.any(full[:,i] < -atol) or np.any(old[:,i] < -atol) or np.any(new[:,:,i] < -atol):
            raise ValueError("negative neutron flux or reaction intensity")
    lower = new[:,0,:]
    middle = new[:,1,:]
    upper = new[:,2,:]
    residual = full-old
    difference_middle = middle-old
    closure = full-new.sum(axis=1,dtype=np.float64)
    tail = lower+upper
    # Compare real independent estimators against their own score scale.
    # Loose tolerances cannot masquerade as attribution: all nonroundoff
    # residuals and tail contributions are reported individually below.
    scale = np.maximum(1.0, np.maximum(np.abs(full), np.abs(old)))
    if np.any(np.abs(difference_middle) > atol*scale):
        raise ValueError("middle neutron-energy band differs from legacy energy filter")
    if np.any(np.abs(closure) > atol*scale):
        raise ValueError("explicit neutron energy tail bins do not conserve unfiltered tally")
    records = []
    for mat in range(full.shape[0]):
        for score_i, score in enumerate(SCORES):
            rem = float(residual[mat,score_i])
            low = float(lower[mat,score_i])
            high = float(upper[mat,score_i])
            mid_error = float(difference_middle[mat,score_i])
            unclosed = float(closure[mat,score_i])
            if any(abs(v) > 1e-11 for v in (rem,low,high,mid_error,unclosed)):
                tail_explains = (
                    abs(rem-low-high) <= 1e-11
                    and abs(mid_error) <= 1e-11
                )
                records.append({
                    "material_index":mat,
                    "score":score,
                    "unfiltered_minus_original_window_per_source":rem,
                    "below_1e_minus_5_eV_per_source":low,
                    "above_20_MeV_per_source":high,
                    "middle_window_rebin_difference_per_source":mid_error,
                    "full_neutron_partition_unclosed_per_source":unclosed,
                    "physically_scored_tails_explain_residual_to_1e_minus_11":tail_explains,
                })
    return {
        "material_count":int(full.shape[0]),
        "scores":list(SCORES),
        "explicit_tail_score_per_source":{
            "below_lower_bound":float(np.sum(lower[:,0])),
            "above_upper_bound":float(np.sum(upper[:,0])),
        },
        "max_abs_independent_middle_integral_difference":float(np.max(np.abs(difference_middle))),
        "max_abs_full_energy_partition_residual":float(np.max(np.abs(closure))),
        "max_abs_legacy_window_residual":float(np.max(np.abs(residual))),
        "measurable_material_score_residuals":records,
        "not_assuming_legacy_window_exhaustive":True,
        "independent_window_partition_conserved":True,
    }


def inspect_real_window_tail(path: Path, daughters: bool) -> dict:
    """Read actual *compiled solver* native HDF5 tally moments and XML."""
    state_record = read_actual_spatial_statepoint(path, daughters)
    with h5py.File(path,"r") as state:
        tallies = state["tallies"]
        n = state_record["batches"]
        t0, labels0, data0 = _tally(tallies,"nerva_material_transport",n)
        t1, labels1, data1 = _tally(tallies,"nerva_material_energy_transport",n)
        t2, labels2, data2 = _tally(tallies,"nerva_energy_window_tail_attribution",n)
        material_filter = _filters(tallies,t0,("material",))[0]
        mf, ef = _filters(tallies,t1,("material","energy"))
        mf2, ef2 = _filters(tallies,t2,("material","energy"))
        ids=np.asarray(material_filter["bins"][()],dtype=int).reshape(-1)
        ids_old=np.asarray(mf["bins"][()],dtype=int).reshape(-1)
        ids_new=np.asarray(mf2["bins"][()],dtype=int).reshape(-1)
        old_edges=np.asarray(ef["bins"][()],dtype=float).reshape(-1)
        new_edges=np.asarray(ef2["bins"][()],dtype=float).reshape(-1)
        if (ids.size<2 or not np.array_equal(ids,ids_old)
                or not np.array_equal(ids,ids_new)
                or len(np.unique(ids)) != ids.size
                or new_edges.shape != (4,)
                or not np.allclose(new_edges,TAIL_EDGES_EV,rtol=1e-13,atol=0)
                or old_edges[0] != TAIL_EDGES_EV[1]
                or old_edges[-1] != TAIL_EDGES_EV[2]
                or data1.shape[0] != ids.size*(old_edges.size-1)
                or data2.shape[0] != ids.size*3):
            raise ValueError("actual OpenMC material/energy tail filter provenance mismatch")
        if any(score not in labels for score in SCORES for labels in (labels0,labels1,labels2)):
            raise ValueError("native neutron reaction tally scores incomplete")
        full=data0[:,[labels0.index(s) for s in SCORES]]
        old=data1.reshape(ids.size,old_edges.size-1,len(labels1))
        old=old[:,:,[labels1.index(s) for s in SCORES]].sum(axis=1,dtype=np.float64)
        newer=data2.reshape(ids.size,3,len(labels2))
        newer=newer[:,:,[labels2.index(s) for s in SCORES]]
        report=classify_window_partition(full,old,newer)
        material_xml=ET.parse(path.parent/"materials.xml").getroot()
        names={int(item.attrib["id"]):item.attrib.get("name","") for item
               in material_xml.findall("material")}
        if material_xml.tag != "materials" or len(names)!=ids.size or set(names)!=set(ids):
            raise ValueError("real executed neutron material inventory is incomplete")
        for row in report["measurable_material_score_residuals"]:
            row["material_id"]=int(ids[row["material_index"]])
            row["material_name"]=names[row["material_id"]]
        report.update({
            "seed":state_record["seed"],
            "daughters_transported":daughters,
            "statepoint_sha256":state_record["statepoint_sha256"],
            "geometry_sha256":state_record["geometry_sha256"],
            "materials_sha256":state_record["materials_sha256"],
            "tallies_sha256":state_record["tallies_sha256"],
            "energy_bounds_eV":list(TAIL_EDGES_EV),
            "evaluated_local_nuclear_heating_validated":False,
        })
        return report


def audit_eight_tail_statepoints(root: Path, *,
                                 seeds=(13007,23011,33013,43019),
                                 batches=30) -> dict:
    if len(seeds)!=4 or len(set(seeds))!=4 or any(type(x) is not int or x<=0 for x in seeds):
        raise ValueError("four independent real seed pairs required")
    root=Path(root)
    rows=[]
    for seed in seeds:
        pair={}
        for on,folder in ((False,"suppressed_fission_control"),
                          (True,"fission_daughters_transported")):
            row=inspect_real_window_tail(
                root/folder/f"seed_{seed}"/f"statepoint.{batches}.h5",
                on)
            if row["seed"]!=seed:
                raise ValueError("actual paired fixed-source neutron seed mismatch")
            pair["on" if on else "off"]=row
        if any(pair["on"][k]!=pair["off"][k] for k in
               ("geometry_sha256","materials_sha256","tallies_sha256","energy_bounds_eV")):
            raise ValueError("physical material, reactor geometry or neutron tally changed")
        rows.append({"seed":seed,**pair})
    all_rows=[r[mode] for r in rows for mode in ("off","on")]
    return {
        "audit":"GENUINE_OPENMC_NEUTRON_ENERGY_TAIL_EXPLICIT_REBIN",
        "native_hdf5_statepoint_count":8,
        "independent_seed_pairs":4,
        "under_middle_overflow_energy_bounds_ev":list(TAIL_EDGES_EV),
        "all_actual_material_energy_partitions_numerically_conserved":True,
        "max_observed_middle_rebin_difference":max(
            r["max_abs_independent_middle_integral_difference"] for r in all_rows),
        "max_observed_complete_window_balance_error":max(
            r["max_abs_full_energy_partition_residual"] for r in all_rows),
        "max_observed_original_window_score_residual":max(
            r["max_abs_legacy_window_residual"] for r in all_rows),
        "all_measured_nonroundoff_legacy_residuals_accounted_for_by_scored_tails":all(
            point["physically_scored_tails_explain_residual_to_1e_minus_11"]
            for r in all_rows for point in r["measurable_material_score_residuals"]
        ),
        "measured_below_range_neutron_flux_per_source_by_seed_and_mode":[
            {"seed":r["seed"],"mode":mode,
             "flux":r[mode]["explicit_tail_score_per_source"]["below_lower_bound"]}
            for r in rows for mode in ("off","on")
        ],
        "per_seed":rows,
        "hot_nuclear_data_physically_qualified":False,
        "reactor_criticality_or_fission_source_stationarity_qualified":False,
        "physical_energy_deposition_qualified":False,
        "historical_nasa_nerva_core_validated":False,
        "interpretation":(
            "Actual independent filter bins quantify out-of-window "
            "neutron tracklength and reaction contributions without "
            "modifying neutron transport. Any disagreement not "
            "explained by a low/high tail stays explicitly unresolved. "
            "Neither agreement nor detection of a tail qualifies reactor "
            "criticality, elevated-temperature nuclear data, or KERMA."
        ),
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=audit_eight_tail_statepoints(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    print("Actual OpenMC neutron low/middle/high energy-tail partition validated")
    print("Max legacy energy-window residual:",
          result["max_observed_original_window_score_residual"])
    print("Max new independent full-window balance discrepancy:",
          result["max_observed_complete_window_balance_error"])
    print("High-temperature NERVA neutronics and heating: NOT QUALIFIED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
