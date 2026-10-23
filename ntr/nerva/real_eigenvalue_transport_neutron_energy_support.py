"""Actual OpenMC eigenvalue neutron incident-energy tally coverage, read-only.

Link two INDEPENDENT HDF5 observations that previous core audits tested
separately: genuine fission-source emission spectrum vs evaluated isotope
energy support (PR #107) and actual *transport* energy-filtered and
unfiltered material neutron tally moments (this audit). They are distinct:
emitted neutron energy does not imply collision/track-length energy coverage.

A residual in the energy-filtered sum is real *unassigned scored tally
weight* but cannot be attributed uniquely to lower/upper energy underflow
without separate wide-range EnergyFilter bins. Never label a near-zero
moment residual as proof of all possible collision energies.
No reactor geometry, temperature, nuclear cross sections, source, solver,
power, k-effective or criticality settings are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .real_core_neutron_energy_closure import _filters, _tally
from .real_fission_source_neutron_energy_support import _digest


AUDITED_SCORES=("flux","absorption","fission","nu-fission")
MATERIAL_IDS=(19,20,21,22)
NATIVE_CLOSURE_ABS=1e-12
NATIVE_CLOSURE_REL=1e-8


def assess_native_material_energy_score_partition(
    native_unfiltered: object,
    native_grouped: object,
    *,
    score_name: str,
)->dict:
    """Evaluate exact 1-to-many tally consistency without erasing residuals."""
    if score_name not in AUDITED_SCORES:
        raise ValueError("unsupported actual neutron collision tally score")
    full=np.asarray(native_unfiltered,dtype=np.float64)
    groups=np.asarray(native_grouped,dtype=np.float64)
    if (full.ndim!=1 or groups.ndim!=2
            or full.shape[0]!=groups.shape[0] or
            full.shape[0]<1 or groups.shape[1]<3
            or not np.isfinite(full).all() or not np.isfinite(groups).all()
            or np.any(full< -1e-12) or np.any(groups< -1e-12)):
        raise ValueError("nonphysical evaluated neutron transport tally moments")
    within=np.sum(groups,axis=1,dtype=np.float64)
    delta=full-within
    if not np.isfinite(delta).all():
        raise ValueError("nonfinite neutron energy tally conservation residual")
    tolerance=NATIVE_CLOSURE_ABS+NATIVE_CLOSURE_REL*np.maximum(
        np.abs(full),np.abs(within)
    )
    closure=np.abs(delta)<=tolerance
    # Report signed residual: signed cancellation is not a measured
    # positive out-of-bounds score or an underflow/overflow attribution.
    return {
        "score":score_name,
        "native_unfiltered_score_per_source":full.tolist(),
        "native_filtered_sum_per_source":within.tolist(),
        "signed_unassigned_energy_window_residual_per_source":delta.tolist(),
        "max_abs_unassigned_score":float(np.max(np.abs(delta))),
        "max_abs_unassigned_fraction_of_unfiltered_score":float(np.max(
            np.abs(delta)/np.maximum(np.abs(full),1e-15)
        )),
        "native_tally_roundoff_compatible_by_material":closure.tolist(),
        "all_material_energy_windows_closed_at_stated_numerical_tolerance":
            bool(np.all(closure)),
        "tail_neutrons_below_or_above_window_individually_attributed":False,
        "observed_energy_window_fraction_is_statistical_source_convergence_test":
            False,
    }


def read_native_real_neutron_energy_tallies(path:Path,
                                           reachable_ids:list[int])->dict:
    path=Path(path)
    with h5py.File(path,"r") as state:
        def txt(x):
            return bytes(x).decode("utf-8").rstrip("\0")
        version=tuple(int(v) for v in state.attrs["openmc_version"])
        if (txt(state.attrs["filetype"])!="statepoint"
                or txt(state["run_mode"][()])!="eigenvalue"
                or version!=(0,15,3)
                or int(state.attrs["photon_transport"])!=0
                or int(state.attrs.get("source_present",0))!=1):
            raise ValueError("not executed genuine OpenMC 0.15.3 eigenvalue neutron HDF5")
        seed=int(state["seed"][()])
        batch=int(state["current_batch"][()])
        inactive=int(state["n_inactive"][()])
        if seed<=0 or not 0<=inactive<batch or batch-inactive<2:
            raise ValueError("invalid actual OpenMC active-batch tally generation count")
        tallies=state["tallies"]
        base,base_labels,base_values=_tally(tallies,"nerva_material_transport",batch-inactive)
        en,en_labels,en_values=_tally(tallies,"nerva_material_energy_transport",batch-inactive)
        base_f=_filters(tallies,base,("material",))[0]
        en_mat_f,en_energy_f=_filters(tallies,en,("material","energy"))
        base_ids=np.asarray(base_f["bins"][()],dtype=int).reshape(-1)
        en_ids=np.asarray(en_mat_f["bins"][()],dtype=int).reshape(-1)
        edges=np.asarray(en_energy_f["bins"][()],dtype=np.float64).reshape(-1)
        groups=len(edges)-1
        if (len(base_ids)!=len(set(base_ids))
                or not np.array_equal(base_ids,en_ids)
                or not set(reachable_ids).issubset(set(base_ids))
                or groups<3 or not np.isfinite(edges).all()
                or not np.all(np.diff(edges)>0)
                or en_values.shape[0]!=len(base_ids)*groups
                or base_values.shape[0]!=len(base_ids)):
            raise ValueError("wrong native energy-filter/material neutron tally mapping")
        source_grid=en_values.reshape(len(base_ids),groups,len(en_labels))
        selected=[int(np.flatnonzero(base_ids==mid)[0]) for mid in reachable_ids]
        scores={}
        for score in AUDITED_SCORES:
            if score not in base_labels or score not in en_labels:
                raise ValueError("missing independent material/energy neutron transport score")
            values=assess_native_material_energy_score_partition(
                base_values[selected,base_labels.index(score)],
                source_grid[selected,:,en_labels.index(score)],
                score_name=score
            )
            scores[score]=values
    return {
        "seed":seed,
        "saved_batch":batch,
        "active_tally_generations":batch-inactive,
        "statepoint_sha256":_digest(path),
        "material_filter_order":base_ids.tolist(),
        "physically_root_reachable_material_ids":reachable_ids,
        "native_energy_filter_edges_eV":edges.tolist(),
        "energy_bin_count":groups,
        "material_incident_energy_score_partitions":scores,
    }


def audit_real_core_neutron_energy_score_support(
    summary_path:Path, cross_sections:Path, source_support_path:Path
)->dict:
    summary_path=Path(summary_path)
    cross_sections=Path(cross_sections)
    source_support_path=Path(source_support_path)
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    prior=json.loads(source_support_path.read_text(encoding="utf-8"))
    if (prior.get("audit")!="GENUINE_OPENMC_FISSION_SOURCE_EVALUATED_NEUTRON_ENERGY_DOMAIN_SUPPORT"
            or prior.get("actual_four_seed_summary_sha256")!=_digest(summary_path)
            or prior.get("actual_evaluated_cross_sections_xml_sha256")!=_digest(cross_sections)
            or prior.get("real_independent_source_seeds")!=4
            or prior.get("genuine_source_hdf5_checkpoint_count")!=20
            or prior.get("physically_reachable_material_ids")!=list(MATERIAL_IDS)
            or not isinstance(prior.get("twenty_verified_source_spectrum_records"),list)
            or len(prior["twenty_verified_source_spectrum_records"])!=20):
        raise ValueError("missing actual evaluated isotope/source energy provenance")
    original={ (x["seed"],x["saved_batch"]):x
              for x in prior["twenty_verified_source_spectrum_records"]}
    if len(original)!=20:
        raise ValueError("duplicate source-energy statepoint identity")
    runs=summary["runs"]
    batches=summary["saved_source_bank_batches"]
    if (len(runs)!=4 or len(batches)!=5
            or batches[-1]!=summary["batches"]
            or any(a>=b for a,b in zip(batches,batches[1:]))):
        raise ValueError("invalid real core energy transport checkpoint schedule")
    low=prior["common_all_material_incident_neutron_energy_min_eV"]
    high=prior["common_all_material_incident_neutron_energy_max_eV"]
    if not 0<=low<high or not np.isfinite([low,high]).all():
        raise ValueError("invalid actual evaluated neutron grid support")
    details=[]
    seen=set()
    for run in runs:
        seed=run["seed"]
        final=Path(run["statepoint_path"])
        if (seed in seen or not final.is_file()
                or _digest(final)!=run["statepoint_sha256"]):
            raise ValueError("wrong actually executed eigenvalue statepoint provenance")
        seen.add(seed)
        for batch in batches:
            path=final.parent/f"statepoint.{batch:0{len(str(summary['batches']))}d}.h5"
            result=read_native_real_neutron_energy_tallies(
                path,list(MATERIAL_IDS)
            )
            key=(seed,batch)
            old=original.get(key)
            if (result["seed"]!=seed or result["saved_batch"]!=batch
                    or old is None
                    or old["statepoint_sha256"]!=result["statepoint_sha256"]):
                raise ValueError("native neutron tally differs from verified emitted source bank")
            edges=result["native_energy_filter_edges_eV"]
            # Native evaluated isotope support must include the *entire*
            # transport scoring window, not merely its endpoints on paper.
            if edges[0]<low-1e-12 or edges[-1]>high+1e-8:
                raise ValueError("actual transport score window exceeds common evaluated isotope grid")
            result["filtered_window_within_genuine_evaluated_isotope_grids"]=True
            result["emitted_source_weight_in_grid_from_independent_audit"]=(
                old["core_all_isotope_support"]["inside_grid_weight_fraction"]
            )
            details.append(result)
    if len(details)!=20 or len(seen)!=4:
        raise ValueError("missing real neutron incident-energy collision score checkpoint")
    diagnostics={
        score:{
            "all_statepoint_material_energy_windows_numerically_closed":all(
                item["material_incident_energy_score_partitions"][score][
                    "all_material_energy_windows_closed_at_stated_numerical_tolerance"]
                for item in details
            ),
            "largest_signed_unassigned_residual_absolute":max(
                item["material_incident_energy_score_partitions"][score][
                    "max_abs_unassigned_score"] for item in details
            ),
            "maximum_absolute_unassigned_fraction":max(
                item["material_incident_energy_score_partitions"][score][
                    "max_abs_unassigned_fraction_of_unfiltered_score"] for item in details
            )
        } for score in AUDITED_SCORES
    }
    return {
        "audit":"GENUINE_OPENMC_EIGENVALUE_MATERIAL_NEUTRON_INCIDENT_ENERGY_SCORE_SUPPORT",
        "real_four_seed_transport_summary_sha256":_digest(summary_path),
        "independently_verified_fission_source_energy_support_sha256":
            _digest(source_support_path),
        "evaluated_neutron_cross_sections_xml_sha256":_digest(cross_sections),
        "native_statepoint_checkpoint_count":len(details),
        "independent_eigenvalue_source_seed_count":len(seen),
        "physically_root_reachable_material_ids":list(MATERIAL_IDS),
        "all_checked_energy_filters_within_evaluated_isotope_support":True,
        "all_score_partitions_per_checkpoint":details,
        "per_score_evaluated_incident_energy_coverage":diagnostics,
        "unfiltered_and_energy_filtered_material_scores_reconciled":True,
        "all_possible_neutron_collision_energies_observed":False,
        "energy_window_underflow_and_overflow_individually_identified":False,
        "hot_nuclear_data_or_bound_scattering_qualified":False,
        "source_bank_stationarity_or_criticality_qualified":False,
        "valid_reactor_local_heating_calculated":False,
        "interpretation":(
            "True neutron eigenvalue statepoint material-tally flux and "
            "reaction score means resolved into evaluated-energy group "
            "tallies for twenty independent HDF5 checkpoint identities. "
            "Any roundoff-scale residual is explicitly reported; large "
            "residuals are unresolved rather than presumed to be a low/high "
            "energy tail. Even exact sampled tally closure does not certify "
            "unsampled neutron collision energies or NERVA hot physics."
        )
    }


def main()->int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary",type=Path,required=True)
    ap.add_argument("--cross-sections",type=Path,required=True)
    ap.add_argument("--source-support",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    result=audit_real_core_neutron_energy_score_support(
        args.summary,args.cross_sections,args.source_support
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Real native neutron transport tally checkpoints:",
          result["native_statepoint_checkpoint_count"])
    for name,item in result["per_score_evaluated_incident_energy_coverage"].items():
        print("Material/energy score window",name,
              "numerically closes:",item["all_statepoint_material_energy_windows_numerically_closed"],
              "maximum nonassignable fraction:",item["maximum_absolute_unassigned_fraction"])
    print("All possible neutron collisions, source stationarity and hot NERVA: UNQUALIFIED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
