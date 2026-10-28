"""Stage42: genuine OpenMC source genealogy when user scoring tallies are absent.

This is a CONTROLLED NUMERICAL EXPERIMENT on the same cold, subcritical
NERVA-derived demonstrator (not historical NASA NERVA). Compare neutron
source-bank/entropy/k trajectories with 20 and 40 inactive generations in:
  A) previous Stage41 real coupled neutron/photon + user scoring,
  B) NEW actual OpenMC coupled neutron/photon with no user scoring.
No geometry, materials, evaluated cross section or neutron source changes.

If B still diverges at generation 23, the user tally scoring activation
hypothesis is falsified; other inactive-batch effects must be investigated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .real_coupled_burnin_sensitivity import strict_native_identity
from .real_coupled_initial_source_relaxation import _sha, _native_heating
from .real_coupled_spatial_heating_stability import total_variation
from .source_bank_convergence import read_source_bank

MODES=("legacy-broad","central-compact")
SEED=13007
BATCHES=(60,100)
FISSION_MESH=(4,4,10)


def validate_native_tally_control(baseline: Path,
                                  scoring_disabled: Path,
                                  *, mode: str, inactive: int) -> dict:
    """Original input SHA and actual HDF5 bank, not JSON approximation."""
    if mode not in MODES or inactive not in (20,40):
        raise ValueError("unknown initial source or inactive generation count")
    paths=(baseline,scoring_disabled)
    config=[]
    hashes=[]
    for index,path in enumerate(paths):
        p=path/"transport_provenance.json"
        if not p.is_file():
            raise FileNotFoundError("original coupled OpenMC provenance mandatory: "+str(p))
        report=json.loads(p.read_text(encoding="utf-8"))
        cfg=dict(report.get("config",{}))
        if (cfg.pop("diagnostic_no_user_tallies",False) is not bool(index)
                or cfg.get("seed") != SEED
                or cfg.get("inactive") != inactive
                or cfg.get("batches") != 100
                or cfg.get("particles") != 3000
                or cfg.get("photon_transport") is not True
                or cfg.get("initial_fission_source") != mode
                or report.get("openmc_version") not in ("0.15.3","v0.15.3")
                or cfg.get("statepoint_batches") not in ([60,100],[30,60,100])):
            raise ValueError("real native user-tally controlled configuration incorrect")
        # Archived Stage41 has original scoring enabled; Stage42 source-only
        # has exactly same geometry/material evaluated data and same start.
        cfg.pop("statepoint_batches")
        for name in ("geometry.xml","materials.xml","settings.xml"):
            filename=path/name
            if (not filename.is_file()
                    or _sha(filename) != report.get("xml_sha256",{}).get(name)):
                raise ValueError("physical source-controlled OpenMC XML SHA changed")
        physical=tuple(report["xml_sha256"][name] for name in ("geometry.xml","materials.xml"))
        cross=report.get("nuclear_data",{}).get("cross_sections_sha256")
        if not cross:
            raise ValueError("original evaluated neutron/photon index not verified")
        hashes.append((physical,cross,report["xml_sha256"]["settings.xml"]))
        if index==0:
            tally_path=path/"tallies.xml"
            if (not tally_path.is_file()
                    or _sha(tally_path)!=report["xml_sha256"].get("tallies.xml")):
                raise ValueError("source control does not have original user heating tally")
        else:
            if (path/"tallies.xml").is_file():
                # Model with zero scoring tallies should never contain user
                # score definitions: this is checked below by native StatePoint.
                xml=(path/"tallies.xml").read_text(encoding="utf-8")
                if "<tally " in xml or "<tally>" in xml:
                    raise ValueError("source-only control included a scoring tally")
        config.append(cfg)
    if (config[0]!=config[1] or hashes[0]!=hashes[1]):
        raise ValueError("real source-only experiment altered settings, geometry or data")
    # Histogram must use the SAME physically defined native source-mesh\n    # bounds as prior validated Stage37–41 source convergence tests.\n    original_heating = _native_heating(baseline / 'statepoint.100.h5')\n    actual=[]
    for index,path in enumerate(paths):
        hists=[]
        for batch in BATCHES:
            native=path/f"statepoint.{batch:03d}.h5"
            if not native.is_file() or not native.stat().st_size:
                raise FileNotFoundError("native eigenvalue StatePoint absent: "+str(native))
            snap=strict_native_identity(native,seed=SEED,inactive=inactive,batch=batch)
            with h5py.File(native,"r") as h:
                # A source-only HDF5 must not have user tally scores.
                # This is stronger than checking Python input XML only.
                if index==1 and "tallies" in h:
                    actual_tallies=h["tallies"]
                    if "ids" in actual_tallies and np.asarray(
                            actual_tallies["ids"][()]).size:
                        raise ValueError("source-only native C++ HDF5 stored scoring tally IDs")
            bank,metadata=read_source_bank(
                native,lower_cm=original_heating['mesh_lower_left_cm'],
                upper_cm=original_heating['mesh_upper_right_cm'],
                bins=FISSION_MESH)
            # Same fixed histogram physical extents across modes.
            if (metadata["seed"]!=SEED
                    or metadata["completed_batch"]!=batch
                    or metadata["statepoint_sha256"]!=_sha(native)):
                raise ValueError("native C++ OpenMC RNG/source bank SHA disagrees")
            hists.append((snap,bank))
        for v0,v1 in zip(hists,hists[1:]):
            if (not np.array_equal(v0[0]["k"],v1[0]["k"][:60])
                    or not np.array_equal(v0[0]["entropy"],v1[0]["entropy"][:60])):
                raise ValueError("original native C++ within-run source prefix invalid")
        actual.append(hists)
    baseline_k=actual[0][-1][0]["k"]
    candidate_k=actual[1][-1][0]["k"]
    difference=np.abs(baseline_k-candidate_k)
    substantial=np.flatnonzero(difference>1e-8)
    a=actual[0][-1][0]["source_bank"]
    b=actual[1][-1][0]["source_bank"]
    return {
        "mode":mode,
        "inactive":inactive,
        "identical_physical_geometry_material_nuclear_data_source_and_settings_proven":True,
        "native_source_only_hdf5_has_no_user_scoring_tallies":True,
        "first_k_difference_larger_than_1e_8_baseline_vs_no_tallies":
            int(substantial[0]+1) if substantial.size else None,
        "max_generation_k_difference_baseline_vs_no_tallies":
            float(difference.max()),
        "entire_source_bank_exact_at_60_and_100":[
            bool(np.array_equal(actual[0][j][0]["source_bank"],
                                actual[1][j][0]["source_bank"])) for j in range(2)],
        "original_source_bank_histogram_TV_60_and_100":[
            float(total_variation(actual[0][j][1],actual[1][j][1]))
            for j in range(2)],
        "source_entropy_history_exact":
            bool(np.array_equal(actual[0][-1][0]["entropy"],
                                actual[1][-1][0]["entropy"])),
        "original_neutron_photon_transport_changed_to_neutron_only":False,
        "source_only_candidate_physical_heating_scored":False,
    }


def native_inactive_sensitivity(first: Path, second: Path,
                                *, mode: str) -> dict:
    """Compare source-only 20 versus 40 inactive, same source, seed and data."""
    if mode not in MODES:
        raise ValueError("invalid original fission source mode")
    data=[]
    for path,inactive in ((first,20),(second,40)):
        p=path/"transport_provenance.json"
        if not p.is_file():
            raise FileNotFoundError("source-only original OpenMC provenance absent")
        cfg=json.loads(p.read_text(encoding="utf-8")).get("config",{})
        if (cfg.get("seed")!=SEED or cfg.get("initial_fission_source")!=mode
                or cfg.get("diagnostic_no_user_tallies") is not True
                or cfg.get("inactive")!=inactive):
            raise ValueError("uncontrolled native source-only inactive-setting experiment")
        hist=[]
        for batch in BATCHES:
            f=path/f"statepoint.{batch:03d}.h5"
            if not f.is_file():
                raise FileNotFoundError("native source-only transport StatePoint missing")
            identity=strict_native_identity(f,seed=SEED,inactive=inactive,batch=batch)
            with h5py.File(f,"r") as h:
                if "tallies" in h and "ids" in h["tallies"] and len(h["tallies/ids"])!=0:
                    raise ValueError("real source-only native scoring tallies not disabled")
            hist.append(identity)
        data.append(hist)
    k=np.abs(data[0][-1]["k"]-data[1][-1]["k"])
    meaningful=np.flatnonzero(k>1e-8)
    return {
        "first_meaningful_k_generation_difference_in_source_only_runs":
            int(meaningful[0]+1) if meaningful.size else None,
        "max_abs_generation_k_difference":float(k.max()),
        "native_exact_full_3000_site_banks_match_at_60_and_100":[
            bool(np.array_equal(data[0][i]["source_bank"],
                                data[1][i]["source_bank"]))
            for i in range(2)],
        "source_entropy_full_100_generations_exact":
            bool(np.array_equal(data[0][-1]["entropy"],
                                data[1][-1]["entropy"])),
        "user_scoring_tallies_removed_in_both_original_runs":True,
    }


def audit_native_source_tally_activation(base: Path | str,
                                         no_tally: Path | str) -> dict:
    old=Path(base)
    root=Path(no_tally)
    per_mode=[]
    for mode in MODES:
        diagnostics=[]
        for inactive in (20,40):
            old_folder=old/("original20" if inactive==20 else "changed40")/mode/f"seed_{SEED}"
            new_folder=root/("inactive20" if inactive==20 else "inactive40")/mode/f"seed_{SEED}"
            diagnostics.append(validate_native_tally_control(
                old_folder,new_folder,mode=mode,inactive=inactive))
        source_only=native_inactive_sensitivity(
            root/"inactive20"/mode/f"seed_{SEED}",
            root/"inactive40"/mode/f"seed_{SEED}",
            mode=mode)
        per_mode.append({
            "initial_fission_source":mode,
            "scored_vs_no_user_tallies_at_fixed_inactive":diagnostics,
            "20_vs_40_inactive_without_any_user_tallies":source_only,
        })
    return {
        "schema":"nuke-stage42-genuine-openmc-user-tally-activation-source-control-v1",
        "genuine_actual_openmc_hdf5_checkpoint_files_inspected":16,
        "original_scored_OpenMC_Cplusplus_source_runs":4,
        "new_source_only_OpenMC_Cplusplus_source_runs":4,
        "photon_transport_retained_in_all_eight_original_executions":True,
        "all_four_original_source_only_runs_have_no_user_scoring_tallies":True,
        "per_source_initialization":per_mode,
        "user_tally_activation_CAUSALLY_changes_neutron_genealogy_PROVEN":False,
        "source_stationarity_statistically_qualified":False,
        "hot_historical_NERVA_or_physical_watts_qualified":False,
        "interpretation":[
            "This tests whether removal of user scoring tallies eliminates the inactive-boundary source divergence.",
            "An original neutron source-only transport StatePoint does NOT measure reactor heating.",
            "No criticality, enrichment, physical core geometry, evaluated nuclear data or temperature is altered.",
            "The original source remains a far-subcritical conceptual NERVA-derived surrogate.",
        ],
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage41-scored-originals",type=Path,required=True)
    p.add_argument("--new-no-user-tally-runs",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    d=audit_native_source_tally_activation(a.stage41_scored_originals,
                                          a.new_no_user_tally_runs)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(d,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for row in d["per_source_initialization"]:
        print(row["initial_fission_source"],
              "source-only 20→40 inactive:",
              row["20_vs_40_inactive_without_any_user_tallies"])
    print("Original native HDF5 inspected:",d[
        "genuine_actual_openmc_hdf5_checkpoint_files_inspected"])
    print("NO source stationarity, hot NERVA or reactor watts certification")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
