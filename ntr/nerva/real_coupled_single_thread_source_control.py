"""Stage41: original OpenMC C++ one-thread inactive-batch sensitivity control.

Six REAL fixed-reactor neutron+photon eigenvalue histories:
  broad/compact initial sources x (20-inactive original, 20-inactive exact
  replay, 40-inactive changed tally schedule), same RNG seed 13007.
All are 100 batches, 3000 source particles/generation, 1 OpenMP thread.

This diagnoses whether the generation-23 source-path divergence reported
at 2 threads is present with a single thread. All inferences are from
actual OpenMC C++ HDF5 source banks and disjoint native energy tallies.
No source-statistical stationarity, reactor design optimization or watts.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .real_coupled_three_arm_source_control import (
    compare_native_three_arm_paths,
    compare_original_late_heating,
)
from .real_coupled_three_arm_source_control import _original_native_pair
from .real_coupled_initial_source_relaxation import _sha

MODES=("legacy-broad","central-compact")
ARMS=(("original20",20),("repeat20",20),("changed40",40))
SEED=13007
CHECKPOINTS=(60,100)


def verify_three_native_original_inputs(paths: list[Path], mode: str) -> dict:
    """Require identical physical reactor/data; A/B also identical settings.

    Compare original file hashes, not config claims. C may differ only in
    inactive-batch setting; must have same checkpoint schedule.
    """
    if len(paths)!=3 or mode not in MODES:
        raise ValueError("expected three real source arms for a supported mode")
    config_list=[]
    sha_list=[]
    executable_list=[]
    for (arm,inactive),folder in zip(ARMS,paths):
        provenance=folder/"transport_provenance.json"
        if not provenance.is_file():
            raise FileNotFoundError("native OpenMC transport provenance missing: "+str(provenance))
        raw=json.loads(provenance.read_text(encoding="utf-8"))
        cfg=dict(raw.get("config",{}))
        if (cfg.get("seed")!=SEED or cfg.get("initial_fission_source")!=mode
                or cfg.get("particles")!=3000 or cfg.get("batches")!=100
                or cfg.get("inactive")!=inactive
                or cfg.get("statepoint_batches")!=[60,100]
                or cfg.get("photon_transport") is not True
                or raw.get("openmc_version") not in ("0.15.3","v0.15.3")):
            raise ValueError("native source, photons, batches or evaluated transport identity changed")
        xml={}
        for name in ("geometry.xml","materials.xml","tallies.xml","settings.xml"):
            path=folder/name
            if (not path.is_file() or not raw.get("xml_sha256",{}).get(name)
                    or raw["xml_sha256"][name] != _sha(path)):
                raise ValueError("native physical OpenMC XML content SHA verification failed")
            xml[name]=raw["xml_sha256"][name]
        data=raw.get("nuclear_data",{}).get("cross_sections_sha256")
        if not data:
            raise ValueError("original neutron/photon evaluated XS index missing")
        config_list.append(cfg)
        sha_list.append((xml,data))
        executable_list.append(str(raw.get("openmc_version","")))
    base=dict(config_list[0])
    for j,cfg in enumerate(config_list):
        stripped=dict(cfg)
        removed=stripped.pop("inactive")
        if removed != ARMS[j][1] or stripped!=dict(base,**{}):
            # Compare without inactive against the 20-inactive reference.
            reference=dict(base)
            reference.pop("inactive")
            if stripped!=reference:
                raise ValueError("original nuclear transport config differs beyond tally inactive")
    physical_keys=("geometry.xml","materials.xml","tallies.xml")
    if any(sha_list[0][0][name]!=ref[0][name]
           for ref in sha_list[1:] for name in physical_keys):
        raise ValueError("reactor geometry/material/tally XML modified across controls")
    if any(ref[1]!=sha_list[0][1] for ref in sha_list[1:]):
        raise ValueError("evaluated neutron/photon nuclear data differs across controls")
    if (sha_list[0][0]["settings.xml"]!=sha_list[1][0]["settings.xml"]):
        raise ValueError("unchanged-run settings XML not equal")
    # Settings in changed40 must differ, otherwise the test didn't change tally burn-in.
    if sha_list[0][0]["settings.xml"]==sha_list[2][0]["settings.xml"]:
        raise ValueError("changed-inactive source run did not change settings.xml")
    if len(set(executable_list))!=1:
        raise ValueError("original C++ OpenMC transport version changed")
    return {
        "physical_geometry_materials_tally_XML_identical":True,
        "evaluated_nuclear_data_index_SHA_identical":True,
        "unchanged_20_and_repeat_20_settings_SHA_exact":True,
        "different_40_inactive_settings_SHA_confirmed":True,
        "OpenMC_engine_version":executable_list[0],
        "num_threads_requested_by_GitHub_Actions_OMP_NUM_THREADS":1,
        "original_Cplusplus_runtime_thread_count_directly_observed":False,
    }


def audit_native_one_thread_experiment(root: Path | str) -> dict:
    folder=Path(root)
    runs=[]
    for mode in MODES:
        dirs=[folder/arm/mode/f"seed_{SEED}" for arm,_ in ARMS]
        provenance=verify_three_native_original_inputs(dirs,mode)
        fields=[]
        for directory,(_,inactive) in zip(dirs,ARMS):
            fields.append(_original_native_pair(directory,SEED,inactive))
        genealogies=compare_native_three_arm_paths(
            fields[0][0],fields[1][0],fields[2][0],[x[2] for x in fields])
        heating=compare_original_late_heating([x[1] for x in fields])
        first_shift=genealogies["per_native_checkpoint"][-1][
            "changed20_vs_40_first_material_k_difference_gt_1e_8"]
        original_k=fields[0][0][-1]["k"]
        repeat_k=fields[1][0][-1]["k"]
        changed_k=fields[2][0][-1]["k"]
        # Mean k comparisons are descriptive and do NOT purport to be
        # uncertainty-calibrated correlated-generation eigenvalue results.
        runs.append({
            "initial_source_mode":mode,
            "random_seed_label":SEED,
            "same_actual_reactor_evaluated_data_and_engine_provenance":provenance,
            "native_source_trajectory":genealogies,
            "native_photon_neutron_disjoint_heating":heating,
            "first_meaningful_generation_k_difference":first_shift,
            "k_generation_21_to_60_mean_in_each_execution":[
                float(np.mean(x[20:60])) for x in
                (original_k,repeat_k,changed_k)],
            "k_generation_61_to_100_mean_in_each_execution":[
                float(np.mean(x[60:100])) for x in
                (original_k,repeat_k,changed_k)],
        })
    same_exact=all(c["native_source_trajectory"][
        "unchanged_replay_all_fission_bank_bytes_and_entropy_exact"] for c in runs)
    change_observed=all(c["native_source_trajectory"][
        "changed_inactive_bank_difference_observed"] for c in runs)
    return {
        "schema":"nuke-stage41-genuine-openmc-single-thread-inactive-source-control-v1",
        "native_OpenMC_neutron_photon_histories_executed":6,
        "genuine_original_hdf5_checkpoints_60_100":12,
        "single_OMP_thread_requested_on_all_original_Cplusplus_runs":True,
        "identical_transport_input_and_nuclear_data_fingerprints_verified":True,
        "all_unchanged20_replay_source_banks_and_entropy_bit_exact":bool(same_exact),
        "all_changed40_source_banks_differ_from_20":bool(change_observed),
        "mean_final_source_bank_TV_unchanged_20_vs_20":float(np.mean([
            r["native_source_trajectory"]["per_native_checkpoint"][-1][
                "unchanged20_vs_20_spatial_fission_bank_TV"] for r in runs])),
        "mean_final_source_bank_TV_changed_20_vs_40":float(np.mean([
            r["native_source_trajectory"]["per_native_checkpoint"][-1][
                "changed20_vs_40_spatial_fission_bank_TV"] for r in runs])),
        "mean_last_window_3D_heat_TV_unchanged20_vs_20":float(np.mean([
            r["native_photon_neutron_disjoint_heating"][
                "unchanged20_vs_20_final_heating_shape_TV"] for r in runs])),
        "mean_last_window_3D_heat_TV_changed20_vs_40":float(np.mean([
            r["native_photon_neutron_disjoint_heating"][
                "changed20_vs_40_final_heating_shape_TV"] for r in runs])),
        "source_and_heat_by_initial_source":runs,
        "one_thread_source_history_stationarity_qualified":False,
        "two_thread_to_single_thread_causal_RNG_mechanism_isolated":False,
        "real_hot_nasa_nerva_core_or_watts_qualified":False,
        "limitations":[
            "Single OpenMP thread requested via CI; actual C++ OpenMC runtime thread count not independently measured.",
            "Original six executions are NOT six independent neutron genealogies.",
            "Changing tally burn-in has not changed reactor materials, geometry or nuclear data.",
            "No reactor power or watt normalization: energy is eV per MC starting neutron.",
            "Two initial source modes and one RNG seed are insufficient for calibrated uncertainty.",
        ],
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    r=audit_native_one_thread_experiment(args.root)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Genuine original OpenMC C++ source histories:",r[
        "native_OpenMC_neutron_photon_histories_executed"])
    for mode in r["source_and_heat_by_initial_source"]:
        print(mode["initial_source_mode"],
              "unchanged exact:",mode["native_source_trajectory"][
                  "unchanged_replay_all_fission_bank_bytes_and_entropy_exact"],
              "changed first substantial k generation:",mode[
                  "first_meaningful_generation_k_difference"])
    print("Final source-bank TV baseline/revised:",
          r["mean_final_source_bank_TV_unchanged_20_vs_20"],
          r["mean_final_source_bank_TV_changed_20_vs_40"])
    print("Last active window native 3D heat TV baseline/revised:",
          r["mean_last_window_3D_heat_TV_unchanged20_vs_20"],
          r["mean_last_window_3D_heat_TV_changed20_vs_40"])
    print("Hot NERVA criticality, fission-source stationarity and watts: NOT qualified")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
