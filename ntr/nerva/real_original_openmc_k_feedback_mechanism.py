"""Stage44: experimentally observed OpenMC k-feedback source genealogy mechanism.

Upstream pinned OpenMC v0.15.3 C++:
  src/eigenvalue.cpp: inactive => keff = current generation k
                       active => keff = mean of active generation k
  src/physics.cpp:     nu_t = p.wgt() / simulation::keff * weight *
                               neutron_xs.nu_fission / neutron_xs.total
On the original native 20-vs-40-inactive cold demonstrator source run,
the sources and generation k stay identical through batch 22.
At the end of batch 22, 20-inactive uses mean(k21,k22); 40-inactive
uses k22. That difference changes the NEXT generation's expected fission
source production, matching first genuine bank divergence at batch 23.

This is a physical-transport numerical mechanism with independently
observed HDF5 inputs, not a claim of hot NERVA or source stationarity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

MODELS = ("legacy-broad", "central-compact")
BATCHES = (19,20,21,22,23,24,60,100)
TAG = "v0.15.3"
# Git blob object hashes of official openmc-dev/openmc v0.15.3 source files.
UPSTREAM_GIT_BLOB_SHA = {
    "src/eigenvalue.cpp": "a2120a006d5758df3bd831f8b813ad2338e8b9d6",
    "src/physics.cpp": "41509af97be90e8853bd9bb6be0d252312069735",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _cpp_provenance(upstream_dir: Path) -> dict:
    """Validate upstream C++ implementation, not guessed Python physics."""
    snippets = {
        "src/eigenvalue.cpp": (
            "simulation::current_batch > settings::n_inactive",
            "simulation::keff = simulation::k_generation[i];",
            "simulation::keff = simulation::k_sum[0] / n;",
        ),
        "src/physics.cpp": (
            "p.wgt() / simulation::keff * weight *",
            "p.neutron_xs(i_nuclide).nu_fission",
        ),
    }
    result={}
    for name,patterns in snippets.items():
        path=upstream_dir/name
        if not path.is_file():
            raise FileNotFoundError("pinned official OpenMC C++ v0.15.3 source missing: "+str(path))
        data=path.read_bytes()
        git_blob_sha=hashlib.sha1(
            ("blob "+str(len(data))+"\\0").encode()+data
        ).hexdigest()
        if git_blob_sha!=UPSTREAM_GIT_BLOB_SHA[name]:
            raise ValueError("official OpenMC C++ version/source Git blob SHA mismatch: "+name)
        txt=data.decode("utf-8")
        if any(fragment not in txt for fragment in patterns):
            raise ValueError("required OpenMC inactive-k/source-fission coupling absent")
        result[name]={"github_tag":TAG,"git_blob_sha1":git_blob_sha,
                      "content_sha256":_sha256(data),
                      "upstream_neutronics_operations_verified":list(patterns)}
    return result


def _native(path: Path, *, inactive: int, batch: int) -> dict:
    if not path.is_file() or path.stat().st_size<1000:
        raise FileNotFoundError("actual original OpenMC HDF5 source bank missing "+str(path))
    with h5py.File(path,"r") as h:
        if (bytes(h.attrs.get("filetype",b"")).rstrip(b"\0")!=b"statepoint"
                or bytes(h["run_mode"][()]).rstrip(b"\0")!=b"eigenvalue"
                or int(h.attrs.get("source_present",0))!=1
                or int(h.attrs.get("photon_transport",0))!=1
                or int(h["seed"][()])!=13007
                or int(h["current_batch"][()])!=batch
                or int(h["n_inactive"][()])!=inactive
                or int(h["n_batches"][()])!=100
                or int(h["n_particles"][()])!=3000
                or int(h["generations_per_batch"][()])!=1
                or int(h["n_realizations"][()])!=max(0,batch-inactive)):
            raise ValueError("original OpenMC C++ neutron/photon eigenvalue HDF5 identity invalid")
        source=np.asarray(h["source_bank"][()])
        k=np.asarray(h["k_generation"][()],dtype=np.float64)
        ent=np.asarray(h["entropy"][()],dtype=np.float64)
    if (source.shape!=(3000,) or source.dtype.names is None
            or "r" not in source.dtype.names
            or k.shape!=(batch,) or ent.shape!=(batch,)
            or not np.isfinite(k).all() or not np.isfinite(ent).all()
            or np.any(k<=0)):
        raise ValueError("native C++ fission source/k/entropy physical arrays invalid")
    return {"k":k,"entropy":ent,"source":source,
            "source_bank_sha256":_sha256(source.tobytes()),
            "hdf5_sha256":_sha256(path.read_bytes())}


def physical_source_feedback(
    active20: dict[int,dict],inactive40: dict[int,dict]
) -> dict:
    """Link observed native generation-22 histories to next-bank difference.

    This derives the actual relative expected-neutron multiplicity factor;
    it does NOT predict stochastic sampled bank coordinates or physical W.
    """
    if tuple(active20)!=BATCHES or tuple(inactive40)!=BATCHES:
        raise ValueError("all 19–24, 60,100 genuine native checkpoints required")
    for dataset in (active20,inactive40):
        for b in BATCHES[:-1]:
            if not np.array_equal(dataset[b]["k"],dataset[100]["k"][:b]):
                raise ValueError("same original OpenMC generation-k prefix mismatch")
            if not np.array_equal(dataset[b]["entropy"],dataset[100]["entropy"][:b]):
                raise ValueError("same original OpenMC source entropy prefix mismatch")
    for batch in (19,20,21,22):
        a,b=active20[batch],inactive40[batch]
        if (not np.array_equal(a["source"],b["source"])
                or not np.array_equal(a["entropy"],b["entropy"])
                or not np.allclose(a["k"],b["k"],atol=1e-14,rtol=0)):
            raise ValueError("native source history did not agree before first source divergence")
    k21=float(active20[22]["k"][20])
    k22=float(active20[22]["k"][21])
    # For one generation per batch, 20-inactive has TWO active generations
    # at end of batch22; 40-inactive is still running inactive history.
    feedback20=0.5*(k21+k22)
    feedback40=k22
    ratio=feedback40/feedback20
    if not (math.isfinite(ratio) and k21>0 and k22>0
            and not math.isclose(feedback20,feedback40,rel_tol=1e-11)):
        raise ValueError("end-generation keff feedback branch did not diverge")
    same23=np.array_equal(active20[23]["source"],inactive40[23]["source"])
    if same23:
        raise ValueError("no observed original source bank divergence at batch 23")
    delta=np.flatnonzero(active20[23]["source"]!=inactive40[23]["source"])
    if not delta.size:
        raise ValueError("native original source_bank equality contradictory")
    k23a=float(active20[23]["k"][22])
    k23b=float(inactive40[23]["k"][22])
    return {
        "k21_both_native_histories":k21,
        "k22_both_native_histories":k22,
        "k_feedback_for_generation23_after_20_inactive":feedback20,
        "k_feedback_for_generation23_after_40_inactive":feedback40,
        "relative_expected_fission_site_production_factor_active20_over_inactive40":
            ratio,
        "relative_expected_fission_site_production_difference_percent":
            (ratio-1.)*100.,
        "the_two_neutron_sources_match_exactly_through_batch22":True,
        "first_observed_original_source_bank_divergence_batch":23,
        "first_original_fission_bank_site_difference_1_based":int(delta[0]+1),
        "source_banks_at_batch23_identical":same23,
        "native_k23_after_changed_feedback":[k23a,k23b],
        "first_meaningful_k23_difference":abs(k23a-k23b),
        "source_and_entropy_equal_at_60_and_100":[
            bool(np.array_equal(active20[b]["source"],inactive40[b]["source"])
                 and np.array_equal(active20[b]["entropy"],
                                    inactive40[b]["entropy"]))
            for b in (60,100)],
        "this_ratio_does_NOT_predict_individual_fission_site_positions":True,
    }


def audit_original_cpp_feedback(real_source_root: Path|str,
                                upstream_cpp_root: Path|str) -> dict:
    root=Path(real_source_root)
    provenance=_cpp_provenance(Path(upstream_cpp_root))
    cases=[]
    for mode in MODELS:
        actual=[]
        for inactive in (20,40):
            folder=root/f"inactive{inactive}"/mode/"seed_13007"
            p=folder/"transport_provenance.json"
            if not p.is_file():
                raise FileNotFoundError("genuine OpenMC C++ transport provenance required")
            d=json.loads(p.read_text(encoding="utf-8"))
            cfg=d.get("config",{})
            if (cfg.get("seed")!=13007
                    or cfg.get("inactive")!=inactive
                    or cfg.get("batches")!=100
                    or cfg.get("particles")!=3000
                    or cfg.get("photon_transport") is not True
                    or cfg.get("initial_fission_source")!=mode
                    or cfg.get("diagnostic_no_user_tallies") is not True
                    or cfg.get("diagnostic_allow_inactive_statepoints") is not True
                    or cfg.get("statepoint_batches")!=list(
                        range(19,31))+[60,100]
                    or d.get("openmc_version") not in ("0.15.3","v0.15.3")):
                raise ValueError("genuine neutron/photons fixed-core experiment inconsistent")
            for name in ("geometry.xml","materials.xml","settings.xml"):
                if (not (folder/name).is_file() or
                        _sha256((folder/name).read_bytes())!=
                        d.get("xml_sha256",{}).get(name)):
                    raise ValueError("original OpenMC physical input data SHA mismatch")
            if not d.get("nuclear_data",{}).get("cross_sections_sha256"):
                raise ValueError("evaluated neutron/photon data provenance missing")
            history={batch:_native(folder/f"statepoint.{batch:03d}.h5",
                                   inactive=inactive,batch=batch)
                     for batch in BATCHES}
            # Prove actual final original binary is same as provenance.
            if history[100]["hdf5_sha256"] != d["statepoint"]["sha256"]:
                raise ValueError("final original OpenMC HDF5 provenance SHA mismatch")
            actual.append((history,d))
        original,candidate=(x[0] for x in actual)
        cfgA,cfgB=(x[1] for x in actual)
        for name in ("geometry.xml","materials.xml"):
            if cfgA["xml_sha256"][name]!=cfgB["xml_sha256"][name]:
                raise ValueError("physical fuel, geometry, neutron materials changed")
        if cfgA["nuclear_data"]["cross_sections_sha256"]!=cfgB[
                "nuclear_data"]["cross_sections_sha256"]:
            raise ValueError("evaluated nuclear physics file changed")
        result=physical_source_feedback(original,candidate)
        cases.append({"initial_source_mode":mode,"source_feedback":result,
                      "actual_same_evaluated_neutron_data_geometry_materials":True})
    return {
        "schema":"nuke-stage44-original-openmc-v0153-k-feedback-neutron-genealogy-v1",
        "genuine_native_openmc_neutron_photon_hdf5_checked":2*len(MODELS)*len(BATCHES),
        "original_scientific_HDF5_first_source_difference_verified":True,
        "upstream_official_openmc_v0153_CPP_code_sha_verified":provenance,
        "original_source_modes":cases,
        "mechanism_consistent_with_upstream_inverse_k_fission_source_production":True,
        "the_exact_stochastic_source_site_ancestry_Causally_proven":False,
        "hot_nasa_nerva_reactor_geometry_or_thermal_power_qualified":False,
        "notes":[
            "Upstream C++ k branch depends on inactive generation count.",
            "Upstream C++ expected fission-site weight is proportional to inverse k.",
            "The two observed source banks match through generation 22.",
            "The inactive setting changes the feedback k used by generation 23.",
            "The first observed original full source-bank difference is generation 23.",
            "No attempt was made to tune NERVA reactor k, fuel, drums, or geometry.",
        ],
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native-original-early-banks",required=True,type=Path)
    p.add_argument("--upstream-openmc-v0153-cpp",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    r=audit_original_cpp_feedback(a.native_original_early_banks,
                                  a.upstream_openmc_v0153_cpp)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    for case in r["original_source_modes"]:
        f=case["source_feedback"]
        print(case["initial_source_mode"],"k21=",f["k21_both_native_histories"],
              "k22=",f["k22_both_native_histories"],
              "k feedback active/inactive=",f[
                  "k_feedback_for_generation23_after_20_inactive"],f[
                  "k_feedback_for_generation23_after_40_inactive"],
              "relative predicted fission production (%):",f[
                  "relative_expected_fission_site_production_difference_percent"],
              "first actual source-bank change:",f[
                  "first_observed_original_source_bank_divergence_batch"])
    print("Source stationarity, validated hot NERVA and W/cm3: NOT qualified.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
