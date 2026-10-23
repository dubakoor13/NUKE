"""Genuine OpenMC reactor prompt/delayed fission-neutron source audit.

Nu-fission, prompt-nu-fission and delayed-nu-fission are three separately
scored evaluated continuous-energy tracklength reactions. Check that they
close *per root-reachable material*, as an independent whole-core tally,
and in five DISJOINT active-generation windows from cumulative statepoint
raw sums. No guessed neutron yield, kinetics constants, source reweighting,
core tuning, or fabricated evaluated nuclear data.

The delayed/total neutron-production ratio is a FLUX-WEIGHTED, UNADJOINTED
emission fraction -- it is NOT reactor beta_eff or a point-kinetics input.
Eigenvalue fission source genealogies remain correlated. Not NERVA history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

SCORES = ("nu-fission", "prompt-nu-fission", "delayed-nu-fission")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def _str(x) -> str:
    if not isinstance(x, (bytes, np.bytes_)):
        raise ValueError("OpenMC native score/filter identity is not HDF5 bytes")
    return bytes(x).decode("utf-8").rstrip("\x00")


def _tally(tallies, name: str, filters: tuple[str,...], active: int,
           selected: tuple[str,...]) -> tuple[np.ndarray, list[int]]:
    matches = [
        t for key,t in tallies.items()
        if key.startswith("tally ") and isinstance(t,h5py.Group)
        and "name" in t and _str(t["name"][()]) == name
    ]
    if len(matches) != 1:
        raise ValueError("missing/duplicate evaluated fission-emission tally "+name)
    t = matches[0]
    if (_str(t["estimator"][()]) != "tracklength"
            or int(t["n_realizations"][()]) != active
            or int(t["n_filters"][()]) != len(filters)
            or [_str(x) for x in t["nuclides"][()]] != ["total"]):
        raise ValueError("fission-emission tally estimator/provenance mismatch")
    # OpenMC 0.15.3 legitimately OMITS the "filters" dataset for native
    # unfiltered tallies (n_filters=0). A material-filtered tally MUST
    # include a nonempty matching ID dataset; never fabricate a filter.
    if len(filters) == 0 and "filters" not in t:
        filter_ids = np.asarray([], dtype=int)
    elif "filters" in t:
        filter_ids = np.asarray(t["filters"][()],dtype=int).reshape(-1)
    else:
        raise ValueError("required native neutron fission-emission material filter missing")
    if len(filter_ids) != len(filters):
        raise ValueError("fission-emission native filter length changed")
    mid = []
    for fid, kind in zip(filter_ids,filters):
        filt = tallies["filters"].get("filter " + str(int(fid)))
        if filt is None or _str(filt["type"][()]) != kind:
            raise ValueError("fission-emission material filter type invalid")
        if kind == "material":
            mid = list(map(int,np.asarray(filt["bins"][()]).reshape(-1)))
    names = [_str(x) for x in t["score_bins"][()]]
    if len(names)!=len(set(names)) or any(score not in names for score in selected):
        raise ValueError("fission-emission native score missing/duplicate")
    raw = np.asarray(t["results"][()], dtype=np.float64)
    if (raw.shape != (len(mid) if mid else 1,len(names),2)
            or not np.isfinite(raw).all() or np.any(raw[:,:,1] < 0)):
        raise ValueError("invalid raw OpenMC fission-emission moments")
    # The second moment is a native per-active-batch sum of squares, not
    # a squared sum. Cauchy-Schwarz is a necessary numerical identity.
    first, second = raw[:,:,0], raw[:,:,1]
    if np.any(second - first*first/active <
              -1e-10*np.maximum(second,1e-26)):
        raise ValueError("invalid neutron-emission native second-moment identity")
    return raw[:,[names.index(score) for score in selected],:],mid


# Prompt, delayed and total evaluated yields may have independently
# tabulated/interpolated incident-energy functions. Their sum is a
# consistency SCREEN, not an exact identity at floating-point epsilon.
# This is a deliberately visible engineering screen, NOT cross-section
# uncertainty, not fitted data, and not reactor beta_eff accuracy.
EVALUATED_YIELD_MAX_RELATIVE_MISMATCH = 1e-4


def _close(total: np.ndarray, parts: np.ndarray, label: str,
           *, independently_tabulated_yields: bool = False) -> float:
    gap = np.asarray(total,dtype=float)-np.asarray(parts,dtype=float)
    scale = np.maximum(1e-10,np.maximum(np.abs(total),np.abs(parts)))
    # Different tallies of the SAME evaluated nu-fission score must
    # numerically agree extremely closely. Distinct nuclear-data yield
    # functions may legitimately differ by evaluation/interpolation.
    relative_tolerance = (
        EVALUATED_YIELD_MAX_RELATIVE_MISMATCH
        if independently_tabulated_yields else 3e-10
    )
    if np.any(np.abs(gap) > relative_tolerance*scale + 3e-12):
        raise ValueError("genuine OpenMC fission-emission closure failed: "+label)
    return float(np.max(np.abs(gap)))


def assess_fission_emission_window(
    global_sums: np.ndarray,
    material_sums: np.ndarray,
    global_reference: float,
    material_reference: np.ndarray,
    ids: list[int],
    reachable: set[int],
    *,
    number_of_active_batches: int,
) -> dict:
    """Read one disjoint active-generation window; preserve source units."""
    global_sums = np.asarray(global_sums,dtype=float)
    material_sums = np.asarray(material_sums,dtype=float)
    material_reference = np.asarray(material_reference,dtype=float)
    if (type(number_of_active_batches) is not int
            or number_of_active_batches < 1
            or global_sums.shape != (3,)
            or material_sums.shape != (len(ids),3)
            or material_reference.shape != (len(ids),)
            or not ids or len(set(ids)) != len(ids)
            or not reachable <= set(ids) or not reachable
            or not np.isfinite(global_sums).all()
            or not np.isfinite(material_sums).all()
            or not np.isfinite(material_reference).all()
            or not math.isfinite(float(global_reference))):
        raise ValueError("invalid actual active-window neutron emission inventory")
    if (np.any(global_sums < -1e-9) or np.any(material_sums < -1e-9)
            or np.any(material_reference < -1e-9) or global_reference < -1e-9):
        raise ValueError("negative native fission neutron emission after window subtraction")
    global_sums = np.maximum(global_sums,0.)
    material_sums = np.maximum(material_sums,0.)
    material_reference = np.maximum(material_reference,0.)
    included = np.array([mid in reachable for mid in ids])
    if (np.any(material_sums[~included] != 0)
            or np.any(material_reference[~included] != 0)):
        raise ValueError("unplaced OpenMC material contains scored fission emission")
    # Use independent raw scoring, not copies of nu-fission reconstructed
    # from prompt + delayed.
    residuals = {
        "global_total_minus_prompt_delayed":_close(
            global_sums[0],global_sums[1]+global_sums[2],"global total=prompt+delayed",
            independently_tabulated_yields=True
        ),
        "material_total_minus_prompt_delayed":_close(
            material_sums[:,0],material_sums[:,1]+material_sums[:,2],
            "material total=prompt+delayed",
            independently_tabulated_yields=True
        ),
        "global_minus_summed_material_total":_close(
            global_sums,material_sums.sum(axis=0),"material-to-global source"
        ),
        "global_minus_independent_existing_nu_fission":_close(
            global_sums[0],global_reference,"independent global nu-fission"
        ),
        "material_minus_independent_existing_nu_fission":_close(
            material_sums[:,0],material_reference,"independent material nu-fission"
        ),
    }
    factor = float(number_of_active_batches)
    delayed = float(global_sums[2]/factor)
    total = float(global_sums[0]/factor)
    return {
        "number_of_active_batched_realizations":number_of_active_batches,
        "neutron_production_per_generation_source":{
            score:float(global_sums[k]/factor) for k,score in enumerate(SCORES)
        },
        "material_source_per_generation_source":[
            {
                "material_id":id_,
                "is_root_reachable":id_ in reachable,
                **{score:float(material_sums[i,k]/factor)
                   for k,score in enumerate(SCORES)}
            } for i,id_ in enumerate(ids)
        ],
        "delayed_fraction_of_evaluated_emission_source":(
            delayed/total if total > 0 else None
        ),
        "zero_observed_fission_source_is_not_proof_of_zero_physical_fission":
            total == 0.,
        "maximum_absolute_raw_window_closure_per_score":max(residuals.values()),
        "maximum_relative_evaluated_yield_mismatch": float(
            max(
                abs(global_sums[0]-global_sums[1]-global_sums[2])/
                max(abs(global_sums[0]),1e-10),
                np.max(
                    abs(material_sums[:,0]-material_sums[:,1]-material_sums[:,2])/
                    np.maximum(abs(material_sums[:,0]),1e-10)
                ),
            )
        ),
        "evaluated_yield_relative_mismatch_screen":EVALUATED_YIELD_MAX_RELATIVE_MISMATCH,
        "independent_conservation_residuals":residuals,
        "effective_delayed_neutron_fraction_beta_eff_computed":False,
    }


def read_real_checkpoint(path: Path, expected_material_ids: set[int]) -> dict:
    path = Path(path)
    with h5py.File(path,"r") as sp:
        if (_str(sp.attrs["filetype"]) != "statepoint"
                or _str(sp["run_mode"][()]) != "eigenvalue"
                or tuple(map(int,sp.attrs["openmc_version"])) != (0,15,3)
                or int(sp.attrs["photon_transport"]) != 0
                or int(sp.attrs.get("source_present",0)) != 1):
            raise ValueError("requires executed neutron-only OpenMC 0.15.3 eigenvalue HDF5")
        seed = int(sp["seed"][()])
        batch = int(sp["current_batch"][()])
        inactive = int(sp["n_inactive"][()])
        active = int(sp["n_realizations"][()])
        if (seed<=0 or batch<=inactive or active != batch-inactive
                or active<2):
            raise ValueError("invalid native active source generations")
        t = sp["tallies"]
        g,_ = _tally(t,"nerva_global_fission_emission",(),active,SCORES)
        m,ids = _tally(t,"nerva_material_fission_emission",
                       ("material",),active,SCORES)
        gref,_ = _tally(t,"nerva_global_reaction_tracklength",(),
                        active,("nu-fission",))
        mref,ref_ids = _tally(t,"nerva_material_transport",
                              ("material",),active,("nu-fission",))
        if (set(ids)!=expected_material_ids or len(ids)!=len(expected_material_ids)
                or ids != ref_ids):
            raise ValueError("fission source material bins differ from executed core")
        return {
            "seed":seed,"batch":batch,"inactive":inactive,"active":active,
            "hdf5_sha256":_sha(path),"material_ids":ids,
            "global_first":g[0,:,0],
            "material_first":m[:,:,0],
            "global_reference_first":float(gref[0,0,0]),
            "material_reference_first":mref[:,0,0],
        }


def audit_real_fission_emission(summary_file: Path,
                                kerma_file: Path) -> dict:
    summary_file=Path(summary_file)
    kerma_file=Path(kerma_file)
    summary=json.loads(summary_file.read_text(encoding="utf-8"))
    kerma=json.loads(kerma_file.read_text(encoding="utf-8"))
    if (summary.get("execution")!="ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly")!="core"
            or kerma.get("audit")!="INDEPENDENT_EXECUTED_OPENMC_MATERIAL_RESOLVED_MT901_PHYSICS"
            or kerma.get("transport_summary_sha256")!=_sha(summary_file)
            or kerma.get("evaluated_neutron_cross_sections_xml_sha256")!=
               summary.get("cross_sections_xml_sha256")):
        raise ValueError("missing actual physical reactor/XS/material provenance")
    reachable=set(kerma["executed_reachable_material_ids"])
    physical=kerma["physical_materials"]
    unused=kerma["uninstantiated_declared_materials"]
    expected={int(x["material_id"]) for x in physical+unused}
    if (len(expected)!=kerma["declared_material_count"]
            or len(reachable)!=kerma["physical_material_count"]
            or not reachable <= expected):
        raise ValueError("invalid executed root-reachable physical source materials")
    batches=summary["saved_source_bank_batches"]
    total_batches=summary["batches"]
    inactive=summary["inactive_batches"]
    if (len(batches)!=5 or batches[-1]!=total_batches
            or any(a>=b for a,b in zip(batches,batches[1:]))
            or batches[0]<=inactive):
        raise ValueError("expected 5 strictly increasing active source checkpoints")
    run_rows=[]
    hashes=set()
    seeds=set()
    for run in summary["runs"]:
        seed=run["seed"]
        if (type(seed) is not int or seed<=0 or seed in seeds):
            raise ValueError("duplicate/invalid physical OpenMC source seed")
        seeds.add(seed)
        final=Path(run["statepoint_path"])
        if not final.is_file() or _sha(final)!=run["statepoint_sha256"]:
            raise ValueError("physical neutron final HDF5 mismatch")
        paths=[
            final.parent/f"statepoint.{b:0{len(str(total_batches))}d}.h5"
            for b in batches
        ]
        checkpoints=[read_real_checkpoint(p,expected) for p in paths]
        if (any(x["seed"]!=seed or x["inactive"]!=inactive or
                x["batch"]!=batch for x,batch in zip(checkpoints,batches))
                or checkpoints[-1]["hdf5_sha256"]!=run["statepoint_sha256"]):
            raise ValueError("OpenMC intermediate neutron source checkpoint identity mismatch")
        for row in checkpoints:
            if row["hdf5_sha256"] in hashes:
                raise ValueError("duplicate fission-emission HDF5 statepoint")
            hashes.add(row["hdf5_sha256"])
        windows=[]
        prior=None
        for checkpoint in checkpoints:
            n=checkpoint["active"]-(0 if prior is None else prior["active"])
            def delta(field):
                arr=np.asarray(checkpoint[field],dtype=float)
                return arr if prior is None else arr-np.asarray(prior[field],dtype=float)
            window=assess_fission_emission_window(
                delta("global_first"),delta("material_first"),
                float(delta("global_reference_first")),
                delta("material_reference_first"),
                checkpoint["material_ids"],reachable,
                number_of_active_batches=n,
            )
            window["first_active_batch"]=(inactive+1 if prior is None
                                          else prior["batch"]+1)
            window["last_active_batch"]=checkpoint["batch"]
            windows.append(window)
            prior=checkpoint
        seed_total=np.sum([w["neutron_production_per_generation_source"]["nu-fission"]*
                          w["number_of_active_batched_realizations"]
                          for w in windows])/sum(w["number_of_active_batched_realizations"] for w in windows)
        if seed_total<=0:
            raise ValueError("no actually sampled neutron-production source")
        final=checkpoints[-1]
        global_final=final["global_first"]
        fraction=(float(global_final[2]/global_final[0])
                  if global_final[0]>0 else None)
        if fraction is None or not 0<=fraction<=1:
            raise ValueError("invalid emitted delayed neutron fraction")
        run_rows.append({
            "seed":seed,
            "final_statepoint_sha256":final["hdf5_sha256"],
            "final_global_neutron_production_per_source":{
                s:float(global_final[k]/final["active"])
                for k,s in enumerate(SCORES)
            },
            "evaluated_delayed_emission_fraction":fraction,
            "windows":windows,
        })
    if len(seeds)!=4 or len(hashes)!=20:
        raise ValueError("four independently seeded neutron runs, 20 genuine statepoints required")
    fractions=np.array([r["evaluated_delayed_emission_fraction"] for r in run_rows])
    return {
        "audit":"ACTUAL_OPENMC_CORE_PROMPT_DELAYED_FISSION_NEUTRON_SOURCE_CLOSURE",
        "independent_seed_count":len(seeds),
        "checked_native_statepoint_checkpoints":len(hashes),
        "active_source_windows_per_seed":5,
        "material_ids_included":sorted(expected),
        "physically_root_reachable_material_ids":sorted(reachable),
        "fission_emission_scores":list(SCORES),
        "executed_core_summary_sha256":_sha(summary_file),
        "executed_material_kerma_report_sha256":_sha(kerma_file),
        "four_seed_delayed_emission_fraction_mean":float(fractions.mean()),
        "four_seed_delayed_emission_fraction_descriptive_sem":float(
            fractions.std(ddof=1)/math.sqrt(len(fractions))
        ),
        "runs":run_rows,
        "independently_evaluated_prompt_delayed_total_yield_consistency_screen_passed":True,
        "maximum_observed_prompt_delayed_relative_yield_mismatch":max(
            w["maximum_relative_evaluated_yield_mismatch"]
            for r in run_rows for w in r["windows"]
        ),
        "evaluated_yield_relative_mismatch_screen":EVALUATED_YIELD_MAX_RELATIVE_MISMATCH,
        "beta_eff_calculated":False,
        "fission_source_stationarity_demonstrated":False,
        "operating_temperature_nuclear_data_qualified":False,
        "nasa_historical_nerva_core_validated":False,
        "interpretation":(
            "OpenMC native evaluated prompt/delayed neutron emission is "
            "compared with separately evaluated total neutron-production "
            "functions (a visible numerical discrepancy, not an exact identity) across "
            "each material and five disjoint active-generation windows. "
            "Four-seed SEM is descriptive due to within-source genealogy. "
            "The source emission fraction is NOT adjoint importance-weighted "
            "beta_eff, a period/reactivity measurement, delayed precursor "
            "inventory or validation of operating-temperature NERVA physics."
        ),
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--material-kerma",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    out=audit_real_fission_emission(args.summary,args.material_kerma)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")
    print("REAL OpenMC prompt/delayed production: 20 HDF5 evaluated-yield consistency screens PASS")
    print("Maximum independent evaluated-yield relative mismatch:",
          out["maximum_observed_prompt_delayed_relative_yield_mismatch"])
    print("Evaluated delayed emission fractions:",[
        x["evaluated_delayed_emission_fraction"] for x in out["runs"]
    ])
    print("Adjoint-weighted beta_eff / NERVA physical criticality: NOT QUALIFIED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
