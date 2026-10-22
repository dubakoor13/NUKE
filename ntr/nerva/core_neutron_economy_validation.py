"""Observed neutron-generation economy from genuine OpenMC C++ transport.

Independently reconcile native analog vacuum leakage and multiplicative
scattering with neutron absorption, while preserving the distinction between
NEUTRONS ALIVE IN ONE GENERATION and daughter neutrons BANKED into the
NEXT eigenvalue generation. Reconcile material/incident-energy tracklength
reactions and four native combined-k estimates, with actual statepoint
digests at 20 saved checkpoints.

No adjustment of geometry, enrichment, materials, control drums, source
population, operating temperatures, or evaluated cross sections. This is
NOT an optimization, nuclear-weapon analysis, source-convergence certificate,
historical NASA benchmark or local nuclear-energy-deposition model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _sha(file: Path) -> str:
    h=hashlib.sha256()
    with Path(file).open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def _positive(value, label: str, *, allow_zero: bool = False) -> float:
    if isinstance(value,bool) or not isinstance(value,(int,float)):
        raise ValueError("invalid actual neutron " + label)
    v=float(value)
    if not math.isfinite(v) or (v < 0 if allow_zero else v <= 0):
        raise ValueError("invalid actual neutron " + label)
    return v


def analyze_neutron_economy_window(
    analog: dict, tracklength: dict, material_ids: list[int],
    reachable_ids: set[int], fuel_material_id: int = 19,
) -> dict:
    """Check physically conservative neutron budget with raw scored inputs.

    Physical identity (per starting neutron):
        absorption + vacuum leakage - (nu-scatter - scatter) = 1.
    Fission daughters are next-generation source, NOT current-generation
    transported neutrons. Leakage >1 is possible after (n,2n) scattering.
    """
    first=int(analog["first_active_batch"])
    last=int(analog["last_active_batch"])
    n=int(analog["active_realizations"])
    if (first<1 or last<first or n!=last-first+1
            or tracklength["first_active_batch"]!=first
            or tracklength["last_active_batch"]!=last
            or tracklength["number_of_active_generations"]!=n
            or analog["statepoint_sha256"]!=tracklength["statepoint_sha256"]):
        raise ValueError("native analog and filtered neutron window identity mismatch")
    absorption=_positive(analog["analog_absorption_per_source"],"analog absorption")
    leakage=_positive(analog["native_vacuum_leakage_per_source"],"native vacuum leakage")
    scattering=_positive(analog["net_multiplicative_scattering_per_source"],
                         "multiplicative scattering",allow_zero=True)
    analog_fission=_positive(analog["analog_nu_fission_per_source"],
                            "analog next-generation fission source")
    original_abs=_positive(tracklength["global_reaction_rates_per_source"]["absorption"],
                           "tracklength absorption")
    original_fission=_positive(tracklength["global_reaction_rates_per_source"]["nu-fission"],
                               "tracklength banked fission")
    abs_material=np.asarray(
        tracklength["material_reaction_rates_per_source"]["absorption"],dtype=float
    )
    fission_material=np.asarray(
        tracklength["material_reaction_rates_per_source"]["nu-fission"],dtype=float
    )
    if (len(material_ids)!=len(set(material_ids)) or len(material_ids)<2
            or abs_material.shape!=(len(material_ids),)
            or fission_material.shape!=(len(material_ids),)
            or not np.isfinite(abs_material).all()
            or not np.isfinite(fission_material).all()
            or np.any(abs_material<0) or np.any(fission_material<0)
            or fuel_material_id not in reachable_ids):
        raise ValueError("invalid native material-resolved neutron reaction rates")
    abs_delta=float(np.sum(abs_material)-original_abs)
    fission_delta=float(np.sum(fission_material)-original_fission)
    if abs(abs_delta)>1e-6 or abs(fission_delta)>1e-6:
        raise ValueError("global neutron reactions not conserved by physical material tally")
    by_id={i:(float(a),float(f)) for i,a,f in zip(material_ids,abs_material,fission_material)}
    for ident in set(material_ids)-reachable_ids:
        if by_id[ident]!=(0.,0.):
            raise ValueError("neutron reactions in unreachable nuclear material")
    for ident in reachable_ids-{fuel_material_id}:
        if by_id[ident][1]>max(1e-10,1e-6*original_fission):
            raise ValueError("neutron fission wrongly scored outside physical fuel material")
    if (by_id[fuel_material_id][1]<=0
            or set(material_ids)!=reachable_ids.union(set(material_ids)-reachable_ids)):
        raise ValueError("no genuine fission-producing core material")
    balance=absorption+leakage-scattering
    if abs(balance-1.)>1e-7:
        raise ValueError("actual neutron generation leakage/absorption balance violated")
    if (abs(float(analog["conservation_balance_per_source"])-balance)>1e-7
            or not analog["neutron_balance_strict_physical_screen_passed"]):
        raise ValueError("independent native neutron-balance calculation disagrees")
    # Analog and tracklength estimates differ stochastically even when
    # scored on identical events. A permissive mismatch screen is NOT a
    # formal uncertainty/significance test.
    if (abs(original_abs-absorption)>max(.012,.3*original_abs)
            or abs(original_fission-analog_fission)>max(.008,.5*original_fission)):
        raise ValueError("large analog and tracklength neutron reaction disagreement")
    total_term=leakage+absorption
    return {
        "first_active_batch":first,
        "last_active_batch":last,
        "active_generations":n,
        "statepoint_sha256":analog["statepoint_sha256"],
        "analog_absorption_per_starting_neutron":absorption,
        "native_vacuum_leakage_per_starting_neutron":leakage,
        "net_multiplicative_scattering_per_starting_neutron":scattering,
        "conservative_neutron_balance":balance,
        "native_neutron_balance_abs_residual":abs(balance-1.),
        "tracklength_absorption_per_starting_neutron":original_abs,
        "tracklength_banked_fission_neutrons_per_starting_neutron":original_fission,
        "analog_banked_fission_neutrons_per_starting_neutron":analog_fission,
        "fuel_fission_source_yield_per_starting_neutron":
            by_id[fuel_material_id][1],
        "absorption_by_reachable_material":{
            str(i):by_id[i][0] for i in sorted(reachable_ids)
        },
        "leakage_fraction_of_total_neutron_terminations":
            leakage/total_term,
        "absorption_fraction_of_total_neutron_terminations":
            absorption/total_term,
        "leakage_is_over_one_due_to_neutron_multiplication":
            leakage>1 and scattering>0,
        "fractions_are_not_independent_neutron_probabilities":True,
    }


def audit_real_neutron_economy(
    summary_file: Path, leakage_file: Path, reactions_file: Path,
    ensemble_file: Path, criticality_file: Path,
) -> dict:
    paths=[Path(x) for x in (
        summary_file, leakage_file, reactions_file, ensemble_file, criticality_file
    )]
    summary, leakage, reactions, ensemble, criticality=[
        json.loads(p.read_text(encoding="utf-8")) for p in paths
    ]
    expected=_sha(paths[0])
    if (summary.get("execution")!="ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly")!="core"
            or leakage.get("analysis")!="ACTUAL_OPENMC_GLOBAL_NEUTRON_LEAKAGE_ACTIVE_WINDOW_BALANCE"
            or reactions.get("analysis")!="ACTUAL_OPENMC_DISJOINT_NEUTRON_REACTION_HISTORY_WINDOWS"
            or ensemble.get("analysis")!="GENUINE_OPENMC_CROSS_SEED_CORE_NEUTRON_REACTION_RATES"
            or criticality.get("audit")!="GENUINE_OPENMC_CORE_MULTIPLICATION_VERSUS_CRITICALITY_CONDITION"
            or any(obj.get(key)!=expected for obj,key in (
                (leakage,"real_transport_summary_sha256"),
                (reactions,"real_transport_summary_sha256"),
                (ensemble,"real_transport_summary_sha256"),
                (criticality,"transport_summary_sha256")
            ))):
        raise ValueError("unmatched genuine neutron source/transport physical provenance")
    material_ids=ensemble["material_ids"]
    if (len(material_ids)!=9 or material_ids!=list(range(19,28))
            or not ensemble.get("nasa_historical_nerva_neutron_physics_validated") is False):
        raise ValueError("different cold conceptual core material reaction ensemble")
    reachable=set(range(19,23))
    unused=set(material_ids)-reachable
    runs=summary.get("runs")
    leak_rows=leakage.get("per_seed_neutron_conservation")
    track_rows=reactions.get("observed_seed_history")
    k_rows=criticality.get("per_seed_physical_neutron_multiplication")
    if (not isinstance(runs,list) or len(runs)!=4
            or not isinstance(leak_rows,list) or len(leak_rows)!=4
            or not isinstance(track_rows,list) or len(track_rows)!=4
            or not isinstance(k_rows,list) or len(k_rows)!=4
            or leakage.get("independent_seed_count")!=4
            or reactions.get("independent_seed_count")!=4
            or ensemble.get("independent_seed_count")!=4
            or criticality.get("independent_seed_count")!=4):
        raise ValueError("requires four actual independent OpenMC neutron ensembles")
    checkpoints=summary.get("saved_source_bank_batches")
    if not isinstance(checkpoints,list) or checkpoints!=[40,60,80,100,120]:
        raise ValueError("genuine executed source checkpoint schedule differs")
    seeds=set()
    seed_rows=[]
    all_hashes=set()
    for i,(run,lr,tr,kr) in enumerate(zip(runs,leak_rows,track_rows,k_rows)):
        seed=run["seed"]
        final=Path(run["statepoint_path"])
        if (seed in seeds or not final.is_file()
                or _sha(final)!=run["statepoint_sha256"]
                or any(x["seed"]!=seed for x in (lr,tr,kr))):
            raise ValueError("actual four-seed fission/absorption/leakage provenance differs")
        seeds.add(seed)
        lw=lr["disjoint_windows"]
        tw=tr["windows"]
        if len(lw)!=5 or len(tw)!=5:
            raise ValueError("five actual disjoint OpenMC active windows required")
        produced=[]
        for row_a,row_t in zip(lw,tw):
            batch=row_a["last_active_batch"]
            statepoint=final.parent/f"statepoint.{batch:03d}.h5"
            if (not statepoint.is_file()
                    or row_a["statepoint_sha256"]!=_sha(statepoint)
                    or row_a["statepoint_sha256"] in all_hashes):
                raise ValueError("HDF5 checkpoint neutron-fate evidence has been changed")
            all_hashes.add(row_a["statepoint_sha256"])
            produced.append(analyze_neutron_economy_window(
                row_a,row_t,material_ids,reachable
            ))
        if (produced[-1]["statepoint_sha256"]!=run["statepoint_sha256"]
                or produced[0]["first_active_batch"]!=31
                or produced[-1]["last_active_batch"]!=120
                or any(b["first_active_batch"]!=a["last_active_batch"]+1
                       for a,b in zip(produced,produced[1:]))
                or sum(r["active_generations"] for r in produced)!=90):
            raise ValueError("wrong executed active generation interval coverage")
        # Reconstruct the native neutron fate per source from disjoint
        # *generation-weighted* windows. Never average five unequal windows.
        total=sum(v["active_generations"] for v in produced)
        key_names=[
            "analog_absorption_per_starting_neutron",
            "native_vacuum_leakage_per_starting_neutron",
            "net_multiplicative_scattering_per_starting_neutron",
            "tracklength_absorption_per_starting_neutron",
            "tracklength_banked_fission_neutrons_per_starting_neutron",
            "analog_banked_fission_neutrons_per_starting_neutron",
        ]
        weighted={
            key:sum(r[key]*r["active_generations"] for r in produced)/total
            for key in key_names
        }
        for score,key in (
            ("absorption","tracklength_absorption_per_starting_neutron"),
            ("nu-fission","tracklength_banked_fission_neutrons_per_starting_neutron")
        ):
            original=ensemble["reactions"][score]["per_seed_global_rate_per_source"][i]
            if not math.isclose(weighted[key],original,rel_tol=5e-9,abs_tol=1e-9):
                raise ValueError("genuine disjoint neutron source and ensemble rate mismatch")
        if abs(weighted["tracklength_banked_fission_neutrons_per_starting_neutron"]
               -kr["combined_k"])>max(.005,10*kr["reported_k_sem"]):
            raise ValueError("native k multiplication and next-generation fission source disagree")
        loss=weighted["native_vacuum_leakage_per_starting_neutron"]+weighted[
            "analog_absorption_per_starting_neutron"
        ]
        seed_rows.append({
            "seed":seed,
            "final_statepoint_sha256":run["statepoint_sha256"],
            "disjoint_native_neutron_windows":produced,
            "weighted_active_generation_economy":weighted,
            "native_combined_k":kr["combined_k"],
            "mean_vacuum_leakage_fraction_of_all_terminals":
                weighted["native_vacuum_leakage_per_starting_neutron"]/loss,
            "physical_absorption_fraction_of_all_terminals":
                weighted["analog_absorption_per_starting_neutron"]/loss,
        })
    means={
        key:float(np.mean([v["weighted_active_generation_economy"][key]
                           for v in seed_rows]))
        for key in key_names
    }
    leakage_dominates=all(
        v["mean_vacuum_leakage_fraction_of_all_terminals"]>.9
        for v in seed_rows
    )
    if abs(means["analog_absorption_per_starting_neutron"]
           +means["native_vacuum_leakage_per_starting_neutron"]
           -means["net_multiplicative_scattering_per_starting_neutron"]
           -1.)>1e-7:
        raise ValueError("aggregate neutron-generation physical conservation lost")
    return {
        "audit":"GENUINE_OPENMC_MEASURED_NEUTRON_FATE_AND_FISSION_BANK_ECONOMY",
        "real_transport_summary_sha256":expected,
        "source_evidence_sha256":{p.name:_sha(p) for p in paths[1:]},
        "independent_seed_count":len(seeds),
        "saved_checkpoint_count":len(all_hashes),
        "evaluated_nuclear_data_sha256":summary["cross_sections_xml_sha256"],
        "mean_per_starting_neutron":means,
        "vacuum_leakage_dominates_terminal_neutron_losses":leakage_dominates,
        "per_seed":seed_rows,
        "independent_neutron_genealogy_or_spatial_source_stationarity_validated":False,
        "physically_validated_critical_nerva_core":False,
        "local_energy_deposition_physically_validated":False,
        "neutron_fate_fraction_is_source_independent_particle_probability":False,
        "interpretation":(
            "Native OpenMC simulated one-generation neutron losses close exactly "
            "as absorption + vacuum leakage - multiplicative scattering = 1 "
            "per initial eigenvalue source neutron. Fission daughters are "
            "banked for the NEXT generation and are not current-generation "
            "multiplicative scattering. A leakage rate >1 is not a "
            "probability error where in-generation scattering multiplies "
            "neutrons. Terminal loss fractions are ratios of observed "
            "termination EVENT COUNTS, not probabilities of independent "
            "particle histories. This physical accounting cannot establish "
            "converged sources, high-temperature nuclear data, reactor "
            "criticality, energy deposition or NASA reference fidelity."
        ),
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("summary","leakage","reactions","ensemble","criticality","output"):
        p.add_argument("--"+key,type=Path,required=True)
    args=p.parse_args()
    report=audit_real_neutron_economy(
        args.summary,args.leakage,args.reactions,args.ensemble,args.criticality
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Genuine physical per-generation neutron fate: PASS")
    print("Four-seed event-rate means:",report["mean_per_starting_neutron"])
    print("Vacuum leakage dominates:",report[
        "vacuum_leakage_dominates_terminal_neutron_losses"
    ])
    print("NERVA source convergence, criticality and heating: NOT VALIDATED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
