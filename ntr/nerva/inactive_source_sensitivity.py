"""Genuine OpenMC source-relaxation test: 30 versus 110 inactive batches.

Run the EXACT same cold conceptual NERVA-derived core, evaluated data,
temperatures and 3000-source-neutron population with two source burn-ins.
Each ensemble then records the SAME 90 active generations with five matched
relative active checkpoints. Four disjoint RNG seeds per arm expose observed
fission-source spatial population collapse and between-seed variability.

Increasing inactive generations is numerical verification, not a NERVA
reactor design or criticality optimization. A passed integrity check cannot
prove independent fission ancestors, stationary source or NASA benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .source_resolution_audit import audit_real_transport_summary


XML_NAMES = ("geometry.xml", "materials.xml", "tallies.xml")
CONFIG_VARIABLES = {"seed", "batches", "inactive", "statepoint_batches"}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_arm(root: Path) -> tuple[dict, list[dict], dict]:
    root = Path(root)
    summary_file = root / "real_core_transport_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if (summary.get("execution") != "ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly") != "core"
            or summary.get("rings") != 3):
        raise ValueError("requires actual OpenMC 3-ring core neutron transport")
    runs = summary.get("runs", [])
    if len(runs) < 4 or len(set(x["seed"] for x in runs)) != len(runs):
        raise ValueError("each source-burn-in arm requires >=4 distinct seeds")
    if summary.get("cold_xs_smoke_test_only") is not True:
        raise ValueError("compare only the matched cold OpenMC surrogate")
    if (not isinstance(summary.get("batches"), int)
            or not isinstance(summary.get("inactive_batches"), int)
            or not isinstance(summary.get("particles_per_generation"), int)):
        raise ValueError("invalid actual OpenMC generation metadata")
    if summary["batches"] <= summary["inactive_batches"]:
        raise ValueError("source burn-in exceeds total generations")
    requested = summary.get("saved_source_bank_batches", [])
    expected_active = [n-summary["inactive_batches"] for n in requested]
    if (len(requested) != 5 or expected_active[0] <= 0
            or requested[-1] != summary["batches"]
            or any(a >= b for a,b in zip(expected_active, expected_active[1:]))):
        raise ValueError("source checkpoints must map to five increasing active batches")

    provenances = []
    for run in runs:
        seed = run["seed"]
        point = Path(run["statepoint_path"])
        if not point.is_file() or run["statepoint_sha256"] != _sha256(point):
            raise ValueError("actual statepoint SHA256 or file invalid")
        info = run.get("hdf5_eigenvalue_integrity", {})
        if (info.get("seed") != seed
                or info.get("particles_per_generation") != summary["particles_per_generation"]
                or info.get("inactive_batches") != summary["inactive_batches"]
                or info.get("current_batch") != summary["batches"]):
            raise ValueError("physical statepoint generation metadata disagree")
        provenance_path = point.parent / "transport_provenance.json"
        p = json.loads(provenance_path.read_text(encoding="utf-8"))
        cfg = p.get("config", {})
        if (p.get("assembly") != "core" or cfg.get("seed") != seed
                or cfg.get("particles") != summary["particles_per_generation"]
                or cfg.get("batches") != summary["batches"]
                or cfg.get("inactive") != summary["inactive_batches"]
                or cfg.get("statepoint_batches") != requested
                or p.get("nuclear_data", {}).get("cross_sections_sha256")
                  != summary.get("cross_sections_xml_sha256")):
            raise ValueError("evaluated neutron transport provenance differs from statepoint")
        for name in XML_NAMES:
            actual = point.parent / name
            if (not actual.is_file()
                    or p.get("xml_sha256", {}).get(name) != _sha256(actual)):
                raise ValueError("actual reactor material/geometry/tallies SHA256 differs")
        provenances.append(p)

    spatial = audit_real_transport_summary(summary_file)
    return summary, provenances, spatial


def _same_physical_core(a: dict, b: dict) -> None:
    if (a.get("assembly") != "core" or b.get("assembly") != "core"):
        raise ValueError("comparison must use same physical core")
    if a.get("openmc_version") != b.get("openmc_version"):
        raise ValueError("different OpenMC transport version")
    for name in XML_NAMES:
        if (not a["xml_sha256"].get(name)
                or a["xml_sha256"].get(name) != b["xml_sha256"].get(name)):
            raise ValueError("changed reactor geometry/material/tally definition: " + name)
    x = a["nuclear_data"]["cross_sections_sha256"]
    y = b["nuclear_data"]["cross_sections_sha256"]
    if not x or x != y:
        raise ValueError("evaluated neutron cross-section index changed")
    ca = {k:v for k,v in a["config"].items() if k not in CONFIG_VARIABLES}
    cb = {k:v for k,v in b["config"].items() if k not in CONFIG_VARIABLES}
    if ca != cb:
        raise ValueError("physical core or neutron source population changed")


def assess_burnin_evidence(
    early: dict, extended: dict,
    early_audit: dict, extended_audit: dict
) -> dict:
    """Pure numeric reduction of independently already-validated OpenMC runs."""
    if early.get("assembly") != "core" or extended.get("assembly") != "core":
        raise ValueError("only identical core neutron source histories are comparable")
    particles = early["particles_per_generation"]
    if (particles < 100 or extended["particles_per_generation"] != particles):
        raise ValueError("source particle populations differ")
    counts = [s["batches"]-s["inactive_batches"] for s in (early, extended)]
    if counts[0] < 30 or counts[0] != counts[1]:
        raise ValueError("comparison must use same nontrivial number of ACTIVE generations")
    if not early["inactive_batches"] < extended["inactive_batches"]:
        raise ValueError("second OpenMC arm must increase INACTIVE source-relaxation generations")
    offsets = [
        [n-s["inactive_batches"] for n in s["saved_source_bank_batches"]]
        for s in (early, extended)
    ]
    if (offsets[0] != offsets[1] or len(offsets[0]) < 3
            or offsets[0][-1] != counts[0]):
        raise ValueError("different active source checkpoint observation windows")
    seeds = [set(run["seed"] for run in s["runs"]) for s in (early,extended)]
    if any(len(z) < 4 for z in seeds) or seeds[0] & seeds[1]:
        raise ValueError("four nonoverlapping independent seeds needed for each arm")
    if early.get("cross_sections_xml_sha256") != extended.get("cross_sections_xml_sha256"):
        raise ValueError("different evaluated physical nuclear data")
    lower = early["independent_seed_source_bank_shape_diagnostics"]
    upper = extended["independent_seed_source_bank_shape_diagnostics"]
    for key in ("spatial_lower_cm","spatial_upper_cm"):
        if lower.get(key) != upper.get(key):
            raise ValueError("neutron source spatial scoring bounds differ")

    def reduce(summary: dict, audit: dict) -> dict:
        if (audit.get("analysis") != "EXACT_SAME_OPENMC_SOURCE_BANKS_ON_NESTED_SPATIAL_GRIDS"
                or not audit.get("grid_resolution_audit_passed")
                or audit.get("source_stationarity_validated") is not False):
            raise ValueError("genuine multiresolution fission-source audit required")
        resolutions = audit["resolutions"]
        if [x["bins_per_axis"] for x in resolutions] != [2,4,8]:
            raise ValueError("source TV must be checked on nested 2³ 4³ 8³")
        if (audit.get("nested_jensen_shannon_information_refinement_integrity_passed") is not True
                or audit.get("jensen_shannon_total_variation_pinsker_inequality_passed") is not True):
            raise ValueError("native nested-grid Jensen-Shannon source information audit required")
        prior_js=None
        for level in resolutions:
            js=float(level.get("final_mean_cross_seed_jensen_shannon_nats",float("nan")))
            tv=float(level.get("final_mean_cross_seed_tv",float("nan")))
            if (not math.isfinite(js) or not math.isfinite(tv)
                    or not 0<=tv<=1 or not 0<=js<=math.log(2.)+2e-12
                    or js+2e-12<.5*tv*tv):
                raise ValueError("source JSD/TV information inequality violated")
            if prior_js is not None and js+2e-12<prior_js:
                raise ValueError("source Jensen-Shannon nested grid refinement decreased")
            prior_js=js
        source = summary["multi_checkpoint_source_bank_trajectory"]
        last = [run["snapshots"][-1] for run in source["runs"]]
        if len(last) != len(summary["runs"]):
            raise ValueError("different real source-bank seed population")
        n=summary["particles_per_generation"]
        fractions=np.array([
            float(p["inverse_simpson_distinct_position_count"])/n
            for p in last], dtype=float)
        if not np.isfinite(fractions).all() or np.any(fractions <= 0) or np.any(fractions > 1):
            raise ValueError("invalid physical effective source-site fractions")
        final_mean_tv=float(resolutions[-1]["final_mean_cross_seed_tv"])
        if not math.isfinite(final_mean_tv) or not 0 <= final_mean_tv <= 1:
            raise ValueError("invalid actual source spatial total-variation distance")
        k=float(summary["run_statistics"]["inverse_variance_mean_keff"])
        sigma=float(summary["run_statistics"]["dispersion_inflated_mean_keff_error"])
        if not math.isfinite(k) or not math.isfinite(sigma) or k <= 0 or sigma <= 0:
            raise ValueError("invalid observed native eigenvalue statistics")
        return {
            "inactive_generations":summary["inactive_batches"],
            "total_generations":summary["batches"],
            "active_generations":summary["batches"]-summary["inactive_batches"],
            "seed_count":len(last),
            "particles_per_generation":n,
            "k_eff_mean":k,
            "descriptive_k_eff_sem_with_seed_dispersion":sigma,
            "final_mean_spatial_TV_2cube":float(resolutions[0]["final_mean_cross_seed_tv"]),
            "final_mean_spatial_TV_8cube":final_mean_tv,
            "final_mean_spatial_JSD_nats_2cube":
                float(resolutions[0]["final_mean_cross_seed_jensen_shannon_nats"]),
            "final_mean_spatial_JSD_nats_8cube":
                float(resolutions[-1]["final_mean_cross_seed_jensen_shannon_nats"]),
            "nested_JSD_TV_information_bounds_verified":True,
            "final_mean_effective_distinct_position_fraction":float(fractions.mean()),
            "final_min_effective_distinct_position_fraction":float(fractions.min()),
            "minimum_effective_fraction_all_checkpoints":
                float(resolutions[-1]["minimum_effective_distinct_position_fraction"]),
            "source_diversity_screen_status":audit["source_diversity_screen_status"],
            "final_snapshot_sha256":[p["statepoint_sha256"] for p in last],
        }
    a=reduce(early,early_audit)
    b=reduce(extended,extended_audit)
    difference=b["k_eff_mean"]-a["k_eff_mean"]
    return {
        "analysis":"GENUINE_OPENMC_NEUTRON_SOURCE_INACTIVE_GENERATION_SENSITIVITY",
        "physical_reactor_core_unchanged":True,
        "same_number_of_active_tally_generations":True,
        "identical_relative_source_checkpoint_active_batches":offsets[0],
        "source_a":a,
        "source_b":b,
        "observed_k_eff_difference_after_longer_inactive":difference,
        "difference_over_formal_combined_seed_dispersion_sem":
            abs(difference)/math.hypot(
                a["descriptive_k_eff_sem_with_seed_dispersion"],
                b["descriptive_k_eff_sem_with_seed_dispersion"]),
        "difference_in_fine_grid_cross_seed_source_TV":
            b["final_mean_spatial_TV_8cube"]-a["final_mean_spatial_TV_8cube"],
        "difference_in_fine_grid_cross_seed_JSD_nats":
            b["final_mean_spatial_JSD_nats_8cube"]-
            a["final_mean_spatial_JSD_nats_8cube"],
        "both_source_arms_passed_information_inequalities":True,
        "source_information_divergence_is_a_convergence_probability":False,
        "difference_in_mean_effective_position_fraction":
            b["final_mean_effective_distinct_position_fraction"]-
            a["final_mean_effective_distinct_position_fraction"],
        "fission_source_stationarity_independently_proven":False,
        "historical_nasa_nerva_reactor_validated":False,
        "interpretation":(
            "Same reactor, material inventory, neutron data, source particles, "
            "90 active batches and matched relative checkpoints; more INACTIVE "
            "generations test fission-source relaxation. Distinct RNG seeds "
            "and correlated fission descendants preclude a causal test or "
            "independent-neutron p-value. Source-site cloning and across-seed "
            "distributions remain diagnostic, not convergence certification. "
            "The corresponding Jensen-Shannon information differences are "
            "checked against TV and nested-grid refinement but are NOT "
            "reactor convergence probabilities or independent fission histories. "
            "No nuclear design, control, enrichment or reactor optimization."
        ),
    }


def assess_actual_source_burnin(short_root: Path, long_root: Path) -> dict:
    early,p0,audit0=_load_arm(Path(short_root))
    extended,p1,audit1=_load_arm(Path(long_root))
    for p in p0+p1:
        _same_physical_core(p0[0],p)
    result=assess_burnin_evidence(early,extended,audit0,audit1)
    result.update({
        "nuclear_data_index_sha256":early["cross_sections_xml_sha256"],
        "source_a_transport_summary_sha256":
            _sha256(Path(short_root)/"real_core_transport_summary.json"),
        "source_b_transport_summary_sha256":
            _sha256(Path(long_root)/"real_core_transport_summary.json"),
    })
    return result


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--short",required=True,type=Path)
    p.add_argument("--extended",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    report=assess_actual_source_burnin(args.short,args.extended)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    for name in ("source_a","source_b"):
        item=report[name]
        print(name,"inactive",item["inactive_generations"],
              "keff",item["k_eff_mean"],
              "cross-seed 8^3 TV",item["final_mean_spatial_TV_8cube"],
              "cross-seed 8^3 JSD nats",item["final_mean_spatial_JSD_nats_8cube"],
              "effective-site fraction",item["final_mean_effective_distinct_position_fraction"],
              "diversity screen",item["source_diversity_screen_status"])
    print("Source convergence proven: NO; NASA historical core benchmark: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
