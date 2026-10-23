"""Numerical population-size sensitivity of REAL OpenMC eigenvalue runs.

Compare two physical OpenMC transport ensembles that differ in Monte Carlo
source particles per generation but NOT in geometry, materials, cross
sections, temperature, batches, tally definitions or source checkpoints.

Evidence comes from actual statepoint-derived NUKE JSON and matching
transport provenance. This is NOT a proof of source convergence: the
effective POSITION count of a resampled source bank is not the number
of genealogically independent neutron histories. The comparison never
adjusts core neutronics to produce a target multiplication factor.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, median
import re


def _checked_float(value: object, label: str, *, strictly_positive: bool = False) -> float:
    try:
        v = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label} not numeric") from exc
    if not math.isfinite(v) or (strictly_positive and v <= 0):
        raise ValueError(f"{label} invalid or nonfinite")
    return v


def _load_evidence(root: Path) -> tuple[dict, list[dict], dict]:
    file = root / "real_core_transport_summary.json"
    summary = json.loads(file.read_text(encoding="utf-8"))
    if summary.get("execution") != "ACTUAL_OPENMC_TRANSPORT":
        raise ValueError("only genuine OpenMC neutron transport reports accepted")
    if summary.get("assembly") != "core":
        raise ValueError("different OpenMC assembly or missing core transport")
    histories = summary.get("particles_per_generation")
    if isinstance(histories, bool) or not isinstance(histories, int) or histories < 100:
        raise ValueError("OpenMC particles-per-generation metadata invalid")
    results = summary.get("runs")
    if not isinstance(results, list) or len(results) < 4:
        raise ValueError("at least four independent OpenMC seeds per population")
    seeds = [row.get("seed") for row in results]
    if any(not isinstance(seed, int) or seed <= 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("OpenMC random seeds must be positive and unique")
    if sorted(seeds) != sorted(summary["run_statistics"]["seeds"]):
        raise ValueError("OpenMC run/summary random seed list inconsistent")
    if summary["run_statistics"].get("independent_seed_count") != len(seeds):
        raise ValueError("OpenMC independent-seed metadata inconsistent")
    for row in results:
        hdf5 = row.get("hdf5_eigenvalue_integrity", {})
        if (
            hdf5.get("particles_per_generation") != histories
            or hdf5.get("seed") != row["seed"]
            or hdf5.get("current_batch") != summary["batches"]
            or hdf5.get("inactive_batches") != summary["inactive_batches"]
        ):
            raise ValueError("actual OpenMC HDF5 statepoint eigenvalue metadata inconsistent")
        if not re.fullmatch("[a-f0-9]{64}", row.get("statepoint_sha256", "")):
            raise ValueError("actual OpenMC SHA256 statepoint digest missing")
        _checked_float(row.get("keff_mean"), "per-seed keff", strictly_positive=True)
        _checked_float(row.get("keff_std_dev"), "per-seed keff error", strictly_positive=True)
    if len(set(row["statepoint_sha256"] for row in results)) != len(results):
        raise ValueError("OpenMC final statepoint reused")
    source = summary.get("multi_checkpoint_source_bank_trajectory")
    if not isinstance(source, dict) or source.get("independent_seed_count") != len(seeds):
        raise ValueError("actual OpenMC multibatch source history absent")
    batches = source.get("saved_batches")
    if (not isinstance(batches, list) or len(batches) < 3
            or batches[-1] != summary["batches"]
            or batches[0] <= summary["inactive_batches"]
            or any(a >= b for a, b in zip(batches, batches[1:]))):
        raise ValueError("actual OpenMC source-bank checkpoint history invalid")
    seed_source = source.get("runs")
    if (not isinstance(seed_source, list) or len(seed_source) != len(seeds)
            or {row.get("seed") for row in seed_source} != set(seeds)):
        raise ValueError("source trajectories must match independent transport seeds")
    for record in seed_source:
        snapshots = record.get("snapshots")
        if not isinstance(snapshots, list) or len(snapshots) != len(batches):
            raise ValueError("missing actual source-bank checkpoints")
        for expected_batch, snap in zip(batches, snapshots):
            if (snap.get("completed_batch") != expected_batch
                    or snap.get("seed") != record["seed"]
                    or snap.get("source_sites") != histories):
                raise ValueError("OpenMC source-bank history does not match particle population")
            effective = _checked_float(
                snap.get("inverse_simpson_distinct_position_count"),
                "inverse-Simpson source diversity", strictly_positive=True
            )
            if effective > histories * (1 + 1e-12):
                raise ValueError("source effective distinct positions exceed source population")
            if not 1 <= snap.get("exact_distinct_site_positions", 0) <= histories:
                raise ValueError("invalid number of exact fission source positions")
    provenance = []
    for seed in seeds:
        path = root / f"seed_{seed}" / "transport_provenance.json"
        p = json.loads(path.read_text(encoding="utf-8"))
        cfg = p.get("config", {})
        if (
            p.get("assembly") != summary["assembly"]
            or cfg.get("particles") != histories
            or cfg.get("seed") != seed
            or cfg.get("batches") != summary["batches"]
            or cfg.get("inactive") != summary["inactive_batches"]
            or cfg.get("statepoint_batches") != batches
            or p.get("nuclear_data", {}).get("cross_sections_sha256")
              != summary.get("cross_sections_xml_sha256")
        ):
            raise ValueError("transport provenance does not match actual OpenMC report")
        provenance.append(p)
    return summary, provenance, source


def _physical_identity(a: dict, b: dict) -> None:
    """Only the RNG seed and source-particle POPULATION may differ."""
    for key in ("geometry.xml", "materials.xml", "tallies.xml"):
        h1 = a.get("xml_sha256", {}).get(key)
        h2 = b.get("xml_sha256", {}).get(key)
        if not isinstance(h1, str) or not h1 or h1 != h2:
            raise ValueError(f"OpenMC {key} differs; invalid population-size comparison")
    x1 = a.get("nuclear_data", {}).get("cross_sections_sha256")
    x2 = b.get("nuclear_data", {}).get("cross_sections_sha256")
    if not isinstance(x1, str) or not x1 or x1 != x2:
        raise ValueError("evaluated neutron cross-section identity differs")
    for p in (a, b):
        if p.get("assembly") != "core":
            raise ValueError("only the same OpenMC reactor core is comparable")
    mutable = {"seed", "particles"}
    cfg1 = {k: v for k, v in a["config"].items() if k not in mutable}
    cfg2 = {k: v for k, v in b["config"].items() if k not in mutable}
    if cfg1 != cfg2:
        raise ValueError("reactor physics or Monte Carlo batch schedule differs")


def assess_native_spatial_information_series(source: dict) -> dict:
    """Source-shape information at EXACTLY the already scored 4³ grid.

    Matched JSD/TV diagnostics must satisfy JSD >= TV²/2 for natural
    logarithms. This is a deterministic identity, not an iid neutron
    significance test or evidence of fission-source stationarity.
    """
    if source.get("jensen_shannon_divergence_natural_logarithm_and_no_pseudocounts") is not True:
        raise ValueError("real OpenMC source missing unmodified Jensen-Shannon provenance")
    batches=source.get("saved_batches")
    if not isinstance(batches,list) or len(batches)<3:
        raise ValueError("real source checkpoint schedule missing")
    series=(
        ("between_seed_mean_total_variation_by_batch",
         "between_seed_mean_jensen_shannon_nats_by_batch",len(batches)),
        ("mean_within_seed_adjacent_total_variation_by_interval",
         "mean_within_seed_adjacent_jensen_shannon_nats_by_interval",
         len(batches)-1),
    )
    outputs={}
    for tv_name,js_name,length in series:
        tv=source.get(tv_name)
        js=source.get(js_name)
        if (not isinstance(tv,list) or not isinstance(js,list)
                or len(tv)!=length or len(js)!=length):
            raise ValueError("source Jensen-Shannon / TV checkpoint arrays missing or misaligned")
        checked_tv=[_checked_float(v,"source TV") for v in tv]
        checked_js=[_checked_float(v,"source Jensen-Shannon information") for v in js]
        if any(not 0<=t<=1 or not 0<=j<=math.log(2)+2e-12
               or j+2e-12<.5*t*t
               for t,j in zip(checked_tv,checked_js)):
            raise ValueError("source JSD/TV Pinsker information inequality violated")
        outputs[js_name]=checked_js
    return {
        "final_between_seed_jensen_shannon_nats":
            outputs["between_seed_mean_jensen_shannon_nats_by_batch"][-1],
        "mean_adjacent_jensen_shannon_nats":mean(
            outputs["mean_within_seed_adjacent_jensen_shannon_nats_by_interval"]
        ),
        "source_jensen_shannon_and_tv_pinsker_consistent":True,
        "source_information_is_convergence_proof":False,
        "checkpoint_information_by_batch_nats":
            outputs["between_seed_mean_jensen_shannon_nats_by_batch"],
        "temporal_information_by_interval_nats":
            outputs["mean_within_seed_adjacent_jensen_shannon_nats_by_interval"],
    }


def assess_population_sensitivity(low_root: Path, high_root: Path) -> dict:
    """Numerical sensitivity only; no independent-source convergence claim."""
    low, low_provs, low_sources = _load_evidence(Path(low_root))
    high, high_provs, high_sources = _load_evidence(Path(high_root))
    lo_n = low["particles_per_generation"]
    hi_n = high["particles_per_generation"]
    if not lo_n < hi_n:
        raise ValueError("population sweep must strictly increase source particles")
    for item in (*low_provs, *high_provs):
        _physical_identity(low_provs[0], item)
    seeds_lo = {row["seed"] for row in low["runs"]}
    seeds_hi = {row["seed"] for row in high["runs"]}
    if seeds_lo & seeds_hi:
        raise ValueError("population ensembles must use disjoint independent random seeds")
    hashes = [
        row["statepoint_sha256"] for row in low["runs"] + high["runs"]
    ]
    if len(set(hashes)) != len(hashes):
        raise ValueError("population ensembles reuse physical OpenMC statepoint")

    def _stats(summary: dict, source: dict) -> dict:
        count = summary["particles_per_generation"]
        final = [r["snapshots"][-1] for r in source["runs"]]
        effective = [
            float(r["inverse_simpson_distinct_position_count"]) for r in final
        ]
        unique = [int(r["exact_distinct_site_positions"]) for r in final]
        mean_tv = _checked_float(
            source["between_seed_mean_total_variation_by_batch"][-1],
            "final between-seed source TV"
        )
        if not 0 <= mean_tv <= 1:
            raise ValueError("source TV must be a probability distance")
        temporal = [
            _checked_float(v, "within-seed source TV")
            for v in source["mean_within_seed_adjacent_total_variation_by_interval"]
        ]
        if len(temporal) != len(source["saved_batches"]) - 1 or any(
            not 0 <= v <= 1 for v in temporal
        ):
            raise ValueError("source trajectory TV intervals invalid")
        js=assess_native_spatial_information_series(source)
        k = _checked_float(
            summary["run_statistics"]["inverse_variance_mean_keff"],
            "ensemble k-effective", strictly_positive=True
        )
        err = _checked_float(
            summary["run_statistics"]["dispersion_inflated_mean_keff_error"],
            "ensemble statistical standard error", strictly_positive=True
        )
        return {
            "particles_per_generation": count,
            "independent_seed_count": len(source["runs"]),
            "ensemble_mean_keff": k,
            "ensemble_dispersion_adjusted_standard_error": err,
            "final_mean_exact_distinct_positions": mean(unique),
            "final_mean_inverse_simpson_effective_positions": mean(effective),
            "final_median_inverse_simpson_effective_positions": median(effective),
            "final_effective_position_fraction_of_bank": mean(effective) / count,
            "final_between_seed_source_total_variation": mean_tv,
            "mean_adjacent_source_total_variation": mean(temporal),
            "adjacent_source_total_variation_by_interval": temporal,
            "minimum_effective_source_positions": min(effective),
            "maximum_effective_source_positions": max(effective),
            **js,
        }

    small = _stats(low, low_sources)
    large = _stats(high, high_sources)
    norm = math.hypot(
        small["ensemble_dispersion_adjusted_standard_error"],
        large["ensemble_dispersion_adjusted_standard_error"]
    )
    z = abs(small["ensemble_mean_keff"] - large["ensemble_mean_keff"]) / norm
    return {
        "comparison": "real_OpenMC_source_population_numerical_sensitivity",
        "core_unchanged": True,
        "evaluated_cross_sections_sha256": low["cross_sections_xml_sha256"],
        "saved_source_batches": low_sources["saved_batches"],
        "low_population": small,
        "high_population": large,
        "effective_position_count_growth_factor": (
            large["final_mean_inverse_simpson_effective_positions"] /
            small["final_mean_inverse_simpson_effective_positions"]
        ),
        "effective_position_fraction_growth_factor": (
            large["final_effective_position_fraction_of_bank"] /
            small["final_effective_position_fraction_of_bank"]
        ),
        "k_effective_difference": (
            large["ensemble_mean_keff"] - small["ensemble_mean_keff"]
        ),
        "k_effective_difference_over_formal_combined_standard_error": z,
        "change_in_between_seed_source_TV": (
            large["final_between_seed_source_total_variation"] -
            small["final_between_seed_source_total_variation"]
        ),
        "change_in_mean_adjacent_source_TV": (
            large["mean_adjacent_source_total_variation"] -
            small["mean_adjacent_source_total_variation"]
        ),
        "change_in_final_between_seed_source_JSD_nats":(
            large["final_between_seed_jensen_shannon_nats"] -
            small["final_between_seed_jensen_shannon_nats"]
        ),
        "change_in_mean_adjacent_source_JSD_nats":(
            large["mean_adjacent_jensen_shannon_nats"] -
            small["mean_adjacent_jensen_shannon_nats"]
        ),
        "both_population_source_information_bounds_verified":True,
        "jensen_shannon_source_differences_are_confidence_intervals":False,
        "fission_source_convergence_demonstrated": False,
        "source_site_effective_count_is_genealogically_independent_neutron_count":False,
        "nasa_nerva_historical_model_validated": False,
        "interpretation": (
            "Sensitivity to generation population at FIXED geometry, materials, "
            "neutron cross sections, and batch schedule. Different seeds were "
            "used at each level. Changes in effective distinct source positions "
            "and source-shape TV are observed sample properties, not proof of "
            "independent fission histories. Population-size agreement in k-eff "
            "does not prove equilibrium source or remove evaluated-data bias. "
            "The independently scored symmetric JSD (natural logarithms) "
            "and TV are constrained by Pinsker's mathematical inequality, "
            "but neither supplies an independent-fission-source p-value. "
            "This comparison is a numerical transport experiment, not a "
            "criticality design optimization or hot NERVA benchmark."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assess_population_sensitivity(args.low, args.high)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for label in ("low_population", "high_population"):
        item = report[label]
        print(
            label, "particles", item["particles_per_generation"],
            "keff", item["ensemble_mean_keff"],
            "source-site ESS", item["final_mean_inverse_simpson_effective_positions"],
            "cross-seed TV", item["final_between_seed_source_total_variation"],
            "time-adjacent TV", item["mean_adjacent_source_total_variation"],
            "cross-seed JSD nats",item["final_between_seed_jensen_shannon_nats"],
            "time-adjacent JSD nats",item["mean_adjacent_jensen_shannon_nats"]
        )
    print("Fission-source convergence demonstrated: NO")
    print("Real-population sensitivity report:", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
