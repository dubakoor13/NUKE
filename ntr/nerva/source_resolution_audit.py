"""Resolution-sensitivity QA of genuine OpenMC fission-source HDF5 banks.

A 4x4x4 histogram alone cannot establish that a fission-source shape is
stationary. This audit repeats the SAME saved HDF5 banks on nested 2/4/8
grids. For fixed distributions total-variation distance cannot decrease
under bin refinement (data-processing inequality). Violation indicates
a numerical/provenance error, not favorable reactor physics.

Exact-coordinate source cloning is measured independently of binning.
A configurable population-diversity threshold is an engineering screen,
NOT a universal criterion for criticality or source stationarity.
There is NO core-design, enrichment or reactivity optimization here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from .source_bank_trajectory import source_bank_trajectory


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested_resolutions(sides: tuple[int, ...]) -> None:
    if (len(sides) < 2
            or any(isinstance(n, bool) or not isinstance(n, int)
                   or n < 2 or n > 32 for n in sides)
            or any(right <= left or right % left != 0
                   for left, right in zip(sides, sides[1:]))):
        raise ValueError("source grids must be strictly nested increasing integers in [2,32]")


def audit_source_grids(
    trajectories: list[list[Path]], *,
    lower_cm: tuple[float, float, float],
    upper_cm: tuple[float, float, float],
    sides: tuple[int, ...] = (2, 4, 8),
    minimum_effective_fraction: float = 0.05,
) -> dict:
    """Compare identical saved source banks across nested 3-D histograms.

    A positive spatial TV or a large resolution change is evidence about
    observed samples, not a hypothesis test: genealogical correlations
    invalidate naive per-neutron independent-sample uncertainty estimates.
    """
    _nested_resolutions(sides)
    if (not math.isfinite(minimum_effective_fraction)
            or not 0.0 < minimum_effective_fraction <= 1.0):
        raise ValueError("minimum effective source fraction must lie in (0,1]")
    outputs = []
    reference = None
    for side in sides:
        report = source_bank_trajectory(
            trajectories, lower_cm=lower_cm, upper_cm=upper_cm,
            bins=(side, side, side),
        )
        snapshots = [entry["snapshots"] for entry in report["runs"]]
        sizes = [float(meta["source_sites"])
                 for run in snapshots for meta in run]
        if not sizes or any(n <= 0 for n in sizes):
            raise ValueError("source bank has invalid nominal particle population")
        effective_fractions = [
            float(meta["inverse_simpson_distinct_position_count"]) /
            float(meta["source_sites"])
            for run in snapshots for meta in run
        ]
        exact_unique_fractions = [
            float(meta["distinct_position_fraction"])
            for run in snapshots for meta in run
        ]
        smallest_effective = min(effective_fractions)
        smallest_distinct = min(exact_unique_fractions)
        if (not math.isfinite(smallest_effective)
                or smallest_effective <= 0 or smallest_effective > 1.00000001):
            raise ValueError("invalid inverse-Simpson source-position diversity")
        final_cross = report["between_seed_mean_total_variation_by_batch"][-1]
        final_max = report["between_seed_max_total_variation_by_batch"][-1]
        adjacent = report["mean_within_seed_adjacent_total_variation_by_interval"]
        pooled_adjacent = report["ensemble_mean_adjacent_total_variation_by_interval"]
        # A complementary information-theoretic metric provides independent
        # math/implementation consistency checks of genuine source shapes.
        # Unlike entropy alone, JSD compares matched distributions on a grid.
        pooled_js=report["ensemble_mean_adjacent_jensen_shannon_nats_by_interval"]
        record = {
            "bins_per_axis": side,
            "number_of_spatial_bins": side ** 3,
            "saved_batches": report["saved_batches"],
            "independent_seed_count": report["independent_seed_count"],
            "mean_cross_seed_tv_per_checkpoint":
                report["between_seed_mean_total_variation_by_batch"],
            "maximum_cross_seed_tv_per_checkpoint":
                report["between_seed_max_total_variation_by_batch"],
            "mean_within_seed_tv_per_interval": adjacent,
            "pooled_mean_source_tv_per_interval": pooled_adjacent,
            "mean_cross_seed_jensen_shannon_nats_per_checkpoint":
                report["between_seed_mean_jensen_shannon_nats_by_batch"],
            "maximum_cross_seed_jensen_shannon_nats_per_checkpoint":
                report["between_seed_max_jensen_shannon_nats_by_batch"],
            "mean_within_seed_jensen_shannon_nats_per_interval":
                report["mean_within_seed_adjacent_jensen_shannon_nats_by_interval"],
            "pooled_source_jensen_shannon_nats_per_interval":pooled_js,
            "final_mean_cross_seed_jensen_shannon_nats":
                report["between_seed_mean_jensen_shannon_nats_by_batch"][-1],
            "final_max_cross_seed_jensen_shannon_nats":
                report["between_seed_max_jensen_shannon_nats_by_batch"][-1],
            "final_mean_cross_seed_tv": final_cross,
            "final_max_cross_seed_tv": final_max,
            "minimum_effective_distinct_position_fraction": smallest_effective,
            "minimum_exact_unique_position_fraction": smallest_distinct,
            "minimum_effective_distinct_position_count": min(
                float(meta["inverse_simpson_distinct_position_count"])
                for run in snapshots for meta in run
            ),
            "minimum_nominal_bank_population": min(sizes),
            "all_checkpoint_statepoint_sha256": [
                [meta["statepoint_sha256"] for meta in run]
                for run in snapshots
            ],
        }
        # Universal information-theoretic Pinsker inequality for Jensen-
        # Shannon (natural logs): JSD(P,Q) >= TV(P,Q)^2/2. Applied to
        # real within-seed, between-seed and pooled spatial comparisons.
        # Average JSD also exceeds (average TV)^2/2 by Jensen's inequality.
        for tv_key, info_key in (
            ("mean_cross_seed_tv_per_checkpoint",
             "mean_cross_seed_jensen_shannon_nats_per_checkpoint"),
            ("maximum_cross_seed_tv_per_checkpoint",
             "maximum_cross_seed_jensen_shannon_nats_per_checkpoint"),
            ("mean_within_seed_tv_per_interval",
             "mean_within_seed_jensen_shannon_nats_per_interval"),
            ("pooled_mean_source_tv_per_interval",
             "pooled_source_jensen_shannon_nats_per_interval"),
        ):
            tv=record[tv_key]
            info=record[info_key]
            if (len(tv)!=len(info)
                    or any(not math.isfinite(j) or j<-2e-12
                           or j>math.log(2.)+2e-12
                           or j+2e-12<0.5*v*v
                           for v,j in zip(tv,info))):
                raise ValueError(
                    "source Jensen-Shannon/total-variation information "
                    "inequality violated on genuine neutron source banks"
                )
        if reference is not None:
            if record["all_checkpoint_statepoint_sha256"] != reference[
                    "all_checkpoint_statepoint_sha256"]:
                raise ValueError("different source statepoints across grid resolutions")
            if record["saved_batches"] != reference["saved_batches"]:
                raise ValueError("different source checkpoints across grids")
            for metric in ("minimum_effective_distinct_position_fraction",
                           "minimum_exact_unique_position_fraction",
                           "minimum_effective_distinct_position_count",
                           "minimum_nominal_bank_population"):
                if not math.isclose(record[metric], reference[metric],
                                    rel_tol=1e-12, abs_tol=1e-10):
                    raise ValueError("source cloning changed with histogram resolution")
            # Total variation is contractive under coarse graining.
            for metric in (
                "mean_cross_seed_tv_per_checkpoint",
                "maximum_cross_seed_tv_per_checkpoint",
                "mean_within_seed_tv_per_interval",
                "pooled_mean_source_tv_per_interval",
            ):
                finer = record[metric]
                coarser = outputs[-1][metric]
                if len(finer) != len(coarser) or any(
                    f + 2e-12 < c for f, c in zip(finer, coarser)
                ):
                    raise ValueError(
                        "nested source-grid total variation decreased; "
                        "source histogram or statepoint integrity is broken"
                    )
            # JSD also satisfies the data-processing inequality on
            # nested bins. Check all time/seed comparisons, not just the
            # favorable final snapshot. There is no arbitrary source
            # convergence acceptance threshold in this exact identity.
            for metric in (
                "mean_cross_seed_jensen_shannon_nats_per_checkpoint",
                "maximum_cross_seed_jensen_shannon_nats_per_checkpoint",
                "mean_within_seed_jensen_shannon_nats_per_interval",
                "pooled_source_jensen_shannon_nats_per_interval",
            ):
                finer = record[metric]
                coarser = outputs[-1][metric]
                if (len(finer)!=len(coarser)
                        or any(f<-2e-12 or f>math.log(2.)+2e-12
                               for f in finer)
                        or any(f+2e-12<c for f,c in zip(finer,coarser))):
                    raise ValueError(
                        "nested source-grid Jensen-Shannon divergence "
                        "decreased; numerical energy/probability "
                        "histogram integrity is broken"
                    )
        else:
            reference = record
        outputs.append(record)

    min_fraction = reference["minimum_effective_distinct_position_fraction"]
    low_diversity = min_fraction < minimum_effective_fraction
    return {
        "analysis": "EXACT_SAME_OPENMC_SOURCE_BANKS_ON_NESTED_SPATIAL_GRIDS",
        "grid_resolution_audit_passed": True,
        "nested_jensen_shannon_information_refinement_integrity_passed":True,
        "jensen_shannon_total_variation_pinsker_inequality_passed":True,
        "jensen_shannon_divergence_nats_not_stationarity_probability":True,
        "source_diversity_screen_threshold_fraction": minimum_effective_fraction,
        "source_diversity_screen_status": (
            "FAIL_SOURCE_CLONING_SCREEN" if low_diversity
            else "NOT_REJECTED_BY_CLONING_SCREEN"
        ),
        "coarsest_final_cross_seed_tv": outputs[0]["final_mean_cross_seed_tv"],
        "finest_final_cross_seed_tv": outputs[-1]["final_mean_cross_seed_tv"],
        "coarsest_final_cross_seed_jensen_shannon_nats":
            outputs[0]["final_mean_cross_seed_jensen_shannon_nats"],
        "finest_final_cross_seed_jensen_shannon_nats":
            outputs[-1]["final_mean_cross_seed_jensen_shannon_nats"],
        "resolutions": outputs,
        "source_stationarity_validated": False,
        "reactor_criticality_validated": False,
        "nasa_historical_core_validated": False,
        "interpretation": (
            "Repeated measurements use the same genuine statepoint source-bank "
            "coordinates at each resolution. Monotone-TV AND monotone "
            "Jensen-Shannon (natural-log information) under grid refinement "
            "are mathematical consistency checks, not statistical evidence "
            "of stationarity. The configurable inverse-Simpson "
            "diversity fraction is only a screen for duplicated fission-site "
            "positions; even distinct sites may share ancestors. Cross-seed "
            "and adjacent-source TV are descriptive, not uncertainty-calibrated "
            "stationarity tests. Passing either screen cannot validate criticality, "
            "operating-temperature transport or historical NASA reactor physics."
        ),
    }


def audit_real_transport_summary(
    summary_file: Path, *,
    sides: tuple[int, ...] = (2, 4, 8),
    minimum_effective_fraction: float = 0.05,
) -> dict:
    """Require the genuine transport runner's statepoint paths and digests.

    The provenance fields must agree with existing files. A checksum is an
    integrity check, not independent proof of nuclear-data validity.
    """
    summary = json.loads(Path(summary_file).read_text(encoding="utf-8"))
    if summary.get("execution") != "ACTUAL_OPENMC_TRANSPORT":
        raise ValueError("requires actual OpenMC transport summary, not fixtures")
    if summary.get("assembly") != "core":
        raise ValueError("audit requires the documented core assembly")
    runs = summary.get("runs")
    batches = summary.get("saved_source_bank_batches")
    total_batches = summary.get("batches")
    if (not isinstance(runs, list) or len(runs) < 4
            or not isinstance(batches, list) or len(batches) < 3
            or not isinstance(total_batches, int) or total_batches < 3
            or batches[-1] != total_batches
            or any(not isinstance(n, int) or n < 1 for n in batches)
            or any(a >= b for a, b in zip(batches, batches[1:]))):
        raise ValueError("invalid real OpenMC checkpoint schedule or seed list")
    shape = summary.get("independent_seed_source_bank_shape_diagnostics")
    if not isinstance(shape, dict):
        raise ValueError("missing real source spatial scoring bounds")
    lower = tuple(shape.get("spatial_lower_cm", ()))
    upper = tuple(shape.get("spatial_upper_cm", ()))
    seeds = []
    trajectories = []
    digits = len(str(total_batches))
    for run in runs:
        seed = run.get("seed")
        file = Path(run.get("statepoint_path", ""))
        if (not isinstance(seed, int) or seed <= 0
                or not file.is_file()
                or run.get("statepoint_sha256") != _sha256(file)):
            raise ValueError("real statepoint file/seed/SHA-256 mismatch")
        seeds.append(seed)
        paths = [
            file.parent / f"statepoint.{batch:0{digits}d}.h5"
            for batch in batches
        ]
        if paths[-1].resolve() != file.resolve() or any(
                not path.is_file() for path in paths):
            raise ValueError("missing genuine OpenMC source checkpoint statepoint")
        trajectories.append(paths)
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate OpenMC seeds in transport summary")
    out = audit_source_grids(
        trajectories, lower_cm=lower, upper_cm=upper,
        sides=sides, minimum_effective_fraction=minimum_effective_fraction,
    )
    if [entry["seed"] for entry in summary["multi_checkpoint_source_bank_trajectory"]["runs"]] != seeds:
        raise ValueError("source checkpoint provenance disagrees with run seeds")
    out["real_transport_summary_sha256"] = _sha256(Path(summary_file))
    out["real_transport_statepoint_sha256"] = [
        run["statepoint_sha256"] for run in runs
    ]
    out["cross_sections_xml_sha256"] = summary.get("cross_sections_xml_sha256")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-effective-fraction", type=float, default=0.05)
    args = parser.parse_args()
    result = audit_real_transport_summary(
        args.summary, minimum_effective_fraction=args.min_effective_fraction
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8"
    )
    print("Source-bank multiresolution integrity: PASS")
    print("Minimum effective source-site fraction:",
          result["resolutions"][0]["minimum_effective_distinct_position_fraction"])
    print("Final cross-seed TV coarse -> fine:",
          result["coarsest_final_cross_seed_tv"], "->",
          result["finest_final_cross_seed_tv"])
    print("Source cloning:", result["source_diversity_screen_status"])
    print("Source stationarity / NASA core fidelity demonstrated: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
