"""Synthetic HDF5 regression for nested-grid fission-source QA mathematics.

These fixtures are not Monte Carlo reactor simulations. Real transport is
exercised separately by the OpenMC CI workflow.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from .source_resolution_audit import (
    audit_source_grids,
    audit_real_transport_summary,
)
from .validate_source_bank_convergence import fake


def reject(fn, phrase: str) -> None:
    try:
        fn()
    except ValueError as error:
        assert phrase in str(error), str(error)
    else:
        raise AssertionError("invalid source physics QA accepted: " + phrase)


def main() -> int:
    lower = (-5., -5., -66.)
    upper = (5., 5., 66.)
    seeds = (13007, 23011, 33013, 43019)
    batches = (40, 60, 80, 120)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trajectories = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            path_list = []
            for batch in batches:
                xyz = rng.uniform(-4., 4., size=(800, 3))
                xyz[:, 0] = np.clip(
                    xyz[:, 0] + (seed // 10000 - 2) * 0.1,
                    -4.9, 4.9
                )
                xyz[:, 2] *= 7.
                path = root / f"{seed}_{batch}.h5"
                fake(path, seed=seed, batch=batch, xyz=xyz)
                path_list.append(path)
            trajectories.append(path_list)

        report = audit_source_grids(
            trajectories, lower_cm=lower, upper_cm=upper
        )
        resolutions = report["resolutions"]
        assert [r["bins_per_axis"] for r in resolutions] == [2, 4, 8]
        assert report["grid_resolution_audit_passed"] is True
        assert report["nested_jensen_shannon_information_refinement_integrity_passed"]
        assert report["jensen_shannon_divergence_nats_not_stationarity_probability"]
        assert 0<=report["coarsest_final_cross_seed_jensen_shannon_nats"]<=np.log(2)
        assert (report["finest_final_cross_seed_jensen_shannon_nats"]+2e-12
                >=report["coarsest_final_cross_seed_jensen_shannon_nats"])
        for coarse,fine in zip(resolutions,resolutions[1:]):
            for metric in (
                "mean_cross_seed_jensen_shannon_nats_per_checkpoint",
                "maximum_cross_seed_jensen_shannon_nats_per_checkpoint",
                "mean_within_seed_jensen_shannon_nats_per_interval",
                "pooled_source_jensen_shannon_nats_per_interval",
            ):
                assert len(coarse[metric])==len(fine[metric])
                assert all(0<=a<=np.log(2)+1e-12 and
                           b+2e-12>=a for a,b in zip(coarse[metric],fine[metric]))
        assert report["source_diversity_screen_status"] == (
            "NOT_REJECTED_BY_CLONING_SCREEN"
        )
        assert report["source_stationarity_validated"] is False
        assert report["reactor_criticality_validated"] is False
        assert report["nasa_historical_core_validated"] is False
        assert all(
            abs(r["minimum_effective_distinct_position_fraction"] - 1.) < 1e-12
            for r in resolutions
        )
        assert all(
            c["final_mean_cross_seed_tv"] <= f["final_mean_cross_seed_tv"] + 1e-12
            for c, f in zip(resolutions, resolutions[1:])
        )

        # Even when every saved batch is SPATIALLY IDENTICAL, a source bank
        # with just two cloned positions must not be certified as stationary.
        for i, seed in enumerate(seeds):
            centers = np.array([
                [0.1 + 0.8 * i, 0.2, 0.3],
                [0.2 + 0.8 * i, 0.3, 0.4]
            ])
            positions = np.repeat(centers, 400, axis=0)
            for j, path in enumerate(trajectories[i]):
                fake(path, seed=seed, batch=batches[j], xyz=positions)
        replicated = audit_source_grids(
            trajectories, lower_cm=lower, upper_cm=upper
        )
        assert replicated["source_diversity_screen_status"] == (
            "FAIL_SOURCE_CLONING_SCREEN"
        )
        assert replicated["nested_jensen_shannon_information_refinement_integrity_passed"]
        assert replicated["source_stationarity_validated"] is False
        assert replicated["resolutions"][0][
            "minimum_effective_distinct_position_count"] == 2.
        assert all(
            all(value == 0. for value in item["mean_within_seed_tv_per_interval"])
            for item in replicated["resolutions"]
        )
        assert replicated["finest_final_cross_seed_tv"] >= (
            replicated["coarsest_final_cross_seed_tv"]
        )
        assert (replicated["finest_final_cross_seed_jensen_shannon_nats"]+2e-12
                >=replicated["coarsest_final_cross_seed_jensen_shannon_nats"])
        assert all(
            all(abs(v)<2e-12 for v in item[
                "mean_within_seed_jensen_shannon_nats_per_interval"
            ]) for item in replicated["resolutions"]
        )

        reject(lambda: audit_source_grids(
            trajectories, lower_cm=lower, upper_cm=upper, sides=(4, 2)
        ), "strictly nested")
        reject(lambda: audit_source_grids(
            trajectories, lower_cm=lower, upper_cm=upper, sides=(2, 3, 6)
        ), "strictly nested")
        reject(lambda: audit_source_grids(
            trajectories, lower_cm=lower, upper_cm=upper,
            minimum_effective_fraction=1.01
        ), "minimum effective")
        reject(lambda: audit_source_grids(
            trajectories[:3], lower_cm=lower, upper_cm=upper
        ), "at least four")
        reject(lambda: audit_source_grids(
            [trajectories[0]] * 4, lower_cm=lower, upper_cm=upper
        ), "duplicate statepoint")
        reject(lambda: audit_source_grids(
            trajectories, lower_cm=(-0.1, -0.1, -0.1),
            upper_cm=(0.1, 0.1, 0.1)
        ), "outside")

        # A test fixture summary cannot impersonate actual executed OpenMC.
        summary = root / "synthetic_summary.json"
        summary.write_text(json.dumps({"execution": "SYNTHETIC"}))
        reject(lambda: audit_real_transport_summary(summary),
               "requires actual OpenMC transport")

        # An invented or altered actual-statepoint digest must fail closed.
        summary.write_text(json.dumps({
            "execution": "ACTUAL_OPENMC_TRANSPORT",
            "assembly": "core",
            "runs": [{
                "seed": seed, "statepoint_path": str(traj[-1]),
                "statepoint_sha256": "not_a_real_hash"
            } for seed, traj in zip(seeds, trajectories)],
            "saved_source_bank_batches": list(batches),
            "batches": 120,
            "independent_seed_source_bank_shape_diagnostics": {
                "spatial_lower_cm": lower, "spatial_upper_cm": upper
            },
            "multi_checkpoint_source_bank_trajectory": {
                "runs": [{"seed": seed} for seed in seeds]
            },
        }))
        reject(lambda: audit_real_transport_summary(summary),
               "statepoint file/seed/SHA-256 mismatch")

    print("NERVA nested-grid fission-source QA: PASS")
    print("  HDF5 inputs generated synthetically; real OpenMC transport: NO")
    print("  cloned banks rejected even when adjacent source TV equals zero")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
