"""Synthetic HDF5 regression for source-bank trajectory mathematics.

These are generated numerical fixtures, NOT transport calculations. The
production GitHub workflow separately requires actual OpenMC HDF5 files.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from .source_bank_trajectory import source_bank_trajectory
from .validate_source_bank_convergence import fake


def reject(fn, phrase: str) -> None:
    try:
        fn()
    except ValueError as error:
        assert phrase in str(error), str(error)
    else:
        raise AssertionError("invalid fission bank accepted: " + phrase)


def main() -> int:
    rng = np.random.default_rng(20261021)
    bounds = (-5., -5., -66.), (5., 5., 66.)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        trajectories = []
        for seed in (13007, 23011, 33013, 43019):
            paths = []
            for i, batch in enumerate((40, 60, 80, 100, 120)):
                xyz = rng.normal(size=(1000, 3))
                xyz[:, 2] = 6. * xyz[:, 2] + i * 2.5
                f = root / f"{seed}_{batch}.h5"
                fake(f, seed=seed, batch=batch, xyz=xyz)
                paths.append(f)
            trajectories.append(paths)
        report = source_bank_trajectory(
            trajectories, lower_cm=bounds[0], upper_cm=bounds[1]
        )
        assert report["saved_batches"] == [40, 60, 80, 100, 120]
        assert report["independent_seed_count"] == 4
        assert len(report["mean_within_seed_adjacent_total_variation_by_interval"]) == 4
        assert len(report["between_seed_mean_total_variation_by_batch"]) == 5
        assert len(report["ensemble_mean_adjacent_total_variation_by_interval"]) == 4
        assert 0 <= min(report["between_seed_mean_total_variation_by_batch"])
        assert max(report["between_seed_max_total_variation_by_batch"]) <= 1
        assert report["source_stationarity_proven"] is False
        assert report["historical_nasa_nerva_benchmark_validated"] is False
        assert all(
            r["nominal_bank_size_is_independent_sample_count"] is False
            for r in report["runs"]
        )
        assert all(
            r["minimum_distinct_position_fraction"] == 1.
            for r in report["runs"]
        )

        clone = np.repeat(
            np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]),
            500, axis=0
        )
        for f, batch in zip(trajectories[0], (40, 60, 80, 100, 120)):
            fake(f, seed=13007, batch=batch, xyz=clone)
        cloned = source_bank_trajectory(
            trajectories, lower_cm=bounds[0], upper_cm=bounds[1]
        )
        assert cloned["runs"][0]["minimum_distinct_position_fraction"] == 0.002
        assert cloned["runs"][0]["minimum_inverse_simpson_position_count"] == 2.
        assert max(cloned["runs"][0]["adjacent_snapshot_total_variation"]) == 0.

        reject(lambda: source_bank_trajectory(
            trajectories[:3], lower_cm=bounds[0], upper_cm=bounds[1]
        ), "at least four")
        reject(lambda: source_bank_trajectory(
            [trajectories[0][:2]] + trajectories[1:],
            lower_cm=bounds[0], upper_cm=bounds[1]
        ), ">=3")
        reject(lambda: source_bank_trajectory(
            trajectories[:3] + [trajectories[0]],
            lower_cm=bounds[0], upper_cm=bounds[1]
        ), "duplicate statepoint")
        reject(lambda: source_bank_trajectory(
            [trajectories[0][::-1]] + trajectories[1:],
            lower_cm=bounds[0], upper_cm=bounds[1]
        ), "strictly increasing")

        # A same-seed trajectory cannot borrow a foreign snapshot.
        reject(lambda: source_bank_trajectory(
            [[trajectories[0][0], trajectories[1][1], *trajectories[0][2:]],
             *trajectories[1:]],
            lower_cm=bounds[0], upper_cm=bounds[1]
        ), "different seeds")

        shifted = root / "23011_80_shifted.h5"
        fake(shifted, seed=23011, batch=85, xyz=rng.normal(size=(1000,3)))
        other = [list(traj) for traj in trajectories]
        other[1][2] = shifted
        reject(lambda: source_bank_trajectory(
            other, lower_cm=bounds[0], upper_cm=bounds[1]
        ), "checkpoint batches")

    print("NERVA fission-source multibatch spatial stationarity diagnostics: PASS")
    print("  synthetic structured HDF5 only; neutron transport executed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
