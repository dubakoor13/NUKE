"""Adversarial numerical tests; synthetic HDF5 is NOT reactor validation."""
from __future__ import annotations

import math
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .core_mt901_reaction_exposure import (
    SCORES, evaluate_reaction_exposure, read_real_material_reactions,
)


def reject(fn, text: str) -> None:
    try:
        fn()
    except ValueError as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError("invalid neutron material reaction evidence accepted")


def fake_hdf5(path: Path, *, bad_second: bool = False) -> None:
    n = 90
    # Deliberately include an unplaced material with exactly zero tallies.
    mean = np.array([
        [10., 0.45, 0.06, 0.14],
        [4., 0.04, 0., 0.],
        [2., 0.02, 0., 0.],
        [0., 0., 0., 0.],
    ])
    scored = np.zeros((4, 4, 2), dtype=float)
    scored[:, :, 0] = mean*n
    scored[:, :, 1] = (mean**2 + 1e-2)*n
    if bad_second:
        scored[0, 1, 1] = 0.
    with h5py.File(path, "w") as state:
        state.attrs["filetype"] = np.bytes_("statepoint")
        state.attrs["photon_transport"] = 0
        state.attrs["openmc_version"] = [0, 15, 3]
        state.create_dataset("run_mode", data=np.bytes_("eigenvalue"))
        state.create_dataset("seed", data=13007)
        state.create_dataset("current_batch", data=120)
        state.create_dataset("n_inactive", data=30)
        tallies = state.create_group("tallies")
        filters = tallies.create_group("filters")
        f = filters.create_group("filter 1")
        f.create_dataset("type", data=np.bytes_("material"))
        f.create_dataset("bins", data=[19, 20, 21, 22])
        t = tallies.create_group("tally 13")
        t.create_dataset("name", data=np.bytes_("nerva_material_transport"))
        t.create_dataset("estimator", data=np.bytes_("tracklength"))
        t.create_dataset("filters", data=[1])
        t.create_dataset("n_filters", data=1)
        t.create_dataset("n_realizations", data=n)
        t.create_dataset("nuclides", data=[np.bytes_("total")])
        t.create_dataset("score_bins", data=[np.bytes_(x) for x in SCORES])
        t.create_dataset("results", data=scored)


def main() -> int:
    ids = [19, 20, 21, 22]
    val = np.array([[5.0, .7, .2, .48],
                    [3.0, .1, 0., 0.],
                    [1.0, .02, 0., 0.],
                    [0., 0., 0., 0.]])
    ready = {19: False, 20: False, 21: True, 22: False}
    got = evaluate_reaction_exposure(
        ids, val, ready, reachable_ids={19, 20, 21}
    )
    score = got["scores"]["absorption"]
    assert math.isclose(score["total_per_fission_source"], .82)
    assert math.isclose(score["in_materials_without_full_mt901_per_source"], .8)
    assert math.isclose(score["fraction_of_scored_activity_without_full_mt901"], .8/.82)
    assert got["mt901_missing_material_ids"] == [19,20]
    assert got["thermal_power_or_missing_heat_fraction_computed"] is False

    ready_all_missing = {i: False for i in ids}
    all_missing = evaluate_reaction_exposure(
        ids, val, ready_all_missing, reachable_ids={19,20,21}
    )
    assert all(
        x["fraction_of_scored_activity_without_full_mt901"] == 1.
        for x in all_missing["scores"].values()
        if x["total_per_fission_source"] > 0
    )

    zero = evaluate_reaction_exposure(
        ids, np.zeros((4,4)), ready_all_missing, reachable_ids={19,20,21}
    )
    assert zero["scores"]["nu-fission"]["fraction_of_scored_activity_without_full_mt901"] is None
    assert zero["scores"]["nu-fission"]["unobserved_reaction_is_not_a_zero_physical_rate"]
    corrupted = val.copy()
    corrupted[3,0] = .0001
    reject(lambda: evaluate_reaction_exposure(
        ids, corrupted, ready, reachable_ids={19,20,21}
    ), "unreachable")
    corrupted = val.copy()
    corrupted[0,1] = -.1
    reject(lambda: evaluate_reaction_exposure(
        ids, corrupted, ready, reachable_ids={19,20,21}
    ), "invalid genuine")
    reject(lambda: evaluate_reaction_exposure(
        ids, val, {19:False,20:False,21:True}, reachable_ids={19,20,21}
    ), "invalid genuine")

    with TemporaryDirectory() as td:
        file = Path(td)/"statepoint.120.h5"
        fake_hdf5(file)
        scored = read_real_material_reactions(file, set(ids))
        assert scored["seed"] == 13007 and scored["active_realizations"] == 90
        assert np.allclose(scored["values"], [
            [10.,.45,.06,.14], [4.,.04,0.,0.],
            [2.,.02,0.,0.], [0.,0.,0.,0.]
        ])
        fake_hdf5(file, bad_second=True)
        reject(lambda: read_real_material_reactions(file, set(ids)),
               "second-moment identity")
        fake_hdf5(file)
        reject(lambda: read_real_material_reactions(file, {19,20,21,25}),
               "material inventory")
    print("Reactor material MT901 reaction-exposure regressions: PASS (synthetic)")
    print("Actual OpenMC C++ HDF5 physical evidence required in separate CI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
