"""Synthetic regression of REAL-formatted OpenMC source-site neutron spectra.

Every fixture is intentionally artificial. The real C++/NNDC OpenMC
workflow separately reads actual saved neutron fission sources from HDF5.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .source_energy_audit import (
    SOURCE_ENERGY_EDGES_EV, read_source_energy,
    assess_source_energy_trajectories, audit_actual_source_energy_summary,
)


def fake(path: Path, *, seed: int, batch: int, energy: np.ndarray,
         omit_energy: bool = False, zero_weight: bool = False) -> None:
    coords = np.dtype([("x", "f8"), ("y", "f8"), ("z", "f8")])
    fields = [("r", coords), ("wgt", "f8")]
    if not omit_energy:
        fields.append(("E", "f8"))
    sites = np.zeros(len(energy), dtype=np.dtype(fields))
    sites["r"]["x"] = 1.
    sites["r"]["y"] = 2.
    sites["r"]["z"] = 3.
    sites["wgt"] = 1.0
    if zero_weight:
        sites["wgt"][0] = 0.
    if not omit_energy:
        sites["E"] = energy
    with h5py.File(path, "w") as state:
        state.attrs["filetype"] = np.bytes_("statepoint")
        state.attrs["source_present"] = 1
        state.create_dataset("run_mode", data=np.bytes_("eigenvalue"))
        state.create_dataset("seed", data=seed)
        state.create_dataset("current_batch", data=batch)
        state.create_dataset("source_bank", data=sites)


def reject(action, substring: str) -> None:
    try:
        action()
    except ValueError as error:
        assert substring in str(error), str(error)
    else:
        raise AssertionError("invalid neutron-source HDF5 accepted: " + substring)


def main() -> int:
    seeds = (13007, 23011, 33013, 43019)
    batches = (40, 60, 80, 100, 120)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        all_paths = []
        for seed in seeds:
            paths = []
            rng = np.random.default_rng(seed)
            for batch in batches:
                # 1000 synthetic fission-like energy draws, with all
                # coordinates identical to mimic a fully cloned spatial bank.
                E = np.clip(rng.lognormal(np.log(1.8e6), 0.5, 1000),
                            1e4, 25e6)
                if seed == seeds[-1] and batch == batches[-1]:
                    E[:] = 9e6  # independent-seed spectral-shift control
                path = root / f"{seed}_statepoint.{batch:03d}.h5"
                fake(path, seed=seed, batch=batch, energy=E)
                paths.append(path)
            all_paths.append(paths)
        histogram, meta = read_source_energy(all_paths[0][-1])
        assert np.isclose(histogram.sum(), 1.)
        assert meta["seed"] == seeds[0]
        assert meta["source_site_count"] == 1000
        assert len(histogram) == len(SOURCE_ENERGY_EDGES_EV) - 1
        assert 1e5 < meta["weighted_mean_energy_eV"] < 1e7

        result = assess_source_energy_trajectories(all_paths)
        assert result["independent_seed_count"] == 4
        assert result["saved_source_batches"] == list(batches)
        assert result["final_cross_seed_energy_spectrum_tv"] > 0.3
        assert len(result["mean_within_seed_adjacent_energy_spectrum_tv"]) == 4
        assert result["fission_source_spectral_stationarity_validated"] is False
        assert result["fission_source_spatial_stationarity_validated"] is False
        assert result["nasa_nerva_physical_reactor_validated"] is False
        assert all(len(run["snapshots"]) == len(batches) for run in result["runs"])
        # Stable global energy averages do NOT validate a fully cloned
        # spatial bank; there is no proof despite numerical goodness.
        stable_paths = []
        for seed in seeds:
            these = []
            for batch in batches:
                path = root / f"stable_{seed}_{batch}.h5"
                fake(path, seed=seed, batch=batch,
                     energy=np.full(300, 1.2e6))
                these.append(path)
            stable_paths.append(these)
        stable = assess_source_energy_trajectories(stable_paths)
        assert stable["final_cross_seed_energy_spectrum_tv"] == 0.
        assert all(v == 0 for v in stable[
            "mean_within_seed_adjacent_energy_spectrum_tv"])
        assert stable["fission_source_spectral_stationarity_validated"] is False

        reject(lambda: assess_source_energy_trajectories(all_paths[:3]),
               "at least four")
        reject(lambda: assess_source_energy_trajectories(
            [all_paths[0]] * 4), "duplicate statepoint")
        reject(lambda: assess_source_energy_trajectories(
            [all_paths[0][::-1]] + all_paths[1:]), "nonincreasing")
        reject(lambda: assess_source_energy_trajectories(
            [all_paths[0][:-1]] + all_paths[1:]), "same three")
        reject(lambda: read_source_energy(
            all_paths[0][0], edges_eV=(0., 1e6, 5e5)),
            "boundaries")

        bad = root / "bad_energy.h5"
        fake(bad, seed=11003, batch=120,
             energy=np.array([2e6, np.nan, 3e6]))
        reject(lambda: read_source_energy(bad), "nonphysical")
        fake(bad, seed=11003, batch=120,
             energy=np.array([2e6, 31e6, 3e6]))
        reject(lambda: read_source_energy(bad), "outside")
        fake(bad, seed=11003, batch=120,
             energy=np.array([2e6, 3e6, 4e6]), zero_weight=True)
        reject(lambda: read_source_energy(bad), "nonphysical")
        fake(bad, seed=11003, batch=120,
             energy=np.array([2e6, 3e6, 4e6]), omit_energy=True)
        reject(lambda: read_source_energy(bad), "complete")

        summary = root / "summary.json"
        summary.write_text(json.dumps({"execution": "SYNTHETIC"}))
        reject(lambda: audit_actual_source_energy_summary(summary),
               "genuine core")
        from .source_energy_audit import _digest
        summary.write_text(json.dumps({
            "execution": "ACTUAL_OPENMC_TRANSPORT",
            "assembly": "core",
            "batches": 120,
            "saved_source_bank_batches": list(batches),
            "runs": [
                {"seed": seed, "statepoint_path": str(paths[-1]),
                 "statepoint_sha256": "forged-sha"}
                for seed, paths in zip(seeds, all_paths)
            ],
        }))
        reject(lambda: audit_actual_source_energy_summary(summary),
               "SHA-256 mismatch")

    print("NERVA fission-source neutron energy drift, numerical/HDF5 QA: PASS")
    print("  artificial HDF5 inputs; real neutron transport separately executed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
