"""Stage35: native OpenMC 3D deposited-heating stability across active windows.

Reconstruct 3D coupled neutron/photon energy-deposition means from 12 executed
OpenMC 0.15.3 statepoints (four seeds, batches 30/45/65). Analyze the
three *disjoint* active windows; measure resolution-sensitive spatial shape
changes without invoking new transport, reactor tuning, or false IID errors.

Every reported density is eV / Monte Carlo source / cm^3, NOT watts/cm^3.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from .real_coupled_disjoint_heating import (
    _native_tallies,
    audit_real_disjoint_heating,
    disjoint_raw_moments,
)
from .real_coupled_source_temporal import CHECKPOINTS, SEEDS

_FINE = (12, 12, 10)
_SCALES = ((12, 12, 10), (6, 6, 10), (3, 3, 10), (1, 1, 10))
_WINDOWS = ((21, 30), (31, 45), (46, 65))


def conservative_block_sum(field: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Conservative 3D block integration; never interpolate or renormalize."""
    a = np.asarray(field, dtype=float)
    target = tuple(shape)
    if (a.ndim != 3 or len(target) != 3
            or any(not isinstance(x, int) or isinstance(x, bool) or x <= 0 for x in target)
            or any(size % step for size, step in zip(a.shape, target))
            or not np.isfinite(a).all() or np.any(a < 0)):
        raise ValueError("invalid original 3D mesh for conservative block aggregation")
    sizes = tuple(size // count for size, count in zip(a.shape, target))
    out = a.reshape(target[0], sizes[0],
                    target[1], sizes[1],
                    target[2], sizes[2]).sum(axis=(1, 3, 5))
    if not math.isclose(float(out.sum()), float(a.sum()), rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("3D block aggregation does not conserve deposited energy")
    return out


def total_variation(a: np.ndarray, b: np.ndarray) -> float:
    """Total variation of two *probability* distributions (not raw heating)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if (x.shape != y.shape or x.ndim < 1 or x.size < 1
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or np.any(x < 0) or np.any(y < 0)
            or not np.isclose(x.sum(), 1.0, atol=1e-10, rtol=0.)
            or not np.isclose(y.sum(), 1.0, atol=1e-10, rtol=0.)):
        raise ValueError("spatial heating TV requires normalized valid probability arrays")
    return float(0.5 * np.sum(np.abs(x - y)))


def measure_3d_heating_stability(values: np.ndarray) -> dict:
    """3D spatial changes, multiscale TV and paired independent-seed variation.

    Four seeds and correlated fission ancestors cannot define a qualified
    reactor heating confidence bound or prove fission-source stationarity.
    """
    energy = np.asarray(values, dtype=float)
    if (energy.shape != (4, 3, *_FINE)
            or not np.isfinite(energy).all() or np.any(energy < 0)):
        raise ValueError("four real seeded 3D heating window fields required")
    totals = energy.sum(axis=(2, 3, 4))
    if np.any(totals <= 0) or not np.isfinite(totals).all():
        raise ValueError("zero/nonfinite physical coupled deposited heat")
    fractions = energy / totals[:, :, None, None, None]
    measures = []
    for scale in _SCALES:
        reduced = np.stack([
            np.stack([conservative_block_sum(fractions[i, w], scale)
                      for w in range(3)])
            for i in range(4)
        ])
        by_seed = []
        for i, seed in enumerate(SEEDS):
            t = [total_variation(reduced[i, u], reduced[i, v])
                 for u, v in ((0, 1), (1, 2), (0, 2))]
            by_seed.append({
                "seed": seed,
                "fraction_TV_early_middle": t[0],
                "fraction_TV_middle_late": t[1],
                "fraction_TV_early_late": t[2],
            })
        across = []
        for window in range(3):
            pairs = [total_variation(reduced[a, window], reduced[b, window])
                     for a, b in combinations(range(4), 2)]
            across.append({
                "active_batches": list(_WINDOWS[window]),
                "mean_between_seed_TV": float(np.mean(pairs)),
                "max_between_seed_TV": float(max(pairs)),
                "all_six_pairwise_between_seed_TV": pairs,
            })
        measures.append({
            "spatial_bins_xyz": list(scale),
            "per_seed_temporal": by_seed,
            "mean_early_late_TV": float(np.mean(
                [row["fraction_TV_early_late"] for row in by_seed]
            )),
            "between_seed_by_window": across,
        })
    for fine, coarse in zip(measures, measures[1:]):
        for a, b in zip(fine["per_seed_temporal"], coarse["per_seed_temporal"]):
            for name in ("fraction_TV_early_middle", "fraction_TV_middle_late",
                         "fraction_TV_early_late"):
                if b[name] > a[name] + 1e-12:
                    raise ValueError("physical heating TV increased after conservative coarsening")
        for a, b in zip(fine["between_seed_by_window"], coarse["between_seed_by_window"]):
            for tv_a, tv_b in zip(a["all_six_pairwise_between_seed_TV"],
                                  b["all_six_pairwise_between_seed_TV"]):
                if tv_b > tv_a + 1e-12:
                    raise ValueError("between-seed heating TV increased after coarsening")
    mean_frac = np.mean(fractions, axis=0)
    sd = np.std(fractions, axis=0, ddof=1)
    diagnostics = []
    for window in range(3):
        average = mean_frac[window]
        active = average > 0.01 * np.max(average)
        relative_sd = sd[window][active] / average[active]
        diagnostics.append({
            "active_batches": list(_WINDOWS[window]),
            "spatially_active_voxel_count": int(np.count_nonzero(active)),
            "active_voxel_relative_across_seed_SD_p90_descriptive":
                float(np.percentile(relative_sd, 90)) if relative_sd.size else None,
            "largest_voxel_fraction_across_seed_average": float(np.max(average)),
            "largest_voxel_index_xyz_of_seed_average":
                list(np.unravel_index(int(np.argmax(average)), _FINE)),
            "summed_seed_mean_heat_fraction": float(np.sum(average)),
        })
    return {
        "source_normalized_total_heating_by_seed_and_window_ev": totals.tolist(),
        "scale_sensitivity": measures,
        "across_seed_hotspot_and_variability": diagnostics,
        "axial_mean_early_late_TV": measures[-1]["mean_early_late_TV"],
        "native_independent_source_stationarity_qualified": False,
        "spatial_voxel_heating_uncertainty_calibrated": False,
        "hot_nerva_reactor_power_density_in_watts_qualified": False,
    }


def audit_actual_3d_heating(root: str | Path, stage34_json: str | Path) -> tuple[dict, dict]:
    """Read native 12-statepoint evidence and archived independently checked heat."""
    base = Path(root)
    archive = Path(stage34_json)
    if not archive.is_file():
        raise FileNotFoundError("accepted Stage34 original HDF5 window audit JSON required")
    independent = json.loads(archive.read_text(encoding="utf-8"))
    if (independent.get("schema") != "nuke-actual-coupled-disjoint-active-heating-stage34-v1"
            or independent.get("genuine_openmc_statepoints_reverified") != 12
            or independent.get("all_raw_moment_telescope_and_final_heat_identical") is not True
            or independent.get("fission_source_stationarity_qualified") is not False
            or independent.get("real_hot_nerva_physical_watts_qualified") is not False):
        raise ValueError("original accepted disjoint heating evidence invalid")
    recomputed = audit_real_disjoint_heating(base)
    signature = recomputed["native_filter_identity"]
    if (signature.get("mesh_dimensions_xyz") != list(_FINE)
            or len(independent.get("per_seed", [])) != 4):
        raise ValueError("original native 3D heating identity changed")
    extent = np.asarray(signature["mesh_upper_right_cm"], dtype=float) - np.asarray(
        signature["mesh_lower_left_cm"], dtype=float)
    if extent.shape != (3,) or not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError("invalid physically scored OpenMC volume")
    voxel_volume = float(np.prod(extent / np.asarray(_FINE, dtype=float)))
    if not math.isfinite(voxel_volume) or voxel_volume <= 0:
        raise ValueError("invalid native OpenMC voxel volume")

    spatial = []
    for i, seed in enumerate(SEEDS):
        snapshots = [_native_tallies(base / f"seed_{seed}" / f"statepoint.{batch}.h5")
                     for batch in CHECKPOINTS]
        if (any(r["identity"]["seed"] != seed for r in snapshots)
                or any(r["signature"] != signature for r in snapshots)
                or [r["identity"]["batch"] for r in snapshots] != list(CHECKPOINTS)):
            raise ValueError("native heating physical mesh/seed/estimator identity changed")
        windows = disjoint_raw_moments(
            [r["raw"]["mesh"]["sum"] for r in snapshots],
            [r["raw"]["mesh"]["sum_sq"] for r in snapshots],
            [r["identity"]["batch"] - 20 for r in snapshots],
            label="actual OpenMC 3D deposited heating",
        )
        fields = []
        for j, item in enumerate(windows):
            mean = np.asarray(item["mean"], dtype=float)
            if mean.shape != _FINE or not np.isfinite(mean).all() or np.any(mean < -1e-8):
                raise ValueError("invalid native OpenMC per-voxel deposited heating")
            mean = np.maximum(mean, 0.)
            original = independent["per_seed"][i]["disjoint_heating_windows"][j]
            matched = recomputed["per_seed"][i]["disjoint_heating_windows"][j]
            for reference in (original, matched):
                if (original["active_batch_window"] != list(_WINDOWS[j])
                        or not math.isclose(float(mean.sum()),
                            reference["heating_ev_per_monte_carlo_start"],
                            rel_tol=1e-10, abs_tol=1e-7)
                        or not np.allclose(mean.sum(axis=(0, 1)),
                            reference["axial_heating_ev_per_source"],
                            rtol=1e-10, atol=1e-7)):
                    raise ValueError("actual 3D heating conflicts with original Stage34 physical evidence")
            fields.append(mean)
        if independent["per_seed"][i]["seed"] != seed:
            raise ValueError("native archived independent RNG seed order mismatch")
        spatial.append(np.stack(fields))
    arrays = np.stack(spatial)
    metrics = measure_3d_heating_stability(arrays)
    expected = float(independent["mean_observed_axial_heating_fraction_first_last_TV"])
    if not math.isclose(metrics["axial_mean_early_late_TV"], expected,
                        rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("coarse integrated axial 3D heating failed Stage34 agreement")
    fractions = arrays / arrays.sum(axis=(2, 3, 4))[:, :, None, None, None]
    return {
        "schema": "nuke-real-coupled-3d-window-heating-stability-stage35-v1",
        "evidence": "12_ORIGINAL_OPENMC_0153_NATIVE_NEUTRON_PHOTON_TALLY_HDF5",
        "original_native_hdf5_checked": 12,
        "four_rng_seeds": list(SEEDS),
        "three_disjoint_active_windows": [list(w) for w in _WINDOWS],
        "mesh_lower_left_cm": signature["mesh_lower_left_cm"],
        "mesh_upper_right_cm": signature["mesh_upper_right_cm"],
        "mesh_dimensions_xyz": list(_FINE),
        "physical_heating_mesh_voxel_volume_cm3": voxel_volume,
        "independent_accepted_stage34_axial_heating_verified": True,
        "spatial_heating_statistics": metrics,
        "absolute_physical_reactor_watts_known": False,
        "source_stationarity_and_spatial_heating_uncertainty_qualified": False,
        "limitations": [
            "Cold, far-subcritical uncalibrated NERVA-derived conceptual core.",
            "Real heating energy density is eV/Monte Carlo starting neutron/cm^3, NOT W/cm^3.",
            "Stage34 nonoverlapping active windows share correlated fission-source genealogy.",
            "Fine 3D voxel spatial variation is estimator/mesh/source dependent.",
            "Four independent RNG seeds cannot calibrate voxelwise confidence coverage.",
            "Source-bank snapshots and active-batch heating are not identical observables.",
        ],
    }, {
        "deposited_energy_ev_per_mc_start": arrays,
        "deposited_energy_density_ev_per_mc_start_per_cm3": arrays / voxel_volume,
        "spatial_heating_probability": fractions,
        "mesh_lower_left_cm": np.asarray(signature["mesh_lower_left_cm"]),
        "mesh_upper_right_cm": np.asarray(signature["mesh_upper_right_cm"]),
        "rng_seeds": np.asarray(SEEDS),
        "active_windows": np.asarray(_WINDOWS),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage34-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report, arrays = audit_actual_3d_heating(args.root, args.stage34_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    npz = args.output.with_suffix(".npz")
    np.savez_compressed(npz, **arrays)
    report["exact_3d_fields_npz_name"] = npz.name
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                           encoding="utf-8")
    measures = report["spatial_heating_statistics"]["scale_sensitivity"]
    print("REAL native OpenMC HDF5:", report["original_native_hdf5_checked"])
    for level in measures:
        print("3D resolution", level["spatial_bins_xyz"],
              "mean early->late deposited heat TV:", level["mean_early_late_TV"])
    print("Stationary physically qualified hot NERVA reactor: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
