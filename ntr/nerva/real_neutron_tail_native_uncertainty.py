"""Read native OpenMC batch-to-batch uncertainty on rare neutron energy tails.

The genuine cold fixed-source OpenMC experiment scores underflow [0,1e-5)
eV and overflow (20 MeV, 10 GeV] in independent material x energy tallies.
This read-only audit distinguishes (a) arithmetic closure of the first
moments from (b) the actual native uncertainty of those scored quantities.
A source/geometry/XS configuration is NOT modified.

OpenMC statepoint tally results contain cumulative raw SUM and SUM-OF-SQUARES
over active batches. They are not means or variances. Each fixed-source batch
is a fresh set of prescribed source histories; their native batch SEM is only
a Monte Carlo statistical diagnostic, not cross-section/model uncertainty.
Material totals cannot be assigned a SEM by adding their marginal SEMs:
the same particles create material/energy-bin covariance, which these raw
marginal tally results do not store. Four independent-run statistics live in
real_neutron_tail_replica_statistics.py.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .real_neutron_energy_tail_attribution import (
    SCORES, TAIL_EDGES_EV, inspect_real_window_tail,
)
from .real_core_neutron_energy_closure import _tally, _filters

TAIL_SLICES = ((0, "below_1e_minus_5_eV"), (2, "above_20_MeV"))


def native_batch_moments(raw: np.ndarray, batches: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-batch means and sample SEM from native OpenMC SUM and SUMSQ.

    Input must have shape (filter bins, scores, [sum, sum_sq]).
    NO assumption of independent tally scores or material bins is made.
    """
    moments = np.asarray(raw, dtype=np.float64)
    if (type(batches) is not int or batches < 3 or moments.ndim != 3
            or moments.shape[-1] != 2 or moments.shape[0] == 0
            or moments.shape[1] == 0 or not np.isfinite(moments).all()):
        raise ValueError("invalid native OpenMC scored batch moments")
    s, ss = moments[:, :, 0], moments[:, :, 1]
    if np.any(ss < 0):
        raise ValueError("negative native squared neutron tally moments")
    n = float(batches)
    residual = ss - s*s/n
    # Allow only floating-point roundoff in nonnegative Cauchy-Schwarz
    # sample-variance identity. Do not convert a material discrepancy into
    # physical neutron tail evidence.
    tolerance = 1e-12 * np.maximum(
        np.maximum(np.abs(ss), np.abs(s*s/n)), 1e-30
    )
    if np.any(residual < -tolerance):
        raise ValueError("native neutron tally second moments violate variance identity")
    sem = np.sqrt(np.maximum(residual, 0.0)/(n*(n-1.0)))
    mean = s/n
    if not np.isfinite(sem).all() or not np.isfinite(mean).all():
        raise ValueError("nonfinite native neutron batch uncertainty")
    return mean, sem


def read_actual_native_tail_uncertainty(path: Path, daughters: bool) -> dict:
    """Audit real OpenMC HDF5 with no synthetic or unscored band extrapolation."""
    path = Path(path)
    verified = inspect_real_window_tail(path, daughters)
    with h5py.File(path, "r") as state:
        if (bytes(state.attrs["filetype"]).rstrip(b"\x00") != b"statepoint"
                or bytes(state["run_mode"][()]).rstrip(b"\x00") != b"fixed source"
                or tuple(int(v) for v in state.attrs["openmc_version"]) != (0,15,3)
                or int(state.attrs["photon_transport"]) != 0):
            raise ValueError("requires actual neutron-only OpenMC 0.15.3 fixed-source HDF5")
        n = int(state["current_batch"][()])
        if n < 3:
            raise ValueError("too few actual fixed-source batches")
        tallies = state["tallies"]
        tally, labels, reference_means = _tally(
            tallies, "nerva_energy_window_tail_attribution", n
        )
        filters = _filters(tallies, tally, ("material", "energy"))
        material_ids = np.asarray(filters[0]["bins"][()], dtype=int).reshape(-1)
        edges = np.asarray(filters[1]["bins"][()], dtype=float).reshape(-1)
        if (not np.array_equal(edges, np.asarray(TAIL_EDGES_EV))
                or material_ids.size != verified["material_count"]
                or len(np.unique(material_ids)) != material_ids.size
                or any(score not in labels for score in SCORES)):
            raise ValueError("native material/energy-bin provenance mismatch")
        # Independent energy-only tally covers the whole scored domain.
        # Its native SUMSQ captures covariance among material regions.
        global_tally, global_labels, global_reference = _tally(
            tallies, "nerva_global_energy_window_tail", n
        )
        energy_filter = _filters(tallies, global_tally, ("energy",))[0]
        global_edges = np.asarray(energy_filter["bins"][()], dtype=float).reshape(-1)
        if (not np.array_equal(global_edges, np.asarray(TAIL_EDGES_EV))
                or any(score not in global_labels for score in SCORES)):
            raise ValueError("whole-core native incident-energy scoring provenance changed")
        global_raw = np.asarray(global_tally["results"][()], dtype=np.float64)
        global_mean_raw, global_sem_raw = native_batch_moments(global_raw, n)
        if (global_mean_raw.shape != (3, len(global_labels))
                or not np.allclose(global_mean_raw, global_reference, rtol=1e-13, atol=1e-20)):
            raise ValueError("whole-core native first/second neutron moments incomplete")
        global_indices = [global_labels.index(score) for score in SCORES]
        global_mean = global_mean_raw[:, global_indices]
        global_sem = global_sem_raw[:, global_indices]
        if np.any(global_mean[:, :4] < -1e-13):
            raise ValueError("negative directly scored whole-core neutron reactions")
        raw = np.asarray(tally["results"][()], dtype=np.float64)
        mean, sem = native_batch_moments(raw, n)
        if (reference_means.shape != mean.shape
                or not np.allclose(reference_means, mean, rtol=1e-13, atol=1e-20)
                or mean.shape[0] != material_ids.size * 3):
            raise ValueError("raw native OpenMC first moment disagrees with physical tally")
        indices = [labels.index(s) for s in SCORES]
        mean = mean[:, indices].reshape(material_ids.size, 3, len(SCORES))
        sem = sem[:, indices].reshape(material_ids.size, 3, len(SCORES))
        # Heating-local may be signed. Other reactions/flux must be >=0.
        scale = np.maximum(1.0, np.max(np.abs(mean), axis=(0,1)))
        if np.any(mean[:, :, :4] < -1e-13*scale[:4]):
            raise ValueError("negative native neutron tracklength/reaction mean")
        # Rates, unlike tracklength flux in void, should be recovered
        # by summing the independently scored material reaction tallies.
        material_total = mean.sum(axis=0, dtype=np.float64)
        reaction_delta = global_mean[:, 1:] - material_total[:, 1:]
        reaction_scale = np.maximum(1.0, np.maximum(
            np.abs(global_mean[:, 1:]), np.abs(material_total[:, 1:])
        ))
        if np.any(np.abs(reaction_delta) > 1e-9 * reaction_scale):
            raise ValueError("whole-core native energy-tail reaction not recovered by material tally")
        global_direct = {}
        for band, band_name in TAIL_SLICES:
            global_direct[band_name] = {
                score: {
                    "mean_per_prescribed_source": float(global_mean[band, k]),
                    "native_whole_domain_batch_sem_per_source": float(global_sem[band, k]),
                    "zero_observations_do_not_bound_rare_event_rate": True,
                } for k, score in enumerate(SCORES)
            }
        observed_low = float(np.sum(mean[:, 0, 0], dtype=np.float64))
        observed_high = float(np.sum(mean[:, 2, 0], dtype=np.float64))
        previous = verified["explicit_tail_score_per_source"]
        if (not math.isclose(observed_low, previous["below_lower_bound"], abs_tol=1e-12, rel_tol=1e-10)
                or not math.isclose(observed_high, previous["above_upper_bound"], abs_tol=1e-12, rel_tol=1e-10)):
            raise ValueError("native energy-tail means mismatch prior independent material tally")

        records = []
        zero_bins = 0
        for j, mid in enumerate(material_ids):
            for band, band_name in TAIL_SLICES:
                for k, score in enumerate(SCORES):
                    m = float(mean[j, band, k])
                    se = float(sem[j, band, k])
                    if m == 0.0 and se == 0.0:
                        zero_bins += 1
                        continue
                    records.append({
                        "material_id": int(mid),
                        "energy_tail": band_name,
                        "score": score,
                        "mean_per_prescribed_source": m,
                        "openmc_native_batch_sem_per_source": se,
                        "native_batch_relative_sem": se / abs(m) if m else None,
                        "numerical_zero_confidence_claimed": False,
                    })
        if len(records) + zero_bins != material_ids.size * len(TAIL_SLICES) * len(SCORES):
            raise ValueError("incomplete native neutron energy-tail variance inventory")
        return {
            "seed": verified["seed"],
            "fission_daughters_transported": daughters,
            "statepoint_sha256": verified["statepoint_sha256"],
            "geometry_sha256": verified["geometry_sha256"],
            "materials_sha256": verified["materials_sha256"],
            "tallies_sha256": verified["tallies_sha256"],
            "native_realizations": n,
            "physical_material_count": int(material_ids.size),
            "tail_energy_bounds_eV": list(TAIL_EDGES_EV),
            "native_per_material_energy_score_nonzero_records": records,
            "unobserved_zero_moment_material_energy_scores": zero_bins,
            "native_batch_sem_available_only_per_material_energy_score": True,
            "whole_domain_tail_uncertainty_scored_directly_with_covariance": True,
            "direct_whole_domain_energy_tail_native_moments": global_direct,
            "largest_abs_global_vs_material_reaction_moment_error": float(
                np.max(np.abs(reaction_delta))),
            "global_minus_material_flux_by_energy_band": (
                global_mean[:,0]-material_total[:,0]
            ).tolist(),
            "global_flux_may_include_void_tracklength": True,
            "material_summed_tail_sem_unavailable_without_covariance": True,
            "mean_total_sub_1e_minus_5_eV_flux_per_source": observed_low,
            "mean_total_above_20_MeV_flux_per_source": observed_high,
            "hot_reactor_physics_or_heating_qualified": False,
        }


def audit_eight_actual_native_tail_uncertainties(
    root: Path, *, seeds=(13007,23011,33013,43019), batches=30,
) -> dict:
    if (len(seeds) != 4 or len(set(seeds)) != 4
            or any(type(seed) is not int or seed <= 0 for seed in seeds)
            or type(batches) is not int or batches < 3):
        raise ValueError("four independent actual fixed-source seeds required")
    rows, hashes = [], set()
    identity = None
    for seed in seeds:
        pair = {"seed": seed}
        for daughters, mode, folder in (
            (False, "off", "suppressed_fission_control"),
            (True, "on", "fission_daughters_transported"),
        ):
            sp = Path(root)/folder/f"seed_{seed}"/f"statepoint.{batches}.h5"
            row = read_actual_native_tail_uncertainty(sp, daughters)
            if row["seed"] != seed or row["native_realizations"] != batches:
                raise ValueError("OpenMC native tally source seed/batch mismatch")
            if row["statepoint_sha256"] in hashes:
                raise ValueError("duplicate physically executed OpenMC HDF5 statepoint")
            hashes.add(row["statepoint_sha256"])
            current = (
                row["geometry_sha256"], row["materials_sha256"],
                row["tallies_sha256"], row["physical_material_count"],
            )
            if identity is None:
                identity = current
            elif identity != current:
                raise ValueError("physical reactor or neutron tally differs across replicas")
            pair[mode] = row
        rows.append(pair)
    return {
        "audit": "GENUINE_OPENMC_FIXED_SOURCE_NATIVE_NEUTRON_TAIL_BATCH_UNCERTAINTY",
        "actual_neutron_statepoints": len(hashes),
        "independent_seed_pairs": len(seeds),
        "native_batch_statistics_per_material_energy_score": True,
        "summed_material_tail_uncertainties_not_invented": True,
        "direct_whole_domain_tail_sem_available": True,
        "global_energy_tail_reaction_partition_tested_independently": True,
        "first_moments_agree_with_independent_energy_partition": True,
        "per_seed": rows,
        "criticality_validated": False,
        "physical_nuclear_heating_validated": False,
        "hot_temperature_cross_sections_validated": False,
        "nasa_nerva_historical_core_validated": False,
        "interpretation": (
            "Real OpenMC raw cumulative per-batch SUM/SUMSQ quantify native "
            "Monte Carlo uncertainty for each separately scored material and "
            "energy-tail reaction. Zero scored samples are not proof that the "
            "physical rare-event rate is zero. Independent material/energy "
            "bins share particle histories: adding marginal SEMs to obtain "
            "whole-core tail uncertainties is not statistically valid without "
            "their covariance. Correlated histories, nuclear-data, fuel and "
            "operating-temperature fidelity remain separate limitations."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = audit_eight_actual_native_tail_uncertainties(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    active = sum(len(r[mode]["native_per_material_energy_score_nonzero_records"])
                 for r in result["per_seed"] for mode in ("off","on"))
    print("REAL OpenMC 8-statepoint native neutron-tail first and second moments: PASS")
    print("Nonzero material/energy-score estimates:", active)
    print("Core-level tail SEM unavailable without material-bin covariance")
    print("Hot NERVA core, thermal KERMA and criticality: NOT QUALIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
