"""Stage 32: inspect actual final-generation fission banks in Stage 31 coupled HDF5.

A final bank is a correlated *snapshot*, not an independent-neutron sample,
not a temporal convergence study, and not a hot NERVA reactor qualification.
Only read existing executed OpenMC statepoints and their pinned Stage 31 audit.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .source_bank_convergence import read_source_bank

_EXPECTED_SEEDS = (13007, 23011, 33013, 43019)


def final_bank_spatial_statistics(histograms: np.ndarray) -> dict:
    """Descriptive spatial distances between independent *bank snapshots*.

    A bank's sites are genealogically correlated; no per-particle IID tests,
    confidence bands, or automatic source-stationarity acceptance are made.
    """
    h = np.asarray(histograms, dtype=float)
    if (h.ndim != 4 or h.shape[0] < 4 or min(h.shape[1:]) < 2
            or not np.isfinite(h).all() or np.any(h < 0)):
        raise ValueError("invalid real fission source histogram ensemble")
    totals = np.sum(h, axis=(1, 2, 3))
    if not np.allclose(totals, 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("fission bank histogram does not conserve source weight")
    distances = [
        {
            "seed_row_i": i,
            "seed_row_j": j,
            "total_variation": float(0.5 * np.sum(np.abs(h[i] - h[j]))),
        }
        for i, j in combinations(range(h.shape[0]), 2)
    ]
    mean_bank = np.mean(h, axis=0)
    axial = np.sum(h, axis=(1, 2))
    axial_sd = np.std(axial, axis=0, ddof=1)
    return {
        "bank_count": int(h.shape[0]),
        "spatial_bins_xyz": list(h.shape[1:]),
        "pairwise_final_bank_total_variation": distances,
        "mean_pairwise_final_bank_total_variation": float(np.mean([x["total_variation"] for x in distances])),
        "max_pairwise_final_bank_total_variation": float(np.max([x["total_variation"] for x in distances])),
        "mean_spatial_source_fraction": mean_bank.tolist(),
        "per_seed_final_bank_axial_fractions": axial.tolist(),
        "axial_mean_source_fraction": np.mean(axial, axis=0).tolist(),
        "axial_between_seed_sample_sd": axial_sd.tolist(),
        "source_stationarity_established": False,
        "source_site_positions_are_independent_fission_histories": False,
    }


def audit_stage31_final_fission_banks(root: str | Path) -> dict:
    """Reopen four archived original StatePoints; fail on absent native evidence."""
    base = Path(root)
    report_path = base / "actual_four_seed_coupled_heating_covariance.json"
    if not report_path.is_file():
        raise FileNotFoundError("archived Stage31 real four-seed heating ledger required")
    prior = json.loads(report_path.read_text(encoding="utf-8"))
    if (prior.get("schema") != "nuke-executed-coupled-neutron-photon-four-seed-stage31-v1"
            or prior.get("real_native_coupled_hdf5_count") != 4
            or prior.get("original_independent_openmc_rng_seeds") != list(_EXPECTED_SEEDS)
            or not prior.get("all_electron_positron_neutron_photon_energy_deposition_partitions_conserved")
            or prior.get("statistically_calibrated_fission_source_convergence") is not False
            or prior.get("real_neutron_photon_heating_hot_NERVA_physically_qualified") is not False):
        raise ValueError("unqualified or incompatible Stage31 genuine heating evidence")
    per_seed = prior.get("per_actual_seed", [])
    if len(per_seed) != 4 or [x.get("seed") for x in per_seed] != list(_EXPECTED_SEEDS):
        raise ValueError("source RNG seed order mismatches original coupled heating")
    low = tuple(prior["mesh_lower_left_cm"])
    high = tuple(prior["mesh_upper_right_cm"])
    # Restrict observations to the physically recorded energy-mesh envelope.
    # No artificial extrapolation of out-of-domain fission source sites.
    bins = (4, 4, int(prior["mesh_dimensions_xyz"][2]))
    histograms = []
    records = []
    hashes = set()
    for original in per_seed:
        seed = original["seed"]
        source = base / f"seed_{seed}" / "statepoint.65.h5"
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError("missing actual OpenMC native source HDF5: " + str(source))
        field, metadata = read_source_bank(
            source, lower_cm=low, upper_cm=high, bins=bins
        )
        expected_hash = original["native_statepoint_sha256"]
        if (metadata["seed"] != seed or metadata["completed_batch"] != 65
                or metadata["statepoint_sha256"] != expected_hash
                or expected_hash in hashes):
            raise ValueError("final fission bank not from pinned original heating StatePoint")
        hashes.add(expected_hash)
        histograms.append(field)
        records.append({
            "seed": seed,
            "source_bank": metadata,
            "heating_ev_per_monte_carlo_start": original["native_total_heating_ev_per_source"],
            "native_statepoint_sha256": expected_hash,
        })
    statistics = final_bank_spatial_statistics(np.stack(histograms, axis=0))
    return {
        "schema": "nuke-original-coupled-final-source-banks-stage32-v1",
        "evidence": "FOUR_PINNED_EXECUTED_OPENMC_COUPLED_STATEPOINT_SOURCE_BANKS",
        "seed_order": list(_EXPECTED_SEEDS),
        "actual_source_bank_count": len(records),
        "native_statepoint_hashes_match_original_heating_ledger": True,
        "source_bank_observations": records,
        "spatial_diagnostics": statistics,
        "fission_source_stationarity_qualified": False,
        "reason_stationarity_not_qualified": (
            "Only last-generation source banks from each seed were retained; "
            "no same-seed earlier temporal checkpoints or independent "
            "genealogical dependence assessment exist in this artifact."
        ),
        "historical_hot_nerva_or_reactor_watts_qualified": False,
        "limitations": [
            "Fission-bank positions are correlated by neutron genealogy and resampling.",
            "Final-generation source snapshots are not the 45-batch averaged heating source distribution.",
            "Total variation is descriptive across seeds, not a source-convergence p-value.",
            "No critical reactor, NASA fidelity or hot hydrogen properties inferred.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_stage31_final_fission_banks(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Verified actual final fission banks:", result["actual_source_bank_count"])
    print("Mean pairwise spatial TV:", result["spatial_diagnostics"]["mean_pairwise_final_bank_total_variation"])
    print("Stationary fission source qualified: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
