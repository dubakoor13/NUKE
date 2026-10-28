"""Stage 33: genuine coupled OpenMC temporal fission-source audit.

Four independently seeded executions each save real OpenMC StatePoints at
batches 30, 45, and 65. Measurements are from fission-source bank snapshots,
not independent fission histories or certified operating reactor physics.
No design, enrichment, control, cross section or temperature tuning.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .source_bank_convergence import read_source_bank

SEEDS = (13007, 23011, 33013, 43019)
CHECKPOINTS = (30, 45, 65)
FINE_BINS = (4, 4, 10)


def _digest(file: Path) -> str:
    h = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hdf5_identity(path: Path, *, expected_particles: int = 3000) -> dict:
    """Raw native OpenMC eigenvalue history, with explicit photon-mode checks."""
    try:
        with h5py.File(path, "r") as h:
            if (bytes(h.attrs["filetype"]).rstrip(b"\0") != b"statepoint"
                    or bytes(h["run_mode"][()]).rstrip(b"\0") != b"eigenvalue"
                    or int(h.attrs["photon_transport"]) != 1
                    or int(h.attrs.get("source_present", 0)) != 1):
                raise ValueError("requires native coupled OpenMC eigenvalue source statepoint")
            seed = int(h["seed"][()])
            batch = int(h["current_batch"][()])
            total = int(h["n_batches"][()])
            inactive = int(h["n_inactive"][()])
            particles = int(h["n_particles"][()])
            generations = int(h["generations_per_batch"][()])
            realizations = int(h["n_realizations"][()])
            k = np.asarray(h["k_generation"][()], dtype=float)
            entropy = np.asarray(h["entropy"][()], dtype=float)
    except (OSError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError("missing native OpenMC coupled fission-source history") from exc
    if (seed not in SEEDS or batch not in CHECKPOINTS or total != 65
            or inactive != 20 or particles != expected_particles or generations != 1
            or realizations != batch - inactive or k.shape != (batch,)
            or entropy.shape != (batch,) or not np.isfinite(k).all()
            or not np.isfinite(entropy).all()
            or np.any(k < 0) or np.any(entropy < 0)):
        raise ValueError("incorrect native coupled OpenMC batch, source, k or entropy identity")
    return {
        "seed": seed, "batch": batch, "n_batches": total,
        "inactive": inactive, "n_particles": particles,
        "generations_per_batch": generations,
        "realizations": realizations, "k": k, "entropy": entropy,
    }


def verify_native_checkpoint_prefixes(records: list[dict]) -> dict:
    """Require the saved k/entropy prefixes to describe ONE exact MC history."""
    if len(records) != 3 or [r["batch"] for r in records] != list(CHECKPOINTS):
        raise ValueError("expected exact three original OpenMC temporal checkpoints")
    if len({r["seed"] for r in records}) != 1:
        raise ValueError("different RNG seeds cannot be same history")
    for earlier, later in zip(records, records[1:]):
        n = earlier["batch"]
        if (not np.array_equal(earlier["k"], later["k"][:n])
                or not np.array_equal(earlier["entropy"], later["entropy"][:n])):
            raise ValueError("real OpenMC generation history prefix changed across checkpoints")
    last = records[-1]
    segments = [
        {"batches": [21, 30], "mean_k_generation": float(np.mean(last["k"][20:30])),
         "mean_source_entropy_nats": float(np.mean(last["entropy"][20:30]))},
        {"batches": [31, 45], "mean_k_generation": float(np.mean(last["k"][30:45])),
         "mean_source_entropy_nats": float(np.mean(last["entropy"][30:45]))},
        {"batches": [46, 65], "mean_k_generation": float(np.mean(last["k"][45:65])),
         "mean_source_entropy_nats": float(np.mean(last["entropy"][45:65]))},
    ]
    return {
        "seed": last["seed"],
        "exact_history_prefixes_verified": True,
        "descriptive_generation_keff_and_entropy_windows": segments,
        "first_active_window_vs_last_k_generation_shift":
            segments[-1]["mean_k_generation"] - segments[0]["mean_k_generation"],
        "first_active_window_vs_last_entropy_nats_shift":
            segments[-1]["mean_source_entropy_nats"] - segments[0]["mean_source_entropy_nats"],
        "generation_k_samples_independent": False,
        "shannon_entropy_stability_proves_source_convergence": False,
    }


def temporal_source_statistics(fields: np.ndarray) -> dict:
    """Observed same-seed and different-seed spatial TV, at two mesh scales.

    Coarsening a normalized spatial distribution cannot increase total
    variation. Use this as a numerical check against indexing/mesh errors.
    """
    a = np.asarray(fields, dtype=float)
    if (a.shape != (4, 3, *FINE_BINS) or not np.isfinite(a).all()
            or np.any(a < 0) or
            not np.allclose(a.sum(axis=(2, 3, 4)), 1., rtol=0, atol=1e-10)):
        raise ValueError("invalid four-seed three-checkpoint probability histograms")
    coarse = a.reshape(4, 3, 2, 2, 2, 2, 5, 2).sum(axis=(3, 5, 7))
    per_seed = []
    tv = lambda x, y: float(0.5 * np.sum(np.abs(x - y)))
    for i, seed in enumerate(SEEDS):
        fine = [tv(a[i, x], a[i, y]) for x, y in ((0, 1), (1, 2), (0, 2))]
        low = [tv(coarse[i, x], coarse[i, y]) for x, y in ((0, 1), (1, 2), (0, 2))]
        if any(y > x + 1e-12 for x, y in zip(fine, low)):
            raise ValueError("source spatial TV violates mesh-coarsening inequality")
        per_seed.append({
            "seed": seed,
            "same_seed_TV_batch30_to_45": fine[0],
            "same_seed_TV_batch45_to_65": fine[1],
            "same_seed_TV_batch30_to_65": fine[2],
            "coarse_2x2x5_same_seed_TV_batch30_to_65": low[2],
            "axial_source_fractions_at_30_45_65":
                a[i].sum(axis=(1, 2)).tolist(),
        })
    per_time = []
    for j, batch in enumerate(CHECKPOINTS):
        pairwise = [tv(a[i, j], a[k, j]) for i, k in combinations(range(4), 2)]
        coarse_pairwise = [
            tv(coarse[i, j], coarse[k, j]) for i, k in combinations(range(4), 2)
        ]
        if any(y > x + 1e-12 for x, y in zip(pairwise, coarse_pairwise)):
            raise ValueError("between-seed spatial TV violates coarsening inequality")
        per_time.append({
            "batch": batch,
            "mean_between_seed_TV_4x4x10": float(np.mean(pairwise)),
            "max_between_seed_TV_4x4x10": float(max(pairwise)),
            "all_six_between_seed_TV_4x4x10": pairwise,
            "mean_between_seed_TV_2x2x5": float(np.mean(coarse_pairwise)),
        })
    return {
        "fine_spatial_bins_xyz": list(FINE_BINS),
        "coarse_spatial_bins_xyz": [2, 2, 5],
        "same_seed_temporal_source_variation": per_seed,
        "between_seed_source_variation_at_each_batch": per_time,
        "mean_30_to_65_within_seed_TV": float(np.mean(
            [row["same_seed_TV_batch30_to_65"] for row in per_seed]
        )),
        "source_stationarity_independently_qualified": False,
        "histogram_bin_coarsening_variation_increase_detected": False,
        "native_bank_sites_are_iid_neutron_histories": False,
    }


def audit_real_coupled_temporal_source(root: str | Path) -> dict:
    """Require 12 new native coupled HDF5 checkpoints, no synthetic fallback."""
    base = Path(root)
    ledger = base / "actual_four_seed_coupled_heating_covariance.json"
    if not ledger.is_file():
        raise FileNotFoundError("original four-seed coupled heating audit is required")
    evidence = json.loads(ledger.read_text(encoding="utf-8"))
    if (evidence.get("schema") != "nuke-executed-coupled-neutron-photon-four-seed-stage31-v1"
            or evidence.get("original_independent_openmc_rng_seeds") != list(SEEDS)
            or evidence.get("real_native_coupled_hdf5_count") != 4
            or not evidence.get("all_electron_positron_neutron_photon_energy_deposition_partitions_conserved")
            or evidence.get("statistically_calibrated_fission_source_convergence") is not False
            or evidence.get("real_neutron_photon_heating_hot_NERVA_physically_qualified") is not False):
        raise ValueError("genuine coupled heating evidence unqualified")
    heating = evidence["per_actual_seed"]
    if [entry.get("seed") for entry in heating] != list(SEEDS):
        raise ValueError("independent seed ordering disagrees with coupled heating")
    lo = tuple(evidence["mesh_lower_left_cm"])
    hi = tuple(evidence["mesh_upper_right_cm"])
    if evidence.get("mesh_dimensions_xyz") != [12, 12, 10]:
        raise ValueError("native coupled heating mesh changed")
    unique_hashes = set()
    histories = []
    fields = []
    for seed, original in zip(SEEDS, heating):
        entries = []
        field_rows = []
        native = []
        folder = base / f"seed_{seed}"
        provenance_path = folder / "transport_provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError("genuine native transport provenance missing")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        cfg = provenance.get("config", {})
        if (cfg.get("seed") != seed or cfg.get("statepoint_batches") != list(CHECKPOINTS)
                or not cfg.get("photon_transport") or cfg.get("particles") != 3000
                or cfg.get("batches") != 65 or cfg.get("inactive") != 20):
            raise ValueError("native OpenMC multi-checkpoint transport provenance mismatch")
        for batch in CHECKPOINTS:
            statepoint = folder / f"statepoint.{batch}.h5"
            if not statepoint.is_file() or statepoint.stat().st_size == 0:
                raise FileNotFoundError("original checkpoint statepoint missing: " + str(statepoint))
            parsed = _hdf5_identity(statepoint)
            field, meta = read_source_bank(statepoint, lower_cm=lo, upper_cm=hi, bins=FINE_BINS)
            if (parsed["seed"] != seed or parsed["batch"] != batch
                    or meta["seed"] != seed or meta["completed_batch"] != batch
                    or parsed["realizations"] != batch-20
                    or meta["statepoint_sha256"] in unique_hashes):
                raise ValueError("genuine checkpoint identity/source bank mismatch")
            if batch == 65 and (meta["statepoint_sha256"] != original["native_statepoint_sha256"]
                    or meta["statepoint_sha256"] != provenance["statepoint"]["sha256"]):
                raise ValueError("native final statepoint does not match coupled heating identity")
            unique_hashes.add(meta["statepoint_sha256"])
            field_rows.append(field)
            native.append(parsed)
            entries.append({
                "batch": batch,
                "source_bank": meta,
                "actual_active_tally_realizations": parsed["realizations"],
                "native_generation_keff_at_batch": float(parsed["k"][-1]),
                "native_generation_entropy_at_batch_nats": float(parsed["entropy"][-1]),
            })
        chain = verify_native_checkpoint_prefixes(native)
        histories.append({"seed": seed, "checkpoints": entries, "eigen_history": chain})
        fields.append(np.stack(field_rows))
    result = temporal_source_statistics(np.stack(fields))
    return {
        "schema": "nuke-real-openmc-coupled-temporal-source-stage33-v1",
        "evidence": "TWELVE_NEW_ACTUAL_OPENMC_0153_NEUTRON_PHOTON_STATEPOINT_HDF5",
        "seed_order": list(SEEDS),
        "batch_checkpoints": list(CHECKPOINTS),
        "native_original_checkpoint_count": len(unique_hashes),
        "all_same_seed_native_generation_history_prefixes_exact": True,
        "all_final_native_sha_match_genuine_coupled_heating": True,
        "nuclear_and_geometry_input_qualification":
            "Enforced by Stage31 four-run heating audit run on the same new statepoints",
        "per_seed_real_transport": histories,
        "source_drift_and_resolution": result,
        "physically_stationary_source_qualified": False,
        "independent_critical_hot_nerva_watts_qualified": False,
        "limitations": [
            "Within-seed source checkpoints share neutronic ancestry and are correlated.",
            "Only three late and active-phase snapshots per RNG seed; no source-independence claim.",
            "Cross-seed TV and entropy are descriptive diagnostics, not formal stationarity tests.",
            "Cold 3-ring conceptual surrogate remains far from verified operating NERVA.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    r = audit_real_coupled_temporal_source(a.root)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("REAL coupled native checkpoints:", r["native_original_checkpoint_count"])
    print("Observed mean same-seed source TV 30->65:",
          r["source_drift_and_resolution"]["mean_30_to_65_within_seed_TV"])
    print("Source stationarity or hot-core fidelity qualified: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
