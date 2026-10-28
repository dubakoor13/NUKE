"""Stage36: ORIGINAL OpenMC population-size sensitivity, no reactor redesign.

Compare four 3000-history/generation native coupled OpenMC runs (Stage33)
against four NEW 6000-history/generation coupled neutron/photon runs.
Both use identical material/geometry/tally XML, nuclear-data fingerprint,
temperatures, batches and independently simulated stochastic histories.
Each produces native source banks and disjoint energy moments at 30/45/65.

Results are descriptive for a cold, highly subcritical NERVA-derived surrogate,
NOT reactor design convergence, confidence intervals, physical power or NASA
benchmark certification. Seed numbers shared between populations are labels,
NOT evidence of valid common-random-numbers pairing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np

from .real_coupled_source_temporal import (
    SEEDS, CHECKPOINTS, _hdf5_identity, verify_native_checkpoint_prefixes,
)
from .real_coupled_disjoint_heating import _native_tallies, disjoint_raw_moments
from .real_coupled_spatial_heating_stability import (
    conservative_block_sum, total_variation,
)
from .source_bank_convergence import read_source_bank

POPULATIONS = (3000, 6000)
SOURCE_BINS = (4, 4, 10)
HEAT_BINS = (12, 12, 10)
SCALES = ((12, 12, 10), (6, 6, 10), (3, 3, 10), (1, 1, 10))
ACTIVE_WINDOWS = ((21, 30), (31, 45), (46, 65))


def _population_statistics(sources: np.ndarray, heating: np.ndarray) -> dict:
    """Compare spatial probability distributions across source populations.

    Exact same RNG seed labels at different particles/generation are NOT
    treated as paired CRN draws; only descriptive differences are shown.
    """
    s = np.asarray(sources, dtype=float)
    h = np.asarray(heating, dtype=float)
    if (s.shape != (2, 4, 3, *SOURCE_BINS)
            or h.shape != (2, 4, 3, *HEAT_BINS)
            or not np.isfinite(s).all() or not np.isfinite(h).all()
            or np.any(s < 0) or np.any(h < 0)
            or not np.allclose(s.sum(axis=(3, 4, 5)), 1., rtol=0, atol=1e-10)
            or np.any(h.sum(axis=(3, 4, 5)) <= 0)):
        raise ValueError("valid two-population four-seed native source/heat tensors mandatory")
    ht = h.sum(axis=(3, 4, 5))
    hfrac = h / ht[:, :, :, None, None, None]
    per_population = []
    for p, population in enumerate(POPULATIONS):
        late_bank = s[p, :, -1]
        bank_mean = late_bank.mean(axis=0)
        late_heat = hfrac[p, :, -1]
        per_population.append({
            "particles_per_generation": population,
            "observed_mean_source_TV_batch30_to_65": float(np.mean([
                total_variation(s[p, i, 0], s[p, i, 2]) for i in range(4)
            ])),
            "observed_final_source_between_seed_pairwise_TV": [
                total_variation(late_bank[i], late_bank[j])
                for i in range(4) for j in range(i + 1, 4)
            ],
            "observed_mean_fine_3D_heating_TV_first_to_last": float(np.mean([
                total_variation(hfrac[p, i, 0], hfrac[p, i, 2])
                for i in range(4)
            ])),
            "final_fission_source_mean_probability": bank_mean.tolist(),
            "observed_heating_ev_per_monte_carlo_start_per_window":
                np.mean(ht[p], axis=0).tolist(),
            "observed_late_heating_between_seed_total_SD":
                float(np.std(ht[p, :, -1], ddof=1)),
            "observed_late_heating_between_seed_pairwise_TV": [
                total_variation(late_heat[i], late_heat[j])
                for i in range(4) for j in range(i + 1, 4)
            ],
        })
    multi_scale = []
    for scale in SCALES:
        coarsened = np.stack([
            np.stack([
                np.stack([
                    conservative_block_sum(hfrac[p, i, j], scale)
                    for j in range(3)])
                for i in range(4)])
            for p in range(2)])
        per_population_temporal = [
            float(np.mean([
                total_variation(coarsened[p, i, 0], coarsened[p, i, 2])
                for i in range(4)]))
            for p in range(2)
        ]
        mean_late0 = np.mean(coarsened[0, :, 2], axis=0)
        mean_late1 = np.mean(coarsened[1, :, 2], axis=0)
        multi_scale.append({
            "heating_mesh_bins_xyz": list(scale),
            "within_3000_mean_early_to_late_TV": per_population_temporal[0],
            "within_6000_mean_early_to_late_TV": per_population_temporal[1],
            "cross_population_final_seed_average_heating_TV":
                total_variation(mean_late0, mean_late1),
            "cross_population_4_seed_mean_late_heating_bin_shape_normalized":
                [mean_late0.tolist(), mean_late1.tolist()] if scale == (1, 1, 10) else None,
        })
    for fine, coarse in zip(multi_scale, multi_scale[1:]):
        for name in ("within_3000_mean_early_to_late_TV",
                     "within_6000_mean_early_to_late_TV",
                     "cross_population_final_seed_average_heating_TV"):
            if coarse[name] > fine[name] + 1e-12:
                raise ValueError("coarsening increased heating total variation")
    return {
        "populations_particles_per_generation": list(POPULATIONS),
        "independent_seed_labels_shared_without_CRN_pairing": list(SEEDS),
        "source_final_seed_mean_4x4x10_TV_cross_population":
            total_variation(per_population[0]["final_fission_source_mean_probability"],
                            per_population[1]["final_fission_source_mean_probability"]),
        "per_population_diagnostics": per_population,
        "heating_spatial_resolution_sensitivity": multi_scale,
        "higher_particle_count_guarantees_reduced_bias": False,
        "source_stationarity_or_uncertainty_qualified": False,
        "nasa_hot_nerva_reactor_power_qualified": False,
    }


def _verify_cross_population_identity(
    old: dict, new: dict, old_run: dict, new_run: dict,
    seed: int,
) -> None:
    """Require exact same model except source-particle count."""
    configs = []
    for prov, pop in ((old, 3000), (new, 6000)):
        cfg = dict(prov.get("config", {}))
        if (cfg.pop("seed", None) != seed
                or cfg.pop("particles", None) != pop
                or cfg.get("statepoint_batches") != [30, 45, 65]
                or cfg.get("batches") != 65 or cfg.get("inactive") != 20
                or cfg.get("photon_transport") is not True
                or prov.get("nuclear_data", {}).get("cross_sections_sha256") is None):
            raise ValueError("genuine coupled population provenance parameters incorrect")
        configs.append(cfg)
    if configs[0] != configs[1]:
        raise ValueError("physical reactor configuration differs beyond particle count")
    for filename in ("geometry.xml", "materials.xml", "tallies.xml"):
        a = old.get("xml_sha256", {}).get(filename)
        b = new.get("xml_sha256", {}).get(filename)
        if not a or a != b:
            raise ValueError("physical OpenMC geometry, material or tally XML changed")
    if (old["nuclear_data"]["cross_sections_sha256"]
            != new["nuclear_data"]["cross_sections_sha256"]):
        raise ValueError("different evaluated neutron/photon data across populations")
    if old_run["native_total_heating_ev_per_source"] <= 0 or new_run["native_total_heating_ev_per_source"] <= 0:
        raise ValueError("invalid original native deposited energy")


def audit_real_population_sensitivity(
    baseline_root: str | Path, candidate_root: str | Path
) -> tuple[dict, dict]:
    base_roots = (Path(baseline_root), Path(candidate_root))
    ledgers = []
    for folder in base_roots:
        original = folder / "actual_four_seed_coupled_heating_covariance.json"
        if not original.is_file():
            raise FileNotFoundError("two original OpenMC 4-seed coupled-heating ledgers required")
        ledger = json.loads(original.read_text(encoding="utf-8"))
        if (ledger.get("schema") != "nuke-executed-coupled-neutron-photon-four-seed-stage31-v1"
                or ledger.get("original_independent_openmc_rng_seeds") != list(SEEDS)
                or ledger.get("real_native_coupled_hdf5_count") != 4
                or ledger.get("all_electron_positron_neutron_photon_energy_deposition_partitions_conserved") is not True
                or ledger.get("statistically_calibrated_fission_source_convergence") is not False
                or ledger.get("real_neutron_photon_heating_hot_NERVA_physically_qualified") is not False):
            raise ValueError("unverified genuine coupled neutron/photon HDF5 ledger")
        ledgers.append(ledger)
    for key in ("mesh_lower_left_cm", "mesh_upper_right_cm",
                "mesh_dimensions_xyz", "material_filter_bin_ids"):
        if ledgers[0].get(key) != ledgers[1].get(key):
            raise ValueError("physical OpenMC energy mesh/material identities changed")
    if ledgers[0]["mesh_dimensions_xyz"] != list(HEAT_BINS):
        raise ValueError("native mesh scoring dimensions unexpectedly changed")
    low = tuple(ledgers[0]["mesh_lower_left_cm"])
    high = tuple(ledgers[0]["mesh_upper_right_cm"])
    collected_sources = []
    collected_heating = []
    observation = []
    fingerprints = set()
    for p, population in enumerate(POPULATIONS):
        seeded_sources = []
        seeded_heating = []
        by_seed = []
        for i, seed in enumerate(SEEDS):
            folder = base_roots[p] / f"seed_{seed}"
            provenance_path = folder / "transport_provenance.json"
            if not provenance_path.is_file():
                raise FileNotFoundError("actual native OpenMC transport provenance required")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if p == 0:
                compare_folder = base_roots[1] / f"seed_{seed}"
                other_path = compare_folder / "transport_provenance.json"
                if not other_path.is_file():
                    raise FileNotFoundError("actual high-population transport provenance required")
                _verify_cross_population_identity(
                    provenance, json.loads(other_path.read_text(encoding="utf-8")),
                    ledgers[0]["per_actual_seed"][i],
                    ledgers[1]["per_actual_seed"][i], seed
                )
            snapshots = []
            source_fields = []
            sources = []
            identity = []
            for batch in CHECKPOINTS:
                f = folder / f"statepoint.{batch}.h5"
                if not f.is_file() or f.stat().st_size <= 0:
                    raise FileNotFoundError("original OpenMC population checkpoint missing: " + str(f))
                raw, metadata = read_source_bank(
                    f, lower_cm=low, upper_cm=high, bins=SOURCE_BINS)
                ident = _hdf5_identity(f, expected_particles=population)
                if (metadata["seed"] != seed or metadata["completed_batch"] != batch
                        or ident["seed"] != seed or ident["batch"] != batch
                        or ident["realizations"] != batch - 20
                        or metadata["statepoint_sha256"] in fingerprints):
                    raise ValueError("untrusted original source bank or repeated statepoint")
                fingerprints.add(metadata["statepoint_sha256"])
                if batch == 65 and (
                    metadata["statepoint_sha256"] != ledgers[p]["per_actual_seed"][i]["native_statepoint_sha256"]
                    or metadata["statepoint_sha256"] != provenance["statepoint"]["sha256"]
                ):
                    raise ValueError("final native OpenMC binary hash disagrees with executed coupled heating")
                score = _native_tallies(f, expected_particles=population)
                if (score["signature"]["mesh_dimensions_xyz"] != list(HEAT_BINS)
                        or score["signature"]["mesh_lower_left_cm"] != list(low)
                        or score["signature"]["mesh_upper_right_cm"] != list(high)):
                    raise ValueError("different original mesh physical extent")
                snapshots.append(score)
                source_fields.append(raw)
                sources.append(metadata)
                identity.append(ident)
            verify_native_checkpoint_prefixes(identity)
            domain_windows = {}
            for domain in ("mesh", "materials", "particles"):
                domain_windows[domain] = disjoint_raw_moments(
                    [x["raw"][domain]["sum"] for x in snapshots],
                    [x["raw"][domain]["sum_sq"] for x in snapshots],
                    [10, 25, 45], label=f"original {population} {seed} {domain}")
            heat = []
            for j in range(3):
                mesh = domain_windows["mesh"][j]["mean"]
                if (mesh.shape != HEAT_BINS or not np.isfinite(mesh).all()
                        or np.any(mesh < -1e-8)):
                    raise ValueError("invalid measured native spatial heating")
                mesh = np.maximum(mesh, 0.)
                total = float(np.sum(mesh))
                if total <= 0:
                    raise ValueError("original deposited heat must be positive")
                for domain in ("materials", "particles"):
                    other = float(np.sum(domain_windows[domain][j]["mean"]))
                    if not math.isclose(total, other, rel_tol=1e-9, abs_tol=1e-6):
                        raise ValueError("native nonoverlapping deposited-energy partition not conserved")
                heat.append(mesh)
            expected = ledgers[p]["per_actual_seed"][i]["native_total_heating_ev_per_source"]
            recovered = math.fsum(
                heat[j].sum() * n for j, n in enumerate((10,15,20))
            ) / 45
            if not math.isclose(recovered, expected, rel_tol=1e-10, abs_tol=1e-6):
                raise ValueError("source-normalized final energy does not conserve original OpenMC")
            seeded_sources.append(np.stack(source_fields))
            seeded_heating.append(np.stack(heat))
            by_seed.append({
                "seed": seed, "population": population,
                "final_native_statepoint_sha256": sources[-1]["statepoint_sha256"],
                "first_final_exact_generation_history_prefix_verified": True,
                "final_source_exact_unique_position_fraction": sources[-1]["distinct_position_fraction"],
                "final_source_inverse_simpson_distinct_position_index":
                    sources[-1]["inverse_simpson_distinct_position_count"],
                "final_largest_repeated_source_position_weight_fraction":
                    sources[-1]["largest_replicated_position_weight_fraction"],
                "final_total_heating_ev_per_mc_start": float(recovered),
            })
        collected_sources.append(np.stack(seeded_sources))
        collected_heating.append(np.stack(seeded_heating))
        observation.append(by_seed)
    source_array = np.stack(collected_sources)
    heat_array = np.stack(collected_heating)
    stats = _population_statistics(source_array, heat_array)
    return {
        "schema": "nuke-real-original-openmc-coupled-population-sensitivity-stage36-v1",
        "evidence": "24_ACTUAL_OPENMC_0153_NEUTRON_PHOTON_SOURCE_HEATING_STATEPOINTS",
        "physical_source_HDF5_count": len(fingerprints),
        "populations": list(POPULATIONS),
        "independent_seed_labels": list(SEEDS),
        "checkpoints": list(CHECKPOINTS),
        "identical_geometry_material_tally_xml_and_evaluated_data_across_populations": True,
        "all_disjoint_material_particle_mesh_energy_partitions_close": True,
        "per_original_simulation": observation,
        "population_and_spatial_heating_diagnostics": stats,
        "source_population_sensitivity_observed_NOT_validated": True,
        "statistically_independent_fission_site_counts_qualified": False,
        "reactor_hot_core_criticality_or_watts_qualified": False,
        "notes": [
            "Same nominal seed at different population sizes is NOT a paired CRN experiment.",
            "Bank source sites are genealogically correlated; no particle-level IID inference.",
            "Three disjoint batch windows remain dependent through fission ancestry.",
            "Observed changes combine Monte Carlo noise, source bias and spatial-resolution effects.",
            "Heating is eV per starting Monte Carlo neutron, NOT physical reactor watts.",
            "Cold far-subcritical conceptual NERVA-derived geometry is not NASA hot-core validation.",
        ],
    }, {
        "source_probability": source_array,
        "heating_ev_per_mc_start": heat_array,
        "population_sizes": np.asarray(POPULATIONS),
        "rng_seeds": np.asarray(SEEDS),
        "checkpoints": np.asarray(CHECKPOINTS),
        "mesh_lower_left_cm": np.asarray(low),
        "mesh_upper_right_cm": np.asarray(high),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-3000", type=Path, required=True)
    parser.add_argument("--candidate-6000", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report, raw = audit_real_population_sensitivity(
        args.baseline_3000, args.candidate_6000)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    npz_path = args.output.with_suffix(".npz")
    np.savez_compressed(npz_path, **raw)
    report["original_native_population_fields_npz"] = npz_path.name
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    d = report["population_and_spatial_heating_diagnostics"]
    print("Original OpenMC coupled HDF5 checkpoints:", report["physical_source_HDF5_count"])
    for p in d["per_population_diagnostics"]:
        print("Particles/gen", p["particles_per_generation"],
              "observed source spatial TV 30->65:", p["observed_mean_source_TV_batch30_to_65"],
              "3D heating TV early->late:", p["observed_mean_fine_3D_heating_TV_first_to_last"])
    print("Population final mean fission source TV:",
          d["source_final_seed_mean_4x4x10_TV_cross_population"])
    print("Hot reactor physics or source statistical convergence CERTIFIED: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
