"""Deterministic synthetic regression for Monte Carlo source-population comparison.

All fixtures are fabricated JSON and NOT claimed as neutron transport.
The separate Docker integration workflow executes the real OpenMC solver.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

from .source_population_sensitivity import assess_population_sensitivity


def _make_case(root: Path, particles: int, seeds: tuple[int, ...], *,
               effective_fraction: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    batches = [40, 60, 80, 100, 120]
    rows = []
    trajectories = []
    for seed in seeds:
        digest = f"{seed:064x}"
        records = [{
            "seed": seed,
            "completed_batch": batch,
            "source_sites": particles,
            "exact_distinct_site_positions": max(2, particles // 20),
            "inverse_simpson_distinct_position_count": effective_fraction * particles,
            "statepoint_sha256": f"{seed * 1000 + batch:064x}",
        } for batch in batches]
        records[-1]["statepoint_sha256"] = digest
        trajectories.append({"seed": seed, "snapshots": records})
        rows.append({
            "seed": seed,
            "statepoint_sha256": digest,
            "keff_mean": 0.0117 + seed * 1e-12,
            "keff_std_dev": .0002,
            "hdf5_eigenvalue_integrity": {
                "seed": seed, "particles_per_generation": particles,
                "current_batch": 120, "inactive_batches": 30,
            },
        })
        prov = {
            "assembly": "core",
            "openmc_version": "0.15.3",
            "xml_sha256": {
                "geometry.xml": "geometry-identity",
                "materials.xml": "materials-identity",
                "tallies.xml": "tallies-identity",
            },
            "nuclear_data": {"cross_sections_sha256": "evaluated-xs"},
            "config": {
                "seed": seed, "particles": particles,
                "batches": 120, "inactive": 30,
                "statepoint_batches": batches,
                "core_rings": 3, "fuel_temperature_k": None,
                "hydrogen_temperature_k": 293.6059129982851,
                "photon_transport": False,
            },
        }
        folder = root / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        (folder / "transport_provenance.json").write_text(json.dumps(prov))
    summary = {
        "execution": "ACTUAL_OPENMC_TRANSPORT",  # Synthetic structural fixture!
        "assembly": "core", "particles_per_generation": particles,
        "batches": 120, "inactive_batches": 30,
        "cross_sections_xml_sha256": "evaluated-xs",
        "run_statistics": {
            "seeds": list(seeds), "independent_seed_count": len(seeds),
            "inverse_variance_mean_keff": .01170,
            "dispersion_inflated_mean_keff_error": .0001,
        },
        "runs": rows,
        "multi_checkpoint_source_bank_trajectory": {
            "independent_seed_count": len(seeds),
            "saved_batches": batches, "runs": trajectories,
            "mean_within_seed_adjacent_total_variation_by_interval":
                [.68, .69, .71, .70],
            "between_seed_mean_total_variation_by_batch":
                [.65, .64, .68, .69, .67],
        },
    }
    (root / "real_core_transport_summary.json").write_text(json.dumps(summary))


def _modify(path: Path, edit):
    original = json.loads(path.read_text())
    modified = deepcopy(original)
    edit(modified)
    path.write_text(json.dumps(modified))
    return original


def reject(action, phrase: str):
    try:
        action()
    except ValueError as exc:
        assert phrase in str(exc), str(exc)
    else:
        raise AssertionError("invalid population evidence accepted: " + phrase)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        small = root / "3000"
        large = root / "6000"
        seeds_small = (13007, 23011, 33013, 43019)
        seeds_large = (53003, 63007, 73009, 83003)
        _make_case(small, 3000, seeds_small, effective_fraction=.01)
        _make_case(large, 6000, seeds_large, effective_fraction=.018)
        report = assess_population_sensitivity(small, large)
        assert report["comparison"] == "real_OpenMC_source_population_numerical_sensitivity"
        assert report["core_unchanged"]
        assert report["low_population"]["particles_per_generation"] == 3000
        assert report["high_population"]["particles_per_generation"] == 6000
        assert abs(report["effective_position_count_growth_factor"] - 3.6) < 1e-12
        assert abs(report["effective_position_fraction_growth_factor"] - 1.8) < 1e-12
        assert report["fission_source_convergence_demonstrated"] is False
        assert report["nasa_nerva_historical_model_validated"] is False

        path = large / "seed_53003" / "transport_provenance.json"
        original = _modify(
            path, lambda x: x["xml_sha256"].update({"materials.xml": "changed"})
        )
        reject(lambda: assess_population_sensitivity(small, large), "materials.xml")
        path.write_text(json.dumps(original))

        original = _modify(
            path, lambda x: x["config"].update({"hydrogen_temperature_k": 600})
        )
        reject(lambda: assess_population_sensitivity(small, large), "physics")
        path.write_text(json.dumps(original))

        original = _modify(
            path, lambda x: x["nuclear_data"].update({
                "cross_sections_sha256": "another-evaluation"
            })
        )
        reject(lambda: assess_population_sensitivity(small, large), "cross")
        path.write_text(json.dumps(original))

        path = large / "real_core_transport_summary.json"
        original = _modify(
            path, lambda x: x["runs"][0]["hdf5_eigenvalue_integrity"].update({
                "particles_per_generation": 100
            })
        )
        reject(lambda: assess_population_sensitivity(small, large), "HDF5")
        path.write_text(json.dumps(original))

        original = _modify(
            path, lambda x: x["multi_checkpoint_source_bank_trajectory"]["runs"][0][
                "snapshots"][-1].update({
                    "inverse_simpson_distinct_position_count": 0
                })
        )
        reject(lambda: assess_population_sensitivity(small, large), "inverse-Simpson")
        path.write_text(json.dumps(original))

        original = _modify(
            path, lambda x: x["multi_checkpoint_source_bank_trajectory"].update({
                "between_seed_mean_total_variation_by_batch": [0,0,0,0,2]
            })
        )
        reject(lambda: assess_population_sensitivity(small, large), "source TV")
        path.write_text(json.dumps(original))

        original = _modify(path, lambda x: x.update({"execution":"SYNTHETIC"}))
        reject(lambda: assess_population_sensitivity(small, large), "genuine")
        path.write_text(json.dumps(original))

    print("NERVA population-size sensitivity, seed and XS integrity: PASS")
    print("  synthetic fixture only; actual neutron transport executed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
