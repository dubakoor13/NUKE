"""Stage49 — actual OpenMC fission-source *phase-space* versus position diversity.

Reinterpret the Stage48 source-degeneracy finding using all 16 original
evaluated 750/900 K neutron C++ HDF5 StatePoints. Equal birth positions
are NOT identical neutron states or proof of equal/independent genealogies:
the original OpenMC source_bank stores position, direction, energy, time,
weight, delayed group and particle. Count exact duplicated FULL records
separately from shared birth positions, quantify full phase-space richness,
and audit the actual native C++ multiplication estimators.

Compare only public historical *element-level* geometry and avoid blending
heritage NERVA element sizes with a proposed SNRE reactor layout. No design,
fuel loading, geometry, thermal state, enrichment or control modifications.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .config import NervaConfig
from .core_criticality_physical_qualification import read_executed_multiplication
from .physical_core_reference_validation import assess_core_fidelity
from .real_multitemperature_independent_seed_neutronics import (
    TEMPS, SEEDS, CHECKPOINTS, PARTICLES, N_BATCHES, N_INACTIVE, _sha,
)

EXPECTED_SOURCE_FIELDS = (
    "r", "u", "E", "time", "wgt", "delayed_group", "surf_id", "particle"
)


def characterize_original_neutron_source(source: np.ndarray) -> dict:
    """Measure actual full neutron birth phase-space vs exact repeated positions.

    This also accepts small synthetic structured unit arrays, but the native
    acceptance path separately verifies real OpenMC HDF5 and all 3000 sites.
    """
    if (source.ndim!=1 or source.size<2
            or source.dtype.names!=EXPECTED_SOURCE_FIELDS):
        raise ValueError("genuine OpenMC neutron source record fields required")
    n=int(source.size)
    pos=np.column_stack([source["r"][axis] for axis in ("x","y","z")])
    direction=np.column_stack([source["u"][axis] for axis in ("x","y","z")])
    energy=np.asarray(source["E"],dtype=float)
    time=np.asarray(source["time"],dtype=float)
    wgt=np.asarray(source["wgt"],dtype=float)
    norms=np.linalg.norm(direction,axis=1)
    if (not np.isfinite(pos).all() or not np.isfinite(direction).all()
            or not np.isfinite(energy).all() or not np.isfinite(time).all()
            or not np.isfinite(wgt).all() or np.any(wgt<=0)
            or np.any(energy<=0) or np.max(np.abs(norms-1.))>1e-9):
        raise ValueError("nonphysical neutron source birth position/direction/energy/weight")
    unique_pos,inverse,counts=np.unique(
        pos,axis=0,return_inverse=True,return_counts=True)
    n_pos=len(unique_pos)
    n_pe=len(np.unique(np.column_stack((pos,energy)),axis=0))
    n_peu=len(np.unique(np.column_stack((pos,direction,energy)),axis=0))
    complete,full_counts=np.unique(source,return_counts=True)
    n_full=len(complete)
    if not n_pos<=n_pe<=n_peu<=n_full<=n:
        raise ValueError("physical neutron full-source uniqueness hierarchy violated")
    site_w=np.bincount(inverse,weights=wgt,minlength=n_pos)
    p=site_w/np.sum(site_w)
    return {
        "exact_unique_birth_positions":n_pos,
        "exact_unique_position_and_energy":n_pe,
        "exact_unique_position_direction_energy":n_peu,
        "exact_unique_complete_source_particle_records":n_full,
        "exact_repeated_complete_source_records":n-n_full,
        "source_sites_sharing_birth_position":n-n_pos,
        "max_particles_at_one_exact_birth_position":int(np.max(counts)),
        "max_identical_complete_source_records":int(np.max(full_counts)),
        "largest_weighted_single_birth_position_fraction":float(np.max(p)),
        "inverse_simpson_birth_position_count":float(1./np.sum(p*p)),
        "fraction_unique_birth_positions":n_pos/n,
        "fraction_unique_complete_particle_records":n_full/n,
        "source_birth_energy_eV_min_median_max":[
            float(np.min(energy)),float(np.median(energy)),float(np.max(energy))],
        "source_birth_energy_fraction_above_100keV":float(
            np.sum(wgt[energy>=100000.])/np.sum(wgt)),
        "source_direction_max_unit_norm_error":float(np.max(np.abs(norms-1.))),
    }


def native_phase_space(file: Path, *, seed: int, batch: int) -> dict:
    """Refuse JSON mock or incomplete neutron StatePoint; inspect FULL C++ bank."""
    _sha(file)
    with h5py.File(file, "r") as state:
        if (bytes(state.attrs.get("filetype", b"")).rstrip(b"\0") != b"statepoint"
                or bytes(state["run_mode"][()]).rstrip(b"\0") != b"eigenvalue"
                or int(state.attrs.get("source_present", 0)) != 1
                or int(state["seed"][()]) != seed
                or int(state["current_batch"][()]) != batch
                or int(state["n_particles"][()]) != PARTICLES
                or int(state["n_batches"][()]) != N_BATCHES
                or int(state["n_inactive"][()]) != N_INACTIVE
                or int(state["n_realizations"][()]) != batch-N_INACTIVE
                or int(state["generations_per_batch"][()]) != 1
                or tuple(int(x) for x in state.attrs["openmc_version"])
                != (0, 15, 3)):
            raise ValueError("actual original OpenMC v0.15.3 neutron statepoint identity invalid")
        source = np.asarray(state["source_bank"][()])
        entropy = np.asarray(state["entropy"][()], dtype=float)
        k_generation = np.asarray(state["k_generation"][()], dtype=float)
    if (source.shape != (PARTICLES,)
            or source.dtype.names != EXPECTED_SOURCE_FIELDS
            or entropy.shape != (batch,) or k_generation.shape != (batch,)
            or not np.isfinite(entropy).all()
            or not np.isfinite(k_generation).all() or np.any(k_generation <= 0)):
        raise ValueError("native OpenMC source particle record or histories malformed")

    measures=characterize_original_neutron_source(source)

    scored = read_executed_multiplication(file)
    if scored["seed"] != seed or scored["completed_batch"] != batch:
        raise ValueError("genuine collision/absorption/tracklength estimator seed mismatch")
    return {
        "native_statepoint_sha256": _sha(file),
        "seed": seed,
        "batch": batch,
        "source_sites": PARTICLES,
        **measures,
        "native_combined_k": scored["combined_k_effective"],
        "native_combined_k_reported_sem": scored["openmc_reported_combined_k_standard_error"],
        "native_collision_absorption_tracklength_k": {
            name: scored["native_neutron_global_estimators"][name]["mean"]
            for name in ("k-collision", "k-absorption", "k-tracklength")
        },
        "native_leakage_tally_score_per_active_batch_NOT_a_leakage_probability":
            scored["native_neutron_global_estimators"]["leakage"]["mean"],
        "source_shannon_entropy_at_batch": float(entropy[-1]),
        "birth_positions_are_not_independent_genealogies": True,
        "individual_neutron_histories_proven_independent": False,
    }


def audit_actual_full_source_phase_space(root: Path | str) -> dict:
    root = Path(root)
    all_results = []
    geometry_sha = set()
    for temp in TEMPS:
        rows = []
        material_hashes = set()
        for seed in SEEDS:
            folder = root/f"T{temp}K"/f"seed_{seed}"
            prov = json.loads((folder/"transport_provenance.json").read_text(encoding="utf-8"))
            cfg = prov["config"]
            if (cfg.get("seed") != seed
                    or cfg.get("batches") != N_BATCHES
                    or cfg.get("inactive") != N_INACTIVE
                    or cfg.get("particles") != PARTICLES
                    or cfg.get("statepoint_batches") != list(CHECKPOINTS)
                    or cfg.get("neutron_temperature_method") != "interpolation"
                    or cfg.get("photon_transport") is not False):
                raise ValueError("original neutron independent source histories not comparable")
            for name in ("fuel_temperature_k", "hydrogen_temperature_k",
                         "bound_solid_temperature_k", "structural_solid_temperature_k"):
                if float(cfg[name]) != float(temp):
                    raise ValueError("actual fission-source temperature changed among cases")
            for name in ("geometry.xml", "materials.xml", "settings.xml"):
                if _sha(folder/name) != prov["xml_sha256"][name]:
                    raise ValueError("physical OpenMC XML SHA mismatch")
            geometry_sha.add(prov["xml_sha256"]["geometry.xml"])
            material_hashes.add(prov["xml_sha256"]["materials.xml"])
            checkpoints = []
            for batch in CHECKPOINTS:
                state = native_phase_space(folder/f"statepoint.{batch:03d}.h5",
                                           seed=seed, batch=batch)
                if batch == 100 and state["native_statepoint_sha256"] != prov["statepoint"]["sha256"]:
                    raise ValueError("actual original final HDF5 provenance mismatch")
                checkpoints.append(state)
            rows.append({
                "seed":seed, "native_checkpoints":checkpoints,
                "change_in_unique_full_phase_source_60_to_100": (
                    checkpoints[1]["exact_unique_complete_source_particle_records"]
                    - checkpoints[0]["exact_unique_complete_source_particle_records"]
                ),
                "change_in_unique_birth_positions_60_to_100": (
                    checkpoints[1]["exact_unique_birth_positions"]
                    - checkpoints[0]["exact_unique_birth_positions"]
                ),
            })
        if len(material_hashes) != 1:
            raise ValueError("actual neutron composition/density changed across seeds")
        late = [x["native_checkpoints"][1] for x in rows]
        all_results.append({
            "actual_evaluated_temperature_K":temp,
            "independent_rng_seed_count":len(rows),
            "original_cpp_source_observations":rows,
            "distinct_position_range_at_100": [
                min(r["exact_unique_birth_positions"] for r in late),
                max(r["exact_unique_birth_positions"] for r in late),
            ],
            "distinct_full_phase_source_record_range_at_100": [
                min(r["exact_unique_complete_source_particle_records"] for r in late),
                max(r["exact_unique_complete_source_particle_records"] for r in late),
            ],
            "maximum_exact_full_source_record_clones_at_100": max(
                r["exact_repeated_complete_source_records"] for r in late
            ),
            "mean_weighted_energy_fraction_of_source_above_100keV_at_100":
                float(np.mean([r["source_birth_energy_fraction_above_100keV"] for r in late])),
            "mean_native_k_combined_at_100":
                float(np.mean([r["native_combined_k"] for r in late])),
        })
    if len(geometry_sha) != 1:
        raise ValueError("actual conceptual reactor geometry changed among temperatures")
    historical = assess_core_fidelity(NervaConfig(core_rings=3), assembly="core")
    if (not historical["heritage_element_geometry_matches"]
            or historical["proposed_snre_bulk_geometry_matches"]
            or historical["overall_historical_core_validation"] != "NOT_VALIDATED"):
        raise ValueError("NASA distinct heritage-element vs proposed SNRE topology qualification failed")
    return {
        "schema":"nuke-stage49-actual-16-OpenMC-neutron-source-full-phase-space-and-fidelity-v1",
        "original_openmc_v0153_hdf5_inspected":16,
        "genuine_evaluated_material_temperature_K":list(TEMPS),
        "unique_independent_rng_seeds":list(SEEDS),
        "same_conceptual_19_element_geometry_all_cases":True,
        "per_temperature":all_results,
        "historical_nasa_geometry_fidelity_diagnostic":historical,
        "positions_not_equivalent_to_unique_neutron_source_phase_states":True,
        "distinct_complete_neutron_source_states_not_proven_independent_genealogies":True,
        "source_stationarity_demonstrated":False,
        "historical_NASA_NERVA_reactor_validated":False,
        "physical_reactor_watts_or_thermal_feedback_validated":False,
        "explanation":[
            "Exact duplicate source POSITIONS may have distinct directions/energies and are NOT full-source duplicates.",
            "Even unique full source records do not prove genealogical independence across fission families.",
            "The source bank samples fission birth spectrum; it does not measure equilibrium core neutron flux spectrum.",
            "Native leakage score in eigenvalue global_tallies is a tally estimator, NOT automatically a leakage probability.",
            "Original heritage fuel-element dimensional agreement does not imply proposed SNRE core-level validation.",
            "The model is strongly subcritical with unqualified fundamental source stationarity.",
        ],
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage48-original-hdf5",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    report=audit_actual_full_source_phase_space(args.stage48_original_hdf5)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    for x in report["per_temperature"]:
        print(x["actual_evaluated_temperature_K"],
              "unique positions at100",x["distinct_position_range_at_100"],
              "FULL source records at100",x["distinct_full_phase_source_record_range_at_100"])
    print("Neutron source stationarity and historical NASA NERVA: NOT qualified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
