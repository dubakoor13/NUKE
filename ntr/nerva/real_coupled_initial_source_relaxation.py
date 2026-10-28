"""Stage37: actual OpenMC initial-source dependence and active-window stability.

Two initial fissionable-only boxes (broad/compact), identical reactor geometry,
materials, neutron/photon data and scores; 4 independent RNG seeds per box.
Transport: 3000 particles/generation, 100 total batches, 20 inactive.
Native checkpoints at batch 30, 60, 100; genuine 24 HDF5 required.

Descriptive source/heat diagnostics cannot establish a statistically stationary
fission source, reactor criticality, historic NASA NERVA or reactor watts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .source_bank_convergence import read_source_bank
from .real_coupled_spatial_heating_stability import (
    conservative_block_sum, total_variation,
)

MODES = ("legacy-broad", "central-compact")
SEEDS = (13007, 23011, 33013, 43019)
BATCHES = (30, 60, 100)
PARTICLES = 3000
N_INACTIVE = 20
WINDOWS = ((21, 30), (31, 60), (61, 100))
MESH = (12, 12, 10)
SOURCE_BINS = (4, 4, 10)
SCALES = ((12, 12, 10), (6, 6, 10), (3, 3, 10), (1, 1, 10))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def _identity(file: Path, *, seed: int, batch: int) -> dict:
    """Reject fabricated, uncoupled, wrong-length or wrong-source statepoints."""
    try:
        with h5py.File(file, "r") as h:
            if (bytes(h.attrs["filetype"]).rstrip(b"\0") != b"statepoint"
                    or bytes(h["run_mode"][()]).rstrip(b"\0") != b"eigenvalue"
                    or int(h.attrs.get("photon_transport", 0)) != 1
                    or int(h.attrs.get("source_present", 0)) != 1):
                raise ValueError("requires genuine coupled neutron/photon eigenvalue bank")
            extracted = {
                "seed": int(h["seed"][()]),
                "batch": int(h["current_batch"][()]),
                "configured_batches": int(h["n_batches"][()]),
                "inactive": int(h["n_inactive"][()]),
                "particles": int(h["n_particles"][()]),
                "generations_per_batch": int(h["generations_per_batch"][()]),
                "realizations": int(h["n_realizations"][()]),
                "k": np.asarray(h["k_generation"][()], dtype=float),
                "entropy": np.asarray(h["entropy"][()], dtype=float),
            }
    except (OSError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError("invalid original native OpenMC StatePoint: " + str(file)) from exc
    r = extracted
    if (r["seed"] != seed or r["batch"] != batch
            or r["configured_batches"] != 100
            or r["inactive"] != N_INACTIVE or r["particles"] != PARTICLES
            or r["generations_per_batch"] != 1
            or r["realizations"] != batch-N_INACTIVE
            or r["k"].shape != (batch,) or r["entropy"].shape != (batch,)
            or not np.isfinite(r["k"]).all() or np.any(r["k"] < 0)
            or not np.isfinite(r["entropy"]).all() or np.any(r["entropy"] < 0)):
        raise ValueError("wrong original OpenMC batches/seed/source histories")
    return r


def _native_heating(file: Path) -> dict:
    """Use installed official OpenMC native HDF5 reader, never local fake tallies."""
    import openmc
    names = ("nerva_3d_neutronics", "nerva_material_transport",
             "nerva_particle_heating")
    with openmc.StatePoint(str(file), autolink=False) as sp:
        matching = {}
        for tally in sp.tallies.values():
            if tally.name in names:
                if tally.name in matching:
                    raise ValueError("duplicate real coupled heating tally")
                matching[tally.name] = tally
        if set(matching) != set(names):
            raise ValueError("missing real neutron/photon energy partitions")
        core, material, particle = [matching[name] for name in names]
        if (len(core.filters) != 1
                or not isinstance(core.filters[0], openmc.MeshFilter)
                or len(material.filters) != 1
                or not isinstance(material.filters[0], openmc.MaterialFilter)
                or len(particle.filters) != 1
                or not isinstance(particle.filters[0], openmc.ParticleFilter)):
            raise ValueError("native neutron/photon heating filters not physical")
        mesh = core.filters[0].mesh
        dims = tuple(int(v) for v in mesh.dimension)
        if (dims != MESH
                or tuple(str(x) for x in particle.filters[0].bins) !=
                ("neutron", "photon", "electron", "positron")
                or not np.isfinite(np.asarray(mesh.lower_left)).all()
                or not np.isfinite(np.asarray(mesh.upper_right)).all()):
            raise ValueError("wrong original coupled mesh/particle heating")
        def tally_sum(tally, is_mesh):
            if ("heating" not in tally.scores or len(tally.nuclides) != 1
                    or str(tally.nuclides[0]) != "total"):
                raise ValueError("original heating score was changed")
            if is_mesh:
                ix = list(tally.scores).index("heating")
                return np.asarray(
                    tally.get_reshaped_data(value="sum", expand_dims=True)[..., 0, ix],
                    dtype=float)
            return np.asarray(tally.get_values(scores=["heating"], value="sum"),
                              dtype=float).reshape(-1)
        return {
            "mesh": tally_sum(core, True),
            "materials": tally_sum(material, False),
            "particles": tally_sum(particle, False),
            "mesh_lower_left_cm": [float(v) for v in mesh.lower_left],
            "mesh_upper_right_cm": [float(v) for v in mesh.upper_right],
            "material_bin_ids": [str(getattr(x, "id", x)) for x in material.filters[0].bins],
            "estimators": [str(x.estimator) for x in (core,material,particle)],
            "realizations": [int(x.num_realizations) for x in (core,material,particle)],
        }


def disjoint_windows(cumulative: list[np.ndarray],
                     counts: tuple[int, int, int] = (10, 40, 80)) -> np.ndarray:
    """Recover the 10/30/40 active Monte Carlo-batch window mean fields."""
    if (len(cumulative) != 3 or tuple(counts) != (10, 40, 80)):
        raise ValueError("requires genuine 30/60/100 checkpoints, 20 inactive")
    values = [np.asarray(a, dtype=float) for a in cumulative]
    if (not values[0].size
            or any(a.shape != values[0].shape
                   or not np.isfinite(a).all() or np.any(a < 0) for a in values)):
        raise ValueError("invalid original cumulative native tally moments")
    previous = np.zeros_like(values[0])
    prev_n = 0
    windows = []
    for current, n in zip(values, counts):
        delta = current - previous
        tol = 1e-12*np.maximum(1., np.maximum(current,previous))
        if np.any(delta < -tol):
            raise ValueError("nonmonotonic original cumulative neutron/photon deposition")
        windows.append(np.maximum(delta, 0.) / (n-prev_n))
        previous, prev_n = current, n
    recovered = (10*windows[0] + 30*windows[1] + 40*windows[2]) / 80
    if not np.allclose(recovered, values[-1]/80, rtol=1e-12, atol=1e-7):
        raise ValueError("failed native heating moment telescoping")
    return np.stack(windows)


def numerical_comparison(source_fields: np.ndarray,
                         heat_fields: np.ndarray) -> dict:
    """Mathematically conservative, shape-only TV for real native samples."""
    s, h = np.asarray(source_fields, float), np.asarray(heat_fields, float)
    if (s.shape != (2, 4, 3, *SOURCE_BINS)
            or h.shape != (2, 4, 3, *MESH)
            or not np.isfinite(s).all() or not np.isfinite(h).all()
            or np.any(s < 0) or np.any(h < 0)
            or not np.allclose(s.sum(axis=(3, 4, 5)), 1., atol=1e-10, rtol=0)
            or np.any(h.sum(axis=(3,4,5)) <= 0)):
        raise ValueError("two original initial-source cases and 24 positive native fields mandatory")
    heat_total = h.sum(axis=(3,4,5))
    hf = h / heat_total[:, :, :, None, None, None]
    result = {
        "two_initial_source_distributions": list(MODES),
        "per_case": [],
        "cross_initialization": {},
        "source_stationarity_proven": False,
        "real_nerva_operating_power_or_temperature_qualified": False,
        "independent_MC_error_coverage_established": False,
    }
    for i, mode in enumerate(MODES):
        per_seed_source = [
            total_variation(s[i, k, 0], s[i, k, 2]) for k in range(4)]
        per_seed_heating = [
            total_variation(hf[i,k,0], hf[i,k,2]) for k in range(4)]
        result["per_case"].append({
            "initial_fission_source": mode,
            "mean_source_TV_batch30_to_100": float(np.mean(per_seed_source)),
            "four_individual_source_TV_batch30_to_100": per_seed_source,
            "mean_3d_heating_TV_active_21_30_to_61_100":
                float(np.mean(per_seed_heating)),
            "four_individual_3d_heating_TV_active_21_30_to_61_100": per_seed_heating,
            "four_seed_mean_deposited_ev_per_source_in_disjoint_windows":
                np.mean(heat_total[i], axis=0).tolist(),
            "last_window_heating_between_seed_sample_SD_ev_per_source":
                float(np.std(heat_total[i,:,2],ddof=1)),
            "mean_final_source_bank_4x4x10_probability": s[i,:,2].mean(axis=0).tolist(),
        })
    source_diffs = []
    for j, batch in enumerate(BATCHES):
        source_diffs.append({
            "batch": batch,
            "cross_initialization_seed_average_source_TV":
                total_variation(s[0,:,j].mean(axis=0),s[1,:,j].mean(axis=0)),
        })
    scales = []
    for scale in SCALES:
        reduced = np.stack([
            np.stack([
                np.stack([
                    conservative_block_sum(hf[m,k,j],scale)
                    for j in range(3)])
                for k in range(4)])
            for m in range(2)])
        scales.append({
            "mesh_bins_xyz": list(scale),
            "cross_initialization_last_window_heating_shape_TV":
                total_variation(reduced[0,:,2].mean(axis=0),
                                reduced[1,:,2].mean(axis=0)),
            "mean_temporal_early_to_late_TV_by_source_initialization": [
                float(np.mean([
                    total_variation(reduced[m,k,0],reduced[m,k,2])
                    for k in range(4)])) for m in range(2)
            ],
        })
    for a,b in zip(scales,scales[1:]):
        if (b["cross_initialization_last_window_heating_shape_TV"] >
            a["cross_initialization_last_window_heating_shape_TV"] + 1e-12
                or any(x > y + 1e-12 for x,y in zip(
                    b["mean_temporal_early_to_late_TV_by_source_initialization"],
                    a["mean_temporal_early_to_late_TV_by_source_initialization"]))):
            raise ValueError("spatial source-conditioned heating TV increases after conservative coarsening")
    result["cross_initialization"] = {
        "source_bank_comparison_at_three_batch_checkpoints": source_diffs,
        "late_window_heating_shape_by_mesh_resolution": scales,
        "same_integer_RNG_seed_between_different_source_boxes_IS_NOT_valid_CRN_pairing": True,
    }
    return result


def audit_original_initialization_experiment(root: str | Path) -> tuple[dict,dict]:
    folder = Path(root)
    all_source, all_heating, observations = [], [], []
    physically_same = None
    nuclear_same = None
    recorded_hashes: set[str] = set()
    for mode in MODES:
        mode_sources, mode_heating, mode_meta = [], [], []
        for seed in SEEDS:
            run = folder / mode / f"seed_{seed}"
            prov_path = run / "transport_provenance.json"
            data_path = run / "nuclear_data_report.json"
            if not prov_path.is_file() or not data_path.is_file():
                raise FileNotFoundError("original source-mode native provenance or nuclear data missing")
            prov = json.loads(prov_path.read_text())
            data = json.loads(data_path.read_text())
            cfg = prov.get("config",{})
            if (cfg.get("initial_fission_source") != mode
                    or cfg.get("seed") != seed or cfg.get("particles") != PARTICLES
                    or cfg.get("statepoint_batches") != list(BATCHES)
                    or cfg.get("batches") != 100 or cfg.get("inactive") != 20
                    or cfg.get("photon_transport") is not True
                    or not data.get("ready")):
                raise ValueError("genuine source-initialization experiment provenance invalid")
            for name in ("geometry.xml","materials.xml","tallies.xml","settings.xml"):
                expected = prov.get("xml_sha256",{}).get(name)
                if not expected or _sha(run/name) != expected:
                    raise ValueError("actual OpenMC input XML SHA fingerprint missing or forged")
            physical = tuple(prov["xml_sha256"][x] for x in (
                "geometry.xml","materials.xml","tallies.xml"))
            nuclear = prov["nuclear_data"].get("cross_sections_sha256")
            if not nuclear or nuclear != data.get("sha256"):
                raise ValueError("original evaluated data index SHA mismatch")
            if physically_same is None:
                physically_same = physical
                nuclear_same = nuclear
            elif physical != physically_same or nuclear != nuclear_same:
                raise ValueError("same reactor geometry/material/tallies/data mandatory for source experiment")
            meta, sources, summed, histories = [], [], [], []
            heat_signature = None
            for batch in BATCHES:
                file = run / f"statepoint.{batch}.h5"
                if not file.is_file() or not file.stat().st_size:
                    raise FileNotFoundError("actual OpenMC source checkpoint missing: "+str(file))
                native = _identity(file,seed=seed,batch=batch)
                heating = _native_heating(file)
                if heating["realizations"] != [batch-N_INACTIVE]*3:
                    raise ValueError("genuine coupled heating tally realization count mismatch")
                signature = (heating["mesh_lower_left_cm"],
                             heating["mesh_upper_right_cm"],
                             heating["material_bin_ids"],heating["estimators"])
                if heat_signature is None:
                    heat_signature = signature
                elif signature != heat_signature:
                    raise ValueError("native scorer/mesh changed within source history")
                source, source_meta = read_source_bank(
                    file,lower_cm=heating["mesh_lower_left_cm"],
                    upper_cm=heating["mesh_upper_right_cm"],
                    bins=SOURCE_BINS)
                if (source_meta["seed"] != seed
                        or source_meta["completed_batch"] != batch
                        or source_meta["statepoint_sha256"] in recorded_hashes):
                    raise ValueError("source bank identity or uniqueness failed")
                if batch == 100 and source_meta["statepoint_sha256"] != prov["statepoint"]["sha256"]:
                    raise ValueError("actual final source hash disagrees with transport provenance")
                recorded_hashes.add(source_meta["statepoint_sha256"])
                histories.append(native)
                summed.append(heating)
                sources.append(source)
                meta.append(source_meta)
            for early,late in zip(histories,histories[1:]):
                size=early["batch"]
                if (not np.array_equal(early["k"],late["k"][:size])
                        or not np.array_equal(early["entropy"],late["entropy"][:size])):
                    raise ValueError("native OpenMC k-generation/entropy history prefixes inconsistent")
            groups = {}
            for tally in ("mesh","materials","particles"):
                groups[tally] = disjoint_windows([x[tally] for x in summed])
            for window in range(3):
                total = float(np.sum(groups["mesh"][window]))
                if total <= 0:
                    raise ValueError("original neutron/photon heating not positive")
                for name in ("materials","particles"):
                    candidate = float(np.sum(groups[name][window]))
                    if not math.isclose(total,candidate,rel_tol=1e-9,abs_tol=1e-6):
                        raise ValueError("genuine original disjoint mesh/material/particle energy violation")
            h = groups["mesh"]
            mode_sources.append(np.stack(sources))
            mode_heating.append(h)
            mode_meta.append({
                "seed": seed, "initial_source_mode": mode,
                "last_original_statepoint_sha256": meta[-1]["statepoint_sha256"],
                "exact_k_generation_and_shannon_source_entropy_prefixes_verified": True,
                "final_source_distinct_site_fraction": meta[-1]["distinct_position_fraction"],
                "final_source_inverse_simpson_position_diversity":
                    meta[-1]["inverse_simpson_distinct_position_count"],
                "source_k_generation_mean_21_30": float(np.mean(histories[-1]["k"][20:30])),
                "source_k_generation_mean_61_100": float(np.mean(histories[-1]["k"][60:100])),
                "entropy_mean_21_30": float(np.mean(histories[-1]["entropy"][20:30])),
                "entropy_mean_61_100": float(np.mean(histories[-1]["entropy"][60:100])),
                "deposition_ev_per_source_21_30_31_60_61_100":
                    h.sum(axis=(1,2,3)).tolist(),
            })
        all_source.append(np.stack(mode_sources))
        all_heating.append(np.stack(mode_heating))
        observations.append(mode_meta)
    source_arr = np.stack(all_source)
    heating_arr = np.stack(all_heating)
    diagnostics = numerical_comparison(source_arr, heating_arr)
    return {
        "schema": "nuke-stage37-eight-original-coupled-openmc-initial-source-relaxation-v1",
        "genuine_coupled_native_HDF5_count": len(recorded_hashes),
        "real_independent_openmc_source_histories": 8,
        "physical_reactor_geometry_material_tally_and_evaluated_data_identical": True,
        "two_source_initializations": list(MODES),
        "independent_seed_labels": list(SEEDS),
        "native_statepoint_batch_checkpoints": list(BATCHES),
        "inactive_generations_per_run": N_INACTIVE,
        "particle_starts_per_generation": PARTICLES,
        "active_heating_windows": [list(a) for a in WINDOWS],
        "each_original_source_history": observations,
        "source_heating_diagnostics": diagnostics,
        "statistically_stationary_source_qualified": False,
        "physical_hot_nerva_power_watts_qualified": False,
        "nasa_original_reactor_benchmark_matched": False,
        "limitations": [
            "Different original initial source boxes, but *same* physical reactor and nuclear data.",
            "Same seed integers across source boxes do NOT establish common-random-number pairing.",
            "Eigenvalue source resampling produces correlated or replicated fission offspring.",
            "Source-bank snapshots are single-generation banks, not independent-source samples.",
            "Disjoint active-window heating shares source genealogy across batches.",
            "Cold highly subcritical surrogate, not historical critical/hot NASA NERVA.",
            "Energy remains eV per Monte Carlo starting neutron, not reactor watts.",
        ],
    }, {
        "native_source_probability_4x4x10": source_arr,
        "native_coupled_heating_ev_per_start_12x12x10": heating_arr,
        "initial_source_modes": np.asarray(MODES),
        "seed_labels": np.asarray(SEEDS),
        "checkpoint_batches": np.asarray(BATCHES),
        "active_heating_windows": np.asarray(WINDOWS),
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result, fields=audit_original_initialization_experiment(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    npz=a.output.with_suffix(".npz")
    np.savez_compressed(npz,**fields)
    result["original_source_and_heat_fields_npz"]=npz.name
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Original OpenMC coupled neutron/photon HDF5: ",result["genuine_coupled_native_HDF5_count"])
    for row in result["source_heating_diagnostics"]["per_case"]:
        print(row["initial_fission_source"],"30->100 fission-source TV",
              row["mean_source_TV_batch30_to_100"],"3D disjoint heating TV",
              row["mean_3d_heating_TV_active_21_30_to_61_100"])
    for compare in result["source_heating_diagnostics"]["cross_initialization"]["source_bank_comparison_at_three_batch_checkpoints"]:
        print("initial-source spatial TV at generation",compare["batch"],":",
              compare["cross_initialization_seed_average_source_TV"])
    print("Hot NERVA stationarity, criticality or physical watts qualified: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
