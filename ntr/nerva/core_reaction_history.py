"""Actual OpenMC active-batch WINDOW neutron reaction stability audit.

OpenMC saved intermediate eigenvalue statepoints report CUMULATIVE sums of
tracklength tallies over all active batches to date. Averaging their means
at checkpoints is mathematically incorrect. For disjoint windows, subtract
raw cumulative reaction sums and divide by the exact change in active
realizations. This does NOT make windows independent: fission descendants
remain correlated across batches and seeds.

All physics comes from genuine OpenMC saved HDF5. Do not tune enrichment,
geometry, control drums, source strength, or power. This is a diagnostic of
a cold conceptual core, NOT stationary-source or NASA NERVA certification.
"""
from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

SCORES = ("absorption", "nu-fission")
TALLY_NAMES = (
    "nerva_global_reaction_tracklength",
    "nerva_material_transport",
    "nerva_material_energy_transport",
)


def sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with Path(file).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").rstrip("\x00")
    raise ValueError("OpenMC tally name/score/estimator must be HDF5 bytes")


def _read_tally(tallies, name: str) -> tuple[np.ndarray, int, tuple[str, ...]]:
    matches = [obj for key,obj in tallies.items()
               if key.startswith("tally ") and isinstance(obj, h5py.Group)
               and "name" in obj and _text(obj["name"][()]) == name]
    if len(matches) != 1:
        raise ValueError("missing or duplicate genuine OpenMC tally " + name)
    obj = matches[0]
    if _text(obj["estimator"][()]) != "tracklength":
        raise ValueError("reaction windows require matching TRACKLENGTH estimators")
    names = tuple(_text(value) for value in obj["score_bins"][()])
    if len(set(names)) != len(names) or any(score not in names for score in SCORES):
        raise ValueError("actual neutron absorption/nu-fission score missing")
    n = int(obj["n_realizations"][()])
    raw = np.asarray(obj["results"][()], dtype=float)
    if (n < 1 or raw.ndim != 3 or raw.shape[-1] != 2
            or raw.shape[1] != len(names)
            or not np.isfinite(raw).all()
            or np.any(raw < 0)):
        raise ValueError("nonphysical or incomplete cumulative OpenMC tally moments")
    return raw[:, [names.index(s) for s in SCORES], 0], n, names


def read_checkpoint(path: Path) -> dict:
    path = Path(path)
    try:
        with h5py.File(path, "r") as h:
            if (bytes(h.attrs["filetype"]).rstrip(b"\x00") != b"statepoint"
                    or bytes(h["run_mode"][()]).rstrip(b"\x00") != b"eigenvalue"
                    or int(h.attrs["photon_transport"]) != 0):
                raise ValueError("reaction window requires real neutron-only eigenvalue statepoint")
            seed = int(h["seed"][()])
            batch = int(h["current_batch"][()])
            inactive = int(h["n_inactive"][()])
            particles = int(h["n_particles"][()])
            if seed < 1 or batch <= inactive or particles < 2:
                raise ValueError("unusable real OpenMC active generation count")
            tallies = h["tallies"]
            arrays = {}
            for name in TALLY_NAMES:
                arr, n, _ = _read_tally(tallies, name)
                if n != batch - inactive:
                    raise ValueError("real tally realization count disagrees with active generations")
                arrays[name] = arr
            if arrays[TALLY_NAMES[0]].shape != (1, 2):
                raise ValueError("whole-core tracklength neutron reaction is not unfiltered")
            material_count = arrays[TALLY_NAMES[1]].shape[0]
            if (material_count < 2
                    or arrays[TALLY_NAMES[2]].shape[0] % material_count != 0
                    or arrays[TALLY_NAMES[2]].shape[0] <= material_count):
                raise ValueError("invalid material by incident-energy reaction partitions")
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError("invalid saved OpenMC neutron reaction HDF5") from exc
    return {
        "seed": seed,
        "completed_batch": batch,
        "inactive_batches": inactive,
        "particles_per_generation": particles,
        "active_realizations": batch-inactive,
        "statepoint_sha256": sha256(path),
        "global": arrays[TALLY_NAMES[0]],
        "material": arrays[TALLY_NAMES[1]],
        "incident_energy": arrays[TALLY_NAMES[2]],
    }


def calculate_disjoint_windows(rows: list[dict]) -> dict:
    if len(rows) < 3:
        raise ValueError("at least three genuine OpenMC source checkpoints required")
    seeds = {r["seed"] for r in rows}
    inactive = {r["inactive_batches"] for r in rows}
    particles = {r["particles_per_generation"] for r in rows}
    if len(seeds) != 1 or len(inactive) != 1 or len(particles) != 1:
        raise ValueError("changed physical neutron model within source seed")
    batches = [r["completed_batch"] for r in rows]
    if any(a >= b for a,b in zip(batches,batches[1:])):
        raise ValueError("source checkpoints not strictly increasing")
    prior = {key:np.zeros_like(rows[0][key]) for key in
             ("global","material","incident_energy")}
    old_count = 0
    windows = []
    for entry in rows:
        n = entry["active_realizations"]
        increment = n - old_count
        if increment < 1 or n != entry["completed_batch"]-entry["inactive_batches"]:
            raise ValueError("invalid disjoint active generation duration")
        values = {}
        for key in prior:
            now = entry[key]
            if now.shape != prior[key].shape:
                raise ValueError("physical reaction filter structure changed between batches")
            delta = now-prior[key]
            if np.any(delta < -1e-9):
                raise ValueError("negative disjoint neutron reaction accumulation")
            values[key] = delta/increment
            prior[key] = now
        global_rate = values["global"].reshape(2)
        material_rate = values["material"].sum(axis=0)
        energy_rate = values["incident_energy"].sum(axis=0)
        if (np.any(global_rate <= 0) or
                not np.allclose(global_rate,material_rate,rtol=0,atol=1e-6) or
                not np.allclose(global_rate,energy_rate,rtol=0,atol=1e-6)):
            raise ValueError("disjoint physical neutron reaction partitions do not conserve")
        windows.append({
            "first_active_batch": int(entry["completed_batch"]-increment+1),
            "last_active_batch": int(entry["completed_batch"]),
            "number_of_active_generations": increment,
            "global_reaction_rates_per_source": {
                score:float(global_rate[i]) for i,score in enumerate(SCORES)},
            "material_reaction_rates_per_source": {
                score:values["material"][:,i].tolist()
                for i,score in enumerate(SCORES)},
            "incident_energy_reaction_rates_per_source": {
                score:values["incident_energy"][:,i].tolist()
                for i,score in enumerate(SCORES)},
            "statepoint_sha256":entry["statepoint_sha256"],
        })
        old_count = n

    full = rows[-1]
    final_mean = full["global"][0,:]/full["active_realizations"]
    weighted_mean = sum(
        window["number_of_active_generations"]
        * np.array([window["global_reaction_rates_per_source"][s] for s in SCORES])
        for window in windows
    )/old_count
    if not np.allclose(weighted_mean, final_mean,rtol=0,atol=2e-12):
        raise ValueError("disjoint active reaction windows fail to reconstruct final OpenMC tally")
    return {
        "seed":rows[0]["seed"],
        "inactive_batches":rows[0]["inactive_batches"],
        "particles_per_generation":rows[0]["particles_per_generation"],
        "saved_source_batches":batches,
        "windows":windows,
        "final_cumulative_reaction_rates_per_source":{
            score:float(final_mean[i]) for i,score in enumerate(SCORES)},
        "final_cumulative_reaction_reconstruction_passed":True,
    }


def audit_real_reaction_windows(summary: Path) -> dict:
    summary = Path(summary)
    result = json.loads(summary.read_text(encoding="utf-8"))
    if (result.get("execution") != "ACTUAL_OPENMC_TRANSPORT"
            or result.get("assembly") != "core"):
        raise ValueError("requires genuine OpenMC core neutron transport, not fixture")
    runs = result.get("runs")
    batches = result.get("saved_source_bank_batches")
    total = result.get("batches")
    if (not isinstance(runs,list) or len(runs)<4
            or not isinstance(batches,list) or len(batches)<3
            or not isinstance(total,int) or batches[-1]!=total):
        raise ValueError("invalid physical OpenMC multi-seed checkpoint schedule")
    digits = len(str(total))
    samples = []
    digests = set()
    model_hashes = {key:set() for key in ("geometry.xml","materials.xml","tallies.xml")}
    for run in runs:
        last = Path(run["statepoint_path"])
        if not last.is_file() or sha256(last)!=run.get("statepoint_sha256"):
            raise ValueError("genuine final OpenMC statepoint hash mismatch")
        source = json.loads((last.parent/"transport_provenance.json").read_text())
        if (source.get("config",{}).get("seed") != run.get("seed")
                or source.get("nuclear_data",{}).get("cross_sections_sha256")
                != result.get("cross_sections_xml_sha256")):
            raise ValueError("OpenMC random seed or physical evaluated data mismatch")
        for name in model_hashes:
            f = last.parent/name
            if not f.is_file() or sha256(f)!=source.get("xml_sha256",{}).get(name):
                raise ValueError("unverified physical model input XML")
            model_hashes[name].add(sha256(f))
        filenames = [last.parent/f"statepoint.{n:0{digits}d}.h5" for n in batches]
        if filenames[-1].resolve()!=last.resolve():
            raise ValueError("genuine statepoint does not match checkpoint schedule")
        statepoints = [read_checkpoint(file) for file in filenames]
        if any(row["statepoint_sha256"] in digests for row in statepoints):
            raise ValueError("duplicate statepoint reused in neutron reaction ensemble")
        digests.update(row["statepoint_sha256"] for row in statepoints)
        run_info = calculate_disjoint_windows(statepoints)
        if run_info["seed"] != run["seed"] or run_info["saved_source_batches"]!=batches:
            raise ValueError("independent seed or source-batch checkpoint identity mismatch")
        samples.append(run_info)
    if (len({s["seed"] for s in samples})!=len(samples)
            or any(len(value)!=1 for value in model_hashes.values())):
        raise ValueError("different geometry/material/tallies among independent seeds")
    n_windows = len(samples[0]["windows"])
    for s in samples:
        if len(s["windows"])!=n_windows:
            raise ValueError("unequal physical source history observation windows")
    per_score = {}
    for score in SCORES:
        matrix = np.asarray([
            [w["global_reaction_rates_per_source"][score]
             for w in sample["windows"]]
            for sample in samples],dtype=float)
        if np.any(matrix <= 0) or not np.isfinite(matrix).all():
            raise ValueError("invalid physical neutron reaction window rates")
        window_means = matrix.mean(axis=0)
        per_score[score] = {
            "mean_across_seed_reaction_per_active_window":window_means.tolist(),
            "cross_seed_window_coefficient_of_variation":
                (matrix.std(axis=0,ddof=1)/window_means).tolist(),
            "mean_first_to_last_window_fractional_change":
                float(window_means[-1]/window_means[0]-1),
            "maximum_seed_relative_window_deviation_from_own_final_cumulative":float(max(
                abs(matrix[i,j] /
                    samples[i]["final_cumulative_reaction_rates_per_source"][score]-1)
                for i in range(len(samples)) for j in range(n_windows)
            )),
            "all_seed_window_reaction_rates_per_source":matrix.tolist(),
        }
    return {
        "analysis":"ACTUAL_OPENMC_DISJOINT_NEUTRON_REACTION_HISTORY_WINDOWS",
        "real_transport_summary_sha256":sha256(summary),
        "independent_seed_count":len(samples),
        "observed_seed_history":samples,
        "physical_neutron_reactions":per_score,
        "all_reaction_window_partition_and_reconstruction_checks_passed":True,
        "independent_fission_source_stationarity_proven":False,
        "historical_nasa_core_validated":False,
        "interpretation":(
            "Checkpoint raw OpenMC tally sums are cumulative from the end of "
            "inactive source generations. Subtracting those cumulative sums "
            "reconstructs disjoint windows without re-running neutrons. "
            "All windows preserve independently filtered whole-core "
            "material and neutron incident-energy reaction rates. No "
            "independence, uncertainty-calibrated stationarity, or valid "
            "NASA NERVA reactor performance is implied: active generations "
            "share correlated fission ancestry, and the surrogate remains "
            "strongly subcritical with severe spatial source cloning."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    report = audit_real_reaction_windows(args.summary)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Genuine OpenMC disjoint active-neutron reaction window QA: PASS")
    for score in SCORES:
        item=report["physical_neutron_reactions"][score]
        print(score,"active-window means",item["mean_across_seed_reaction_per_active_window"])
        print(score,"cross-seed window CV",item["cross_seed_window_coefficient_of_variation"])
    print("Independent source stationarity and NASA reactor validated: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
