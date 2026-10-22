"""Read-only physical direction/energy and cloning audit of REAL OpenMC fission banks.

Each bank particle may have a distinct emission direction and energy while
sharing one exact fission position with hundreds of other resampled sites.
A near-isotropic angular distribution must never be accepted as independent
fission histories or as a stationary fundamental-mode source.

Uses persisted structured HDF5 data. No changes to nuclear-data values,
materials, reactor geometry, criticality, or OpenMC transport settings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bytes(value) -> bytes:
    if not isinstance(value, (bytes, np.bytes_)):
        raise ValueError("invalid actual OpenMC HDF5 identity")
    return bytes(value).rstrip(b"\0")


def read_real_bank_angular_physics(file: Path) -> tuple[dict, set[tuple[float, float, float]]]:
    """Read neutron source direction, energy, and exact position multiplicity."""
    file = Path(file)
    with h5py.File(file, "r") as h:
        if (_bytes(h.attrs["filetype"]) != b"statepoint"
                or _bytes(h["run_mode"][()]) != b"eigenvalue"
                or int(h.attrs.get("source_present", 0)) != 1
                or int(h.attrs.get("photon_transport", -1)) != 0):
            raise ValueError("requires genuine neutron-only OpenMC eigenvalue source bank")
        seed = int(h["seed"][()])
        batch = int(h["current_batch"][()])
        nominal = int(h["n_particles"][()])
        source = h["source_bank"][()]
    fields = {"r", "u", "E", "wgt", "time", "particle"}
    if (source.ndim != 1 or source.dtype.names is None
            or not fields.issubset(source.dtype.names)
            or source["r"].dtype.names != ("x", "y", "z")
            or source["u"].dtype.names != ("x", "y", "z")
            or seed <= 0 or batch <= 0 or nominal < 2
            or len(source) != nominal):
        raise ValueError("inconsistent actual OpenMC fission source population or vector fields")
    xyz = np.column_stack([source["r"][c] for c in ("x", "y", "z")]).astype(float)
    uv = np.column_stack([source["u"][c] for c in ("x", "y", "z")]).astype(float)
    energy = np.asarray(source["E"], dtype=float)
    weight = np.asarray(source["wgt"], dtype=float)
    event_time = np.asarray(source["time"], dtype=float)
    if (not np.isfinite(xyz).all() or not np.isfinite(uv).all()
            or not np.isfinite(energy).all() or not np.isfinite(weight).all()
            or not np.isfinite(event_time).all()
            or np.any(weight <= 0) or np.any(energy <= 0)
            or np.any(event_time < 0)
            or np.any(source["particle"] != 0)):
        raise ValueError("nonphysical neutron source position/direction/energy/time/particle")
    norm = np.linalg.norm(uv, axis=1)
    deviation = float(np.max(np.abs(norm - 1.)))
    if deviation > 1e-8:
        raise ValueError("actual neutron emission direction is not a unit vector")
    total_weight = float(weight.sum(dtype=np.float64))
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ValueError("invalid total fission source weight")

    # Spatial fission coordinates, not the angular or energy resampling,
    # determine the number of exact distinct fission-source *positions*.
    unique, inverse, counts = np.unique(
        xyz, axis=0, return_inverse=True, return_counts=True
    )
    position_weight = np.bincount(inverse, weights=weight, minlength=len(unique))
    probability = position_weight / total_weight
    effective_positions = float(1. / np.sum(probability**2, dtype=np.float64))
    most_repeated = int(np.argmax(counts))
    repeated = inverse == most_repeated
    # Do NOT treat different directions/energies from the same location as
    # separate fission sites or independent neutron genealogies.
    top_direction_count = int(len(np.unique(uv[repeated], axis=0)))
    top_energy_count = int(len(np.unique(energy[repeated])))
    dipole = np.average(uv, axis=0, weights=weight)
    moment = (uv * weight[:, None]).T @ uv / total_weight
    moment = (moment + moment.T) * .5
    eigvals = np.linalg.eigvalsh(moment)
    if (not np.isfinite(eigvals).all() or eigvals[0] < -1e-10
            or abs(float(np.trace(moment)) - 1.) > 2e-8):
        raise ValueError("nonphysical normalized neutron emission angular moments")
    positions = {tuple(float(v) for v in p) for p in unique}
    return {
        "seed": seed,
        "checkpoint_batch": batch,
        "actual_statepoint_sha256": _sha256(file),
        "nominal_source_particles": nominal,
        "distinct_exact_source_positions": int(len(unique)),
        "effective_source_position_count_inverse_simpson": effective_positions,
        "effective_position_fraction_of_nominal_population": (
            effective_positions / nominal
        ),
        "max_clones_at_exact_same_position": int(counts[most_repeated]),
        "unique_directions_at_most_cloned_position": top_direction_count,
        "unique_energies_at_most_cloned_position": top_energy_count,
        "unique_emission_energies_in_bank": int(len(np.unique(energy))),
        "actual_neutron_direction_max_unit_norm_error": deviation,
        "mean_source_emission_direction_vector": dipole.tolist(),
        "mean_source_emission_direction_magnitude": float(np.linalg.norm(dipole)),
        "normalized_direction_second_moment": moment.tolist(),
        "direction_second_moment_eigenvalues": eigvals.tolist(),
        "direction_second_moment_deviation_from_isotropic": float(
            np.linalg.norm(moment - np.eye(3) / 3.)
        ),
        "emission_energy_min_eV": float(np.min(energy)),
        "emission_energy_max_eV": float(np.max(energy)),
        "emission_energy_weighted_mean_eV": float(
            np.dot(energy, weight) / total_weight
        ),
        "saved_source_emission_time_min": float(np.min(event_time)),
        "saved_source_emission_time_max": float(np.max(event_time)),
        "actual_source_weight_sum": total_weight,
        "angular_isotropy_or_fission_genealogy_independence_proven": False,
    }, positions


def audit_real_source_angular_physics(summary_file: Path) -> dict:
    """Validate four genuine source trajectories with >=3 matched checkpoints."""
    summary_file = Path(summary_file)
    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    if (payload.get("execution") != "ACTUAL_OPENMC_TRANSPORT"
            or payload.get("assembly") != "core"):
        raise ValueError("requires actual OpenMC core transport summary")
    runs = payload.get("runs")
    batches = payload.get("saved_source_bank_batches")
    terminal = payload.get("batches")
    if (not isinstance(runs, list) or len(runs) < 4
            or not isinstance(batches, list) or len(batches) < 3
            or any(type(n) is not int or n <= 0 for n in batches)
            or any(b >= c for b, c in zip(batches, batches[1:]))
            or type(terminal) is not int or terminal != batches[-1]):
        raise ValueError("invalid independent seed or source checkpoint schedule")
    records = []
    seeds = set()
    hashes = set()
    for run in runs:
        seed = run.get("seed")
        final = Path(run.get("statepoint_path", ""))
        if (type(seed) is not int or seed <= 0 or seed in seeds
                or not final.is_file()
                or _sha256(final) != run.get("statepoint_sha256")):
            raise ValueError("real fission bank seed or statepoint provenance mismatch")
        seeds.add(seed)
        width = len(str(terminal))
        checkpoints = [
            final.parent / f"statepoint.{batch:0{width}d}.h5"
            for batch in batches
        ]
        if checkpoints[-1].resolve() != final.resolve():
            raise ValueError("final saved source checkpoint mismatch")
        snapshot_rows = []
        distinct = []
        for batch, file in zip(batches, checkpoints):
            if not file.is_file():
                raise ValueError("missing actual OpenMC source checkpoint")
            row, positions = read_real_bank_angular_physics(file)
            if (row["seed"] != seed or row["checkpoint_batch"] != batch
                    or row["actual_statepoint_sha256"] in hashes):
                raise ValueError("source checkpoint identity/seed mismatch")
            hashes.add(row["actual_statepoint_sha256"])
            if (snapshot_rows and
                    row["nominal_source_particles"] != snapshot_rows[0]["nominal_source_particles"]):
                raise ValueError("neutron source population changed inside one executed seed")
            snapshot_rows.append(row)
            distinct.append(positions)
        if (snapshot_rows[-1]["actual_statepoint_sha256"]
                != run["statepoint_sha256"]):
            raise ValueError("executed final source bank digest changed")
        carryover = [
            len(before.intersection(after))
            for before, after in zip(distinct, distinct[1:])
        ]
        records.append({
            "seed": seed,
            "checkpoints": snapshot_rows,
            "exact_positions_shared_across_adjacent_saved_checkpoints": carryover,
            "minimum_effective_position_fraction": float(min(
                r["effective_position_fraction_of_nominal_population"]
                for r in snapshot_rows
            )),
            "maximum_exact_position_cloning": int(max(
                r["max_clones_at_exact_same_position"] for r in snapshot_rows
            )),
        })

    final_rows = [record["checkpoints"][-1] for record in records]
    all_rows = [x for r in records for x in r["checkpoints"]]
    return {
        "audit": "REAL_OPENMC_FISSION_BANK_DIRECTION_ENERGY_AND_SPATIAL_CLONING",
        "actual_transport_summary_sha256": _sha256(summary_file),
        "actual_independent_seed_count": len(seeds),
        "matched_checkpoint_batches": batches,
        "source_direction_vectors_unit_normalized": True,
        "source_energy_positive_and_finite": True,
        "smallest_effective_position_fraction": float(min(
            x["effective_position_fraction_of_nominal_population"]
            for x in all_rows
        )),
        "largest_exact_position_clone_multiplicity": int(max(
            x["max_clones_at_exact_same_position"] for x in all_rows
        )),
        "final_bank_effective_position_counts": [
            x["effective_source_position_count_inverse_simpson"]
            for x in final_rows
        ],
        "final_bank_mean_emission_direction_magnitudes": [
            x["mean_source_emission_direction_magnitude"] for x in final_rows
        ],
        "independent_seed_real_source_records": records,
        "nominal_source_population_as_independent_fission_histories": False,
        "near_isotropic_directions_prove_spatial_source_convergence": False,
        "fission_source_stationarity_demonstrated": False,
        "historical_nasa_nerva_core_physically_validated": False,
        "interpretation": (
            "Actual neutron emission directions are unit vectors and angular "
            "moments are computed from saved physical HDF5. Repeated exact "
            "fission-source positions with distinct emission directions/energies "
            "reflect source-bank resampling, NOT independent fission locations. "
            "Exact position carryover is checked across stored checkpoint banks, "
            "but absence of identical coordinates does not prove independent "
            "histories, modal stationarity or credible NERVA criticality. "
            "Emission source time metadata is not reactor transient time."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    report = audit_real_source_angular_physics(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("Actual OpenMC source direction/energy physical HDF5 integrity: PASS")
    print("Final inverse-Simpson effective source positions:",
          report["final_bank_effective_position_counts"])
    print("Worst exact spatial source cloning:",
          report["largest_exact_position_clone_multiplicity"])
    print("Fission-source modal stationarity: NOT ESTABLISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
