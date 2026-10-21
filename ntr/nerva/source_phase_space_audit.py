"""Independent REAL OpenMC fission-source joint position-energy QA.

An apparently stationary fission-source neutron ENERGY spectrum can conceal
a spatially collapsed or drifting eigenvalue fission-source distribution.
Audit the exact same HDF5 source particles jointly in three spatial
coordinates and emission energy, then independently reconcile BOTH marginals
with the separate real spatial and emission-energy reports.

These are data-conservation identities, not tests of historical NERVA
reactor fidelity or fundamental-mode/source stationarity. No material,
geometry, control, enrichment, source spectrum or reactor power is tuned.
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

from .source_energy_audit import SOURCE_ENERGY_EDGES_EV


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(a - b).sum(dtype=np.float64))


def joint_position_energy_bank(
    statepoint: Path, *,
    lower_cm: tuple[float, float, float],
    upper_cm: tuple[float, float, float],
    spatial_sides: tuple[int, ...] = (2, 4, 8),
    edges_eV: tuple[float, ...] = SOURCE_ENERGY_EDGES_EV,
) -> tuple[dict[int, np.ndarray], dict]:
    """Score the actual saved neutron fission source in x/y/z/E.

    Every source particle must be accounted for, without replacing a
    missing energy with a flux-derived distribution. Coarse-to-fine spatial
    grids partition the *same* physical volume and the *same* energy bins.
    """
    lo = np.asarray(lower_cm, dtype=float)
    hi = np.asarray(upper_cm, dtype=float)
    energy_edges = np.asarray(edges_eV, dtype=float)
    if (lo.shape != (3,) or hi.shape != (3,)
            or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(hi <= lo)):
        raise ValueError("invalid shared spatial source envelope")
    if (len(spatial_sides) < 2 or any(
            isinstance(n, bool) or not isinstance(n, int)
            or n < 2 or n > 32 for n in spatial_sides)
            or any(right <= left or right % left != 0 for
                   left, right in zip(spatial_sides, spatial_sides[1:]))):
        raise ValueError("spatial source grids must be strictly nested")
    if (energy_edges.ndim != 1 or len(energy_edges) < 3
            or not np.isfinite(energy_edges).all()
            or energy_edges[0] != 0
            or np.any(np.diff(energy_edges) <= 0)):
        raise ValueError("invalid fission-source energy interval boundaries")
    path = Path(statepoint)
    try:
        with h5py.File(path, "r") as h:
            if (bytes(h.attrs["filetype"]).rstrip(b"\x00") != b"statepoint"
                    or bytes(h["run_mode"][()]).rstrip(b"\x00") != b"eigenvalue"
                    or int(h.attrs["source_present"]) != 1
                    or int(h.attrs["photon_transport"]) != 0):
                raise ValueError("requires saved neutron-only OpenMC eigenvalue source")
            source = h["source_bank"][()]
            seed = int(h["seed"][()])
            batch = int(h["current_batch"][()])
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError("missing complete actual OpenMC neutron source HDF5") from exc

    required = {"r", "E", "wgt"}
    if (source.ndim != 1 or len(source) < 2
            or source.dtype.names is None
            or not required.issubset(source.dtype.names)
            or source["r"].dtype.names != ("x", "y", "z")):
        raise ValueError("incomplete OpenMC structured fission-source record")
    if "particle" in source.dtype.names and np.any(source["particle"] != 0):
        raise ValueError("non-neutron site in neutron-only fission source bank")
    xyz = np.column_stack([
        source["r"][axis] for axis in ("x", "y", "z")
    ]).astype(float)
    E = np.asarray(source["E"], dtype=float)
    w = np.asarray(source["wgt"], dtype=float)
    if (not np.isfinite(xyz).all() or not np.isfinite(E).all()
            or not np.isfinite(w).all() or np.any(w <= 0)
            or np.any(E <= 0)):
        raise ValueError("nonfinite or nonphysical source position/energy/weight")
    if (np.any(xyz < lo) or np.any(xyz > hi)):
        raise ValueError("source outside common physical spatial envelope")
    if np.any(E > energy_edges[-1]):
        raise ValueError("source neutron outside full scoring energy window")
    if seed <= 0 or batch < 1:
        raise ValueError("invalid source seed or batch identity")
    total = float(w.sum(dtype=np.float64))
    fields = {}
    four = np.column_stack((xyz, E))
    for side in spatial_sides:
        spatial = [np.linspace(lo[j], hi[j], side + 1)
                   for j in range(3)]
        hist, _ = np.histogramdd(
            four, bins=(*spatial, energy_edges), weights=w
        )
        counted = float(hist.sum(dtype=np.float64))
        if not math.isclose(counted, total, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("joint source space-energy histogram lost neutrons")
        field = hist / total
        if (not np.isfinite(field).all() or
                not math.isclose(float(field.sum()), 1.,
                                 rel_tol=1e-12, abs_tol=1e-10)):
            raise ValueError("invalid normalized source phase-space distribution")
        fields[side] = field
    return fields, {
        "seed": seed,
        "completed_batch": batch,
        "source_sites": len(E),
        "total_source_weight": total,
        "mean_initial_source_energy_eV": float(np.dot(E, w) / total),
        "source_energy_min_eV": float(E.min()),
        "source_energy_max_eV": float(E.max()),
        "statepoint_sha256": _sha(path),
    }


def _ensemble_stat(field: np.ndarray) -> dict:
    """Describe correlated source-bank changes without treating sites iid."""
    nseeds, nbatches = field.shape[:2]
    between = []
    within = []
    for batch in range(nbatches):
        pair = [
            _tv(field[i, batch], field[j, batch])
            for i, j in combinations(range(nseeds), 2)
        ]
        between.append(float(np.mean(pair)))
    for batch in range(nbatches - 1):
        within.append(float(np.mean([
            _tv(field[i, batch], field[i, batch+1])
            for i in range(nseeds)
        ])))
    return {
        "mean_cross_seed_tv_per_checkpoint": between,
        "mean_within_seed_adjacent_tv_per_interval": within,
    }


def audit_joint_phase_space(
    trajectories: list[list[Path]], *,
    lower_cm: tuple[float, float, float],
    upper_cm: tuple[float, float, float],
    sides: tuple[int, ...] = (2, 4, 8),
    edges_eV: tuple[float, ...] = SOURCE_ENERGY_EDGES_EV,
) -> dict:
    """Score identical actual seed/batch source banks across nested x/y/z/E."""
    if len(trajectories) < 4 or len(trajectories[0]) < 3 or any(
            len(t) != len(trajectories[0]) for t in trajectories):
        raise ValueError("four independent seeds and >=3 matched checkpoints required")
    all_fields = {n: [] for n in sides}
    metadata = []
    seen_hash = set()
    seen_seed = set()
    reference_batches = None
    for paths in trajectories:
        rows = []
        run_field = {n: [] for n in sides}
        for file in paths:
            fields, meta = joint_position_energy_bank(
                Path(file), lower_cm=lower_cm, upper_cm=upper_cm,
                spatial_sides=sides, edges_eV=edges_eV,
            )
            if meta["statepoint_sha256"] in seen_hash:
                raise ValueError("duplicate source-bank statepoint contents")
            seen_hash.add(meta["statepoint_sha256"])
            for side in sides:
                run_field[side].append(fields[side])
            rows.append(meta)
        seeds = {m["seed"] for m in rows}
        if len(seeds) != 1:
            raise ValueError("mixed seeds within source trajectory")
        seed = seeds.pop()
        if seed in seen_seed:
            raise ValueError("duplicate OpenMC independent source seeds")
        seen_seed.add(seed)
        batches = [m["completed_batch"] for m in rows]
        if any(a >= b for a, b in zip(batches, batches[1:])):
            raise ValueError("source checkpoint batches not increasing")
        if reference_batches is None:
            reference_batches = batches
        elif batches != reference_batches:
            raise ValueError("different source batches across seed trajectories")
        metadata.append(rows)
        for side in sides:
            all_fields[side].append(np.stack(run_field[side]))
    resolutions = []
    previous_fields = None
    for side in sides:
        joint = np.stack(all_fields[side])
        spatial = joint.sum(axis=-1)
        energy = joint.sum(axis=(2, 3, 4))
        joint_stats = _ensemble_stat(joint)
        spatial_stats = _ensemble_stat(spatial)
        energy_stats = _ensemble_stat(energy)
        # Total-variation contractivity under marginalization is exact.
        for key in ("mean_cross_seed_tv_per_checkpoint",
                    "mean_within_seed_adjacent_tv_per_interval"):
            for ref in (spatial_stats, energy_stats):
                if any(a + 3e-12 < b for a, b in
                       zip(joint_stats[key], ref[key])):
                    raise ValueError("source joint-space marginal TV exceeds joint TV")
        if previous_fields is not None:
            factor = side // previous_fields.shape[2]
            smaller = joint.reshape(
                joint.shape[0], joint.shape[1],
                previous_fields.shape[2], factor,
                previous_fields.shape[3], factor,
                previous_fields.shape[4], factor,
                joint.shape[-1]
            ).sum(axis=(3, 5, 7))
            if not np.allclose(smaller, previous_fields, rtol=0, atol=3e-12):
                raise ValueError("fine source grid does not conserve coarse 4D phase space")
        previous_fields = joint
        resolutions.append({
            "spatial_bins_per_axis": side,
            "number_of_joint_spatial_energy_bins": int(np.prod(joint.shape[2:])),
            "joint": joint_stats,
            "spatial_marginal": spatial_stats,
            "energy_marginal": energy_stats,
        })
    baseline = resolutions[0]["energy_marginal"]
    for row in resolutions[1:]:
        for key in baseline:
            if not np.allclose(row["energy_marginal"][key], baseline[key],
                               rtol=0, atol=3e-12):
                raise ValueError("emission-energy marginal differs by spatial grid")
    return {
        "analysis": "ACTUAL_OPENMC_SAME_SOURCE_PARTICLES_JOINT_X_Y_Z_E",
        "independent_seed_count": len(metadata),
        "saved_source_batches": reference_batches,
        "energy_edges_eV": [float(v) for v in edges_eV],
        "source_statepoint_sha256": [
            [m["statepoint_sha256"] for m in rows] for rows in metadata
        ],
        "per_seed_source_evidence": metadata,
        "resolutions": resolutions,
        "energy_spatial_marginal_and_nested_conservation_passed": True,
        "fission_source_stationarity_validated": False,
        "core_neutron_transport_physics_historically_validated": False,
        "interpretation": (
            "All neutron source phase-space weight is accounted for. "
            "Joint x/y/z/E total variation is never smaller than either "
            "spatial or energy marginal and coarsening conserves weights. "
            "Highly cloned source banks and within-seed genealogical "
            "dependence invalidate independent-neutron p-values. "
            "Physical eigenmode convergence and NASA fidelity remain unverified."
        ),
    }


def audit_real_joint_summary(
    summary_path: Path, *,
    spatial_reference: Path,
    energy_reference: Path,
) -> dict:
    summary_path = Path(summary_path)
    original = json.loads(summary_path.read_text(encoding="utf-8"))
    if (original.get("execution") != "ACTUAL_OPENMC_TRANSPORT"
            or original.get("assembly") != "core"):
        raise ValueError("requires genuine OpenMC core transport summary")
    runs = original.get("runs", [])
    batches = original.get("saved_source_bank_batches", [])
    total = original.get("batches")
    if (len(runs) < 4 or not isinstance(total, int) or total < 3
            or len(batches) < 3 or batches[-1] != total
            or any(not isinstance(b, int) or b < 1 for b in batches)
            or any(a >= b for a, b in zip(batches, batches[1:]))):
        raise ValueError("invalid actual OpenMC independent seed schedule")
    shape = original.get("independent_seed_source_bank_shape_diagnostics")
    if not isinstance(shape, dict):
        raise ValueError("missing actual OpenMC spatial source envelope")
    lower = tuple(shape["spatial_lower_cm"])
    upper = tuple(shape["spatial_upper_cm"])
    digits = len(str(total))
    paths = []
    for run in runs:
        final = Path(run["statepoint_path"])
        if (not final.is_file() or
                run.get("statepoint_sha256") != _sha(final)):
            raise ValueError("real statepoint path or SHA-256 mismatch")
        snapshots = [
            final.parent / f"statepoint.{batch:0{digits}d}.h5"
            for batch in batches
        ]
        if snapshots[-1].resolve() != final.resolve():
            raise ValueError("OpenMC final saved source checkpoint mismatch")
        paths.append(snapshots)
    report = audit_joint_phase_space(paths, lower_cm=lower, upper_cm=upper)
    if [entry[0]["seed"] for entry in report["per_seed_source_evidence"]] != [
            run["seed"] for run in runs]:
        raise ValueError("genuine HDF5 neutron source seed mismatch")

    spatial = json.loads(Path(spatial_reference).read_text(encoding="utf-8"))
    energy = json.loads(Path(energy_reference).read_text(encoding="utf-8"))
    if (not spatial.get("grid_resolution_audit_passed")
            or spatial.get("source_stationarity_validated") is not False
            or energy.get("fission_source_spectral_stationarity_validated") is not False
            or energy.get("neutron_energy_bin_edges_eV") != report["energy_edges_eV"]
            or spatial.get("all_checkpoint_statepoint_sha256", None) is not None):
        # Each spatial-resolution record has its own raw SHA matrix; the
        # top-level spatial audit does not duplicate it.
        raise ValueError("invalid physical reference source audits")
    hashes = report["source_statepoint_sha256"]
    if (energy.get("real_transport_summary_sha256") != _sha(summary_path)
            or spatial.get("real_transport_summary_sha256") != _sha(summary_path)
            or any(row["all_checkpoint_statepoint_sha256"] != hashes
                   for row in spatial["resolutions"])):
        raise ValueError("joint and spatial/energy audits have different statepoints")
    for resolution, prior in zip(report["resolutions"], spatial["resolutions"]):
        if resolution["spatial_bins_per_axis"] != prior["bins_per_axis"]:
            raise ValueError("spatial reference resolution different from joint audit")
        joint_space = resolution["spatial_marginal"][
            "mean_cross_seed_tv_per_checkpoint"]
        prior_space = prior["mean_cross_seed_tv_per_checkpoint"]
        if not np.allclose(joint_space, prior_space, rtol=0, atol=3e-11):
            raise ValueError("actual spatial source marginal differs from independent audit")
        observed = resolution["energy_marginal"][
            "mean_cross_seed_tv_per_checkpoint"]
        expected = energy["mean_cross_seed_energy_spectrum_tv_by_checkpoint"]
        if not np.allclose(observed, expected, rtol=0, atol=3e-11):
            raise ValueError("actual source energy marginal differs from independent audit")
    report["real_transport_summary_sha256"] = _sha(summary_path)
    report["real_spatial_source_audit_sha256"] = _sha(Path(spatial_reference))
    report["real_energy_source_audit_sha256"] = _sha(Path(energy_reference))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--spatial-audit", type=Path, required=True)
    p.add_argument("--energy-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit_real_joint_summary(
        args.summary, spatial_reference=args.spatial_audit,
        energy_reference=args.energy_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("Genuine OpenMC joint source x/y/z/E numerical QA: PASS")
    for row in report["resolutions"]:
        print("  spatial side", row["spatial_bins_per_axis"],
              "joint TV", row["joint"]["mean_cross_seed_tv_per_checkpoint"][-1],
              "spatial TV", row["spatial_marginal"][
                  "mean_cross_seed_tv_per_checkpoint"][-1],
              "energy TV", row["energy_marginal"][
                  "mean_cross_seed_tv_per_checkpoint"][-1])
    print("Fission-source spatial/energy stationarity validated: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
