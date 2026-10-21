"""Energy-resolved fission-source QA on REAL OpenMC eigenvalue source banks.

Spatial source stationarity does not establish fission-neutron energy
stationarity. Read the actual source_bank.E and source_bank.wgt arrays at
successive statepoints from four genuinely independent seeds, preserving
eigenvalue-history correlations and raw SHA-256 file provenance.

The histogram bins are diagnostic scores, NOT source physics definitions,
reactor-design inputs, or optimization. A stable spectrum cannot prove
fundamental-mode spatial source convergence or NASA NERVA fidelity.
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


# Predeclared source-neutron scoring grid: 0 to 30 MeV, subdividing the
# typical fission-emission spectrum. No out-of-range neutrons are dropped.
SOURCE_ENERGY_EDGES_EV = (
    0., 1.e5, 3.e5, 6.e5, 1.e6, 2.e6, 3.e6, 5.e6, 1.e7, 3.e7
)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _string(value) -> bytes:
    try:
        return bytes(value).rstrip(b"\x00")
    except (ValueError, TypeError) as error:
        raise ValueError("invalid OpenMC statepoint HDF5 bytestring") from error


def read_source_energy(
    path: Path,
    *, edges_eV: tuple[float, ...] = SOURCE_ENERGY_EDGES_EV
) -> tuple[np.ndarray, dict]:
    """Extract authentic saved source-site neutron energies and weights.

    Missing E, invalid energy/weight, or an incorrectly clipped scoring
    range rejects the file. Never substitute mesh flux energy or artificial
    spectrum shapes for the initial fission-source particles.
    """
    edges = np.asarray(edges_eV, dtype=float)
    if (edges.ndim != 1 or len(edges) < 3
            or not np.isfinite(edges).all() or edges[0] != 0
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("source energy group boundaries invalid")
    path = Path(path)
    try:
        with h5py.File(path, "r") as state:
            if (_string(state.attrs.get("filetype")) != b"statepoint"
                    or _string(state["run_mode"][()]) != b"eigenvalue"):
                raise ValueError("source energies require OpenMC eigenvalue statepoint")
            if int(state.attrs.get("source_present", 0)) != 1:
                raise ValueError("OpenMC eigenvalue source bank was not saved")
            sites = state["source_bank"][()]
            if (sites.ndim != 1 or sites.size < 2
                    or sites.dtype.names is None
                    or not {"E", "wgt", "r"}.issubset(sites.dtype.names)):
                raise ValueError("not a complete OpenMC fission source-site bank")
            energy = np.asarray(sites["E"], dtype=float)
            weight = np.asarray(sites["wgt"], dtype=float)
            seed = int(state["seed"][()])
            batch = int(state["current_batch"][()])
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError("unable to read complete OpenMC energy source bank") from exc
    if seed <= 0 or batch <= 0:
        raise ValueError("invalid seed/batch in actual source statepoint")
    if (not np.isfinite(energy).all() or not np.isfinite(weight).all()
            or np.any(energy <= 0) or np.any(weight <= 0)):
        raise ValueError("nonphysical neutron source energy or weight")
    if np.any(energy < edges[0]) or np.any(energy > edges[-1]):
        raise ValueError("source neutron energy outside full scoring coverage")
    histogram = np.histogram(energy, bins=edges, weights=weight)[0]
    total = float(np.sum(weight, dtype=np.float64))
    accounted = float(np.sum(histogram, dtype=np.float64))
    if total <= 0 or not math.isclose(total, accounted, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError("neutron source energy histogram lost particle weight")
    fractions = histogram / total
    energy_mean = float(np.dot(energy, weight) / total)
    return fractions, {
        "seed": seed,
        "completed_batch": batch,
        "source_site_count": int(len(energy)),
        "total_source_weight": total,
        "weighted_mean_energy_eV": energy_mean,
        "minimum_source_energy_eV": float(energy.min()),
        "maximum_source_energy_eV": float(energy.max()),
        "statepoint_sha256": _digest(path),
    }


def assess_source_energy_trajectories(
    trajectories: list[list[Path]], *,
    edges_eV: tuple[float, ...] = SOURCE_ENERGY_EDGES_EV
) -> dict:
    """Compare >=3 checkpoints for >=4 independent physical OpenMC seeds.

    Between-seed spectra are independently initialized, but within-seed
    generations and many particles within a bank share fission ancestry.
    Therefore spectral TV and energy means are DESCRIPTIVE only, not
    calibrated hypothesis tests or source-convergence acceptance evidence.
    """
    if not isinstance(trajectories, list) or len(trajectories) < 4:
        raise ValueError("at least four independent OpenMC seed trajectories required")
    count = len(trajectories[0])
    if count < 3 or any(len(run) != count for run in trajectories):
        raise ValueError("same three or more source-bank checkpoints required")
    all_meta = []
    all_fraction = []
    saved_batches = None
    seed_numbers = set()
    digests = set()
    for paths in trajectories:
        bank = []
        metadata = []
        for p in paths:
            fractions, meta = read_source_energy(p, edges_eV=edges_eV)
            if meta["statepoint_sha256"] in digests:
                raise ValueError("duplicate statepoint HDF5 source bank reused")
            digests.add(meta["statepoint_sha256"])
            bank.append(fractions)
            metadata.append(meta)
        seeds = {x["seed"] for x in metadata}
        if len(seeds) != 1:
            raise ValueError("source snapshots from mixed random seeds")
        seed = seeds.pop()
        if seed in seed_numbers:
            raise ValueError("duplicate independent OpenMC source seed")
        seed_numbers.add(seed)
        batches = [x["completed_batch"] for x in metadata]
        if any(a >= b for a, b in zip(batches, batches[1:])):
            raise ValueError("nonincreasing neutron source checkpoint batches")
        if saved_batches is None:
            saved_batches = batches
        elif batches != saved_batches:
            raise ValueError("different source checkpoint schedule between seeds")
        all_meta.append(metadata)
        all_fraction.append(np.stack(bank))

    fields = np.stack(all_fraction)
    n_seeds, n_batches = fields.shape[:2]
    within_seed_tv = 0.5 * np.abs(np.diff(fields, axis=1)).sum(axis=-1)
    within_seed_first_to_last = 0.5 * np.abs(fields[:, -1] - fields[:, 0]).sum(axis=-1)
    cross_seed_mean = []
    cross_seed_max = []
    for i in range(n_batches):
        tvs = [
            0.5 * float(np.abs(fields[a, i] - fields[b, i]).sum())
            for a, b in combinations(range(n_seeds), 2)
        ]
        cross_seed_mean.append(float(np.mean(tvs)))
        cross_seed_max.append(float(np.max(tvs)))
    pooled = fields.mean(axis=0)
    pooled_change = 0.5 * np.abs(np.diff(pooled, axis=0)).sum(axis=-1)
    run_summaries = []
    for i, meta in enumerate(all_meta):
        run_summaries.append({
            "seed": meta[0]["seed"],
            "snapshots": meta,
            "within_seed_adjacent_energy_spectrum_tv": within_seed_tv[i].tolist(),
            "initial_to_final_energy_spectrum_tv": float(within_seed_first_to_last[i]),
        })
    return {
        "analysis": "GENUINE_OPENMC_FISSION_SOURCE_ENERGY_HDF5",
        "neutron_energy_bin_edges_eV": [float(x) for x in edges_eV],
        "independent_seed_count": n_seeds,
        "saved_source_batches": saved_batches,
        "mean_cross_seed_energy_spectrum_tv_by_checkpoint": cross_seed_mean,
        "max_cross_seed_energy_spectrum_tv_by_checkpoint": cross_seed_max,
        "mean_within_seed_adjacent_energy_spectrum_tv": within_seed_tv.mean(axis=0).tolist(),
        "pooled_adjacent_energy_spectrum_tv": pooled_change.tolist(),
        "final_cross_seed_energy_spectrum_tv": cross_seed_mean[-1],
        "mean_initial_to_final_within_seed_energy_spectrum_tv": float(
            within_seed_first_to_last.mean()
        ),
        "runs": run_summaries,
        "fission_source_spectral_stationarity_validated": False,
        "fission_source_spatial_stationarity_validated": False,
        "nasa_nerva_physical_reactor_validated": False,
        "interpretation": (
            "These are real saved source neutron energy spectra. Histograms "
            "conserve all positive initial neutron weights in a predeclared "
            "0–30 MeV window. This is not a test of thermal flux, "
            "neutron absorption energy, deposited heat, or evaluated KERMA. "
            "Within-generation and across-generation source-site genealogies "
            "are correlated. Even stable fission-source energy spectra do "
            "not establish spatial stationarity, reliable k-effective, "
            "a hot NERVA model, or comparison to measured NASA benchmarks."
        ),
    }


def audit_actual_source_energy_summary(summary_path: Path) -> dict:
    path = Path(summary_path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if (summary.get("execution") != "ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly") != "core"):
        raise ValueError("requires genuine core OpenMC transport summary")
    runs = summary.get("runs", [])
    batches = summary.get("saved_source_bank_batches", [])
    total = summary.get("batches")
    if (not isinstance(runs, list) or len(runs) < 4
            or not isinstance(total, int) or total < 3
            or not isinstance(batches, list) or len(batches) < 3
            or batches[-1] != total
            or any(not isinstance(b, int) or b < 1 for b in batches)
            or any(a >= b for a, b in zip(batches, batches[1:]))):
        raise ValueError("invalid independent-seed OpenMC checkpoint provenance")
    digits = len(str(total))
    paths = []
    for run in runs:
        final = Path(run["statepoint_path"])
        if (not final.is_file() or run.get("statepoint_sha256") != _digest(final)):
            raise ValueError("final OpenMC source statepoint SHA-256 mismatch")
        snapshots = [
            final.parent / f"statepoint.{batch:0{digits}d}.h5"
            for batch in batches
        ]
        if snapshots[-1].resolve() != final.resolve():
            raise ValueError("real checkpoint paths disagree with final statepoint")
        paths.append(snapshots)
    report = assess_source_energy_trajectories(paths)
    if [item["seed"] for item in report["runs"]] != [r["seed"] for r in runs]:
        raise ValueError("different source seeds from real OpenMC summary")
    report["real_transport_summary_sha256"] = _digest(path)
    report["evaluated_cross_sections_xml_sha256"] = summary.get(
        "cross_sections_xml_sha256"
    )
    return report


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--summary", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    report = audit_actual_source_energy_summary(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("Actual neutron fission-source energy spectrum HDF5 integrity: PASS")
    print("Final cross-seed energy-spectrum TV:", report[
        "final_cross_seed_energy_spectrum_tv"])
    print("Source spatial/spectral stationarity validated: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
