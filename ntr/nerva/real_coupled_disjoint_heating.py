"""Stage34: recover disjoint active-generation heating from genuine OpenMC HDF5.

Cumulative OpenMC tally SUM and SUM_SQ snapshots at 30/45/65 provide
non-overlapping active batch intervals [21,30], [31,45], [46,65].
This is a deterministic difference of *native* Monte Carlo tally moments,
not subtraction of correlated SEMs nor a claim of independent generations.
Cold nonstationary source: physical confidence coverage remains unqualified.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .real_coupled_source_temporal import (
    CHECKPOINTS, SEEDS, _hdf5_identity, audit_real_coupled_temporal_source,
)

_DOMAINS = ("mesh", "materials", "particles")


def disjoint_raw_moments(cumulative_sums, cumulative_squared_sums, counts, *, label):
    """Recover nonoverlapping batch sum, sum-of-squares and descriptive variance.

    No SEM of a sum is inferred: OpenMC active batches and bins are correlated.
    """
    s = [np.asarray(v, dtype=np.float64) for v in cumulative_sums]
    q = [np.asarray(v, dtype=np.float64) for v in cumulative_squared_sums]
    n = [int(v) for v in counts]
    if (len(s) != 3 or len(q) != 3 or len(n) != 3 or n != [10, 25, 45]
            or s[0].ndim < 1 or not s[0].size
            or any(x.shape != s[0].shape for x in (*s, *q))
            or any(not np.isfinite(x).all() for x in (*s, *q))
            or any(np.any(x < 0) for x in (*s, *q))):
        raise ValueError("invalid cumulative native heating moments: " + label)
    start_s = np.zeros_like(s[0])
    start_q = np.zeros_like(q[0])
    previous_n = 0
    intervals = []
    for end_s, end_q, end_n in zip(s, q, n):
        dn = end_n - previous_n
        ds = end_s - start_s
        dq = end_q - start_q
        # Roundoff tolerance uses the cumulative magnitude for each bin.
        scale_s = np.maximum(1.0, np.maximum(np.abs(end_s), np.abs(start_s)))
        scale_q = np.maximum(1.0, np.maximum(np.abs(end_q), np.abs(start_q)))
        if np.any(ds < -1e-12 * scale_s) or np.any(dq < -1e-12 * scale_q):
            raise ValueError("native cumulative heating moments decreased: " + label)
        mean = ds / dn
        # Within-window per-bin batch scatter, NOT IID sampling uncertainty.
        centered = dq - np.square(ds) / dn
        eps = 2e-11 * np.maximum(1.0, dq + np.square(ds) / dn)
        if np.any(centered < -eps):
            raise ValueError("native raw heating second moment violates Cauchy: " + label)
        variance = np.maximum(centered, 0.) / (dn-1)
        intervals.append({
            "active_realizations": dn, "raw_sum": ds,
            "raw_sum_sq": dq, "mean": mean,
            "descriptive_sample_batch_variance_not_IID": variance,
        })
        start_s, start_q, previous_n = end_s, end_q, end_n
    if (not np.allclose(sum((v["raw_sum"] for v in intervals)), s[-1],
                        rtol=1e-12, atol=1e-7)
            or not np.allclose(sum((v["raw_sum_sq"] for v in intervals)), q[-1],
                               rtol=1e-12, atol=1e-7)):
        raise ValueError("original cumulative HDF5 moment telescoping mismatch")
    return intervals


def _native_tallies(path: Path) -> dict:
    """Read original OpenMC tally moments, preserving native mesh axis order."""
    import openmc

    identity = _hdf5_identity(path)
    with openmc.StatePoint(str(path), autolink=False) as statepoint:
        by_name = {}
        for tally in statepoint.tallies.values():
            if tally.name in ("nerva_3d_neutronics", "nerva_material_transport",
                              "nerva_particle_heating"):
                if tally.name in by_name:
                    raise ValueError("duplicate original native heating tally")
                by_name[tally.name] = tally
        if set(by_name) != {
                "nerva_3d_neutronics", "nerva_material_transport",
                "nerva_particle_heating"}:
            raise ValueError("missing original neutron/photon deposited-energy tally")
        core = by_name["nerva_3d_neutronics"]
        material = by_name["nerva_material_transport"]
        particle = by_name["nerva_particle_heating"]
        if (len(core.filters) != 1 or
                not isinstance(core.filters[0], openmc.MeshFilter) or
                len(material.filters) != 1 or
                not isinstance(material.filters[0], openmc.MaterialFilter) or
                len(particle.filters) != 1 or
                not isinstance(particle.filters[0], openmc.ParticleFilter)):
            raise ValueError("incompatible native heating filters")
        mesh = core.filters[0].mesh
        if (tuple(mesh.dimension) != (12, 12, 10)
                or not np.isfinite(mesh.lower_left).all()
                or not np.isfinite(mesh.upper_right).all()
                or tuple(str(x) for x in particle.filters[0].bins) !=
                ("neutron", "photon", "electron", "positron")):
            raise ValueError("wrong original material mesh or particle physics")
        active = identity["batch"] - identity["inactive"]
        if any(t.num_realizations != active for t in (core, material, particle)):
            raise ValueError("real OpenMC active generation tally count mismatch")

        def score(tally, *, expanded=False):
            if "heating" not in tally.scores or len(tally.nuclides) != 1:
                raise ValueError("native heating score/nuclide identity missing")
            if expanded:
                index = tuple(tally.scores).index("heating")
                return {
                    kind: np.asarray(
                        tally.get_reshaped_data(value=kind, expand_dims=True)[..., 0, index],
                        dtype=float,
                    )
                    for kind in ("sum", "sum_sq")
                }
            return {
                kind: np.asarray(tally.get_values(scores=["heating"], value=kind),
                                 dtype=float).reshape(-1)
                for kind in ("sum", "sum_sq")
            }

        signature = {
            "mesh_dimensions_xyz": [int(x) for x in mesh.dimension],
            "mesh_lower_left_cm": [float(x) for x in mesh.lower_left],
            "mesh_upper_right_cm": [float(x) for x in mesh.upper_right],
            "material_filter_ids": [
                str(getattr(v, "id", v)) for v in material.filters[0].bins
            ],
            "particle_filter_order": [str(v) for v in particle.filters[0].bins],
            "native_heating_estimators": [
                str(v.estimator) for v in (core, material, particle)
            ],
        }
        return {
            "identity": identity,
            "signature": signature,
            "raw": {
                "mesh": score(core, expanded=True),
                "materials": score(material),
                "particles": score(particle),
            },
        }


def _relative_closure(reference: float, other: float) -> float:
    if reference <= 0 or not math.isfinite(reference) or not math.isfinite(other):
        raise ValueError("zero/invalid disjoint OpenMC deposited heating")
    result = (other - reference) / reference
    if abs(result) > 1e-9:
        raise ValueError("actual disjoint-batch coupled heating partition not conserved")
    return result


def audit_real_disjoint_heating(root: str | Path) -> dict:
    """Reaudit Stage33 original 12 binaries and subtract raw native moments."""
    base = Path(root)
    temporal = audit_real_coupled_temporal_source(base)
    final_heating = json.loads((base/"actual_four_seed_coupled_heating_covariance.json").read_text())
    matching = None
    per_seed = []
    totals = []
    axial_profiles = []
    for seed, accepted in zip(SEEDS, final_heating["per_actual_seed"]):
        records = []
        folder = base / f"seed_{seed}"
        for batch in CHECKPOINTS:
            record = _native_tallies(folder/f"statepoint.{batch}.h5")
            if record["identity"]["seed"] != seed or record["identity"]["batch"] != batch:
                raise ValueError("native temporal heating checkpoint identity mismatch")
            if matching is None:
                matching = record["signature"]
            elif record["signature"] != matching:
                raise ValueError("different physical mesh, material, particle or estimator across checkpoints")
            records.append(record)
        moments = {}
        for domain in _DOMAINS:
            moments[domain] = disjoint_raw_moments(
                [x["raw"][domain]["sum"] for x in records],
                [x["raw"][domain]["sum_sq"] for x in records],
                [x["identity"]["batch"]-20 for x in records],
                label=domain,
            )
        # Original OpenMC native final tally sums must equal actual Stage31
        # reported mean/source times final 45 active tally realizations.
        final_mesh_ev = float(np.sum(records[-1]["raw"]["mesh"]["sum"])/45)
        _relative_closure(accepted["native_total_heating_ev_per_source"], final_mesh_ev)
        seed_windows = []
        window_totals = []
        window_axial = []
        for j, (start, end) in enumerate(((21, 30), (31, 45), (46, 65))):
            mesh_mean = moments["mesh"][j]["mean"]
            material_mean = moments["materials"][j]["mean"]
            particle_mean = moments["particles"][j]["mean"]
            total = float(np.sum(mesh_mean))
            if total <= 0 or not np.isfinite(mesh_mean).all():
                raise ValueError("invalid original temporal deposited heating")
            mat_res = _relative_closure(total, float(np.sum(material_mean)))
            particle_res = _relative_closure(total, float(np.sum(particle_mean)))
            axial = mesh_mean.sum(axis=(0, 1))
            if not np.isclose(float(np.sum(axial)), total, rtol=1e-12, atol=1e-7):
                raise ValueError("native temporal axial energy did not conserve")
            axial_fraction = axial / total
            if np.any(axial_fraction < -1e-12):
                raise ValueError("negative physical axial deposited heating")
            window_totals.append(total)
            window_axial.append(axial_fraction)
            seed_windows.append({
                "active_batch_window": [start, end],
                "active_realizations": end-start+1,
                "heating_ev_per_monte_carlo_start": total,
                "material_mesh_relative_residual": mat_res,
                "all_four_particle_mesh_relative_residual": particle_res,
                "particle_heat_ev_per_source": particle_mean.tolist(),
                "material_heat_ev_per_source": material_mean.tolist(),
                "axial_heating_ev_per_source": axial.tolist(),
                "axial_heating_fraction": axial_fraction.tolist(),
                "no_independent_batch_SEM_inferred": True,
            })
        # Weighted recovery from nonoverlapping windows to native 45-batch
        # actual coupled heating. Do not average 10/15/20-batch means equally.
        combined = math.fsum(x*n for x,n in zip(window_totals,(10,15,20)))/45
        _relative_closure(final_mesh_ev, combined)
        heat_tv = .5*float(np.sum(np.abs(window_axial[0]-window_axial[2])))
        per_seed.append({
            "seed": seed,
            "native_final_statepoint_sha256":
                temporal["per_seed_real_transport"][len(per_seed)]["checkpoints"][-1]["source_bank"]["statepoint_sha256"],
            "disjoint_heating_windows": seed_windows,
            "axial_heating_fraction_TV_first_to_last": heat_tv,
            "source_bank_spatial_TV_30_to_65":
                temporal["source_drift_and_resolution"]["same_seed_temporal_source_variation"][len(per_seed)]["same_seed_TV_batch30_to_65"],
            "weighted_window_heating_recovers_original_final": True,
        })
        totals.append(window_totals)
        axial_profiles.append(window_axial)
    total_arr = np.asarray(totals, dtype=float)
    return {
        "schema": "nuke-actual-coupled-disjoint-active-heating-stage34-v1",
        "evidence": "12_ORIGINAL_OPENMC_0153_COUPLED_NEUTRON_PHOTON_NATIVE_TALLY_SUM_SUMSQ",
        "genuine_openmc_statepoints_reverified": 12,
        "original_seeds": list(SEEDS),
        "active_windows": [[21, 30], [31, 45], [46, 65]],
        "all_raw_moment_telescope_and_final_heat_identical": True,
        "native_filter_identity": matching,
        "per_seed": per_seed,
        "window_mean_heating_across_seeds_ev_per_start": np.mean(total_arr, axis=0).tolist(),
        "window_total_heating_between_seed_sample_SD": np.std(total_arr, axis=0, ddof=1).tolist(),
        "mean_observed_axial_heating_fraction_first_last_TV":
            float(np.mean([x["axial_heating_fraction_TV_first_to_last"] for x in per_seed])),
        "mean_observed_source_spatial_TV_30_to_65":
            temporal["source_drift_and_resolution"]["mean_30_to_65_within_seed_TV"],
        "cumulative_checkpoint_means_are_independent": False,
        "separate_active_generation_batches_statistically_independent": False,
        "cross_bin_within_window_covariances_available": False,
        "fission_source_stationarity_qualified": False,
        "real_hot_nerva_physical_watts_qualified": False,
        "nasa_historical_benchmark_validated": False,
        "notes": [
            "Window heating is disjoint in active-batch index but shares correlated neutron histories.",
            "Particle/material/mesh sum conservation is checked on every actual nonoverlapping window.",
            "OpenMC raw cumulative first/second tally moments telescope without summing SEMs.",
            "Distinct source and heat histograms are different physical observables: no causal inference.",
            "No thermal-power normalization, criticality, or hot material qualification.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_real_disjoint_heating(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("Original native coupled OpenMC checkpoint count:",
          result["genuine_openmc_statepoints_reverified"])
    print("Real disjoint-window heat eV/source:",
          result["window_mean_heating_across_seeds_ev_per_start"])
    print("First-to-last axial heating TV:",
          result["mean_observed_axial_heating_fraction_first_last_TV"])
    print("Stationary hot NASA NERVA source or real watts qualified: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
