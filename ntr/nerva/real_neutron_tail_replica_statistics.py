"""Four-replica uncertainty of REAL fixed-source OpenMC neutron energy tails.

Reads the eight-statepoint HDF5-derived JSON from
real_neutron_energy_tail_attribution. This postprocessor does not run OpenMC,
invent cross sections, fit an NERVA core, or convert statistical detection
into criticality, power, or NASA historical validation.

A partition closure test asks whether recorded scores add up. This module
asks a different question: whether low/high incident-energy tracklength tails
are reproducible across four independent seed pairs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# Two-sided 95% Student-t critical value, degrees of freedom = 4 - 1.
T95_DF3 = 3.182446305284263
SCORE_KEYS = ("below_lower_bound", "above_upper_bound")
MODES = ("off", "on")
EXPECTED_EDGES_EV = (0.0, 1.0e-5, 2.0e7, 1.0e10)


def four_replica_statistics(values, *, signed: bool = False) -> dict:
    """Unadjusted between-seed t interval, NOT a per-history tally error.

    n=4 makes these screening estimates noisy. Exactly zero observed tails
    cannot establish that an unobserved rare tail is physically impossible.
    """
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (4,) or not np.isfinite(a).all():
        raise ValueError("four finite independent replica estimates required")
    if not signed and np.any(a < 0.0):
        raise ValueError("negative physically scored neutron tail tracklength")
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    sem = sd / 2.0
    lo = mean - T95_DF3 * sem
    hi = mean + T95_DF3 * sem
    # Constant four-point samples have no identifiable nonzero between-seed
    # variance. A degenerate CI is NOT presented as strong statistical proof.
    variation_measurable = sd > 0.0
    return {
        "samples_per_source": [float(x) for x in a],
        "mean_per_source": mean,
        "sample_standard_deviation_per_source": sd,
        "between_replica_standard_error_per_source": sem,
        "exploratory_unadjusted_t95_interval_per_source": [lo, hi],
        "independent_replica_count": 4,
        "replica_variance_estimable": variation_measurable,
        "zero_excluded_by_exploratory_interval": (
            bool((lo > 0.0 or hi < 0.0) and variation_measurable)
        ),
        "all_four_observations_zero": bool(np.all(a == 0.0)),
    }


def summarize_real_tail_replicas(audit: dict) -> dict:
    """Require real experiment identity before summarizing replica variation."""
    if (not isinstance(audit, dict)
            or audit.get("audit") != "GENUINE_OPENMC_NEUTRON_ENERGY_TAIL_EXPLICIT_REBIN"
            or audit.get("native_hdf5_statepoint_count") != 8
            or audit.get("independent_seed_pairs") != 4
            or audit.get("all_actual_material_energy_partitions_numerically_conserved") is not True
            or audit.get("all_measured_nonroundoff_legacy_residuals_accounted_for_by_scored_tails") is not True
            or audit.get("hot_nuclear_data_physically_qualified") is not False
            or audit.get("historical_nasa_nerva_core_validated") is not False):
        raise ValueError("real fixed-source neutron energy-tail provenance or closure missing")
    rows = audit.get("per_seed")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("four paired physical seed records required")
    seeds = []
    state_hashes = []
    physical_identity = None
    tails = {mode: {key: [] for key in SCORE_KEYS} for mode in MODES}
    for record in rows:
        seed = record.get("seed")
        if type(seed) is not int or seed <= 0 or seed in seeds:
            raise ValueError("repeated or invalid independent neutron seed")
        seeds.append(seed)
        for mode in MODES:
            point = record.get(mode)
            if not isinstance(point, dict) or point.get("seed") != seed:
                raise ValueError("missing physical HDF5 seed or transport mode")
            if point.get("daughters_transported") is not (mode == "on"):
                raise ValueError("fission daughter transport status mismatch")
            if point.get("independent_window_partition_conserved") is not True:
                raise ValueError("unclosed native neutron energy tail partition")
            if point.get("evaluated_local_nuclear_heating_validated") is not False:
                raise ValueError("unqualified local nuclear heating claim")
            if point.get("material_count", 0) < 2:
                raise ValueError("physical material inventory incomplete")
            edges = point.get("energy_bounds_eV")
            if (not isinstance(edges, list) or len(edges) != 4
                    or tuple(edges) != EXPECTED_EDGES_EV):
                raise ValueError("executed neutron energy-filter bounds changed")
            hashes = tuple(point.get(name) for name in
                           ("geometry_sha256", "materials_sha256",
                            "tallies_sha256"))
            if any(not isinstance(h, str) or len(h) != 64
                   or any(c not in "0123456789abcdef" for c in h)
                   for h in hashes):
                raise ValueError("missing exact OpenMC geometry/material/tally identity")
            identity = (hashes, point["material_count"])
            if physical_identity is None:
                physical_identity = identity
            elif physical_identity != identity:
                raise ValueError("physical OpenMC model or tally changed between seeds")
            state_hash = point.get("statepoint_sha256")
            if (not isinstance(state_hash, str) or len(state_hash) != 64
                    or any(c not in "0123456789abcdef" for c in state_hash)
                    or state_hash in state_hashes):
                raise ValueError("duplicate or missing actual OpenMC statepoint SHA256")
            state_hashes.append(state_hash)
            source_tails = point.get("explicit_tail_score_per_source")
            if not isinstance(source_tails, dict):
                raise ValueError("actual OpenMC low/high neutron score missing")
            for key in SCORE_KEYS:
                v = source_tails.get(key)
                if (type(v) not in (int, float) or not math.isfinite(v)
                        or v < 0.0):
                    raise ValueError("invalid physical low/high neutron tail flux")
                tails[mode][key].append(float(v))
    stats = {
        mode: {key: four_replica_statistics(tails[mode][key])
               for key in SCORE_KEYS}
        for mode in MODES
    }
    paired = {
        key: four_replica_statistics(
            np.asarray(tails["on"][key]) - np.asarray(tails["off"][key]),
            signed=True,
        )
        for key in SCORE_KEYS
    }
    return {
        "audit": "REAL_OPENMC_NEUTRON_TAIL_FOUR_PAIR_STATISTICAL_SCREEN",
        "source_audit": audit["audit"],
        "independent_seed_pairs": 4,
        "native_hdf5_statepoints": 8,
        "ordered_seed_ids": seeds,
        "individually_distinct_statepoint_sha256_count": len(state_hashes),
        "geometry_materials_and_tallies_byte_identical_across_all_eight": True,
        "energy_bounds_eV": list(EXPECTED_EDGES_EV),
        "units": "native neutron tracklength tally per prescribed external source neutron",
        "tail_flux_by_fission_daughter_mode": stats,
        "paired_on_minus_off": paired,
        "confidence_interval": "exploratory two-sided 95% t; n=4, df=3, not multiple-comparison adjusted",
        "no_detected_tail_not_a_physical_zero_bound": True,
        "same_seed_pairing_does_not_mean_identical_particle_histories": True,
        "nuclear_heating_or_power_physically_qualified": False,
        "eigenvalue_or_fission_source_convergence_validated": False,
        "nasa_historical_nerva_validated": False,
        "interpretation": (
            "Four independent transport seeds quantify between-run variation "
            "in directly scored neutron tracklength tails. The paired contrast "
            "is descriptive, not a criticality or hot-core result. Identical or "
            "zero recorded replicas give no statistical bound on rare unseen "
            "neutron energies. No multiplicity correction or source convergence "
            "claim is implied by these exploratory intervals."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    audit = json.loads(args.input.read_text(encoding="utf-8"))
    result = summarize_real_tail_replicas(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("REAL OpenMC energy-tail four-seed uncertainty report: generated")
    for mode in MODES:
        low = result["tail_flux_by_fission_daughter_mode"][mode]["below_lower_bound"]
        print(mode, "low-energy flux tail mean/SEM:", low["mean_per_source"],
              low["between_replica_standard_error_per_source"])
    print("Historical NERVA core and elevated-temperature physics: NOT QUALIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
