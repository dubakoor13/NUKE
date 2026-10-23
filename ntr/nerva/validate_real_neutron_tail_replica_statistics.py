"""Adversarial, synthetic tests for real-statepoint replica statistics only."""
from __future__ import annotations

from copy import deepcopy
import math

from .real_neutron_tail_replica_statistics import (
    T95_DF3, four_replica_statistics, summarize_real_tail_replicas,
)


def reject(fn, message: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert message in str(exc), (message, str(exc))
    else:
        raise AssertionError("invalid real OpenMC replica evidence was accepted")


def example() -> dict:
    lows_off = (2.0e-8, 4.0e-8, 6.0e-8, 8.0e-8)
    lows_on = (3.0e-8, 7.0e-8, 7.0e-8, 13.0e-8)
    rows = []
    for i, seed in enumerate((13007, 23011, 33013, 43019)):
        record = {"seed": seed}
        for mode in ("off", "on"):
            record[mode] = {
                "seed": seed,
                "daughters_transported": mode == "on",
                "independent_window_partition_conserved": True,
                "evaluated_local_nuclear_heating_validated": False,
                "material_count": 9,
                "energy_bounds_eV": [0.0, 1.0e-5, 2.0e7, 1.0e10],
                "geometry_sha256": "a" * 64,
                "materials_sha256": "b" * 64,
                "tallies_sha256": "c" * 64,
                "statepoint_sha256": format(2 * i + (mode == "on") + 1, "064x"),
                "explicit_tail_score_per_source": {
                    "below_lower_bound": (
                        lows_on[i] if mode == "on" else lows_off[i]
                    ),
                    "above_upper_bound": 0.0,
                },
            }
        rows.append(record)
    return {
        "audit": "GENUINE_OPENMC_NEUTRON_ENERGY_TAIL_EXPLICIT_REBIN",
        "native_hdf5_statepoint_count": 8,
        "independent_seed_pairs": 4,
        "all_actual_material_energy_partitions_numerically_conserved": True,
        "all_measured_nonroundoff_legacy_residuals_accounted_for_by_scored_tails": True,
        "hot_nuclear_data_physically_qualified": False,
        "historical_nasa_nerva_core_validated": False,
        "per_seed": rows,
    }


def main() -> int:
    assert abs(T95_DF3 - 3.182446305284263) < 1e-12
    result = summarize_real_tail_replicas(example())
    assert result["native_hdf5_statepoints"] == 8
    assert result["individually_distinct_statepoint_sha256_count"] == 8
    off = result["tail_flux_by_fission_daughter_mode"]["off"]["below_lower_bound"]
    assert math.isclose(off["mean_per_source"], 5e-8, rel_tol=1e-12)
    assert off["zero_excluded_by_exploratory_interval"]
    contrast = result["paired_on_minus_off"]["below_lower_bound"]
    assert math.isclose(contrast["mean_per_source"], 2.5e-8, rel_tol=1e-12)
    assert not contrast["zero_excluded_by_exploratory_interval"]
    zero = result["tail_flux_by_fission_daughter_mode"]["off"]["above_upper_bound"]
    assert zero["all_four_observations_zero"]
    assert not zero["replica_variance_estimable"]
    assert not zero["zero_excluded_by_exploratory_interval"]
    assert not result["nuclear_heating_or_power_physically_qualified"]

    # Identical positive replicas are NOT assigned artificial zero-variance
    # certainty about a physical effect.
    constant = four_replica_statistics([1e-8] * 4)
    assert not constant["replica_variance_estimable"]
    assert not constant["zero_excluded_by_exploratory_interval"]
    reject(lambda: four_replica_statistics([1.0] * 3),
           "four finite independent")
    reject(lambda: four_replica_statistics([1.0, math.nan, 3.0, 4.0]),
           "four finite independent")
    reject(lambda: four_replica_statistics([-1.0, 2.0, 3.0, 4.0]),
           "negative physically scored")

    bad = deepcopy(example())
    bad["per_seed"][1]["seed"] = bad["per_seed"][0]["seed"]
    reject(lambda: summarize_real_tail_replicas(bad),
           "repeated or invalid independent")
    bad = deepcopy(example())
    bad["per_seed"][2]["on"]["statepoint_sha256"] = (
        bad["per_seed"][0]["off"]["statepoint_sha256"]
    )
    reject(lambda: summarize_real_tail_replicas(bad),
           "duplicate or missing actual")
    bad = deepcopy(example())
    bad["per_seed"][1]["off"]["materials_sha256"] = "e" * 64
    reject(lambda: summarize_real_tail_replicas(bad),
           "physical OpenMC model or tally changed")
    bad = deepcopy(example())
    bad["per_seed"][3]["on"]["daughters_transported"] = False
    reject(lambda: summarize_real_tail_replicas(bad),
           "fission daughter transport status")
    bad = deepcopy(example())
    bad["per_seed"][0]["on"]["explicit_tail_score_per_source"]["below_lower_bound"] = -1e-9
    reject(lambda: summarize_real_tail_replicas(bad),
           "invalid physical low/high")
    bad = deepcopy(example())
    bad["per_seed"][0]["off"]["energy_bounds_eV"][2] = 1e7
    reject(lambda: summarize_real_tail_replicas(bad),
           "executed neutron energy-filter bounds")
    bad = deepcopy(example())
    bad["all_measured_nonroundoff_legacy_residuals_accounted_for_by_scored_tails"] = False
    reject(lambda: summarize_real_tail_replicas(bad),
           "real fixed-source neutron energy-tail provenance")
    print("Four real-OpenMC seed-pair tail statistics: PASS (synthetic input tests)")
    print("Physical evidence must come from actual eight-statepoint CI artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
