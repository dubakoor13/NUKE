"""Adversarial synthetic regression of per-seed correlated neutron accounting.

These four values are invented mathematical fixtures, NOT actual transport.
"""
from __future__ import annotations

from copy import deepcopy

from .neutron_economy_uncertainty import quantify_seed_economy_uncertainty, SCORES


def rejected(rows, message):
    try:
        quantify_seed_economy_uncertainty(rows)
    except (ValueError, KeyError) as exc:
        assert message in str(exc), (message, str(exc))
    else:
        raise AssertionError("invalid neutron accounting passed: " + message)


def main() -> int:
    a = [.04, .05, .06, .045]
    scatter = [.06, .03, .08, .02]
    rows = []
    for i in range(4):
        leakage = 1 + scatter[i] - a[i]
        vals = [a[i], leakage, scatter[i], a[i] + .001,
                .012 + .0005*i, .0118 + .0005*i]
        rows.append({"seed": 101 + i,
                     "weighted_active_generation_economy": dict(zip(SCORES, vals))})
    report = quantify_seed_economy_uncertainty(rows)
    assert report["joint_neutron_balance_covariance_standard_error"] < 1e-9
    assert report["incorrect_independent_terms_balance_standard_error"] > .001
    assert abs(report["paired_tracklength_minus_analog_absorption"]["mean_difference"] - .001) < 1e-15
    assert report["paired_tracklength_minus_analog_absorption"]["between_seed_paired_standard_error"] < 1e-15
    assert len(report["between_seed_sample_covariance"]) == len(SCORES)
    assert report["conditional_t95_intervals_per_starting_neutron"][SCORES[0]][0] < .04
    assert report["formal_monte_carlo_confidence_coverage_established"] is False
    invalid = deepcopy(rows)
    invalid[1]["seed"] = invalid[0]["seed"]
    rejected(invalid, "duplicate or invalid")
    rejected(rows[:3], "exactly four")
    invalid = deepcopy(rows)
    invalid[0]["weighted_active_generation_economy"][SCORES[0]] = float("nan")
    rejected(invalid, "invalid per-seed")
    invalid = deepcopy(rows)
    invalid[0]["weighted_active_generation_economy"][SCORES[1]] += .1
    rejected(invalid, "conservation identity")
    invalid = deepcopy(rows)
    invalid[0]["weighted_active_generation_economy"].pop(SCORES[4])
    rejected(invalid, "missing numerical")
    print("Correlated four-seed neutron economy and paired estimators: PASS (synthetic only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
