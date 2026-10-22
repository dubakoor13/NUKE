"""Correlated four-seed uncertainty diagnostics for actual OpenMC neutron economy.

Work with independently executed seed-level means, NOT correlated eigenvalue
batches or individual histories. Preserve the joint absorption/leakage/
secondary-scattering covariance and paired analog/tracklength comparisons.
All intervals are descriptive/conditional: neither spatial source stationarity
nor NASA reactor fidelity follows from four apparently independent RNG seeds.
"""
from __future__ import annotations

import math

import numpy as np


SCORES = (
    "analog_absorption_per_starting_neutron",
    "native_vacuum_leakage_per_starting_neutron",
    "net_multiplicative_scattering_per_starting_neutron",
    "tracklength_absorption_per_starting_neutron",
    "tracklength_banked_fission_neutrons_per_starting_neutron",
    "analog_banked_fission_neutrons_per_starting_neutron",
)
# Two-sided 95% Student t critical value, 3 degrees of freedom (exactly
# four independent seed means). Never interpret as a genealogy-aware CI.
T95_DF3 = 3.182446305284263


def quantify_seed_economy_uncertainty(per_seed: list[dict]) -> dict:
    """Preserve joint covariance and within-seed pairing in neutron scores.

    Requires four seed means each formed by generation-weighting five
    disjoint observed active windows from the SAME OpenMC calculation.
    """
    if not isinstance(per_seed, list) or len(per_seed) != 4:
        raise ValueError("requires exactly four independent neutron seed means")
    seeds = [entry.get("seed") for entry in per_seed]
    if (any(type(s) is not int or s <= 0 for s in seeds)
            or len(set(seeds)) != 4):
        raise ValueError("duplicate or invalid independent neutron seeds")
    try:
        scores = np.asarray([
            [entry["weighted_active_generation_economy"][key] for key in SCORES]
            for entry in per_seed
        ], dtype=np.float64)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("missing numerical per-seed neutron economy") from exc
    if (scores.shape != (4, len(SCORES))
            or not np.isfinite(scores).all()
            or np.any(scores[:, [0, 1, 3, 4, 5]] <= 0)
            or np.any(scores[:, 2] < 0)):
        raise ValueError("invalid per-seed neutron economy reaction or fate score")

    # The covariance must retain the negative cross terms, rather than
    # quadratically adding independent absorption/leakage/scatter SEMs.
    closure = scores[:, 0] + scores[:, 1] - scores[:, 2] - 1.0
    if np.max(np.abs(closure)) > 1e-7:
        raise ValueError("correlated neutron fate conservation identity violated")

    means = scores.mean(axis=0)
    covariance = np.cov(scores, rowvar=False, ddof=1)
    standard_errors = np.sqrt(np.maximum(0., np.diag(covariance)) / 4.)
    balance_coefficients = np.array([1., 1., -1., 0., 0., 0.])
    joint_variance = float(balance_coefficients @ covariance @ balance_coefficients)
    scale = max(1.0, float(np.max(np.abs(covariance))))
    if not math.isfinite(joint_variance) or joint_variance < -1e-12*scale:
        raise ValueError("invalid joint neutron-balance covariance")
    joint_sem = math.sqrt(max(0., joint_variance) / 4.)
    if joint_sem > 1e-7:
        raise ValueError("correlated neutron-balance uncertainty contradicts closure")
    falsely_independent_sem = float(np.linalg.norm(standard_errors[:3]))

    # Difference the two estimators INSIDE each independent seed before
    # computing between-seed uncertainty: both score the same histories.
    def paired(left: int, right: int) -> dict:
        differences = scores[:, left] - scores[:, right]
        mean_difference = float(differences.mean())
        sem = float(np.std(differences, ddof=1) / math.sqrt(4.))
        return {
            "per_seed_differences": differences.tolist(),
            "mean_difference": mean_difference,
            "between_seed_paired_standard_error": sem,
            "conditional_t95_interval": [
                mean_difference - T95_DF3 * sem,
                mean_difference + T95_DF3 * sem,
            ],
        }

    return {
        "statistics": "FOUR_INDEPENDENT_OPENMC_SEED_MEANS_CORRELATED_NEUTRON_ECONOMY",
        "independent_seed_count": 4,
        "seeds": seeds,
        "score_order": list(SCORES),
        "mean_per_starting_neutron": dict(zip(SCORES, means.tolist())),
        "between_seed_sample_covariance": covariance.tolist(),
        "between_seed_standard_error_of_mean": dict(zip(SCORES, standard_errors.tolist())),
        "conditional_t95_intervals_per_starting_neutron": {
            key: [float(mean - T95_DF3*sem), float(mean + T95_DF3*sem)]
            for key, mean, sem in zip(SCORES, means, standard_errors)
        },
        "maximum_absolute_per_seed_neutron_balance_residual": float(
            np.max(np.abs(closure))
        ),
        "joint_neutron_balance_covariance_standard_error": joint_sem,
        "incorrect_independent_terms_balance_standard_error": falsely_independent_sem,
        "paired_tracklength_minus_analog_absorption": paired(3, 0),
        "paired_tracklength_minus_analog_fission_neutron_yield": paired(4, 5),
        "t_critical_df3_two_sided_95_percent": T95_DF3,
        "source_bank_stationarity_validated": False,
        "historical_nerva_neutron_physics_validated": False,
        "formal_monte_carlo_confidence_coverage_established": False,
        "interpretation": (
            "The four RNG seeds are the sampling units. Preserve correlations "
            "among terminal absorption, vacuum leakage and neutron-producing "
            "scattering, and pair tracklength with analog tallies inside each "
            "seed. Student-t ranges are conditional descriptive estimates, "
            "not evidence of converged fission-source genealogies, reactor "
            "criticality, hot evaluated nuclear data or NASA fidelity."
        ),
    }
