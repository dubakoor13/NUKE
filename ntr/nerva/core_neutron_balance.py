"""Physical neutron accounting from genuine OpenMC eigenvalue statepoints.

For a generation with one starting neutron, fission daughters are BANKED for
a later eigenvalue generation; they are not added to the current source
neutron transport. Scattering with multiplicity adds neutrons to the same
generation. The generation-wide neutron balance is thus:

    absorption + leakage - (nu_scatter - scatter) ~= 1

All reaction tallies must span the WHOLE GEOMETRY (not an engineering mesh).
The leakage term is OpenMC's native GLOBAL leakage estimator rather than a
signed surface-current sum with arbitrarily oriented surface normals.

This numerical closure is a real core-transport physics check, not a NASA
NERVA benchmark or fission-source stationarity certification. The physical
identity, covariance limitations, and input HDF5 hash are recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import openmc


def _scalar_tally(tally: openmc.Tally, score: str) -> tuple[float, float]:
    if score not in tally.scores:
        raise ValueError(f"neutron balance tally lacks score {score}")
    if any(not isinstance(f, openmc.ParticleFilter) for f in tally.filters):
        raise ValueError("global neutron balance must not be spatially clipped")
    if tally.filters:
        if len(tally.filters) != 1 or tuple(tally.filters[0].bins) != ("neutron",):
            raise ValueError("global neutron balance must tally neutrons alone")
    def single(stat: str) -> float:
        value = np.asarray(tally.get_values(scores=[score], value=stat), dtype=float)
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError(f"invalid {score} {stat}: expected one real finite bin")
        return float(value.reshape(-1)[0])
    mean, sem = single("mean"), single("std_dev")
    if mean < 0 or sem < 0:
        raise ValueError(f"negative {score} neutron reaction score/error")
    return mean, sem


def _global_leakage(sp: openmc.StatePoint) -> tuple[float, float, float | None, float | None]:
    gt = sp.global_tallies
    names = [
        name.decode().strip() if isinstance(name, bytes) else str(name).strip()
        for name in gt["name"]
    ]
    if len(set(names)) != len(names) or "leakage" not in names:
        raise ValueError("OpenMC global leakage estimator missing or duplicated")
    i = names.index("leakage")
    leakage = float(gt["mean"][i])
    error = float(gt["std_dev"][i])
    if not all(map(math.isfinite, (leakage, error))) or leakage < 0 or error < 0:
        raise ValueError("invalid native OpenMC vacuum leakage estimate")
    if "k-tracklength" in names:
        j = names.index("k-tracklength")
        k = float(gt["mean"][j])
        sigma = float(gt["std_dev"][j])
        if not all(map(math.isfinite, (k, sigma))) or k <= 0 or sigma < 0:
            raise ValueError("invalid native OpenMC k-tracklength production estimate")
        return leakage, error, k, sigma
    return leakage, error, None, None


def _close_balance(
    rates: dict[str, tuple[float, float]],
    leakage: tuple[float, float],
    native_k: tuple[float | None, float | None],
    *,
    abs_tolerance: float = 0.02,
    sigma_multiplier: float = 5.0,
) -> dict:
    """Conservative reported-error screen, never a joint confidence claim.

    Unknown tally cross-covariances can make quadrature of marginal SEM
    spuriously small. Sum the absolute coefficients times SEM: by triangle
    inequality this bounds the standard deviation of the signed linear
    combination, provided the reported batch SEMs are valid.
    """
    for score in ("absorption", "scatter", "nu-scatter", "nu-fission"):
        if score not in rates:
            raise ValueError(f"required physical neutron score missing: {score}")
        mean, sigma = rates[score]
        if not math.isfinite(mean) or mean < 0 or not math.isfinite(sigma) or sigma < 0:
            raise ValueError(f"nonfinite/negative {score} rate or error")
    leak, leak_sem = leakage
    if any(not math.isfinite(x) or x < 0 for x in (leak, leak_sem)):
        raise ValueError("invalid global neutron leakage")
    if not math.isfinite(abs_tolerance) or not 0 < abs_tolerance < 1:
        raise ValueError("absolute tolerance must be finite inside (0,1)")
    if not math.isfinite(sigma_multiplier) or not 1 <= sigma_multiplier <= 10:
        raise ValueError("sigma multiplier must be finite between 1 and 10")

    absorption = rates["absorption"][0]
    net_multiplicative_scattering = rates["nu-scatter"][0] - rates["scatter"][0]
    outflow = absorption + leak - net_multiplicative_scattering
    signed_residual = outflow - 1.0
    sem_bound = (
        leak_sem + rates["absorption"][1]
        + rates["nu-scatter"][1] + rates["scatter"][1]
    )
    threshold = max(abs_tolerance, sigma_multiplier * sem_bound)
    native_k_value, native_k_sigma = native_k
    nu_fiss, nu_fiss_sem = rates["nu-fission"]
    production_delta = None if native_k_value is None else nu_fiss - native_k_value
    production_sem_bound = (
        None if native_k_value is None
        else nu_fiss_sem + float(native_k_sigma)
    )
    production_tolerance = (
        None if native_k_value is None
        else max(abs_tolerance * max(1.0, native_k_value),
                 sigma_multiplier * production_sem_bound)
    )
    balance_pass = abs(signed_residual) <= threshold
    production_pass = (
        None if production_delta is None
        else abs(production_delta) <= production_tolerance
    )
    return {
        "source_normalization": "one transported source neutron per eigenvalue generation",
        "physical_scores_per_source_neutron": {
            k: {"mean":float(v[0]), "reported_standard_error":float(v[1])}
            for k,v in rates.items()
        },
        "native_global_vacuum_leakage_per_source":{
            "mean":leak,"reported_standard_error":leak_sem,
        },
        "multiplicative_scattering_net_neutrons_per_source":net_multiplicative_scattering,
        "balance_disappearance_per_source":outflow,
        "balance_signed_residual_minus_one":signed_residual,
        "conservative_residual_standard_error_upper_bound":sem_bound,
        "screen_threshold_per_source":threshold,
        "neutron_balance_screen_passed":balance_pass,
        "nu_fission_vs_native_tracklength_k":{
            "nu_fission_production_per_source":nu_fiss,
            "native_k_tracklength":native_k_value,
            "signed_difference":production_delta,
            "conservative_standard_error_upper_bound":production_sem_bound,
            "screen_threshold":production_tolerance,
            "screen_passed":production_pass,
        },
        "is_core_physically_validated_nasa_nerva":False,
        "fission_source_stationarity_proven":False,
        "interpretation":(
            "Whole-geometry neutron conservation including live neutron "
            "multiplication from scattering. Fission secondaries are BANKED "
            "for the next eigenvalue source generation. Native global OpenMC "
            "leakage avoids signed-vacuum-surface cancellation. Marginal "
            "reported tally SEMs are correlated, so their SUM is a "
            "conservative error bound, not an exact joint variance. Passing "
            "these screens is necessary physics implementation evidence, "
            "not independent source convergence or measured NASA fidelity."
        ),
    }


def inspect_real_statepoint(path: Path) -> dict:
    path=Path(path)
    if not path.is_file():
        raise ValueError("genuine OpenMC eigenvalue statepoint file missing")
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024),b""):
            digest.update(chunk)
    with openmc.StatePoint(path,autolink=False) as sp:
        if sp.run_mode != "eigenvalue":
            raise ValueError("neutron conservation applies to eigenvalue source bank")
        if not sp.tallies_present:
            raise ValueError("real OpenMC statepoint contains no tallies")
        tally=sp.get_tally(name="nerva_global_neutron_balance")
        rates={score:_scalar_tally(tally,score)
               for score in ("absorption","scatter","nu-scatter","nu-fission")}
        native=_global_leakage(sp)
        report=_close_balance(rates,native[:2],native[2:])
        report.update({
            "source":"GENUINE_OPENMC_EIGENVALUE_STATEPOINT",
            "statepoint_path":str(path),
            "statepoint_sha256":digest.hexdigest(),
            "batches":int(sp.current_batch),
            "inactive_batches":int(sp.n_inactive),
            "active_tally_realizations":int(sp.n_realizations),
        })
    return report


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statepoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--strict",action="store_true",
                   help="Fail when real neutron conservation or fission-production consistency screens fail")
    a=p.parse_args()
    result=inspect_real_statepoint(a.statepoint)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Neutron balance (absorption + leakage - net scatter multiplication):",
          result["balance_disappearance_per_source"])
    print("Signed residual:",result["balance_signed_residual_minus_one"])
    print("Conservation screen:",result["neutron_balance_screen_passed"])
    print("Output:",a.output)
    if a.strict and (
        not result["neutron_balance_screen_passed"] or
        result["nu_fission_vs_native_tracklength_k"]["screen_passed"] is False
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
