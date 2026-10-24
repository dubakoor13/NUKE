"""Source-independent delayed-neutron importance calculation.

This module checks the algebra of a multigroup fission-production split:

 F_total = F_prompt + F_delayed
 beta_eff = <psi,F_delayed(phi)> / <psi,F_total(phi)>

It is a deterministic cross-check, NOT a NERVA model, evaluated nuclear data,
verified OpenMC adjoint, or a substitute for source-specific delayed spectra.

Arrays are [cell, incident_or_emitted_energy_group]. Macroscopic nu*Sigma_f
is [1/m], forward flux and adjoint are nonnegative, and cell_volumes_m3 is
[m3]. Each spectrum is per-cell, per-emitted-group and sums to 1. Beta
specifies the delayed emission fraction per INCIDENT fission energy group;
spectra determine the EMISSION energy group. All spectra and beta are supplied
by the caller; this module cannot derive them from k_eff or material density.

The same cell-centred, volume-integrated bilinear inner product convention
must be used for both prompt and delayed fission operators. If passing an
adjoint produced by a discrete matrix solver, make sure its left eigenvector
matches the integrated operator represented by these weights. Avoid adding
cell volumes twice when pre-integrating the fission operator upstream.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ImportanceResult:
    beta_effective: float
    beta_unweighted: float
    importance_ratio: float
    total_weighted_production: float
    delayed_weighted_production: float
    prompt_weighted_production: float
    delayed_unweighted_production: float
    total_unweighted_production: float
    # Rate density integrated in cell volume; arbitrary flux normalization.
    delayed_emission: np.ndarray
    prompt_emission: np.ndarray
    beta_group_contributions: np.ndarray
    assumptions: str


def _finite_nonnegative(name: str, array: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    value = np.asarray(array, dtype=float)
    if value.shape != shape or np.any(~np.isfinite(value)) or np.any(value < 0):
        raise ValueError(f"{name}: expected finite nonnegative array of shape {shape}")
    return value


def delayed_importance(
    forward_flux: np.ndarray,
    adjoint_flux: np.ndarray,
    cell_volumes_m3: np.ndarray,
    nu_sigma_f_inv_m: np.ndarray,
    beta_by_incident_group: np.ndarray,
    prompt_spectrum: np.ndarray,
    delayed_spectrum: np.ndarray,
) -> ImportanceResult:
    """Calculate the fission-operator importance ratio for supplied group data.

    No physical neutron energy spectra are inferred. In particular, delayed
    neutron importance is NOT the adjoint at the INCIDENT energy group.
    A legitimate beta_eff may be greater or smaller than raw beta.
    """
    phi = np.asarray(forward_flux, dtype=float)
    if phi.ndim != 2 or phi.shape[0] < 1 or phi.shape[1] < 1:
        raise ValueError("forward_flux must have shape [cell,group]")
    nc, ng = phi.shape
    phi = _finite_nonnegative("forward_flux", phi, (nc,ng))
    psi = _finite_nonnegative("adjoint_flux", adjoint_flux, (nc,ng))
    v = _finite_nonnegative("cell_volumes_m3", cell_volumes_m3, (nc,))
    if np.any(v <= 0):
        raise ValueError("cell volumes must be positive")
    xs = _finite_nonnegative("nu_sigma_f_inv_m", nu_sigma_f_inv_m, (nc,ng))
    beta = _finite_nonnegative("beta_by_incident_group", beta_by_incident_group, (nc,ng))
    if np.any(beta >= 1):
        raise ValueError("delayed fractions must be < 1")
    chi_p = _finite_nonnegative("prompt_spectrum", prompt_spectrum, (nc,ng))
    chi_d = _finite_nonnegative("delayed_spectrum", delayed_spectrum, (nc,ng))
    if not np.allclose(chi_p.sum(axis=1), 1., rtol=0, atol=2e-12):
        raise ValueError("prompt birth spectrum must sum to 1 in each cell")
    if not np.allclose(chi_d.sum(axis=1), 1., rtol=0, atol=2e-12):
        raise ValueError("delayed birth spectrum must sum to 1 in each cell")

    incoming_fission = phi * xs
    prompt_births = np.sum(incoming_fission * (1.-beta), axis=1)
    delayed_births_by_incident = incoming_fission * beta
    delayed_births = delayed_births_by_incident.sum(axis=1)
    prompt_emission = chi_p * prompt_births[:,None] * v[:,None]
    delayed_emission = chi_d * delayed_births[:,None] * v[:,None]
    total_emission = prompt_emission + delayed_emission

    wp = float(np.sum(psi * prompt_emission))
    wd = float(np.sum(psi * delayed_emission))
    wu = float(np.sum(total_emission))
    du = float(np.sum(delayed_emission))
    denominator = wp + wd
    if not all(math.isfinite(x) for x in (wp,wd,wu,du,denominator)) or denominator <= 0 or wu <= 0:
        raise ValueError("no positive finite fission production or adjoint importance")
    if wd < 0 or wd > denominator:
        raise ArithmeticError("nonphysical weighted production partition")
    # The correct importance is the DELAYED BIRTH energy-group importance.
    delayed_birth_importance = np.sum(psi * chi_d, axis=1)
    beta_group_contributions = np.sum(
        v[:,None] * delayed_births_by_incident * delayed_birth_importance[:,None],
        axis=0
    ) / denominator
    beff = wd / denominator
    bunw = du / wu
    if not math.isclose(float(beta_group_contributions.sum()),beff,rel_tol=1e-10,abs_tol=1e-12):
        raise ArithmeticError("delayed importance group contributions do not close")
    return ImportanceResult(
        beta_effective=beff, beta_unweighted=bunw,
        importance_ratio=beff/bunw if bunw > 0 else float("nan"),
        total_weighted_production=denominator, delayed_weighted_production=wd,
        prompt_weighted_production=wp,
        delayed_unweighted_production=du,total_unweighted_production=wu,
        delayed_emission=delayed_emission,prompt_emission=prompt_emission,
        beta_group_contributions=beta_group_contributions,
        assumptions="SUPPLIED multigroup data only; no physically validated NERVA neutron data"
    )
