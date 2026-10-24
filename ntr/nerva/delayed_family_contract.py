"""Family-resolved delayed fission operator and strictly checked solver contracts.

No nuclear library is bundled. Cross sections, source spectra, delayed yields,
precursor decay constants, forward/adjoint flux and mesh geometry are required
from the caller. This is a finite-dimensional discrete operator audit, NOT an
OpenMC calculation or a physically qualified NERVA kinetics model.

Dimensions:
  phi, psi, nuSigmaF, chi_prompt       [cell, neutron_energy_group]
  beta_incident                         [cell, incident_group, family]
  chi_delayed                           [cell, family, emitted_group]
  lambda_family_s_inv                   [family]
  cell_volume_m3                        [cell]

The same energy group ordering is used for incident and emitted groups. The
bilinear pairing is sum_c,g psi[c,g] * Fphi[c,g] * V[c]. Therefore psi must be
consistent with the *unintegrated* production operator; if it comes from a
cell-integrated discrete left eigenproblem, its normalization/volume gauge
must be converted before passing it here. Never multiply by cell volume twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np


@dataclass(frozen=True)
class DelayedFamilyResult:
    """Frozen-state, adjoint-weighted family production statistics."""
    source_id: str
    decay_constants_s_inv: tuple[float, ...]
    beta_effective_by_family: tuple[float, ...]
    beta_unweighted_by_family: tuple[float, ...]
    beta_effective: float
    beta_unweighted: float
    importance_ratio: float
    delayed_emission_by_family: np.ndarray  # integrated [cell,family,emitted_group]
    prompt_emission: np.ndarray             # integrated [cell,emitted_group]
    weighted_delayed_by_family: tuple[float, ...]
    weighted_prompt: float
    weighted_total: float

    def point_kinetics_kwargs(
        self, *, generation_time_s: float, reactivity: float,
        source_amplitude_per_s: float,
    ) -> dict[str, Any]:
        """Inputs for FX2-TH Stage-1 KineticsCoefficients (not solver execution).

        Lambda/rho/source are provided, never inferred from a k-effective
        snapshot. This contract only describes a frozen, externally qualified
        spatial/energy state; no temperature or spatial feedback is implied.
        """
        for name, x in (("generation_time_s",generation_time_s),
                        ("reactivity",reactivity),
                        ("source_amplitude_per_s",source_amplitude_per_s)):
            if not math.isfinite(x):
                raise ValueError(name + " must be finite")
        if generation_time_s <= 0 or reactivity >= 1 or source_amplitude_per_s < 0:
            raise ValueError("invalid kinetics parameter domain")
        if not self.beta_effective < 1:
            raise ValueError("beta_eff must be below unity")
        return dict(generation_time_s=float(generation_time_s),
                    beta_groups=self.beta_effective_by_family,
                    lambda_groups_s_inv=self.decay_constants_s_inv,
                    reactivity=float(reactivity),
                    source_amplitude_per_s=float(source_amplitude_per_s))


def _nonnegative(a: object, name: str, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(a,dtype=float)
    if (result.shape != shape or not np.all(np.isfinite(result))
            or np.any(result < 0)):
        raise ValueError(f"{name} must be finite nonnegative with shape {shape}")
    return result


def family_importance(
    *,
    source_id: str,
    forward_flux: np.ndarray,
    adjoint_flux: np.ndarray,
    cell_volumes_m3: np.ndarray,
    nu_sigma_f_inv_m: np.ndarray,
    beta_by_incident_group_and_family: np.ndarray,
    prompt_spectrum: np.ndarray,
    delayed_spectra_by_family: np.ndarray,
    decay_constants_s_inv: np.ndarray,
) -> DelayedFamilyResult:
    """Return adjoint-weighted delayed family fractions for supplied data.

    The array of prompt and delayed production is integrated over each cell
    volume. Beta fractions are per incident fission-neutron energy group.
    Delayed emission spectra are per delayed family and per emitted group.
    Beta fractions may vary across space and incident energy. They are not
    silently replaced with a U-235 mixture constant.
    """
    if not isinstance(source_id,str) or not source_id.strip():
        raise ValueError("source_id must identify the provided data provenance")
    phi = np.asarray(forward_flux,dtype=float)
    if phi.ndim != 2 or min(phi.shape) < 1:
        raise ValueError("forward_flux must be [cell,group]")
    nc, ng = phi.shape
    precursor_decay = np.asarray(decay_constants_s_inv,dtype=float)
    if precursor_decay.ndim != 1 or precursor_decay.size < 1:
        raise ValueError("at least one delayed-neutron family required")
    nf = precursor_decay.size
    phi = _nonnegative(phi,"forward_flux",(nc,ng))
    psi = _nonnegative(adjoint_flux,"adjoint_flux",(nc,ng))
    nu_f = _nonnegative(nu_sigma_f_inv_m,"nu_sigma_f_inv_m",(nc,ng))
    volumes = _nonnegative(cell_volumes_m3,"cell_volumes_m3",(nc,))
    beta = _nonnegative(beta_by_incident_group_and_family,
                        "beta_by_incident_group_and_family",(nc,ng,nf))
    chi_prompt = _nonnegative(prompt_spectrum,"prompt_spectrum",(nc,ng))
    chi_delayed = _nonnegative(delayed_spectra_by_family,
                              "delayed_spectra_by_family",(nc,nf,ng))
    lam = _nonnegative(precursor_decay,"decay_constants_s_inv",(nf,))
    if np.any(volumes <= 0) or np.any(lam <= 0):
        raise ValueError("volumes and decay constants must be strictly positive")
    if np.any(beta.sum(axis=-1) >= 1.0):
        raise ValueError("sum of family beta per incident group must be < 1")
    if not np.allclose(chi_prompt.sum(axis=-1), 1., atol=2e-12, rtol=0):
        raise ValueError("prompt_spectrum must sum to 1 in each cell")
    if not np.allclose(chi_delayed.sum(axis=-1), 1., atol=2e-12, rtol=0):
        raise ValueError("each delayed family spectrum must sum to 1")

    fission = phi * nu_f
    total_beta = beta.sum(axis=-1)
    prompt_birth = np.sum(fission * (1.0-total_beta),axis=1)
    family_birth = np.einsum('cg,cgf->cf', fission, beta)
    prompt_emission = volumes[:,None] * prompt_birth[:,None] * chi_prompt
    delayed_emission = volumes[:,None,None] * family_birth[:,:,None] * chi_delayed
    weighted_prompt = float(np.einsum('cg,cg->',psi,prompt_emission))
    weighted_fam = np.einsum('cg,cfg->f',psi,delayed_emission)
    weighted_total = float(weighted_prompt + weighted_fam.sum())
    total_emission = float(prompt_emission.sum()+delayed_emission.sum())
    if not (math.isfinite(weighted_total) and weighted_total > 0
            and math.isfinite(total_emission) and total_emission > 0):
        raise ValueError("adjoint-weighted fission production must be positive finite")
    effective = weighted_fam/weighted_total
    raw = np.sum(delayed_emission,axis=(0,2))/total_emission
    beta_eff=float(effective.sum())
    beta_raw=float(raw.sum())
    if not (0 <= beta_eff <= 1 and 0 <= beta_raw < 1):
        raise ArithmeticError("nonphysical delayed emission partition")
    # Production split identity is exact for every incident group and cell.
    balance = (prompt_emission.sum()+delayed_emission.sum()
               - float(np.sum(volumes[:,None]*fission)))
    if abs(balance)>1e-11*max(total_emission,1.):
        raise ArithmeticError("prompt/delayed neutron production does not close")
    return DelayedFamilyResult(
        source_id=source_id,
        decay_constants_s_inv=tuple(float(x) for x in lam),
        beta_effective_by_family=tuple(float(x) for x in effective),
        beta_unweighted_by_family=tuple(float(x) for x in raw),
        beta_effective=beta_eff, beta_unweighted=beta_raw,
        importance_ratio=beta_eff/beta_raw if beta_raw>0 else float('nan'),
        delayed_emission_by_family=delayed_emission,
        prompt_emission=prompt_emission,
        weighted_delayed_by_family=tuple(float(x) for x in weighted_fam),
        weighted_prompt=weighted_prompt,weighted_total=weighted_total,
    )


def legacy_spacetime_contract(
    *, family_result: DelayedFamilyResult,
    beta_by_incident_group_and_family: np.ndarray,
    delayed_spectra_by_family: np.ndarray,
    prompt_spectrum: np.ndarray,
    grid_fission_spectrum: np.ndarray,
) -> dict[str, tuple[float,...]]:
    """Return Stage-3 raw beta/decays ONLY when its assumptions are exact.

    Stage-3 FX2TH DelayedData has a global beta vector and a SINGLE emission
    spectrum shared with prompt fission. It cannot accept spatial- or group-
    varying beta, or family-specific delayed birth spectra. Reject incompatible
    data instead of silently using beta_eff as a physical precursor yield.
    Group speeds remain independently required and are NOT fabricated.
    """
    beta=np.asarray(beta_by_incident_group_and_family,dtype=float)
    spectra=np.asarray(delayed_spectra_by_family,dtype=float)
    chi_p=np.asarray(prompt_spectrum,dtype=float)
    chi_grid=np.asarray(grid_fission_spectrum,dtype=float)
    if beta.ndim!=3 or min(beta.shape)<1:
        raise ValueError("beta must be [cell,incident_group,family]")
    nc,ng,nf=beta.shape
    if (nf!=len(family_result.decay_constants_s_inv) or
            spectra.shape!=(nc,nf,ng) or chi_p.shape!=(nc,ng) or
            chi_grid.shape!=(nc,ng)):
        raise ValueError("incompatible legacy shapes")
    if (not np.all(np.isfinite(beta)) or
            not np.allclose(beta,beta[0,0,:][None,None,:],rtol=0,atol=1e-13)):
        raise ValueError("Stage-3 requires beta constant across cells and incident groups")
    if (not np.allclose(spectra,chi_grid[:,None,:],atol=1e-13,rtol=0) or
            not np.allclose(chi_p,chi_grid,atol=1e-13,rtol=0)):
        raise ValueError("Stage-3 cannot represent distinct delayed/prompt birth spectra")
    return dict(beta=tuple(float(v) for v in beta[0,0,:]),
                decay_s_inv=family_result.decay_constants_s_inv)
