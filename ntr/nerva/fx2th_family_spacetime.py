"""Cellwise delayed-family space-time multigroup diffusion (mathematical benchmark).

Source lineage: ANL-78-97 FX2-TH, time-dependent multigroup diffusion and
spatial delayed-neutron inventories, as independently implemented here.

This solver accepts an EXTERNAL conservative *cell-integrated* loss matrix
(A) for (cell,group) flux unknowns. Nothing in this module manufactures
cross sections, OpenMC transport, control-drum worth, or thermal feedback.

The source contract shares the Stage-6 family-resolved arrays:
  flux[c,g], nuSigmaF[c,g], beta[c,g,f], chi_prompt[c,g],
  chi_delayed[c,f,g], lambda[f], volumes[c].
  A has dimensions [number of cell/group unknowns]^2, with volume already
  applied to reaction/removal terms. M[c,g]=V[c]/v[g].

Equations (C is the *cell-integrated* number of precursors):
  M d(phi)/dt = -A phi + F_prompt(phi) + emission(C) + V q
  dC[c,f]/dt = sum_g V[c] beta[c,g,f] nuSigmaF[c,g] phi[c,g] - lambda[f] C[c,f]
  emission[c,g] = sum_f chi_delayed[c,f,g] lambda[f] C[c,f].

Implicit backward Euler uses algebraic elimination of all C[c,f].
This advances spatially nonuniform precursor inventories, including
family-specific birth spectra, without the lossy Stage-3 common-spectrum
assumption. An input-to-output ledger independently checks both equations.

The result is NOT a source-faithful original FX2-TH port, a NERVA reactor
transient, thermal feedback, kinetic effective-parameter derivation, or
validated continuous-energy transport model. Do not confuse a synthetic
numerical check with experimental nuclear-physics evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from scipy.sparse import block_diag, csc_matrix, diags, issparse
from scipy.sparse.linalg import spsolve

from .delayed_family_contract import family_importance


def _array(x: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f"{name} must be finite nonnegative {shape}")
    return a


@dataclass(frozen=True)
class SpatialFamilyData:
    """One externally supplied material state on a fixed mesh and group grid."""
    source_id: str
    integrated_loss: csc_matrix
    volumes_m3: np.ndarray
    group_speeds_m_s: np.ndarray
    nu_sigma_f_inv_m: np.ndarray
    beta_by_incident_group_and_family: np.ndarray
    prompt_spectrum: np.ndarray
    delayed_spectra_by_family: np.ndarray
    decay_constants_s_inv: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("nonblank source_id must identify input provenance")
        xs = np.asarray(self.nu_sigma_f_inv_m, dtype=float)
        if xs.ndim != 2 or min(xs.shape) < 1:
            raise ValueError("nuSigmaF must be [cell,energy_group]")
        nc, ng = xs.shape
        lam = np.asarray(self.decay_constants_s_inv, dtype=float)
        if lam.ndim != 1 or len(lam) < 1:
            raise ValueError("at least one precursor family required")
        nf = len(lam)
        _array(xs, (nc, ng), "nuSigmaF")
        v = _array(self.volumes_m3, (nc,), "volumes")
        speed = _array(self.group_speeds_m_s, (ng,), "group speeds")
        rates = _array(lam, (nf,), "decay rates")
        beta = _array(self.beta_by_incident_group_and_family, (nc,ng,nf), "beta")
        chi_p = _array(self.prompt_spectrum, (nc,ng), "prompt spectrum")
        chi_d = _array(self.delayed_spectra_by_family, (nc,nf,ng), "family spectra")
        if np.any(v<=0) or np.any(speed<=0) or np.any(rates<=0):
            raise ValueError("volumes, speeds and decay constants must be positive")
        if np.any(beta.sum(axis=2) >= 1):
            raise ValueError("sum of beta for incident group must be below 1")
        if not np.allclose(chi_p.sum(axis=1),1.,rtol=0.,atol=1e-12):
            raise ValueError("each prompt spectrum must sum to one")
        if not np.allclose(chi_d.sum(axis=2),1.,rtol=0.,atol=1e-12):
            raise ValueError("each delayed spectrum must sum to one")
        A = self.integrated_loss
        if not issparse(A):
            raise ValueError("integrated_loss must be a SciPy sparse matrix")
        if A.shape != (nc*ng,nc*ng):
            raise ValueError("loss matrix shape mismatch")
        if not np.all(np.isfinite(A.data)):
            raise ValueError("loss matrix must be finite")
        # This is a loss/transfer M-matrix convention. Positive offdiagonal
        # coupling would be a different equation/phase convention.
        arr = A.tocoo()
        if np.any(arr.data[arr.row != arr.col] > 1e-13):
            raise ValueError("positive off-diagonal loss couplings are not permitted")
        if np.any(A.diagonal() <= 0):
            raise ValueError("positive loss diagonal required")

    @property
    def cells(self) -> int:
        return int(np.asarray(self.nu_sigma_f_inv_m).shape[0])

    @property
    def groups(self) -> int:
        return int(np.asarray(self.nu_sigma_f_inv_m).shape[1])

    @property
    def families(self) -> int:
        return len(self.decay_constants_s_inv)

    @property
    def mass_diagonal(self) -> np.ndarray:
        """[cell,group] integrated inverse-speed mass for time derivative."""
        return (np.asarray(self.volumes_m3)[:,None] /
                np.asarray(self.group_speeds_m_s)[None,:]).reshape(-1)


@dataclass(frozen=True)
class SpatialFamilyState:
    time_s: float
    flux: np.ndarray             # [cell,group] scalar flux
    precursors: np.ndarray       # [cell,family] integrated precursor inventory


@dataclass(frozen=True)
class SpatialFamilyStep:
    state: SpatialFamilyState
    flux_equation_residual: float
    precursor_equation_residual: float
    global_balance_residual: float
    neutron_inventory: float
    total_precursor_inventory: float
    prompt_emission: np.ndarray         # integrated, [cell,group]
    delayed_emission: np.ndarray        # integrated, [cell,group]
    produced_precursors: np.ndarray     # rates integrated per cell, [cell,family]
    absolute_global_balance: float


def stationary_precursors(data: SpatialFamilyData, flux: np.ndarray) -> np.ndarray:
    phi = _array(flux, (data.cells,data.groups), "flux")
    birth = np.einsum("cg,cg,cgf,c->cf", phi, data.nu_sigma_f_inv_m,
                     data.beta_by_incident_group_and_family, data.volumes_m3)
    return birth / np.asarray(data.decay_constants_s_inv)[None,:]


def _production(data: SpatialFamilyData, flux: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    fission = np.asarray(data.nu_sigma_f_inv_m)*flux
    prompt_rate = np.einsum("cg,cg->c", fission,
                            1.-np.sum(data.beta_by_incident_group_and_family,axis=2))
    prompt = np.asarray(data.volumes_m3)[:,None] * prompt_rate[:,None] * data.prompt_spectrum
    precursor_rate = np.einsum("cg,cgf,c->cf", fission,
                                data.beta_by_incident_group_and_family, data.volumes_m3)
    return prompt, precursor_rate


def _emission(data: SpatialFamilyData, C: np.ndarray) -> np.ndarray:
    return np.einsum("cf,cfg,f->cg", C, data.delayed_spectra_by_family,
                     data.decay_constants_s_inv)


def _effective_prompt_delayed_matrix(data: SpatialFamilyData, dt: float) -> csc_matrix:
    """F_prompt + Σ_f dt*lambda_f/(1+dt*lambda_f) F_delayed_family."""
    beta = np.asarray(data.beta_by_incident_group_and_family)
    fraction = dt*np.asarray(data.decay_constants_s_inv) / (
        1.+dt*np.asarray(data.decay_constants_s_inv))
    blocks=[]
    for cell in range(data.cells):
        source = np.asarray(data.volumes_m3)[cell]*np.asarray(data.nu_sigma_f_inv_m)[cell]
        bp = np.asarray(data.prompt_spectrum)[cell,:,None] * (
            source*(1-beta[cell].sum(axis=1)))[None,:]
        bd = np.einsum("fg,hf,h,f->gh",
                       data.delayed_spectra_by_family[cell],
                       beta[cell],source,fraction)
        blocks.append(bp+bd)
    return block_diag(blocks,format="csc")


def advance_spatial_families(
    data: SpatialFamilyData,
    previous: SpatialFamilyState,
    dt_s: float,
    external_source_per_m3: np.ndarray | None = None,
) -> SpatialFamilyStep:
    """Backward Euler, including exact local algebraic precursor elimination."""
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("dt_s must be finite and positive")
    if not math.isfinite(previous.time_s) or previous.time_s<0:
        raise ValueError("state time must be finite and nonnegative")
    old_phi = _array(previous.flux, (data.cells,data.groups), "old flux")
    old_c = _array(previous.precursors, (data.cells,data.families), "old precursors")
    q = (np.zeros_like(old_phi) if external_source_per_m3 is None else
         _array(external_source_per_m3,old_phi.shape,"external source"))
    M=data.mass_diagonal
    lam=np.asarray(data.decay_constants_s_inv)
    kold=lam/(1.+dt_s*lam)
    old_emit=np.einsum("cf,cfg,f->cg",old_c,data.delayed_spectra_by_family,kold)
    rhs=(M/dt_s)*old_phi.flatten() + old_emit.flatten() + (
        q*np.asarray(data.volumes_m3)[:,None]).flatten()
    B=(data.integrated_loss.tocsc() + diags(M/dt_s,format="csc")
       - _effective_prompt_delayed_matrix(data,dt_s))
    new_phi=np.asarray(spsolve(B,rhs),dtype=float)
    if np.any(~np.isfinite(new_phi)) or np.any(new_phi<-1e-10):
        raise ArithmeticError("implicit family spatial solve produced nonphysical flux")
    new_phi=np.maximum(new_phi,0.).reshape(data.cells,data.groups)
    prompt,produced=_production(data,new_phi)
    new_c=(old_c+dt_s*produced)/(1.+dt_s*lam[None,:])
    emission=_emission(data,new_c)

    rflux=(M*(new_phi-old_phi).flatten()/dt_s
           +data.integrated_loss@new_phi.flatten()
           -prompt.flatten()-emission.flatten()
           -(q*np.asarray(data.volumes_m3)[:,None]).flatten())
    rc=(new_c-old_c)/dt_s - produced + new_c*lam[None,:]
    fscale=max(1.,float(np.linalg.norm((M/dt_s)*old_phi.flatten(),ord=np.inf)),
               float(np.linalg.norm(prompt.flatten()+emission.flatten(),ord=np.inf)))
    cscale=max(1.,float(np.linalg.norm(produced,ord=np.inf)))
    absolute_balance=(float(np.sum(M*(new_phi-old_phi).flatten()/dt_s))
                       +float(np.sum((new_c-old_c)/dt_s))
                       +float(np.sum(data.integrated_loss@new_phi.flatten()))
                       -float(np.sum(np.asarray(data.volumes_m3)[:,None] *
                                     data.nu_sigma_f_inv_m*new_phi))
                       -float(np.sum(q*np.asarray(data.volumes_m3)[:,None])))
    bscale=max(1.,float(np.sum(np.abs(data.integrated_loss@new_phi.flatten()))),
               float(np.sum(np.asarray(data.volumes_m3)[:,None] *
                                     data.nu_sigma_f_inv_m*new_phi)))
    return SpatialFamilyStep(
        state=SpatialFamilyState(previous.time_s+dt_s,new_phi,new_c),
        flux_equation_residual=float(np.max(np.abs(rflux))/fscale),
        precursor_equation_residual=float(np.max(np.abs(rc))/cscale),
        global_balance_residual=abs(absolute_balance)/bscale,
        neutron_inventory=float(np.sum(M*new_phi.flatten())),
        total_precursor_inventory=float(new_c.sum()),
        prompt_emission=prompt,
        delayed_emission=emission,
        produced_precursors=produced,
        absolute_global_balance=absolute_balance,
    )


def importance_snapshot(
    data: SpatialFamilyData, state: SpatialFamilyState, adjoint_flux: np.ndarray
):
    """Re-use Stage 6 importance contract; do not derive adjoint from flux."""
    return family_importance(
        source_id=data.source_id,
        forward_flux=state.flux,
        adjoint_flux=adjoint_flux,
        cell_volumes_m3=data.volumes_m3,
        nu_sigma_f_inv_m=data.nu_sigma_f_inv_m,
        beta_by_incident_group_and_family=data.beta_by_incident_group_and_family,
        prompt_spectrum=data.prompt_spectrum,
        delayed_spectra_by_family=data.delayed_spectra_by_family,
        decay_constants_s_inv=data.decay_constants_s_inv,
    )


def evolve_external_states(
    data_states: Sequence[SpatialFamilyData],
    initial: SpatialFamilyState,
    dt_s: float,
    sources: Sequence[np.ndarray] | None = None,
) -> tuple[SpatialFamilyStep,...]:
    """Evolve under externally specified material states on the same mesh.

    Does not calculate thermal response. If the mesh or group set changes,
    conservative remapping and cross-section energy-bin reconciliation must
    be handled upstream; those changes are rejected here.
    """
    if not data_states:
        raise ValueError("one or more externally supplied coefficient states required")
    sizes={(d.cells,d.groups,d.families) for d in data_states}
    if len(sizes)!=1 or len({tuple(np.asarray(d.volumes_m3)) for d in data_states})!=1:
        raise ValueError("mesh and energy/precursor dimensions must remain fixed")
    if sources is None:
        sources=[np.zeros((data_states[0].cells,data_states[0].groups)) for _ in data_states]
    if len(sources)!=len(data_states):
        raise ValueError("source schedule length mismatch")
    steps=[]
    state=initial
    for d,q in zip(data_states,sources):
        step=advance_spatial_families(d,state,dt_s,q)
        steps.append(step)
        state=step.state
    return tuple(steps)
