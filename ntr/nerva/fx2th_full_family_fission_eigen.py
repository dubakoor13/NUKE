"""Generalized spatial multigroup fission eigenproblem with delayed birth spectra.

This Stage-15 implementation removes the common-spectrum restriction of
earlier NUKE r-z (Stage 12) and Cartesian (Stage 11) eigen references.

For each spatial cell c, incident energy group h, emitted energy group g,
and delayed-neutron family f:

    F_prompt[c,g,h] = V[c] * chi_prompt[c,g]
                      * nuSigmaF[c,h] * (1-sum_f beta[c,h,f])
    F_delayed_f[c,g,h] = V[c] * chi_delayed[c,f,g]
                         * nuSigmaF[c,h] * beta[c,h,f]
    F_total = F_prompt + sum_f F_delayed_f

The stationary problem is A phi = (1/k) F_total phi, with adjoint
A.T psi = (1/k) F_total.T psi; A already includes cell-integrated
absorption, scattering and leakage. Do NOT apply volume twice.

The generalized fission operator is independent of family decay constants
at equilibrium because steady local precursor inventories satisfy
lambda[f]*C[c,f] = V[c]*sum_h beta[c,h,f]*nuSigmaF[c,h]*phi[c,h].
The family kinetics in Stage 7 and second-order Stage 14 should preserve
this stationary state when k=1, even when all emitted spectra differ.

All spatial dimensions, energy groups, material data and spectra are
externally provided, without default fuel mixtures or reactor parameters.
This is a numerical multigroup diffusion/kinetics check, NOT real OpenMC
transport, qualified NERVA nuclear data or historical reactor validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import block_diag, csc_matrix
from scipy.sparse.linalg import splu

from .fx2th_family_spacetime import (
    SpatialFamilyData, SpatialFamilyState, stationary_precursors,
    advance_spatial_families, importance_snapshot,
)


@dataclass(frozen=True)
class ResolvedFissionMatrices:
    prompt: csc_matrix
    delayed_by_family: tuple[csc_matrix, ...]
    total: csc_matrix


@dataclass(frozen=True)
class FamilyEigenReference:
    keff: float
    forward_flux: np.ndarray                  # [cell,group]
    adjoint_flux: np.ndarray                  # [cell,group]
    forward_residual: float
    adjoint_residual: float
    forward_iterations: int
    adjoint_iterations: int
    prompt_generation_time_s: float
    beta_effective_by_family: tuple[float, ...]
    beta_unweighted_by_family: tuple[float, ...]
    fission_adjoint_pairing: float
    total_fission_births: float
    prompt_fission_births: float
    delayed_fission_births: float
    source_id: str


@dataclass(frozen=True)
class StationaryFamilyVerification:
    initial: SpatialFamilyState
    implicit_euler_flux_error: float
    implicit_euler_precursor_error: float
    implicit_euler_balance_residual: float
    sdirk2_flux_error: float
    sdirk2_precursor_error: float
    sdirk2_balance_residual: float
    reference: FamilyEigenReference


def assemble_family_fission(data: SpatialFamilyData) -> ResolvedFissionMatrices:
    """Construct individual family birth operators on the Stage-7 ordering."""
    nc,ng,nf=data.cells,data.groups,data.families
    volume=np.asarray(data.volumes_m3,dtype=float)
    xs=np.asarray(data.nu_sigma_f_inv_m,dtype=float)
    beta=np.asarray(data.beta_by_incident_group_and_family,dtype=float)
    chi_p=np.asarray(data.prompt_spectrum,dtype=float)
    chi_d=np.asarray(data.delayed_spectra_by_family,dtype=float)
    # Each block maps incoming energy group h to emitted group g.
    pblocks=[]
    dblocks=[[] for _ in range(nf)]
    for c in range(nc):
        births=volume[c]*xs[c]
        prompt=births*(1.-beta[c].sum(axis=1))
        pblocks.append(np.outer(chi_p[c],prompt))
        for f in range(nf):
            dblocks[f].append(np.outer(chi_d[c,f],births*beta[c,:,f]))
    # Use sparse *matrix* inputs for predictable SciPy sparse types, and
    # eliminate explicit numerical zeros; a beta=0 family has no operator.
    p=block_diag([csc_matrix(b) for b in pblocks],format="csc")
    p.eliminate_zeros()
    delayed_list=[]
    for chunks in dblocks:
        f=block_diag([csc_matrix(b) for b in chunks],format="csc")
        f.eliminate_zeros()
        delayed_list.append(f)
    delayed=tuple(delayed_list)
    total=p.copy()
    for mat in delayed:
        total=total+mat
    if (not np.all(np.isfinite(total.data)) or
        np.any(total.data<0) or total.shape!=(nc*ng,nc*ng) or
        float(total.sum())<=0):
        raise ValueError("invalid total family-resolved fission production matrix")
    return ResolvedFissionMatrices(p,delayed,total.tocsc())


def _power_mode(A:csc_matrix,F:csc_matrix, *,
                adjoint:bool, tolerance:float, max_iterations:int):
    solver=splu(A.tocsc())
    v=np.ones(A.shape[0],dtype=float)
    v/=v.sum()
    AA=A.T if adjoint else A
    FF=F.T if adjoint else F
    for i in range(1,max_iterations+1):
        rhs=FF@v
        out=np.asarray(solver.solve(rhs,trans="T" if adjoint else "N"),
                       dtype=float)
        if not np.all(np.isfinite(out)) or np.any(out< -1e-13):
            raise ArithmeticError("nonphysical generalized fission iterate")
        out=np.maximum(out,0.)
        k=float(out.sum())
        if not math.isfinite(k) or k<=0:
            raise ArithmeticError("nonpositive generalized fission eigenvalue")
        out/=k
        res=float(np.linalg.norm(AA@out-(FF@out)/k,ord=np.inf)/
                  max(1e-30,float(np.linalg.norm(AA@out,ord=np.inf)),
                      float(np.linalg.norm((FF@out)/k,ord=np.inf))))
        change=float(np.linalg.norm(out-v,ord=np.inf))
        if res<tolerance and change<tolerance:
            return k,out,res,i
        v=out
    raise RuntimeError("multigroup family fission eigenmode did not converge")


def solve_family_eigen(
    data:SpatialFamilyData,
    *, tolerance:float=1e-10, max_iterations:int=10000,
) -> FamilyEigenReference:
    """Solve total prompt+delayed equilibrium fission, forward and adjoint."""
    if (not math.isfinite(tolerance) or not 0<tolerance<1e-3 or
        not isinstance(max_iterations,int) or max_iterations<1):
        raise ValueError("invalid generalized eigen tolerance/iteration limit")
    mats=assemble_family_fission(data)
    A=data.integrated_loss.tocsc()
    F=mats.total
    k,phi,fr,nf=_power_mode(A,F,adjoint=False,
                             tolerance=tolerance,max_iterations=max_iterations)
    kt,psi,ar,na=_power_mode(A,F,adjoint=True,
                             tolerance=tolerance,max_iterations=max_iterations)
    if abs(k-kt)>max(1e-10,30*tolerance*max(k,kt)):
        raise ArithmeticError("forward and adjoint family eigenvalues disagree")
    mass=data.mass_diagonal
    # Forward arbitrary scalar is fixed by the cell-integrated neutron
    # population, not by imposing an unqualified physical power in watts.
    pop=float(np.dot(mass,phi))
    if not math.isfinite(pop) or pop<=0:
        raise ArithmeticError("invalid neutron population mass")
    phi/=pop
    pairing=float(psi@(F@phi))
    if not math.isfinite(pairing) or pairing<=0:
        raise ArithmeticError("zero adjoint-weighted production")
    psi/=pairing
    nc,ng=data.cells,data.groups
    snapshot=importance_snapshot(
        data,
        SpatialFamilyState(0.,phi.reshape(nc,ng),
                           np.zeros((nc,data.families))),
        psi.reshape(nc,ng))
    if not math.isclose(snapshot.weighted_total,1.,rel_tol=2e-8,abs_tol=2e-9):
        raise ArithmeticError("family production and sparse F adjoint pairing disagree")
    # The kinetic spectral parameter uses the TOTAL equilibrium fission
    # operator; call it adjoint-weighted generation time, not a measured
    # prompt neutron lifetime or a NERVA transient power constant.
    lifetime=float(np.dot(psi,mass*phi))
    prompt=float(np.sum(mats.prompt@phi))
    delayed=float(sum(float(np.sum(x@phi)) for x in mats.delayed_by_family))
    full=float(np.sum(F@phi))
    if not math.isclose(prompt+delayed,full,rel_tol=1e-11,abs_tol=1e-11):
        raise ArithmeticError("prompt+delayed production failed algebraic closure")
    return FamilyEigenReference(
        keff=k,
        forward_flux=phi.reshape(nc,ng),
        adjoint_flux=psi.reshape(nc,ng),
        forward_residual=fr,
        adjoint_residual=ar,
        forward_iterations=nf,
        adjoint_iterations=na,
        prompt_generation_time_s=lifetime,
        beta_effective_by_family=snapshot.beta_effective_by_family,
        beta_unweighted_by_family=snapshot.beta_unweighted_by_family,
        fission_adjoint_pairing=float(psi@(F@phi)),
        total_fission_births=full,
        prompt_fission_births=prompt,
        delayed_fission_births=delayed,
        source_id=data.source_id,
    )


def verify_stationary_family_eigen(
    data:SpatialFamilyData,
    *, critical_tolerance:float=1e-8,
    timestep_s:float=.1,
) -> StationaryFamilyVerification:
    """Check a critical family eigenmode against BOTH time integrators.

    If k != 1 (without an external source), the eigenvector is not a
    stationary state and verification fails rather than claiming it is.
    """
    from .fx2th_sdirk2_neutron_kinetics import (
        build_core_operator,advance_sdirk2,
    )
    if (not math.isfinite(timestep_s) or timestep_s<=0 or
        not math.isfinite(critical_tolerance) or critical_tolerance<=0):
        raise ValueError("positive finite timestep and critical tolerance required")
    ref=solve_family_eigen(data)
    if abs(ref.keff-1.)>critical_tolerance:
        raise ValueError("full-family equilibrium requires k-effective near 1")
    phi=ref.forward_flux
    C=stationary_precursors(data,phi)
    initial=SpatialFamilyState(0.,phi,C)
    be=advance_spatial_families(data,initial,timestep_s)
    sdirk=advance_sdirk2(build_core_operator(data),data,initial,timestep_s)
    def differences(v:SpatialFamilyState):
        return (float(np.max(np.abs(v.flux-phi))),
                float(np.max(np.abs(v.precursors-C))))
    bf,bc=differences(be.state)
    sf,sc=differences(sdirk.state)
    # Relative to the normalized physical state scale. Do not accept
    # nonstationarity hidden by a large absolute flux/precursor amplitude.
    flux_tol=2e-7*max(1.,float(np.max(phi)))
    C_tol=2e-7*max(1.,float(np.max(C)))
    if max(bf,sf)>flux_tol or max(bc,sc)>C_tol:
        raise ArithmeticError("stationary family operator is inconsistent with kinetics")
    if max(be.global_balance_residual,
           sdirk.global_balance_relative_residual)>1e-8:
        raise ArithmeticError("stationary family neutron inventory is not conserved")
    return StationaryFamilyVerification(
        initial=initial,
        implicit_euler_flux_error=bf,
        implicit_euler_precursor_error=bc,
        implicit_euler_balance_residual=be.global_balance_residual,
        sdirk2_flux_error=sf,
        sdirk2_precursor_error=sc,
        sdirk2_balance_residual=sdirk.global_balance_relative_residual,
        reference=ref,
    )
