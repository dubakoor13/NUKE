"""Exact 2D FV diffusion -> spatial family kinetics and adjoint reference.

Links TWO already independently tested NUKE numerical components:

    Stage 2: assemble_diffusion(DiffusionGrid) -> integrated A, F.
    Stage 7: SpatialFamilyData(A, V, beta, chi_p, chi_d, lambda, speed)

Both flatten [y,x,group] to [cell,group] with cell=(y*nx+x).
For a cell with x width dx and y width dy, V=dx*dy*1 m depth.
A contains V*absorption/outscatter and conservative face currents;
do not repeat volume scaling on A. F contains V*chi*nuSigmaF.

This interface has no NERVA geometry/cross-sections, evaluated nuclear
data, OpenMC solve or actual reactor power. All nuclear kinetic material
parameters are caller-supplied. Values and verification are synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .fx2th_diffusion import (
    DiffusionGrid, DiffusionMatrices, EigenSolution,
    assemble_diffusion, solve_keff
)
from .fx2th_family_spacetime import (
    SpatialFamilyData, SpatialFamilyState,
    stationary_precursors, advance_spatial_families,
    importance_snapshot,
)


def _positive_array(value, shape, name):
    a=np.asarray(value,dtype=float)
    if (a.shape!=shape or not np.all(np.isfinite(a)) or np.any(a<=0)):
        raise ValueError(name+" must be finite positive with shape "+str(shape))
    return a


@dataclass(frozen=True)
class CartesianFamilyAssembly:
    data: SpatialFamilyData
    diffusion: DiffusionMatrices
    shape_yxg: tuple[int,int,int]

    def reshape_flux(self, flux_yxg: np.ndarray) -> np.ndarray:
        shape=self.shape_yxg
        v=np.asarray(flux_yxg,dtype=float)
        if v.shape!=shape or not np.all(np.isfinite(v)) or np.any(v<0):
            raise ValueError("flux must be nonnegative finite [y,x,group]")
        return v.reshape(shape[0]*shape[1],shape[2])

    def image_flux(self, flux_cg: np.ndarray) -> np.ndarray:
        v=np.asarray(flux_cg,dtype=float)
        if v.shape!=(self.data.cells,self.data.groups):
            raise ValueError("flux cell/group shape incompatible with 2D mesh")
        return v.reshape(self.shape_yxg)

    def boundary_leakage(self, flux_cg: np.ndarray) -> float:
        phi=np.asarray(flux_cg,dtype=float)
        if phi.shape!=(self.data.cells,self.data.groups):
            raise ValueError("flux shape mismatch")
        return float(np.dot(self.diffusion.boundary_current_coefficient,
                            phi.flatten()))

    def diffusion_loss_partition(self, phi_cg: np.ndarray,
                                 grid: DiffusionGrid) -> tuple[float,float,float]:
        """(net integrated removal, absorption, external leakage).

        Group scattering and interior face currents must CANCEL in global sum.
        """
        phi=np.asarray(phi_cg,dtype=float)
        if phi.shape!=(self.data.cells,self.data.groups):
            raise ValueError("flux shape mismatch")
        volume=self.diffusion.cell_volume_m3.flatten()
        a=float(np.sum(volume[:,None]*
                       np.asarray(grid.absorption_inv_m).reshape(phi.shape)*phi))
        l=self.boundary_leakage(phi)
        total=float(np.sum(self.diffusion.loss @ phi.flatten()))
        return total,a,l


@dataclass(frozen=True)
class EigenKineticsReference:
    keff: float
    forward: np.ndarray
    adjoint: np.ndarray
    generation_time_s: float
    beta_effective_by_family: tuple[float,...]
    beta_unweighted_by_family: tuple[float,...]
    equilibrium_state: SpatialFamilyState | None
    forward_integrated_residual: float
    adjoint_integrated_residual: float
    importance_fission_normalization: float


def cartesian_to_spatial_families(
    grid: DiffusionGrid, *,
    source_id: str,
    beta_by_incident_group_and_family: np.ndarray,
    delayed_spectra_by_family: np.ndarray,
    decay_constants_s_inv: np.ndarray,
    group_speeds_m_s: np.ndarray,
    prompt_spectrum: np.ndarray | None=None,
) -> CartesianFamilyAssembly:
    """Build a Stage 7 spatial-family model from the actual Stage 2 FV A.

    No coefficients are rescaled. In particular, the cell-integrated A is
    transferred unchanged and volumes are used ONCE for fission production,
    precursor births and neutron time-derivative mass.
    """
    mat=assemble_diffusion(grid)
    ny,nx,ng=mat.shape
    nc=ny*nx
    lambdas=np.asarray(decay_constants_s_inv,dtype=float)
    if lambdas.ndim!=1 or len(lambdas)<1:
        raise ValueError("at least one delayed family required")
    nf=lambdas.size
    speed=_positive_array(group_speeds_m_s,(ng,),"speeds")
    beta=np.asarray(beta_by_incident_group_and_family,dtype=float)
    if beta.shape != (nc,ng,nf):
        raise ValueError("beta must be [cell,incident_group,family]")
    chi_d=np.asarray(delayed_spectra_by_family,dtype=float)
    if chi_d.shape!=(nc,nf,ng):
        raise ValueError("delayed spectra must be [cell,family,emitted_group]")
    grid_chi=np.asarray(grid.fission_spectrum,dtype=float).reshape(nc,ng)
    prompt=grid_chi if prompt_spectrum is None else np.asarray(prompt_spectrum,dtype=float)
    if prompt.shape!=(nc,ng):
        raise ValueError("prompt spectrum must be [cell,emitted_group]")
    data=SpatialFamilyData(
        source_id=source_id,
        integrated_loss=mat.loss.copy(),
        volumes_m3=mat.cell_volume_m3.reshape(nc).copy(),
        group_speeds_m_s=speed.copy(),
        nu_sigma_f_inv_m=np.asarray(grid.nu_fission_inv_m,dtype=float)
                              .reshape(nc,ng).copy(),
        beta_by_incident_group_and_family=beta.copy(),
        prompt_spectrum=prompt.copy(),
        delayed_spectra_by_family=chi_d.copy(),
        decay_constants_s_inv=lambdas.copy()
    )
    return CartesianFamilyAssembly(data,mat,mat.shape)


def _is_same_birth_spectrum(assembly: CartesianFamilyAssembly,grid: DiffusionGrid):
    nc,ng=assembly.data.cells,assembly.data.groups
    chi=np.asarray(grid.fission_spectrum).reshape(nc,ng)
    return (np.allclose(assembly.data.prompt_spectrum,chi,rtol=0,atol=1e-13)
            and np.allclose(assembly.data.delayed_spectra_by_family,
                            chi[:,None,:],rtol=0,atol=1e-13))


def diffusion_eigen_kinetics_reference(
    assembly: CartesianFamilyAssembly,grid: DiffusionGrid, *,
    critical_tolerance:float=1e-9,
    eigen_tolerance:float=1e-11
) -> EigenKineticsReference:
    """Verify static forward/adjoint pairing; provide equilibrium when valid.

    A left eigenvector of the *cell-integrated* A,F is already the correct
    importance weighting for the integrated source F phi. Do NOT multiply
    its pairing by V again or convert psi to psi/V: the integrated source
    is already V-weighted. Stage 6 importance_snapshot() itself applies V
    to fission source densities, which produces exactly this pairing.
    """
    if (not math.isfinite(critical_tolerance) or critical_tolerance<=0
            or not math.isfinite(eigen_tolerance) or eigen_tolerance<=0):
        raise ValueError("positive finite eigen tolerances required")
    m=assemble_diffusion(grid)
    if (m.shape!=assembly.shape_yxg or m.loss.shape!=assembly.diffusion.loss.shape
            or (m.loss-assembly.diffusion.loss).nnz!=0
            or (m.fission-assembly.diffusion.fission).nnz!=0):
        raise ValueError("adapter and supplied diffusion model are incompatible")
    solution=solve_keff(grid,tolerance=eigen_tolerance)
    phi=solution.forward.flatten()
    psi=solution.adjoint.flatten()
    prod=m.fission@phi
    pairing=float(psi@prod)
    if not math.isclose(pairing,1.,rel_tol=1e-9,abs_tol=1e-10):
        raise ArithmeticError("adjoint fission-weight pairing must be 1")
    M=assembly.data.mass_diagonal
    generation=float(np.dot(psi,M*phi)/pairing)
    if not math.isfinite(generation) or generation<=0:
        raise ArithmeticError("invalid prompt generation time")
    impact=importance_snapshot(
        assembly.data,
        SpatialFamilyState(0.,solution.forward.reshape(assembly.data.cells,
                                                        assembly.data.groups),
                           np.zeros((assembly.data.cells,assembly.data.families))),
        solution.adjoint.reshape(assembly.data.cells,assembly.data.groups)
    )
    if not _is_same_birth_spectrum(assembly,grid):
        raise ValueError(
            "Stage 2 static eigenmode cannot be used with different delayed/"
            "prompt birth spectra; the resulting total fission operator differs"
        )
    if not math.isclose(impact.weighted_total,pairing,rel_tol=1e-8,abs_tol=1e-9):
        raise ArithmeticError("family spectral partition does not reproduce FV F")
    equilibrium=None
    if abs(solution.keff-1.)<=critical_tolerance:
        flux=solution.forward.reshape(assembly.data.cells,assembly.data.groups)
        C=stationary_precursors(assembly.data,flux)
        equilibrium=SpatialFamilyState(0.,flux,C)
        step=advance_spatial_families(assembly.data,equilibrium,0.1)
        flux_res=float(np.linalg.norm(step.state.flux-flux,ord=np.inf))
        inventory_res=float(np.linalg.norm(step.state.precursors-C,ord=np.inf))
        if (flux_res>5e-8*max(1.,float(np.linalg.norm(flux,ord=np.inf)))
            or inventory_res>5e-8*max(1.,float(np.linalg.norm(C,ord=np.inf)))):
            raise ArithmeticError("critical eigenmode is not stationary in family equations")
    ra=m.loss@phi-(m.fission@phi)/solution.keff
    rb=m.loss.T@psi-(m.fission.T@psi)/solution.keff
    scale_a=max(1.,float(np.linalg.norm(m.loss@phi,ord=np.inf)))
    scale_b=max(1.,float(np.linalg.norm(m.loss.T@psi,ord=np.inf)))
    return EigenKineticsReference(
        keff=solution.keff,forward=solution.forward,adjoint=solution.adjoint,
        generation_time_s=generation,
        beta_effective_by_family=impact.beta_effective_by_family,
        beta_unweighted_by_family=impact.beta_unweighted_by_family,
        equilibrium_state=equilibrium,
        forward_integrated_residual=float(np.linalg.norm(ra,ord=np.inf)/scale_a),
        adjoint_integrated_residual=float(np.linalg.norm(rb,ord=np.inf)/scale_b),
        importance_fission_normalization=pairing
    )
