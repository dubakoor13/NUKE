"""Conservative axisymmetric (r,z) multigroup neutron diffusion.

An independently assembled *cell-integrated* finite-volume benchmark for
general nonuniform cylindrical meshes and group-resolved heterogeneous
synthetic cross sections. Unlike the older NUKE r-z eigenvalue-only module,
this exports sparse A/F matrices with physical cylindrical cell volumes for
the Stage-7 spatial delayed-neutron family transient.

A*phi = F*phi/k, with A the removal + leakage - inscatter operator;
F the TOTAL fission birth operator. Radial axis symmetry requires ZERO
current at r=0. A zero-flux outer/axial face is a Dirichlet extrapolated
boundary, NOT a transport vacuum/Marshak boundary.

All geometry is in m; macroscopic cross sections are inverse m; D in m.
The annulus volume is pi*(r_outer**2 - r_inner**2)*dz, and interfaces
use exact surface area divided by summed half-cell resistances. The sparse
index is ((ir*nz+iz)*ng+group), i.e. (r,z,g) row-major.

This is mathematics on externally supplied coefficients, not the physical
NERVA hexagonal fuel/core, evaluated XS, actual OpenMC transport or original
historical FX2-TH source.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

_VALID_BC = ("reflecting","zero_flux")


@dataclass(frozen=True)
class RZGrid:
    radial_edges_m: np.ndarray
    axial_edges_m: np.ndarray
    diffusion_m: np.ndarray           # [nr,nz,G]
    absorption_inv_m: np.ndarray      # [nr,nz,G]
    scattering_inv_m: np.ndarray      # [nr,nz,G_from,G_to]
    nu_fission_inv_m: np.ndarray      # [nr,nz,G]
    fission_spectrum: np.ndarray      # [nr,nz,G_emitted]
    outer_boundary: str = "zero_flux"
    lower_boundary: str = "zero_flux"
    upper_boundary: str = "zero_flux"

    def __post_init__(self):
        r=np.asarray(self.radial_edges_m,dtype=float)
        z=np.asarray(self.axial_edges_m,dtype=float)
        if (r.ndim!=1 or z.ndim!=1 or len(r)<2 or len(z)<2 or
            not np.all(np.isfinite(r)) or not np.all(np.isfinite(z)) or
            not math.isclose(float(r[0]),0.,abs_tol=1e-14) or
            np.any(np.diff(r)<=0) or np.any(np.diff(z)<=0)):
            raise ValueError("RZ edges must increase, with radial axis at r=0")
        nr,nz=len(r)-1,len(z)-1
        D=np.asarray(self.diffusion_m,dtype=float)
        if D.ndim!=3 or D.shape[:2]!=(nr,nz) or D.shape[-1]<1:
            raise ValueError("D requires [radial,axial,group] shape")
        ng=D.shape[-1]
        for name,a,positive in (
            ("D",D,True),("absorption",self.absorption_inv_m,True),
            ("nuSigmaF",self.nu_fission_inv_m,False),
            ("chi",self.fission_spectrum,False)):
            arr=np.asarray(a,dtype=float)
            if (arr.shape!=(nr,nz,ng) or not np.all(np.isfinite(arr)) or
                 np.any(arr<=0 if positive else arr<0)):
                raise ValueError(name+" must be finite "+("positive" if positive else "nonnegative"))
        S=np.asarray(self.scattering_inv_m,dtype=float)
        if (S.shape!=(nr,nz,ng,ng) or np.any(~np.isfinite(S)) or
            np.any(S<0) or np.any(np.diagonal(S,axis1=-2,axis2=-1)!=0)):
            raise ValueError("scattering must be nonnegative, with zero diagonal")
        if (not np.allclose(np.sum(self.fission_spectrum,axis=2),1,atol=1e-12,rtol=0)
            or not np.any(np.asarray(self.nu_fission_inv_m)>0)):
            raise ValueError("fission production/spectrum invalid")
        if any(x not in _VALID_BC for x in
               (self.outer_boundary,self.lower_boundary,self.upper_boundary)):
            raise ValueError("outer, lower and upper boundaries require reflecting or zero_flux")


@dataclass(frozen=True)
class RZMatrices:
    loss: csc_matrix
    fission: csc_matrix
    cell_volumes_m3: np.ndarray            # [nr,nz]
    boundary_current_coefficients: np.ndarray # [nr,nz,G]
    shape: tuple[int,int,int]


@dataclass(frozen=True)
class RZEigen:
    keff: float
    forward: np.ndarray    # [nr,nz,G]
    adjoint: np.ndarray
    forward_residual: float
    adjoint_residual: float
    forward_iterations: int
    adjoint_iterations: int
    absorption_rate: float
    leakage_rate: float
    fission_rate: float
    balance_residual: float
    biorthogonal_fission_norm: float


def assemble_rz(grid: RZGrid) -> RZMatrices:
    """Conservative finite volume, harmonic diffusion and exact annular area."""
    r=np.asarray(grid.radial_edges_m,dtype=float)
    z=np.asarray(grid.axial_edges_m,dtype=float)
    nr,nz,ng=np.asarray(grid.diffusion_m).shape
    D=np.asarray(grid.diffusion_m,dtype=float)
    Sa=np.asarray(grid.absorption_inv_m,dtype=float)
    S=np.asarray(grid.scattering_inv_m,dtype=float)
    Nf=np.asarray(grid.nu_fission_inv_m,dtype=float)
    chi=np.asarray(grid.fission_spectrum,dtype=float)
    dr=np.diff(r);dz=np.diff(z)
    v=np.pi*(r[1:]**2-r[:-1]**2)[:,None]*dz[None,:]
    size=nr*nz*ng
    diag=np.zeros(size,dtype=float)
    leakage=np.zeros(size,dtype=float)
    ri=[];ci=[];va=[]
    fri=[];fci=[];fva=[]
    def index(i,j,g):return (i*nz+j)*ng+g
    for i in range(nr):
        cross_section=np.pi*(r[i+1]**2-r[i]**2)
        for j in range(nz):
            V=v[i,j]
            for g in range(ng):
                k=index(i,j,g)
                diag[k]+=V*(Sa[i,j,g]+S[i,j,g].sum())
                for h in range(ng):
                    if h!=g and S[i,j,h,g]:
                        ri.append(k);ci.append(index(i,j,h));va.append(-V*S[i,j,h,g])
                    prod=V*chi[i,j,g]*Nf[i,j,h]
                    if prod:
                        fri.append(k);fci.append(index(i,j,h));fva.append(prod)
                # Interior annular interfaces, with resistance from each
                # cell centre to the shared radial face. Two equal and opposite
                # transmissibilities ensure exact neutron current conservation.
                for di in (-1,1):
                    ii=i+di
                    face=r[i] if di<0 else r[i+1]
                    if 0<=ii<nr:
                        area=2*np.pi*face*dz[j]
                        t=area/(dr[i]/(2*D[i,j,g])+dr[ii]/(2*D[ii,j,g]))
                        diag[k]+=t
                        ri.append(k);ci.append(index(ii,j,g));va.append(-t)
                    elif di==1 and grid.outer_boundary=="zero_flux":
                        t=2*np.pi*face*dz[j]/(dr[i]/(2*D[i,j,g]))
                        diag[k]+=t;leakage[k]+=t
                    # i=0 axis: face radius = 0, so strictly zero current.
                # Axial interfaces have constant physical annular area.
                for dj in (-1,1):
                    jj=j+dj
                    if 0<=jj<nz:
                        t=cross_section/(dz[j]/(2*D[i,j,g])+
                                         dz[jj]/(2*D[i,jj,g]))
                        diag[k]+=t
                        ri.append(k);ci.append(index(i,jj,g));va.append(-t)
                    else:
                        boundary=grid.lower_boundary if dj<0 else grid.upper_boundary
                        if boundary=="zero_flux":
                            t=cross_section/(dz[j]/(2*D[i,j,g]))
                            diag[k]+=t;leakage[k]+=t
    for i in range(size):
        ri.append(i);ci.append(i);va.append(diag[i])
    A=coo_matrix((va,(ri,ci)),shape=(size,size)).tocsc()
    F=coo_matrix((fva,(fri,fci)),shape=(size,size)).tocsc()
    return RZMatrices(A,F,v,leakage.reshape(nr,nz,ng),(nr,nz,ng))


def _dominant(m:RZMatrices,*,transpose:bool,tol:float,maxiter:int):
    A,F=m.loss,m.fission
    solver=splu(A)
    vec=np.ones(A.shape[0]);vec/=np.sum(vec)
    for iteration in range(1,maxiter+1):
        nextv=(solver.solve(F.T@vec,trans="T") if transpose
               else solver.solve(F@vec))
        if np.any(~np.isfinite(nextv)) or np.any(nextv< -1e-12):
            raise ArithmeticError("nonphysical eigenvector iterate")
        nextv=np.maximum(nextv,0)
        k=float(np.sum(nextv))
        if not math.isfinite(k) or k<=0:
            raise ArithmeticError("nonpositive eigenvalue")
        nextv/=k
        AA=A.T if transpose else A
        FF=F.T if transpose else F
        error=float(np.linalg.norm(AA@nextv-(FF@nextv)/k,ord=np.inf)/
                    max(1e-30,np.linalg.norm(AA@nextv,ord=np.inf),
                        np.linalg.norm((FF@nextv)/k,ord=np.inf)))
        if error<tol and np.linalg.norm(nextv-vec,ord=np.inf)<tol:
            return k,nextv,error,iteration
        vec=nextv
    raise RuntimeError("r-z dominant mode did not converge")


def solve_rz_keff(grid:RZGrid,*,tolerance:float=1e-10,
                  max_iterations:int=5000) -> RZEigen:
    if not math.isfinite(tolerance) or not 0<tolerance<=1e-3:
        raise ValueError("positive tolerance required")
    if not isinstance(max_iterations,int) or max_iterations<1:
        raise ValueError("max_iterations must be >=1")
    m=assemble_rz(grid)
    k,phi,ef,ni=_dominant(m,transpose=False,tol=tolerance,maxiter=max_iterations)
    kt,psi,et,nj=_dominant(m,transpose=True,tol=tolerance,maxiter=max_iterations)
    if abs(kt-k)>max(1e-11,30*tolerance*max(k,kt)):
        raise ArithmeticError("forward and adjoint eigenvalues disagree")
    vols=np.repeat(m.cell_volumes_m3.reshape(-1),m.shape[-1])
    phi=phi/float(np.dot(vols,phi))
    pairing=float(psi@(m.fission@phi))
    if pairing<=0:
        raise ArithmeticError("zero adjoint fission importance")
    psi/=pairing
    absorption=float(np.sum(vols*np.asarray(grid.absorption_inv_m).flatten()*phi))
    escape=float(np.dot(m.boundary_current_coefficients.flatten(),phi))
    prod=float(np.sum(m.fission@phi))
    balance=absorption+escape-prod/k
    return RZEigen(k,phi.reshape(m.shape),psi.reshape(m.shape),ef,et,ni,nj,
                   absorption,escape,prod,balance,float(psi@(m.fission@phi)))


def rz_to_spatial_families(
    grid:RZGrid,*,source_id:str,group_speeds_m_s:np.ndarray,
    beta_by_incident_group_and_family:np.ndarray,
    delayed_spectra_by_family:np.ndarray,
    decay_constants_s_inv:np.ndarray,
    prompt_spectrum:np.ndarray|None=None,
):
    """Construct real Stage-7 spatial-family data from the SAME r-z operator."""
    from .fx2th_family_spacetime import SpatialFamilyData
    m=assemble_rz(grid)
    nr,nz,ng=m.shape;nc=nr*nz
    lam=np.asarray(decay_constants_s_inv,dtype=float)
    if lam.ndim!=1 or lam.size<1:
        raise ValueError("delayed families required")
    nf=lam.size
    beta=np.asarray(beta_by_incident_group_and_family,dtype=float)
    chid=np.asarray(delayed_spectra_by_family,dtype=float)
    if beta.shape!=(nc,ng,nf) or chid.shape!=(nc,nf,ng):
        raise ValueError("family arrays must be [cell,group,family] and [cell,family,group]")
    chi=(np.asarray(grid.fission_spectrum).reshape(nc,ng) if prompt_spectrum is None
         else np.asarray(prompt_spectrum,dtype=float))
    return SpatialFamilyData(
        source_id=source_id,integrated_loss=m.loss,
        volumes_m3=m.cell_volumes_m3.flatten(),
        group_speeds_m_s=np.asarray(group_speeds_m_s,dtype=float),
        nu_sigma_f_inv_m=np.asarray(grid.nu_fission_inv_m).reshape(nc,ng),
        beta_by_incident_group_and_family=beta,
        prompt_spectrum=chi,delayed_spectra_by_family=chid,
        decay_constants_s_inv=lam)


def rz_critical_family_reference(grid:RZGrid,data,*,tolerance:float=1e-8):
    """Verify a mathematically critical r-z eigenmode is stationary in time.

    Only valid if Stage-7 prompt AND delayed spectra agree with the static
    r-z fission birth operator. Otherwise return a clear error: the
    stationary fission operator itself has changed.
    """
    from .fx2th_family_spacetime import (
        SpatialFamilyState,stationary_precursors,advance_spatial_families)
    m=assemble_rz(grid)
    nr,nz,ng=m.shape
    if data.cells!=nr*nz or data.groups!=ng:
        raise ValueError("spatial kinetics mesh/groups differ from r-z")
    if ((m.loss-data.integrated_loss).nnz>0 or
        not np.array_equal(data.volumes_m3,m.cell_volumes_m3.flatten()) or
        not np.array_equal(data.nu_sigma_f_inv_m,
                           grid.nu_fission_inv_m.reshape(data.cells,ng))):
        raise ValueError("spatial family data incompatible with r-z matrices")
    chi=np.asarray(grid.fission_spectrum).reshape(data.cells,ng)
    if (not np.allclose(data.prompt_spectrum,chi,rtol=0,atol=1e-13) or
        not np.allclose(data.delayed_spectra_by_family,chi[:,None,:],
                        rtol=0,atol=1e-13)):
        raise ValueError("distinct family birth spectra change the reference fission matrix")
    eig=solve_rz_keff(grid)
    if abs(eig.keff-1)>tolerance:
        raise ValueError("stationary precursor test requires k-effective ≈ 1")
    flux=eig.forward.reshape(data.cells,ng)
    initial=SpatialFamilyState(0.,flux,stationary_precursors(data,flux))
    stepped=advance_spatial_families(data,initial,.1)
    if not np.allclose(stepped.state.flux,initial.flux,rtol=1e-8,atol=1e-10):
        raise ArithmeticError("critical r-z spatial shape not stationary in family kinetics")
    if not np.allclose(stepped.state.precursors,initial.precursors,
                       rtol=1e-8,atol=1e-10):
        raise ArithmeticError("delayed family equilibrium not stationary")
    return eig,initial
