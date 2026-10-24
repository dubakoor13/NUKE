"""Independent 2D Cartesian multigroup diffusion benchmark (FX2-TH inspired).

Provenance: ANL-78-97, sec II.A-B, cell integrated Eq II.5-II.9.
No historical FORTRAN source has been used. Not an OpenMC transport model,
not NERVA reactor geometry or materials, not reactor kinetics/time evolution.

Conventions:
  phi[y,x,g] is cell-average scalar flux, in arbitrary normalization.
  scattering[y,x,g_from,g_to] excludes within-group self-scatter.
  Loss A = absorption + out-scatter + Fick leakage - in-scatter.
  Production F = chi[g_to] * sum(nu_sigma_f[g_from] * phi[g_from]).
  Eigenproblem A phi = F phi/k; adjoint A.T psi = F.T psi/k.

All macroscopic cross sections [1/m], D [m], coordinates [m].
Finite-volume balances are per unit out-of-plane depth, hence cell volumes
are physical areas [m^2] times a unit depth of 1 m.
Boundary choices: reflecting (zero normal current) and zero_flux
(zero scalar flux AT the boundary face, not a Marshak/vacuum BC).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu


BOUNDARY_SIDES = ('left', 'right', 'bottom', 'top')
VALID_BOUNDARIES = ('reflecting', 'zero_flux')


@dataclass(frozen=True)
class DiffusionGrid:
    x_edges_m: np.ndarray
    y_edges_m: np.ndarray
    diffusion_m: np.ndarray
    absorption_inv_m: np.ndarray
    scattering_inv_m: np.ndarray
    nu_fission_inv_m: np.ndarray
    fission_spectrum: np.ndarray
    boundaries: Mapping[str, str]

    def __post_init__(self) -> None:
        x = np.asarray(self.x_edges_m, dtype=float)
        y = np.asarray(self.y_edges_m, dtype=float)
        if (x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2
                or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
                or np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)):
            raise ValueError('edges must be strictly increasing and finite')
        ny, nx = len(y)-1, len(x)-1
        D = np.asarray(self.diffusion_m, dtype=float)
        if D.ndim != 3 or D.shape[:2] != (ny, nx) or D.shape[2] < 1:
            raise ValueError('diffusion shape must be [ny,nx,group]')
        ng = D.shape[2]
        vectors = (('absorption',self.absorption_inv_m),
                   ('nu_fission',self.nu_fission_inv_m),
                   ('fission_spectrum',self.fission_spectrum))
        for name, value in vectors:
            a=np.asarray(value,dtype=float)
            if a.shape != D.shape or not np.all(np.isfinite(a)) or np.any(a<0):
                raise ValueError(f'{name} invalid value/shape')
        s = np.asarray(self.scattering_inv_m, dtype=float)
        if (s.shape != (ny,nx,ng,ng) or not np.all(np.isfinite(s))
                or np.any(s < 0)):
            raise ValueError('scattering shape or values invalid')
        if np.any(np.diagonal(s,axis1=-2,axis2=-1) != 0):
            raise ValueError('self scattering must be zero (already cancels)')
        if not np.all(np.isfinite(D)) or np.any(D<=0):
            raise ValueError('diffusion coefficients must be finite positive')
        if not np.allclose(np.sum(self.fission_spectrum,axis=-1),1.,atol=1e-12,rtol=0):
            raise ValueError('fission spectrum must sum to 1 in each cell')
        if not np.any(np.asarray(self.nu_fission_inv_m)>0):
            raise ValueError('fission spectrum eigenproblem needs a fission source')
        if set(self.boundaries) != set(BOUNDARY_SIDES) or any(
                v not in VALID_BOUNDARIES for v in self.boundaries.values()):
            raise ValueError('all four boundary conditions required')
        # Requiring absorption in every group makes inverse-power iteration
        # well-defined with fully reflecting outer boundaries.
        if np.any(np.asarray(self.absorption_inv_m)<=0):
            raise ValueError('absorption must be positive in this benchmark')


@dataclass(frozen=True)
class DiffusionMatrices:
    loss: csc_matrix
    fission: csc_matrix
    boundary_current_coefficient: np.ndarray
    cell_volume_m3: np.ndarray
    shape: tuple[int,int,int]


@dataclass(frozen=True)
class EigenSolution:
    keff: float
    forward: np.ndarray
    adjoint: np.ndarray
    forward_residual: float
    adjoint_residual: float
    forward_iterations: int
    adjoint_iterations: int
    biorthogonal_fission_norm: float
    absorption_neutrons_per_s: float
    boundary_leakage_neutrons_per_s: float
    fission_neutrons_per_s: float
    global_neutron_balance_residual: float


def assemble_diffusion(grid: DiffusionGrid) -> DiffusionMatrices:
    """Conservative harmonic-interface finite volume stencil.

    Indexed [row (y),col (x),group]; per unit out-of-plane depth.
    Two neighboring cells share exactly the same face coefficient.
    """
    dx=np.diff(np.asarray(grid.x_edges_m,dtype=float))
    dy=np.diff(np.asarray(grid.y_edges_m,dtype=float))
    ny,nx,ng=np.asarray(grid.diffusion_m).shape
    D=np.asarray(grid.diffusion_m,dtype=float)
    Sa=np.asarray(grid.absorption_inv_m,dtype=float)
    S=np.asarray(grid.scattering_inv_m,dtype=float)
    nf=np.asarray(grid.nu_fission_inv_m,dtype=float)
    chi=np.asarray(grid.fission_spectrum,dtype=float)
    nv=ny*nx*ng
    diag=np.zeros(nv)
    leakage=np.zeros(nv)
    rows=[]; cols=[]; vals=[]
    frows=[]; fcols=[]; fvals=[]
    cell_volumes=np.empty((ny,nx))
    def idx(iy,ix,g): return ((iy*nx+ix)*ng+g)
    def offdiag(i,j,value):
        rows.append(i);cols.append(j);vals.append(value)
    for iy in range(ny):
        for ix in range(nx):
            vol=float(dx[ix]*dy[iy]) # unit depth
            cell_volumes[iy,ix]=vol
            for g in range(ng):
                i=idx(iy,ix,g)
                diag[i] += vol*(Sa[iy,ix,g]+S[iy,ix,g,:].sum())
                for g_from in range(ng):
                    if g_from != g and S[iy,ix,g_from,g] != 0:
                        offdiag(i,idx(iy,ix,g_from),-vol*S[iy,ix,g_from,g])
                    production=vol*chi[iy,ix,g]*nf[iy,ix,g_from]
                    if production>0:
                        frows.append(i);fcols.append(idx(iy,ix,g_from));fvals.append(production)
                # face at left/right of cell
                for is_left in (True,False):
                    jx=ix-1 if is_left else ix+1
                    if 0<=jx<nx:
                        gap=dx[ix]/(2*D[iy,ix,g])+dx[jx]/(2*D[iy,jx,g])
                        t=dy[iy]/gap
                        diag[i]+=t
                        offdiag(i,idx(iy,jx,g),-t)
                    else:
                        side='left' if is_left else 'right'
                        if grid.boundaries[side]=='zero_flux':
                            t=2*D[iy,ix,g]*dy[iy]/dx[ix]
                            diag[i]+=t
                            leakage[i]+=t
                # face at bottom/top of cell
                for is_bottom in (True,False):
                    jy=iy-1 if is_bottom else iy+1
                    if 0<=jy<ny:
                        gap=dy[iy]/(2*D[iy,ix,g])+dy[jy]/(2*D[jy,ix,g])
                        t=dx[ix]/gap
                        diag[i]+=t
                        offdiag(i,idx(jy,ix,g),-t)
                    else:
                        side='bottom' if is_bottom else 'top'
                        if grid.boundaries[side]=='zero_flux':
                            t=2*D[iy,ix,g]*dx[ix]/dy[iy]
                            diag[i]+=t
                            leakage[i]+=t
    for i in range(nv):
        rows.append(i);cols.append(i);vals.append(diag[i])
    A=coo_matrix((vals,(rows,cols)),shape=(nv,nv)).tocsc()
    F=coo_matrix((fvals,(frows,fcols)),shape=(nv,nv)).tocsc()
    return DiffusionMatrices(A,F,leakage,cell_volumes,(ny,nx,ng))


def _iterate(matrices: DiffusionMatrices, adjoint: bool, tol: float,
             maxiter: int) -> tuple[float,np.ndarray,float,int]:
    A,F=matrices.loss,matrices.fission
    lu=splu(A)
    vec=np.ones(A.shape[0],dtype=float)
    vec/=vec.sum()
    for n in range(1,maxiter+1):
        if adjoint:
            nxt=lu.solve(F.T @ vec,trans='T')
            AA=A.T; FF=F.T
        else:
            nxt=lu.solve(F @ vec)
            AA=A;FF=F
        if np.any(~np.isfinite(nxt)) or np.any(nxt < -1e-12):
            raise ArithmeticError('invalid multigroup eigenvector iterate')
        nxt=np.maximum(nxt,0)
        k=float(np.sum(nxt))
        if not math.isfinite(k) or k<=0:
            raise ArithmeticError('nonpositive multigroup eigenvalue estimate')
        nxt/=k
        residual_vec=(AA @ nxt)-(FF @ nxt)/k
        scale=max(np.linalg.norm(AA @ nxt,ord=np.inf),
                  np.linalg.norm((FF @ nxt)/k,ord=np.inf),1e-30)
        residual=float(np.linalg.norm(residual_vec,ord=np.inf)/scale)
        if residual < tol and np.linalg.norm(nxt-vec,ord=np.inf)<tol:
            return k,nxt,residual,n
        vec=nxt
    raise RuntimeError(f'{"adjoint" if adjoint else "forward"} eigen iteration did not converge within {maxiter} iterations')


def solve_keff(grid: DiffusionGrid, *, tolerance:float=1e-10,
               max_iterations:int=5000) -> EigenSolution:
    if not math.isfinite(tolerance) or tolerance<=0 or tolerance>1e-2:
        raise ValueError('invalid eigen residual tolerance')
    if not isinstance(max_iterations,int) or max_iterations<1:
        raise ValueError('invalid max_iterations')
    m=assemble_diffusion(grid)
    k, phi, f_res, fi=_iterate(m,False,tolerance,max_iterations)
    ka,psi,a_res,ai=_iterate(m,True,tolerance,max_iterations)
    if abs(k-ka)> max(tolerance*max(k,ka)*20,1e-12):
        raise ArithmeticError('forward and adjoint eigenvalues disagree')
    # Normalize forward flux by integrated scalar flux; normalize adjoint
    # so psi^T F phi = 1, a reproducible bi-orthogonal gauge.
    vols=np.repeat(m.cell_volume_m3.flatten(),m.shape[2])
    phi=phi/float(np.dot(phi,vols))
    norm=float(psi @ (m.fission @ phi))
    if norm<=0 or not math.isfinite(norm):
        raise ArithmeticError('invalid biorthogonal fission pairing')
    psi=psi/norm
    absvec=np.asarray(grid.absorption_inv_m).flatten()*vols
    absorption=float(np.dot(absvec,phi))
    escaped=float(np.dot(m.boundary_current_coefficient,phi))
    produced=float(np.sum(m.fission @ phi))
    return EigenSolution(
        k,phi.reshape(m.shape),psi.reshape(m.shape),f_res,a_res,fi,ai,
        float(psi @ (m.fission @ phi)),absorption,escaped,produced,
        absorption+escaped-produced/k
    )


def solve_fixed_source(grid: DiffusionGrid, source_neutrons_s_m3: np.ndarray,
                       *, minimum_subcritical_margin:float=1e-7) -> np.ndarray:
    """Subcritical steady response (A-F) phi = source*volume.

    This is a diffusion benchmark; not a chain-multiplying Monte Carlo run.
    """
    m=assemble_diffusion(grid)
    q=np.asarray(source_neutrons_s_m3,dtype=float)
    if q.shape!=m.shape or not np.all(np.isfinite(q)) or np.any(q<0):
        raise ValueError('source must be [ny,nx,group] finite and nonnegative')
    if not math.isfinite(minimum_subcritical_margin) or minimum_subcritical_margin<=0:
        raise ValueError('invalid subcritical margin')
    k,_,_,_=_iterate(m,False,1e-9,5000)
    if k>=1-minimum_subcritical_margin:
        raise ValueError('source response requires a subcritical configuration')
    qv=(q*m.cell_volume_m3[:,:,None]).flatten()
    phi=splu(m.loss-m.fission).solve(qv)
    if np.any(~np.isfinite(phi)) or np.any(phi< -1e-12):
        raise ArithmeticError('nonphysical fixed source flux')
    return np.maximum(phi,0).reshape(m.shape)
