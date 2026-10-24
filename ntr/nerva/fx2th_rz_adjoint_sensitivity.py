"""Exact first-order material perturbations of NUKE's spatial core eigenproblem.

For Stage-15 full-family fission and Stage-12 conservative axisymmetric r-z
loss matrices, the generalized eigenproblem is

    A(phi) = F(phi)/k,       psi.T A = psi.T F/k.

For externally supplied parameter theta (possibly temperature, but NO
temperature laws are inferred), at fixed mesh and fixed material structure:

    (1/k) dk/dtheta = [psi.T (dF/dtheta - k*dA/dtheta) phi]
                      / [psi.T F phi],
    d(1-1/k)/dtheta = (dk/dtheta)/k**2.

dA includes differentiated harmonic face transmissibilities and boundary
current terms as well as group scattering/removal. dF uses explicitly
different PROMPT and individual DELAYED family emitted energy spectra;
beta is indexed by incident energy group and delayed family.

This code does not supply dXS/dT, data covariance, NERVA compositions or
hot-core reactor design values. Such coefficients must be qualified and
provided by external nuclear-data/state reconstruction. Geometry, mesh
edges, group speeds, delayed-family decay constants and volumes remain
FIXED for the derivative. Derivatives are signed, unlike baseline XS.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, block_diag, issparse

from .fx2th_rz_family_diffusion import RZGrid, assemble_rz
from .fx2th_family_spacetime import SpatialFamilyData
from .fx2th_full_family_fission_eigen import (
    FamilyEigenReference, assemble_family_fission, solve_family_eigen,
)


@dataclass(frozen=True)
class RZLossDerivative:
    matrix: csc_matrix
    removal: csc_matrix
    leakage: csc_matrix
    scattering: csc_matrix
    source_id: str


@dataclass(frozen=True)
class FissionDerivative:
    prompt: csc_matrix
    delayed_by_family: tuple[csc_matrix, ...]
    total: csc_matrix


@dataclass(frozen=True)
class CoreEigenSensitivity:
    keff: float
    d_keff_d_parameter: float
    d_log_keff_d_parameter: float
    d_reactivity_d_parameter: float
    loss_contribution_dlogk: float
    prompt_fission_contribution_dlogk: float
    delayed_fission_contribution_dlogk: tuple[float, ...]
    pairing_psi_f_phi: float
    source_id: str


def _signed(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: finite signed derivative shape {shape} required")
    return a


def _zero_or_signed(name: str, value, shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape,dtype=float) if value is None else _signed(name,value,shape)


def assemble_family_fission_derivative(
    data: SpatialFamilyData, *,
    d_nu_sigma_f_inv_m=None,
    d_beta_by_incident_group_and_family=None,
    d_prompt_spectrum=None,
    d_delayed_spectra_by_family=None,
) -> FissionDerivative:
    """Analytically differentiate the fixed-geometry prompt + delayed source.

    Spectrum tangents must sum to ZERO over EMITTED energy group, preserving
    their baseline sum-to-one normalization. The beta tangent may have
    signed components and preserves the existing group/family ordering.
    An identity direction for nuSigmaF=nuSigmaF produces dF=F.
    """
    nc,ng,nf=data.cells,data.groups,data.families
    dx=_zero_or_signed("d_nu_sigma_f_inv_m",d_nu_sigma_f_inv_m,(nc,ng))
    db=_zero_or_signed("d_beta",d_beta_by_incident_group_and_family,(nc,ng,nf))
    dp=_zero_or_signed("d_prompt_spectrum",d_prompt_spectrum,(nc,ng))
    dd=_zero_or_signed("d_delayed_spectra",d_delayed_spectra_by_family,(nc,nf,ng))
    if (not np.allclose(dp.sum(axis=1),0.,rtol=0,atol=1e-12) or
        not np.allclose(dd.sum(axis=2),0.,rtol=0,atol=1e-12)):
        raise ValueError("spectrum derivatives must sum to zero over emitted group")
    v=np.asarray(data.volumes_m3)
    xs=np.asarray(data.nu_sigma_f_inv_m)
    beta=np.asarray(data.beta_by_incident_group_and_family)
    chip=np.asarray(data.prompt_spectrum)
    chid=np.asarray(data.delayed_spectra_by_family)
    pb=[]
    family=[[] for _ in range(nf)]
    for c in range(nc):
        p_rate=v[c]*xs[c]*(1.-beta[c].sum(axis=1))
        dp_rate=v[c]*(dx[c]*(1.-beta[c].sum(axis=1)) -
                         xs[c]*db[c].sum(axis=1))
        pb.append(np.outer(dp[c],p_rate)+np.outer(chip[c],dp_rate))
        for f in range(nf):
            d_rate=v[c]*(dx[c]*beta[c,:,f]+xs[c]*db[c,:,f])
            base_rate=v[c]*xs[c]*beta[c,:,f]
            family[f].append(np.outer(dd[c,f],base_rate) +
                             np.outer(chid[c,f],d_rate))
    prompt=block_diag([csc_matrix(m) for m in pb],format="csc")
    prompt.eliminate_zeros()
    delayed=[]
    for blocks in family:
        s=block_diag([csc_matrix(m) for m in blocks],format="csc")
        s.eliminate_zeros()
        delayed.append(s)
    total=prompt.copy()
    for block in delayed:
        total=total+block
    total.eliminate_zeros()
    return FissionDerivative(prompt,tuple(delayed),total.tocsc())


def _coo(n, rows, cols, vals) -> csc_matrix:
    m=coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsc()
    m.sum_duplicates()
    m.eliminate_zeros()
    return m


def assemble_rz_loss_derivative(
    grid: RZGrid, *,
    d_diffusion_m=None,
    d_absorption_inv_m=None,
    d_scattering_inv_m=None,
) -> RZLossDerivative:
    """Differentiate EXACT annular FV A at fixed cell boundaries.

    The harmonic interface coefficient T = area/(l_i/D_i+l_j/D_j)
    has derivative
      dT = area*(l_i*dD_i/D_i**2+l_j*dD_j/D_j**2)
                  /(l_i/D_i+l_j/D_j)**2.
    At a zero-flux boundary the centre-to-face resistance alone applies.
    """
    shape=np.asarray(grid.diffusion_m).shape
    nr,nz,ng=shape
    dD=_zero_or_signed("d_diffusion_m",d_diffusion_m,shape)
    dSa=_zero_or_signed("d_absorption",d_absorption_inv_m,shape)
    dS=_zero_or_signed("d_scattering",d_scattering_inv_m,shape+(ng,))
    if np.any(np.diagonal(dS,axis1=2,axis2=3)!=0):
        raise ValueError("self-scattering derivatives must have zero diagonal")
    rr=np.asarray(grid.radial_edges_m,dtype=float)
    zz=np.asarray(grid.axial_edges_m,dtype=float)
    dr=np.diff(rr); dz=np.diff(zz)
    D=np.asarray(grid.diffusion_m,dtype=float)
    V=np.pi*(rr[1:]**2-rr[:-1]**2)[:,None]*dz[None,:]
    n=nr*nz*ng
    parts=[([],[],[]) for _ in range(3)]  # absorption, interface/boundary, scattering
    def add(category,row,col,value):
        if value!=0.:
            a,b,c=parts[category]
            a.append(row);b.append(col);c.append(float(value))
    def ix(i,j,g):
        return (i*nz+j)*ng+g
    def face(area,l_i,D_i,d_i,l_j,D_j,d_j):
        R=l_i/D_i+l_j/D_j
        return area*(l_i*d_i/(D_i*D_i)+l_j*d_j/(D_j*D_j))/(R*R)

    for i in range(nr):
        axial_area=np.pi*(rr[i+1]**2-rr[i]**2)
        for j in range(nz):
            for g in range(ng):
                k=ix(i,j,g)
                add(0,k,k,V[i,j]*dSa[i,j,g])
                add(2,k,k,V[i,j]*float(dS[i,j,g].sum()))
                for h in range(ng):
                    if h!=g:
                        add(2,k,ix(i,j,h),-V[i,j]*dS[i,j,h,g])
                for di in (-1,1):
                    ii=i+di
                    face_r=rr[i] if di<0 else rr[i+1]
                    area=2*np.pi*face_r*dz[j]
                    if 0<=ii<nr:
                        dt=face(area,dr[i]/2,D[i,j,g],dD[i,j,g],
                                    dr[ii]/2,D[ii,j,g],dD[ii,j,g])
                        add(1,k,k,dt)
                        add(1,k,ix(ii,j,g),-dt)
                    elif di==1 and grid.outer_boundary=="zero_flux":
                        dt=area*2*dD[i,j,g]/dr[i]
                        add(1,k,k,dt)
                    # The cylindrical symmetry axis has exactly zero area.
                for dj in (-1,1):
                    jj=j+dj
                    if 0<=jj<nz:
                        dt=face(axial_area,dz[j]/2,D[i,j,g],dD[i,j,g],
                                          dz[jj]/2,D[i,jj,g],dD[i,jj,g])
                        add(1,k,k,dt)
                        add(1,k,ix(i,jj,g),-dt)
                    else:
                        boundary=grid.lower_boundary if dj<0 else grid.upper_boundary
                        if boundary=="zero_flux":
                            dt=axial_area*2*dD[i,j,g]/dz[j]
                            add(1,k,k,dt)
    pieces=[_coo(n,*part) for part in parts]
    total=pieces[0]+pieces[1]+pieces[2]
    total.sum_duplicates()
    total.eliminate_zeros()
    return RZLossDerivative(total.tocsc(),pieces[0],pieces[1],pieces[2],
                            "fixed-geometry axisymmetric r-z first-order material tangent")


def evaluate_core_eigen_sensitivity(
    data: SpatialFamilyData,
    d_loss: csc_matrix,
    d_fission: FissionDerivative,
    *, eigen: FamilyEigenReference | None = None,
) -> CoreEigenSensitivity:
    """Return analytic first derivative of k and reactivity 1-1/k.

    The reference and tangent MUST correspond to the identical nuclear
    material and mesh. By default solve a fresh eigenproblem, rather than
    trust a potentially stale previously computed eigenpair. The returned
    sensitivities are per unit externally specified scalar perturbation;
    this module cannot infer whether that scalar means kelvin or another
    independent material-state variable.
    """
    if not issparse(d_loss):
        raise TypeError("loss tangent must be a SciPy sparse matrix")
    ref=solve_family_eigen(data) if eigen is None else eigen
    if ref.source_id!=data.source_id:
        raise ValueError("eigen reference and nuclear-data source IDs differ")
    n=data.cells*data.groups
    if (d_loss.shape!=(n,n) or
        d_fission.total.shape!=(n,n) or
        any(x.shape!=(n,n) for x in
            (d_fission.prompt,*d_fission.delayed_by_family)) or
        len(d_fission.delayed_by_family)!=data.families):
        raise ValueError("loss/fission derivative matrix dimensions mismatch")
    if (not np.all(np.isfinite(d_loss.data)) or
        not np.all(np.isfinite(d_fission.total.data)) or
        any(not np.all(np.isfinite(x.data))
            for x in (d_fission.prompt,*d_fission.delayed_by_family))):
        raise ValueError("tangent operators must be finite")
    phi=np.asarray(ref.forward_flux).flatten()
    psi=np.asarray(ref.adjoint_flux).flatten()
    base=assemble_family_fission(data).total
    norm=float(psi@(base@phi))
    if not math.isfinite(norm) or norm<=0:
        raise ArithmeticError("adjoint fission normalization invalid")
    k=float(ref.keff)
    if not math.isfinite(k) or k<=0:
        raise ArithmeticError("invalid keff reference")
    # Component contributions are derivatives of ln(k), not dk.
    loss=-k*float(psi@(d_loss@phi))/norm
    prompt=float(psi@(d_fission.prompt@phi))/norm
    delayed=tuple(float(psi@(m@phi))/norm for m in d_fission.delayed_by_family)
    dlog=loss+prompt+sum(delayed)
    dk=k*dlog
    return CoreEigenSensitivity(
        keff=k, d_keff_d_parameter=dk,
        d_log_keff_d_parameter=dlog,
        d_reactivity_d_parameter=dk/(k*k),
        loss_contribution_dlogk=loss,
        prompt_fission_contribution_dlogk=prompt,
        delayed_fission_contribution_dlogk=delayed,
        pairing_psi_f_phi=norm,
        source_id=data.source_id,
    )


def evaluate_rz_material_sensitivity(
    grid:RZGrid,data:SpatialFamilyData, *,
    d_diffusion_m=None, d_absorption_inv_m=None, d_scattering_inv_m=None,
    d_nu_sigma_f_inv_m=None, d_beta_by_incident_group_and_family=None,
    d_prompt_spectrum=None,d_delayed_spectra_by_family=None,
) -> CoreEigenSensitivity:
    """Single entry point for all fixed-mesh r-z material derivatives."""
    assembled=assemble_rz(grid)
    if (assembled.loss.shape!=data.integrated_loss.shape or
        (assembled.loss-data.integrated_loss).nnz!=0 or
        not np.array_equal(assembled.cell_volumes_m3.flatten(),data.volumes_m3)
        or not np.array_equal(
            np.asarray(grid.nu_fission_inv_m).reshape(data.cells,data.groups),
            data.nu_sigma_f_inv_m)):
        raise ValueError("r-z loss or fission coefficients differ from kinetic data")
    loss=assemble_rz_loss_derivative(
        grid,d_diffusion_m=d_diffusion_m,
        d_absorption_inv_m=d_absorption_inv_m,
        d_scattering_inv_m=d_scattering_inv_m)
    fission=assemble_family_fission_derivative(
        data,d_nu_sigma_f_inv_m=d_nu_sigma_f_inv_m,
        d_beta_by_incident_group_and_family=d_beta_by_incident_group_and_family,
        d_prompt_spectrum=d_prompt_spectrum,
        d_delayed_spectra_by_family=d_delayed_spectra_by_family)
    return evaluate_core_eigen_sensitivity(data,loss.matrix,fission)
