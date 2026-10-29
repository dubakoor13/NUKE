"""Stage81: conservative nonorthogonal INTERIOR fuel finite-volume faces.

Stage80 corrected the curved FUEL/COATING boundary jump using actual
material-centroid gradients. But the Stage53 INTERNAL fuel face fluxes
still used Cartesian cell-centre spacing even when the true centroids
move tangentially in hex-clipped/hole-intersected cutcells.

For every existing Stage53 open interior face, construct:
  n: outward Cartesian face normal from the lower-index owner i to j
  d_n = (c_j-c_i)·n > 0       [m]
  d_t = (c_j-c_i)·t          [m]
  A: EXACT Stage53 unblocked fuel-face area (not a full Cartesian face)
  g = k_harmonic*A/d_n       [W/K]
  F_i->j = g * (T_i-T_j + d_t * (grad_T_avg·t))   [W]

The + sign is essential: for an affine field T=a*x+b*y, it yields
F_i->j = -k_harmonic*A*(grad_T·n) exactly, independently of centroid
skew and mesh resolution.

F is added to cell i and subtracted from cell j. Consequently,
the global internal conduction sum cancels exactly, and all Stage75
fuel/ZrC/H2 deposition and wall-energy ledgers remain untouched.

The tangential gradient is a DEFERRED correction derived from the
Stage80 real-fuel-centroid LS reconstruction. It is not a monotone
MPFA/MFD scheme, not a proof of energy-stable nonlinear iteration,
and not a validated curved-hole manufactured PDE solution.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fuel_cutcell_axial_transport import AxialCutCellModel
from .fuel_wall_nonorthogonal import (
    FuelGradientReconstruction,reconstruct_fuel_gradient,
)


@dataclass(frozen=True)
class InteriorFuelFaceCorrection:
    """Stage53 physical face geometry and Stage80 true-centroid skew."""
    face_cell_i: np.ndarray
    face_cell_j: np.ndarray
    physical_face_full_area_m2: np.ndarray
    normal_distance_m: np.ndarray
    tangential_offset_m: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    modified_full_length_geometry_m: np.ndarray
    reference_full_length_geometry_m: np.ndarray
    gradient: FuelGradientReconstruction

    @property
    def face_count(self) -> int:
        return len(self.face_cell_i)


def build_interior_fuel_face_correction(
    model:AxialCutCellModel,
    gradient:FuelGradientReconstruction,
) -> InteriorFuelFaceCorrection:
    """Exact open face area, centroid normal gap, and skew for every face."""
    base=model.transverse
    pair=np.asarray(base.lateral_face_pairs,dtype=int)
    nf=int(base.n_fuel)
    centers=np.asarray(gradient.centroids_cm,dtype=float)*.01
    if (pair.ndim!=2 or pair.shape[1]!=2 or
            centers.shape!=(nf,2) or
            not np.all(np.isfinite(centers))):
        raise ValueError("interior fuel face reconstruction geometry mismatch")
    nface=len(pair)
    if nface<1:
        raise ValueError("embedded fuel domain has no interior conduction faces")
    ids=np.asarray(base.cell_ids,dtype=int)
    if ids.ndim!=2:
        raise ValueError("missing fuel cell adjacency topology")
    original_locations=np.full((nf,2),-1,dtype=int)
    for iy,ix in zip(*np.where(ids>=0)):
        original_locations[int(ids[iy,ix])]=[iy,ix]
    if np.any(original_locations<0):
        raise ArithmeticError("not every fuel cutcell has a real Cartesian location")
    left=pair[:,0]
    right=pair[:,1]
    if np.any(left<0) or np.any(right<0) or np.any(left>=nf) or np.any(right>=nf):
        raise ValueError("invalid fuel face cell indices")
    # Cartesian connectivity determines each physical face NORMAL. The
    # actual centroids determine distance and nonorthogonal offset.
    delta_index=original_locations[right]-original_locations[left]
    vertical=(delta_index[:,0]==0)&(delta_index[:,1]==1)
    horizontal=(delta_index[:,0]==1)&(delta_index[:,1]==0)
    if np.any(~(vertical|horizontal)):
        raise ArithmeticError("fuel face is not a valid directed neighbour")
    nx=vertical.astype(float)
    ny=horizontal.astype(float)
    d=centers[right]-centers[left]
    normal_distance=d[:,0]*nx+d[:,1]*ny
    tangent_offset=-d[:,0]*ny+d[:,1]*nx
    if np.any(normal_distance<=0) or not np.all(np.isfinite(normal_distance)):
        raise ArithmeticError("nonpositive real-centroid interior normal gap")
    ref=np.asarray(base.fuel_conductivity_w_m_k,dtype=float)
    if ref.shape!=(nf,) or np.any(ref<=0):
        raise ValueError("reference fuel conductivity invalid")
    harmonic=2*ref[left]*ref[right]/(ref[left]+ref[right])
    original_geom=np.asarray(base.lateral_face_conductance_w_k,dtype=float)/harmonic
    if (original_geom.shape!=(nface,) or np.any(original_geom<=0) or
            not np.all(np.isfinite(original_geom))):
        raise ArithmeticError("Stage53 interior physical face geometry invalid")
    dx=float(np.diff(base.x_edges_cm)[0])*.01
    dy=float(np.diff(base.y_edges_cm)[0])*.01
    nominal=np.where(vertical,dx,dy)
    # Original geometry = A / nominal_gap. Recover the exact unblocked
    # physical fuel area and build new A / real-centroid normal gap.
    area=original_geom*nominal
    updated_geometry=area/normal_distance
    if np.any(area<=0) or np.any(updated_geometry<=0):
        raise ArithmeticError("invalid true open fuel face/centroid geometry")
    return InteriorFuelFaceCorrection(
        face_cell_i=left.copy(),face_cell_j=right.copy(),
        physical_face_full_area_m2=area,
        normal_distance_m=normal_distance,
        tangential_offset_m=tangent_offset,
        normal_x=nx,normal_y=ny,
        modified_full_length_geometry_m=updated_geometry,
        reference_full_length_geometry_m=original_geom,
        gradient=gradient,
    )


def nonorthogonal_face_heat_w(
    faces:InteriorFuelFaceCorrection,
    fuel_temperature_k:np.ndarray,
    conductivity_w_m_k:np.ndarray,
    axial_fraction:np.ndarray,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return signed [z,face] physical TOTAL conduction W and explicit skew W.

    First term is positive symmetric TPFA with the NORMAL true centroid
    gap. Second term is a conservative, explicitly recomputed gradient
    correction; it is NOT silently included in the frozen-k Jacobian.
    """
    t=np.asarray(fuel_temperature_k,dtype=float)
    k=np.asarray(conductivity_w_m_k,dtype=float)
    frac=np.asarray(axial_fraction,dtype=float)
    if (t.ndim!=2 or k.shape!=t or
            t.shape[1]!=faces.gradient.centroids_cm.shape[0] or
            frac.shape!=(t.shape[0],) or
            not all(np.all(np.isfinite(x)) for x in (t,k,frac)) or
            np.any(k<=0) or np.any(frac<=0)):
        raise ValueError("nonorthogonal fuel face requires positive valid T,k,axial fraction")
    i=faces.face_cell_i
    j=faces.face_cell_j
    kh=2*k[:,i]*k[:,j]/(k[:,i]+k[:,j])
    base=kh*faces.modified_full_length_geometry_m[None,:]*frac[:,None]
    gx,gy=reconstruct_fuel_gradient(faces.gradient,t)
    gt_i=-gx[:,i]*faces.normal_y[None,:]+gy[:,i]*faces.normal_x[None,:]
    gt_j=-gx[:,j]*faces.normal_y[None,:]+gy[:,j]*faces.normal_x[None,:]
    gt=.5*(gt_i+gt_j)
    correction=base*faces.tangential_offset_m[None,:]*gt
    signed=base*(t[:,i]-t[:,j])+correction
    if not all(np.all(np.isfinite(x)) for x in (base,correction,signed)):
        raise ArithmeticError("nonfinite conservative interior fuel flux")
    return signed,correction,base


def conservative_face_divergence_w(
    faces:InteriorFuelFaceCorrection,
    face_heat_w:np.ndarray,
    n_fuel:int,
) -> np.ndarray:
    """Equal/opposite face power, shape [z,fuel], global sum zero."""
    power=np.asarray(face_heat_w,dtype=float)
    if (power.ndim!=2 or power.shape[1]!=faces.face_count
            or n_fuel!=faces.gradient.centroids_cm.shape[0]
            or not np.all(np.isfinite(power))):
        raise ValueError("invalid face heat array for conservative divergence")
    out=np.zeros((power.shape[0],n_fuel),dtype=float)
    for z in range(power.shape[0]):
        np.add.at(out[z],faces.face_cell_i,power[z])
        np.add.at(out[z],faces.face_cell_j,-power[z])
    err=np.max(np.abs(np.sum(out,axis=1)))
    scale=max(1.,float(np.max(np.abs(power))),float(np.sum(np.abs(power))))
    if err>2e-12*scale:
        raise ArithmeticError("interior face energy diverged globally")
    return out
