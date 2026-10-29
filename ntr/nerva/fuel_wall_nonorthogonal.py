"""Stage80: conservative deferred nonorthogonal fuel/curved-wall heat flux.

Stage79 established the TRUE fuel-volume centroid and the curved contact
arc centroid/normal. The two-point flux approximation omitted temperature
variation along the tangent between these points. For a locally LINEAR
temperature field, subtract g_t * d_t from the fuel-to-wall temperature
jump before multiplying by the positive normal thermal conductance.

The fuel gradient is reconstructed with a weighted 2D least-squares
stencil over actual material centroids, NOT rectangular cell centres.
Reconstruction is linearly exact in the fuel-only material domain when
neighbours span both spatial dimensions. Every heat flux is posted with
opposite sign to fuel and ZrC: no heat is created at the contact.

This is an opt-in deferred correction, not a full nonorthogonal
multi-point flux approximation. Strongly anisotropic fuel conduction,
sliver conditioning, M-matrix/maximum-principle guarantees and independent
curved manufactured steady-state heat-solution convergence remain open.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .fuel_cutcell_conduction_2d import _clip_polygon
from .fuel_embedded_boundary_moments import fuel_cutcell_volume_centroid_cm
from .layout import coolant_channel_positions
from .config import NervaConfig
from .fuel_cutcell_axial_transport import AxialCutCellModel


@dataclass(frozen=True)
class FuelGradientReconstruction:
    """Area/centroid-consistent [n_fuel,n_fuel] gradient operators in 1/m."""
    centroids_cm: np.ndarray
    gradient_x_m_inv: csr_matrix
    gradient_y_m_inv: csr_matrix
    minimum_neighbours: int
    maximum_gradient_condition: float
    exact_material_area_residual_cm2: float


def build_fuel_gradient_reconstruction(
    config:NervaConfig,
    solid:AxialCutCellModel,
    *,
    radius_in_cells:int=3,
    maximum_condition:float=1e9,
) -> FuelGradientReconstruction:
    """Construct exact-centroid local 2D LS gradients from real cutcell IDs.

    Weighted least squares (Tj-Ti) = grad_i · (xj-xi), with weights
    1/|xj-xi|². The sparse coefficients have units 1/m. Stencils that
    cannot support a unique 2D linear gradient FAIL CLOSED.
    """
    if (not isinstance(radius_in_cells,int)
            or not 2<=radius_in_cells<=6
            or not math.isfinite(maximum_condition)
            or maximum_condition<=1.):
        raise ValueError("fuel LS stencil radius/condition guard invalid")
    fuel=solid.transverse
    ids=np.asarray(fuel.cell_ids,dtype=int)
    nf=int(fuel.n_fuel)
    if ids.ndim!=2 or nf<4:
        raise ValueError("missing real transverse fuel cutcell topology")
    ny,nx=ids.shape
    xx=np.asarray(fuel.x_edges_cm,dtype=float)
    yy=np.asarray(fuel.y_edges_cm,dtype=float)
    if len(xx)!=nx+1 or len(yy)!=ny+1:
        raise ValueError("fuel cutcell edges do not match IDs")
    coords=np.full((nf,2),np.nan)
    ap=config.inner_fuel_flat_to_flat_cm/2.
    radius=config.coated_channel_radius_cm
    holes=coolant_channel_positions(config)
    max_area_error=0.
    for iy,ix in zip(*np.where(ids>=0)):
        i=int(ids[iy,ix])
        polygon=_clip_polygon([
            (float(xx[ix]),float(yy[iy])),
            (float(xx[ix+1]),float(yy[iy])),
            (float(xx[ix+1]),float(yy[iy+1])),
            (float(xx[ix]),float(yy[iy+1]))],ap)
        ref=float(fuel.fuel_areas_cm2[i])
        area,cx,cy=fuel_cutcell_volume_centroid_cm(
            polygon,holes,radius,exact_fuel_area_cm2=ref)
        max_area_error=max(max_area_error,abs(area-ref))
        coords[i,:]=cx,cy
    if not np.all(np.isfinite(coords)):
        raise ArithmeticError("fuel LS reconstruction missing a real material centroid")
    rows=[]; cols=[]; data_x=[]; data_y=[]
    min_n=100000
    max_cond=0.
    for iy,ix in zip(*np.where(ids>=0)):
        i=int(ids[iy,ix])
        lower_y=max(0,iy-radius_in_cells)
        upper_y=min(ny,iy+radius_in_cells+1)
        lower_x=max(0,ix-radius_in_cells)
        upper_x=min(nx,ix+radius_in_cells+1)
        neigh=ids[lower_y:upper_y,lower_x:upper_x].ravel()
        neigh=np.unique(neigh[(neigh>=0)&(neigh!=i)])
        if len(neigh)<2:
            raise ArithmeticError("physical fuel LS cell has insufficient neighbours")
        d=(coords[neigh]-coords[i])*0.01
        norm2=np.einsum("ij,ij->i",d,d)
        keep=norm2>1e-22
        neigh=neigh[keep]
        d=d[keep]
        norm2=norm2[keep]
        if len(neigh)<2:
            raise ArithmeticError("fuel LS real-centroid neighbours coincide")
        w=1./norm2
        gram=d.T @ (d*w[:,None])
        cond=float(np.linalg.cond(gram))
        if not math.isfinite(cond) or cond>maximum_condition:
            raise ArithmeticError("physical fuel LS stencil cannot resolve two gradients")
        projection=np.linalg.solve(gram,(d*w[:,None]).T)
        min_n=min(min_n,len(neigh))
        max_cond=max(max_cond,cond)
        # Each row coefficient belongs to neighbour j, and the centre
        # row coefficient is negative SUM_j (so constants cancel).
        all_cols=np.concatenate((neigh,[i]))
        px=np.concatenate((projection[0],[-np.sum(projection[0])]))
        py=np.concatenate((projection[1],[-np.sum(projection[1])]))
        rows.extend([i]*len(all_cols))
        cols.extend(all_cols.tolist())
        data_x.extend(px.tolist())
        data_y.extend(py.tolist())
    shape=(nf,nf)
    gx=coo_matrix((data_x,(rows,cols)),shape=shape).tocsr()
    gy=coo_matrix((data_y,(rows,cols)),shape=shape).tocsr()
    if (not np.all(np.isfinite(gx.data))
            or not np.all(np.isfinite(gy.data))):
        raise ArithmeticError("nonfinite embedded fuel gradient reconstruction")
    # Verify exact reproduction of constant and linear (x,y) fields in
    # REAL physical centroid positions, at setup time.
    one=np.ones(nf)
    xm=coords[:,0]*.01
    ym=coords[:,1]*.01
    scale=max(1.,float(np.max(np.abs(xm))),float(np.max(np.abs(ym))))
    check=[
        float(np.max(np.abs(gx@one))),float(np.max(np.abs(gy@one))),
        float(np.max(np.abs(gx@xm-1.))),float(np.max(np.abs(gy@ym-1.))),
        float(np.max(np.abs(gx@ym))),float(np.max(np.abs(gy@xm))),
    ]
    if max(check)>2e-8/scale:
        raise ArithmeticError("fuel LS operator fails linear-gradient reproduction")
    return FuelGradientReconstruction(
        centroids_cm=coords,
        gradient_x_m_inv=gx,gradient_y_m_inv=gy,
        minimum_neighbours=min_n,
        maximum_gradient_condition=max_cond,
        exact_material_area_residual_cm2=max_area_error,
    )


def reconstruct_fuel_gradient(
    reconstruction:FuelGradientReconstruction,
    fuel_temperature_k:np.ndarray,
) -> tuple[np.ndarray,np.ndarray]:
    """Return actual 2D fuel temperature gradients [K/m] for each z-zone."""
    temp=np.asarray(fuel_temperature_k,dtype=float)
    n=reconstruction.centroids_cm.shape[0]
    if (temp.ndim!=2 or temp.shape[1]!=n
            or not np.all(np.isfinite(temp))):
        raise ValueError("fuel gradient state must be finite [z,material-cutcell]")
    gx=(reconstruction.gradient_x_m_inv@temp.T).T
    gy=(reconstruction.gradient_y_m_inv@temp.T).T
    if not np.all(np.isfinite(gx)) or not np.all(np.isfinite(gy)):
        raise ArithmeticError("nonfinite nonorthogonal fuel gradient")
    return np.asarray(gx),np.asarray(gy)


def tangential_wall_temperature_correction_k(
    gx:np.ndarray,gy:np.ndarray,
    cell_indices:np.ndarray,
    tangent_m:np.ndarray,
    normal_x:np.ndarray,normal_y:np.ndarray,
) -> np.ndarray:
    """Exact tangential ΔT correction for affine T and a curved arc.

    With wall-to-centroid displacement = d_n*n + d_t*t:
    T_cell - T_wall = d_n*(grad·n) + d_t*(grad·t).
    The corrected normal jump subtracts d_t*(grad·t), with the signed
    d_t from Stage79 physical fuel and curved contact first moments.
    """
    if (gx.shape!=gy.shape or gx.ndim!=2
            or tangent_m.shape!=cell_indices.shape
            or tangent_m.shape!=normal_x.shape
            or tangent_m.shape!=normal_y.shape):
        raise ValueError("fuel nonorthogonal correction shapes mismatch")
    if (np.any(cell_indices<0) or
            np.any(cell_indices>=gx.shape[1])
            or not all(np.all(np.isfinite(x))
                       for x in (gx,gy,tangent_m,normal_x,normal_y))):
        raise ValueError("nonphysical fuel nonorthogonal correction inputs")
    n=np.hypot(normal_x,normal_y)
    if np.any(n<=1e-12):
        raise ValueError("zero average curved fuel-wall normal")
    ux=normal_x/n
    uy=normal_y/n
    tangent_grad=-gx[:,cell_indices]*uy[None,:]+gy[:,cell_indices]*ux[None,:]
    return tangent_grad*tangent_m[None,:]
