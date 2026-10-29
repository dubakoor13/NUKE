"""Stage61 geometrically conservative OpenMC 3D fuel mesh -> thermal fuel cut cells.

Numerical fidelity change from Stage59/60: **do not sum entire-reactor fuel
tallies onto a single fuel element**. The source mesh XY bounds must match
the specific resolved fuel hex's bounding rectangle. Heat for each native
source voxel is assumed uniform per unit physical fuel-only overlap volume
*inside that voxel*, rather than uniformly over the whole element.

Intersection areas are analytic: axis-aligned rectangle intersections,
true inner hex clipping, and exact circular coating holes (Stage53 kernel).
No discretized circle polygons, centre-point sampling or fitted hot-power
distribution is introduced. All source voxel weights must partition fuel
volume and conserve the original energy, not merely yield a near-zero sum.

The native material filter does NOT identify fuel element instances. A
matching bounding box plus externally justified isolated-element identity
is necessary; caller must separately establish this run geometry. This
module does not read OpenMC geometry files or establish historically
physical fuel loading, power normalization or neutron source convergence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .config import NervaConfig
from .layout import coolant_channel_positions
from .fuel_cutcell_conduction_2d import (
    _clip_polygon, _area_polygon, _disk_polygon_intersection_area,
)
from .fuel_cutcell_axial_transport import AxialCutCellModel
from .openmc_fuel_axial_heat_handoff import (
    EV_TO_J, ResolvedFuelHeatingInput, build_axial_overlap_m,
    check_heating_for_coupled_core,
)
from .openmc_native_fuel_statepoint import NativeFuelHeatingTally


@dataclass(frozen=True)
class NativeSpatialFuelHeating:
    """True source-voxel XY pattern retained under bounded local spreading.

    source_xy_to_fuel_weights is a sparse [fuel-cut-cell, source-XY-bin]
    matrix. Each nonempty source column sums to unity (conservation).
    source coverage is restricted to ONE physical fuel-element footprint.
    """
    resolved: ResolvedFuelHeatingInput
    source_xy_to_fuel_weights: csr_matrix
    source_xy_fuel_area_cm2: np.ndarray
    fuel_cell_uncertainty_bound_w: np.ndarray
    axial_uncertainty_bound_w: np.ndarray
    total_uncertainty_bound_w: float
    local_heating_approximation: str = (
        "each source 3D voxel uniform power density ONLY over exact fuel "
        "within that voxel; source voxel internal spatial shape unknown"
    )
    provenance_requirement: str = (
        "caller must independently prove selected native fuel tally mesh "
        "contains exactly one intended physical fuel-element instance"
    )


def _fuel_only_rectangle_area(
    x0: float, x1: float, y0: float, y1: float,
    ap: float, holes: tuple[tuple[float,float],...], radius: float,
) -> float:
    if x1<=x0 or y1<=y0:
        return 0.
    poly=_clip_polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)],ap)
    if len(poly)<3:
        return 0.
    total=_area_polygon(poly)
    if total<=0:
        return 0.
    removed=0.
    for hx,hy in holes:
        if (hx+radius<x0 or hx-radius>x1 or
                hy+radius<y0 or hy-radius>y1):
            continue
        removed+=max(0.,_disk_polygon_intersection_area(
            poly,hx,hy,radius))
    area=total-removed
    if area< -2e-10:
        raise ArithmeticError("source overlap has negative exact fuel area")
    return max(0.,area)


def _check_geometry(
    config: NervaConfig, native: NativeFuelHeatingTally,
    solid: AxialCutCellModel,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,tuple[tuple[float,float],...]]:
    config.validate()
    nx,ny,nz=native.mesh_dimensions
    src_low=np.asarray(native.source_mesh_origin_cm,dtype=float)
    src_high=np.asarray(native.source_mesh_upper_cm,dtype=float)
    targets=solid.transverse
    if (src_low.shape!=(3,) or src_high.shape!=(3,) or
            not np.all(np.isfinite(src_low)) or
            not np.all(np.isfinite(src_high))):
        raise ValueError("native OpenMC source has invalid mesh geometry")
    # This is the essential scope gate. The built-in NERVA reactor mesh
    # deliberately covers the entire reactor, and MUST fail here.
    expected_xy=np.array([
        targets.x_edges_cm[0],targets.y_edges_cm[0],
        targets.x_edges_cm[-1],targets.y_edges_cm[-1]])
    native_xy=np.array([src_low[0],src_low[1],
                        src_high[0],src_high[1]])
    if not np.allclose(native_xy,expected_xy,rtol=0.,atol=2e-10):
        raise ValueError(
            "whole-core or displaced OpenMC mesh cannot be mapped into "
            "one physical NERVA fuel element; require isolated-element tally")
    if abs(float(src_high[2]-src_low[2]) -
           100*float(solid.axial_edges_m[-1]))>2e-9:
        raise ValueError("native source fuel length differs from solid fuel")
    if (not np.all(np.isfinite(targets.fuel_areas_cm2)) or
            np.any(targets.fuel_areas_cm2<=0)
            or targets.cell_ids.shape!=(
                len(targets.y_edges_cm)-1,len(targets.x_edges_cm)-1)):
        raise ValueError("target cut-cell layout or physical fuel area invalid")
    holes=tuple((float(x),float(y)) for x,y in
                coolant_channel_positions(config))
    if len(holes)!=19:
        raise ValueError("source physical hex requires 19 coated holes")
    sx=np.linspace(src_low[0],src_high[0],nx+1)
    sy=np.linspace(src_low[1],src_high[1],ny+1)
    sz=(np.linspace(src_low[2],src_high[2],nz+1)-src_low[2])*.01
    return sx,sy,sz,holes


def _geometric_source_weights(
    config: NervaConfig,
    native: NativeFuelHeatingTally,
    solid: AxialCutCellModel,
) -> tuple[csr_matrix,np.ndarray,np.ndarray]:
    sx,sy,sz,holes=_check_geometry(config,native,solid)
    target=solid.transverse
    ap=config.inner_fuel_flat_to_flat_cm/2
    radius=config.coated_channel_radius_cm
    src_nx=len(sx)-1
    src_ny=len(sy)-1
    ns=src_nx*src_ny
    nf=target.n_fuel
    source_area=np.zeros(ns)
    row=[]
    col=[]
    val=[]
    weighted_target=np.zeros(nf)
    for a in range(src_nx):
        for b in range(src_ny):
            srci=a*src_ny+b
            xa,xb=float(sx[a]),float(sx[a+1])
            ya,yb=float(sy[b]),float(sy[b+1])
            direct=_fuel_only_rectangle_area(
                xa,xb,ya,yb,ap,holes,radius)
            source_area[srci]=direct
            if direct<=1e-14:
                continue
            # Only target rectangles overlapping this source rectangle
            # need precise hex-and-hole clipping. Avoid O(Nsrc*Ntarget).
            ia=max(0,np.searchsorted(target.x_edges_cm,xa,side="right")-1)
            ib=min(len(target.x_edges_cm)-2,
                   np.searchsorted(target.x_edges_cm,xb,side="left"))
            ja=max(0,np.searchsorted(target.y_edges_cm,ya,side="right")-1)
            jb=min(len(target.y_edges_cm)-2,
                   np.searchsorted(target.y_edges_cm,yb,side="left"))
            accumulated=0.
            for j in range(ja,jb+1):
                y0=max(ya,float(target.y_edges_cm[j]))
                y1=min(yb,float(target.y_edges_cm[j+1]))
                if y1<=y0:
                    continue
                for i in range(ia,ib+1):
                    fuel_id=int(target.cell_ids[j,i])
                    if fuel_id<0:
                        continue
                    x0=max(xa,float(target.x_edges_cm[i]))
                    x1=min(xb,float(target.x_edges_cm[i+1]))
                    if x1<=x0:
                        continue
                    intersect=_fuel_only_rectangle_area(
                        x0,x1,y0,y1,ap,holes,radius)
                    if intersect<=1e-14:
                        continue
                    row.append(fuel_id)
                    col.append(srci)
                    val.append(intersect/direct)
                    weighted_target[fuel_id]+=intersect
                    accumulated+=intersect
            if abs(accumulated-direct)>max(2e-10,2e-10*direct):
                raise ArithmeticError(
                    "source XY fuel area not partitioned by thermal cut cells")
    if not np.allclose(
        weighted_target,target.fuel_areas_cm2,
        rtol=2e-10,atol=2e-10):
        raise ArithmeticError(
            "source mesh misses fuel or duplicates thermal cut-cell geometry")
    weights=coo_matrix(
        (val,(row,col)),shape=(nf,ns)).tocsr()
    weights.sum_duplicates()
    sums=np.asarray(weights.sum(axis=0)).ravel()
    mask=source_area>1e-14
    if (not np.allclose(sums[mask],1,rtol=2e-10,atol=2e-10)
            or not np.allclose(sums[~mask],0,atol=2e-10)):
        raise ArithmeticError("XY fuel source weights not conservative")
    return weights,source_area,sz


def project_native_3d_fuel_heating(
    config: NervaConfig,
    native: NativeFuelHeatingTally,
    model: AxialCutCellModel,
    *,
    isolated_element_geometry_confirmed: bool,
) -> NativeSpatialFuelHeating:
    """Map native x,y,z fuel material tally without collapsing XY profiles.

    NOTE: `isolated_element_geometry_confirmed` is an externally asserted
    constraint, not cryptographic proof. Do not use for full-core tally.
    """
    if isolated_element_geometry_confirmed is not True:
        raise ValueError(
            "isolated single-fuel-element geometry must be independently confirmed")
    weights,source_area,z=_geometric_source_weights(config,native,model)
    nx,ny,nz=native.mesh_dimensions
    mean=np.asarray(native.source_mesh_mean_ev_per_source,dtype=float)
    stderr=np.asarray(native.source_mesh_stderr_ev_per_source,dtype=float)
    if (mean.shape!=(nx,ny,nz) or stderr.shape!=(nx,ny,nz)
            or not np.all(np.isfinite(mean)) or
            not np.all(np.isfinite(stderr)) or
            np.any(mean<0) or np.any(stderr<0)):
        raise ValueError("native 3D mean or marginal error arrays invalid")
    source_scores=mean.reshape((nx*ny,nz))
    source_errors=stderr.reshape((nx*ny,nz))
    empty=source_area<=1e-14
    if (np.any(source_scores[empty]>1e-12)
            or np.any(source_errors[empty]>1e-12)):
        raise ValueError("nonzero fuel heating occurs in mesh bin with no fuel")
    native_axial=np.asarray(native.score.fuel_neutron_ev_per_source)
    if (native_axial.shape!=(nz,) or not np.allclose(
            source_scores.sum(axis=0),native_axial,
            atol=1e-10,rtol=1e-12)):
        raise ValueError("native score and mesh voxel means disagree")
    overlap=build_axial_overlap_m(z,model.axial_edges_m)
    fz=overlap/np.diff(z)[None,:]
    joule_rate=native.score.source_rate_particles_per_s*EV_TO_J
    # First preserve the source x/y power profile under true material
    # intersections, then remap z independently using true slab overlaps.
    watts=(weights @ (source_scores @ fz.T)).T*joule_rate
    bounds=(weights @ (source_errors @ fz.T)).T*joule_rate
    original_axis=source_scores.sum(axis=0)*joule_rate
    mapped_axis=watts.sum(axis=1)
    total=float(np.sum(original_axis))
    err=float(np.sum(watts)-total)
    if abs(err)>max(2e-12,2e-11*total):
        raise ArithmeticError("3D native fuel heat mapping violates power balance")
    if not np.all(np.isfinite(watts)) or np.any(watts<-1e-12):
        raise ArithmeticError("invalid fuel heat transfer from 3D source")
    fraction=np.asarray(model.transverse.fuel_areas_cm2,dtype=float)
    fraction=fraction/fraction.sum()
    resolved=ResolvedFuelHeatingInput(
        fuel_deposition_w=watts,
        source_axial_bin_w=original_axis,
        mapped_axial_bin_w=mapped_axis,
        axial_overlap_m=overlap,
        target_axial_edges_m=model.axial_edges_m.copy(),
        target_fuel_areas_cm2=model.transverse.fuel_areas_cm2.copy(),
        transverse_fuel_weights=fraction,
        fuel_neutron_power_w=total,
        fuel_photon_power_w=0.,
        unresolved_nonfuel_power_w=0.,
        fuel_power_balance_residual_w=err,
        statepoint_sha256=native.statepoint_sha256,
        nuclear_data_sha256=native.score.nuclear_data_sha256,
        source_rate_basis=native.score.source_rate_basis,
        transverse_approximation=(
            "original native 3D source XY distribution retained; "
            "uniform physical fuel heating per original source voxel"),
    )
    return NativeSpatialFuelHeating(
        resolved=resolved,
        source_xy_to_fuel_weights=weights,
        source_xy_fuel_area_cm2=source_area.reshape((nx,ny)),
        fuel_cell_uncertainty_bound_w=bounds,
        axial_uncertainty_bound_w=np.sum(bounds,axis=1),
        total_uncertainty_bound_w=float(np.sum(bounds)),
    )
