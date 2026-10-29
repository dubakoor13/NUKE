"""Stage73 geometry-exact fuel/ZrC/H2 angular contact and power projection.

This is the missing physical CONTACT operator for Stage72's separate
axial/azimuthal ZrC finite volumes. The fuel boundary is the actual
Stage53 *outer coated-cylinder circumference*, split by Cartesian
fuel-cutcell edges, six hex flats and angular sector boundaries.
The H2 boundary is the separately clipped *inner bore circumference*.
Each arc segment is owned by exactly one physical fuel cell, coating
angular sector or H2 channel. No adjacent-channel thermal shortcut.

Cross-interface watts use the SAME signed power on both sides, so source
allocation is conservative. These interfaces alone do NOT activate the
Stage72 angular state inside the Stage58 coupled time integrator; that
requires an additional time-marching integration stage and its first-law
tests. Source rates/material properties used in tests remain synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .fuel_cutcell_conduction_2d import _NORMALS, _inside_hex, _clip_polygon
from .fuel_embedded_boundary_moments import (
    fuel_cutcell_volume_centroid_cm,
    circular_arc_centroid_and_normal_cm,
    fuel_wall_moment_normal_depth_m,
)
from .fuel_cutcell_axial_transport import AxialCutCellModel
from .fuel_cutcell_nonlinear_enthalpy import NonlinearThermalProperties
from .layout import coolant_channel_positions
from .config import NervaConfig
from .resolved_coating_thermal import ResolvedCoatingModel
from .coating_angular_fvm import AngularCoatingMesh


@dataclass(frozen=True)
class AngularCoatingInterfaces:
    """Physically partitioned surface areas across [channel,angle] sectors.

    fuel_outer_full_m2 [nf,19*n_theta] is TOTAL active-length area.
    h2_inner_full_m2 [19,n_theta] is total active-length inner area.
    During each axial step multiply both by dz / active_length.
    """
    fuel_outer_full_m2: csr_matrix
    h2_inner_full_m2: np.ndarray
    sector_count: int
    fuel_arc_residual_m2: float
    hydrogen_arc_residual_m2: float
    # Area-weighted local half-depth, built from exact Stage53 cutcell
    # footprint and circular contact ARC normal. Optional to preserve
    # older construction; selected explicitly by the Stage78 model.
    fuel_wall_half_depth_m: csr_matrix | None = None
    fuel_wall_moment_depth_m: csr_matrix | None = None
    fuel_wall_tangential_offset_m: csr_matrix | None = None
    fuel_wall_signed_normal_depth_m: csr_matrix | None = None
    nonorthogonal_wall_fallback_count: int = 0


@dataclass(frozen=True)
class AngularInterfacePower:
    """One signed fuel->coating flux, and one coating->gas flux, in watts."""
    fuel_to_coating_by_fuel_w: np.ndarray
    fuel_to_coating_by_sector_w: np.ndarray
    coating_to_hydrogen_by_sector_w: np.ndarray
    coating_to_hydrogen_by_channel_w: np.ndarray
    equal_opposite_fuel_interface_residual_w: float
    equal_opposite_hydrogen_interface_residual_w: float


def _angle_events(cx:float,cy:float,r:float,
                  x_edges:np.ndarray|None,y_edges:np.ndarray|None,
                  ap:float, angular_edges:np.ndarray) -> list[float]:
    events=list(float(v) for v in angular_edges)
    def add(t):
        v=float(t%(2*math.pi))
        if 1e-13<v<2*math.pi-1e-13:
            events.append(v)
    if x_edges is not None:
        for edge in x_edges:
            v=(float(edge)-cx)/r
            if -1.<v<1.:
                a=math.acos(v)
                add(a)
                add(-a)
    if y_edges is not None:
        for edge in y_edges:
            v=(float(edge)-cy)/r
            if -1.<v<1.:
                a=math.asin(v)
                add(a)
                add(math.pi-a)
    for nx,ny in _NORMALS:
        v=(ap-nx*cx-ny*cy)/r
        if -1.<v<1.:
            a=math.atan2(ny,nx)
            d=math.acos(v)
            add(a+d)
            add(a-d)
    events.sort()
    result=[events[0]]
    for v in events[1:]:
        if v-result[-1]>2e-13:
            result.append(v)
    return result


def local_fuel_wall_half_depth_m(
    clipped_cell_polygon_cm: list[tuple[float,float]],
    wall_point_cm: tuple[float,float],
    outward_from_bore: tuple[float,float],
) -> float:
    """Positive local half-depth from curved wall into available fuel.

    For a planar boundary orthogonal to n and a clipped rectangular fuel
    cell extending h outward, returns h/2 EXACTLY (the 1D cell-centroid
    distance). For oblique/curved cutcells, approximates the depth as half
    the largest projected positive extent of the physical hex-clipped
    polygon. It is NOT an exact fuel-volume centroid; convergence against
    a manufactured curved-boundary solution is still required.
    """
    if len(clipped_cell_polygon_cm)<3:
        raise ValueError("fuel wall needs a physical clipped cell polygon")
    nx,ny=(float(outward_from_bore[0]),float(outward_from_bore[1]))
    norm=math.hypot(nx,ny)
    if abs(norm-1.)>1e-10:
        raise ValueError("fuel-wall outward normal must have unit magnitude")
    px,py=wall_point_cm
    distances=[
        nx*(vx-px)+ny*(vy-py)
        for vx,vy in clipped_cell_polygon_cm
    ]
    width=max(distances)
    if not math.isfinite(width) or width<=1e-13:
        raise ArithmeticError("no physical fuel half-depth across curved wall")
    return .5*width*.01


def build_angular_coating_interfaces(
    config:NervaConfig,
    solid:AxialCutCellModel,
    coating:ResolvedCoatingModel,
    angular:AngularCoatingMesh,
) -> AngularCoatingInterfaces:
    """Exact arc splitting and independent Stage53/Stage70 total-area checks."""
    angular_edges=np.asarray(angular.angular_edges_rad,dtype=float)
    n=len(angular_edges)-1
    if (n<8 or n>96 or angular.sector_area_m2.shape!=(19,n)
            or angular_edges.shape!=(n+1,) or
            not np.all(np.diff(angular_edges)>0)
            or not np.isclose(angular_edges[0],0.,atol=1e-13)
            or not np.isclose(angular_edges[-1],2*math.pi,atol=1e-13)):
        raise ValueError("angular coating mesh has invalid periodic sector edges")
    fuel=solid.transverse
    xedge=fuel.x_edges_cm
    yedge=fuel.y_edges_cm
    ids=fuel.cell_ids
    ap=config.inner_fuel_flat_to_flat_cm/2.
    r_outer=config.coated_channel_radius_cm
    r_bore=config.channel_radius_cm
    axial_length=float(solid.axial_edges_m[-1]-solid.axial_edges_m[0])
    if axial_length<=0:
        raise ValueError("fuel element axial length must be positive")
    coords=coolant_channel_positions(config)
    rows=[]
    cols=[]
    areas=[]
    depth_area_m3=[]
    moment_depth_area_m3=[]
    signed_normal_area_m3=[]
    tangent_offset_area_m3=[]
    fallback_count=0
    # Per-cell total physical cylindrical contact length gives a
    # conservative positive effective control-volume depth when a
    # curved face is nonorthogonal to its whole-cell material centroid.
    total_fuel_contact_m2=np.asarray(
        fuel.fuel_zrc_contact_m2.sum(axis=1)).ravel()
    polygon_cache={}
    centroid_cache={}
    inner=np.zeros((19,n),dtype=float)
    for channel,(cx,cy) in enumerate(coords):
        # Outer coating/fuel arcs require a unique physical cutcell owner.
        cuts=_angle_events(
            cx,cy,r_outer,xedge,yedge,ap,angular_edges)
        for left,right in zip(cuts,cuts[1:]):
            if right-left<1e-13:
                continue
            theta=(left+right)/2.
            x=cx+r_outer*math.cos(theta)
            y=cy+r_outer*math.sin(theta)
            if not _inside_hex(x,y,ap):
                continue
            ix=int(np.searchsorted(xedge,x,side="right")-1)
            iy=int(np.searchsorted(yedge,y,side="right")-1)
            if not (0<=ix<len(xedge)-1 and 0<=iy<len(yedge)-1):
                raise ArithmeticError("outer coating arc beyond fuel mesh")
            fuel_cell=int(ids[iy,ix])
            if fuel_cell<0:
                raise ArithmeticError("coating physical outer arc without fuel cutcell")
            angular_cell=int(np.searchsorted(
                angular_edges,theta,side="right")-1)
            if not 0<=angular_cell<n:
                raise ArithmeticError("outer arc angular bin invalid")
            rows.append(fuel_cell)
            cols.append(channel*n+angular_cell)
            patch_area=r_outer*(right-left)*.01*axial_length
            areas.append(patch_area)
            # Account for the true locally available fuel depth, not
            # the single global min(dx,dy)/2 for every curved contact.
            poly=polygon_cache.get(fuel_cell)
            if poly is None:
                x0,x1=float(xedge[ix]),float(xedge[ix+1])
                y0,y1=float(yedge[iy]),float(yedge[iy+1])
                poly=_clip_polygon(
                    [(x0,y0),(x1,y0),(x1,y1),(x0,y1)],ap)
                polygon_cache[fuel_cell]=poly
            arc_normal=(math.cos(theta),math.sin(theta))
            depth=local_fuel_wall_half_depth_m(poly,(x,y),arc_normal)
            depth_area_m3.append(patch_area*depth)
            material_centroid=centroid_cache.get(fuel_cell)
            if material_centroid is None:
                _,xx,yy=fuel_cutcell_volume_centroid_cm(
                    poly,coords,r_outer,
                    exact_fuel_area_cm2=float(fuel.fuel_areas_cm2[fuel_cell]))
                material_centroid=(xx,yy)
                centroid_cache[fuel_cell]=material_centroid
            physical_arc=circular_arc_centroid_and_normal_cm(
                (cx,cy),r_outer,float(left),float(right))
            signed_normal,tangent=fuel_wall_moment_normal_depth_m(
                material_centroid,physical_arc,
                allow_nonorthogonal_negative=True)
            signed_normal_area_m3.append(patch_area*signed_normal)
            if signed_normal>0.:
                moment_depth=signed_normal
            else:
                # A centroid lying behind this local tangent plane
                # has NO valid normal two-point path. The exact fuel
                # control-volume / total true cylindrical interface
                # area provides a separate positive first-order
                # volume-to-wall half-depth. It is explicitly counted,
                # never masqueraded as a true centroid projection.
                wall_area=float(total_fuel_contact_m2[fuel_cell])
                if wall_area<=0.:
                    raise ArithmeticError("no physical cutcell contact area for fallback")
                volume_cm2=float(fuel.fuel_areas_cm2[fuel_cell])
                moment_depth=.5*volume_cm2*1e-4*axial_length/wall_area
                if not math.isfinite(moment_depth) or moment_depth<=0.:
                    raise ArithmeticError("nonphysical fuel volume/contact fallback")
                fallback_count+=1
            moment_depth_area_m3.append(patch_area*moment_depth)
            tangent_offset_area_m3.append(patch_area*tangent)
        # The inner bore/H2 area is separately clipped to physical hex.
        cuts=_angle_events(
            cx,cy,r_bore,None,None,ap,angular_edges)
        for left,right in zip(cuts,cuts[1:]):
            theta=(left+right)/2.
            x=cx+r_bore*math.cos(theta)
            y=cy+r_bore*math.sin(theta)
            if not _inside_hex(x,y,ap):
                continue
            bin_index=int(np.searchsorted(
                angular_edges,theta,side="right")-1)
            inner[channel,bin_index]+=r_bore*(right-left)*.01*axial_length
    outer=coo_matrix(
        (areas,(rows,cols)),
        shape=(fuel.n_fuel,19*n)).tocsr()
    weighted=coo_matrix(
        (depth_area_m3,(rows,cols)),shape=outer.shape).tocsr()
    if (not np.array_equal(weighted.indptr,outer.indptr)
            or not np.array_equal(weighted.indices,outer.indices)):
        raise ArithmeticError("local half-depth contact topology mismatch")
    depths=weighted.data/outer.data
    if not np.all(np.isfinite(depths)) or np.any(depths<=0):
        raise ArithmeticError("nonphysical embedded fuel contact half-depth")
    local_depth=csr_matrix(
        (depths,outer.indices.copy(),outer.indptr.copy()),shape=outer.shape)
    def aggregate_moments(values, *, positive):
        moment=coo_matrix(
            (values,(rows,cols)),shape=outer.shape).tocsr()
        if (not np.array_equal(moment.indptr,outer.indptr)
                or not np.array_equal(moment.indices,outer.indices)):
            raise ArithmeticError("curved fuel moment/contact topology mismatch")
        aggregate=moment.data/outer.data
        if not np.all(np.isfinite(aggregate)) or (
                positive and np.any(aggregate<=0)):
            raise ArithmeticError("invalid physical normal fuel-wall centroid distance")
        return csr_matrix(
            (aggregate,outer.indices.copy(),outer.indptr.copy()),
            shape=outer.shape)
    moment_depth=aggregate_moments(moment_depth_area_m3,positive=True)
    signed_normal=aggregate_moments(
        signed_normal_area_m3,positive=False)
    tangent_offset=aggregate_moments(tangent_offset_area_m3,positive=False)
    reference=coating.fuel_arc_m2
    if reference.shape!=(fuel.n_fuel,19):
        raise ValueError("Stage70 coating contact reference has wrong shape")
    compressed=outer.toarray().reshape((fuel.n_fuel,19,n)).sum(axis=2)
    fuel_res=float(np.max(np.abs(compressed-reference.toarray())))
    h2_res=float(np.max(np.abs(
        inner.sum(axis=1)-coating.inner_bore_area_m2.sum(axis=0))))
    if fuel_res>4e-11 or h2_res>4e-11:
        raise ArithmeticError(
            f"angular coating arc split loses actual fuel/H2 contact: "
            f"fuel {fuel_res}, coolant {h2_res} m2")
    if np.any(outer.data<0) or np.any(inner<0):
        raise ArithmeticError("negative physical coating boundary area")
    if np.any(
            inner[~angular.sector_active]>1e-13):
        raise ArithmeticError("inner H2 bore arc without coating sector mass")
    return AngularCoatingInterfaces(
        fuel_outer_full_m2=outer,
        h2_inner_full_m2=inner,
        sector_count=n,
        fuel_arc_residual_m2=fuel_res,
        hydrogen_arc_residual_m2=h2_res,
        fuel_wall_half_depth_m=local_depth,
        fuel_wall_moment_depth_m=moment_depth,
        fuel_wall_tangential_offset_m=tangent_offset,
        fuel_wall_signed_normal_depth_m=signed_normal,
        nonorthogonal_wall_fallback_count=fallback_count,
    )


def angular_interface_power(
    coating:ResolvedCoatingModel,
    model:AxialCutCellModel,
    laws:NonlinearThermalProperties,
    contacts:AngularCoatingInterfaces,
    fuel_temperature_k:np.ndarray,
    coating_temperature_k:np.ndarray,
    hydrogen_temperature_k:np.ndarray,
) -> AngularInterfacePower:
    """Physical per-angle contact W; no source generated or counted twice."""
    nz,nfuel=fuel_temperature_k.shape
    na=contacts.sector_count
    if (coating_temperature_k.shape!=(nz,19,na)
            or hydrogen_temperature_k.shape!=(nz,19)
            or nfuel!=model.transverse.n_fuel
            or contacts.fuel_outer_full_m2.shape!=(nfuel,19*na)
            or not np.all(np.isfinite(fuel_temperature_k))
            or not np.all(np.isfinite(coating_temperature_k))
            or not np.all(np.isfinite(hydrogen_temperature_k))):
        raise ValueError("angular fuel/coating/hydrogen interface shape mismatch")
    kf=laws.fuel_k_w_m_k.at(fuel_temperature_k)
    kc=laws.coating_k_w_m_k.at(coating_temperature_k)
    film=laws.hydrogen_film_w_m2_k.at(hydrogen_temperature_k)
    rim=contacts.fuel_outer_full_m2.tocoo()
    fidx=rim.row.astype(int)
    channel=(rim.col//na).astype(int)
    ai=(rim.col%na).astype(int)
    length=float(model.axial_edges_m[-1]-model.axial_edges_m[0])
    dz=np.diff(model.axial_edges_m)
    arc=dz[:,None]*rim.data[None,:]/length
    r_outer=coating.outer_radius_m
    r_mid=coating.geometric_mid_radius_m
    bore=coating.inner_radius_m
    if coating.fuel_wall_distance_model=="moment_normal_depth":
        dm=contacts.fuel_wall_moment_depth_m
        if dm is None or dm.shape!=contacts.fuel_outer_full_m2.shape:
            raise ValueError("missing physical fuel centroid/arc moment distances")
        local=np.asarray(dm[fidx,rim.col]).ravel()
        if np.any(local<=0) or not np.all(np.isfinite(local)):
            raise ArithmeticError("nonphysical positive fuel centroid-normal wall distance")
    elif coating.fuel_wall_distance_model=="local_half_depth":
        dm=contacts.fuel_wall_half_depth_m
        if dm is None or dm.shape!=contacts.fuel_outer_full_m2.shape:
            raise ValueError("missing local fuel wall half-depth contact geometry")
        local=np.asarray(dm[fidx,rim.col]).ravel()
        if np.any(local<=0) or not np.all(np.isfinite(local)):
            raise ArithmeticError("invalid local fuel wall half-depth")
    elif coating.fuel_wall_distance_model=="global_half_cell":
        local=coating.half_fuel_distance_m
    else:
        raise ValueError("unknown fuel-wall embedded-boundary distance model")
    resistance=(
        local/kf[:,fidx]
        +r_outer*math.log(r_outer/r_mid)/kc[:,channel,ai]
        +coating.contact_resistance_m2_k_w[:,channel])
    if np.any(resistance<=0):
        raise ArithmeticError("unphysical angular coating contact resistance")
    flow=arc/resistance*(
        fuel_temperature_k[:,fidx]-coating_temperature_k[:,channel,ai])
    fuel_out=np.zeros((nz,nfuel))
    coat_in=np.zeros((nz,19,na))
    for z in range(nz):
        np.add.at(fuel_out[z],fidx,flow[z])
        np.add.at(coat_in[z],(channel,ai),flow[z])
    inner=contacts.h2_inner_full_m2[None,:,:]*dz[:,None,None]/length
    film3=film[:,:,None]
    radius_term=bore*math.log(r_mid/bore)/kc
    h2_out=np.zeros_like(coating_temperature_k)
    conductance=np.zeros_like(coating_temperature_k)
    np.divide(inner, radius_term+np.where(film3>0,1/np.maximum(film3,1e-300),
                                         np.inf),
              out=conductance,where=(film3>0))
    h2_out=conductance*(
        coating_temperature_k-hydrogen_temperature_k[:,:,None])
    h2_by_ch=h2_out.sum(axis=2)
    balance_f=float(np.sum(fuel_out)-np.sum(coat_in))
    balance_h=float(np.sum(h2_out)-np.sum(h2_by_ch))
    if (abs(balance_f)>1e-10*max(1.,abs(float(fuel_out.sum())))
            or abs(balance_h)>1e-10*max(1.,abs(float(h2_out.sum())))):
        raise ArithmeticError("angular coating wall flux not antisymmetric")
    return AngularInterfacePower(
        fuel_to_coating_by_fuel_w=fuel_out,
        fuel_to_coating_by_sector_w=coat_in,
        coating_to_hydrogen_by_sector_w=h2_out,
        coating_to_hydrogen_by_channel_w=h2_by_ch,
        equal_opposite_fuel_interface_residual_w=balance_f,
        equal_opposite_hydrogen_interface_residual_w=balance_h,
    )


def distribute_channel_coating_power(
    power_w:np.ndarray,angular:AngularCoatingMesh,
) -> np.ndarray:
    """Conservatively distribute unresolved [z,19] into angular sectors.

    Area-proportional distribution is only a declared UNIFORM volumetric
    source approximation. It MUST NOT be mistaken for an OpenMC spatial
    heating measurement or an angularly resolved transport tally.
    """
    p=np.asarray(power_w,dtype=float)
    weights=angular.sector_area_m2
    nz=angular.sector_mass_kg.shape[0]
    if (p.shape!=(nz,19) or not np.all(np.isfinite(p))
            or np.any(p<0) or np.any(weights.sum(axis=1)<=0)):
        raise ValueError("coating source must be nonnegative finite [z,19] W")
    fraction=weights/weights.sum(axis=1)[:,None]
    distributed=p[:,:,None]*fraction[None,:,:]
    if not np.allclose(distributed.sum(axis=2),p,atol=5e-13,rtol=2e-15):
        raise ArithmeticError("coating heat source angular partition not conservative")
    return distributed
