"""Geometry-resolved planar cut-cell fuel/H2 heat transport.

The solid is the y-oriented inner fuel hexagon minus 19 *actual* circular
ZrC-coated channel footprints. Exact polygon/disk intersections define cell
heat capacity; exact cut-face segments define internal heat flux; angular
intersection of each circle with the Cartesian grid and inner hex gives
per-cell fuel/ZrC contact arc lengths.

The scheme is conservative, SPD for pure conduction, implicit in coupled
hydrogen temperatures, and first order near embedded curved boundaries.
Cell-centre two-point flux and the half-grid boundary resistance are
approximations: it is NOT an unstructured body-fitted or validated local
thermo-mechanical reactor calculation. Source watts and material properties
are explicit inputs, not inferred from neutron criticality.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags, bmat
from scipy.sparse.linalg import spsolve

from .config import NervaConfig
from .fuel_sector_geometry import fuel_sector_fuel_areas_cm2
from .layout import coolant_channel_positions, minimum_channel_ligament_cm

_NORMALS = ((1.0, 0.0), (-1.0, 0.0),
            (0.5, math.sqrt(3.0)/2.0), (-0.5, -math.sqrt(3.0)/2.0),
            (0.5, -math.sqrt(3.0)/2.0), (-0.5, math.sqrt(3.0)/2.0))


@dataclass(frozen=True)
class CutCellFuel2D:
    """Physical fuel-only volumes, adiabatic exterior, 19 H2 passages."""

    x_edges_cm: np.ndarray
    y_edges_cm: np.ndarray
    cell_ids: np.ndarray
    cell_centers_cm: np.ndarray
    fuel_areas_cm2: np.ndarray
    fuel_volumes_m3: np.ndarray
    fuel_heat_capacity_j_k: np.ndarray
    fuel_conductivity_w_m_k: np.ndarray
    fuel_zrc_contact_m2: csr_matrix
    fuel_h2_conductance_w_k: csr_matrix
    lateral_face_pairs: np.ndarray
    lateral_face_conductance_w_k: np.ndarray
    lateral_laplacian_w_k: csr_matrix
    axial_length_m: float

    @property
    def n_fuel(self) -> int:
        return len(self.fuel_areas_cm2)

    @property
    def channel_fuel_interface_m2(self) -> np.ndarray:
        return np.asarray(self.fuel_zrc_contact_m2.sum(axis=0)).ravel()


@dataclass(frozen=True)
class CutCellStep:
    fuel_temperature_k: np.ndarray
    hydrogen_temperature_k: np.ndarray
    fuel_storage_j: float
    hydrogen_storage_j: float
    coolant_export_j: float
    deposited_j: float
    wall_exchange_j: float
    first_law_residual_j: float


def _cross(p: tuple[float, float], q: tuple[float, float]) -> float:
    return p[0]*q[1]-p[1]*q[0]


def _clip_polygon(poly: list[tuple[float, float]], ap: float
                  ) -> list[tuple[float, float]]:
    """Sutherland-Hodgman exact straight-sided polygon/hex clipping."""
    for nx, ny in _NORMALS:
        if not poly:
            return []
        new = []
        p = poly[-1]
        fp = nx*p[0]+ny*p[1]-ap
        for q in poly:
            fq = nx*q[0]+ny*q[1]-ap
            inp, inq = fp <= 0., fq <= 0.
            if inp != inq:
                frac = fp/(fp-fq)
                new.append((p[0]+frac*(q[0]-p[0]),
                            p[1]+frac*(q[1]-p[1])))
            if inq:
                new.append(q)
            p, fp = q, fq
        poly = new
    return poly


def _area_polygon(poly: list[tuple[float, float]]) -> float:
    return 0.5*sum(_cross(a,b) for a,b in zip(poly,poly[1:]+poly[:1]))


def _disk_polygon_intersection_area(
    poly: list[tuple[float, float]], cx: float, cy: float, radius: float
) -> float:
    """Analytic polygon/circle area by split segment triangles and sectors.

    Implements an exact planar integral, not point sampling or a 64-gon
    approximation; 'exact' is understood to floating-point precision.
    """
    if len(poly) < 3:
        return 0.
    area = 0.
    r2 = radius*radius
    for a, b in zip(poly,poly[1:]+poly[:1]):
        px, py = a[0]-cx, a[1]-cy
        dx, dy = b[0]-a[0], b[1]-a[1]
        aa = dx*dx+dy*dy
        if aa <= 1e-30:
            continue
        bb = 2*(px*dx+py*dy)
        cc = px*px+py*py-r2
        disc = bb*bb-4*aa*cc
        cuts = [0.,1.]
        if disc > 0.:
            sd = math.sqrt(disc)
            for t in ((-bb-sd)/(2*aa),(-bb+sd)/(2*aa)):
                if 0. < t < 1.:
                    cuts.append(t)
        cuts.sort()
        for u,v in zip(cuts,cuts[1:]):
            if v-u <= 1e-15:
                continue
            sx, sy = px+u*dx, py+u*dy
            tx, ty = px+v*dx, py+v*dy
            mx,my = 0.5*(sx+tx),0.5*(sy+ty)
            cross = sx*ty-sy*tx
            if mx*mx+my*my <= r2:
                area += 0.5*cross
            else:
                area += 0.5*r2*math.atan2(cross,sx*tx+sy*ty)
    return area


def _inside_hex(x: float, y: float, ap: float) -> bool:
    return all(nx*x+ny*y <= ap+1e-13 for nx,ny in _NORMALS)


def _line_fuel_length_cm(
    p: tuple[float,float], q: tuple[float,float], ap: float,
    holes: list[tuple[float,float]], radius: float,
) -> float:
    """Exact hex-clipped segment length minus disjoint circular intervals."""
    dx,dy = q[0]-p[0],q[1]-p[1]
    full=math.hypot(dx,dy)
    if full <= 0:
        return 0.
    lo,hi=0.,1.
    for nx,ny in _NORMALS:
        f=nx*p[0]+ny*p[1]-ap
        slope=nx*dx+ny*dy
        if abs(slope) < 1e-14:
            if f > 1e-12:
                return 0.
            continue
        t=-f/slope
        if slope > 0:
            hi=min(hi,t)
        else:
            lo=max(lo,t)
        if hi<=lo:
            return 0.
    lo=max(0.,lo)
    hi=min(1.,hi)
    if hi<=lo:
        return 0.
    removed=[]
    for cx,cy in holes:
        vx,vy=p[0]-cx,p[1]-cy
        aa=dx*dx+dy*dy
        bb=2*(vx*dx+vy*dy)
        cc=vx*vx+vy*vy-radius*radius
        disc=bb*bb-4*aa*cc
        if disc <= 0:
            continue
        d=math.sqrt(disc)
        s=max(lo,(-bb-d)/(2*aa))
        t=min(hi,(-bb+d)/(2*aa))
        if t>s:
            removed.append((s,t))
    removed.sort()
    length=hi-lo
    end=lo
    for s,t in removed:
        if t<=end:
            continue
        length-=t-max(s,end)
        end=t
    if length < -1e-12:
        raise ArithmeticError("overlapping coolant-hole footprints on a face")
    return max(0.,length)*full


def _circle_contact_arcs_cm(
    x_edges: np.ndarray, y_edges: np.ndarray, ids: np.ndarray,
    holes: list[tuple[float,float]], r: float, ap: float,
) -> list[tuple[int,int,float]]:
    """Exact circular arc clipped by each Cartesian cell and the inner hex."""
    arc=[]
    for h,(cx,cy) in enumerate(holes):
        events=[0.,2.*math.pi]
        def add(theta: float) -> None:
            theta=theta%(2.*math.pi)
            if 1e-13 < theta < 2.*math.pi-1e-13:
                events.append(theta)
        for edge in x_edges:
            v=(float(edge)-cx)/r
            if -1. < v < 1.:
                a=math.acos(v)
                add(a)
                add(-a)
        for edge in y_edges:
            v=(float(edge)-cy)/r
            if -1. < v < 1.:
                a=math.asin(v)
                add(a)
                add(math.pi-a)
        for nx,ny in _NORMALS:
            v=(ap-nx*cx-ny*cy)/r
            if -1. < v < 1.:
                a=math.atan2(ny,nx)
                d=math.acos(v)
                add(a+d)
                add(a-d)
        angles=sorted(set(events))
        for t0,t1 in zip(angles,angles[1:]):
            if t1-t0 <= 1e-13:
                continue
            theta=.5*(t0+t1)
            x=cx+r*math.cos(theta)
            y=cy+r*math.sin(theta)
            if not _inside_hex(x,y,ap):
                continue
            i=int(np.searchsorted(x_edges,x,side="right")-1)
            j=int(np.searchsorted(y_edges,y,side="right")-1)
            if not (0<=i<len(x_edges)-1 and 0<=j<len(y_edges)-1):
                raise ArithmeticError("channel contact outside fuel mesh")
            cell=int(ids[j,i])
            if cell<0:
                raise ArithmeticError("physical channel surface has no fuel cell")
            arc.append((cell,h,r*(t1-t0)))
    return arc


def _properties(name: str, values: object, n: int, *, positive: bool=True
                ) -> np.ndarray:
    v=np.asarray(values,dtype=float)
    if v.ndim==0:
        v=np.full(n,float(v))
    if v.shape!=(n,) or not np.all(np.isfinite(v)) or (
        np.any(v<=0) if positive else np.any(v<0)
    ):
        raise ValueError(f"{name} requires finite {'positive' if positive else 'nonnegative'} scalar or {n}-vector")
    return v


def build_cutcell_fuel_2d(
    config: NervaConfig,
    *,
    nx: int=48,
    ny: int=56,
    conductivity_w_m_k: object,
    density_kg_m3: object,
    heat_capacity_j_kg_k: object,
    zrc_conductivity_w_m_k: object,
    hydrogen_film_w_m2_k: object,
) -> CutCellFuel2D:
    """Build the actual noncircular fuel-only embedded-boundary heat operator.

    All dimensional properties must be supplied; boundary outside the inner
    hex is currently ADIABATIC. Each coolant hole has one H2 bulk unknown.
    """
    config.validate()
    if (not isinstance(nx,int) or isinstance(nx,bool) or not 20<=nx<=256
            or not isinstance(ny,int) or isinstance(ny,bool) or not 20<=ny<=256):
        raise ValueError("mesh nx and ny must be integers in [20,256]")
    if minimum_channel_ligament_cm(config)<=0:
        raise ValueError("coated channel disks must not overlap")
    ap=config.inner_fuel_flat_to_flat_cm/2
    edge=config.inner_fuel_edge_length_cm
    xedge=np.linspace(-ap,ap,nx+1)
    yedge=np.linspace(-edge,edge,ny+1)
    dx=xedge[1]-xedge[0]
    dy=yedge[1]-yedge[0]
    radius=config.coated_channel_radius_cm
    holes=coolant_channel_positions(config)
    if len(holes)!=19:
        raise ValueError("exactly 19 coated coolant channels required")
    # Reject geometry outside the one-hex footprint/fully contained bore
    # regime qualified in the original Stage51 analytical geometry audit.
    normals=_NORMALS
    for x,y in holes:
        distances=[ap-a*x-b*y for a,b in normals]
        if min(distances)<=config.channel_radius_cm:
            raise ValueError("hydrogen bore outside the inner hex")
        if sum(d<radius for d in distances)>1:
            raise ValueError("two-flat coating clipping unqualified")
    ids=np.full((ny,nx),-1,dtype=int)
    centers=[]
    areas=[]
    for j in range(ny):
        y0,y1=float(yedge[j]),float(yedge[j+1])
        for i in range(nx):
            x0,x1=float(xedge[i]),float(xedge[i+1])
            polygon=_clip_polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)],ap)
            if len(polygon)<3:
                continue
            original=_area_polygon(polygon)
            if original<=1e-17:
                continue
            removed=0.
            for cx,cy in holes:
                if (cx+radius<x0 or cx-radius>x1 or
                    cy+radius<y0 or cy-radius>y1):
                    continue
                disk=_disk_polygon_intersection_area(polygon,cx,cy,radius)
                if disk<-1e-12 or disk>original+1e-12:
                    raise ArithmeticError("invalid analytic circle/polygon area")
                removed+=max(0.,disk)
            fuel=original-removed
            if fuel < -1e-11:
                raise ArithmeticError("physical fuel cell has negative exact area")
            if fuel<=1e-14:
                continue
            ids[j,i]=len(areas)
            centers.append(((x0+x1)*.5,(y0+y1)*.5))
            areas.append(fuel)
    areas_np=np.asarray(areas)
    centers_np=np.asarray(centers)
    n=len(areas)
    actual=float(areas_np.sum())
    expected=float(np.sum(fuel_sector_fuel_areas_cm2(config)))
    if abs(actual-expected)>3e-10:
        raise ArithmeticError(
            f"cut-cell fuel area does not match OpenMC Stage51 CSG: {actual} vs {expected}"
        )
    k=_properties("fuel conductivity",conductivity_w_m_k,n)
    rho=_properties("fuel density",density_kg_m3,n)
    cp=_properties("fuel heat capacity",heat_capacity_j_kg_k,n)
    kz=_properties("ZrC conductivity",zrc_conductivity_w_m_k,19)
    film=_properties("hydrogen-side film",hydrogen_film_w_m2_k,19,positive=False)
    length=config.active_length_cm*0.01
    volumes=areas_np*1e-4*length
    heat_capacity=volumes*rho*cp

    # Interior cut face flux: exact available fuel segment, harmonic k,
    # symmetric edge conductance. Surfaces outside Ω are insulating.
    edges=[]
    conductances=[]
    for j in range(ny):
        y0,y1=float(yedge[j]),float(yedge[j+1])
        for i in range(1,nx):
            a,b=int(ids[j,i-1]),int(ids[j,i])
            if a<0 or b<0:
                continue
            face=_line_fuel_length_cm(
                (float(xedge[i]),y0),(float(xedge[i]),y1),ap,holes,radius)
            if face<=1e-12:
                continue
            kh=2*k[a]*k[b]/(k[a]+k[b])
            edges.append((a,b))
            conductances.append(kh*(face*.01)*length/(dx*.01))
    for j in range(1,ny):
        y=float(yedge[j])
        for i in range(nx):
            a,b=int(ids[j-1,i]),int(ids[j,i])
            if a<0 or b<0:
                continue
            face=_line_fuel_length_cm(
                (float(xedge[i]),y),(float(xedge[i+1]),y),ap,holes,radius)
            if face<=1e-12:
                continue
            kh=2*k[a]*k[b]/(k[a]+k[b])
            edges.append((a,b))
            conductances.append(kh*(face*.01)*length/(dy*.01))
    if not edges:
        raise ArithmeticError("cut-cell fuel graph has no conductive faces")
    conductances_np=np.asarray(conductances)
    diag=np.zeros(n)
    rr=[]
    cc=[]
    vv=[]
    for (a,b),g in zip(edges,conductances):
        if g<=0 or not math.isfinite(g):
            raise ArithmeticError("invalid finite-volume heat conductance")
        diag[a]+=g
        diag[b]+=g
        rr.extend((a,b))
        cc.extend((b,a))
        vv.extend((-g,-g))
    for a,value in enumerate(diag):
        rr.append(a)
        cc.append(a)
        vv.append(float(value))
    L=coo_matrix((vv,(rr,cc)),shape=(n,n)).tocsr()
    L.sum_duplicates()

    # True fuel/ZrC coating boundary arcs attributed to cut cells;
    # H2 film acts on corresponding bore arcs with the same angular fraction.
    arcs=_circle_contact_arcs_cm(xedge,yedge,ids,holes,radius,ap)
    area_rows=[]
    area_cols=[]
    area_vals=[]
    wall_vals=[]
    bore_r=config.channel_radius_cm*.01
    coat_r=radius*.01
    delta=.5*min(dx,dy)*.01
    for a,h,arc_cm in arcs:
        frac=arc_cm/(2*math.pi*radius)
        A=(arc_cm*.01)*length
        area_rows.append(a)
        area_cols.append(h)
        area_vals.append(A)
        if film[h]==0.:
            wall_vals.append(0.)
        else:
            # First-order centre-to-curved-wall fuel resistance (vanishes
            # with refinement), exact cylindrical ZrC and H2 film series.
            R_fuel=delta/(k[a]*A)
            R_coat=math.log(coat_r/bore_r)/(2*math.pi*length*frac*kz[h])
            R_film=1./(2*math.pi*bore_r*length*frac*film[h])
            wall_vals.append(1./(R_fuel+R_coat+R_film))
    contact=coo_matrix((area_vals,(area_rows,area_cols)),
                       shape=(n,19)).tocsr()
    wall=coo_matrix((wall_vals,(area_rows,area_cols)),
                    shape=(n,19)).tocsr()
    contact.sum_duplicates()
    wall.sum_duplicates()
    for h,(cx,cy) in enumerate(holes):
        d=min(ap-nx0*cx-ny0*cy for nx0,ny0 in _NORMALS)
        missing=2*math.acos(d/radius)*radius if d<radius else 0.
        reference=(2*math.pi*radius-missing)*.01*length
        observed=float(contact[:,h].sum())
        if abs(reference-observed)>1e-11:
            raise ArithmeticError("embedded-boundary channel arc conservation failed")
    if np.max(np.abs(np.asarray(L.sum(axis=1))))>1e-9:
        raise ArithmeticError("lateral diffusion violates constant-mode closure")
    return CutCellFuel2D(
        x_edges_cm=xedge,y_edges_cm=yedge,cell_ids=ids,
        cell_centers_cm=centers_np,fuel_areas_cm2=areas_np,
        fuel_volumes_m3=volumes,fuel_heat_capacity_j_k=heat_capacity,
        fuel_conductivity_w_m_k=k,
        fuel_zrc_contact_m2=contact,fuel_h2_conductance_w_k=wall,
        lateral_face_pairs=np.asarray(edges,dtype=int),
        lateral_face_conductance_w_k=conductances_np,
        lateral_laplacian_w_k=L,axial_length_m=length,
    )


def advance_cutcell_fuel_hydrogen(
    op: CutCellFuel2D,
    fuel_temperature_k: object,
    hydrogen_temperature_k: object,
    *,
    dt_s: float,
    fuel_source_w: object,
    hydrogen_source_w: object,
    hydrogen_heat_capacity_j_k: object,
    hydrogen_flow_heat_capacity_w_k: object,
    hydrogen_inlet_temperature_k: object,
) -> CutCellStep:
    """Implicit conservative spatial 2D fuel and 19-channel hydrogen solve.

    Fuel: Cf(T-Told)/dt + L T + W1*T - W*Th = Qf
    H2:   Ch(Th-Thold)/dt + Wt1*Th-Wt*T + F*(Th-Tin) = Qh
    """
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("dt_s must be positive finite seconds")
    n=op.n_fuel
    tf=_properties("fuel old temperature",fuel_temperature_k,n)
    th=_properties("hydrogen old temperature",hydrogen_temperature_k,19)
    qf=np.asarray(fuel_source_w,dtype=float)
    qh=np.asarray(hydrogen_source_w,dtype=float)
    if qf.shape!=(n,) or qh.shape!=(19,) or not (
        np.all(np.isfinite(qf)) and np.all(np.isfinite(qh))
    ):
        raise ValueError("power vectors must have valid dimensions and finite watts")
    ch=_properties("hydrogen storage J/K",hydrogen_heat_capacity_j_k,19)
    flow=_properties("hydrogen flow heat capacity W/K",
                     hydrogen_flow_heat_capacity_w_k,19,positive=False)
    inlet=_properties("hydrogen inlet temperature K",hydrogen_inlet_temperature_k,19)
    cf=op.fuel_heat_capacity_j_k
    W=op.fuel_h2_conductance_w_k
    wall_f=np.asarray(W.sum(axis=1)).ravel()
    wall_h=np.asarray(W.sum(axis=0)).ravel()
    A=bmat([
        [op.lateral_laplacian_w_k+diags(cf/dt_s+wall_f),-W],
        [-W.T,diags(ch/dt_s+wall_h+flow)],
    ],format="csc")
    rhs=np.concatenate((cf*tf/dt_s+qf,ch*th/dt_s+qh+flow*inlet))
    solved=np.asarray(spsolve(A,rhs),dtype=float)
    if solved.shape!=(n+19,) or not np.all(np.isfinite(solved)) or np.any(solved<=0):
        raise ArithmeticError("cut-cell heat equation returned nonphysical temperatures")
    f,h=solved[:n],solved[n:]
    uf=float(np.dot(cf,f-tf))
    uh=float(np.dot(ch,h-th))
    exported=float(dt_s*np.dot(flow,h-inlet))
    deposited=float(dt_s*(qf.sum()+qh.sum()))
    wall_exchange=float(dt_s*np.dot(wall_f,f)-dt_s*np.dot(wall_h,h))
    residual=uf+uh+exported-deposited
    scale=max(1.,abs(uf),abs(uh),abs(exported),abs(deposited))
    if abs(residual)>5e-9*scale:
        raise ArithmeticError("cut-cell fuel/H2 first-law energy conservation failed")
    return CutCellStep(f,h,uf,uh,exported,deposited,wall_exchange,residual)
