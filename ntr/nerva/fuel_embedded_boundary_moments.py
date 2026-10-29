"""Stage79 analytic first moments for hex-clipped fuel and circular holes.

The actual Stage53 fuel cutcell is a polygon clipped by six hex facets,
MINUS any coated circular hole footprints. This module computes signed
first Cartesian moments of those material regions analytically (straight
segments and circular sectors), instead of treating the rectangular
Cartesian centre as the physical fuel-volume centroid.

For a curved wall arc, integrate the actual surface position and normal
over its angular interval; its centroid and exact mean outward normal
are NOT replaced by a planar boundary or polygonized circle.

A one-point/two-point fuel-to-coating interface transfer remains a
first-order/oblique FV closure. Nonorthogonal tangential offsets must
be reported; accurate reconstruction of tangential gradients and
independent curved-boundary manufactured solution remain TODO.
"""
from __future__ import annotations

import math

import numpy as np


def polygon_area_moments_cm(
    vertices: list[tuple[float,float]],
) -> tuple[float,float,float]:
    """Return signed (area [cm2], x-first-moment [cm3], y-first-moment).

    Input is an oriented, non-self-intersecting Cartesian/hex polygon.
    """
    if len(vertices)<3:
        return 0.,0.,0.
    a=mx=my=0.
    for p,q in zip(vertices,vertices[1:]+vertices[:1]):
        cross=p[0]*q[1]-q[0]*p[1]
        a+=cross
        mx+=(p[0]+q[0])*cross
        my+=(p[1]+q[1])*cross
    return .5*a,mx/6.,my/6.


def disk_polygon_area_moments_cm(
    vertices: list[tuple[float,float]],
    center:tuple[float,float],
    radius_cm:float,
) -> tuple[float,float,float]:
    """Signed analytic polygon∩disk area and its first moments.

    Standard exact edge-sector decomposition: subdivide each polygon
    segment at both circle intersections. For any resulting segment
    within the disk add the signed triangle (origin,p,q), otherwise
    the signed circular sector joining its endpoint angles. Triangle
    centroids and circle-sector first moments are integrated exactly.
    Results are translated from circle-centred coordinates to global.
    No polygonal approximation of the circumference is introduced.
    """
    if (not math.isfinite(radius_cm) or radius_cm<=0
            or not all(math.isfinite(v) for point in vertices for v in point)
            or len(center)!=2 or not all(math.isfinite(v) for v in center)):
        raise ValueError("finite disk and polygon geometry is required")
    if len(vertices)<3:
        return 0.,0.,0.
    cx,cy=center
    r2=radius_cm*radius_cm
    area=mx=my=0.
    relative=[(float(x)-cx,float(y)-cy) for x,y in vertices]
    for p,q in zip(relative,relative[1:]+relative[:1]):
        px,py=p
        dx,dy=q[0]-px,q[1]-py
        aa=dx*dx+dy*dy
        if aa<=1e-30:
            continue
        bb=2.*(px*dx+py*dy)
        cc=px*px+py*py-r2
        cuts=[0.,1.]
        discr=bb*bb-4.*aa*cc
        if discr>0:
            root=math.sqrt(discr)
            for t in ((-bb-root)/(2.*aa),(-bb+root)/(2.*aa)):
                if 0.<t<1.:
                    cuts.append(t)
        cuts.sort()
        for lo,hi in zip(cuts,cuts[1:]):
            if hi-lo<=1e-15:
                continue
            sx,sy=px+lo*dx,py+lo*dy
            tx,ty=px+hi*dx,py+hi*dy
            mx0,my0=(sx+tx)*.5,(sy+ty)*.5
            cross=sx*ty-sy*tx
            if mx0*mx0+my0*my0<=r2:
                a=.5*cross
                area+=a
                mx+=a*(sx+tx)/3.
                my+=a*(sy+ty)/3.
            else:
                delta=math.atan2(cross,sx*tx+sy*ty)
                start=math.atan2(sy,sx)
                area+=.5*r2*delta
                mx+=radius_cm**3/3.*(
                    math.sin(start+delta)-math.sin(start))
                my+=radius_cm**3/3.*(
                    math.cos(start)-math.cos(start+delta))
    return area,mx+cx*area,my+cy*area


def fuel_cutcell_volume_centroid_cm(
    polygon:list[tuple[float,float]],
    holes:list[tuple[float,float]],
    coated_radius_cm:float,
    *,
    exact_fuel_area_cm2:float|None=None,
) -> tuple[float,float,float]:
    """Area and exact first moments of inner-hex polygon MINUS holes.

    Uses the same oriented hex-clipped polygon and true channel cylinders
    as Stage53.  Each disjoint hole's intersection moments are subtracted.
    Reject large cancellation or nonpositive fuel cells; no geometric
    centroid inferred from the rectangular Cartesian box centre.
    """
    area,mx,my=polygon_area_moments_cm(polygon)
    if area<=0 or not math.isfinite(area):
        raise ValueError("invalid physical hex-clipped fuel polygon")
    if not math.isfinite(coated_radius_cm) or coated_radius_cm<=0:
        raise ValueError("nonpositive coated cylinder radius")
    x0=min(x for x,y in polygon)
    x1=max(x for x,y in polygon)
    y0=min(y for x,y in polygon)
    y1=max(y for x,y in polygon)
    for cx,cy in holes:
        if (cx+coated_radius_cm<x0 or cx-coated_radius_cm>x1
                or cy+coated_radius_cm<y0 or
                cy-coated_radius_cm>y1):
            continue
        a,hx,hy=disk_polygon_area_moments_cm(
            polygon,(cx,cy),coated_radius_cm)
        if a< -1e-12:
            raise ArithmeticError("negative polygon/circle intersection area")
        area-=max(0.,a)
        mx-=hx
        my-=hy
    if area<=1e-14 or not math.isfinite(area):
        raise ArithmeticError("invalid nonpositive fuel material centroid area")
    if exact_fuel_area_cm2 is not None and abs(
            area-exact_fuel_area_cm2)>max(1e-11,1e-7*exact_fuel_area_cm2):
        raise ArithmeticError("first-moment fuel material area differs from Stage53")
    x=mx/area
    y=my/area
    if not all(math.isfinite(v) for v in (x,y)):
        raise ArithmeticError("nonfinite actual fuel material centroid")
    return area,x,y


def circular_arc_centroid_and_normal_cm(
    center:tuple[float,float],radius_cm:float,
    theta_start:float,theta_stop:float,
) -> tuple[float,float,float,float,float]:
    """Analytic arc length, arc centroid (x,y) and mean outward normal.

    Mean normal is dimensionless and has magnitude <= 1, rather than
    pretending the curved finite-angle arc has a single exact unit normal.
    The vector is for a wall normal OUTWARD from coating hole into fuel.
    """
    if (not all(math.isfinite(v) for v in
                (*center,radius_cm,theta_start,theta_stop))
            or radius_cm<=0 or theta_stop<=theta_start
            or theta_stop-theta_start>2.*math.pi+1e-10):
        raise ValueError("invalid physical circular contact arc")
    span=theta_stop-theta_start
    middle=.5*(theta_start+theta_stop)
    factor=math.sin(.5*span)/(.5*span)
    nx=factor*math.cos(middle)
    ny=factor*math.sin(middle)
    cx,cy=center
    return (radius_cm*span,
            cx+radius_cm*nx,cy+radius_cm*ny,nx,ny)


def fuel_wall_moment_normal_depth_m(
    fuel_centroid_cm:tuple[float,float],
    circular_arc:tuple[float,float,float,float,float],
) -> tuple[float,float]:
    """Signed fuel-centroid-to-arc normal and tangential offsets in metres.

    The actual averaged arc normal is converted to a unit direction.
    Returning a TANGENTIAL displacement separately exposes the
    nonorthogonal FV cross-diffusion correction that Stage79 does NOT
    yet implement. Reject nonpositive normal depth; no guessed clamp.
    """
    length,px,py,nx,ny=circular_arc
    norm=math.hypot(nx,ny)
    if (not math.isfinite(length) or length<=0 or norm<=1e-10
            or not all(math.isfinite(z) for z in fuel_centroid_cm)):
        raise ValueError("invalid curved-wall averaged normal")
    ux,uy=nx/norm,ny/norm
    vx=fuel_centroid_cm[0]-px
    vy=fuel_centroid_cm[1]-py
    normal=.01*(vx*ux+vy*uy)
    tangent=.01*(-vx*uy+vy*ux)
    if normal<=0 or not math.isfinite(normal):
        raise ArithmeticError("nonpositive fuel-centroid normal distance to wall")
    if not math.isfinite(tangent):
        raise ArithmeticError("nonfinite nonorthogonal fuel-wall offset")
    return normal,tangent
