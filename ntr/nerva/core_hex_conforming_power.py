"""Conservative Cartesian heating-mesh -> conforming triangular-prism mapper.

All input density fields must ALREADY be spatial energy deposition [W/cm3]
for the indicated solid material, not neutron flux, kappa-fission, or an
unqualified reactor-total tally. This code does not attempt to repair
missing local KERMA or infer a source rate from keff.

For each Cartesian XY voxel and target triangle, the planar intersection
area is computed by convex half-space polygon clipping (not centroid
sampling). For each axial voxel and thermal bin, their exact 1-D overlap
is integrated. Each prism power is sum(q_voxel * area_overlap * dz_overlap).

This is an exact projection of PIECEWISE-CONSTANT voxel heating into the
mesh in the geometric limit; it cannot recover subvoxel tally structure.
Fuel and ZrC have distinct explicitly supplied density fields.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from .core_hex_conforming_mesh import HexTriangleSection


@dataclass(frozen=True)
class CartesianSolidPower:
    """OpenMC-like Cartesian power fields; SI heating per cm3, bounds in cm."""
    x_edges_cm: np.ndarray
    y_edges_cm: np.ndarray
    z_edges_cm: np.ndarray
    fuel_density_w_cm3: np.ndarray
    coating_density_w_cm3: np.ndarray

    def validate(self)->None:
        axes=(self.x_edges_cm,self.y_edges_cm,self.z_edges_cm)
        for a in axes:
            a=np.asarray(a,dtype=float)
            if (a.ndim!=1 or a.size<2 or not np.isfinite(a).all()
                    or np.any(np.diff(a)<=0)):
                raise ValueError("invalid Cartesian power mesh axes")
        shape=tuple(len(a)-1 for a in axes)
        for name,a in (("fuel",self.fuel_density_w_cm3),
                       ("ZrC",self.coating_density_w_cm3)):
            x=np.asarray(a,dtype=float)
            if x.shape!=shape or not np.isfinite(x).all() or np.any(x<0):
                raise ValueError(f"{name} field requires nonnegative [nx,ny,nz] W/cm3")


@dataclass(frozen=True)
class PrismPowerProjection:
    solid_prism_power_w: np.ndarray       # [target_axial, triangles]
    fuel_power_w: float
    coating_power_w: float
    projection_is_physics_qualified: bool=False


def _clip_against(
    polygon:list[tuple[float,float]],axis:int,
    edge:float,keep_greater:bool,
)->list[tuple[float,float]]:
    """Clip a convex polygon against one axis-aligned half-space."""
    if not polygon:return []
    output=[]
    p0=polygon[-1]
    v0=p0[axis]-edge
    inside0=v0>=-1e-12 if keep_greater else v0<=1e-12
    for p1 in polygon:
        v1=p1[axis]-edge
        inside1=v1>=-1e-12 if keep_greater else v1<=1e-12
        if inside0!=inside1:
            denom=v0-v1
            if abs(denom)>1e-30:
                f=v0/denom
                output.append((
                    p0[0]+f*(p1[0]-p0[0]),
                    p0[1]+f*(p1[1]-p0[1])))
        if inside1:
            output.append(p1)
        p0,v0,inside0=p1,v1,inside1
    return output


def _rectangle_intersection_area(
    triangle:np.ndarray,
    xmin:float,xmax:float,ymin:float,ymax:float,
)->float:
    """Exact triangle-axis-aligned-rectangle area [cm2], up to FP roundoff."""
    poly=[(float(x),float(y)) for x,y in triangle]
    for axis,edge,greater in (
        (0,xmin,True),(0,xmax,False),(1,ymin,True),(1,ymax,False)):
        poly=_clip_against(poly,axis,edge,greater)
        if len(poly)<3:return 0.
    area=0.
    for a,b in zip(poly,poly[1:]+poly[:1]):
        area+=a[0]*b[1]-b[0]*a[1]
    return abs(area)*0.5


def project_cartesian_deposition_to_prisms(
    section:HexTriangleSection,
    target_z_edges_m:np.ndarray,
    source:CartesianSolidPower,
)->PrismPowerProjection:
    """Map evaluated material-separated voxel energy onto the 3-D thermal FE.

    The thermal mesh must be strictly enclosed by the source Cartesian
    mesh. This avoids silently losing an unbounded fraction of heat.
    The result is W per fuel/ZrC triangular prism, which the thermal FEM
    distributes consistently to its six solid nodes.
    """
    section.validate()
    source.validate()
    target_z=np.asarray(target_z_edges_m,dtype=float)*100.
    if (target_z.ndim!=1 or len(target_z)<2
            or not np.isfinite(target_z).all() or
            np.any(np.diff(target_z)<=0)):
        raise ValueError("invalid target thermal axial grid")
    xe=np.asarray(source.x_edges_cm)
    ye=np.asarray(source.y_edges_cm)
    ze=np.asarray(source.z_edges_cm)
    xy=np.asarray(section.xy_m,dtype=float)*100.
    tol=1e-8
    if (np.min(xy[:,0])<xe[0]-tol or np.max(xy[:,0])>xe[-1]+tol
            or np.min(xy[:,1])<ye[0]-tol or np.max(xy[:,1])>ye[-1]+tol
            or target_z[0]<ze[0]-tol or target_z[-1]>ze[-1]+tol):
        raise ValueError("thermal domain extends beyond nuclear heating mesh")
    material=np.asarray(section.material,dtype=int)
    out=np.zeros((len(target_z)-1,len(section.triangles)))
    fields=(np.asarray(source.fuel_density_w_cm3,dtype=float),
            np.asarray(source.coating_density_w_cm3,dtype=float))
    # Sparse XY voxel overlap: only raster cells intersecting the
    # triangle's bounding rectangle can contribute energy.
    overlaps:list[list[tuple[int,int,float]]]=[]
    for indices in section.triangles:
        tri=xy[indices]
        lo=tri.min(axis=0);hi=tri.max(axis=0)
        ix0=max(0,int(np.searchsorted(xe,lo[0],side="right")-1))
        iy0=max(0,int(np.searchsorted(ye,lo[1],side="right")-1))
        ix1=min(len(xe)-1,int(np.searchsorted(xe,hi[0],side="left")+1))
        iy1=min(len(ye)-1,int(np.searchsorted(ye,hi[1],side="left")+1))
        cells=[]
        for i in range(ix0,ix1):
            for j in range(iy0,iy1):
                area=_rectangle_intersection_area(
                    tri,xe[i],xe[i+1],ye[j],ye[j+1])
                if area>0:cells.append((i,j,area))
        if not cells:
            raise RuntimeError("no heating mesh cells intersect target triangle")
        overlaps.append(cells)
    # Axial projection integrates exact overlap length, not centroid z.
    for iz,(zlo,zhi) in enumerate(zip(target_z[:-1],target_z[1:])):
        k0=max(0,int(np.searchsorted(ze,zlo,side="right")-1))
        k1=min(len(ze)-1,int(np.searchsorted(ze,zhi,side="left")+1))
        for e,cells in enumerate(overlaps):
            field=fields[material[e]]
            power=0.
            for k in range(k0,k1):
                dz=max(0.,min(zhi,ze[k+1])-max(zlo,ze[k]))
                if dz<=0:continue
                for ix,iy,area in cells:
                    power+=float(field[ix,iy,k])*area*dz
            out[iz,e]=power
    if not np.isfinite(out).all() or np.any(out<0):
        raise RuntimeError("nonphysical Cartesian->prism deposition output")
    return PrismPowerProjection(
        solid_prism_power_w=out,
        fuel_power_w=float(np.sum(out[:,material==0])),
        coating_power_w=float(np.sum(out[:,material==1])),
    )
