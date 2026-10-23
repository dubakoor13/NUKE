"""Conforming 2-D Gmsh cross-section for the NUKE 19-bore hexagonal fuel element.

Explicit fuel XY triangles, ZrC coating rings, and 19 coolant voids. The
mesh is extruded implicitly by the companion 3-D thermal FEM solver.
Gmsh is optional and imported only when generating a new mesh. This
geometry is the existing NUKE demonstration geometry, not a NASA drawing.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from pathlib import Path
import numpy as np
from .config import NervaConfig
from .layout import coolant_channel_positions


@dataclass(frozen=True)
class HexTriangleSection:
    """2-D conforming first-order triangles; 0=fuel, 1=ZrC."""
    xy_m: np.ndarray
    triangles: np.ndarray
    material: np.ndarray
    wall_edges: np.ndarray
    wall_channel: np.ndarray

    def validate(self)->None:
        xy=np.asarray(self.xy_m)
        tri=np.asarray(self.triangles)
        material=np.asarray(self.material)
        edges=np.asarray(self.wall_edges)
        channels=np.asarray(self.wall_channel)
        if (xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<4
            or tri.ndim!=2 or tri.shape[1]!=3 or len(tri)<1
            or tri.dtype.kind not in "iu" or edges.ndim!=2
            or edges.shape[1]!=2 or len(edges)<1 or edges.dtype.kind not in "iu"
            or material.shape!=(len(tri),) or channels.shape!=(len(edges),)
            or not np.isfinite(xy).all()
            or np.any(tri<0) or np.any(tri>=len(xy))
            or np.any(edges<0) or np.any(edges>=len(xy))
            or np.any((material!=0)&(material!=1))
            or np.any((channels<0)|(channels>=19))):
            raise ValueError("invalid conforming triangle mesh")
        p=xy[tri]
        signed=0.5*((p[:,1,0]-p[:,0,0])*(p[:,2,1]-p[:,0,1])
                    -(p[:,2,0]-p[:,0,0])*(p[:,1,1]-p[:,0,1]))
        if np.any(signed<=0):
            raise ValueError("reversed or degenerate triangles")
        if np.any(np.linalg.norm(xy[edges[:,0]]-xy[edges[:,1]],axis=1)<=0):
            raise ValueError("zero-length channel wall")
        if len(np.unique(tri))!=len(xy):
            raise ValueError("orphan mesh nodes")

    @property
    def triangle_area_m2(self)->np.ndarray:
        p=self.xy_m[self.triangles]
        return 0.5*np.abs(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]))

    @property
    def triangle_centroid_m(self)->np.ndarray:
        return self.xy_m[self.triangles].mean(axis=1)


def make_nerva_hex_triangle_section(
    *,config:NervaConfig|None=None,
    fuel_mesh_size_m:float=6e-4,
    coating_mesh_size_m:float=8e-5,
    output_msh:Path|None=None,
)->HexTriangleSection:
    """Conforming outer-element hex minus 19 bores plus their coating annuli.

    The external perimeter ZrC skin is not explicitly meshed. The inner
    fuel hex would cut six bore coatings, so never use it as this envelope.
    """
    cfg=config or NervaConfig()
    cfg.validate()
    if any(not math.isfinite(x) or x<=0 for x in
           (fuel_mesh_size_m,coating_mesh_size_m)):
        raise ValueError("positive mesh sizes required")
    if coating_mesh_size_m>fuel_mesh_size_m:
        raise ValueError("coating mesh must be at least as fine as fuel")
    try:
        import gmsh
    except ImportError as exc:
        raise RuntimeError("Install optional gmsh and GLU runtime") from exc
    if gmsh.isInitialized():
        raise RuntimeError("Gmsh is already initialized")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal",0)
        gmsh.model.add("NUKE_NERVA_19_bores_thermal")
        geo=gmsh.model.geo
        # The inner-fuel hex (outer ZrC perimeter removed) intersects six
        # of the outer bore coatings in the currently documented NUKE
        # surrogate: margin -0.00363 cm. The outer fuel ELEMENT hex
        # physically contains them, with only +0.00137 cm clearance.
        # Do not move/enlarge bores or pretend the outer coating is meshed.
        side=cfg.fuel_edge_length_cm*0.01
        x=0.5*math.sqrt(3)*side
        vertices=np.array([
            [0.,side],[-x,side/2],[-x,-side/2],
            [0.,-side],[x,-side/2],[x,side/2]
        ],dtype=float)
        # Every coated hole must fit strictly inside the external hex.
        radius=cfg.coated_channel_radius_cm*0.01
        for cx_cm,cy_cm in coolant_channel_positions(cfg):
            pt=np.array([cx_cm,cy_cm],dtype=float)*0.01
            for edge_start,edge_end in zip(vertices,np.roll(vertices,-1,axis=0)):
                vector=edge_end-edge_start
                inward=(vector[0]*(pt-edge_start)[1]-
                        vector[1]*(pt-edge_start)[0])/np.linalg.norm(vector)
                if inward<=radius:
                    raise ValueError(
                        "NUKE coated coolant bore overlaps external hex boundary; "
                        "refusing invalid conforming geometry")
        points=[geo.addPoint(float(x),float(y),0.,fuel_mesh_size_m)
                for x,y in vertices]
        boundary=[geo.addLine(points[j],points[(j+1)%6]) for j in range(6)]
        hex_loop=geo.addCurveLoop(boundary)
        outer_loops=[]
        coating_surfaces=[]
        wall_arcs=[]
        for cx_cm,cy_cm in coolant_channel_positions(cfg):
            cx=cx_cm*0.01
            cy=cy_cm*0.01
            center=geo.addPoint(cx,cy,0.,coating_mesh_size_m)
            loops=[]
            for radius in (cfg.coated_channel_radius_cm*0.01,
                           cfg.channel_radius_cm*0.01):
                ps=[geo.addPoint(cx+radius*math.cos(j*math.pi/2),
                                 cy+radius*math.sin(j*math.pi/2),
                                 0.,coating_mesh_size_m) for j in range(4)]
                arcs=[geo.addCircleArc(ps[j],center,ps[(j+1)%4])
                      for j in range(4)]
                loops.append((geo.addCurveLoop(arcs),arcs))
            outer,inner=loops
            outer_loops.append(outer[0])
            coating_surfaces.append(
                geo.addPlaneSurface([outer[0],inner[0]]))
            wall_arcs.append(inner[1])
        fuel_surface=geo.addPlaneSurface([hex_loop,*outer_loops])
        geo.synchronize()
        gmsh.model.addPhysicalGroup(2,[fuel_surface],1)
        gmsh.model.setPhysicalName(2,1,"fuel")
        gmsh.model.addPhysicalGroup(2,coating_surfaces,2)
        gmsh.model.setPhysicalName(2,2,"zrc_coating")
        for idx,arcs in enumerate(wall_arcs):
            tag=100+idx
            gmsh.model.addPhysicalGroup(1,arcs,tag)
            gmsh.model.setPhysicalName(1,tag,f"channel_{idx+1:02d}")
        gmsh.option.setNumber("Mesh.ElementOrder",1)
        gmsh.option.setNumber("Mesh.Algorithm",6)
        gmsh.option.setNumber("Mesh.MeshSizeMax",fuel_mesh_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMin",coating_mesh_size_m*0.5)
        gmsh.model.mesh.generate(2)
        if output_msh is not None:
            path=Path(output_msh)
            path.parent.mkdir(parents=True,exist_ok=True)
            gmsh.write(str(path))
        tags,xyz,_=gmsh.model.mesh.getNodes()
        tags=np.asarray(tags,dtype=np.int64)
        xy=np.asarray(xyz,dtype=float).reshape(-1,3)[:,:2]
        index={int(tag):i for i,tag in enumerate(tags)}
        triangles=[];material=[]
        for surf,kind in [(fuel_surface,0),*[(s,1) for s in coating_surfaces]]:
            types,_,connections=gmsh.model.mesh.getElements(2,surf)
            for tp,nodes in zip(types,connections):
                if int(tp)!=2:
                    raise RuntimeError("expected P1 triangular elements")
                array=np.asarray(nodes,dtype=np.int64).reshape(-1,3)
                triangles.extend([[index[int(v)] for v in row] for row in array])
                material.extend([kind]*len(array))
        wall_edges=[];wall_channel=[]
        for channel,arcs in enumerate(wall_arcs):
            for arc in arcs:
                types,_,connections=gmsh.model.mesh.getElements(1,arc)
                for tp,nodes in zip(types,connections):
                    if int(tp)!=1:
                        raise RuntimeError("expected P1 wall segments")
                    array=np.asarray(nodes,dtype=np.int64).reshape(-1,2)
                    wall_edges.extend([[index[int(v)] for v in row] for row in array])
                    wall_channel.extend([channel]*len(array))
        tri=np.asarray(triangles,dtype=np.int64)
        for row in tri:
            p=xy[row]
            cross=(p[1,0]-p[0,0])*(p[2,1]-p[0,1])-(
                p[2,0]-p[0,0])*(p[1,1]-p[0,1])
            if cross<0:
                row[1],row[2]=row[2],row[1]
        wall_edges=np.asarray(wall_edges,dtype=np.int64).reshape(-1,2)
        used=np.unique(np.r_[tri.reshape(-1),wall_edges.reshape(-1)])
        mapping=np.full(len(xy),-1,dtype=int)
        mapping[used]=np.arange(len(used))
        section=HexTriangleSection(
            xy[used],mapping[tri],np.asarray(material,dtype=int),
            mapping[wall_edges],np.asarray(wall_channel,dtype=int))
        section.validate()
        if len(np.unique(section.wall_channel))!=19:
            raise RuntimeError("missing coated coolant channel boundary")
        if not np.any(section.material==0) or not np.any(section.material==1):
            raise RuntimeError("fuel/coating tagging failed")
        return section
    finally:
        gmsh.finalize()
