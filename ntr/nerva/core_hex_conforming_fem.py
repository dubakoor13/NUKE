"""Conforming triangular-prism 3-D fuel/coat/H2 thermal finite elements.

A single 2-D conforming P1 triangle mesh of a hexagonal fuel element with
19 circular coolant bores and individually meshed coating rings is extruded
along a common axial grid. Each 6-node triangular prism uses tensor-product
P1(x,y)*P1(z) Galerkin conduction; continuous nodal temperatures share the
actual coating/fuel interfaces, unlike the annulus/Voronoi approximations.

At every segment of each bore, a consistent wall-surface finite-element
convection matrix is coupled monolithically to the corresponding 1-D H2
enthalpy control-volume temperature. Gas uses backward-Euler stored
enthalpy and implicit upwind axial advection. The solid uses lumped volume
heat capacity with exact secant enthalpy for a linear cp(T) law. Both
coefficients and conductivity can change during nonlinear Picard iteration.

External reactor nuclear deposition is required in watts per 2-D triangle
per axial bin. The solver cannot infer local KERMA from a keff tally,
assume historical reactor power, validate fuel properties, or compute
pressure/flow feedback. Outer hex faces and both axial ends are insulated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .core_conjugate_fv import CoreSolidLaw, _gas_enthalpy, _gas_secant
from .core_hex_lateral_fv import HexCoolantBoundary
from .core_hex_conforming_mesh import HexTriangleSection
from .core_hex_exterior_surface import (
    CoreExteriorHeatBoundary, exterior_fuel_edges,
    exterior_heat_coefficients, exterior_boundary_power,
)


@dataclass(frozen=True)
class ConformingCoreState:
    solid_temperature_k: np.ndarray      # [nz+1, n_xy_nodes]
    hydrogen_temperature_k: np.ndarray   # [nz, 19]


@dataclass(frozen=True)
class ConformingCoreStep:
    state: ConformingCoreState
    wall_to_hydrogen_w: np.ndarray        # [nz, 19] W
    stored_solid_j: float
    stored_hydrogen_j: float
    inlet_enthalpy_w: float
    outlet_enthalpy_w: float
    first_law_residual: float             # W steady, J transient
    iterations: int
    transient: bool
    exterior_heat_loss_w: float = 0.0


def _triangle_operators(xy:np.ndarray,tri:np.ndarray
                        )->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Positive triangle areas, exact P1 planar mass/stiffness templates."""
    p=np.asarray(xy,dtype=float)[tri]
    edge1=p[:,1]-p[:,0]
    edge2=p[:,2]-p[:,0]
    determinant=edge1[:,0]*edge2[:,1]-edge1[:,1]*edge2[:,0]
    if np.any(determinant<=0):
        raise ValueError("mesh has degenerate or inverted fuel triangles")
    area=0.5*determinant
    grad=np.stack((
        np.stack((p[:,1,1]-p[:,2,1],p[:,2,0]-p[:,1,0]),axis=1),
        np.stack((p[:,2,1]-p[:,0,1],p[:,0,0]-p[:,2,0]),axis=1),
        np.stack((p[:,0,1]-p[:,1,1],p[:,1,0]-p[:,0,0]),axis=1),
    ),axis=1)/(2*area[:,None,None])
    stiffness=area[:,None,None]*np.einsum("tik,tjk->tij",grad,grad)
    mass=area[:,None,None]*np.array([
        [2,1,1],[1,2,1],[1,1,2]],dtype=float)[None,:,:]/12
    return area,mass,stiffness


def _source_array(
    section:HexTriangleSection,z_edges_m:np.ndarray,
    source_w:np.ndarray,
)->np.ndarray:
    """Power in W per [z_bin,2D_triangle] extruded prism."""
    a=np.asarray(source_w,dtype=float)
    shape=(len(z_edges_m)-1,len(section.triangles))
    if a.shape!=shape or not np.isfinite(a).all() or np.any(a<0):
        raise ValueError("solid deposition must be finite [z,triangle] W >= 0")
    return a


def solve_conforming_core_thermal(
    section:HexTriangleSection,
    z_edges_m:np.ndarray,
    previous:ConformingCoreState,
    boundary:HexCoolantBoundary,
    fuel:CoreSolidLaw,
    coating:CoreSolidLaw,
    solid_power_w:np.ndarray,
    *,
    dt_s:float|None=None,
    hydrogen_kinetic_power_w:np.ndarray|None=None,
    exterior_boundary:CoreExteriorHeatBoundary|None=None,
    max_iterations:int=60,
    tolerance_k:float=1e-8,
    relaxation:float=0.8,
    balance_tolerance:float=1e-7,
)->ConformingCoreStep:
    """Solve simultaneous fuel, ZrC and 19 H2-channel energy equations.

    The finite-element mesh must explicitly represent the material regions,
    rather than transfer temperatures through artificial sector resistors.
    Within each prism, k is evaluated at its six-node mean temperature.
    """
    section.validate()
    fuel.validate()
    coating.validate()
    z=np.asarray(z_edges_m,dtype=float)
    if (z.ndim!=1 or len(z)<2 or not np.isfinite(z).all()
            or np.any(np.diff(z)<=0)):
        raise ValueError("invalid 3-D axial mesh")
    nz=len(z)-1
    boundary.validate(nz)
    power=_source_array(section,z,solid_power_w)
    # Steady compressible channel momentum coupling supplies the signed
    # kinetic-energy change -mdot*(v_out^2-v_in^2)/2 per H2 cell.
    # Without this, prescribing pressure loss while solving only static
    # gas enthalpy would violate stagnation-energy conservation.
    kinetic=np.zeros((nz,19),dtype=float)
    if hydrogen_kinetic_power_w is not None:
        kinetic=np.asarray(hydrogen_kinetic_power_w,dtype=float)
        if (kinetic.shape!=(nz,19) or not np.isfinite(kinetic).all()):
            raise ValueError("kinetic power must be finite [z,19] W")
        if dt_s is not None:
            raise ValueError(
                "quasi-steady kinetic-power correction is not supported "
                "with transient H2 inventory; use steady state only")
    nxy=len(section.xy_m)
    solid_initial=np.asarray(previous.solid_temperature_k,dtype=float)
    gas_initial=np.asarray(previous.hydrogen_temperature_k,dtype=float)
    if (solid_initial.shape!=(nz+1,nxy)
            or gas_initial.shape!=(nz,19)
            or any(not np.isfinite(t).all() or np.any(t<=0)
                   for t in (solid_initial,gas_initial))):
        raise ValueError("invalid conforming thermal starting fields")
    if (dt_s is not None and (not math.isfinite(dt_s) or dt_s<=0)
            or max_iterations<2 or not math.isfinite(tolerance_k)
            or tolerance_k<=0 or not 0<relaxation<=1
            or not math.isfinite(balance_tolerance) or balance_tolerance<=0):
        raise ValueError("invalid implicit timestep/nonlinear controls")
    gas_law=boundary.gas_boundary()
    for t in gas_initial.flat:
        _gas_enthalpy(float(t),gas_law)
    triangle=np.asarray(section.triangles,dtype=int)
    xy=np.asarray(section.xy_m,dtype=float)
    area,m2d,k2d=_triangle_operators(xy,triangle)
    dz=np.diff(z)
    ntri=len(triangle)
    ns=(nz+1)*nxy
    total=ns+nz*19
    material=np.asarray(section.material,dtype=int)
    law_by_material=(fuel,coating)
    solid=solid_initial.copy()
    gas=gas_initial.copy()
    mass_1d=np.array([[2.,1.],[1.,2.]])/6
    stiff_1d=np.array([[1.,-1.],[-1.,1.]])
    m_edge=np.array([[2.,1.],[1.,2.]])/6
    wall_edges=np.asarray(section.wall_edges,dtype=int)
    wall_channel=np.asarray(section.wall_channel,dtype=int)
    wall_length=np.linalg.norm(
        xy[wall_edges[:,0]]-xy[wall_edges[:,1]],axis=1)
    outer_edges=(exterior_fuel_edges(section) if exterior_boundary is not None
                 else None)
    if outer_edges is not None:
        exterior_boundary.expanded(nz,len(outer_edges))
        outer_length=np.linalg.norm(
            xy[outer_edges[:,1]]-xy[outer_edges[:,0]],axis=1)

    for iteration in range(1,max_iterations+1):
        row:list[int]=[]
        col:list[int]=[]
        val:list[float]=[]
        diag=np.zeros(total)
        rhs=np.zeros(total)

        def dense_block(indices:np.ndarray,mat:np.ndarray)->None:
            for a in range(len(indices)):
                idx=int(indices[a])
                for b in range(len(indices)):
                    row.append(idx)
                    col.append(int(indices[b]))
                    val.append(float(mat[a,b]))

        # True XY triangle / Z line tensor-product volume FE.
        for i in range(nz):
            mz=dz[i]*mass_1d
            kz=stiff_1d/dz[i]
            for e,nodes in enumerate(triangle):
                indices=np.r_[nodes+i*nxy,nodes+(i+1)*nxy]
                law=law_by_material[int(material[e])]
                temps=solid.reshape(-1)[indices]
                old=solid_initial.reshape(-1)[indices]
                cond=law.conductivity(float(temps.mean()))
                K=cond*(np.kron(mz,k2d[e])+np.kron(kz,m2d[e]))
                dense_block(indices,K)
                source=float(power[i,e])/6.
                volume_six=area[e]*dz[i]/6.
                for node_index,now,prior in zip(indices,temps,old):
                    idx=int(node_index)
                    rhs[idx]+=source
                    if dt_s is not None:
                        capacity=(volume_six*law.density_kg_m3*
                            law.secant_cp(float(prior),float(now))/dt_s)
                        diag[idx]+=capacity
                        rhs[idx]+=capacity*prior

        # Shared discrete wall flux: positive from coating into hydrogen.
        for i in range(nz):
            mz=dz[i]*mass_1d
            for edge_index,(a,b) in enumerate(wall_edges):
                c=int(wall_channel[edge_index])
                h=float(boundary.film_w_m2_k[i,c])
                nodes=np.array((a,b,a+nxy*(i+1),b+nxy*(i+1)),
                               dtype=int)
                nodes[:2]+=i*nxy
                wall_mat=h*wall_length[edge_index]*np.kron(mz,m_edge)
                dense_block(nodes,wall_mat)
                gidx=ns+i*19+c
                weights=wall_mat.sum(axis=1)
                for node_index,w in zip(nodes,weights):
                    row.append(int(node_index))
                    col.append(int(gidx))
                    val.append(-float(w))
                    row.append(int(gidx))
                    col.append(int(node_index))
                    val.append(-float(w))
                diag[gidx]+=float(wall_mat.sum())

        # External contact / radiation to an EXPLICITLY prescribed reservoir.
        # Consistent quadratic 4-node surface matrix, same formulation as
        # the H2 wall coupling but without a solved external gas node.
        if outer_edges is not None:
            _,ext_h,ext_bath,_,_=exterior_heat_coefficients(
                solid,z,section,exterior_boundary,edges=outer_edges)
            for i in range(nz):
                mz=dz[i]*mass_1d
                for e,(a,b) in enumerate(outer_edges):
                    h=float(ext_h[i,e])
                    if h==0.:
                        continue
                    nodes=np.array((a,b,a+nxy*(i+1),b+nxy*(i+1)),
                                   dtype=int)
                    nodes[:2]+=i*nxy
                    matrix=h*outer_length[e]*np.kron(mz,m_edge)
                    dense_block(nodes,matrix)
                    weights=matrix.sum(axis=1)
                    for node_index,w in zip(nodes,weights):
                        rhs[int(node_index)]+=float(w*ext_bath[i,e])
        # Hydrogen enthalpy storage, direct nuclear heating, implicit upwind.
        for i in range(nz):
            for c in range(19):
                idx=ns+i*19+c
                rhs[idx]+=float(boundary.direct_hydrogen_power_w[i,c]+kinetic[i,c])
                if dt_s is not None:
                    gas_capacity=(boundary.hydrogen_mass_kg[i,c]*
                        _gas_secant(float(gas_initial[i,c]),
                                    float(gas[i,c]),gas_law)/dt_s)
                    diag[idx]+=gas_capacity
                    rhs[idx]+=gas_capacity*gas_initial[i,c]
                upstream=(float(boundary.inlet_temperature_k[c])
                          if i==0 else float(gas[i-1,c]))
                adv=boundary.mdot_kg_s[c]*_gas_secant(
                    upstream,float(gas[i,c]),gas_law)
                diag[idx]+=adv
                if i==0:
                    rhs[idx]+=adv*boundary.inlet_temperature_k[c]
                else:
                    row.append(idx)
                    col.append(ns+(i-1)*19+c)
                    val.append(-float(adv))
        indices=np.arange(total,dtype=int)
        row.extend(indices.tolist())
        col.extend(indices.tolist())
        val.extend(diag.tolist())
        A=coo_matrix((val,(row,col)),shape=(total,total)).tocsr()
        solved=np.asarray(spsolve(A,rhs),dtype=float)
        if not np.isfinite(solved).all() or np.min(solved)<=0:
            raise RuntimeError("conforming 3-D thermal linearization diverged")
        solid_new=solved[:ns].reshape((nz+1,nxy))
        gas_new=solved[ns:].reshape((nz,19))
        for t in gas_new.flat:
            _gas_enthalpy(float(t),gas_law)
        maximum_delta=max(
            float(np.max(np.abs(solid_new-solid))),
            float(np.max(np.abs(gas_new-gas))))
        solid=relaxation*solid_new+(1-relaxation)*solid
        gas=relaxation*gas_new+(1-relaxation)*gas
        if maximum_delta<tolerance_k:
            break
    else:
        raise RuntimeError("conforming 3-D core nonlinear solve did not converge")

    wall=np.zeros((nz,19))
    for i in range(nz):
        for edge_index,(a,b) in enumerate(wall_edges):
            c=int(wall_channel[edge_index])
            solid_average=(solid[i,a]+solid[i,b]+
                           solid[i+1,a]+solid[i+1,b])/4
            wall[i,c]+=(boundary.film_w_m2_k[i,c]*
                        wall_length[edge_index]*dz[i]*
                        (solid_average-gas[i,c]))
    stored_solid=0.
    stored_gas=0.
    if dt_s is not None:
        stored_solid=math.fsum(
            law_by_material[int(material[e])].density_kg_m3*
            area[e]*dz[i]/6.*
            law_by_material[int(material[e])].enthalpy_change(
                float(solid_initial.reshape(-1)[idx]),
                float(solid.reshape(-1)[idx]))
            for i in range(nz) for e,nodes in enumerate(triangle)
            for idx in np.r_[nodes+i*nxy,nodes+(i+1)*nxy])
        stored_gas=math.fsum(
            boundary.hydrogen_mass_kg[i,c]*
            (_gas_enthalpy(float(gas[i,c]),gas_law)-
             _gas_enthalpy(float(gas_initial[i,c]),gas_law))
            for i in range(nz) for c in range(19))
    enthalpy_in=math.fsum(
        boundary.mdot_kg_s[c]*
        _gas_enthalpy(float(boundary.inlet_temperature_k[c]),gas_law)
        for c in range(19))
    enthalpy_out=math.fsum(
        boundary.mdot_kg_s[c]*
        _gas_enthalpy(float(gas[-1,c]),gas_law)
        for c in range(19))
    exterior_loss=0.
    if exterior_boundary is not None:
        _,exterior_loss=exterior_boundary_power(
            solid,z,section,exterior_boundary)
    source_total=float(power.sum()+np.sum(boundary.direct_hydrogen_power_w)+
                       np.sum(kinetic))
    residual=(stored_solid+stored_gas-
              (source_total+enthalpy_in-enthalpy_out-exterior_loss)*
              (dt_s if dt_s is not None else 1.))
    scale=max(1.,abs(source_total),abs(enthalpy_in-enthalpy_out),
              abs(exterior_loss))
    scale*=dt_s if dt_s is not None else 1.
    if abs(residual)>balance_tolerance*scale+1e-6:
        raise RuntimeError(
            f"conforming core conjugate first-law failure: {residual:g}")
    return ConformingCoreStep(
        ConformingCoreState(solid,gas),wall,stored_solid,stored_gas,
        enthalpy_in,enthalpy_out,residual,iteration,dt_s is not None,
        exterior_loss)
