"""Monolithic *two-element* conforming reactor-core thermal/H2 solver.

Unlike two independent elements with prescribed outer bath temperatures,
this assembles one global sparse system with a symmetric nonmatching-mesh
mortar contact operator: the interface heat leaving element A is EXACTLY
the heat entering B. Fuel and coating are resolved in 3D P1 prisms within
each element and both 19-channel hydrogen energy networks are coupled
through their own wall heat-transfer matrices.

This is the first physically conservative inter-element solid-conduction
coding step. Contact conductance remains a supplied constitutive input;
the module DOES NOT model a historical NERVA full core, tie-tube system,
hot neutron transport, or compressible coolant pressure equations.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix, block_diag
from scipy.sparse.linalg import spsolve

from .core_hex_conforming_mesh import HexTriangleSection
from .core_hex_conforming_fem import (
    ConformingCoreState,_triangle_operators,_source_array,
)
from .core_conjugate_fv import CoreSolidLaw,_gas_enthalpy,_gas_secant
from .core_solid_material_tables import TabulatedCoreSolidLaw
from .core_hex_lateral_fv import HexCoolantBoundary
from .core_hex_interelement_mortar import MortarContact


@dataclass(frozen=True)
class ElementPairCoreStep:
    element_a: ConformingCoreState
    element_b: ConformingCoreState
    contact_a_to_b_w: float
    element_a_first_law_residual: float   # W steady; J transient
    element_b_first_law_residual: float
    combined_first_law_residual: float
    iterations: int
    transient: bool


def _local_thermal_system(
    section:HexTriangleSection,
    z:np.ndarray,
    old:ConformingCoreState,
    guess:ConformingCoreState,
    h2:HexCoolantBoundary,
    fuel:CoreSolidLaw|TabulatedCoreSolidLaw,
    coat:CoreSolidLaw|TabulatedCoreSolidLaw,
    power:np.ndarray,
    dt_s:float|None,
):
    """Local 3D Galerkin fuel/coating + implicit 19-channel H2 energy matrix."""
    nz=len(z)-1
    nxy=len(section.xy_m)
    ns=(nz+1)*nxy
    n=ns+nz*19
    q=_source_array(section,z,power)
    tri=np.asarray(section.triangles,dtype=int)
    mat=np.asarray(section.material,dtype=int)
    area,mxy,kxy=_triangle_operators(section.xy_m,tri)
    law=(fuel,coat)
    s=np.asarray(guess.solid_temperature_k,dtype=float)
    s_old=np.asarray(old.solid_temperature_k,dtype=float)
    g=np.asarray(guess.hydrogen_temperature_k,dtype=float)
    g_old=np.asarray(old.hydrogen_temperature_k,dtype=float)
    if (s.shape!=(nz+1,nxy) or s_old.shape!=s.shape
            or g.shape!=(nz,19) or g_old.shape!=g.shape
            or any(not np.isfinite(v).all() or np.any(v<=0)
                   for v in (s,s_old,g,g_old))):
        raise ValueError("invalid element-pair local core temperature state")
    diag=np.zeros(n)
    rhs=np.zeros(n)
    rr=[];cc=[];vv=[]
    m1=np.array([[2.,1.],[1.,2.]])/6.
    k1=np.array([[1.,-1.],[-1.,1.]])
    me=np.array([[2.,1.],[1.,2.]])/6.
    dz=np.diff(z)

    def block(indices:np.ndarray,mat:np.ndarray)->None:
        for a in range(len(indices)):
            for b in range(len(indices)):
                rr.append(int(indices[a]))
                cc.append(int(indices[b]))
                vv.append(float(mat[a,b]))

    for i in range(nz):
        mz=dz[i]*m1
        kz=k1/dz[i]
        for e,vertices in enumerate(tri):
            idx=np.r_[vertices+i*nxy,vertices+(i+1)*nxy]
            medium=law[int(mat[e])]
            temperature=s.reshape(-1)[idx]
            before=s_old.reshape(-1)[idx]
            conduct=medium.conductivity(float(np.mean(temperature)))
            K=conduct*(np.kron(mz,kxy[e])+np.kron(kz,mxy[e]))
            block(idx,K)
            source=float(q[i,e]/6.)
            dvol=area[e]*dz[i]/6.
            for node,prev,now in zip(idx,before,temperature):
                node=int(node)
                rhs[node]+=source
                if dt_s is not None:
                    cap=medium.density_kg_m3*dvol*medium.secant_cp(
                        float(prev),float(now))/dt_s
                    diag[node]+=cap
                    rhs[node]+=cap*prev

        for edge,chan in zip(section.wall_edges,section.wall_channel):
            a,b=map(int,edge)
            c=int(chan)
            idx=np.array(
                (a+i*nxy,b+i*nxy,a+(i+1)*nxy,b+(i+1)*nxy))
            length=float(np.linalg.norm(section.xy_m[b]-section.xy_m[a]))
            W=float(h2.film_w_m2_k[i,c])*length*np.kron(mz,me)
            block(idx,W)
            gas_node=ns+i*19+c
            weights=W.sum(axis=1)
            for j,w in zip(idx,weights):
                rr.extend((int(j),gas_node))
                cc.extend((gas_node,int(j)))
                vv.extend((-float(w),-float(w)))
            diag[gas_node]+=float(W.sum())
        gas_law=h2.gas_boundary()
        for c in range(19):
            index=ns+i*19+c
            rhs[index]+=h2.direct_hydrogen_power_w[i,c]
            if dt_s is not None:
                cap=h2.hydrogen_mass_kg[i,c]*_gas_secant(
                    float(g_old[i,c]),float(g[i,c]),gas_law)/dt_s
                diag[index]+=cap
                rhs[index]+=cap*g_old[i,c]
            inlet=(float(h2.inlet_temperature_k[c]) if i==0
                   else float(g[i-1,c]))
            adv=h2.mdot_kg_s[c]*_gas_secant(
                inlet,float(g[i,c]),gas_law)
            diag[index]+=adv
            if i==0:
                rhs[index]+=adv*h2.inlet_temperature_k[c]
            else:
                rr.append(index)
                cc.append(ns+(i-1)*19+c)
                vv.append(-float(adv))
    ident=np.arange(n,dtype=int)
    rr.extend(ident.tolist())
    cc.extend(ident.tolist())
    vv.extend(diag.tolist())
    return coo_matrix((vv,(rr,cc)),shape=(n,n)).tocsr(),rhs


def _element_storage_and_flow(
    section:HexTriangleSection,
    z:np.ndarray,
    before:ConformingCoreState,
    after:ConformingCoreState,
    h2:HexCoolantBoundary,
    fuel:CoreSolidLaw|TabulatedCoreSolidLaw,
    coating:CoreSolidLaw|TabulatedCoreSolidLaw,
    dt_s:float|None,
)->tuple[float,float,float]:
    """Return stored solid+gas energy J, incoming/outgoing H2 enthalpy W."""
    nz=len(z)-1
    nxy=len(section.xy_m)
    laws=(fuel,coating)
    dE=0.
    if dt_s is not None:
        area=section.triangle_area_m2
        dt=np.diff(z)
        dE=math.fsum(
            laws[int(section.material[e])].density_kg_m3*area[e]*dt[i]/6.*
            laws[int(section.material[e])].enthalpy_change(
                float(before.solid_temperature_k.reshape(-1)[idx]),
                float(after.solid_temperature_k.reshape(-1)[idx]))
            for i in range(nz) for e,vertices in enumerate(section.triangles)
            for idx in np.r_[vertices+i*nxy,vertices+(i+1)*nxy])
        gaslaw=h2.gas_boundary()
        dE+=math.fsum(
            h2.hydrogen_mass_kg[i,c]*(
                _gas_enthalpy(float(after.hydrogen_temperature_k[i,c]),gaslaw)-
                _gas_enthalpy(float(before.hydrogen_temperature_k[i,c]),gaslaw))
            for i in range(nz) for c in range(19))
    gaslaw=h2.gas_boundary()
    incoming=math.fsum(
        h2.mdot_kg_s[c]*
        _gas_enthalpy(float(h2.inlet_temperature_k[c]),gaslaw)
        for c in range(19))
    outgoing=math.fsum(
        h2.mdot_kg_s[c]*
        _gas_enthalpy(float(after.hydrogen_temperature_k[-1,c]),gaslaw)
        for c in range(19))
    return dE,incoming,outgoing


def solve_two_element_conforming_core(
    section_a:HexTriangleSection,
    section_b:HexTriangleSection,
    z_edges_m:np.ndarray,
    previous_a:ConformingCoreState,
    previous_b:ConformingCoreState,
    boundary_a:HexCoolantBoundary,
    boundary_b:HexCoolantBoundary,
    fuel:CoreSolidLaw|TabulatedCoreSolidLaw,
    coating:CoreSolidLaw|TabulatedCoreSolidLaw,
    source_a_w:np.ndarray,
    source_b_w:np.ndarray,
    mortar:MortarContact,
    *,
    dt_s:float|None=None,
    max_iterations:int=70,
    relaxation:float=0.8,
    tolerance_k:float=1e-8,
    first_law_tolerance:float=1e-7,
)->ElementPairCoreStep:
    """Monolithic two-element conduction/contact + 38 H2 streams."""
    section_a.validate();section_b.validate()
    fuel.validate();coating.validate()
    z=np.asarray(z_edges_m,dtype=float)
    if (z.ndim!=1 or z.size<2 or np.any(np.diff(z)<=0)
            or not np.isfinite(z).all()):
        raise ValueError("invalid paired-element axial edges")
    nz=len(z)-1
    boundary_a.validate(nz);boundary_b.validate(nz)
    qa=_source_array(section_a,z,source_a_w)
    qb=_source_array(section_b,z,source_b_w)
    na=(nz+1)*len(section_a.xy_m)
    nb=(nz+1)*len(section_b.xy_m)
    ga=nz*19
    gb=nz*19
    if (mortar.matrix_w_k.shape!=(na+nb,na+nb)
            or mortar.element_a_solid_dofs!=na
            or mortar.element_b_solid_dofs!=nb
            or max_iterations<2 or not 0<relaxation<=1
            or not math.isfinite(tolerance_k) or tolerance_k<=0
            or not math.isfinite(first_law_tolerance)
            or first_law_tolerance<=0
            or dt_s is not None and (not math.isfinite(dt_s) or dt_s<=0)):
        raise ValueError("invalid paired core matrices/time/nonlinear controls")
    # A local unknowns = [A-solid,A-gas]; B = [B-solid,B-gas].
    # Contact mortar unknowns = [A-solid,B-solid].
    m=mortar.matrix_w_k.tocoo()
    mapping=np.r_[np.arange(na),na+ga+np.arange(nb)]
    contact=coo_matrix(
        (m.data,(mapping[m.row],mapping[m.col])),
        shape=(na+ga+nb+gb,na+ga+nb+gb)).tocsr()
    guess_a=previous_a
    guess_b=previous_b
    for iteration in range(1,max_iterations+1):
        Aa,ra=_local_thermal_system(
            section_a,z,previous_a,guess_a,boundary_a,fuel,coating,qa,dt_s)
        Ab,rb=_local_thermal_system(
            section_b,z,previous_b,guess_b,boundary_b,fuel,coating,qb,dt_s)
        A=block_diag((Aa,Ab),format="csr")+contact
        result=np.asarray(spsolve(A,np.r_[ra,rb]),dtype=float)
        if not np.isfinite(result).all() or np.min(result)<=0:
            raise RuntimeError("nonphysical 2-element coupled temperatures")
        trial_a=ConformingCoreState(
            result[:na].reshape((nz+1,len(section_a.xy_m))),
            result[na:na+ga].reshape((nz,19)))
        trial_b=ConformingCoreState(
            result[na+ga:na+ga+nb].reshape((nz+1,len(section_b.xy_m))),
            result[na+ga+nb:].reshape((nz,19)))
        err=max(
            float(np.max(np.abs(trial_a.solid_temperature_k-
                                guess_a.solid_temperature_k))),
            float(np.max(np.abs(trial_a.hydrogen_temperature_k-
                                guess_a.hydrogen_temperature_k))),
            float(np.max(np.abs(trial_b.solid_temperature_k-
                                guess_b.solid_temperature_k))),
            float(np.max(np.abs(trial_b.hydrogen_temperature_k-
                                guess_b.hydrogen_temperature_k))))
        r=relaxation
        guess_a=ConformingCoreState(
            r*trial_a.solid_temperature_k+(1-r)*guess_a.solid_temperature_k,
            r*trial_a.hydrogen_temperature_k+(1-r)*guess_a.hydrogen_temperature_k)
        guess_b=ConformingCoreState(
            r*trial_b.solid_temperature_k+(1-r)*guess_b.solid_temperature_k,
            r*trial_b.hydrogen_temperature_k+(1-r)*guess_b.hydrogen_temperature_k)
        if err<tolerance_k:
            break
    else:
        raise RuntimeError("two-element thermal contact Picard did not converge")
    qcontact=mortar.heat_leaving_element_a_w(
        guess_a.solid_temperature_k,guess_b.solid_temperature_k)
    stored_a,hin_a,hout_a=_element_storage_and_flow(
        section_a,z,previous_a,guess_a,boundary_a,fuel,coating,dt_s)
    stored_b,hin_b,hout_b=_element_storage_and_flow(
        section_b,z,previous_b,guess_b,boundary_b,fuel,coating,dt_s)
    qa_tot=float(np.sum(qa)+np.sum(boundary_a.direct_hydrogen_power_w))
    qb_tot=float(np.sum(qb)+np.sum(boundary_b.direct_hydrogen_power_w))
    delta=dt_s if dt_s is not None else 1.
    residual_a=stored_a-(qa_tot+hin_a-hout_a-qcontact)*delta
    residual_b=stored_b-(qb_tot+hin_b-hout_b+qcontact)*delta
    scale=max(1.,abs(qa_tot)+abs(qb_tot),
              abs(hin_a-hout_a)+abs(hin_b-hout_b),abs(qcontact))*delta
    if max(abs(residual_a),abs(residual_b))>first_law_tolerance*scale+1e-5:
        raise RuntimeError(
            "conservative two-element core first-law did not close: "
            f"A={residual_a:g}, B={residual_b:g}")
    return ElementPairCoreStep(
        guess_a,guess_b,qcontact,residual_a,residual_b,
        residual_a+residual_b,iteration,dt_s is not None)
