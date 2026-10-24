"""Conservative nonmatching-mesh thermal contact between two 3D fuel elements.

The discrete interface energy is

    E_contact = 1/2 * integral_interface H (T_A - T_B)^2 dA,

with H>=0 the prescribed inter-element contact conductance [W/m2/K].
Its Hessian is a SYMMETRIC positive-semidefinite stiffness matrix with
no external bath and no lost contact heat. Equal/opposite fluxes and
energy conservation hold by construction, even if the two separate
conforming fuel-element meshes have different perimeter edge divisions.

This module builds an inter-element MORTAR OPERATOR to add to a future
global multi-element FE thermal matrix. It does not itself solve a
reactor-wide set of elements, neutron transport, or determine contact H.
It supports shared flat faces with arbitrary in-plane unit normals and
element offsets, a common axial grid, and a fully overlapping interface.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .core_hex_conforming_mesh import HexTriangleSection
from .core_hex_exterior_surface import exterior_fuel_edges

_GAUSS = (-1/math.sqrt(3),1/math.sqrt(3))


@dataclass(frozen=True)
class MortarContact:
    """Physical contact assembled on stacked [A solid, B solid] nodal states."""
    matrix_w_k: csr_matrix
    overlap_length_m: float
    interface_area_m2: float
    overlap_pairs: int
    element_a_solid_dofs: int
    element_b_solid_dofs: int

    def heat_leaving_element_a_w(self,
                                 temperature_a_k:np.ndarray,
                                 temperature_b_k:np.ndarray)->float:
        """Signed heat leaving A and entering B (conservative, watts)."""
        a=np.asarray(temperature_a_k,dtype=float).reshape(-1)
        b=np.asarray(temperature_b_k,dtype=float).reshape(-1)
        if (a.size!=self.element_a_solid_dofs or
                b.size!=self.element_b_solid_dofs
                or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError("incorrect contact FE temperature dimensions")
        residual=self.matrix_w_k@np.r_[a,b]
        return float(np.sum(residual[:len(a)]))

    def heat_leaving_element_b_w(self,temperature_a_k:np.ndarray,
                                 temperature_b_k:np.ndarray)->float:
        """Signed heat leaving B and entering A, exactly minus A flux."""
        return -self.heat_leaving_element_a_w(temperature_a_k,temperature_b_k)


def _side(section:HexTriangleSection,offset:np.ndarray,
          normal:np.ndarray,sign:float,tolerance_m:float
          )->tuple[list[tuple[np.ndarray,float,float]],float]:
    outer=exterior_fuel_edges(section)
    xy=np.asarray(section.xy_m,dtype=float)+offset[None,:]
    projection=xy@normal
    plane=(float(np.max(projection)) if sign>0 else float(np.min(projection)))
    tangent=np.array((-normal[1],normal[0]))
    selected=[]
    for edge in outer:
        values=projection[edge]
        if np.max(np.abs(values-plane))>tolerance_m:
            continue
        along=xy[edge]@tangent
        low=float(min(along));high=float(max(along))
        if high-low>tolerance_m:
            selected.append((edge,low,high))
    if not selected:
        raise ValueError("requested normal does not identify a flat fuel side")
    return selected,plane


def assemble_two_element_contact_mortar(
    section_a:HexTriangleSection,
    section_b:HexTriangleSection,
    z_edges_m:np.ndarray,
    *,
    origin_a_m:tuple[float,float]|np.ndarray=(0.,0.),
    origin_b_m:tuple[float,float]|np.ndarray,
    normal_a_to_b:tuple[float,float]|np.ndarray,
    contact_conductance_w_m2_k:float,
    alignment_tolerance_m:float=1e-8,
)->MortarContact:
    """Add equal-opposite nonmatching P1 side-contact conduction.

    Stacks A solid nodes [z,node] then B solid nodes [z,node]. All edge
    intersections are integrated by tensor-product 2x2 Gauss, exact for
    P1(y)*P1(z) temperature basis products on each overlap patch.
    The chosen normal MUST point from A toward B, and A's maximal plane
    along it must coincide with B's minimal plane; holes never appear
    among the eligible perimeter edges.
    """
    section_a.validate()
    section_b.validate()
    z=np.asarray(z_edges_m,dtype=float)
    if z.ndim!=1 or z.size<2 or not np.isfinite(z).all() or np.any(np.diff(z)<=0):
        raise ValueError("contact axial grid must be increasing")
    na=len(section_a.xy_m)
    nb=len(section_b.xy_m)
    origins=(np.asarray(origin_a_m,dtype=float),
             np.asarray(origin_b_m,dtype=float))
    n=np.asarray(normal_a_to_b,dtype=float)
    if (any(o.shape!=(2,) or not np.isfinite(o).all() for o in origins)
            or n.shape!=(2,) or not np.isfinite(n).all()
            or not math.isfinite(float(np.linalg.norm(n)))
            or np.linalg.norm(n)<1e-12
            or not math.isfinite(contact_conductance_w_m2_k)
            or contact_conductance_w_m2_k<0
            or not math.isfinite(alignment_tolerance_m)
            or alignment_tolerance_m<=0):
        raise ValueError("invalid two-element mortar geometry/contact law")
    n=n/np.linalg.norm(n)
    a,plane_a=_side(section_a,origins[0],n,+1,alignment_tolerance_m)
    b,plane_b=_side(section_b,origins[1],n,-1,alignment_tolerance_m)
    if abs(plane_a-plane_b)>alignment_tolerance_m:
        raise ValueError(
            "neighboring fuel faces do not coincide: explicit gap model required")
    length_a=math.fsum(u-l for _,l,u in a)
    length_b=math.fsum(u-l for _,l,u in b)
    nplane=len(z)
    na3=nplane*na
    nb3=nplane*nb
    total=na3+nb3
    row=[];col=[];data=[]
    overlap=0.
    npairs=0
    h=float(contact_conductance_w_m2_k)

    def interval_shapes(low:float,high:float,x:float)->np.ndarray:
        # The physical edge endpoints can be oriented backwards relative to
        # the common tangent. Shape values must match original connectivity.
        if high<=low:
            raise ValueError("zero-length 1D contact interval")
        return np.array([(high-x)/(high-low),(x-low)/(high-low)])

    # Edges in a/b refer to their ORIGINAL vertex order, so evaluate shape
    # coordinates using the actual along-tangent projection at endpoints.
    tangent=np.array([-n[1],n[0]])
    xy_a=np.asarray(section_a.xy_m)+origins[0]
    xy_b=np.asarray(section_b.xy_m)+origins[1]
    dz=np.diff(z)
    for ea,alo,ahi in a:
        sa=xy_a[ea]@tangent
        for eb,blo,bhi in b:
            low=max(alo,blo)
            high=min(ahi,bhi)
            if high<=low:
                continue
            eb0=xy_b[eb]@tangent
            npairs+=1
            overlap+=high-low
            for iz,height in enumerate(dz):
                indices=np.array([
                    int(ea[0])+iz*na,int(ea[1])+iz*na,
                    int(ea[0])+(iz+1)*na,int(ea[1])+(iz+1)*na,
                    na3+int(eb[0])+iz*nb,na3+int(eb[1])+iz*nb,
                    na3+int(eb[0])+(iz+1)*nb,
                    na3+int(eb[1])+(iz+1)*nb],dtype=int)
                element=np.zeros((8,8),dtype=float)
                for gy in _GAUSS:
                    s=.5*(low+high)+.5*(high-low)*gy
                    # Parametric FE shape on each separately meshed side.
                    wa=np.array([(sa[1]-s)/(sa[1]-sa[0]),
                                 (s-sa[0])/(sa[1]-sa[0])])
                    wb=np.array([(eb0[1]-s)/(eb0[1]-eb0[0]),
                                 (s-eb0[0])/(eb0[1]-eb0[0])])
                    for gz in _GAUSS:
                        vz=np.array([.5*(1-gz),.5*(1+gz)])
                        phi=np.r_[np.kron(vz,wa),-np.kron(vz,wb)]
                        # 2x2 Gaussian area integration exact for
                        # quadratic products in both y and z.
                        element+=h*(high-low)*height*.25*np.outer(phi,phi)
                rr,cc=np.indices((8,8))
                row.extend(indices[rr.reshape(-1)].tolist())
                col.extend(indices[cc.reshape(-1)].tolist())
                data.extend(element.reshape(-1).tolist())
    if (abs(overlap-length_a)>max(1e-11,alignment_tolerance_m*len(a))
            or abs(overlap-length_b)>max(1e-11,alignment_tolerance_m*len(b))
            or npairs==0):
        raise ValueError("partial/nonmatching external side overlap unsupported")
    matrix=coo_matrix(
        (data,(row,col)),shape=(total,total)).tocsr()
    return MortarContact(
        matrix_w_k=matrix,
        overlap_length_m=overlap,
        interface_area_m2=overlap*(z[-1]-z[0]),
        overlap_pairs=npairs,
        element_a_solid_dofs=na3,
        element_b_solid_dofs=nb3,
    )
