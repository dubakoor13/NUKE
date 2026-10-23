"""Numerical-only exercise of symmetric nonmatching inter-element contact.

The two synthetic square fuel cross sections have one outer face in common.
Element B subdivides the face into two P1 edges, while element A has one
long edge; the mortar matrix must remain symmetric and conservative.
"""
from __future__ import annotations

import numpy as np

from .validate_core_hex_conforming_fem import toy_section
from .core_hex_conforming_mesh import HexTriangleSection
from .core_hex_interelement_mortar import assemble_two_element_contact_mortar


def split_one_outer_edge(base:HexTriangleSection)->HexTriangleSection:
    # Element B faces A with its LEFT x=-0.4 side; subdivide that
    # physically shared side, not its opposite/right-side perimeter.
    xy=np.vstack((base.xy_m,np.array([[-0.4,0.0]])))
    newnode=len(xy)-1
    tri=[]
    material=[]
    hits=0
    for nodes,mat in zip(base.triangles,base.material):
        if mat==0 and set((11,8)).issubset(set(nodes)):
            v=int(next(k for k in nodes if k not in (11,8)))
            tri.extend(((v,11,newnode),(v,newnode,8)))
            material.extend((mat,mat))
            hits+=1
        else:
            tri.append(tuple(int(k) for k in nodes))
            material.append(int(mat))
    assert hits==1
    new=HexTriangleSection(
        xy,np.asarray(tri,dtype=int),np.asarray(material,dtype=int),
        base.wall_edges.copy(),base.wall_channel.copy())
    new.validate()
    return new


def main()->int:
    a=toy_section()
    b=split_one_outer_edge(a)
    z=np.array([0.,0.09,0.20])
    h=72.5
    contact=assemble_two_element_contact_mortar(
        a,b,z,origin_b_m=(.8,0.),
        normal_a_to_b=(1.,0.),
        contact_conductance_w_m2_k=h)
    assert contact.overlap_pairs==2
    assert np.isclose(contact.overlap_length_m,.8)
    assert np.isclose(contact.interface_area_m2,.16)
    K=contact.matrix_w_k
    assert np.max(np.abs((K-K.T).data))<1e-10 if (K-K.T).nnz else True
    uniform=np.ones(K.shape[0])
    assert np.max(np.abs(K@uniform))<1e-10

    ta=np.full((len(z),len(a.xy_m)),700.)
    tb=np.full((len(z),len(b.xy_m)),500.)
    flux=contact.heat_leaving_element_a_w(ta,tb)
    assert np.isclose(flux,h*.16*200.,rtol=1e-12)
    assert np.isclose(
        flux+contact.heat_leaving_element_b_w(ta,tb),0.,atol=1e-10)

    # Equal global linear temperature field has no physical interface
    # flux even when there is a nonmatching T-shape discretization.
    ta=np.array([
        620.+4*z0+50*a.xy_m[:,1] for z0 in z])
    tb=np.array([
        620.+4*z0+50*b.xy_m[:,1] for z0 in z])
    t=np.r_[ta.reshape(-1),tb.reshape(-1)]
    assert np.max(np.abs(K@t))<1e-9
    assert np.isclose(float(t@K@t),0.,atol=2e-6)
    # The discrete contact energy is always nonnegative.
    t=np.r_[np.full(ta.size,700.),np.full(tb.size,500.)]
    assert t@K@t>0.
    print("Nonmatching two-fuel-element mortar interface operator: PASS")
    print("Overlapping P1 face pairs:",contact.overlap_pairs)
    print("Coupled face area m2:",contact.interface_area_m2)
    print("Uniform 200 K face-to-face transfer W:",flux)
    print("Reactor array / NASA physical qualification: DEFERRED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
