"""Numerical code-smoke: two distinct conforming 3D fuel elements and 38 H2.

Element A has one exposed contact edge; element B's physically identical
side is split into two edges. The GLOBAL monolithic solver conserves heat
flow across their unmatched finite-element interfaces by mortar assembly.

These are deliberately simple synthetic materials, not NASA NERVA benchmark
values. More reactor elements/tie tubes and physical verification are later.
"""
from __future__ import annotations

import numpy as np
from .validate_core_hex_conforming_fem import toy_section
from .validate_core_hex_interelement_mortar import split_one_outer_edge
from .core_hex_interelement_mortar import assemble_two_element_contact_mortar
from .core_hex_conforming_fem import ConformingCoreState
from .core_hex_two_element_fem import solve_two_element_conforming_core
from .core_hex_lateral_fv import HexCoolantBoundary
from .core_conjugate_fv import CoreSolidLaw


def main()->int:
    a=toy_section()
    b=split_one_outer_edge(a)
    z=np.array([0.,.1,.2])
    contact=assemble_two_element_contact_mortar(
        a,b,z,origin_b_m=(.8,0.),
        normal_a_to_b=(1.,0.),
        contact_conductance_w_m2_k=80.)
    sa=ConformingCoreState(
        np.full((3,len(a.xy_m)),650.),np.full((2,19),650.))
    sb=ConformingCoreState(
        np.full((3,len(b.xy_m)),500.),np.full((2,19),500.))
    ba=HexCoolantBoundary(
        np.full(19,.01),np.full(19,650.),
        np.full((2,19),.002),np.full((2,19),1200.),
        np.zeros((2,19)),hydrogen_model="constant")
    bb=HexCoolantBoundary(
        np.full(19,.01),np.full(19,500.),
        np.full((2,19),.002),np.full((2,19),1200.),
        np.zeros((2,19)),hydrogen_model="constant")
    fuel=CoreSolidLaw(1850.,1250.,23.)
    coating=CoreSolidLaw(6400.,650.,19.)
    qa=np.zeros((2,len(a.triangles)))
    qa[:,a.material==0]=1.
    qb=np.zeros((2,len(b.triangles)))
    steady=solve_two_element_conforming_core(
        a,b,z,sa,sb,ba,bb,fuel,coating,qa,qb,contact)
    assert steady.iterations>=2
    assert steady.contact_a_to_b_w>1.
    assert abs(steady.combined_first_law_residual)<1e-5
    assert abs(steady.element_a_first_law_residual)<1e-5
    assert abs(steady.element_b_first_law_residual)<1e-5
    assert np.max(steady.element_a.solid_temperature_k)>np.min(
        steady.element_b.solid_temperature_k)
    transient=solve_two_element_conforming_core(
        a,b,z,sa,sb,ba,bb,fuel,coating,qa,qb,contact,dt_s=.2)
    assert transient.contact_a_to_b_w>0.
    assert abs(transient.combined_first_law_residual)<1e-5
    assert abs(transient.element_a_first_law_residual)<1e-5
    assert abs(transient.element_b_first_law_residual)<1e-5

    # Exactly zero mortar conductivity matches two isolated local
    # thermal solutions and gives no net heat transferred by contact.
    disconnected=assemble_two_element_contact_mortar(
        a,b,z,origin_b_m=(.8,0.),
        normal_a_to_b=(1.,0.),
        contact_conductance_w_m2_k=0.)
    alone=solve_two_element_conforming_core(
        a,b,z,sa,sb,ba,bb,fuel,coating,qa,qb,disconnected)
    assert alone.contact_a_to_b_w==0.
    assert np.max(steady.element_a.solid_temperature_k) < (
        np.max(alone.element_a.solid_temperature_k))
    print("Nonmatching two fuel elements + 38 H2 streams monolithic 3D FEM: PASS")
    print("Interface area m2:",contact.interface_area_m2)
    print("Steady A->B heat W:",steady.contact_a_to_b_w)
    print("Combined steady energy residual W:",steady.combined_first_law_residual)
    print("Combined transient energy residual J:",transient.combined_first_law_residual)
    print("Historical full-core nuclear/thermal validation: DEFERRED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
