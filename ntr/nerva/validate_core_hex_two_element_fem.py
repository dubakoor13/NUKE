"""Numerical code-smoke: two distinct conforming 3D fuel elements and 38 H2.

Element A has one exposed contact edge; element B's physically identical
side is split into two edges. The GLOBAL monolithic solver conserves heat
flow across their unmatched finite-element interfaces by mortar assembly.

These are deliberately simple synthetic materials, not NASA NERVA benchmark
values. More reactor elements/tie tubes and physical verification are later.
"""
from __future__ import annotations

import numpy as np
import json
import sys
import tempfile
from pathlib import Path
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
        a,b,z,sa,sb,ba,bb,fuel,coating,qa,qb,contact,
        dt_s=.2,tolerance_k=1e-11)
    assert transient.contact_a_to_b_w>0.
    assert abs(transient.combined_first_law_residual)<1e-5
    assert abs(transient.element_a_first_law_residual)<1e-5, (
        transient.element_a_first_law_residual)
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
    # Runnable application path: cached meshes + distinct source prism
    # arrays + actual NUKE [element,19,z] flow/heating NPZ.
    from .run_core_hex_conforming_fem import save_section
    from .run_core_hex_two_element_fem import main as run_pair_cli
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        mesh_a=root/"a.npz";mesh_b=root/"b.npz"
        data=root/"reconstruction.npz"
        power_a=root/"power_a.npz";power_b=root/"power_b.npz"
        output=root/"results"
        save_section(mesh_a,a);save_section(mesh_b,b)
        np.savez_compressed(
            data,
            channel_wall_axial_power_w=np.zeros((2,19,2)),
            channel_direct_h2_axial_power_w=np.zeros((2,19,2)),
            branch_mass_flow_kg_s=np.full((2,19),0.01),
            z_edges_m=z)
        np.savez_compressed(power_a,solid_prism_power_w=qa)
        np.savez_compressed(power_b,solid_prism_power_w=qb)
        old_argv=sys.argv
        try:
            sys.argv=[
                "run_core_hex_two_element_fem",str(data),
                "--mesh-a",str(mesh_a),"--mesh-b",str(mesh_b),
                "--element-a-index","0","--element-b-index","1",
                "--origin-b-x-m","0.8","--origin-b-y-m","0",
                "--normal-x","1","--normal-y","0",
                "--contact-conductance-w-m2-k","80",
                "--power-a-npz",str(power_a),
                "--power-b-npz",str(power_b),
                "--inlet-temperature-a-k","650",
                "--inlet-temperature-b-k","500",
                "--initial-temperature-a-k","650",
                "--initial-temperature-b-k","500",
                "--gas-density-kg-m3","3",
                "--gas-film-w-m2-k","1200",
                "--bore-diameter-m","0.002",
                "--hydrogen-model","constant",
                "--fuel-density-kg-m3","1850",
                "--fuel-cp-j-kg-k","1250",
                "--fuel-conductivity-w-m-k","23",
                "--coating-density-kg-m3","6400",
                "--coating-cp-j-kg-k","650",
                "--coating-conductivity-w-m-k","19",
                "--output",str(output)]
            assert run_pair_cli()==0
        finally:
            sys.argv=old_argv
        report=json.loads((output/"two_element_conforming_report.json").read_text())
        assert report["contact_overlap_pairs"]==2
        assert report["steps"][0]["contact_heat_a_to_b_w"]>0.
        with np.load(output/"two_element_conforming_fields.npz") as fields:
            assert fields["hydrogen_a_temperature_k"].shape==(2,19)
            assert fields["hydrogen_b_temperature_k"].shape==(2,19)
    print("Nonmatching two fuel elements + 38 H2 streams monolithic 3D FEM: PASS")
    print("Interface area m2:",contact.interface_area_m2)
    print("Steady A->B heat W:",steady.contact_a_to_b_w)
    print("Combined steady energy residual W:",steady.combined_first_law_residual)
    print("Combined transient energy residual J:",transient.combined_first_law_residual)
    print("Historical full-core nuclear/thermal validation: DEFERRED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
