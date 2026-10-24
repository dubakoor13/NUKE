"""Lightweight executable numerical smoke test for the conforming 3-D thermal FEM.

This deliberately uses a *square annular example*, not a NERVA physical
benchmark. It only proves that mixed fuel/coating triangles, shared wall
nodes, axial solid coupling, hydrogen enthalpy and global first-law
assembly are executable before later physical validation.
"""
from __future__ import annotations
import numpy as np
from .core_hex_conforming_mesh import HexTriangleSection
from .core_hex_conforming_fem import (
    ConformingCoreState,solve_conforming_core_thermal,
)
from .core_hex_lateral_fv import HexCoolantBoundary
from .core_conjugate_fv import CoreSolidLaw
from .core_hex_conforming_power import (
    CartesianSolidPower,project_cartesian_deposition_to_prisms,
)


def toy_section()->HexTriangleSection:
    """Three concentric nested squares, with a conforming coating annulus."""
    nodes=[]
    for r in (0.1,0.2,0.4):
        nodes.extend([(-r,-r),(r,-r),(r,r),(-r,r)])
    tri=[];mat=[]
    for inner,outer,kind in ((0,4,1),(4,8,0)):
        for j in range(4):
            jj=(j+1)%4
            tri.extend([
                [inner+j,outer+j,outer+jj],
                [inner+j,outer+jj,inner+jj],
            ])
            mat.extend([kind,kind])
    boundary=np.array([[i,(i+1)%4] for i in range(4)],dtype=int)
    mesh=HexTriangleSection(
        np.array(nodes,float),np.array(tri,dtype=int),
        np.array(mat,dtype=int),boundary,np.zeros(4,dtype=int))
    mesh.validate()
    return mesh


def main()->int:
    section=toy_section()
    z=np.array([0.,0.10,0.20])
    nz=len(z)-1
    # Exact material-separated voxel->prism integration: each uniform
    # material power density must equal its triangle volume times W/cm3.
    source=CartesianSolidPower(
        x_edges_cm=np.array([-50.,0.,50.]),
        y_edges_cm=np.array([-50.,0.,50.]),
        z_edges_cm=np.array([0.,5.,15.,20.]),
        fuel_density_w_cm3=np.full((2,2,3),2.),
        coating_density_w_cm3=np.full((2,2,3),3.))
    projection=project_cartesian_deposition_to_prisms(section,z,source)
    area=section.triangle_area_m2
    for i in range(nz):
        expected=(area*1e6*(z[i+1]-z[i]) *
                  np.where(section.material==0,2.,3.))
        assert np.allclose(projection.solid_prism_power_w[i],expected,
                           rtol=1e-10,atol=1e-6)
    q=np.zeros((nz,len(section.triangles)))
    q[:,section.material==0]=2.0
    initial=ConformingCoreState(
        np.full((nz+1,len(section.xy_m)),500.),
        np.full((nz,19),500.))
    gas=HexCoolantBoundary(
        np.full(19,0.005),np.full(19,500.),
        np.full((nz,19),0.001),
        np.full((nz,19),1200.),
        np.full((nz,19),0.1),
        hydrogen_model="constant")
    fuel=CoreSolidLaw(1800.,1000.,25.)
    coat=CoreSolidLaw(6400.,650.,19.)
    steady=solve_conforming_core_thermal(
        section,z,initial,gas,fuel,coat,q,tolerance_k=1e-9)
    assert steady.iterations>=2
    assert abs(steady.first_law_residual)<1e-5
    assert np.sum(steady.wall_to_hydrogen_w)>0
    assert np.max(steady.state.solid_temperature_k)>500.
    transient=solve_conforming_core_thermal(
        section,z,initial,gas,fuel,coat,q,
        dt_s=0.1,tolerance_k=1e-9)
    assert transient.stored_solid_j>0
    assert transient.stored_hydrogen_j>0
    assert abs(transient.first_law_residual)<1e-5
    # Zero-source invariant on conforming fuel/coating interfaces.
    zero=HexCoolantBoundary(
        gas.mdot_kg_s,gas.inlet_temperature_k,
        gas.hydrogen_mass_kg,gas.film_w_m2_k,
        np.zeros((nz,19)),hydrogen_model="constant")
    equilibrium=solve_conforming_core_thermal(
        section,z,initial,zero,fuel,coat,np.zeros_like(q),
        dt_s=0.1)
    assert np.max(np.abs(
        equilibrium.state.solid_temperature_k-500.))<1e-6
    assert np.max(np.abs(
        equilibrium.state.hydrogen_temperature_k-500.))<1e-6
    print("Conforming prism 3-D core thermal FEM assembly: PASS")
    print("solid/prism triangles:",len(section.triangles))
    print("steady first law W:",steady.first_law_residual)
    print("transient first law J:",transient.first_law_residual)
    print("NASA physical qualification: NOT PERFORMED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
