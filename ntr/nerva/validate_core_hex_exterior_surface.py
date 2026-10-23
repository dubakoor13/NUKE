"""Executable code-smoke for 3-D exterior thermal contact and greybody loss.

Synthetic P1 geometry and prescribed heat transfer only; NERVA outer
coating/contact measurements and hardware comparison intentionally deferred.
"""
from __future__ import annotations
import numpy as np

from .validate_core_hex_conforming_fem import toy_section
from .core_hex_exterior_surface import (
    CoreExteriorHeatBoundary,exterior_fuel_edges,
    exterior_boundary_power,STEFAN_BOLTZMANN_W_M2_K4,
)
from .core_hex_conforming_fem import (
    ConformingCoreState,solve_conforming_core_thermal,
)
from .core_hex_lateral_fv import HexCoolantBoundary
from .core_conjugate_fv import CoreSolidLaw


def main()->int:
    section=toy_section()
    outer=exterior_fuel_edges(section)
    assert len(outer)==4,outer.shape
    z=np.array([0.,0.10,0.20])
    ns=3; nz=len(z)-1; nxy=len(section.xy_m)
    solid=ConformingCoreState(
        np.full((ns,nxy),600.),np.full((nz,19),600.))
    lengths=np.linalg.norm(
        section.xy_m[outer[:,1]]-section.xy_m[outer[:,0]],axis=1)
    area=np.sum(lengths)*np.diff(z).sum()
    bath=CoreExteriorHeatBoundary(
        contact_conductance_w_m2_k=7.5,
        contact_reservoir_temperature_k=500.,
        emissivity=0.65,
        radiation_environment_temperature_k=450.)
    # Uniform nodal temperature: energy flux must reproduce exact
    # Fourier Robin + Stefan-Boltzmann power times exposed area.
    P,net=exterior_boundary_power(solid.solid_temperature_k,z,section,bath)
    expected=area*(
        7.5*(600.-500.)+
        0.65*STEFAN_BOLTZMANN_W_M2_K4*(600.**4-450.**4))
    assert P.shape==(nz,4)
    assert np.isclose(net,expected,rtol=1e-12)
    adiabatic=CoreExteriorHeatBoundary()
    _,zero=exterior_boundary_power(
        solid.solid_temperature_k,z,section,adiabatic)
    assert zero==0.
    wall=HexCoolantBoundary(
        np.full(19,.005),np.full(19,600.),
        np.full((nz,19),.001),np.full((nz,19),1200.),
        np.zeros((nz,19)),hydrogen_model="constant")
    fuel=CoreSolidLaw(1850.,1250.,23.)
    coat=CoreSolidLaw(6400.,650.,19.)
    heated=solve_conforming_core_thermal(
        section,z,solid,wall,fuel,coat,
        np.zeros((nz,len(section.triangles))),dt_s=.20,
        exterior_boundary=bath,tolerance_k=1e-9)
    assert heated.exterior_heat_loss_w>0
    assert heated.stored_solid_j<0
    assert abs(heated.first_law_residual)<1e-4
    assert np.max(heated.state.solid_temperature_k)<600.001
    # Zero exchange collapses exactly to the original adiabatic solution.
    old=solve_conforming_core_thermal(
        section,z,solid,wall,fuel,coat,
        np.zeros((nz,len(section.triangles))),dt_s=.20)
    same=solve_conforming_core_thermal(
        section,z,solid,wall,fuel,coat,
        np.zeros((nz,len(section.triangles))),dt_s=.20,
        exterior_boundary=adiabatic)
    assert np.allclose(
        old.state.solid_temperature_k,same.state.solid_temperature_k,
        rtol=0,atol=1e-7)
    for bad in (
        CoreExteriorHeatBoundary(-1.),
        CoreExteriorHeatBoundary(emissivity=1.1),
        CoreExteriorHeatBoundary(contact_reservoir_temperature_k=0.)):
        try:
            bad.expanded(nz,len(outer))
        except ValueError:
            pass
        else:
            raise AssertionError("bad exterior boundary not rejected")
    print("Conforming core outer contact + radiation FEM physics: PASS")
    print("Exposed outer edges:",len(outer))
    print("Stefan-Boltzmann & bath outward heat W:",net)
    print("Transient first-law residual J:",heated.first_law_residual)
    print("Physical NERVA contact/emissivity qualification: DEFERRED")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
