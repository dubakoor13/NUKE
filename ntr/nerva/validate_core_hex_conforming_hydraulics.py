"""Small executable steady thermal+subsonic momentum coupled-physics check.

One conforming coating/fuel cross-section with a single actively heated
H2 channel and 18 unheated channels. Test verifies the existing NASA-7
H2 stagnation and momentum solver is actually invoked and coupled to
3-D solid conduction, while reserving historical validation for later.
"""
from __future__ import annotations
import numpy as np

from .validate_core_hex_conforming_fem import toy_section
from .core_conjugate_fv import CoreSolidLaw
from .core_hex_conforming_fem import ConformingCoreState
from .core_hex_conforming_hydraulics import solve_conforming_core_with_hydraulics


def main()->int:
    section=toy_section()
    z=np.array([0.,0.025,0.05])
    nz=len(z)-1
    q=np.zeros((nz,len(section.triangles)))
    q[:,section.material==0]=0.35
    start=ConformingCoreState(
        np.full((nz+1,len(section.xy_m)),600.),
        np.full((nz,19),600.))
    sol=solve_conforming_core_with_hydraulics(
        section,z,start,
        CoreSolidLaw(1850.,1250.,23.),
        CoreSolidLaw(6400.,650.,19.),
        q,
        mass_flow_kg_s=np.full(19,0.003),
        inlet_temperature_k=np.full(19,600.),
        inlet_pressure_pa=np.full(19,15.e6),
        direct_hydrogen_power_w=np.zeros((nz,19)),
        max_outer_iterations=32)
    assert sol.iterations>=1
    assert sol.pressure_pa.shape==(nz,19)
    assert np.isfinite(sol.pressure_pa).all()
    assert np.all(sol.pressure_pa<15.e6)
    assert np.all(sol.mach_number>0)
    assert np.all(sol.mach_number<.85)
    assert np.all(sol.film_w_m2_k>0)
    assert sol.thermal_hydraulic_temperature_error_k<=2.e-3
    assert abs(sol.thermal.first_law_residual)<1e-5
    print("3D conforming solid/H2 steady compressible coupling: PASS")
    print("Outer iterations:",sol.iterations)
    print("Maximum thermal-to-momentum gas temperature mismatch K:",
          sol.thermal_hydraulic_temperature_error_k)
    print("Maximum Mach:",float(np.max(sol.mach_number)))
    print("Core operating point physical verification: DEFERRED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
