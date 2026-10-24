"""Code smoke for property-table exact energy and conforming FEM integration.

Synthetic numerical material functions only. No qualified NERVA property
tables are bundled, and this is not experimental material validation.
"""
from __future__ import annotations

from pathlib import Path
import tempfile
import math
import numpy as np

from .core_solid_material_tables import TabulatedCoreSolidLaw
from .core_hex_conforming_fem import (
    ConformingCoreState, solve_conforming_core_thermal,
)
from .core_hex_lateral_fv import HexCoolantBoundary
from .validate_core_hex_conforming_fem import toy_section


def main()->int:
    t=np.array([200.,500.,800.,1600.,3500.])
    k=np.array([12.,15.,18.,26.,45.])
    cp=1000.+(t-500.)     # synthetic cp(T)=1000 + (T-500) SI
    law=TabulatedCoreSolidLaw(1900.,t,k,cp)
    law.validate()
    assert math.isclose(law.cp(650.),1150.,rel_tol=1e-12)
    assert math.isclose(law.conductivity(650.),16.5,rel_tol=1e-12)
    for a,b in ((500.,800.),(350.,1250.),(1700.,2100.)):
        expected=1000.*(b-a)+0.5*((b-500.)**2-(a-500.)**2)
        assert math.isclose(law.enthalpy_change(a,b),expected,rel_tol=1e-12)
        assert math.isclose(law.enthalpy_change(b,a),-expected,rel_tol=1e-12)
        assert math.isclose(law.secant_cp(a,b),expected/(b-a),rel_tol=1e-12)
    assert law.secant_cp(700.,700.)==law.cp(700.)
    for outside in (199.,3501.,float("nan")):
        try:
            law.cp(outside)
        except ValueError:
            pass
        else:
            raise AssertionError("material-table extrapolation was allowed")
    with tempfile.TemporaryDirectory() as temp:
        path=Path(temp)/"solid.npz"
        np.savez_compressed(path,temperature_k=t,
                            conductivity_w_m_k=k,cp_j_kg_k=cp)
        loaded=TabulatedCoreSolidLaw.from_npz(path,density_kg_m3=1900.)
        assert loaded.enthalpy_change(500.,800.)==law.enthalpy_change(500.,800.)
    mesh=toy_section()
    z=np.array([0.,0.10,0.20])
    q=np.zeros((2,len(mesh.triangles)))
    q[:,mesh.material==0]=2.
    zero=ConformingCoreState(
        np.full((3,len(mesh.xy_m)),500.),np.full((2,19),500.))
    boundary=HexCoolantBoundary(
        np.full(19,.005),np.full(19,500.),
        np.full((2,19),0.001),np.full((2,19),1200.),
        np.full((2,19),0.1),hydrogen_model="constant")
    from .core_conjugate_fv import CoreSolidLaw
    coat=CoreSolidLaw(6400.,650.,19.)
    r=solve_conforming_core_thermal(
        mesh,z,zero,boundary,law,coat,q,
        dt_s=.1,tolerance_k=1e-9)
    assert r.stored_solid_j>0
    assert abs(r.first_law_residual)<1e-5
    print("Tabulated solid k(T), exact cp integral, 3D FEM coupling: PASS")
    print("Conforming FEM tabulated-material transient energy residual J:",
          r.first_law_residual)
    print("Historical NERVA tabulated properties: NOT SUPPLIED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
