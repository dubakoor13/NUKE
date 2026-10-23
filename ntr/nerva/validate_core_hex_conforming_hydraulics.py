"""Small executable steady thermal+subsonic momentum coupled-physics check.

One conforming coating/fuel cross-section with a single actively heated
H2 channel and 18 unheated channels. Test verifies the existing NASA-7
H2 stagnation and momentum solver is actually invoked and coupled to
3-D solid conduction, while reserving historical validation for later.
"""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
import sys
import tempfile

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
    # Exercise real CLI wiring: imported cached 3-D triangle mesh,
    # explicit per-prism source, tabulated fuel k(T)/cp(T), and 19-channel
    # compressible pressure/film calculation in one execution.
    from .run_core_hex_conforming_fem import main as run_cli, save_section
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        mesh_file=root/"toy_mesh.npz"
        source_file=root/"detailed.npz"
        input_file=root/"channel_reconstruction.npz"
        table_file=root/"fuel_table.npz"
        output=root/"results"
        save_section(mesh_file,section)
        np.savez_compressed(source_file,solid_prism_power_w=q[None,:,:])
        np.savez_compressed(
            input_file,
            channel_wall_axial_power_w=np.zeros((1,19,nz)),
            channel_direct_h2_axial_power_w=np.zeros((1,19,nz)),
            branch_mass_flow_kg_s=np.full((1,19),.003),
            z_edges_m=z)
        np.savez_compressed(
            table_file,
            temperature_k=np.array([200.,500.,1500.,3500.]),
            conductivity_w_m_k=np.array([23.,23.,24.,24.]),
            cp_j_kg_k=np.array([1250.,1250.,1300.,1300.]))
        arguments=[
            "run_core_hex_conforming_fem",str(input_file),
            "--mesh-cache",str(mesh_file),
            "--detailed-source",str(source_file),
            "--element-index","0",
            "--inlet-temperature-k","600",
            "--initial-temperature-k","600",
            "--hydrogen-density-kg-m3","3",
            "--film-w-m2-k","1200",
            "--hydrogen-model","nasa7-h2",
            "--couple-hydraulics",
            "--inlet-pressure-pa","15000000",
            "--fuel-density-kg-m3","1850",
            "--fuel-material-npz",str(table_file),
            "--coating-density-kg-m3","6400",
            "--coating-cp-j-kg-k","650",
            "--coating-conductivity-w-m-k","19",
            "--output",str(output),
        ]
        previous_argv=sys.argv
        try:
            sys.argv=arguments
            assert run_cli()==0
        finally:
            sys.argv=previous_argv
        with np.load(output/"conforming_core_hydraulics.npz") as data:
            assert data["pressure_pa"].shape==(1,nz,19)
            assert np.isfinite(data["mach_number"]).all()
        report=json.loads((output/"conforming_core_3d_report.json").read_text())
        assert report["fuel_material_model"]=="tabulated"
        assert report["hydraulic_model"]=="quasi-steady subsonic NASA7 frozen H2"
        assert report["results"][0]["steps"][0]["hydraulic_outer_iterations"]>=1
    print("3D conforming solid/H2 steady compressible coupling: PASS")
    print("Outer iterations:",sol.iterations)
    print("Maximum thermal-to-momentum gas temperature mismatch K:",
          sol.thermal_hydraulic_temperature_error_k)
    print("Maximum Mach:",float(np.max(sol.mach_number)))
    print("Core operating point physical verification: DEFERRED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
