"""Deterministic numerical verification of NERVA 19-bore lateral FV physics.

No source calibration, criticality, nuclear cross sections or NASA benchmarks:
these tests exercise conservative Fourier flux, gas enthalpy and geometry.
"""
from __future__ import annotations

import math
import numpy as np

from .core_conjugate_fv import CoreSolidLaw
from .core_hex_lateral_balance import audit_hex_core_cell_energy
from .core_hex_lateral_fv import (
    HexSectorNetwork,HexCoolantBoundary,HexCoreState,
    solve_hex_core_conjugate,
)


def main()->int:
    mesh=HexSectorNetwork.from_nerva_config()
    assert mesh.fuel_area_m2.shape==(19,)
    assert mesh.shared_pair_indices.shape[1]==2
    assert mesh.shared_pair_indices.shape[0]>=30
    # All interior faces link geometrically distinct channel centers.
    assert np.all(mesh.shared_face_lengths_m>0)
    assert np.all(mesh.channel_pitch_pair_m>0)
    assert all(0<=int(x)<19 for x in mesh.shared_pair_indices.flat)

    z=np.linspace(0.0,0.4,5)
    nz=len(z)-1
    fuel=CoreSolidLaw(1850,1200,24,cp_slope_per_k=3e-5,
                      conductivity_slope_per_k=-2e-5)
    coat=CoreSolidLaw(6400,650,18)
    inlet=np.full(19,500.0)
    q=np.full((nz,19),1.0)
    q[:,0]=100.0 # Asymmetric heat source forces lateral cross-channel flux.
    qh=np.full((nz,19),0.05)
    mass=np.full((nz,19),1.0e-5)
    boundary=HexCoolantBoundary(
        np.full(19,0.003),inlet,mass,np.full((nz,19),5500.0),
        qh,hydrogen_model="constant")
    old=HexCoreState(
        np.full((nz,19),500.0),
        np.full((nz,19),500.0),
        np.full((nz,19),500.0))
    hot=solve_hex_core_conjugate(mesh,z,old,boundary,fuel,coat,q)
    assert hot.iterations>=2
    assert abs(hot.energy_residual)<1e-5
    audit=audit_hex_core_cell_energy(mesh,z,old,hot,boundary,fuel,coat,q,dt_s=None)
    assert audit.maximum_scaled_cell_residual<1e-6, audit.maximum_scaled_cell_residual
    assert audit.global_scaled_residual<1e-7
    assert np.max(np.abs(hot.lateral_face_heat_w))>1e-5
    assert np.all(hot.fuel_to_coating_w>0)
    assert np.all(hot.coating_to_hydrogen_w>0)
    assert float(np.max(hot.state.fuel_temperature_k))>500.0
    emitted=math.fsum(
        boundary.mdot_kg_s[c]*boundary.cp_reference_j_kg_k*
        (hot.state.hydrogen_temperature_k[-1,c]-inlet[c])
        for c in range(19))
    assert math.isclose(emitted,
                        float(np.sum(q+qh)),rel_tol=1e-7,abs_tol=1e-5)

    # Turn off only the lateral face length (retain all masses, heat sources
    # and radial coupling) to verify lateral transport changes hot and
    # adjacent channel temperatures but not the global first law.
    disconnected=HexSectorNetwork(
        mesh.fuel_area_m2,
        mesh.shared_pair_indices,
        mesh.shared_face_lengths_m*1e-12,
        mesh.channel_pitch_pair_m,
        mesh.coolant_bore_radius_m,
        mesh.coating_outer_radius_m)
    decoupled=solve_hex_core_conjugate(
        disconnected,z,old,boundary,fuel,coat,q)
    assert np.max(hot.state.fuel_temperature_k[:,0])<np.max(
        decoupled.state.fuel_temperature_k[:,0])
    assert np.max(np.abs(
        hot.state.fuel_temperature_k-decoupled.state.fuel_temperature_k))>1e-3

    transient=solve_hex_core_conjugate(
        mesh,z,old,boundary,fuel,coat,q,dt_s=0.1,tolerance_k=1e-10)
    assert transient.solid_storage_j>0
    assert transient.hydrogen_storage_j>=0
    assert abs(transient.energy_residual)<1e-5
    audit_t=audit_hex_core_cell_energy(
        mesh,z,old,transient,boundary,fuel,coat,q,dt_s=0.1)
    assert audit_t.maximum_scaled_cell_residual<1e-6, audit_t.maximum_scaled_cell_residual
    assert np.any(transient.lateral_face_heat_w!=0)

    zeros=HexCoolantBoundary(
        boundary.mdot_kg_s,inlet,mass,boundary.film_w_m2_k,
        np.zeros((nz,19)),hydrogen_model="constant")
    equil=solve_hex_core_conjugate(
        mesh,z,old,zeros,fuel,coat,np.zeros((nz,19)),dt_s=0.1)
    assert np.max(np.abs(equil.state.fuel_temperature_k-500.0))<1e-6
    assert np.max(np.abs(equil.state.hydrogen_temperature_k-500.0))<1e-6

    # Contact resistance must increase hottest fuel temperature under
    # fixed nuclear heat, while preserving global enthalpy closure.
    resistive=solve_hex_core_conjugate(
        mesh,z,old,boundary,fuel,coat,q,
        fuel_coating_contact_resistance_m2_k_w=5e-5)
    assert resistive.state.fuel_temperature_k[0,0]>hot.state.fuel_temperature_k[0,0]
    assert abs(resistive.energy_residual)<1e-5
    audit_c=audit_hex_core_cell_energy(
        mesh,z,old,resistive,boundary,fuel,coat,q,
        dt_s=None,contact_resistance_m2_k_w=5e-5)
    assert audit_c.maximum_scaled_cell_residual<1e-6

    for invalid in (
        np.full((nz+1,19),1.0),
        np.full((nz,19),np.nan),
    ):
        try:
            solve_hex_core_conjugate(mesh,z,old,boundary,fuel,coat,invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid heat-field dimensions/values passed")

    print("19-sector lateral Voronoi core FV numerical physics: PASS")
    print("actual shared Voronoi faces:",len(mesh.shared_pair_indices))
    print("max real lateral face heat W:",float(np.max(np.abs(hot.lateral_face_heat_w))))
    print("effect of lateral conduction on hottest fuel K:",
          float(np.max(decoupled.state.fuel_temperature_k)-
                np.max(hot.state.fuel_temperature_k)))
    print("steady energy residual W:",hot.energy_residual)
    print("independent max local scaled residual:",audit.maximum_scaled_cell_residual)
    print("transient energy residual J:",transient.energy_residual)
    print("NERVA historical hot core/criticality validation: NOT PERFORMED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
