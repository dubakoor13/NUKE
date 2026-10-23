"""Numerical tests of combined radial/axial/lateral 19-channel thermal physics.

These checks use externally prescribed synthetic heat rates, not nuclear
transport or NERVA flight-temperature physical reference data.
"""
from __future__ import annotations

import math
import numpy as np

from .core_conjugate_fv import (
    CoreSectorBoundary, CoreSectorGrid, CoreSectorState,
    CoreSolidLaw, solve_core_sector_conjugate,
)
from .core_hex_lateral_fv import HexSectorNetwork, HexCoolantBoundary
from .core_hex_radial_lateral_fv import (
    RadialHexMesh, RadialHexState, solve_radial_hex_core,
)
from .core_hex_radial_lateral_balance import audit_radial_hex_energy


def main()->int:
    network=HexSectorNetwork.from_nerva_config()
    assert len(network.shared_pair_indices)==42
    z=np.linspace(0.,0.45,4)
    mesh=RadialHexMesh.from_network(network,z,coating_cells=2,fuel_cells=6)
    nz,ns,nr=mesh.shape
    assert mesh.radial_edges_m.shape==(19,nr+1)
    fuel=CoreSolidLaw(1850.,1250.,23.,cp_slope_per_k=4e-5,
                      conductivity_slope_per_k=-2e-5)
    coat=CoreSolidLaw(6400.,650.,19.,cp_slope_per_k=3e-5)
    Q=np.full((nz,ns),0.25)
    Q[:,0]=18.
    qh=np.full((nz,ns),0.03)
    gas=HexCoolantBoundary(
        mdot_kg_s=np.full(ns,0.003),
        inlet_temperature_k=np.full(ns,500.),
        hydrogen_mass_kg=np.full((nz,ns),1e-5),
        film_w_m2_k=np.full((nz,ns),5500.),
        direct_hydrogen_power_w=qh,
        hydrogen_model="constant",
    )
    old=RadialHexState(
        solid_temperature_k=np.full((nz,ns,nr),500.),
        hydrogen_temperature_k=np.full((nz,ns),500.))
    full=solve_radial_hex_core(mesh,old,gas,fuel,coat,Q)
    audit=audit_radial_hex_energy(
        mesh,old,full,gas,fuel,coat,Q,dt_s=None)
    assert full.iterations>=2
    assert abs(full.first_law_residual)<1e-5
    assert audit.max_relative_cell_residual<1e-7, audit.max_relative_cell_residual
    assert audit.global_relative_residual<1e-7
    assert np.max(abs(full.lateral_face_heat_w))>1e-3
    assert np.min(full.wall_to_hydrogen_w)>0
    assert np.max(full.state.solid_temperature_k)>500.

    # Disconnect only the transverse interfaces. No source/mass/film or
    # radial/axial geometry changes are allowed in this comparison.
    disconnected=HexSectorNetwork(
        network.fuel_area_m2,
        network.shared_pair_indices,
        network.shared_face_lengths_m*1e-12,
        network.channel_pitch_pair_m,
        network.coolant_bore_radius_m,network.coating_outer_radius_m)
    no_lateral=RadialHexMesh.from_network(
        disconnected,z,coating_cells=2,fuel_cells=6)
    disconnected_result=solve_radial_hex_core(
        no_lateral,old,gas,fuel,coat,Q)
    peak_reduction=(
        np.max(disconnected_result.state.solid_temperature_k)-
        np.max(full.state.solid_temperature_k))
    assert peak_reduction>1e-3, peak_reduction

    # A disconnected sector must match the earlier independent solver's
    # full radial+axial+gas equations, rather than merely its total power.
    isolated_mesh=CoreSectorGrid.equivalent_annulus(
        z,bore_radius_m=network.coolant_bore_radius_m,
        coating_thickness_m=(
            network.coating_outer_radius_m-network.coolant_bore_radius_m),
        fuel_sector_area_m2=float(network.fuel_area_m2[0]),
        coating_cells=2,fuel_cells=6)
    stand_gas=CoreSectorBoundary(
        mass_flow_kg_s=gas.mdot_kg_s[0],
        inlet_temperature_k=gas.inlet_temperature_k[0],
        convection_w_m2_k=gas.film_w_m2_k[:,0],
        direct_hydrogen_heating_w=qh[:,0],
        gas_mass_kg=gas.hydrogen_mass_kg[:,0],
        gas_enthalpy_model="constant")
    stand_state=CoreSectorState(
        old.solid_temperature_k[:,0,:],
        old.hydrogen_temperature_k[:,0])
    independent=solve_core_sector_conjugate(
        isolated_mesh,stand_state,stand_gas,fuel,coat,Q[:,0])
    max_discrepancy=max(
        float(np.max(np.abs(
            disconnected_result.state.solid_temperature_k[:,0,:]-
            independent.state.solid_temperature_k))),
        float(np.max(np.abs(
            disconnected_result.state.hydrogen_temperature_k[:,0]-
            independent.state.hydrogen_temperature_k))))
    assert max_discrepancy<1e-6, max_discrepancy

    transient=solve_radial_hex_core(
        mesh,old,gas,fuel,coat,Q,dt_s=0.10,tolerance_k=1e-10)
    audit_t=audit_radial_hex_energy(
        mesh,old,transient,gas,fuel,coat,Q,dt_s=0.10)
    assert transient.stored_solid_energy_j>0
    assert transient.stored_hydrogen_energy_j>0
    assert abs(transient.first_law_residual)<1e-5
    assert audit_t.max_relative_cell_residual<1e-7

    # Frozen molecular H2 NASA-7: use nonconstant cp(T) while conserving
    # stored H2 mass times enthalpy, not an arbitrary reference cp*T.
    molecular_h2=HexCoolantBoundary(
        mdot_kg_s=gas.mdot_kg_s,
        inlet_temperature_k=np.full(ns,850.),
        hydrogen_mass_kg=gas.hydrogen_mass_kg,
        film_w_m2_k=gas.film_w_m2_k,
        direct_hydrogen_power_w=qh,
        hydrogen_model="nasa7-h2",
    )
    warm=RadialHexState(
        np.full((nz,ns,nr),850.),np.full((nz,ns),850.))
    thermo=solve_radial_hex_core(
        mesh,warm,molecular_h2,fuel,coat,Q,
        dt_s=0.15,tolerance_k=1e-10)
    thermo_audit=audit_radial_hex_energy(
        mesh,warm,thermo,molecular_h2,fuel,coat,Q,dt_s=0.15)
    assert thermo.stored_solid_energy_j>0
    assert thermo.stored_hydrogen_energy_j>0
    assert thermo_audit.max_relative_cell_residual<1e-7
    assert abs(thermo.first_law_residual)<1e-5

    # Explicit 3-D cell-wise heating equivalent to volume-distributed
    # 2-D [z,19] sources must give the identical result.
    volume=mesh.volumes_m3.copy()
    volume[:,:,:mesh.coating_cells]=0
    qcells=Q[:,:,None]*volume/volume.sum(axis=2)[:,:,None]
    explicit=solve_radial_hex_core(mesh,old,gas,fuel,coat,qcells)
    assert np.allclose(
        explicit.state.solid_temperature_k,
        full.state.solid_temperature_k,atol=1e-6,rtol=0)
    assert np.allclose(
        explicit.state.hydrogen_temperature_k,
        full.state.hydrogen_temperature_k,atol=1e-6,rtol=0)

    # Material interface resistance alters solid temperature while leaving
    # externally supplied nuclear source and first-law ledger intact.
    R=4e-5
    resistive=solve_radial_hex_core(
        mesh,old,gas,fuel,coat,Q,contact_resistance_m2_k_w=R)
    audit_r=audit_radial_hex_energy(
        mesh,old,resistive,gas,fuel,coat,Q,dt_s=None,
        contact_resistance_m2_k_w=R)
    assert resistive.state.solid_temperature_k[0,0,-1] > (
        full.state.solid_temperature_k[0,0,-1])
    assert audit_r.max_relative_cell_residual<1e-7

    # With zero nuclear source, initially uniform solid and gas remain
    # exactly in thermal equilibrium (including all neighbor links).
    zero_gas=HexCoolantBoundary(
        gas.mdot_kg_s,gas.inlet_temperature_k,gas.hydrogen_mass_kg,
        gas.film_w_m2_k,np.zeros_like(qh),hydrogen_model="constant")
    zero=solve_radial_hex_core(
        mesh,old,zero_gas,fuel,coat,np.zeros_like(Q),dt_s=0.2)
    assert np.max(np.abs(zero.state.solid_temperature_k-500))<1e-6
    assert np.max(np.abs(zero.state.hydrogen_temperature_k-500))<1e-6

    for invalid in (
        np.full((nz+1,19),2.),
        np.full((nz,19),np.nan),
        np.full((nz,19),-1.),
    ):
        try:
            solve_radial_hex_core(mesh,old,gas,fuel,coat,invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("nonphysical/incorrect heat source accepted")

    print("Unified radial-axial-lateral 19-channel FV: PASS")
    print("  lateral shared faces:",len(network.shared_pair_indices))
    print("  radial solid cells per channel:",nr)
    print("  eliminated-hotspot difference (synthetic) K:",peak_reduction)
    print("  max disconnected-vs-single-channel difference K:",max_discrepancy)
    print("  independent max local scaled residual:",audit.max_relative_cell_residual)
    print("  global steady first-law residual W:",full.first_law_residual)
    print("  global transient first-law residual J:",transient.first_law_residual)
    print("  NASA-7 H2 transient first-law residual J:",thermo.first_law_residual)
    print("  hot OpenMC/NASA physical validation: NOT ESTABLISHED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
