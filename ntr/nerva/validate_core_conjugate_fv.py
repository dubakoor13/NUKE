"""Pure numerical self-tests for the implicit conjugate FV implementation.

These are manufactured/analytical conservation and input-defence checks,
NOT physical qualification of actual NERVA core, OpenMC KERMA or NASA data.
"""
from __future__ import annotations

import math

import numpy as np

from .core_conjugate_fv import (
    CoreSectorBoundary,CoreSectorGrid,CoreSectorState,CoreSolidLaw,
    solve_core_sector_conjugate,
)


def main()->int:
    z=np.linspace(0.0,0.5,6)
    grid=CoreSectorGrid.equivalent_annulus(
        z,bore_radius_m=0.00125,
        coating_thickness_m=0.00005,
        fuel_sector_area_m2=1.8e-5,
        coating_cells=2,fuel_cells=6)
    nz,nr=grid.shape
    fuel=CoreSolidLaw(1850,1250,23,cp_slope_per_k=1e-4,
                      conductivity_slope_per_k=-5e-5)
    coat=CoreSolidLaw(6400,650,19,cp_slope_per_k=4e-5)
    qsol=np.linspace(1.0,2.0,nz)
    qgas=np.linspace(0.08,0.16,nz)
    b=CoreSectorBoundary(
        mass_flow_kg_s=0.002,inlet_temperature_k=500.0,
        convection_w_m2_k=np.full(nz,5000.0),
        direct_hydrogen_heating_w=qgas,
        gas_mass_kg=np.full(nz,1.0e-5),
        gas_enthalpy_model="constant",constant_gas_cp_j_kg_k=14300)
    old=CoreSectorState(np.full((nz,nr),500.0),np.full(nz,500.0))
    steady=solve_core_sector_conjugate(grid,old,b,fuel,coat,qsol)
    assert steady.iterations>=2
    assert np.min(steady.state.solid_temperature_k)>500.0
    assert np.min(steady.convected_wall_heating_w)>0.0
    assert math.isclose(
        steady.convected_wall_heating_w.sum(),qsol.sum(),
        abs_tol=1e-5,rel_tol=1e-6)
    assert math.isclose(
        b.mass_flow_kg_s*b.constant_gas_cp_j_kg_k*
        (steady.state.hydrogen_temperature_k[-1]-b.inlet_temperature_k),
        float(qsol.sum()+qgas.sum()),rel_tol=1e-6,abs_tol=1e-6)
    assert abs(steady.energy_balance_residual_j)<1e-5

    # The transient stores heat in both domains and conserves the FIRST law,
    # rather than forcing all generated heat to coolant in one timestep.
    transient=solve_core_sector_conjugate(
        grid,old,b,fuel,coat,qsol,dt_s=0.20)
    assert transient.iterations>=2
    assert transient.solid_energy_change_j>0
    assert transient.hydrogen_energy_change_j>=0
    assert abs(transient.energy_balance_residual_j)<1e-5
    assert transient.state.solid_temperature_k.shape==(nz,nr)

    # Neutron power can be supplied as an explicitly distributed radial/axial
    # deposition field. Never assume the 1-D shape is an actual spatial tally.
    volumes=grid.volumes_m3.copy()
    volumes[:,:grid.n_coating_cells]=0
    separated=qsol[:,None]*volumes/volumes.sum(axis=1)[:,None]
    again=solve_core_sector_conjugate(grid,old,b,fuel,coat,separated)
    assert np.allclose(
        again.state.solid_temperature_k,
        steady.state.solid_temperature_k,atol=1e-6)
    bad=CoreSectorState(np.ones((nz,nr+1)),np.ones(nz))
    try:
        solve_core_sector_conjugate(grid,bad,b,fuel,coat,qsol)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong solid array geometry passed")

    # Uniform zero-power/zero-direct-source equilibrium: no internal heat,
    # no wall flux, and no stored-energy creation. This exercises the coupled
    # matrix without inventing a thermal source.
    b0=CoreSectorBoundary(
        b.mass_flow_kg_s,500.0,b.convection_w_m2_k,np.zeros(nz),
        b.gas_mass_kg,gas_enthalpy_model="constant")
    zero=solve_core_sector_conjugate(
        grid,old,b0,fuel,coat,np.zeros(nz),dt_s=0.1)
    assert np.max(np.abs(zero.state.solid_temperature_k-500))<1e-6
    assert np.max(np.abs(zero.state.hydrogen_temperature_k-500))<1e-6
    print("Core radial-axial implicit FV numerical self-test: PASS")
    print("Steady channel H2 outlet K:",steady.state.hydrogen_temperature_k[-1])
    print("Transient stored solid J:",transient.solid_energy_change_j)
    print("Historical/hot nuclear physics: NOT VALIDATED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
