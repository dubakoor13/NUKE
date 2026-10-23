"""Pure numerical self-tests for the implicit conjugate FV implementation.

These are manufactured/analytical conservation and input-defence checks,
NOT physical qualification of actual NERVA core, OpenMC KERMA or NASA data.
"""
from __future__ import annotations

import math

import numpy as np

from .core_conjugate_fv_diagnostics import audit_core_sector_discrete_energy
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
    steady_check=audit_core_sector_discrete_energy(
        grid,old,steady,b,fuel,coat,qsol,dt_s=None)
    assert steady_check.max_local_residual_scaled<1e-5
    assert steady_check.global_residual_scaled<1e-7

    # The transient stores heat in both domains and conserves the FIRST law,
    # rather than forcing all generated heat to coolant in one timestep.
    transient=solve_core_sector_conjugate(
        grid,old,b,fuel,coat,qsol,dt_s=0.20)
    assert transient.iterations>=2
    assert transient.solid_energy_change_j>0
    assert transient.hydrogen_energy_change_j>=0
    assert abs(transient.energy_balance_residual_j)<1e-5
    assert transient.state.solid_temperature_k.shape==(nz,nr)
    transient_check=audit_core_sector_discrete_energy(
        grid,old,transient,b,fuel,coat,qsol,dt_s=0.20)
    assert transient_check.max_local_residual_scaled<1e-5
    assert transient_check.global_residual_scaled<1e-7

    # Contact resistance increases the fuel/coating jump at otherwise
    # unchanged heat deposition, without creating/destroying energy.
    contact_r=0.0001
    contact=solve_core_sector_conjugate(
        grid,old,b,fuel,coat,qsol,
        fuel_coating_contact_resistance_m2_k_w=contact_r)
    contact_check=audit_core_sector_discrete_energy(
        grid,old,contact,b,fuel,coat,qsol,dt_s=None,
        fuel_coating_contact_resistance_m2_k_w=contact_r)
    assert contact_check.max_local_residual_scaled<1e-5
    assert contact.state.solid_temperature_k[0,-1]>steady.state.solid_temperature_k[0,-1]

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
    # Independent continuum check: 1-axial-cell radial conduction with
    # uniform annular fuel heating, zero coating generation and an insulated
    # external radius has a closed-form temperature at each fuel radius.
    # Refining the fuel FV grid should approach that analytical field.
    def annulus_error(nfuel:int)->float:
        zm=np.array([0.0,0.20])
        g=CoreSectorGrid.equivalent_annulus(
            zm,bore_radius_m=0.00125,coating_thickness_m=0.00005,
            fuel_sector_area_m2=1.8e-5,coating_cells=2,fuel_cells=nfuel)
        t=CoreSectorState(np.full(g.shape,500.0),np.full(1,500.0))
        coeff=CoreSectorBoundary(
            0.01,500.0,np.array([4500.0]),np.array([0.0]),
            np.array([1.0e-5]),gas_enthalpy_model="constant")
        f=CoreSolidLaw(1850.0,1250.0,23.0)
        c=CoreSolidLaw(6400.0,650.0,19.0)
        result=solve_core_sector_conjugate(g,t,coeff,f,c,np.array([25.0]))
        ri=g.r_edges_m[0]
        rcoat=g.r_edges_m[g.n_coating_cells]
        outer=g.r_edges_m[-1]
        rlast=math.sqrt(0.5*sum(x*x for x in g.r_edges_m[-2:]))
        qprime=25.0/0.20
        qvol=25.0/(1.8e-5*0.20)
        expected=(qprime/(2*math.pi*ri*4500.0)
            +qprime*math.log(rcoat/ri)/(2*math.pi*19.0)
            +qvol/(2*23.0)*(outer**2*math.log(rlast/rcoat)
                                -0.5*(rlast**2-rcoat**2)))
        predicted=(result.state.solid_temperature_k[0,-1]
                   -result.state.hydrogen_temperature_k[0])
        check=audit_core_sector_discrete_energy(
            g,t,result,coeff,f,c,np.array([25.0]),dt_s=None)
        assert check.max_local_residual_scaled<1e-5
        return abs(predicted-expected)
    error_coarse=annulus_error(4)
    error_fine=annulus_error(32)
    assert error_fine<error_coarse, (error_coarse,error_fine)
    print("Core radial-axial implicit FV numerical self-test: PASS")
    print("Independent radial analytical error coarse/fine K:",error_coarse,error_fine)
    print("Steady channel H2 outlet K:",steady.state.hydrogen_temperature_k[-1])
    print("Transient stored solid J:",transient.solid_energy_change_j)
    print("Historical/hot nuclear physics: NOT VALIDATED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
