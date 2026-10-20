"""Regression tests: resolved NERVA fuel channels and tie counterflow.

Source fixtures are mathematical synthetic profiles, not OpenMC transport.
An optional generated fixture is explicitly tagged synthetic in the CLI.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .resolved_transient import (
    ResolvedHeating, ResolvedParameters, initial_resolved_state,
    read_resolved_heating, advance_resolved
)


def write_synthetic_reconstructions(directory: Path) -> tuple[Path,Path,Path]:
    directory.mkdir(parents=True, exist_ok=True)
    shape = (2,2,3)
    total = np.full(shape, 100., dtype=float)
    parts = {
        "fuel_power_density_w_cm3": 0.55*total,
        "fuel_hydrogen_power_density_w_cm3": 0.08*total,
        "tie_power_density_w_cm3": 0.10*total,
        "tie_supply_hydrogen_power_density_w_cm3": 0.03*total,
        "tie_return_hydrogen_power_density_w_cm3": 0.02*total,
    }
    mesh = directory/"mesh_fields.npz"
    np.savez_compressed(
        mesh, power_density_w_cm3=total,
        lower_left_cm=np.zeros(3),
        upper_right_cm=np.array(shape,dtype=float),
        dimension=np.array(shape,dtype=int), **parts,
    )
    fuel = directory/"fuel_reconstruction.npz"
    fw = np.full((2,19,3),660./(2*19*3))
    fd = np.full((2,19,3),96./(2*19*3))
    np.savez_compressed(
        fuel, channel_wall_axial_power_w=fw,
        channel_direct_h2_axial_power_w=fd,
        branch_mass_flow_kg_s=np.full((2,19),0.02),
        z_edges_m=np.linspace(0,1,4),
    )
    tie = directory/"tie_reconstruction.npz"
    np.savez_compressed(
        tie, tie_solid_axial_power_w=np.full((2,3),120./6),
        tie_supply_direct_h2_axial_power_w=np.full((2,3),36./6),
        tie_return_direct_h2_axial_power_w=np.full((2,3),24./6),
        tie_mass_flow_kg_s=np.array([0.05,0.06]),
        z_edges_m=np.linspace(0,1,4),
    )
    return mesh,fuel,tie


def check_error(func, klass=ValueError):
    try:
        func()
    except klass:
        return
    raise AssertionError(f"expected {klass.__name__}")


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--write-fixture",type=Path,
                        help="Leave generated synthetic reconstruction NPZ files at this location")
    opts=parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        path = opts.write_fixture or Path(tmp)
        mesh,fuelfile,tiefile=write_synthetic_reconstructions(path)
        r=read_resolved_heating(fuelfile,tiefile,mesh)
        assert r.fuel_solid_w.shape==(2,19,3)
        assert r.tie_solid_w.shape==(2,3)
        assert math.isclose(r.total_openmc_power_w,1200.)
        assert math.isclose(r.unresolved_power_w,264.)
        assert math.isclose(r.total_fuel_flow_kg_s,0.76)
        assert math.isclose(r.total_tie_flow_kg_s,0.11)

        p=ResolvedParameters(
            fuel_solid_capacity_j_k=5000.,
            fuel_h2_capacity_j_k=500.,
            fuel_ua_w_k=100.,
            tie_solid_capacity_j_k=3000.,
            tie_supply_h2_capacity_j_k=600.,
            tie_return_h2_capacity_j_k=700.,
            tie_supply_ua_w_k=80.,
            tie_return_ua_w_k=90.,
            hydrogen_cp_reference_j_kg_k=14300.,
            enthalpy_model="nasa7-h2",
            max_coupling_iterations=100,
        )
        original=initial_resolved_state(r,solid_temperature_k=500.,
                                        gas_temperature_k=300.)
        step=advance_resolved(r,original,p,300.,0.1)
        assert abs(step.energy_residual_j)<1e-4
        assert math.isclose(step.state.unresolved_energy_j,26.4)
        assert step.fuel_outlet_t_k.shape==(2,19)
        assert step.tie_outlet_t_k.shape==(2,)
        assert step.tie_coupling_max_iterations>=2
        assert np.all(step.fuel_outlet_t_k>300.)
        assert np.all(step.tie_outlet_t_k>300.)
        assert np.all(np.isfinite(step.state.tie_return_t_k))
        assert step.state.tie_return_t_k.min()>0.
        step2=advance_resolved(r,step.state,p,300.,0.1)
        assert abs(step2.energy_residual_j)<1e-4
        assert math.isclose(step2.state.unresolved_energy_j,52.8)

        # Fuel channel individuality: make channel zero in element zero
        # hotter; all other channels have unchanged deposited power.
        fw=r.fuel_solid_w.copy()
        fw[0,0,0]+=250.
        r_hot=replace(r,fuel_solid_w=fw,total_openmc_power_w=1450.)
        r_hot.validate()
        heat=advance_resolved(r_hot,original,p,300.,0.1)
        assert heat.state.fuel_solid_t_k[0,0,0]>heat.state.fuel_solid_t_k[0,1,0]
        assert heat.state.fuel_solid_t_k[0,0,0]>step.state.fuel_solid_t_k[0,0,0]
        assert abs(heat.energy_residual_j)<1e-4

        # Opposing tie flow: heat the RETURN coolant in one high-z cell.
        # At near-end z=0 the return stream has traversed the hot far-end
        # cell, so its temperature must exceed that of a cooler return
        # upstream cell. With zero wall UA/solid heating this is isolated.
        tw=np.zeros((1,3))
        sup0=np.zeros((1,3))
        ret0=np.array([[0.,0.,10000.]])
        rf=np.zeros((1,19,3))
        dr=np.zeros_like(rf)
        isolated=ResolvedHeating(
            rf,dr,np.full((1,19),0.02),tw,sup0,ret0,
            np.array([0.05]),10000.,0.)
        isolated.validate()
        onlygas=replace(p,fuel_ua_w_k=0.,tie_supply_ua_w_k=0.,
                        tie_return_ua_w_k=0.)
        z=initial_resolved_state(isolated,solid_temperature_k=400.,
                                 gas_temperature_k=300.)
        reverse=advance_resolved(isolated,z,onlygas,300.,1.)
        assert abs(reverse.energy_residual_j)<1e-4
        assert np.allclose(reverse.state.tie_solid_t_k,400.,atol=1e-9)
        # With transient storage in each cell, the directly heated far-end
        # cell is hottest; the near-end return outlet responds downstream.
        assert reverse.state.tie_return_t_k[0,-1]>reverse.state.tie_return_t_k[0,0]>300.
        assert reverse.tie_outlet_t_k[0]>300.
        assert np.allclose(reverse.state.tie_supply_t_k,300.,atol=1e-9)
        # Heat solid only: BOTH coolant paths gain heat from the SAME
        # solid, and the final return outlet is physically z=0.
        solid_input=replace(isolated,
            tie_solid_w=np.full((1,3),1500.),
            tie_return_h2_w=np.zeros((1,3)),
            total_openmc_power_w=4500.)
        solid_input.validate()
        both=advance_resolved(solid_input,z,p,300.,2.)
        assert abs(both.energy_residual_j)<1e-4
        assert both.state.tie_supply_t_k[0,-1]>300.
        assert both.state.tie_return_t_k[0,0]>300.
        # A thermally hot solid may still cool while delivering heat to
        # both gas passages. Compare its final state with a zero-source
        # baseline, not with its initial 400 K.
        cold_source=replace(solid_input,
            tie_solid_w=np.zeros((1,3)),
            tie_return_h2_w=np.full((1,3),1.),
            total_openmc_power_w=3.)
        cold_source.validate()
        baseline=advance_resolved(cold_source,z,p,300.,2.)
        assert both.state.tie_solid_t_k[0,0]>baseline.state.tie_solid_t_k[0,0]
        assert both.tie_outlet_t_k[0]>baseline.tie_outlet_t_k[0]

        check_error(lambda:read_resolved_heating(tiefile,fuelfile,mesh),
                    (KeyError,ValueError))
        bad=replace(r,unresolved_power_w=0.)
        check_error(bad.validate)
        badflow=replace(r,fuel_flow_kg_s=np.zeros_like(r.fuel_flow_kg_s))
        check_error(badflow.validate)
        badstate=replace(original,tie_solid_t_k=np.zeros((2,2)))
        check_error(lambda:advance_resolved(r,badstate,p,300.,0.1))
        check_error(lambda:advance_resolved(r,original,p,300.,-1.))
        # Use a higher than supported total heat to trip the thermo domain.
        extreme=replace(isolated,
            tie_return_h2_w=np.full((1,3),1e10),
            total_openmc_power_w=3e10,
        )
        extreme.validate()
        check_error(lambda:advance_resolved(extreme,z,onlygas,300.,1.))
    print("Channel-resolved fuel and tie-counterflow energy conservation: PASS")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
