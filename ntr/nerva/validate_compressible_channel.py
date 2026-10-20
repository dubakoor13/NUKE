"""Physics verification for subsonic compressible NERVA H2 channels.

Checks independently computed total-enthalpy and momentum budgets,
pressure/velocity coupling, axial mass continuity, direct gas deposition,
comparison with low-Mach Darcy limit, and explicit choking/domain gates.
These are analytical/numerical checks with hypothetical conditions, NOT
historical reactor performance measurements.
"""
from __future__ import annotations

import math
import numpy as np

from .compressible_channel import (
    R_H2, SubsonicFlowError, solve_fuel_channel_compressible
)
from .nasa_hydrogen import hydrogen_state
from .thermal import HydrogenProperties, solve_fuel_channel, friction_factor


def expect_rejected(call, klass=ValueError) -> None:
    try:
        call()
    except klass:
        return
    raise AssertionError(f"expected {klass.__name__}")


def evaluate_solution(power, direct, z, mdot, temp, pressure, diameter):
    return solve_fuel_channel_compressible(
        np.array(power,dtype=float),
        np.array(z,dtype=float),
        1,mdot,temp,pressure,diameter,
        axial_direct_coolant_power_w=np.array(direct,dtype=float)
    )


def check_governing_equations(sol, t0, p0, mdot):
    assert sol.mach_number is not None
    assert sol.energy_residual_w is not None
    assert sol.momentum_residual_pa is not None
    assert np.max(np.abs(sol.energy_residual_w))<1e-4
    assert np.max(np.abs(sol.momentum_residual_pa))<1e-3
    assert np.all(sol.mach_number>0)
    assert np.all(sol.mach_number<0.85)
    assert np.all(np.diff(sol.pressure_pa)<0)
    p_in=p0
    t_in=t0
    area=math.pi*sol.channel_diameter_m**2/4
    for i in range(sol.pressure_pa.size):
        p_out=float(sol.pressure_pa[i])
        t_out=float(sol.bulk_temperature_k[i])
        x=hydrogen_state(t_in,p_in,mode="frozen")
        y=hydrogen_state(t_out,p_out,mode="frozen")
        vin=mdot/(x.density_kg_m3*area)
        vout=mdot/(y.density_kg_m3*area)
        deposited=float(sol.channel_power_w[i])
        energy=mdot*((y.enthalpy_j_kg+0.5*vout**2)
                     -(x.enthalpy_j_kg+0.5*vin**2))
        pressure_loss=p_in-p_out
        wall_loss=float(sol.friction_pressure_loss_pa[i])
        accel_loss=float(sol.acceleration_pressure_loss_pa[i])
        assert math.isclose(energy,deposited,rel_tol=3e-8,abs_tol=2e-5)
        assert math.isclose(pressure_loss,wall_loss+accel_loss,
                            rel_tol=4e-7,abs_tol=1e-3)
        assert math.isclose(accel_loss,mdot/area*(vout-vin),
                            rel_tol=2e-9,abs_tol=1e-6)
        t_in,p_in=t_out,p_out


def main() -> int:
    # Independent low Mach Darcy-Weisbach limit. For weak compression,
    # drop approaches f*(L/D)*rho*v²/2 with fixed gas density.
    z=[0.,0.2,0.4,0.6]
    mdot=0.01
    diameter=0.02
    pressure=10e6
    inlet_t=400.
    unheated=evaluate_solution(
        [0,0,0],[0,0,0],z,mdot,inlet_t,pressure,diameter
    )
    check_governing_equations(unheated,inlet_t,pressure,mdot)
    initial=hydrogen_state(inlet_t,pressure,mode="frozen")
    area=math.pi*diameter**2/4
    vin=mdot/(initial.density_kg_m3*area)
    re=(mdot/area)*diameter/1.20e-5
    darcy=friction_factor(re,1e-6/diameter)
    analytic_drop=darcy*0.6/diameter*0.5*initial.density_kg_m3*vin**2
    measured=pressure-unheated.outlet_pressure_pa
    assert math.isclose(measured,analytic_drop,rel_tol=0.02)
    assert max(unheated.mach_number)<0.01

    # Heat-depositing segment conserves total rather than static
    # enthalpy: heating induces acceleration pressure drop.
    z=[0.,0.05,0.12,0.2]
    mdot=0.03
    pressure=3.0e6
    diameter=0.0075
    t0=450.
    q_solid=[2000,3000,2000]
    q_direct=[400,600,1000]
    heated=evaluate_solution(
        q_solid,q_direct,z,mdot,t0,pressure,diameter
    )
    check_governing_equations(heated,t0,pressure,mdot)
    assert math.isclose(heated.absorbed_power_w,9000.)
    assert math.isclose(heated.direct_nuclear_coolant_power_w,2000.)
    assert math.isclose(heated.wall_transferred_power_w,7000.)
    assert heated.outlet_temperature_k>t0
    assert np.all(heated.acceleration_pressure_loss_pa>0)
    assert np.all(heated.friction_pressure_loss_pa>0)
    assert heated.outlet_pressure_pa>0
    assert np.all(heated.wall_temperature_k>heated.bulk_temperature_k-5)
    assert max(heated.mach_number)<0.85

    # Original solver leaves out acceleration and kinetic energy; this
    # must matter for a finite-Mach heated channel.
    legacy=solve_fuel_channel(
        np.array(q_solid),np.array(z),1,mdot,t0,pressure,diameter,
        properties=HydrogenProperties(),
        axial_direct_coolant_power_w=np.array(q_direct)
    )
    assert heated.outlet_pressure_pa<legacy.outlet_pressure_pa
    assert heated.mach_number is not None
    assert legacy.mach_number is None

    # Direct nuclear heating alone MUST NOT be credited as wall power.
    alone=evaluate_solution(
        [0,0],[700,1300],[0,0.1,0.2],0.015,400,7e6,0.01
    )
    assert alone.wall_transferred_power_w==0
    assert alone.direct_nuclear_coolant_power_w==2000
    assert np.all(alone.heat_flux_w_m2==0.)
    assert np.all(alone.wall_temperature_k>400.)
    check_governing_equations(alone,400.,7e6,0.015)

    # Split among representative identical channels: 2 channels
    # receive half of deposited wall and direct coolant power per
    # channel, while their mass flow is specified PER channel.
    split=solve_fuel_channel_compressible(
        np.array([2000.,3000.]),np.array([0.,0.1,0.2]),
        2,0.03,450.,3e6,0.01,
        axial_direct_coolant_power_w=np.array([200.,500.])
    )
    assert math.isclose(split.absorbed_power_w,2850.)
    assert math.isclose(split.direct_nuclear_coolant_power_w,350.)
    check_governing_equations(split,450.,3e6,0.03)

    expect_rejected(lambda:evaluate_solution(
        [100],[0],[0,0.1],1.,500,5e5,0.003
    ),SubsonicFlowError)
    expect_rejected(lambda:evaluate_solution(
        [1e8],[0],[0,.1],0.01,3400,5e6,0.01
    ),SubsonicFlowError)
    expect_rejected(lambda:evaluate_solution(
        [-10],[0],[0,.1],0.01,400,5e6,0.01
    ))
    expect_rejected(lambda:evaluate_solution(
        [10],[0],[0.,0.],0.01,400,5e6,0.01
    ))
    expect_rejected(lambda:solve_fuel_channel_compressible(
        np.array([100.]),np.array([0.,0.1]),1,
        0.01,400,5e6,0.01,property_model=object()
    ))
    print(
        "NERVA compressible H2 mass, momentum, stagnation-energy and "
        "Mach-limit physics: PASS"
    )
    print(
        f"Low-Mach Darcy predicted dp={analytic_drop:.6g} Pa "
        f"vs compressible dp={measured:.6g} Pa"
    )
    print(
        f"Heated channel: exit={heated.outlet_temperature_k:.6g} K, "
        f"dp_friction={sum(heated.friction_pressure_loss_pa):.6g} Pa, "
        f"dp_acceleration={sum(heated.acceleration_pressure_loss_pa):.6g} Pa, "
        f"Mach_max={max(heated.mach_number):.5g}"
    )
    return 0


if __name__=="__main__":
    raise SystemExit(main())
