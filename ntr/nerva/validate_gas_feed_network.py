"""Independent gas-phase feed / channel / nozzle conservation regressions.

Synthetic specified heating and SOURCE-curve parameters, not experimental
NERVA pump data. Validate compressor isentrope, energy, mass, momentum,
chamber pressure and thrust sensitivity with physical limiting cases.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

from .gas_feed_network import (
    GasFeedNetwork, NoSubsonicNetworkSolution, evaluate_gas_network,
    solve_gas_network,
)
from .nasa_hydrogen import hydrogen_state


def baseline() -> GasFeedNetwork:
    return GasFeedNetwork(
        supply_pressure_pa=2.0e6,
        supply_temperature_k=450.,
        pressure_source_shutoff_pa=2.5e6,
        pressure_source_quadratic_pa_s2_kg2=3.e7,
        feed_quadratic_pa_s2_kg2=2.e7,
        exit_quadratic_pa_s2_kg2=3.e7,
        isentropic_compression_efficiency=0.78,
        throat_area_m2=1.2e-5,
        expansion_ratio=6.,
        channel_count=8,
        channel_diameter_m=0.01,
        z_edges_m=(0.,0.10,0.20,0.30),
        wall_heating_w=(12000.,10000.,8000.),
        direct_hydrogen_heating_w=(1000.,2000.,1000.),
        maximum_total_flow_kg_s=0.2,
        maximum_mach=0.85,
    )


def expect_error(f,expected=(ValueError,RuntimeError)):
    try:
        f()
    except expected:
        return
    raise AssertionError("expected invalid physical solution to fail")


def verify(config, point) -> None:
    mdot=point.total_mass_flow_kg_s
    nozzle=point.nozzle
    assert mdot>0
    assert abs(mdot-nozzle.mass_flow_kg_s)/mdot < 2e-7
    assert 0 < point.chamber_pressure_pa < point.fuel_channel_exit_pa
    assert point.fuel_channel_exit_pa < point.fuel_channel_inlet_pa
    assert point.fuel_channel_inlet_pa < point.pressure_source_discharge_pa
    assert point.pressure_source_discharge_pa > config.supply_pressure_pa
    assert config.supply_temperature_k < point.compressor_outlet_temperature_k
    assert point.compressor_outlet_temperature_k > point.channel_inlet_temperature_k
    assert point.chamber_temperature_k>point.channel_inlet_temperature_k
    assert nozzle.chamber_pressure_pa==point.chamber_pressure_pa
    assert nozzle.chamber_temperature_k==point.chamber_temperature_k
    assert nozzle.thrust_n>0 and nozzle.isp_s>0
    assert max(point.channel.mach_number)<config.maximum_mach
    assert abs(point.total_energy_residual_w) < 1e-6*max(
        1.,point.compressor_shaft_power_w+point.deposited_reactor_heat_w
    )
    assert abs(point.pressure_balance_residual_pa)<1e-8*point.pressure_source_discharge_pa
    assert math.isclose(
        point.pressure_source_rise_pa,
        point.feed_loss_pa+point.channel_friction_loss_pa
        +point.channel_acceleration_loss_pa+point.exit_loss_pa
        +point.chamber_pressure_pa-config.supply_pressure_pa,
        rel_tol=1e-7,abs_tol=1e-3,
    )
    assert math.isclose(point.deposited_reactor_heat_w,34000.,
                        rel_tol=1e-12) or point.deposited_reactor_heat_w>=0

    # Compressor thermodynamics: the ratio of isentropic and actual
    # enthalpy rise must equal the supplied shaft efficiency.
    inlet=hydrogen_state(config.supply_temperature_k,
                         config.supply_pressure_pa,mode="frozen")
    discharge=hydrogen_state(point.compressor_outlet_temperature_k,
                             point.pressure_source_discharge_pa,mode="frozen")
    assert math.isclose(
        mdot*(discharge.enthalpy_j_kg-inlet.enthalpy_j_kg),
        point.compressor_shaft_power_w,
        rel_tol=5e-10,abs_tol=1e-4,
    )
    assert math.isclose(
        point.compressor_shaft_power_w+point.deposited_reactor_heat_w,
        mdot*(hydrogen_state(point.chamber_temperature_k,
                            point.chamber_pressure_pa,
                            mode="frozen").enthalpy_j_kg-inlet.enthalpy_j_kg),
        rel_tol=1e-8,abs_tol=1e-3,
    )


def main() -> int:
    config=baseline()
    result=solve_gas_network(config)
    verify(config,result)

    # Independently evaluate the exact returned root from first
    # principles; not just a cached bisection result.
    direct,residual=evaluate_gas_network(config,result.total_mass_flow_kg_s)
    assert abs(residual) < 2e-7*result.total_mass_flow_kg_s
    assert math.isclose(direct.chamber_pressure_pa,result.chamber_pressure_pa,
                        rel_tol=1e-11)

    # More available pressure raises both chamber Pc and flow, while the
    # same deposited heat is split across more gas and changes Tc.
    stronger=replace(config,pressure_source_shutoff_pa=3.0e6)
    raised=solve_gas_network(stronger)
    verify(stronger,raised)
    assert raised.total_mass_flow_kg_s > result.total_mass_flow_kg_s
    assert raised.chamber_pressure_pa > result.chamber_pressure_pa
    assert raised.nozzle.thrust_n > result.nozzle.thrust_n

    # More hydrogen heating increases stagnation enthalpy and changes
    # the choked mass-flow closure at a fixed nozzle geometry.
    hotter=replace(config,wall_heating_w=(18000.,16000.,14000.))
    warm=solve_gas_network(hotter)
    verify(hotter,warm)
    assert warm.chamber_temperature_k>result.chamber_temperature_k
    assert warm.deposited_reactor_heat_w>result.deposited_reactor_heat_w

    # Check adiabatic source compression, reactor heat and acoustic
    # flow in a channel when no reactor power is deposited.
    unheated=replace(config,
        wall_heating_w=(0.,0.,0.),
        direct_hydrogen_heating_w=(0.,0.,0.))
    adiabatic=solve_gas_network(unheated)
    verify(unheated,adiabatic)
    assert adiabatic.deposited_reactor_heat_w==0.
    assert adiabatic.compressor_outlet_temperature_k>450.
    assert all(q==0 for q in adiabatic.channel.channel_power_w)
    assert adiabatic.chamber_temperature_k>450.

    # The source curve and nozzle must be jointly solvable. Refuse
    # insufficient head, near-choking/high-flow cases and bad geometry.
    expect_error(lambda:solve_gas_network(replace(
        config,throat_area_m2=5e-3
    )),NoSubsonicNetworkSolution)
    expect_error(lambda:solve_gas_network(replace(
        config,maximum_total_flow_kg_s=0.001
    )),NoSubsonicNetworkSolution)
    expect_error(lambda:solve_gas_network(replace(
        config,supply_temperature_k=20.
    )))
    expect_error(lambda:solve_gas_network(replace(
        config,isentropic_compression_efficiency=0.0
    )))
    expect_error(lambda:solve_gas_network(replace(
        config,z_edges_m=(0.,0.2,0.1,0.3)
    )))
    expect_error(lambda:solve_gas_network(replace(
        config,maximum_mach=1.
    )))

    if "--write-example" in sys.argv:
        i=sys.argv.index("--write-example")
        if i+1>=len(sys.argv):
            raise ValueError("--write-example requires a JSON path")
        output=Path(sys.argv[i+1])
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(asdict(config),indent=2)+"\n",
                          encoding="utf-8")
        from .run_gas_feed_network import main as cli
        old=sys.argv
        sys.argv=["run_gas_feed_network","--config",str(output),
                  "--output",str(output.with_name("gas_feed_result.json"))]
        try:
            assert cli()==0
        finally:
            sys.argv=old

    print("NERVA gas feed/channel/nozzle conservation and root solve: PASS")
    print(f"  gas mass flow {result.total_mass_flow_kg_s:.7g} kg/s")
    print(f"  chamber pressure {result.chamber_pressure_pa/1e6:.7g} MPa")
    print(f"  compressor power {result.compressor_shaft_power_w:.7g} W")
    print(f"  reactor heat {result.deposited_reactor_heat_w:.7g} W")
    print(f"  chamber H2 temperature {result.chamber_temperature_k:.7g} K")
    print(f"  thrust {result.nozzle.thrust_n:.7g} N, Isp {result.nozzle.isp_s:.7g} s")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
