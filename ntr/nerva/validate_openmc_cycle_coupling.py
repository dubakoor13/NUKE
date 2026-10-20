"""Power-partition and engine-cycle closure tests for the OpenMC heating bridge.

All inputs here are synthetic conservation fixtures, NOT independent
real OpenMC transport nor a validated historical NERVA model.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .openmc_cycle_coupling import (
    check_reconstruction_axial_edges, map_openmc_heating_to_cycle,
)
from .resolved_transient import ResolvedHeating, read_resolved_heating
from .turbine_bleed_cycle import solve_turbine_bleed
from .validate_hot_lh2_cycle import hot_case
from .validate_regenerative_cooling import network
from .validate_resolved_transient import write_synthetic_reconstructions
from .validate_turbine_bleed_cycle import sample_turbine


def rejects(fn) -> None:
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("nonconservative OpenMC/engine mapping accepted")


def main() -> int:
    core = hot_case()
    net = network()
    # Exact axial power profiles of the established *synthetic* full-cycle
    # test, but now they enter through the resolved OpenMC source fields.
    fw = np.broadcast_to(
        np.array([600000., 570000., 480000.])/19.,
        (1,19,3),
    ).copy()
    fd = np.broadcast_to(
        np.array([50000., 50000., 50000.])/19.,
        (1,19,3),
    ).copy()
    ts = np.array([[20000.,30000.,30000.]])
    sd = np.full((1,3),65000./3.)
    rd = np.full((1,3),85000./3.)
    unresolved = 250000.
    power = sum(float(x.sum()) for x in (fw,fd,ts,sd,rd))+unresolved
    data = ResolvedHeating(
        fw,fd,np.full((1,19),0.02),ts,sd,rd,np.array([0.05]),
        power,unresolved,
    )
    data.validate()
    mapped_core, mapped_net, ledger = map_openmc_heating_to_cycle(
        data,core,net,
        fuel_solid_transfer_fraction=1.,
        tie_solid_transfer_fraction=0.,
        tie_solid_supply_fraction=0.5,
        nozzle_wall_heat_w=110000.,
        reflector_wall_heat_w=90000.,
    )
    assert mapped_core.channel_wall_power_w == core.channel_wall_power_w
    assert mapped_core.channel_direct_hydrogen_power_w == core.channel_direct_hydrogen_power_w
    assert mapped_core.preheater_thermal_power_w == core.preheater_thermal_power_w
    assert mapped_net.total_heat_w == net.total_heat_w
    assert ledger.axial_fuel_channels_per_element == 19
    assert abs(ledger.power_balance_error_w) < 1e-7
    assert math.isclose(ledger.reactor_power_w,
                        ledger.routed_openmc_power_w +
                        ledger.retained_or_unmodeled_openmc_power_w,
                        rel_tol=1e-12)
    assert math.isclose(ledger.cycle_heating_power_w,
                        ledger.routed_openmc_power_w +
                        ledger.prescribed_nonopenmc_wall_heat_w,
                        rel_tol=1e-12)
    assert math.isclose(ledger.tie_solid_untransferred_w,80000.)

    # The mapped source fields can drive the EXISTING nonlinear
    # pump/turbine/header/core/nozzle operating-point solver unchanged.
    turbine=sample_turbine()
    reference=solve_turbine_bleed(core,turbine,cooling_network=net)
    coupled=solve_turbine_bleed(mapped_core,turbine,cooling_network=mapped_net)
    assert math.isclose(coupled.tank_mass_flow_kg_s,
                        reference.tank_mass_flow_kg_s,rel_tol=2e-8)
    assert math.isclose(coupled.chamber_pressure_pa,
                        reference.chamber_pressure_pa,rel_tol=2e-8)
    assert abs(coupled.total_cycle_energy_residual_w)<1e-3
    assert abs(coupled.shaft_power_residual_w)<1e-4

    # Moving nuclear heat from solid inventory into tie coolant must
    # conserve *total reactor power* and change only tie coolant
    # power, not fuel direct-H2 deposition or prescribed jacket heat.
    changed_core, changed_net, changed_ledger = map_openmc_heating_to_cycle(
        data,core,net,
        fuel_solid_transfer_fraction=1.,
        tie_solid_transfer_fraction=0.5,
        tie_solid_supply_fraction=0.25,
        nozzle_wall_heat_w=110000.,
        reflector_wall_heat_w=90000.,
    )
    assert math.isclose(changed_net.tie_supply.heat_power_w, 75000.)
    assert math.isclose(changed_net.tie_return.heat_power_w,115000.)
    assert changed_core.channel_wall_power_w==mapped_core.channel_wall_power_w
    assert math.isclose(changed_ledger.retained_or_unmodeled_openmc_power_w,
                        ledger.retained_or_unmodeled_openmc_power_w-40000.)
    assert abs(changed_ledger.power_balance_error_w)<1e-7

    # Never silently move fuel heating into upstream cooling.
    partial_core, partial_net, partial_ledger = map_openmc_heating_to_cycle(
        data,core,net,
        fuel_solid_transfer_fraction=0.5,
        tie_solid_transfer_fraction=0.,
        tie_solid_supply_fraction=0.5,
        nozzle_wall_heat_w=110000.,
        reflector_wall_heat_w=90000.,
    )
    assert math.isclose(partial_ledger.fuel_solid_untransferred_w,825000.)
    assert partial_net.total_heat_w==mapped_net.total_heat_w
    assert math.isclose(sum(partial_core.channel_wall_power_w),825000.)

    rejects(lambda:map_openmc_heating_to_cycle(
        data,core,net,fuel_solid_transfer_fraction=1.1,
        tie_solid_transfer_fraction=0.,tie_solid_supply_fraction=0.,
        nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.))
    rejects(lambda:map_openmc_heating_to_cycle(
        data,core,net,fuel_solid_transfer_fraction=1.,
        tie_solid_transfer_fraction=0.,tie_solid_supply_fraction=-0.1,
        nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.))
    rejects(lambda:map_openmc_heating_to_cycle(
        data,core,net,fuel_solid_transfer_fraction=1.,
        tie_solid_transfer_fraction=0.,tie_solid_supply_fraction=0.,
        nozzle_wall_heat_w=-1.,reflector_wall_heat_w=90000.))
    rejects(lambda:map_openmc_heating_to_cycle(
        data,replace(core,z_edges_m=(0.,0.1,0.2)),
        net,fuel_solid_transfer_fraction=1.,
        tie_solid_transfer_fraction=0.,tie_solid_supply_fraction=0.,
        nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.))

    # Reuse actual existing NPZ file format rather than bypassing the
    # reconstruction reader; all normalized mesh source partitions close.
    with tempfile.TemporaryDirectory() as temp:
        mesh, fuel, tie = write_synthetic_reconstructions(Path(temp))
        reconstructed = read_resolved_heating(fuel,tie,mesh)
        aligned = replace(core,z_edges_m=tuple(np.linspace(0.,1.,4)))
        check_reconstruction_axial_edges(fuel,tie,aligned)
        small_core,small_net,small_ledger = map_openmc_heating_to_cycle(
            reconstructed,aligned,net,
            fuel_solid_transfer_fraction=0.8,
            tie_solid_transfer_fraction=0.5,
            tie_solid_supply_fraction=0.4,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
        )
        assert abs(small_ledger.power_balance_error_w)<1e-7
        assert math.isclose(
            small_ledger.routed_openmc_power_w +
            small_ledger.retained_or_unmodeled_openmc_power_w,
            reconstructed.total_openmc_power_w,rel_tol=1e-12,
        )
        assert small_core.channel_direct_hydrogen_power_w==(32.,32.,32.)
        assert math.isclose(small_net.tie_supply.heat_power_w,60.)
        assert math.isclose(small_net.tie_return.heat_power_w,60.)
        rejects(lambda:check_reconstruction_axial_edges(fuel,tie,core))
        rejects(lambda:map_openmc_heating_to_cycle(
            reconstructed,replace(aligned,z_edges_m=(0.,0.1,0.2)),
            net,fuel_solid_transfer_fraction=1.,
            tie_solid_transfer_fraction=0.,tie_solid_supply_fraction=0.5,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.))

    print("NERVA OpenMC reconstructed heating -> engine/cooling power closure: PASS")
    print(f"  reactor heating: {ledger.reactor_power_w:.8g} W")
    print(f"  routed nuclear heating: {ledger.routed_openmc_power_w:.8g} W")
    print(f"  retained/unmodeled nuclear heating: {ledger.retained_or_unmodeled_openmc_power_w:.8g} W")
    print(f"  externally prescribed wall transfer: {ledger.prescribed_nonopenmc_wall_heat_w:.8g} W")
    print(f"  cycle first law residual: {coupled.total_cycle_energy_residual_w:.8g} W")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
