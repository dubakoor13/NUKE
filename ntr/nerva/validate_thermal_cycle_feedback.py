"""Independent power/source-accounting regression for implicit wall coupling.

Synthetic OpenMC-sized fuel/tie heating; checks transient solid storage,
variable-enthalpy coolant update, and source flow into existing cycle.
No historical NERVA neutronics or geometry is inferred by this test.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .openmc_cycle_coupling import check_reconstruction_axial_edges
from .resolved_transient import (
    ResolvedParameters, read_resolved_heating, initial_resolved_state,
    advance_resolved,
)
from .thermal_cycle_feedback import project_resolved_wall_power
from .validate_resolved_transient import write_synthetic_reconstructions
from .validate_hot_lh2_cycle import hot_case
from .validate_regenerative_cooling import network


def expect_rejected(func):
    try:
        func()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("incorrect wall source mapping accepted")


def main() -> int:
    with tempfile.TemporaryDirectory() as temp:
        mesh,fuel,tie = write_synthetic_reconstructions(Path(temp))
        heat=read_resolved_heating(fuel,tie,mesh)
        core=replace(hot_case(),z_edges_m=(0.,1./3.,2./3.,1.))
        cooling=network()
        params=ResolvedParameters(
            fuel_solid_capacity_j_k=5000.,
            fuel_h2_capacity_j_k=500.,
            fuel_ua_w_k=100.,
            tie_solid_capacity_j_k=3000.,
            tie_supply_h2_capacity_j_k=600.,
            tie_return_h2_capacity_j_k=700.,
            tie_supply_ua_w_k=80.,
            tie_return_ua_w_k=90.,
        )
        check_reconstruction_axial_edges(fuel,tie,core)
        old=initial_resolved_state(heat,solid_temperature_k=500.,
                                   gas_temperature_k=300.)
        dt=0.1
        step=advance_resolved(heat,old,params,300.,dt)
        mapped,net,ledger=project_resolved_wall_power(
            heat,old,step,params,core,cooling,dt,
            nozzle_wall_heat_w=110000.,
            reflector_wall_heat_w=90000.,
        )
        assert np.max(step.state.fuel_solid_t_k)>500.
        assert abs(ledger.resolved_thermal_step_energy_residual_j)<1e-4
        assert abs(ledger.nuclear_power_balance_error_w)<1e-6
        assert abs(ledger.maximum_solid_balance_error_w)<1e-6
        assert math.isclose(
            ledger.reactor_power_w,
            ledger.nuclear_heat_to_coolant_w+
            ledger.solid_energy_storage_rate_w+
            ledger.unresolved_power_w,rel_tol=1e-12,
        )
        assert math.isclose(
            ledger.engine_cycle_heating_power_w,
            ledger.nuclear_heat_to_coolant_w+
            ledger.prescribed_nonopenmc_wall_heat_w,
            rel_tol=1e-12,
        )
        assert math.isclose(
            ledger.fuel_direct_h2_w,float(np.sum(heat.fuel_h2_w)))
        assert math.isclose(
            ledger.tie_supply_direct_h2_w,float(np.sum(heat.tie_supply_h2_w)))
        assert math.isclose(
            ledger.tie_return_direct_h2_w,float(np.sum(heat.tie_return_h2_w)))
        assert math.isclose(
            ledger.fuel_wall_to_h2_w,
            params.fuel_ua_w_k*float(np.sum(
                step.state.fuel_solid_t_k-step.state.fuel_h2_t_k
            )),rel_tol=1e-12)
        assert math.isclose(
            sum(mapped.channel_wall_power_w),
            ledger.fuel_wall_to_h2_w,rel_tol=1e-12)
        assert math.isclose(
            sum(mapped.channel_direct_hydrogen_power_w),
            ledger.fuel_direct_h2_w,rel_tol=1e-12)
        assert math.isclose(net.total_heat_w,mapped.preheater_thermal_power_w)
        assert math.isclose(
            net.tie_supply.heat_power_w,
            ledger.tie_supply_wall_to_h2_w + ledger.tie_supply_direct_h2_w,
        )
        assert math.isclose(
            net.tie_return.heat_power_w,
            ledger.tie_return_wall_to_h2_w + ledger.tie_return_direct_h2_w,
        )
        # The existing percentage model doesn't resolve solid inventory.
        # A deposited watt need not become coolant watt in the same timestep.
        assert not math.isclose(
            ledger.fuel_wall_to_h2_w,float(np.sum(heat.fuel_solid_w)),
            rel_tol=1e-5,
        )
        # Stronger solid-gas conductance changes the delivered watts,
        # while fixed OpenMC nuclear deposition stays exactly conserved.
        better=replace(params,fuel_ua_w_k=200.)
        advanced=advance_resolved(heat,old,better,300.,dt)
        _,_,changed=project_resolved_wall_power(
            heat,old,advanced,better,core,cooling,dt,
            nozzle_wall_heat_w=110000.,
            reflector_wall_heat_w=90000.,
        )
        assert changed.fuel_wall_to_h2_w>ledger.fuel_wall_to_h2_w
        assert abs(changed.nuclear_power_balance_error_w)<1e-6
        assert changed.fuel_solid_storage_rate_w<ledger.fuel_solid_storage_rate_w

        # Fail closed when a solved source violates wall/mass transfer
        # identity, even if its global reactor power is unchanged.
        inconsistent=replace(
            step,state=replace(
                step.state,
                fuel_solid_t_k=step.state.fuel_solid_t_k+5.,
            ),
        )
        expect_rejected(lambda:project_resolved_wall_power(
            heat,old,inconsistent,params,core,cooling,dt,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
        ))
        reverse=replace(
            step,state=replace(
                step.state,
                fuel_solid_t_k=np.full_like(step.state.fuel_solid_t_k,250.),
            ),
        )
        expect_rejected(lambda:project_resolved_wall_power(
            heat,old,reverse,params,core,cooling,dt,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
        ))
        expect_rejected(lambda:project_resolved_wall_power(
            heat,old,step,params,core,cooling,-dt,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
        ))
        expect_rejected(lambda:project_resolved_wall_power(
            heat,old,step,params,core,cooling,dt,
            nozzle_wall_heat_w=-1.,reflector_wall_heat_w=90000.,
        ))
        broken=replace(
            step,state=replace(
                step.state,
                unresolved_energy_j=step.state.unresolved_energy_j+20.,
            ),
        )
        expect_rejected(lambda:project_resolved_wall_power(
            heat,old,broken,params,core,cooling,dt,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
        ))

    print("NERVA implicit solid/H2 wall-transfer to cycle ledger: PASS")
    print(f"  nuclear input: {ledger.reactor_power_w:.9g} W")
    print(f"  nuclear coolant delivery: {ledger.nuclear_heat_to_coolant_w:.9g} W")
    print(f"  net solid inventory rate: {ledger.solid_energy_storage_rate_w:.9g} W")
    print(f"  unresolved power: {ledger.unresolved_power_w:.9g} W")
    print(f"  energy ledger residual: {ledger.nuclear_power_balance_error_w:.9g} W")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
