"""Synthetic numerical validation of fuel/tie FLOW feedback and fixed-point gates.

The mock engine used here is explicitly synthetic. Real nonlinear NERVA
shaft-cycle operation is covered by its own existing physical regressions.
No neutron transport or NASA hardware validation occurs in this test.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np

from .resolved_transient import (
    ResolvedParameters, initial_resolved_state, read_resolved_heating,
    advance_resolved,
)
from .thermal_flow_coupling import couple_resolved_thermal_flows
from .validate_resolved_transient import write_synthetic_reconstructions
from .validate_hot_lh2_cycle import hot_case
from .validate_regenerative_cooling import network
from .validate_turbine_bleed_cycle import sample_turbine


def rejects(action):
    try:
        action()
    except (RuntimeError,ValueError):
        return
    raise AssertionError("invalid or unconverged flow-coupling accepted")


def main() -> int:
    with tempfile.TemporaryDirectory() as temp:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(temp))
        heating=read_resolved_heating(fuel,tie,mesh)
        heating=replace(
            heating,
            fuel_flow_kg_s=np.full_like(heating.fuel_flow_kg_s,0.08/38.),
            tie_flow_kg_s=np.full_like(heating.tie_flow_kg_s,0.022/2.),
        )
        heating.validate()
        core=replace(hot_case(),z_edges_m=(0.,1./3.,2./3.,1.))
        cooling=network()
        turbine=sample_turbine()
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
        old=initial_resolved_state(heating,
                                   solid_temperature_k=500.,
                                   gas_temperature_k=300.)
        calls=[]

        def synthetic_engine(mapped_core,bleed,*,cooling_network,chemistry_mode):
            assert bleed is turbine
            assert chemistry_mode=="frozen"
            assert math.isclose(mapped_core.preheater_thermal_power_w,
                                cooling_network.total_heat_w,rel_tol=1e-12)
            # Weak dependence on independently solved wall-transfer powers.
            # This is a deterministic numerical fixed-point fixture,
            # not an approximation to the real pump/turbine physics.
            f=0.060+2e-9*sum(mapped_core.channel_wall_power_w)
            tieflow=0.020+1e-9*(
                cooling_network.tie_supply.heat_power_w+
                cooling_network.tie_return.heat_power_w
            )
            calls.append((f,tieflow))
            return SimpleNamespace(
                core_mass_flow_kg_s=f,
                regenerative_cooling=SimpleNamespace(
                    tie_tube_mass_flow_kg_s=tieflow
                ),
                warm_header_temperature_k=300.,
                total_cycle_energy_residual_w=0.,
            )

        result=couple_resolved_thermal_flows(
            heating,old,params,core,cooling,turbine,300.,0.1,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
            max_iterations=22,
            relative_flow_tolerance=0.001,
            profile_tolerance_k=0.5,
            relaxation=0.5,
            engine_solver=synthetic_engine,
        )
        assert result.iterations>=2
        assert result.iterations==len(calls)
        assert result.max_relative_flow_error<=0.001
        assert result.max_thermal_profile_update_k<=0.5
        assert math.isclose(
            result.thermal_heating.total_fuel_flow_kg_s,
            result.engine.core_mass_flow_kg_s,rel_tol=0.001)
        assert math.isclose(
            result.thermal_heating.total_tie_flow_kg_s,
            result.engine.regenerative_cooling.tie_tube_mass_flow_kg_s,
            rel_tol=0.001)
        assert result.temperature_boundary_not_closed is True
        assert result.transport_performed is False
        # Within-family channel and tie flow RATIOS remain invariant,
        # avoiding spurious redistribution of per-channel OpenMC sources.
        assert np.allclose(
            result.thermal_heating.fuel_flow_kg_s /
            result.thermal_heating.total_fuel_flow_kg_s,
            heating.fuel_flow_kg_s/heating.total_fuel_flow_kg_s,
            atol=1e-14,rtol=1e-12)
        assert np.allclose(
            result.thermal_heating.tie_flow_kg_s /
            result.thermal_heating.total_tie_flow_kg_s,
            heating.tie_flow_kg_s/heating.total_tie_flow_kg_s,
            atol=1e-14,rtol=1e-12)
        assert math.isclose(
            result.power_ledger.reactor_power_w,
            result.power_ledger.nuclear_heat_to_coolant_w+
            result.power_ledger.solid_energy_storage_rate_w+
            result.power_ledger.unresolved_power_w,
            abs_tol=1e-5)
        assert abs(result.power_ledger.nuclear_power_balance_error_w)<1e-5
        assert abs(result.thermal_step.energy_residual_j)<1e-4
        assert all(
            abs(rec.nuclear_power_balance_error_w)<1e-5
            for rec in result.diagnostics
        )
        assert all(
            rec.cycle_energy_balance_error_w==0.
            for rec in result.diagnostics
        )
        # Fundamental time-integrator guarantee: last candidate is a
        # SINGLE timestep from OLD state, not n successive steps.
        direct=advance_resolved(
            result.thermal_heating,old,params,300.,0.1
        )
        for name in ("fuel_solid_t_k","fuel_h2_t_k","tie_solid_t_k",
                     "tie_supply_t_k","tie_return_t_k"):
            assert np.allclose(
                getattr(result.thermal_step.state,name),
                getattr(direct.state,name),rtol=1e-12,atol=1e-10,
            )
        assert math.isclose(
            result.power_ledger.engine_cycle_heating_power_w,
            result.power_ledger.nuclear_heat_to_coolant_w+200000.,
            rel_tol=1e-12,
        )

        # Flow roots that do not satisfy both groups fail closed.
        rejects(lambda:couple_resolved_thermal_flows(
            heating,old,params,core,cooling,turbine,300.,0.1,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
            max_iterations=2,relative_flow_tolerance=1e-8,
            profile_tolerance_k=1e-10,engine_solver=synthetic_engine,
        ))
        rejects(lambda:couple_resolved_thermal_flows(
            heating,old,params,core,cooling,turbine,300.,0.1,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
            relaxation=0.,engine_solver=synthetic_engine,
        ))
        def no_tie(*args,**kwargs):
            return SimpleNamespace(core_mass_flow_kg_s=0.05,
                                   regenerative_cooling=None)
        rejects(lambda:couple_resolved_thermal_flows(
            heating,old,params,core,cooling,turbine,300.,0.1,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
            engine_solver=no_tie,
        ))

    print("NERVA resolved thermal/core/tie mass-flow fixed point: PASS")
    print(f"  iterations: {result.iterations}")
    print(f"  relative flow error: {result.max_relative_flow_error:.8g}")
    print(f"  thermal profile update: {result.max_thermal_profile_update_k:.8g} K")
    print(f"  reactor power residual: {result.power_ledger.nuclear_power_balance_error_w:.8g} W")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
