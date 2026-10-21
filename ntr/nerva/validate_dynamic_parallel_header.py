"""Dynamic nozzle/reflector pressure-closed branch + tie Euler + H2 header.

Uses actual CoolProp shaft-closed engine reference and native cryogenic
leg momentum/heat equations. Branch flow reacts to evolving header
pressure; tie Euler flux is independent. Combined header mass/energy
close. The pump reservoir and core mass demand remain prescribed and
their open inlet imbalance is NEVER labeled full engine convergence.
Synthetic nuclear power only.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .distributed_tie_euler import (
    DistributedTieParameters,initial_distributed_tie_state,
)
from .dynamic_header_tie import (
    DynamicHeaderParameters,DynamicHeaderTieState,initial_dynamic_header,
)
from .dynamic_parallel_header import (
    solve_nozzle_reflector_to_header,advance_dynamic_parallel_tie_header,
)
from .distributed_tie_engine_boundary import engine_tie_reservoirs
from .isochoric_tie import IsochoricTieParameters
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating
from .turbine_bleed_cycle import solve_turbine_bleed
from .validate_hot_lh2_cycle import hot_case
from .validate_regenerative_cooling import network
from .validate_turbine_bleed_cycle import sample_turbine
from .validate_resolved_transient import write_synthetic_reconstructions


def rejects(fn):
    try:
        fn()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("inadmissible dynamic branch condition accepted")


def main()->int:
    eos=HydrogenEOS()
    net=network()
    shaft=solve_turbine_bleed(
        hot_case(),sample_turbine(),cooling_network=net,eos=eos,
    )
    bc=engine_tie_reservoirs(shaft)
    hydraulic=shaft.regenerative_cooling
    assert hydraulic is not None
    nominal=solve_nozzle_reflector_to_header(
        net,shaft,bc.steady_common_header_pressure_pa,eos=eos
    )
    assert nominal.mass_flow_kg_s>0
    assert abs(nominal.outlet_pressure_residual_pa)<1.
    assert math.isclose(nominal.mass_flow_kg_s,
                        hydraulic.nozzle_reflector_mass_flow_kg_s,
                        rel_tol=1e-3)
    assert math.isclose(
        nominal.outlet_stagnation_enthalpy_j_kg,
        hydraulic.legs[1].outlet_stagnation_enthalpy_j_kg,
        rel_tol=1e-3,
    )
    # Physical monotonic pressure response using the native actual
    # jacket+reflector leg momentum/energy equations.
    low=solve_nozzle_reflector_to_header(
        net,shaft,bc.steady_common_header_pressure_pa-500.,eos=eos
    )
    high=solve_nozzle_reflector_to_header(
        net,shaft,bc.steady_common_header_pressure_pa+500.,eos=eos
    )
    assert low.mass_flow_kg_s>nominal.mass_flow_kg_s
    assert nominal.mass_flow_kg_s>high.mass_flow_kg_s
    assert low.outlet_stagnation_enthalpy_j_kg<(
        high.outlet_stagnation_enthalpy_j_kg
    )
    assert abs(low.outlet_pressure_residual_pa)<1.
    assert abs(high.outlet_pressure_residual_pa)<1.
    assert nominal.jacket.mass_flow_kg_s==nominal.reflector.mass_flow_kg_s
    assert abs(nominal.jacket.energy_residual_w)<0.1
    assert abs(nominal.reflector.energy_residual_w)<0.1
    assert abs(nominal.jacket.momentum_residual_pa)<1.
    assert abs(nominal.reflector.momentum_residual_pa)<1.

    with tempfile.TemporaryDirectory() as td:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(td))
        heating=read_resolved_heating(fuel,tie,mesh)
        params=DistributedTieParameters(
            thermal=IsochoricTieParameters(
                supply_cell_volume_m3=0.0001,
                return_cell_volume_m3=0.0001,
                solid_capacity_j_k=3000.,
                supply_ua_w_k=0.4,
                return_ua_w_k=0.4,
            ),
            flow_area_m2=0.0002,
            darcy_friction_factor=0.04,
            turnaround_loss_coefficient=0.6,
        )
        header=DynamicHeaderParameters(volume_m3=0.0001)
        tie_old=initial_distributed_tie_state(
            heating,params,bc.upstream_pressure_pa,
            bc.upstream_stagnation_enthalpy_j_kg,
            solid_temperature_k=35.,eos=eos,
        )
        state=DynamicHeaderTieState(
            tie_old,initial_dynamic_header(shaft,header,eos=eos),
        )
        dt=1e-7
        first=advance_dynamic_parallel_tie_header(
            heating,state,params,header,net,shaft,dt,eos=eos,
        )
        res=first.combined
        assert res.dynamic_nozzle_branch_pressure_response_active
        assert not first.full_shaft_flow_and_pressure_coupling_converged
        assert first.upstream_pump_pressure_held_fixed
        assert first.downstream_engine_mass_demand_held_fixed
        assert not res.full_shaft_cycle_feedback_converged
        assert abs(first.nozzle_branch.outlet_pressure_residual_pa)<1.
        assert math.isclose(
            res.nozzle_branch_inlet_flow_kg_s,
            first.nozzle_branch.mass_flow_kg_s,rel_tol=1e-12,
        )
        assert math.isclose(
            res.nozzle_branch_inlet_enthalpy_j_kg,
            first.nozzle_branch.outlet_stagnation_enthalpy_j_kg,
            rel_tol=1e-12,
        )
        assert math.isclose(
            first.pump_inflow_mismatch_kg_s,
            res.nozzle_branch_inlet_flow_kg_s+
            res.tie_inlet_flow_kg_s-shaft.tank_mass_flow_kg_s,
            abs_tol=1e-10,
        )
        assert math.isclose(
            res.header_inventory_rate_kg_s,
            res.nozzle_branch_inlet_flow_kg_s+
            res.tie_header_interface_flow_kg_s-
            res.fixed_engine_discharge_flow_kg_s,
            abs_tol=1e-8,
        )
        assert abs(res.header_mass_residual_kg_s)<1e-8
        assert abs(res.header_internal_energy_residual_j)<1e-5
        assert abs(res.combined_tie_header_mass_residual_kg_s)<1e-8
        assert abs(res.combined_tie_solid_header_energy_residual_j)<1e-5
        assert abs(res.tie_step.total_energy_residual_j)<1e-5

        second=advance_dynamic_parallel_tie_header(
            heating,res.state,params,header,net,shaft,dt,eos=eos,
        )
        assert abs(second.nozzle_branch.outlet_pressure_residual_pa)<1.
        assert math.isclose(
            second.nozzle_branch.desired_header_pressure_pa,
            res.header_new_pressure_pa,rel_tol=1e-12,
        )
        assert math.isclose(
            second.combined.header_old_pressure_pa,
            res.header_new_pressure_pa,rel_tol=1e-12,
        )
        assert second.combined.nozzle_branch_inlet_flow_kg_s!=(
            first.combined.nozzle_branch_inlet_flow_kg_s
        )
        assert abs(second.combined.combined_tie_header_mass_residual_kg_s)<1e-8
        assert abs(second.combined.combined_tie_solid_header_energy_residual_j)<1e-5
        # Solve at the perturbed header, and require corresponding
        # hydraulic flow; don't just recopy steady nozzle reference.
        assert math.isclose(
            second.nozzle_branch.mass_flow_kg_s,
            solve_nozzle_reflector_to_header(
                net,shaft,res.header_new_pressure_pa,eos=eos
            ).mass_flow_kg_s,rel_tol=1e-12,
        )

        rejects(lambda:solve_nozzle_reflector_to_header(
            net,shaft,bc.upstream_pressure_pa,eos=eos
        ))
        rejects(lambda:solve_nozzle_reflector_to_header(
            net,shaft,-1.,eos=eos
        ))
        rejects(lambda:solve_nozzle_reflector_to_header(
            net,shaft,1.,eos=eos
        ))
        rejects(lambda:solve_nozzle_reflector_to_header(
            net,shaft,bc.steady_common_header_pressure_pa,
            minimum_reference_fraction=1.2,eos=eos,
        ))
        rejects(lambda:advance_dynamic_parallel_tie_header(
            heating,state,params,header,net,shaft,-dt,eos=eos,
        ))
        rejects(lambda:advance_dynamic_parallel_tie_header(
            heating,state,params,header,net,
            replace(shaft,regenerative_cooling=None),dt,eos=eos,
        ))
    print("NERVA pressure-conditioned nozzle branch / Euler tie / dynamic H2 header: PASS")
    print(f"  steady nozzle reference flow: {hydraulic.nozzle_reflector_mass_flow_kg_s:.9g} kg/s")
    print(f"  dynamic nozzle flow, step 1: {first.nozzle_branch.mass_flow_kg_s:.9g} kg/s")
    print(f"  dynamic nozzle flow, step 2: {second.nozzle_branch.mass_flow_kg_s:.9g} kg/s")
    print(f"  dynamic header pressure, step 2: {second.combined.header_new_pressure_pa:.9g} Pa")
    print(f"  pump-header flow mismatch (not solved): {second.pump_inflow_mismatch_kg_s:.9g} kg/s")
    print(f"  combined mass residual: {second.combined.combined_tie_header_mass_residual_kg_s:.9g} kg/s")
    print(f"  combined energy residual: {second.combined.combined_tie_solid_header_energy_residual_j:.9g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
