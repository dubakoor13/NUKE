"""Real-CoolProp LH2 pump head / finite shaft-rotor energy verification.

Checks existing *synthetic* shaft-closed pump reference, live pressure
and speed perturbations, EOS pump isentropic efficiency, fluid/rotor
total-energy consistency, dynamic manifold mass, independent
two-step feedback, and invalid pump operating envelopes. No NASA
measured turbopump curve or real OpenMC data are used.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .dynamic_pump_shaft import (
    DynamicPumpParameters,DynamicPumpState,
    pump_delivery_at_state,advance_dynamic_pump_tie,
)
from .dual_header_tie import (
    DualHeaderTieState,initial_pump_manifold,
)
from .dynamic_header_tie import (
    DynamicHeaderParameters,DynamicHeaderTieState,
    initial_dynamic_header,header_properties,
)
from .distributed_tie_euler import (
    DistributedTieParameters,initial_distributed_tie_state,
)
from .distributed_tie_engine_boundary import engine_tie_reservoirs
from .isochoric_tie import IsochoricTieParameters
from .lh2_feed_cycle import HydrogenEOS
from .turbine_bleed_cycle import solve_turbine_bleed
from .validate_hot_lh2_cycle import hot_case
from .validate_regenerative_cooling import network
from .validate_turbine_bleed_cycle import sample_turbine
from .validate_resolved_transient import write_synthetic_reconstructions
from .resolved_transient import read_resolved_heating


def rejects(f):
    try:
        f()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("invalid pump hydraulic or shaft state accepted")


def validate_balance(step, old, params, dt, net, heat):
    delivery=step.pump
    fluid=step.fluid
    assert step.pump_mass_flow_follows_pressure_speed
    assert step.rotor_inertia_dynamic
    assert step.turbine_shaft_power_reference_fixed
    assert not step.whole_engine_cycle_converged
    assert not step.historical_nasa_pump_map_validated
    assert fluid.pump_flow_and_head_dynamic
    assert not fluid.pump_flow_and_head_are_reference_values
    assert abs(delivery.manifold_pressure_residual_pa)<1e-5
    assert abs(delivery.pump_enthalpy_power_residual_w)<1e-8
    assert abs(fluid.combined_system_mass_residual_kg_s)<1e-8
    assert abs(fluid.combined_system_total_energy_residual_j)<1e-5
    assert abs(step.rotor_energy_balance_residual_j)<1e-5
    assert abs(step.total_fluid_rotor_energy_balance_residual_j)<1e-5
    assert delivery.mass_flow_kg_s>0
    assert delivery.shaft_power_w>0
    assert math.isclose(
        step.rotor_power_storage_rate_w,
        step.reference_turbine_shaft_power_w-delivery.shaft_power_w,
        rel_tol=1e-8,abs_tol=1e-3,
    )
    assert math.isclose(
        step.rotor_final_energy_j,
        0.5*params.rotor_inertia_kg_m2*
        step.state.rotor_angular_speed_rad_s**2,
        rel_tol=1e-12,
    )
    assert math.isclose(
        fluid.pump_manifold_inventory_rate_kg_s,
        delivery.mass_flow_kg_s-
        fluid.parallel.nozzle_branch.mass_flow_kg_s-
        fluid.tie_supply_interface_mass_flow_kg_s,
        abs_tol=1e-8,
    )
    # Independent FLUID + SHAFT first law: use H2 tank enthalpy,
    # turbine drive into rotor, and external distributed thermal heat.
    oldhyd=old.hydraulic
    newhyd=fluid.state
    cs=heat.tie_solid_w.shape and 3000.
    fluid_storage=(
        (newhyd.pump.internal_energy_j-oldhyd.pump.internal_energy_j)+
        (newhyd.downstream.header.internal_energy_j-
         oldhyd.downstream.header.internal_energy_j)+
        float(np.sum(
            newhyd.downstream.tie.total_energy_j-
            oldhyd.downstream.tie.total_energy_j
        ))+
        cs*float(np.sum(
            newhyd.downstream.tie.solid_temperature_k-
            oldhyd.downstream.tie.solid_temperature_k
        ))
    )
    full_delta=fluid_storage+step.rotor_final_energy_j-step.rotor_initial_energy_j
    q_nozzle=net.nozzle_jacket.heat_power_w+net.reflector.heat_power_w
    q_nuclear=fluid.parallel.combined.tie_step.nuclear_deposition_w
    m_core=fluid.parallel.combined.fixed_engine_discharge_flow_kg_s
    h_header_old=fluid.parallel.combined.header_old_enthalpy_j_kg
    boundary=dt*(
        delivery.mass_flow_kg_s*delivery.tank_enthalpy_j_kg+
        step.reference_turbine_shaft_power_w-
        m_core*h_header_old+q_nozzle+q_nuclear
    )
    assert abs(full_delta-boundary)<1e-5


def main()->int:
    eos=HydrogenEOS()
    core=hot_case()
    net=network()
    shaft=solve_turbine_bleed(
        core,sample_turbine(),eos=eos,cooling_network=net
    )
    nominal=engine_tie_reservoirs(shaft)
    params=DynamicPumpParameters(
        reference_angular_speed_rad_s=1200.,
        rotor_inertia_kg_m2=0.05,
    )
    fixed=pump_delivery_at_state(
        core,params,1200.,nominal.upstream_pressure_pa,eos=eos,
    )
    assert math.isclose(
        fixed.mass_flow_kg_s,shaft.tank_mass_flow_kg_s,
        rel_tol=2e-7,abs_tol=1e-8,
    )
    assert math.isclose(
        fixed.delivered_stagnation_enthalpy_j_kg,
        nominal.upstream_stagnation_enthalpy_j_kg,
        rel_tol=2e-7,abs_tol=1e-5,
    )
    assert math.isclose(
        fixed.shaft_power_w,shaft.pump_required_power_w,
        rel_tol=2e-7,abs_tol=1e-5,
    )
    assert fixed.available_npsh_m>0

    higher_pressure=pump_delivery_at_state(
        core,params,1200.,nominal.upstream_pressure_pa+500.,eos=eos
    )
    lower_pressure=pump_delivery_at_state(
        core,params,1200.,nominal.upstream_pressure_pa-500.,eos=eos
    )
    assert higher_pressure.mass_flow_kg_s<fixed.mass_flow_kg_s
    assert lower_pressure.mass_flow_kg_s>fixed.mass_flow_kg_s
    faster=pump_delivery_at_state(
        core,params,1201.2,nominal.upstream_pressure_pa,eos=eos,
    )
    slower=pump_delivery_at_state(
        core,params,1198.8,nominal.upstream_pressure_pa,eos=eos,
    )
    assert faster.mass_flow_kg_s>fixed.mass_flow_kg_s
    assert slower.mass_flow_kg_s<fixed.mass_flow_kg_s

    with tempfile.TemporaryDirectory() as td:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(td))
        heat=read_resolved_heating(fuel,tie,mesh)
        thermal=IsochoricTieParameters(
            supply_cell_volume_m3=0.0001,
            return_cell_volume_m3=0.0001,
            solid_capacity_j_k=3000.,
            supply_ua_w_k=0.4,return_ua_w_k=0.4,
        )
        flow=DistributedTieParameters(
            thermal=thermal,flow_area_m2=0.0002,
            darcy_friction_factor=0.04,
            turnaround_loss_coefficient=0.6,
        )
        up=DynamicHeaderParameters(volume_m3=0.0001)
        down=DynamicHeaderParameters(volume_m3=0.0001)
        tie_initial=initial_distributed_tie_state(
            heat,flow,nominal.upstream_pressure_pa,
            nominal.upstream_stagnation_enthalpy_j_kg,
            solid_temperature_k=35.,eos=eos,
        )
        initial=DynamicPumpState(
            rotor_angular_speed_rad_s=params.reference_angular_speed_rad_s,
            hydraulic=DualHeaderTieState(
                pump=initial_pump_manifold(shaft,up,eos=eos),
                downstream=DynamicHeaderTieState(
                    tie=tie_initial,
                    header=initial_dynamic_header(shaft,down,eos=eos),
                ),
            ),
        )
        dt=1e-7
        first=advance_dynamic_pump_tie(
            heat,initial,core,params,flow,up,down,net,shaft,dt,eos=eos,
        )
        validate_balance(first,initial,params,dt,net,heat)
        assert math.isclose(
            first.pump.mass_flow_kg_s,shaft.tank_mass_flow_kg_s,
            rel_tol=2e-7,
        )
        second=advance_dynamic_pump_tie(
            heat,first.state,core,params,flow,up,down,net,shaft,dt,eos=eos,
        )
        validate_balance(second,first.state,params,dt,net,heat)
        assert second.pump.mass_flow_kg_s!=first.pump.mass_flow_kg_s
        assert math.isclose(
            second.pump.manifold_pressure_pa,
            first.fluid.pump_new_pressure_pa,rel_tol=1e-11,
        )
        assert second.state.rotor_angular_speed_rad_s!=(
            first.state.rotor_angular_speed_rad_s
        )
        # Start slightly faster: this must change the physical pump
        # supply and subsequent common-header state even though
        # turbine reference delivered power is held unchanged.
        fast_initial=replace(initial,rotor_angular_speed_rad_s=1201.2)
        perturbed=advance_dynamic_pump_tie(
            heat,fast_initial,core,params,flow,up,down,net,shaft,dt,eos=eos,
        )
        validate_balance(perturbed,fast_initial,params,dt,net,heat)
        assert perturbed.pump.mass_flow_kg_s>first.pump.mass_flow_kg_s
        assert perturbed.fluid.state.pump.mass_kg!=first.fluid.state.pump.mass_kg
        # Changing rotor inertia modifies angular acceleration, not
        # instantaneous pump performance at the same speed.
        heavy=replace(params,rotor_inertia_kg_m2=params.rotor_inertia_kg_m2*20)
        heavy_result=advance_dynamic_pump_tie(
            heat,initial,core,heavy,flow,up,down,net,shaft,dt,eos=eos,
        )
        validate_balance(heavy_result,initial,heavy,dt,net,heat)
        assert math.isclose(
            first.pump.mass_flow_kg_s,
            heavy_result.pump.mass_flow_kg_s,rel_tol=1e-12,
        )
        assert abs(heavy_result.state.rotor_angular_speed_rad_s-1200.)<(
            abs(first.state.rotor_angular_speed_rad_s-1200.)
        )
        rejects(lambda:pump_delivery_at_state(
            core,params,1200.,core.tank_pressure_pa+core.pump_shutoff_pressure_rise_pa*1.01,eos=eos
        ))
        rejects(lambda:pump_delivery_at_state(
            core,params,1200.,-1.,eos=eos,
        ))
        rejects(lambda:pump_delivery_at_state(
            core,params,1200.*1.4,nominal.upstream_pressure_pa,eos=eos,
        ))
        rejects(lambda:advance_dynamic_pump_tie(
            heat,initial,core,params,flow,up,down,net,shaft,0.,eos=eos,
        ))
        rejects(lambda:advance_dynamic_pump_tie(
            heat,initial,core,params,flow,up,down,net,shaft,0.1,eos=eos,
        ))
        rejects(lambda:advance_dynamic_pump_tie(
            heat,initial,core,replace(params,rotor_inertia_kg_m2=-1.),
            flow,up,down,net,shaft,dt,eos=eos,
        ))
        rejects(lambda:advance_dynamic_pump_tie(
            heat,initial,core,params,flow,up,down,net,
            replace(shaft,regenerative_cooling=None),dt,eos=eos,
        ))

    print("NERVA dynamic LH2 pump-head/rotor + dual manifolds/tie Euler: PASS")
    print(f"  steady hydraulic reference = {shaft.tank_mass_flow_kg_s:.9g} kg/s")
    print(f"  pump transient flow step 2 = {second.pump.mass_flow_kg_s:.9g} kg/s")
    print(f"  stage-head pressure residual = {second.pump.manifold_pressure_residual_pa:.9g} Pa")
    print(f"  pump shaft power = {second.pump.shaft_power_w:.9g} W")
    print(f"  rotor energy residual = {second.rotor_energy_balance_residual_j:.9g} J")
    print(f"  global fluid+rotor energy residual = {second.total_fluid_rotor_energy_balance_residual_j:.9g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
