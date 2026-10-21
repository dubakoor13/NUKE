"""Real-H2 turbine admission / pump rotor / common-header energy verification.

Synthetic effective turbine area calibrated against a separately
shaft-closed operating point, NOT NASA hardware dimensions.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .dynamic_turbine_admission import (
    TurbineAdmission, calibrate_effective_admission,
    isentropic_hydrogen_admission_mass_flux,
    evaluate_dynamic_turbine, advance_dynamic_turbine_pump,
)
from .dynamic_pump_shaft import DynamicPumpParameters,DynamicPumpState
from .dual_header_tie import DualHeaderTieState,initial_pump_manifold
from .dynamic_header_tie import (
    DynamicHeaderParameters,DynamicHeaderTieState,initial_dynamic_header,
)
from .distributed_tie_engine_boundary import engine_tie_reservoirs
from .distributed_tie_euler import (
    DistributedTieParameters,initial_distributed_tie_state,
)
from .isochoric_tie import IsochoricTieParameters
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating
from .turbine_bleed_cycle import solve_turbine_bleed
from .validate_hot_lh2_cycle import hot_case
from .validate_regenerative_cooling import network
from .validate_turbine_bleed_cycle import sample_turbine
from .validate_resolved_transient import write_synthetic_reconstructions


def rejects(action):
    try:
        action()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("invalid turbine/H2 condition accepted")


def check(step,old,dt,network,params):
    assert step.moving_turbine_bleed_and_shaft_power
    assert step.pump_and_header_flow_pressure_feedback_active
    assert not step.core_nozzle_throat_mass_demand_dynamic
    assert not step.nasa_turbine_admission_area_calibrated
    assert not step.real_openmc_transport_executed
    p=step.pump_and_fluid
    t=step.turbine
    assert p.turbine_shaft_power_dynamic
    assert not p.turbine_shaft_power_reference_fixed
    assert p.fluid.pump_flow_and_head_dynamic
    assert not p.whole_engine_cycle_converged
    assert t.bleed_mass_flow_kg_s>0
    assert t.core_reference_remainder_mass_flow_kg_s>0
    assert t.turbine_gas_power_w>t.delivered_rotor_power_w>0
    assert t.mechanical_loss_w>=0
    assert math.isclose(
        t.bleed_mass_flow_kg_s+t.core_reference_remainder_mass_flow_kg_s,
        step.engine_reference_total_discharge_flow_kg_s,
        abs_tol=1e-12,
    )
    assert math.isclose(
        t.bleed_mass_flow_kg_s*(
            t.inlet_enthalpy_j_kg-t.exhaust_enthalpy_j_kg
        ),t.delivered_rotor_power_w+t.mechanical_loss_w,
        rel_tol=1e-10,
    )
    assert abs(t.turbine_energy_residual_w)<1e-8
    assert abs(step.combined_open_system_energy_residual_j)<1e-5
    assert abs(p.total_fluid_rotor_energy_balance_residual_j)<1e-5
    assert abs(p.fluid.combined_system_total_energy_residual_j)<1e-5
    assert abs(p.fluid.combined_system_mass_residual_kg_s)<1e-8
    assert abs(p.rotor_energy_balance_residual_j)<1e-5
    assert abs(p.pump.manifold_pressure_residual_pa)<1e-5
    assert math.isclose(
        p.rotor_power_storage_rate_w,
        t.delivered_rotor_power_w-p.pump.shaft_power_w,
        rel_tol=1e-9,abs_tol=1e-3,
    )
    assert math.isclose(
        p.rotor_final_energy_j,
        .5*params.rotor_inertia_kg_m2*
        step.state.rotor_angular_speed_rad_s**2,
        rel_tol=1e-12,
    )
    # Independently reconstruct global open-system energy from
    # INVENTORY changes and actual physical boundary sinks.
    a=old.hydraulic
    b=step.state.hydraulic
    cs=3000.
    delta=(
        b.pump.internal_energy_j-a.pump.internal_energy_j+
        b.downstream.header.internal_energy_j-
        a.downstream.header.internal_energy_j+
        float(np.sum(b.downstream.tie.total_energy_j-
                     a.downstream.tie.total_energy_j))+
        cs*float(np.sum(
            b.downstream.tie.solid_temperature_k-
            a.downstream.tie.solid_temperature_k
        ))+
        p.rotor_final_energy_j-p.rotor_initial_energy_j
    )
    q=network.nozzle_jacket.heat_power_w+network.reflector.heat_power_w
    q+=p.fluid.parallel.combined.tie_step.nuclear_deposition_w
    expected=dt*(
        p.pump.mass_flow_kg_s*p.pump.tank_enthalpy_j_kg+q-
        t.core_reference_remainder_mass_flow_kg_s*t.inlet_enthalpy_j_kg-
        t.bleed_mass_flow_kg_s*t.exhaust_enthalpy_j_kg-
        t.mechanical_loss_w
    )
    assert abs(delta-expected)<1e-5


def main()->int:
    eos=HydrogenEOS()
    core=hot_case()
    net=network()
    turbine_config=sample_turbine()
    shaft=solve_turbine_bleed(
        core,turbine_config,cooling_network=net,eos=eos,
    )
    bc=engine_tie_reservoirs(shaft)
    admission=calibrate_effective_admission(shaft,turbine_config,eos=eos)
    params=DynamicPumpParameters(
        reference_angular_speed_rad_s=1200.,
        rotor_inertia_kg_m2=.05,
    )
    base=evaluate_dynamic_turbine(
        admission,bc.steady_common_header_pressure_pa,
        bc.steady_common_header_stagnation_enthalpy_j_kg,
        shaft.tank_mass_flow_kg_s,eos=eos,
    )
    assert math.isclose(base.bleed_mass_flow_kg_s,
                        shaft.turbine_bleed_mass_flow_kg_s,rel_tol=1e-9)
    assert math.isclose(base.delivered_rotor_power_w,
                        shaft.turbine_shaft_delivered_power_w,rel_tol=1e-7)
    assert math.isclose(base.core_reference_remainder_mass_flow_kg_s,
                        shaft.core_mass_flow_kg_s,rel_tol=1e-9)
    assert base.inlet_pressure_pa>base.exhaust_pressure_pa
    high=evaluate_dynamic_turbine(
        admission,bc.steady_common_header_pressure_pa+1000.,
        bc.steady_common_header_stagnation_enthalpy_j_kg,
        shaft.tank_mass_flow_kg_s,eos=eos,
    )
    low=evaluate_dynamic_turbine(
        admission,bc.steady_common_header_pressure_pa-1000.,
        bc.steady_common_header_stagnation_enthalpy_j_kg,
        shaft.tank_mass_flow_kg_s,eos=eos,
    )
    assert high.bleed_mass_flow_kg_s>base.bleed_mass_flow_kg_s
    assert base.bleed_mass_flow_kg_s>low.bleed_mass_flow_kg_s
    assert high.delivered_rotor_power_w>low.delivered_rotor_power_w
    lower_eff=evaluate_dynamic_turbine(
        replace(admission,isentropic_efficiency=.5),
        bc.steady_common_header_pressure_pa,
        bc.steady_common_header_stagnation_enthalpy_j_kg,
        shaft.tank_mass_flow_kg_s,eos=eos,
    )
    assert math.isclose(lower_eff.bleed_mass_flow_kg_s,
                        base.bleed_mass_flow_kg_s,rel_tol=1e-12)
    assert lower_eff.delivered_rotor_power_w<base.delivered_rotor_power_w
    assert lower_eff.exhaust_enthalpy_j_kg>base.exhaust_enthalpy_j_kg
    reduced_outlet=evaluate_dynamic_turbine(
        replace(admission,exhaust_pressure_pa=.60e6),
        bc.steady_common_header_pressure_pa,
        bc.steady_common_header_stagnation_enthalpy_j_kg,
        shaft.tank_mass_flow_kg_s,eos=eos,
    )
    assert reduced_outlet.specific_actual_work_j_kg>base.specific_actual_work_j_kg
    # Choked admission is a static throat constraint: raising
    # downstream exhaust pressure ABOVE the critical ratio should
    # eventually yield unchoked flow with a lower mass flux.
    near_exhaust=.97*bc.steady_common_header_pressure_pa
    g,thr,choked=isentropic_hydrogen_admission_mass_flux(
        bc.steady_common_header_pressure_pa,
        bc.steady_common_header_stagnation_enthalpy_j_kg,
        near_exhaust,eos=eos,
    )
    assert g>0 and not choked
    assert math.isclose(thr,near_exhaust,rel_tol=1e-8)
    if base.is_admission_choked:
        assert math.isclose(
            reduced_outlet.admission_mass_flux_kg_m2_s,
            base.admission_mass_flux_kg_m2_s,rel_tol=1e-5
        )

    with tempfile.TemporaryDirectory() as td:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(td))
        heating=read_resolved_heating(fuel,tie,mesh)
        flow=DistributedTieParameters(
            thermal=IsochoricTieParameters(
                supply_cell_volume_m3=.0001,return_cell_volume_m3=.0001,
                solid_capacity_j_k=3000.,supply_ua_w_k=.4,return_ua_w_k=.4,
            ),
            flow_area_m2=.0002,darcy_friction_factor=.04,
            turnaround_loss_coefficient=.6,
        )
        up=DynamicHeaderParameters(volume_m3=.0001)
        down=DynamicHeaderParameters(volume_m3=.0001)
        initial=DynamicPumpState(
            rotor_angular_speed_rad_s=1200.,
            hydraulic=DualHeaderTieState(
                pump=initial_pump_manifold(shaft,up,eos=eos),
                downstream=DynamicHeaderTieState(
                    tie=initial_distributed_tie_state(
                        heating,flow,bc.upstream_pressure_pa,
                        bc.upstream_stagnation_enthalpy_j_kg,
                        solid_temperature_k=35.,eos=eos,
                    ),
                    header=initial_dynamic_header(shaft,down,eos=eos),
                ),
            ),
        )
        dt=1e-7
        first=advance_dynamic_turbine_pump(
            heating,initial,core,params,admission,flow,up,down,
            net,shaft,dt,eos=eos,
        )
        check(first,initial,dt,net,params)
        second=advance_dynamic_turbine_pump(
            heating,first.state,core,params,admission,flow,up,down,
            net,shaft,dt,eos=eos,
        )
        check(second,first.state,dt,net,params)
        assert second.turbine.inlet_pressure_pa!=first.turbine.inlet_pressure_pa
        assert second.turbine.bleed_mass_flow_kg_s!=first.turbine.bleed_mass_flow_kg_s
        assert second.turbine.delivered_rotor_power_w!=(
            first.turbine.delivered_rotor_power_w
        )
        assert math.isclose(
            second.turbine.inlet_pressure_pa,
            first.pump_and_fluid.fluid.parallel.combined.header_new_pressure_pa,
            rel_tol=1e-12,
        )
        perturbed=advance_dynamic_turbine_pump(
            heating,initial,core,params,
            replace(admission,effective_admission_area_m2=
                    admission.effective_admission_area_m2*.95),
            flow,up,down,net,shaft,dt,eos=eos,
        )
        check(perturbed,initial,dt,net,params)
        assert perturbed.turbine.bleed_mass_flow_kg_s<first.turbine.bleed_mass_flow_kg_s
        assert perturbed.turbine.delivered_rotor_power_w<first.turbine.delivered_rotor_power_w

        rejects(lambda:evaluate_dynamic_turbine(
            replace(admission,effective_admission_area_m2=
                    100*admission.effective_admission_area_m2),
            bc.steady_common_header_pressure_pa,
            bc.steady_common_header_stagnation_enthalpy_j_kg,
            shaft.tank_mass_flow_kg_s,eos=eos,
        ))
        rejects(lambda:evaluate_dynamic_turbine(
            admission,bc.steady_common_header_pressure_pa,
            bc.steady_common_header_stagnation_enthalpy_j_kg,
            0.,eos=eos,
        ))
        rejects(lambda:evaluate_dynamic_turbine(
            admission,admission.exhaust_pressure_pa*.99,
            bc.steady_common_header_stagnation_enthalpy_j_kg,
            shaft.tank_mass_flow_kg_s,eos=eos,
        ))
        rejects(lambda:advance_dynamic_turbine_pump(
            heating,initial,core,params,admission,flow,up,down,
            net,shaft,0.,eos=eos,
        ))
        rejects(lambda:advance_dynamic_turbine_pump(
            heating,initial,core,params,admission,flow,up,down,
            net,shaft,.1,eos=eos,
        ))

    print("NERVA live warm H2 turbine bleed and shaft power -> rotor/tie cooling: PASS")
    print(f"  calibrated effective turbine area: {admission.effective_admission_area_m2:.9g} m2")
    print(f"  reference bleed flow: {base.bleed_mass_flow_kg_s:.9g} kg/s")
    print(f"  dynamic bleed flow, step 2: {second.turbine.bleed_mass_flow_kg_s:.9g} kg/s")
    print(f"  dynamic turbine shaft power: {second.turbine.delivered_rotor_power_w:.9g} W")
    print(f"  rotor stored energy residual: {second.pump_and_fluid.rotor_energy_balance_residual_j:.9g} J")
    print(f"  combined open-system energy residual: {second.combined_open_system_energy_residual_j:.9g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
