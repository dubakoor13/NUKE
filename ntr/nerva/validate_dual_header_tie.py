"""Real-CoolProp dual-manifold hydrogen mass/total-energy regression.

Exercises a shaft-closed reference pump and heating partitions, a live
upstream EOS inventory, the dynamic nozzle/reflector hydraulic root,
a distributed conservative tie Euler, and the common return header.
Tests explicit cross-component flux cancellation, feedback response,
pressure/volume sensitivity, valid EOS and fail-closed time steps.
Heating fields and model geometry are SYNTHETIC, not historical NERVA.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import numpy as np

from .dual_header_tie import (
    DualHeaderTieState,initial_pump_manifold,advance_dual_header_tie,
)
from .dynamic_header_tie import (
    DynamicHeaderParameters,DynamicHeaderTieState,
    header_properties,initial_dynamic_header,
)
from .dynamic_parallel_header import advance_dynamic_parallel_tie_header
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


def rejects(fn):
    try:
        fn()
    except (RuntimeError,ValueError):
        return
    raise AssertionError("invalid pump manifold / tie state accepted")


def check(result,old,dt,network):
    fluid=result.parallel.combined.tie_step
    down=result.parallel.combined
    noz=result.parallel.nozzle_branch
    assert result.dynamic_pump_manifold_active
    assert result.conserved_tie_supply_face
    assert result.pump_flow_and_head_are_reference_values
    assert not result.shaft_turbine_core_nozzle_feedback_converged
    assert not result.actual_openmc_transport_executed
    assert abs(result.combined_system_mass_residual_kg_s)<1e-8
    assert abs(result.combined_system_total_energy_residual_j)<1e-5
    assert abs(result.pump_manifold_mass_residual_kg_s)<1e-8
    assert abs(result.pump_manifold_energy_residual_j)<1e-5
    assert abs(down.combined_tie_header_mass_residual_kg_s)<1e-8
    assert abs(down.combined_tie_solid_header_energy_residual_j)<1e-5
    assert abs(fluid.total_energy_residual_j)<1e-5
    assert abs(fluid.mass_residual_kg_s)<1e-8
    assert abs(noz.outlet_pressure_residual_pa)<1.
    assert math.isclose(result.tie_supply_interface_mass_flow_kg_s,
                        float(np.sum(fluid.interface_mass_flux_kg_s[:,0])),
                        abs_tol=1e-12)
    assert math.isclose(result.tie_supply_interface_total_energy_power_w,
                        float(np.sum(fluid.interface_total_energy_flux_w[:,0])),
                        abs_tol=1e-7)
    m_noz=noz.mass_flow_kg_s
    assert math.isclose(
        result.pump_manifold_inventory_rate_kg_s,
        result.pump_delivered_reference_flow_kg_s-
        m_noz-result.tie_supply_interface_mass_flow_kg_s,
        abs_tol=1e-8,
    )
    assert math.isclose(
        result.pump_manifold_internal_energy_rate_w,
        result.pump_delivered_reference_flow_kg_s*
        result.pump_delivered_reference_enthalpy_j_kg-
        m_noz*result.pump_old_enthalpy_j_kg-
        result.tie_supply_interface_total_energy_power_w,
        abs_tol=0.1,rel_tol=1e-8,
    )
    assert math.isclose(
        m_noz*(noz.outlet_stagnation_enthalpy_j_kg-
               result.pump_old_enthalpy_j_kg),
        network.nozzle_jacket.heat_power_w+
        network.reflector.heat_power_w,
        abs_tol=1e-5,rel_tol=1e-9,
    )
    assert math.isclose(
        result.combined_net_mass_inventory_rate_kg_s,
        result.pump_delivered_reference_flow_kg_s-
        down.fixed_engine_discharge_flow_kg_s,
        abs_tol=1e-8,
    )
    assert result.pump_new_pressure_pa>0
    assert result.state.pump.mass_kg>0
    assert result.state.downstream.header.mass_kg>0


def main()->int:
    eos=HydrogenEOS()
    network_case=network()
    shaft=solve_turbine_bleed(
        hot_case(),sample_turbine(),
        cooling_network=network_case,eos=eos,
    )
    bc=engine_tie_reservoirs(shaft)
    up_params=DynamicHeaderParameters(volume_m3=0.0001)
    down_params=DynamicHeaderParameters(volume_m3=0.0001)
    up_initial=initial_pump_manifold(shaft,up_params,eos=eos)
    p_init,h_init,t_init,rho_init=header_properties(
        up_initial,up_params,eos=eos,
    )
    assert math.isclose(p_init,bc.upstream_pressure_pa,rel_tol=1e-7)
    assert math.isclose(h_init,bc.upstream_stagnation_enthalpy_j_kg,
                        rel_tol=1e-7)
    assert t_init>17 and rho_init>0

    with tempfile.TemporaryDirectory() as tmp:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(tmp))
        heating=read_resolved_heating(fuel,tie,mesh)
        params=DistributedTieParameters(
            thermal=IsochoricTieParameters(
                supply_cell_volume_m3=0.0001,
                return_cell_volume_m3=0.0001,
                solid_capacity_j_k=3000.,
                supply_ua_w_k=0.4,return_ua_w_k=0.4,
            ),
            flow_area_m2=0.0002,
            darcy_friction_factor=0.04,
            turnaround_loss_coefficient=0.6,
        )
        tie_init=initial_distributed_tie_state(
            heating,params,bc.upstream_pressure_pa,
            bc.upstream_stagnation_enthalpy_j_kg,
            solid_temperature_k=35.,eos=eos,
        )
        old=DualHeaderTieState(
            pump=up_initial,
            downstream=DynamicHeaderTieState(
                tie=tie_init,
                header=initial_dynamic_header(shaft,down_params,eos=eos),
            ),
        )
        dt=1e-7
        first=advance_dual_header_tie(
            heating,old,params,up_params,down_params,
            network_case,shaft,dt,eos=eos,
        )
        check(first,old,dt,network_case)
        assert first.pump_new_pressure_pa!=first.pump_old_pressure_pa
        assert first.pump_new_enthalpy_j_kg!=first.pump_old_enthalpy_j_kg
        second=advance_dual_header_tie(
            heating,first.state,params,up_params,down_params,
            network_case,shaft,dt,eos=eos,
        )
        check(second,first.state,dt,network_case)
        assert math.isclose(
            second.pump_old_pressure_pa,first.pump_new_pressure_pa,
            rel_tol=1e-12,
        )
        assert math.isclose(
            second.parallel.nozzle_branch.desired_header_pressure_pa,
            first.parallel.combined.header_new_pressure_pa,
            rel_tol=1e-12,
        )
        # The second Euler and nozzle branch now receive a changed
        # upstream EOS reservoir, not the original fixed pump state.
        frozen=advance_dynamic_parallel_tie_header(
            heating,first.state.downstream,params,down_params,
            network_case,shaft,dt,eos=eos,
        )
        assert not np.allclose(
            second.parallel.combined.tie_step.interface_mass_flux_kg_s[:,0],
            frozen.combined.tie_step.interface_mass_flux_kg_s[:,0],
            atol=1e-12,rtol=1e-8,
        )
        assert second.parallel.nozzle_branch.mass_flow_kg_s!=(
            frozen.nozzle_branch.mass_flow_kg_s
        )
        # Larger upstream volume slows EOS pressure evolution for
        # the same physically computed inlet/outlet mass mismatch.
        big=replace(up_params,volume_m3=up_params.volume_m3*25)
        big_init=initial_pump_manifold(shaft,big,eos=eos)
        bigger=advance_dual_header_tie(
            heating,replace(old,pump=big_init),params,big,down_params,
            network_case,shaft,dt,eos=eos,
        )
        check(bigger,replace(old,pump=big_init),dt,network_case)
        assert abs(bigger.pump_new_pressure_pa-bigger.pump_old_pressure_pa)<(
            abs(first.pump_new_pressure_pa-first.pump_old_pressure_pa)
        )
        # Invalid pressure range, unresolvable CFL and unsupported
        # configurations fail instead of inventing stored H2 mass.
        rejects(lambda:initial_pump_manifold(
            shaft,replace(up_params,volume_m3=-1.),eos=eos
        ))
        rejects(lambda:advance_dual_header_tie(
            heating,old,params,up_params,down_params,
            network_case,shaft,0.,eos=eos,
        ))
        rejects(lambda:advance_dual_header_tie(
            heating,old,params,up_params,down_params,
            network_case,shaft,0.1,eos=eos,
        ))
        rejects(lambda:advance_dual_header_tie(
            heating,replace(old,pump=replace(
                up_initial,mass_kg=0.)),
            params,up_params,down_params,
            network_case,shaft,dt,eos=eos,
        ))
        rejects(lambda:advance_dual_header_tie(
            heating,old,params,up_params,down_params,
            network_case,replace(shaft,regenerative_cooling=None),
            dt,eos=eos,
        ))
    print("NERVA dual live manifolds + tie Euler + nozzle regenerative branch: PASS")
    print(f"  pump header pressure: {first.pump_old_pressure_pa:.9g} -> {first.pump_new_pressure_pa:.9g} Pa")
    print(f"  nozzle branch mass flow step 2: {second.parallel.nozzle_branch.mass_flow_kg_s:.9g} kg/s")
    print(f"  tie inlet mass flux step 2: {second.tie_supply_interface_mass_flow_kg_s:.9g} kg/s")
    print(f"  upstream mass residual: {second.pump_manifold_mass_residual_kg_s:.9g} kg/s")
    print(f"  total system mass residual: {second.combined_system_mass_residual_kg_s:.9g} kg/s")
    print(f"  total system energy residual: {second.combined_system_total_energy_residual_j:.9g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
