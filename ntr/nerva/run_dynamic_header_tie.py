"""Run the two-way dynamic mixing-header / tie-Euler hydrogen transient.

The shaft-closed turbine/regenerator provides fixed reference pump,
nozzle/reflector branch and core/turbine demand. The distributed tie
return exchanges ONE conserved mass and TOTAL-energy flux with the
header; its evolving EOS pressure/h are fed back to the next step.
Full engine pump/turbine/nozzle transient feedback is not solved.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .distributed_tie_euler import (
    DistributedTieParameters,initial_distributed_tie_state,
)
from .distributed_tie_engine_boundary import engine_tie_reservoirs
from .dynamic_header_tie import (
    DynamicHeaderParameters,DynamicHeaderTieState,
    initial_dynamic_header,advance_dynamic_header_tie,
)
from .isochoric_tie import IsochoricTieParameters
from .dynamic_parallel_header import advance_dynamic_parallel_tie_header
from .dual_header_tie import (
    DualHeaderTieState,initial_pump_manifold,advance_dual_header_tie,
)
from .lh2_feed_cycle import LH2Cycle,HydrogenEOS
from .regenerative_cooling import CoolingLeg,CoolingNetwork
from .resolved_transient import read_resolved_heating
from .turbine_bleed_cycle import TurbineBleed,solve_turbine_bleed


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh_fields",type=Path)
    parser.add_argument("fuel_reconstruction",type=Path)
    parser.add_argument("tie_reconstruction",type=Path)
    parser.add_argument("--engine-config",type=Path,required=True)
    parser.add_argument("--chemistry-mode",choices=("frozen","equilibrium"),
                        default="frozen")
    parser.add_argument("--header-volume-m3",type=float,required=True)
    parser.add_argument("--dynamic-nozzle-branch",action="store_true",
                        help="solve nozzle/reflector flow against live header pressure")
    parser.add_argument("--upstream-manifold-volume-m3",type=float,
                        help="also evolve a conservative pump-discharge H2 inventory")
    parser.add_argument("--initial-solid-temperature-k",type=float,required=True)
    parser.add_argument("--supply-cell-volume-m3",type=float,required=True)
    parser.add_argument("--return-cell-volume-m3",type=float,required=True)
    parser.add_argument("--flow-area-m2",type=float,required=True)
    parser.add_argument("--solid-capacity-j-k",type=float,required=True)
    parser.add_argument("--supply-ua-w-k",type=float,required=True)
    parser.add_argument("--return-ua-w-k",type=float,required=True)
    parser.add_argument("--darcy-friction-factor",type=float,default=0.)
    parser.add_argument("--turnaround-loss-coefficient",type=float,default=0.)
    parser.add_argument("--max-cfl",type=float,default=0.45)
    parser.add_argument("--dt-s",type=float,required=True)
    parser.add_argument("--steps",type=int,default=2)
    parser.add_argument("--output",type=Path,
                        default=Path("build/nerva_dynamic_tie_header"))
    args=parser.parse_args()
    if not 1<=args.steps<=10000:
        raise ValueError("invalid number of dynamic-header steps")
    if (args.upstream_manifold_volume_m3 is not None and
            not args.dynamic_nozzle_branch):
        raise ValueError("dynamic upstream manifold requires --dynamic-nozzle-branch")
    raw=json.loads(args.engine_config.read_text(encoding="utf-8"))
    if not isinstance(raw,dict) or set(raw)!={
        "lh2_cycle","turbine","regenerative_cooling"
    }:
        raise ValueError("engine config requires LH2, turbine and regenerative cooling")
    core=LH2Cycle(**raw["lh2_cycle"])
    turbine=TurbineBleed(**raw["turbine"])
    config=raw["regenerative_cooling"]
    names={"nozzle_jacket","reflector","tie_supply","tie_return"}
    if not isinstance(config,dict) or set(config)-names-{"min_branch_fraction"} or not names<=set(config):
        raise ValueError("invalid regenerative four-leg network JSON")
    net=CoolingNetwork(
        **{n:CoolingLeg(**config[n]) for n in names},
        min_branch_fraction=config.get("min_branch_fraction",0.02),
    )
    eos=HydrogenEOS()
    shaft=solve_turbine_bleed(
        core,turbine,cooling_network=net,
        chemistry_mode=args.chemistry_mode,eos=eos,
    )
    bc=engine_tie_reservoirs(shaft)
    heating=read_resolved_heating(
        args.fuel_reconstruction,args.tie_reconstruction,args.mesh_fields,
    )
    thermal=IsochoricTieParameters(
        supply_cell_volume_m3=args.supply_cell_volume_m3,
        return_cell_volume_m3=args.return_cell_volume_m3,
        solid_capacity_j_k=args.solid_capacity_j_k,
        supply_ua_w_k=args.supply_ua_w_k,
        return_ua_w_k=args.return_ua_w_k,
    )
    fluid=DistributedTieParameters(
        thermal=thermal,flow_area_m2=args.flow_area_m2,
        darcy_friction_factor=args.darcy_friction_factor,
        turnaround_loss_coefficient=args.turnaround_loss_coefficient,
        max_cfl=args.max_cfl,
    )
    header=DynamicHeaderParameters(volume_m3=args.header_volume_m3)
    fluid.validate()
    header.validate()
    tie=initial_distributed_tie_state(
        heating,fluid,
        bc.upstream_pressure_pa,
        bc.upstream_stagnation_enthalpy_j_kg,
        solid_temperature_k=args.initial_solid_temperature_k,eos=eos,
    )
    state=DynamicHeaderTieState(
        tie=tie,header=initial_dynamic_header(shaft,header,eos=eos),
    )
    upstream_params=(
        DynamicHeaderParameters(volume_m3=args.upstream_manifold_volume_m3)
        if args.upstream_manifold_volume_m3 is not None else None
    )
    dual=(
        DualHeaderTieState(
            pump=initial_pump_manifold(shaft,upstream_params,eos=eos),
            downstream=state,
        )
        if upstream_params is not None else None
    )
    history=[]
    latest=None
    for i in range(args.steps):
        branch=None
        dual_result=None
        if dual is not None:
            dual_result=advance_dual_header_tie(
                heating,dual,fluid,upstream_params,header,
                net,shaft,args.dt_s,eos=eos,
            )
            dual=dual_result.state
            branch=dual_result.parallel
            latest=branch.combined
        elif args.dynamic_nozzle_branch:
            branch=advance_dynamic_parallel_tie_header(
                heating,state,fluid,header,net,shaft,args.dt_s,eos=eos,
            )
            latest=branch.combined
        else:
            latest=advance_dynamic_header_tie(
                heating,state,fluid,header,shaft,args.dt_s,eos=eos,
            )
        state=latest.state
        history.append({
            "step":i+1,"time_s":(i+1)*args.dt_s,
            "header_pressure_pa":latest.header_new_pressure_pa,
            "header_temperature_k":latest.header_new_temperature_k,
            "header_specific_enthalpy_j_kg":latest.header_new_enthalpy_j_kg,
            "header_mass_kg":state.header.mass_kg,
            "header_internal_energy_j":state.header.internal_energy_j,
            "tie_inlet_flow_kg_s":latest.tie_inlet_flow_kg_s,
            "tie_to_header_flow_kg_s":latest.tie_header_interface_flow_kg_s,
            "tie_to_header_total_energy_w":
                latest.tie_header_interface_total_energy_power_w,
            "nozzle_branch_inlet_kg_s":latest.nozzle_branch_inlet_flow_kg_s,
            "fixed_engine_discharge_kg_s":latest.fixed_engine_discharge_flow_kg_s,
            "dynamic_nozzle_branch_active":
                latest.dynamic_nozzle_branch_pressure_response_active,
            "upstream_pump_flow_mismatch_kg_s":
                latest.upstream_pump_header_mass_flow_mismatch_kg_s,
            "nozzle_branch_header_pressure_residual_pa":(
                branch.nozzle_branch.outlet_pressure_residual_pa
                if branch is not None else None
            ),
            "upstream_manifold_pressure_pa":(
                dual_result.pump_new_pressure_pa
                if dual_result is not None else None
            ),
            "upstream_manifold_enthalpy_j_kg":(
                dual_result.pump_new_enthalpy_j_kg
                if dual_result is not None else None
            ),
            "upstream_manifold_mass_kg":(
                dual.pump.mass_kg if dual is not None else None
            ),
            "total_system_mass_residual_kg_s":(
                dual_result.combined_system_mass_residual_kg_s
                if dual_result is not None else None
            ),
            "total_system_energy_residual_j":(
                dual_result.combined_system_total_energy_residual_j
                if dual_result is not None else None
            ),
            "combined_mass_residual_kg_s":
                latest.combined_tie_header_mass_residual_kg_s,
            "combined_total_energy_residual_j":
                latest.combined_tie_solid_header_energy_residual_j,
            "tie_momentum_residual_n":
                latest.combined_tie_header_momentum_residual_n,
            "acoustic_cfl":latest.tie_step.maximum_cfl,
        })
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    fields=dict(
        tie_mass_kg=state.tie.mass_kg,
        tie_momentum_kg_m_s=state.tie.momentum_kg_m_s,
        tie_total_energy_j=state.tie.total_energy_j,
        tie_solid_temperature_k=state.tie.solid_temperature_k,
        tie_pressure_pa=latest.tie_step.pressure_pa,
        tie_temperature_k=latest.tie_step.temperature_k,
        tie_vapor_quality=latest.tie_step.quality,
        tie_interface_mass_flux_kg_s=
            latest.tie_step.interface_mass_flux_kg_s,
        tie_interface_total_energy_flux_w=
            latest.tie_step.interface_total_energy_flux_w,
        header_mass_kg=np.array(state.header.mass_kg),
        header_internal_energy_j=np.array(state.header.internal_energy_j),
        header_pressure_pa=np.array(latest.header_new_pressure_pa),
        header_enthalpy_j_kg=np.array(latest.header_new_enthalpy_j_kg),
    )
    if dual is not None:
        fields.update(
            upstream_manifold_mass_kg=np.array(dual.pump.mass_kg),
            upstream_manifold_internal_energy_j=np.array(
                dual.pump.internal_energy_j
            ),
            upstream_manifold_pressure_pa=np.array(
                dual_result.pump_new_pressure_pa
            ),
            upstream_manifold_enthalpy_j_kg=np.array(
                dual_result.pump_new_enthalpy_j_kg
            ),
        )
    np.savez_compressed(
        output/"dynamic_tie_header_final_state.npz",**fields,
    )
    report={
        "status":"PASS",
        "tie_header_pressure_enthalpy_feedback_active":True,
        "dynamic_nozzle_branch_header_pressure_feedback_active":
            args.dynamic_nozzle_branch,
        "upstream_pump_header_mass_flow_conserved":dual is not None,
        "upstream_manifold_h2_inventory_active":dual is not None,
        "pump_head_and_pump_mass_flow_are_steady_references":True,
        "full_shaft_cycle_feedback_converged":False,
        "dynamic_pump_turbine_nozzle_flows_solved":False,
        "actual_openmc_transport_executed":False,
        "historical_nerva_hardware_validated":False,
        "steady_reference_tank_flow_kg_s":shaft.tank_mass_flow_kg_s,
        "steady_reference_tie_flow_kg_s":bc.steady_tie_branch_mass_flow_kg_s,
        "header_parameters":asdict(header),
        "upstream_manifold_parameters":(
            asdict(upstream_params) if upstream_params is not None else None
        ),
        "tie_parameters":asdict(fluid),
        "history":history,
    }
    (output/"dynamic_tie_header_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",
        encoding="utf-8",
    )
    print("NERVA two-way tie Euler/common-header feedback: PASS")
    print(f"  header pressure: {latest.header_new_pressure_pa:.9g} Pa")
    print(f"  tie outlet flow: {latest.tie_header_interface_flow_kg_s:.9g} kg/s")
    print(f"  combined mass residual: {latest.combined_tie_header_mass_residual_kg_s:.9g} kg/s")
    print(f"  combined total-energy residual: {latest.combined_tie_solid_header_energy_residual_j:.9g} J")
    print("  full pump/turbine/nozzle transient feedback: NOT SOLVED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
