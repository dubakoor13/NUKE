"""Run shaft-closed NERVA turbine/regenerator, then boundary-drive transient tie Euler.

This is explicitly ONE-WAY conditioning: pump/turbine and regenerative
branch solve establish *steady* pressure/enthalpy reservoir states.
The transient tie Euler mass, momentum and TOTAL energy are independent
and are NOT iterated back to the engine. Instantaneous mismatch between
the tie fluxes and the steady branch mass flow is reported, never
claimed converged. Heating NPZ input is externally supplied; the
command does not execute OpenMC or verify NASA model provenance.
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
from .distributed_tie_engine_boundary import (
    engine_tie_reservoirs,advance_engine_conditioned_distributed_tie,
)
from .isochoric_tie import IsochoricTieParameters
from .lh2_feed_cycle import HydrogenEOS,LH2Cycle
from .regenerative_cooling import CoolingLeg,CoolingNetwork
from .resolved_transient import read_resolved_heating
from .turbine_bleed_cycle import TurbineBleed,solve_turbine_bleed


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh_fields",type=Path)
    p.add_argument("fuel_reconstruction",type=Path)
    p.add_argument("tie_reconstruction",type=Path)
    p.add_argument("--engine-config",type=Path,required=True,
                   help="lh2_cycle, turbine and regenerative_cooling JSON")
    p.add_argument("--chemistry-mode",choices=("frozen","equilibrium"),
                   default="frozen")
    p.add_argument("--solid-temperature-k",type=float,required=True)
    p.add_argument("--supply-cell-volume-m3",type=float,required=True)
    p.add_argument("--return-cell-volume-m3",type=float,required=True)
    p.add_argument("--flow-area-m2",type=float,required=True)
    p.add_argument("--solid-capacity-j-k",type=float,required=True)
    p.add_argument("--supply-ua-w-k",type=float,required=True)
    p.add_argument("--return-ua-w-k",type=float,required=True)
    p.add_argument("--darcy-friction-factor",type=float,default=0.)
    p.add_argument("--turnaround-loss-coefficient",type=float,default=0.)
    p.add_argument("--max-cfl",type=float,default=0.45)
    p.add_argument("--initial-velocity-m-s",type=float,default=0.)
    p.add_argument("--dt-s",type=float,required=True)
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_shaft_bound_tie_euler"))
    a=p.parse_args()
    if not 1<=a.steps<=10000:
        raise ValueError("invalid number of engine-conditioned Euler steps")
    config=json.loads(a.engine_config.read_text(encoding="utf-8"))
    if not isinstance(config,dict) or set(config)!={
        "lh2_cycle","turbine","regenerative_cooling"
    }:
        raise ValueError("engine JSON must have lh2_cycle, turbine, regenerative_cooling")
    core=LH2Cycle(**config["lh2_cycle"])
    turbine=TurbineBleed(**config["turbine"])
    branches=config["regenerative_cooling"]
    expected={"nozzle_jacket","reflector","tie_supply","tie_return"}
    if set(branches)-expected-{"min_branch_fraction"} or not expected<=set(branches):
        raise ValueError("invalid four-leg regenerative cooling definition")
    cooling=CoolingNetwork(
        **{name:CoolingLeg(**branches[name]) for name in expected},
        min_branch_fraction=branches.get("min_branch_fraction",0.02),
    )
    core.validate()
    turbine.validate()
    cooling.validate()
    eos=HydrogenEOS()
    point=solve_turbine_bleed(
        core,turbine,cooling_network=cooling,
        chemistry_mode=a.chemistry_mode,eos=eos,
    )
    reservoirs=engine_tie_reservoirs(point)
    heating=read_resolved_heating(
        a.fuel_reconstruction,a.tie_reconstruction,a.mesh_fields,
    )
    thermal=IsochoricTieParameters(
        supply_cell_volume_m3=a.supply_cell_volume_m3,
        return_cell_volume_m3=a.return_cell_volume_m3,
        solid_capacity_j_k=a.solid_capacity_j_k,
        supply_ua_w_k=a.supply_ua_w_k,
        return_ua_w_k=a.return_ua_w_k,
    )
    params=DistributedTieParameters(
        thermal=thermal,flow_area_m2=a.flow_area_m2,
        darcy_friction_factor=a.darcy_friction_factor,
        turnaround_loss_coefficient=a.turnaround_loss_coefficient,
        max_cfl=a.max_cfl,
    )
    params.validate()
    state=initial_distributed_tie_state(
        heating,params,
        reservoirs.upstream_pressure_pa,
        reservoirs.upstream_stagnation_enthalpy_j_kg,
        solid_temperature_k=a.solid_temperature_k,
        velocity_m_s=a.initial_velocity_m_s,eos=eos,
    )
    history=[]
    result=None
    for i in range(a.steps):
        result=advance_engine_conditioned_distributed_tie(
            heating,state,params,point,a.dt_s,eos=eos,
        )
        state=result.transient.state
        history.append({
            "step":i+1,"elapsed_s":(i+1)*a.dt_s,
            "steady_tie_branch_reference_kg_s":
                result.steady_tie_mass_flow_reference_kg_s,
            "dynamic_inlet_mass_flow_kg_s":
                result.inlet_transient_mass_flow_kg_s,
            "dynamic_outlet_mass_flow_kg_s":
                result.outlet_transient_mass_flow_kg_s,
            "hydrogen_inventory_rate_kg_s":
                result.transient_mass_accumulation_kg_s,
            "inlet_reference_flow_mismatch_kg_s":
                result.inlet_mass_flow_reference_residual_kg_s,
            "outlet_reference_flow_mismatch_kg_s":
                result.outlet_mass_flow_reference_residual_kg_s,
            "mass_residual_kg_s":
                result.transient_mass_conservation_residual_kg_s,
            "momentum_residual_n":
                result.transient_momentum_conservation_residual_n,
            "total_energy_residual_j":
                result.transient_total_energy_conservation_residual_j,
            "acoustic_cfl":result.transient.maximum_cfl,
            "solid_storage_rate_w":result.transient.solid_storage_rate_w,
            "fluid_internal_storage_rate_w":
                result.transient.fluid_internal_energy_storage_rate_w,
            "fluid_kinetic_storage_rate_w":
                result.transient.fluid_kinetic_energy_storage_rate_w,
        })
    path=a.output
    path.mkdir(parents=True,exist_ok=True)
    step=result.transient
    np.savez_compressed(
        path/"shaft_conditioned_tie_euler_state.npz",
        fluid_mass_kg=state.mass_kg,
        fluid_momentum_kg_m_s=state.momentum_kg_m_s,
        fluid_total_energy_j=state.total_energy_j,
        solid_temperature_k=state.solid_temperature_k,
        pressure_pa=step.pressure_pa,
        density_kg_m3=step.density_kg_m3,
        temperature_k=step.temperature_k,
        vapor_quality=step.quality,
        velocity_m_s=step.velocity_m_s,
        interface_mass_flux_kg_s=step.interface_mass_flux_kg_s,
        interface_momentum_flux_n=step.interface_momentum_flux_n,
        interface_total_energy_flux_w=step.interface_total_energy_flux_w,
    )
    report={
        "status":"PASS",
        "engine_steady_shaft_cycle_solved":True,
        "distributed_tie_euler_mass_momentum_total_energy_solved":True,
        "coupling_direction":"one-way steady shaft boundaries to dynamic tie reservoirs",
        "transient_flow_matches_steady_engine":False,
        "full_engine_feedback_converged":False,
        "real_openmc_transport_executed":False,
        "historical_nerva_hardware_validated":False,
        "steady_tank_mass_flow_kg_s":point.tank_mass_flow_kg_s,
        "steady_tie_branch_mass_flow_kg_s":
            reservoirs.steady_tie_branch_mass_flow_kg_s,
        "steady_shaft_residual_w":point.shaft_power_residual_w,
        "steady_nozzle_mass_residual_kg_s":point.nozzle_mass_flow_residual_kg_s,
        "steady_regen_split_pressure_residual_pa":
            point.regenerative_cooling.split_pressure_residual_pa,
        "reservoirs":asdict(reservoirs),
        "distributed_parameters":asdict(params),
        "history":history,
    }
    (path/"shaft_conditioned_tie_euler_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",
        encoding="utf-8",
    )
    print("NERVA shaft-conditioned distributed tie Euler boundary: PASS")
    print(f"  steady tie branch={reservoirs.steady_tie_branch_mass_flow_kg_s:.9g} kg/s")
    print(f"  transient inlet={result.inlet_transient_mass_flow_kg_s:.9g} kg/s")
    print(f"  transient outlet={result.outlet_transient_mass_flow_kg_s:.9g} kg/s")
    print(f"  energy residual={step.total_energy_residual_j:.9g} J")
    print("  full engine transient feedback: NOT SOLVED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
