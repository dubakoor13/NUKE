"""Run coupled cryogenic tie energy + regenerative branch hydraulics.

The pump outlet enthalpy/pressure is evaluated from the existing LH2
pump efficiency/head curve. The full nozzle/turbine/shaft operating
point is NOT re-solved; mass flow is an externally supplied trial.
A two-phase tie control-volume model and steady hydraulic network
exchange mass flow, representative pressure and effective advected
heat until the fixed point meets independent residual thresholds.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .cryogenic_tie_enthalpy import CryogenicTieParameters,CryogenicTieState
from .cryogenic_tie_hydraulic_coupling import (
    couple_cryogenic_tie_hydraulics,pumped_regenerative_inlet,
)
from .lh2_feed_cycle import HydrogenEOS,LH2Cycle
from .regenerative_cooling import CoolingNetwork,CoolingLeg
from .resolved_transient import read_resolved_heating


def _args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh_fields",type=Path)
    p.add_argument("fuel_reconstruction",type=Path)
    p.add_argument("tie_reconstruction",type=Path)
    p.add_argument("--cycle-config",type=Path,required=True,
                   help="JSON with lh2_cycle and regenerative_cooling objects")
    p.add_argument("--total-mass-flow-kg-s",type=float,required=True,
                   help="Prescribed pump operating point / total coolant flow")
    p.add_argument("--initial-pressure-fraction",type=float,default=0.99)
    p.add_argument("--initial-fluid-temperature-k",type=float,required=True)
    p.add_argument("--initial-solid-temperature-k",type=float,required=True)
    p.add_argument("--solid-capacity-j-k",type=float,required=True)
    p.add_argument("--supply-inventory-kg",type=float,required=True)
    p.add_argument("--return-inventory-kg",type=float,required=True)
    p.add_argument("--supply-ua-w-k",type=float,required=True)
    p.add_argument("--return-ua-w-k",type=float,required=True)
    p.add_argument("--nozzle-wall-heat-w",type=float,required=True)
    p.add_argument("--reflector-wall-heat-w",type=float,required=True)
    p.add_argument("--dt-s",type=float,required=True)
    p.add_argument("--max-iterations",type=int,default=15)
    p.add_argument("--relative-flow-tolerance",type=float,default=0.002)
    p.add_argument("--relative-pressure-tolerance",type=float,default=0.002)
    p.add_argument("--relative-source-tolerance",type=float,default=0.002)
    p.add_argument("--relaxation",type=float,default=0.6)
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_cryogenic_tie_hydraulic"))
    return p.parse_args()


def _cooling(d: dict)->CoolingNetwork:
    return CoolingNetwork(
        nozzle_jacket=CoolingLeg(**d["nozzle_jacket"]),
        reflector=CoolingLeg(**d["reflector"]),
        tie_supply=CoolingLeg(**d["tie_supply"]),
        tie_return=CoolingLeg(**d["tie_return"]),
        min_branch_fraction=d.get("min_branch_fraction",0.02),
    )


def main()->int:
    a=_args()
    if (not math.isfinite(a.initial_pressure_fraction) or
            not 0<a.initial_pressure_fraction<=1):
        raise ValueError("initial pressure fraction must lie within (0,1]")
    for label,value in (
        ("initial_fluid_temperature_k",a.initial_fluid_temperature_k),
        ("initial_solid_temperature_k",a.initial_solid_temperature_k),
    ):
        if not math.isfinite(value) or value<=0:
            raise ValueError(f"invalid {label}")
    raw=json.loads(a.cycle_config.read_text(encoding="utf-8"))
    core=LH2Cycle(**raw["lh2_cycle"])
    net=_cooling(raw["regenerative_cooling"])
    core.validate()
    net.validate()
    eos=HydrogenEOS()
    pin,hin=pumped_regenerative_inlet(
        core,a.total_mass_flow_kg_s,eos=eos,
    )
    params=CryogenicTieParameters(
        pressure_pa=pin*a.initial_pressure_fraction,
        solid_capacity_j_k=a.solid_capacity_j_k,
        supply_inventory_kg=a.supply_inventory_kg,
        return_inventory_kg=a.return_inventory_kg,
        supply_ua_w_k=a.supply_ua_w_k,
        return_ua_w_k=a.return_ua_w_k,
    )
    h_old=eos.tp("Hmass",a.initial_fluid_temperature_k,params.pressure_pa)
    heating=read_resolved_heating(
        a.fuel_reconstruction,a.tie_reconstruction,a.mesh_fields,
    )
    initial=CryogenicTieState(
        solid_temperature_k=np.full_like(
            heating.tie_solid_w,a.initial_solid_temperature_k
        ),
        supply_enthalpy_j_kg=np.full_like(heating.tie_solid_w,h_old),
        return_enthalpy_j_kg=np.full_like(heating.tie_solid_w,h_old),
    )
    result=couple_cryogenic_tie_hydraulics(
        heating,initial,params,net,a.total_mass_flow_kg_s,pin,hin,a.dt_s,
        nozzle_wall_heat_w=a.nozzle_wall_heat_w,
        reflector_wall_heat_w=a.reflector_wall_heat_w,
        max_iterations=a.max_iterations,
        relative_flow_tolerance=a.relative_flow_tolerance,
        relative_pressure_tolerance=a.relative_pressure_tolerance,
        relative_source_tolerance=a.relative_source_tolerance,
        relaxation=a.relaxation,
        eos=eos,
    )
    root=a.output
    root.mkdir(parents=True,exist_ok=True)
    step=result.tie_step
    np.savez_compressed(
        root/"tie_hydraulic_fixed_point.npz",
        solid_temperature_k=step.state.solid_temperature_k,
        supply_enthalpy_j_kg=step.state.supply_enthalpy_j_kg,
        return_enthalpy_j_kg=step.state.return_enthalpy_j_kg,
        supply_temperature_k=step.supply_temperature_k,
        return_temperature_k=step.return_temperature_k,
        supply_quality=step.supply_quality,
        return_quality=step.return_quality,
        supply_density_kg_m3=step.supply_density_kg_m3,
        return_density_kg_m3=step.return_density_kg_m3,
        supply_wall_power_w=step.supply_wall_power_w,
        return_wall_power_w=step.return_wall_power_w,
        effective_tie_supply_heat_power_w=np.array(
            result.effective_cooling_network.tie_supply.heat_power_w
        ),
        effective_tie_return_heat_power_w=np.array(
            result.effective_cooling_network.tie_return.heat_power_w
        ),
        tie_flow_kg_s=result.heating.tie_flow_kg_s,
    )
    summary={
        "status":"PASS",
        "source_kind":"externally supplied OpenMC-style NPZ; provenance unverified",
        "real_openmc_transport_executed":False,
        "nerva_hardware_validated":False,
        "full_engine_cycle_converged":False,
        "pressure_field_resolved":False,
        "hydrogen_mass_inventory_updated":False,
        "flow_and_representative_pressure_fixed_point_converged":True,
        "total_pump_mass_flow_kg_s":a.total_mass_flow_kg_s,
        "pump_cooling_inlet_pressure_pa":pin,
        "pump_cooling_inlet_enthalpy_j_kg":hin,
        "dt_s":a.dt_s,
        "iterations":result.iterations,
        "relative_flow_residual":result.relative_flow_residual,
        "relative_pressure_residual":result.relative_pressure_residual,
        "relative_heat_source_change":result.relative_heat_source_change,
        "nuclear_energy_power_residual_w":result.nuclear_energy_power_residual_w,
        "outlet_enthalpy_difference_j_kg":result.outlet_enthalpy_difference_j_kg,
        "tie_nuclear_deposition_w":step.nuclear_deposition_w,
        "tie_solid_energy_storage_rate_w":step.solid_storage_rate_w,
        "tie_hydrogen_energy_storage_rate_w":step.hydrogen_storage_rate_w,
        "tie_advected_to_header_w":(
            result.effective_cooling_network.tie_supply.heat_power_w+
            result.effective_cooling_network.tie_return.heat_power_w
        ),
        "hydraulic_branch_flow_kg_s":result.cooling_point.tie_tube_mass_flow_kg_s,
        "hydraulic_common_header_temperature_k":result.cooling_point.common_header_temperature_k,
        "hydraulic_common_header_pressure_pa":result.cooling_point.common_header_pressure_pa,
        "hydraulic_total_energy_residual_w":result.cooling_point.total_energy_residual_w,
        "hydraulic_split_pressure_residual_pa":result.cooling_point.split_pressure_residual_pa,
        "max_local_solid_energy_residual_w":step.max_local_solid_residual_w,
        "max_local_hydrogen_energy_residual_w":step.max_local_hydrogen_residual_w,
        "inputs":{
            "cycle_config":str(a.cycle_config),
            "mesh_fields":str(a.mesh_fields),
            "fuel_reconstruction":str(a.fuel_reconstruction),
            "tie_reconstruction":str(a.tie_reconstruction),
            "model_parameters":asdict(params),
        },
        "iteration_trace":[asdict(x) for x in result.diagnostics],
    }
    (root/"cryogenic_tie_hydraulic_report.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    print("NERVA real-fluid tie-to-hydraulic branch fixed point: PASS")
    print(f"  iterations={result.iterations}")
    print(f"  tie flow={result.cooling_point.tie_tube_mass_flow_kg_s:.9g} kg/s")
    print(f"  p-header={result.cooling_point.common_header_pressure_pa:.9g} Pa")
    print(f"  nuclear energy residual={result.nuclear_energy_power_residual_w:.9g} W")
    print(f"  report: {root/'cryogenic_tie_hydraulic_report.json'}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
