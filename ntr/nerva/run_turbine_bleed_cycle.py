"""Run shaft-power-closed NERVA LH2 pump / turbine / core / nozzle model.

Input JSON shape:
  {"lh2_cycle": { ... existing LH2Cycle fields ... },
   "turbine": {"exhaust_pressure_pa": 750000,
               "isentropic_efficiency": 0.80,
               "shaft_mechanical_efficiency": 0.90,
               "maximum_bleed_fraction": 0.5}}

All source-head coefficients, energy deposition and turbine
performance parameters must be supplied: none is a recovered NERVA
test engine pump/turbine map. Turbine exhaust thrust is not modeled.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path

import numpy as np

from .lh2_feed_cycle import LH2Cycle
from .turbine_bleed_cycle import TurbineBleed, solve_turbine_bleed
from .regenerative_cooling import CoolingLeg, CoolingNetwork
from .resolved_transient import (
    ResolvedParameters, ResolvedState, read_resolved_heating, advance_resolved,
)
from .thermal_cycle_feedback import project_resolved_wall_power
from .openmc_cycle_coupling import (
    check_reconstruction_axial_edges, map_openmc_heating_to_cycle,
)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--chemistry-mode",choices=("frozen","equilibrium"),
                        default="frozen")
    parser.add_argument("--output",type=Path,
                        default=Path("build/nerva_shaft_closed_cycle.json"))
    # Explicit opt-in mapping of externally provided normalized OpenMC-style
    # power reconstructions. This CLI never executes or verifies transport.
    parser.add_argument("--openmc-mesh-fields", type=Path)
    parser.add_argument("--openmc-fuel-reconstruction", type=Path)
    parser.add_argument("--openmc-tie-reconstruction", type=Path)
    parser.add_argument("--fuel-solid-transfer-fraction", type=float)
    parser.add_argument("--tie-solid-transfer-fraction", type=float)
    parser.add_argument("--tie-solid-supply-fraction", type=float)
    parser.add_argument("--nozzle-wall-heat-w", type=float)
    parser.add_argument("--reflector-wall-heat-w", type=float)
    parser.add_argument("--openmc-transfer-mode",
                        choices=("fractions", "resolved-implicit"),
                        default="fractions")
    parser.add_argument("--resolved-thermal-parameters", type=Path)
    parser.add_argument("--resolved-thermal-state", type=Path)
    parser.add_argument("--resolved-dt-s", type=float)
    parser.add_argument("--resolved-inlet-temperature-k", type=float)
    args=parser.parse_args()
    raw=json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw,dict) or not {"lh2_cycle","turbine"} <= set(raw) or set(raw)-{"lh2_cycle","turbine","regenerative_cooling"}:
        raise ValueError("JSON must contain lh2_cycle and turbine objects")
    def build(cls,item):
        valid={f.name for f in fields(cls)}
        if not isinstance(item,dict) or set(item)-valid:
            raise ValueError(f"unknown {cls.__name__} inputs")
        return cls(**item)
    core=build(LH2Cycle,raw["lh2_cycle"])
    turbine=build(TurbineBleed,raw["turbine"])
    cooling=None
    if "regenerative_cooling" in raw:
        config=raw["regenerative_cooling"]
        expected={"nozzle_jacket","reflector","tie_supply","tie_return"}
        if not isinstance(config,dict) or set(config)-(expected|{"min_branch_fraction"}):
            raise ValueError("unknown regenerative cooling fields")
        sections={}
        for name in expected:
            if name not in config or not isinstance(config[name],dict):
                raise ValueError(f"cooling leg missing: {name}")
            sections[name]=CoolingLeg(**config[name])
        cooling=CoolingNetwork(**sections,
                               min_branch_fraction=config.get("min_branch_fraction",0.02))
    power_bridge=None
    sources=(
        args.openmc_mesh_fields, args.openmc_fuel_reconstruction,
        args.openmc_tie_reconstruction, args.nozzle_wall_heat_w,
        args.reflector_wall_heat_w,
    )
    fractions=(
        args.fuel_solid_transfer_fraction, args.tie_solid_transfer_fraction,
        args.tie_solid_supply_fraction,
    )
    temporal=(
        args.resolved_thermal_parameters, args.resolved_thermal_state,
        args.resolved_dt_s, args.resolved_inlet_temperature_k,
    )
    requested=(args.openmc_transfer_mode=="resolved-implicit"
               or any(v is not None for v in (*sources,*fractions,*temporal)))
    if requested:
        if cooling is None or not all(v is not None for v in sources):
            raise ValueError(
                "OpenMC power mapping requires regenerative_cooling, all "
                "three reconstruction paths and two declared external wall rates"
            )
        if args.openmc_transfer_mode=="fractions":
            if not all(v is not None for v in fractions) or any(
                v is not None for v in temporal
            ):
                raise ValueError(
                    "fraction mode requires three fractions and forbids "
                    "resolved-transient state options"
                )
        else:
            if any(v is not None for v in fractions) or not all(
                v is not None for v in temporal
            ):
                raise ValueError(
                    "resolved-implicit requires state NPZ, parameters JSON, "
                    "positive dt and inlet temperature; fractions are forbidden"
                )
        check_reconstruction_axial_edges(
            args.openmc_fuel_reconstruction,
            args.openmc_tie_reconstruction, core,
        )
        heating=read_resolved_heating(
            args.openmc_fuel_reconstruction,
            args.openmc_tie_reconstruction,
            args.openmc_mesh_fields,
        )
        diagnostics=None
        if args.openmc_transfer_mode=="fractions":
            core,cooling,ledger=map_openmc_heating_to_cycle(
                heating,core,cooling,
                fuel_solid_transfer_fraction=args.fuel_solid_transfer_fraction,
                tie_solid_transfer_fraction=args.tie_solid_transfer_fraction,
                tie_solid_supply_fraction=args.tie_solid_supply_fraction,
                nozzle_wall_heat_w=args.nozzle_wall_heat_w,
                reflector_wall_heat_w=args.reflector_wall_heat_w,
            )
        else:
            raw_parameters=json.loads(
                args.resolved_thermal_parameters.read_text(encoding="utf-8")
            )
            parameters=build(ResolvedParameters,raw_parameters)
            with np.load(args.resolved_thermal_state,allow_pickle=False) as data:
                required=(
                    "fuel_solid_t_k","fuel_h2_t_k","tie_solid_t_k",
                    "tie_supply_t_k","tie_return_t_k","unresolved_energy_j",
                )
                if not set(required).issubset(data.files):
                    raise ValueError("resolved old-state NPZ missing thermal fields")
                stored=np.asarray(data["unresolved_energy_j"],dtype=float)
                if stored.size != 1:
                    raise ValueError("old unresolved reservoir requires scalar joules")
                old_state=ResolvedState(
                    *(np.asarray(data[key],dtype=float).copy()
                      for key in required[:-1]),
                    unresolved_energy_j=float(stored.reshape(-1)[0]),
                )
            step=advance_resolved(
                heating,old_state,parameters,
                args.resolved_inlet_temperature_k,args.resolved_dt_s,
            )
            core,cooling,ledger=project_resolved_wall_power(
                heating,old_state,step,parameters,core,cooling,
                args.resolved_dt_s,
                nozzle_wall_heat_w=args.nozzle_wall_heat_w,
                reflector_wall_heat_w=args.reflector_wall_heat_w,
            )
            diagnostics={
                "thermal_model":"one-way implicit finite-volume solid/H2 "
                                "heat transfer, with assumed cell Cs and UA",
                "old_state_path":str(args.resolved_thermal_state.resolve()),
                "parameters":asdict(parameters),
                "dt_s":args.resolved_dt_s,
                "thermal_inlet_temperature_k":args.resolved_inlet_temperature_k,
                "thermal_reconstruction_fuel_flow_kg_s":
                    heating.total_fuel_flow_kg_s,
                "thermal_reconstruction_tie_flow_kg_s":
                    heating.total_tie_flow_kg_s,
                "thermal_step_energy_residual_j":step.energy_residual_j,
                "thermal_flow_not_iterated_with_engine":True,
                "full_conjugate_cycle_closed":False,
            }
        power_bridge={
            "source_kind":"external NPZ reconstruction, not independently "
                          "verified against an OpenMC statepoint",
            "nuclear_transport_run_in_this_cli":False,
            "transfer_mode":args.openmc_transfer_mode,
            "power_ledger":asdict(ledger),
            "resolved_transient":diagnostics,
            "source_files":{
                "mesh":str(args.openmc_mesh_fields.resolve()),
                "fuel":str(args.openmc_fuel_reconstruction.resolve()),
                "tie":str(args.openmc_tie_reconstruction.resolve()),
            },
            "mapped_cycle_inputs":{
                "lh2_cycle":asdict(core),
                "regenerative_cooling":asdict(cooling),
            },
        }
    point=solve_turbine_bleed(core,turbine,chemistry_mode=args.chemistry_mode,
                              cooling_network=cooling)
    if (power_bridge is not None
            and power_bridge["resolved_transient"] is not None):
        diag=power_bridge["resolved_transient"]
        diag["engine_solved_core_mass_flow_kg_s"]=point.core_mass_flow_kg_s
        diag["engine_solved_tie_mass_flow_kg_s"]=(
            point.regenerative_cooling.tie_tube_mass_flow_kg_s
        )
        diag["fuel_mass_flow_relative_mismatch"]=abs(
            diag["thermal_reconstruction_fuel_flow_kg_s"]-
            point.core_mass_flow_kg_s
        )/point.core_mass_flow_kg_s
        diag["tie_mass_flow_relative_mismatch"]=abs(
            diag["thermal_reconstruction_tie_flow_kg_s"]-
            point.regenerative_cooling.tie_tube_mass_flow_kg_s
        )/point.regenerative_cooling.tie_tube_mass_flow_kg_s
    data={
        "status":"PASS",
        "model":"single bleed turbine shaft drives cryogenic LH2 pump; "
                "lumped preheater and one hot fuel-channel family",
        "real_openmc_transport_performed":False,
        "historical_nerva_validation":False,
        "turbine_exhaust_thrust_included":False,
        "complete_turbomachinery_map":False,
        "parallel_cooling_branches_solved": cooling is not None,
        "reconstructed_openmc_heating_mapped": power_bridge is not None,
        "power_bridge":power_bridge,
        "inputs":raw,
        "point":{k:v for k,v in asdict(point).items() if k!="nozzle"},
        "nozzle":{
            "mass_flow_kg_s":point.nozzle.mass_flow_kg_s,
            "thrust_n":point.nozzle.thrust_n,
            "isp_s":point.nozzle.isp_s,
            "exit_temperature_k":point.nozzle.exit.temperature_k,
            "exit_pressure_pa":point.nozzle.exit.pressure_pa,
            "exit_velocity_m_s":point.nozzle.exit_velocity_m_s,
        },
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(data,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    print("NERVA LH2 pump / warm-H2 turbine bleed / hot core / nozzle: PASS")
    print(f"  total flow: {point.tank_mass_flow_kg_s:.7g} kg/s")
    print(f"  hot-core flow: {point.core_mass_flow_kg_s:.7g} kg/s")
    print(f"  bleed fraction: {point.turbine_bleed_fraction:.6g}")
    if point.regenerative_cooling is not None:
        print(f"  tie-branch fraction: {point.regenerative_cooling.tie_tube_fraction:.6g}")
        print(f"  coolant mixed header T: {point.regenerative_cooling.common_header_temperature_k:.6g} K")
    print(f"  pump shaft: {point.pump_required_power_w:.6g} W; "
          f"turbine delivered: {point.turbine_shaft_delivered_power_w:.6g} W")
    print(f"  chamber: {point.chamber_pressure_pa/1e6:.6g} MPa, "
          f"{point.chamber_temperature_k:.6g} K")
    print(f"  main-nozzle thrust: {point.main_nozzle_thrust_n:.6g} N")
    print(f"  nozzle-only Isp: {point.main_nozzle_isp_s:.6g} s")
    print("  Isp using total tank flow, excluding turbine exhaust thrust: "
          f"{point.effective_isp_excluding_bleed_thrust_s:.6g} s")
    print(f"  global first-law error: {point.total_cycle_energy_residual_w:.7g} W")
    print(f"  report: {args.output}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
