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

from .lh2_feed_cycle import LH2Cycle
from .turbine_bleed_cycle import TurbineBleed, solve_turbine_bleed


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--chemistry-mode",choices=("frozen","equilibrium"),
                        default="frozen")
    parser.add_argument("--output",type=Path,
                        default=Path("build/nerva_shaft_closed_cycle.json"))
    args=parser.parse_args()
    raw=json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(raw,dict) or set(raw)!={"lh2_cycle","turbine"}:
        raise ValueError("JSON must contain lh2_cycle and turbine objects")
    def build(cls,item):
        valid={f.name for f in fields(cls)}
        if not isinstance(item,dict) or set(item)-valid:
            raise ValueError(f"unknown {cls.__name__} inputs")
        return cls(**item)
    core=build(LH2Cycle,raw["lh2_cycle"])
    turbine=build(TurbineBleed,raw["turbine"])
    point=solve_turbine_bleed(core,turbine,chemistry_mode=args.chemistry_mode)
    data={
        "status":"PASS",
        "model":"single bleed turbine shaft drives cryogenic LH2 pump; "
                "lumped preheater and one hot fuel-channel family",
        "real_openmc_transport_performed":False,
        "historical_nerva_validation":False,
        "turbine_exhaust_thrust_included":False,
        "complete_turbomachinery_map":False,
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
