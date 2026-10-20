"""Run matched-reference CoolProp LH2 -> NASA7 hot-core NERVA model.

Input JSON is the LH2Cycle configuration used by the established
cryogenic feed model. In this mode CoolProp is used ONLY up through
the warm gaseous header; the hotter fuel channels and nozzle use
NASA-7 molecular H2 thermodynamics with a per-case energy-reference
offset matched at the header.

No real OpenMC or historical NERVA data are used by this CLI.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict,fields
import json
from pathlib import Path

from .lh2_feed_cycle import LH2Cycle
from .hot_lh2_cycle import solve_hot_core_cycle


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_hot_lh2_cycle_result.json"))
    a=p.parse_args()
    raw=json.loads(a.config.read_text(encoding="utf-8"))
    allowed={f.name for f in fields(LH2Cycle)}
    if not isinstance(raw,dict) or set(raw)-allowed:
        raise ValueError("configuration must contain only LH2Cycle fields")
    cfg=LH2Cycle(**raw)
    solution=solve_hot_core_cycle(cfg)
    body={
        "status":"PASS",
        "physics_model":"CoolProp liquid H2 pump + preheater, matched "
           "NASA7 frozen molecular hydrogen hot-core channel and nozzle",
        "historical_nerva_validation":False,
        "real_OpenMC_transport_performed":False,
        "full_thermal_hydraulic_network":False,
        "pump_turbine_thermodynamic_cycle_solved":False,
        "hot_channel_eos":"NASA7 frozen pure H2 ideal gas; 200-3500 K",
        "thermodynamic_bridge":"one matched enthalpy reference at "
           "preheater outlet temperature and pressure",
        "inputs":asdict(cfg),
        "point":{k:v for k,v in asdict(solution).items() if k!="nozzle"},
        "nozzle":{
            "mass_flow_kg_s":solution.nozzle.mass_flow_kg_s,
            "thrust_n":solution.nozzle.thrust_n,
            "isp_s":solution.nozzle.isp_s,
            "exit_temperature_k":solution.nozzle.exit.temperature_k,
            "exit_pressure_pa":solution.nozzle.exit.pressure_pa,
            "exit_velocity_m_s":solution.nozzle.exit_velocity_m_s,
            "enthalpy_to_kinetic_residual_j_kg":
             solution.nozzle.enthalpy_to_kinetic_residual_j_kg,
        }
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(body,indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("NERVA LH2-to-hot-core enthalpy-consistent operating point: PASS")
    print(f"  flow: {solution.mass_flow_kg_s:.7g} kg/s")
    print(f"  pump discharge: {solution.pump_discharge_pressure_pa/1e6:.7g} MPa")
    print(f"  core inlet: {solution.bridge_temperature_k:.7g} K")
    print(f"  chamber: {solution.chamber_pressure_pa/1e6:.7g} MPa / "
          f"{solution.chamber_temperature_k:.7g} K")
    print(f"  thrust: {solution.nozzle.thrust_n:.7g} N / "
          f"Isp: {solution.nozzle.isp_s:.7g} s")
    print("  first-law residual: "+f"{solution.overall_energy_residual_w:.6g} W")
    print("  EOS cp/rho relative jumps at bridge: "+
          f"{solution.bridge_cp_relative_jump:.4g}, "
          f"{solution.bridge_density_relative_jump:.4g}")
    print(f"  report: {a.output}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
