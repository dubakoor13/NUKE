"""Run the bounded CoolProp LH2 pump/preheater/gas-core/nozzle network.

The JSON file must contain valid LH2Cycle parameters. This simulation
has a real-fluid cryogenic pump, a lumped preheater, and gas-only
subsonic core; it is NOT an experimentally validated NERVA cycle.

  python -m ntr.nerva.run_lh2_feed_cycle \
    --config build/lh2_cycle_example.json \
    --output build/lh2_cycle_result.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path

from .lh2_feed_cycle import LH2Cycle, solve_lh2_cycle


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,default=Path("build/lh2_cycle_result.json"))
    a=p.parse_args()
    data=json.loads(a.config.read_text(encoding="utf-8"))
    names={x.name for x in fields(LH2Cycle)}
    if not isinstance(data,dict) or set(data)-names:
        raise ValueError("expected JSON object of LH2Cycle parameters")
    inp=LH2Cycle(**data)
    inp.validate()
    res=solve_lh2_cycle(inp)
    report={
        "status":"PASS","study_scope":"single-phase LH2 pump and warm-core "
          "real-fluid H2 with lumped preheater and local-gamma nozzle surrogate",
        "historical_nerva_validation":False,
        "full_nerva_turbopump_and_turbine_cycle":False,
        "reactor_thermal_power_source":"externally specified preheater and channel heat",
        "coolprop_hydrogen_eos_upper_validity_k":1000.,
        "configured_warm_gas_upper_k":inp.maximum_eos_temperature_k,
        "inputs":asdict(inp),
        "operating_point":{k:v for k,v in asdict(res).items() if k!="nozzle"},
        "nozzle":asdict(res.nozzle),
    }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("LH2 cryogenic feed / warm H2 core / nozzle coupled case: PASS")
    print(f"  flow: {res.mass_flow_kg_s:.6g} kg/s")
    print(f"  pump discharge: {res.pump_discharge_pressure_pa/1e6:.6g} MPa")
    print(f"  LH2 pump work: {res.pump_shaft_power_w/1e3:.6g} kW")
    print(f"  warm core inlet: {res.core_inlet_temperature_k:.6g} K")
    print(f"  chamber: {res.chamber_pressure_pa/1e6:.6g} MPa, {res.chamber_temperature_k:.6g} K")
    print(f"  thrust/Isp: {res.nozzle.thrust_n:.6g} N, {res.nozzle.isp_s:.6g} s")
    print(f"  mass/energy/pressure residuals: {res.mass_residual_kg_s:.5g} kg/s, "
          f"{res.energy_residual_w:.5g} W, {res.pressure_residual_pa:.5g} Pa")
    print(f"  report: {a.output}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
