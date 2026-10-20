"""Run the single-phase gaseous-H2 reduced feed/core/nozzle network.

Input is a JSON object with every GasFeedNetwork dataclass parameter.
This CLI does NOT size pumps/rotors or perform LH2 two-phase calculations.

  python -m ntr.nerva.run_gas_feed_network \
     --config build/gas_feed_network_case.json \
     --output build/gas_feed_network_result.json

A synthetic/assumed example configuration can be generated via:
  python -m ntr.nerva.validate_gas_feed_network --write-example FILE.json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path

from .gas_feed_network import GasFeedNetwork, solve_gas_network


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_gas_feed_network_result.json"))
    return p.parse_args()


def main() -> int:
    args=parse_args()
    raw=json.loads(args.config.read_text(encoding="utf-8"))
    allowed={x.name for x in fields(GasFeedNetwork)}
    if not isinstance(raw,dict) or set(raw)-allowed:
        raise ValueError("input must be an object of known GasFeedNetwork fields")
    config=GasFeedNetwork(**raw)
    config.validate()
    solution=solve_gas_network(config)
    info={
        "status":"PASS",
        "model":"quasi-steady pure gaseous H2 compressor/feed/fuel-core/nozzle",
        "historical_nerva_validation":False,
        "cryogenic_turbopump_model":False,
        "reactor_power_source":"fixed externally prescribed channel heat",
        "input":asdict(config),
        "operating_point":{
            k:v for k,v in asdict(solution).items()
            if k not in ("channel","nozzle")
        },
        "nozzle":asdict(solution.nozzle),
        "channel":{
            "outlet_temperature_k":solution.channel.outlet_temperature_k,
            "outlet_pressure_pa":solution.channel.outlet_pressure_pa,
            "maximum_mach":float(max(solution.channel.mach_number)),
            "friction_pressure_loss_pa":float(
                sum(solution.channel.friction_pressure_loss_pa)
            ),
            "acceleration_pressure_loss_pa":float(
                sum(solution.channel.acceleration_pressure_loss_pa)
            ),
            "maximum_energy_residual_w":float(
                max(abs(x) for x in solution.channel.energy_residual_w)
            ),
            "maximum_momentum_residual_pa":float(
                max(abs(x) for x in solution.channel.momentum_residual_pa)
            ),
        },
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(
        json.dumps(info,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    print("Gas-phase NERVA feed/core/nozzle network: PASS")
    print(f"  total mass flow {solution.total_mass_flow_kg_s:.7g} kg/s")
    print(f"  chamber pressure {solution.chamber_pressure_pa/1e6:.6g} MPa")
    print(f"  chamber temperature {solution.chamber_temperature_k:.6g} K")
    print(f"  compressor gas-shaft power {solution.compressor_shaft_power_w:.6g} W")
    print(f"  nozzle thrust {solution.nozzle.thrust_n:.6g} N")
    print(f"  nozzle Isp {solution.nozzle.isp_s:.6g} s")
    print(f"  output {args.output}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
