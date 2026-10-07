"""Couple processed OpenMC heating to the generic airbreathing cycle."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .nuclear_ramjet import evaluate_nuclear_ramjet_from_thermal_power


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use generic OpenMC deposited heating as the heat source for the "
            "generic nuclear-airbreathing thermodynamic cycle."
        )
    )
    parser.add_argument(
        "metadata",
        type=Path,
        help="openmc_airheater_metadata.json from postprocessing",
    )
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--ambient-temperature-k", type=float, default=220.0)
    parser.add_argument("--ambient-pressure-kpa", type=float, default=25.0)
    parser.add_argument("--air-mass-flow-kg-s", type=float, default=10.0)
    parser.add_argument(
        "--heat-source",
        choices=("total", "fuel"),
        default="fuel",
        help=(
            "Use all OpenMC deposited power or only the generic fuel-region "
            "deposition as the solid heat source."
        ),
    )
    parser.add_argument(
        "--heat-transfer-efficiency",
        type=float,
        default=0.85,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/openmc_airbreathing_cycle.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    if args.heat_source == "total":
        power_w = float(meta["deposited_power_W"])
    else:
        power_w = float(meta["region_power_W"]["fuel"])

    result = evaluate_nuclear_ramjet_from_thermal_power(
        mach=args.mach,
        ambient_temperature_k=args.ambient_temperature_k,
        ambient_pressure_pa=args.ambient_pressure_kpa * 1000.0,
        air_mass_flow_kg_s=args.air_mass_flow_kg_s,
        deposited_thermal_power_w=power_w,
        heat_transfer_efficiency=args.heat_transfer_efficiency,
    )

    payload = asdict(result)
    payload.update(
        {
            "openmc_metadata": str(args.metadata),
            "selected_heat_source": args.heat_source,
            "selected_deposited_power_W": power_w,
            "heat_transfer_efficiency": args.heat_transfer_efficiency,
            "model_status": (
                "generic OpenMC fixed-source heating coupled to a generic "
                "airbreathing thermodynamic cycle; not a weapon model"
            ),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("OpenMC -> generic airbreathing coupling complete:")
    print(f"  selected deposited power: {power_w/1.0e6:.6f} MW")
    print(
        f"  reactor outlet total temperature: "
        f"{result.reactor_outlet_total_temperature_k:.2f} K"
    )
    print(f"  net thrust: {result.net_thrust_n/1000.0:.3f} kN")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
