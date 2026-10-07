"""CLI for the generic nuclear-airbreathing demonstrator."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .nuclear_ramjet import evaluate_nuclear_ramjet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a generic nuclear-heated airbreathing ramjet cycle. "
            "No weapon-specific geometry, range, or reactor design is modeled."
        )
    )
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--ambient-temperature-k", type=float, default=220.0)
    parser.add_argument("--ambient-pressure-kpa", type=float, default=25.0)
    parser.add_argument("--air-mass-flow-kg-s", type=float, default=10.0)
    parser.add_argument(
        "--reactor-outlet-temperature-k",
        type=float,
        default=1400.0,
    )
    parser.add_argument(
        "--diffuser-pressure-recovery",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--reactor-total-pressure-ratio",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--reactor-heat-transfer-efficiency",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--nozzle-efficiency",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nuclear_airbreathing_demo.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    result = evaluate_nuclear_ramjet(
        mach=args.mach,
        ambient_temperature_k=args.ambient_temperature_k,
        ambient_pressure_pa=args.ambient_pressure_kpa * 1000.0,
        air_mass_flow_kg_s=args.air_mass_flow_kg_s,
        reactor_outlet_total_temperature_k=(
            args.reactor_outlet_temperature_k
        ),
        diffuser_pressure_recovery=args.diffuser_pressure_recovery,
        reactor_total_pressure_ratio=args.reactor_total_pressure_ratio,
        reactor_heat_transfer_efficiency=(
            args.reactor_heat_transfer_efficiency
        ),
        nozzle_efficiency=args.nozzle_efficiency,
    )

    payload = asdict(result)
    payload["model_status"] = (
        "generic nuclear-airbreathing thermodynamic demonstrator; "
        "not a Burevestnik weapon model"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Nuclear-airbreathing demonstrator:")
    print(f"  Mach: {result.mach:.3f}")
    print(
        f"  reactor thermal power: "
        f"{result.reactor_thermal_power_w / 1.0e6:.3f} MW"
    )
    print(
        f"  net thrust: {result.net_thrust_n / 1000.0:.3f} kN"
    )
    print(
        f"  specific thrust: "
        f"{result.specific_thrust_n_s_kg:.3f} N/(kg/s)"
    )
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
