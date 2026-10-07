"""Command-line NERVA engine performance calculation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .historical_presets import (
    LBF_TO_N,
    NERVA_75K_1972,
    NERVA_GRAPHITE_DERIVATIVE,
)
from .nerva_engine import (
    NervaGasModel,
    evaluate_nerva_nozzle,
    throat_area_for_target_thrust,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a first-order NERVA hot-hydrogen nozzle using fixed "
            "expansion-ratio compressible-flow equations."
        )
    )
    parser.add_argument(
        "--preset",
        choices=("1972-nerva", "graphite-derivative"),
        default="1972-nerva",
    )
    parser.add_argument("--gamma", type=float, default=1.35)
    parser.add_argument("--nozzle-efficiency", type=float, default=0.95)
    parser.add_argument("--ambient-pressure-kpa", type=float, default=0.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_engine.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.preset == "1972-nerva":
        preset = NERVA_75K_1972
        chamber_temperature_k = sum(
            preset.chamber_temperature_range_k
        ) / 2.0
        target_thrust_n = 75_000.0 * LBF_TO_N
    else:
        preset = NERVA_GRAPHITE_DERIVATIVE
        chamber_temperature_k = preset.chamber_temperature_k
        target_thrust_n = 75_000.0 * LBF_TO_N

    gas = NervaGasModel(gamma=args.gamma)

    throat_area = throat_area_for_target_thrust(
        target_thrust_n=target_thrust_n,
        chamber_pressure_pa=preset.chamber_pressure_pa,
        chamber_temperature_k=chamber_temperature_k,
        expansion_ratio=preset.nozzle_expansion_ratio,
        ambient_pressure_pa=args.ambient_pressure_kpa * 1000.0,
        nozzle_efficiency=args.nozzle_efficiency,
        gas=gas,
    )

    result = evaluate_nerva_nozzle(
        chamber_pressure_pa=preset.chamber_pressure_pa,
        chamber_temperature_k=chamber_temperature_k,
        throat_area_m2=throat_area,
        expansion_ratio=preset.nozzle_expansion_ratio,
        ambient_pressure_pa=args.ambient_pressure_kpa * 1000.0,
        nozzle_efficiency=args.nozzle_efficiency,
        gas=gas,
    )

    payload = {
        "reference": preset.name,
        "source": preset.source_title,
        "chamber_pressure_MPa": result.chamber_pressure_pa / 1.0e6,
        "chamber_temperature_K": result.chamber_temperature_k,
        "expansion_ratio": result.expansion_ratio,
        "throat_area_m2": result.throat_area_m2,
        "exit_area_m2": result.exit_area_m2,
        "mass_flow_kg_s": result.mass_flow_kg_s,
        "exit_pressure_kPa": result.exit_pressure_pa / 1000.0,
        "exit_mach": result.exit_mach,
        "cstar_m_s": result.characteristic_velocity_m_s,
        "thrust_coefficient": result.thrust_coefficient,
        "thrust_kN": result.thrust_n / 1000.0,
        "Isp_s": result.isp_s,
        "effective_exhaust_velocity_km_s": (
            result.effective_exhaust_velocity_m_s / 1000.0
        ),
        "model": (
            "calorically-perfect hot-H2 nozzle; fixed expansion ratio; "
            "effective gamma/nozzle-efficiency engineering approximation"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA engine calculation complete:")
    print(f"  reference: {preset.name}")
    print(f"  Pc: {payload['chamber_pressure_MPa']:.3f} MPa")
    print(f"  Tc: {payload['chamber_temperature_K']:.1f} K")
    print(f"  mdot H2: {payload['mass_flow_kg_s']:.3f} kg/s")
    print(f"  thrust: {payload['thrust_kN']:.3f} kN")
    print(f"  Isp: {payload['Isp_s']:.3f} s")
    print(f"  throat area: {payload['throat_area_m2']:.6f} m^2")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
