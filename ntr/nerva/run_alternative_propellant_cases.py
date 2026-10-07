"""Export/reconstruct the four source-backed alternative-propellant NTP cases."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .alternative_propellant_cases import FOUR_CASES, MethanePbmCase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export source-backed H-NTP, A-NTP and M-NTP cases."
    )
    parser.add_argument(
        "--case",
        choices=tuple(FOUR_CASES),
        default=None,
        help="Export one case; default exports all four.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/alternative_ntp_cases"),
    )
    return parser.parse_args()


def _write_state_csv(path: Path, case: MethanePbmCase) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "state",
                "mass_flow_kg_s",
                "temperature_K",
                "pressure_MPa",
            )
        )
        for state in case.states:
            writer.writerow(
                (
                    state.state,
                    state.mass_flow_kg_s,
                    state.temperature_k,
                    state.pressure_mpa,
                )
            )


def _case_payload(case):
    payload = asdict(case)
    if isinstance(case, MethanePbmCase):
        payload["derived"] = {
            "thrust_N_from_mdot_Isp": case.derived_thrust_n,
            "thrust_kN_from_mdot_Isp": case.derived_thrust_n / 1000.0,
            "chamber_pressure_MPa": case.chamber_pressure_mpa,
            "pump_pressure_rise_MPa": case.pump_pressure_rise_mpa,
            "pump_power_margin_kW": (
                case.turbine_power_kw - case.pump_power_kw
            ),
        }
    return payload


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = (
        FOUR_CASES
        if args.case is None
        else {args.case: FOUR_CASES[args.case]}
    )

    payload = {
        key: _case_payload(case)
        for key, case in selected.items()
    }

    json_path = args.output_dir / "alternative_ntp_cases.json"
    json_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    for key, case in selected.items():
        if isinstance(case, MethanePbmCase):
            _write_state_csv(
                args.output_dir / f"{key}_states.csv",
                case,
            )

    print(f"reconstructed {len(selected)} source-backed NTP case(s)")
    print(f"  JSON: {json_path}")
    for key, case in selected.items():
        if isinstance(case, MethanePbmCase):
            print(
                f"  {key}: {case.isp_s:.2f} s, "
                f"{case.total_mass_flow_kg_s:.2f} kg/s, "
                f"{case.derived_thrust_n/1000.0:.2f} kN, "
                f"Pc={case.chamber_pressure_mpa:.2f} MPa"
            )
        else:
            print(
                f"  {key}: reference Isp "
                f"{case.reference_isp_s:.0f} s at "
                f"{case.reference_temperature_k:.0f} K"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
