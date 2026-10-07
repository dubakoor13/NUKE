"""Print/export the four source-backed alternative-propellant NTP cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .alternative_propellant_cases import FOUR_CASES


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
        "--output",
        type=Path,
        default=Path("build/alternative_ntp_cases.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.case is None:
        payload = {
            key: asdict(case)
            for key, case in FOUR_CASES.items()
        }
    else:
        payload = {args.case: asdict(FOUR_CASES[args.case])}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {len(payload)} source-backed NTP case(s): {args.output}")
    for key, case in payload.items():
        print(f"  {key}: {case['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
