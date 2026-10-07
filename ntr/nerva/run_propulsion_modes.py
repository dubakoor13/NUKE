"""Export the four requested NTP propulsion families."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .propulsion_modes import (
    FOUR_PROPULSION_MODES,
    lch4_lox_mass_bookkeeping,
    lch4_ntp_reference,
    lh2_lox_lantr_reference,
    lh2_ntp_reference,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lantr-mr", type=float, default=3.0)
    parser.add_argument("--methane-case", type=int, choices=(1, 2), default=2)
    parser.add_argument("--methane-lox-ratio", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/four_propulsion_modes.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "modes": {
            key: asdict(mode)
            for key, mode in FOUR_PROPULSION_MODES.items()
        },
        "operating_points": {
            "lh2_ntp": asdict(lh2_ntp_reference()),
            "lch4_ntp": asdict(
                lch4_ntp_reference(args.methane_case)
            ),
            "lh2_lox_lantr": asdict(
                lh2_lox_lantr_reference(args.lantr_mr)
            ),
            "lch4_lox_augmented": asdict(
                lch4_lox_mass_bookkeeping(
                    args.methane_lox_ratio,
                    methane_case_number=args.methane_case,
                )
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote four propulsion modes: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
