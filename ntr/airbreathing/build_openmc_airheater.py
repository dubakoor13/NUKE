"""Build/run the generic fixed-source OpenMC air-heater model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .openmc_airheater import (
    GenericAirHeaterConfig,
    build_openmc_airheater,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a fixed, generic OpenMC nuclear-air-heater benchmark. "
            "No criticality or weapon-specific optimization is performed."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/openmc_airheater"),
    )
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--particles", type=int, default=5000)
    parser.add_argument(
        "--photon-transport",
        action="store_true",
        help="Enable coupled photon transport when compatible data are available.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "Run OpenMC after exporting XML. Requires a compatible "
            "OPENMC_CROSS_SECTIONS data library."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = GenericAirHeaterConfig(
        batches=args.batches,
        particles=args.particles,
        photon_transport=args.photon_transport,
    )
    build = build_openmc_airheater(config)

    args.output.mkdir(parents=True, exist_ok=True)
    build.model.export_to_xml(directory=args.output)

    print("Generic fixed-source OpenMC air heater exported:")
    print("  run mode: fixed source")
    print("  k-effective calculation: disabled")
    print("  enrichment optimization: disabled")
    print(f"  photon transport: {config.photon_transport}")
    print(f"  mesh: {config.mesh_dimension}")
    print(f"  batches: {config.batches}")
    print(f"  particles/batch: {config.particles}")
    print(f"  output: {args.output}")

    if args.run:
        statepoint = build.model.run(cwd=args.output)
        print(f"  statepoint: {statepoint}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
