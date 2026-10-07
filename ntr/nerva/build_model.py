"""Command-line entry point for the NERVA-derived model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import NervaConfig
from .geometry import cluster_summary, geometry_summary, mixed_core_summary
from .model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a NERVA-derived OpenMC demonstration model."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_demo"),
        help="Directory for exported OpenMC XML files.",
    )
    parser.add_argument(
        "--assembly",
        choices=("core", "cluster", "mixed-core"),
        default="core",
        help="Export the all-fuel core, 6-fuel/1-tie cluster, or mixed fuel/tie core.",
    )
    parser.add_argument(
        "--rings",
        type=int,
        default=3,
        help="Number of hexagonal lattice rings in the all-fuel core demonstrator.",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=4000,
        help="Particles per batch.",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run OpenMC after exporting XML (requires executable + nuclear data).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = NervaConfig(core_rings=args.rings, particles=args.particles)
    config.validate()

    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(config, assembly=args.assembly)
    model.export_to_xml(directory=args.output)

    if args.assembly == "cluster":
        summary = cluster_summary(config)
    elif args.assembly == "mixed-core":
        summary = mixed_core_summary(config)
    else:
        summary = geometry_summary(config)
    print(f"NERVA-derived {args.assembly} model exported:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  output: {args.output}")

    if args.run:
        model.run(cwd=args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
