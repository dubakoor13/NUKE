"""Command-line entry point for the NERVA-derived model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import NervaConfig
from .geometry import cluster_summary, geometry_summary, mixed_core_summary
from .periphery import reactor_summary
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
        choices=("core", "cluster", "mixed-core", "reactor"),
        default="core",
        help="Export a core/cluster model or the full reactor-periphery assembly.",
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
        "--drum-angle",
        type=float,
        default=0.0,
        help="Control-drum absorber angle in degrees; 0 points the absorber half-shell inward.",
    )
    parser.add_argument(
        "--photon-transport",
        action="store_true",
        help=(
            "Enable coupled neutron-photon transport when the installed "
            "OpenMC nuclear-data library supports photon interactions."
        ),
    )
    parser.add_argument(
        "--diagnostic-energy-groups",
        type=int,
        default=80,
        help="Number of logarithmic energy bins for diagnostic spectra.",
    )
    parser.add_argument(
        "--axial-mesh-bins",
        type=int,
        default=96,
        help="Number of fine axial bins for OpenMC profile tallies.",
    )
    parser.add_argument(
        "--no-element-instance-tallies",
        action="store_true",
        help="Disable repeated fuel/tie instance tallies.",
    )
    parser.add_argument(
        "--no-channel-instance-tallies",
        action="store_true",
        help="Disable 19-channel repeated-cell tallies.",
    )
    parser.add_argument(
        "--target-rel-error",
        type=float,
        default=None,
        help=(
            "Optional heating-local relative-error tally trigger. "
            "When set, OpenMC may continue beyond --batches up to "
            "--trigger-max-batches."
        ),
    )
    parser.add_argument(
        "--trigger-max-batches",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--trigger-batch-interval",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run OpenMC after exporting XML (requires executable + nuclear data).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = NervaConfig(
        core_rings=args.rings,
        particles=args.particles,
        control_drum_angle_deg=args.drum_angle,
        photon_transport=args.photon_transport,
        diagnostic_energy_groups=args.diagnostic_energy_groups,
        axial_mesh_bins=args.axial_mesh_bins,
        element_instance_tallies=not args.no_element_instance_tallies,
        channel_instance_tallies=not args.no_channel_instance_tallies,
        tally_rel_err_trigger=args.target_rel_error,
        trigger_max_batches=args.trigger_max_batches,
        trigger_batch_interval=args.trigger_batch_interval,
    )
    config.validate()

    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model(config, assembly=args.assembly)
    model.export_to_xml(directory=args.output)

    if args.assembly == "cluster":
        summary = cluster_summary(config)
    elif args.assembly == "mixed-core":
        summary = mixed_core_summary(config)
    elif args.assembly == "reactor":
        summary = reactor_summary(config)
    else:
        summary = geometry_summary(config)
    print(f"NERVA-derived {args.assembly} model exported:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"  photon transport: {config.photon_transport}")
    print(f"  diagnostic energy groups: {config.diagnostic_energy_groups}")
    print(f"  axial diagnostic bins: {config.axial_mesh_bins}")
    print(f"  element instance tallies: {config.element_instance_tallies}")
    print(f"  channel instance tallies: {config.channel_instance_tallies}")
    print(f"  tally relative-error trigger: {config.tally_rel_err_trigger}")
    print(f"  output: {args.output}")

    if args.run:
        model.run(cwd=args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
