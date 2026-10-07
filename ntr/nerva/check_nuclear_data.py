"""CLI for checking OpenMC nuclear data against the NERVA model."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import NervaConfig
from .nuclear_data import inspect_nuclear_data, write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that a cross_sections.xml contains the neutron data "
            "required by the current low-enrichment NERVA demonstrator."
        )
    )
    parser.add_argument(
        "--cross-sections",
        type=Path,
        default=None,
        help=(
            "cross_sections.xml; defaults to OPENMC_CROSS_SECTIONS."
        ),
    )
    parser.add_argument("--photon-transport", action="store_true")
    parser.add_argument(
        "--skip-file-existence-check",
        action="store_true",
        help="Validate XML inventory without checking referenced HDF5 files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_nuclear_data_report.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = NervaConfig(photon_transport=args.photon_transport)

    report = inspect_nuclear_data(
        cross_sections=args.cross_sections,
        config=config,
        photon_transport=args.photon_transport,
        require_referenced_files=not args.skip_file_existence_check,
    )
    write_report(report, args.output)

    print("NERVA OpenMC nuclear-data preflight:")
    print(f"  cross sections: {report.cross_sections_xml}")
    print(f"  SHA-256: {report.sha256}")
    print(
        f"  required neutron nuclides: "
        f"{len(report.required_neutron_nuclides)}"
    )
    print(
        f"  missing neutron nuclides: "
        f"{len(report.missing_neutron_nuclides)}"
    )
    if report.missing_neutron_nuclides:
        print(
            "  missing: "
            + ", ".join(report.missing_neutron_nuclides)
        )
    print(
        f"  photon library present: "
        f"{report.photon_library_present}"
    )
    print(
        f"  referenced files missing: "
        f"{len(report.referenced_files_missing)}"
    )
    print(f"  READY: {report.ready}")
    print(f"  report: {args.output}")

    return 0 if report.ready else 2


if __name__ == "__main__":
    raise SystemExit(main())
