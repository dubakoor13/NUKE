"""Nuclear-data preflight and provenance helpers for NERVA OpenMC runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from .config import NervaConfig
from .materials import build_materials


@dataclass(frozen=True)
class LibraryEntry:
    data_type: str
    materials: tuple[str, ...]
    path: str
    file_exists: bool


@dataclass(frozen=True)
class NuclearDataReport:
    cross_sections_xml: str
    sha256: str
    neutron_materials: tuple[str, ...]
    photon_materials: tuple[str, ...]
    thermal_materials: tuple[str, ...]
    required_neutron_nuclides: tuple[str, ...]
    missing_neutron_nuclides: tuple[str, ...]
    entries: tuple[LibraryEntry, ...]
    photon_transport_requested: bool
    photon_library_present: bool
    referenced_files_missing: tuple[str, ...]
    ready: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def resolve_cross_sections(
    value: str | Path | None = None,
) -> Path:
    if value is None:
        value = os.environ.get("OPENMC_CROSS_SECTIONS")
    if not value:
        raise ValueError(
            "OpenMC nuclear data are not configured. Supply --cross-sections "
            "or set OPENMC_CROSS_SECTIONS to cross_sections.xml."
        )

    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"cross_sections.xml does not exist: {path}"
        )
    return path


def _library_entries(cross_sections: Path) -> tuple[LibraryEntry, ...]:
    root = ET.parse(cross_sections).getroot()
    base = cross_sections.parent

    entries: list[LibraryEntry] = []
    for node in root.findall(".//library"):
        data_type = node.attrib.get("type", "").strip()
        materials = tuple(
            item
            for item in node.attrib.get("materials", "").split()
            if item
        )
        raw_path = node.attrib.get("path", "").strip()
        file_path = Path(raw_path)
        if raw_path and not file_path.is_absolute():
            file_path = base / file_path

        entries.append(
            LibraryEntry(
                data_type=data_type,
                materials=materials,
                path=raw_path,
                file_exists=(not raw_path) or file_path.is_file(),
            )
        )

    if not entries:
        raise ValueError(
            f"no <library> entries found in {cross_sections}"
        )
    return tuple(entries)


def required_neutron_nuclides(
    config: NervaConfig | None = None,
) -> tuple[str, ...]:
    if config is None:
        config = NervaConfig()
    materials = build_materials(config)

    nuclides: set[str] = set()
    for material in materials.values():
        nuclides.update(material.get_nuclides())
    return tuple(sorted(nuclides))


def inspect_nuclear_data(
    cross_sections: str | Path | None = None,
    config: NervaConfig | None = None,
    photon_transport: bool = False,
    require_referenced_files: bool = True,
) -> NuclearDataReport:
    """Inspect the OpenMC library against the current NERVA materials.

    This is a compatibility/provenance check only. It does not alter model
    composition or search for a critical configuration.
    """
    path = resolve_cross_sections(cross_sections)
    entries = _library_entries(path)

    by_type: dict[str, set[str]] = {}
    for entry in entries:
        by_type.setdefault(entry.data_type, set()).update(entry.materials)

    neutron_materials = by_type.get("neutron", set())
    photon_materials = by_type.get("photon", set())
    thermal_materials = by_type.get("thermal", set())

    required = set(required_neutron_nuclides(config))
    missing = tuple(sorted(required - neutron_materials))

    missing_files = tuple(
        entry.path
        for entry in entries
        if entry.path and not entry.file_exists
    )

    photon_present = bool(photon_materials)
    ready = (
        not missing
        and (not require_referenced_files or not missing_files)
        and (not photon_transport or photon_present)
    )

    return NuclearDataReport(
        cross_sections_xml=str(path),
        sha256=_sha256(path),
        neutron_materials=tuple(sorted(neutron_materials)),
        photon_materials=tuple(sorted(photon_materials)),
        thermal_materials=tuple(sorted(thermal_materials)),
        required_neutron_nuclides=tuple(sorted(required)),
        missing_neutron_nuclides=missing,
        entries=entries,
        photon_transport_requested=bool(photon_transport),
        photon_library_present=photon_present,
        referenced_files_missing=missing_files,
        ready=ready,
    )


def write_report(report: NuclearDataReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
