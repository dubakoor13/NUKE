"""Nuclear-data preflight and provenance helpers for NERVA OpenMC runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from .config import NervaConfig
from .materials import build_materials, required_bound_scattering_tables


@dataclass(frozen=True)
class LibraryEntry:
    data_type: str
    materials: tuple[str, ...]
    path: str
    file_exists: bool
    resolved_path: str


@dataclass(frozen=True)
class NuclearDataReport:
    cross_sections_xml: str
    sha256: str
    neutron_materials: tuple[str, ...]
    photon_materials: tuple[str, ...]
    thermal_materials: tuple[str, ...]
    required_neutron_nuclides: tuple[str, ...]
    missing_neutron_nuclides: tuple[str, ...]
    required_thermal_scattering_tables: tuple[str, ...]
    missing_thermal_scattering_tables: tuple[str, ...]
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
    """Resolve library files exactly once, honoring OpenMC's <directory>.

    For reproducible CLI runs, an explicit <directory> must be absolute.
    OpenMC interprets a relative <directory> against its process working
    directory, which may differ from the Python preflight's working directory
    (model.run is launched in the export directory). Reject that ambiguity
    rather than claiming files were verified at a different location.
    """
    root = ET.parse(cross_sections).getroot()
    directory_node = root.find("directory")
    if directory_node is None:
        library_root = cross_sections.parent.resolve()
    else:
        text = (directory_node.text or "").strip()
        if not text:
            raise ValueError("cross_sections.xml contains empty <directory>")
        library_root = Path(text).expanduser()
        if not library_root.is_absolute():
            raise ValueError(
                "cross_sections.xml <directory> must be absolute for "
                "unambiguous OpenMC transport preflight; "
                "relative directories depend on OpenMC's working directory"
            )
        library_root = library_root.resolve()

    entries: list[LibraryEntry] = []
    for node in root.findall("library"):
        data_type = node.attrib.get("type", "").strip()
        materials = tuple(node.attrib.get("materials", "").split())
        raw_path = node.attrib.get("path", "").strip()
        if not raw_path:
            raise ValueError("cross_sections.xml <library> missing path")
        file_path = Path(raw_path).expanduser()
        if not file_path.is_absolute():
            file_path = library_root / file_path
        file_path = file_path.resolve()
        entries.append(
            LibraryEntry(
                data_type=data_type,
                materials=materials,
                path=raw_path,
                file_exists=file_path.is_file(),
                resolved_path=str(file_path),
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
    needed_bound = set(
        required_bound_scattering_tables(config or NervaConfig())
    )
    missing_thermal = tuple(sorted(needed_bound - thermal_materials))

    missing_files = tuple(
        entry.resolved_path
        for entry in entries
        if not entry.file_exists
    )

    photon_present = bool(photon_materials)
    ready = (
        not missing
        and not missing_thermal
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
        required_thermal_scattering_tables=tuple(sorted(needed_bound)),
        missing_thermal_scattering_tables=missing_thermal,
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
