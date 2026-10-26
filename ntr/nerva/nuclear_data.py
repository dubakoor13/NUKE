"""Nuclear-data preflight and provenance helpers for NERVA OpenMC runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
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
    required_photon_elements: tuple[str, ...]
    missing_photon_elements: tuple[str, ...]
    verified_photon_elements: tuple[str, ...]
    photon_binary_audit_errors: tuple[str, ...]
    photon_binary_sha256: tuple[tuple[str, str], ...]
    photon_evaluated_release_verified: bool
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



def required_photon_elements(config: NervaConfig | None = None) -> tuple[str, ...]:
    """Photon HDF5 is indexed by ELEMENT, not by neutron isotope.

    Conservative requested-material inventory, just as the existing neutron
    preflight uses every declared material. This is not proof of root placement.
    """
    elements = set()
    for isotope in required_neutron_nuclides(config):
        match = re.fullmatch(r"([A-Z][a-z]?)[0-9]+(?:_m[0-9]+)?", isotope)
        if match is None:
            raise ValueError("unrecognized neutron isotope for photon mapping: " + isotope)
        elements.add(match.group(1))
    return tuple(sorted(elements))


def _verified_photon_binaries(
    entries: tuple[LibraryEntry, ...], required: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Read actual photon HDF5 through OpenMC. XML presence is insufficient.

    This validates basic data identity and required photo-atomic reactions,
    NOT the evaluated data source/release or consistency with hot neutronics.
    """
    by_symbol: dict[str, list[LibraryEntry]] = {}
    for entry in entries:
        if entry.data_type != "photon":
            continue
        for name in entry.materials:
            by_symbol.setdefault(name, []).append(entry)
    checked: list[str] = []
    problems: list[str] = []
    hashes: list[tuple[str, str]] = []
    try:
        from openmc.data import IncidentPhoton, ATOMIC_NUMBER
    except (ImportError, AttributeError) as exc:
        return (), ("OpenMC photon HDF5 reader unavailable: "+str(exc),), ()
    for symbol in required:
        rows = by_symbol.get(symbol, [])
        if not rows:
            continue  # missing indexed element is separately reported.
        if len(rows) != 1:
            problems.append(symbol + ": ambiguous duplicate photon index")
            continue
        record = rows[0]
        if len(record.materials) != 1:
            problems.append(symbol + ": multi-element photon HDF5 entry cannot be uniquely verified")
            continue
        if not record.file_exists:
            problems.append(symbol + ": referenced evaluated photon HDF5 missing")
            continue
        try:
            path = Path(record.resolved_path)
            photon = IncidentPhoton.from_hdf5(str(path))
            if getattr(photon, "name", None) != symbol:
                raise ValueError("photon HDF5 element identity mismatch")
            if getattr(photon, "atomic_number", None) != ATOMIC_NUMBER[symbol]:
                raise ValueError("photon HDF5 atomic number mismatch")
            reactions = photon.reactions
            if not all(mt in reactions for mt in (502, 504, 522)):
                raise ValueError("required coherent, incoherent or photoelectric photon reactions absent")
            if getattr(photon, "atomic_relaxation", None) is None:
                raise ValueError("photon atomic relaxation information absent")
            digest = _sha256(path)
        except (OSError, ValueError, KeyError, TypeError, AssertionError, IndexError, RuntimeError) as exc:
            problems.append(symbol + ": rejected actual photon HDF5: " + str(exc))
            continue
        checked.append(symbol)
        hashes.append((symbol, digest))
    return tuple(sorted(checked)), tuple(sorted(problems)), tuple(sorted(hashes))


def inspect_nuclear_data(
    cross_sections: str | Path | None = None,
    config: NervaConfig | None = None,
    photon_transport: bool = False,
    require_referenced_files: bool = True,
) -> NuclearDataReport:
    """Inspect the OpenMC library against the declared NERVA materials.

    Coupled transport requires every required photoatomic ELEMENT in the
    index and genuinely readable HDF5 content with matching identity and
    essential interactions, not just one arbitrary 'photon' entry. This
    compatibility check is NOT hot-core data-release qualification, reactor
    design, or criticality tuning.
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
    needed_photons = required_photon_elements(config) if photon_transport else ()
    missing_photons = tuple(sorted(set(needed_photons) - photon_materials))
    # Even when a caller requests an XML-only dry run, that is insufficient
    # for authorizing a coupled transport. Require real photon HDF5 bytes.
    if photon_transport:
        checked_photons, photon_errors, photon_hashes = _verified_photon_binaries(
            entries, needed_photons)
    else:
        checked_photons, photon_errors, photon_hashes = (), (), ()
    ready = (
        not missing
        and not missing_thermal
        and (not require_referenced_files or not missing_files)
        and (not photon_transport or (
            photon_present and not missing_photons and not photon_errors
            and tuple(sorted(checked_photons)) == needed_photons))
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
        required_photon_elements=needed_photons,
        missing_photon_elements=missing_photons,
        verified_photon_elements=checked_photons,
        photon_binary_audit_errors=photon_errors,
        photon_binary_sha256=photon_hashes,
        photon_evaluated_release_verified=False,
        referenced_files_missing=missing_files,
        ready=ready,
    )


def write_report(report: NuclearDataReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
