"""Read-only MT=901 local-neutron-KERMA audit of evaluated OpenMC nuclear data.

Neutron-only 'heating-local' requires evaluated local energy-deposition
coefficients, typically represented as reaction MT=901 in the nuclide HDF5.
Geometry validity, fissile loading, core criticality and NASA benchmarking
are separate. This audit NEVER manufactures KERMA coefficients, changes
material definitions, substitutes MT=301 for 901 or modifies core design.

See https://docs.openmc.org/en/stable/methods/energy_deposition.html
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import openmc.data

from .audit_nuclear_data_temperatures import _requested_temperatures
from .config import NervaConfig
from .nuclear_data import _library_entries, resolve_cross_sections


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def assess_mt901_inventory(rows: list[dict]) -> dict:
    """Require positively observed MT=901 for every required nuclide.

    This is a necessary *coverage screen* for one local-heating method,
    not validation of KERMA numeric values, evaluated-data provenance,
    reactor temperatures, deposited-power shapes or actual OpenMC tallies.
    """
    if not rows:
        raise ValueError("empty neutron-data nuclide inventory")
    names = [x.get("nuclide") for x in rows]
    if any(not isinstance(n, str) or not n for n in names):
        raise ValueError("nuclide identity missing in nuclear-data audit")
    if len(set(names)) != len(names):
        raise ValueError("duplicate neutron nuclide in MT=901 inventory")
    invalid = []
    for row in rows:
        present = row.get("local_kerma_mt901_present")
        if present is not None and not isinstance(present, bool):
            raise ValueError("local KERMA reaction presence must be boolean or null")
        if present is not True:
            invalid.append(row["nuclide"])
    ready = len(invalid) == 0
    return {
        "number_of_required_neutron_nuclides": len(rows),
        "nuclides_missing_or_unverified_mt901": invalid,
        "local_heating_evaluated_kerma_inventory_complete": ready,
        "neutron_only_heating_local_can_be_claimed_from_inventory": ready,
        "interpretation": (
            "MT=901 presence in EVERY required nuclide is a conservative "
            "data-coverage preflight, not a physical reactor acceptance test. "
            "Actual neutron-local heating tallies must also be nonzero and "
            "numerically verified. MT=301 ordinary heating, kappa-fission and "
            "other scores cannot silently replace missing MT=901. "
            "Identical neutron transport and reaction rates can coexist "
            "with unavailable physical deposited-energy data."
        ),
        "nasa_nerva_core_heating_benchmarked": False,
    }


def audit_kerma_coverage(cross_sections: Path, *,
                         config: NervaConfig | None = None) -> dict:
    cfg = config or NervaConfig(core_rings=3)
    cfg.validate()
    xml = resolve_cross_sections(cross_sections)
    source = {}
    for entry in _library_entries(xml):
        if entry.data_type == "neutron":
            for name in entry.materials:
                if name in source and source[name] != Path(entry.resolved_path):
                    raise ValueError("ambiguous evaluated cross-section file for " + name)
                source[name] = Path(entry.resolved_path)
    requested = _requested_temperatures(cfg)
    records = []
    for nuclide, demands in sorted(requested.items()):
        p = source.get(nuclide)
        row = {
            "nuclide": nuclide,
            "requested_by_material": demands,
            "physical_neutron_hdf5_path": str(p) if p else None,
            "physical_neutron_hdf5_sha256": None,
            "local_kerma_mt901_present": None,
            "ordinary_heating_mt301_present": None,
            "fission_heating_mt318_present": None,
            "library_load_error": None,
        }
        if p is None or not p.is_file():
            row["library_load_error"] = "required physical evaluated neutron HDF5 missing"
        else:
            try:
                data = openmc.data.IncidentNeutron.from_hdf5(p)
                if data.name != nuclide:
                    raise ValueError("evaluated HDF5 nuclide identity mismatch")
                row["physical_neutron_hdf5_sha256"] = _sha256(p)
                row["local_kerma_mt901_present"] = 901 in data.reactions
                row["ordinary_heating_mt301_present"] = 301 in data.reactions
                row["fission_heating_mt318_present"] = 318 in data.reactions
            except Exception as exc:
                row["library_load_error"] = type(exc).__name__ + ": " + str(exc)
        records.append(row)
    overview = assess_mt901_inventory(records)
    return {
        "mode": "ACTUAL_EVALUATED_NEUTRON_CROSS_SECTION_HDF5_INSPECTION",
        "cross_sections_xml": str(xml),
        "cross_sections_xml_sha256": _sha256(xml),
        "requested_model_assembly": "cold conceptual NERVA-derived core",
        "nuclides": records,
        **overview,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cross-sections", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-mt901", action="store_true")
    args = p.parse_args()
    report = audit_kerma_coverage(args.cross_sections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Actual evaluated neutron MT=901 local KERMA coverage:",
          report["local_heating_evaluated_kerma_inventory_complete"])
    for row in report["nuclides"]:
        print(row["nuclide"], "MT301", row["ordinary_heating_mt301_present"],
              "MT901", row["local_kerma_mt901_present"],
              "load_error", row["library_load_error"])
    print("NASA NERVA physical core heating validated: NO")
    if args.require_mt901 and not report["local_heating_evaluated_kerma_inventory_complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
