"""Read-only independent neutron HDF5 temperature/energy-grid integrity check.

Inspect the *executed and root-reachable* OpenMC neutron materials, not all
surrogate material declarations. Load authentic evaluated IncidentNeutron
HDF5 tables. Each temperature's native incident energy grid must be finite,
strictly increasing and nonnegative, with overlapping tabulated domains.
The check neither synthesizes hot microscopic cross sections nor modifies
temperature, material composition, source, geometry or criticality.

Passing a grid-integrity screen alone does NOT qualify hot NERVA transport:
bound thermal-scattering physics, phase constitution, local KERMA, reactor
thermal feedback and physical benchmarking require independent evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import openmc.data

from .core_material_temperature_qualification import (
    audit_actual_core_neutron_temperature,
)
from .nuclear_data import _library_entries


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assess_evaluated_neutron_energy_grids(
    isotope: str, temperatures_K: list[float],
    energies_by_temperature: dict[str, object],
    labels: list[str],
) -> dict:
    """Numerical check only: temperatures ↔ evaluated energy grids."""
    if (not isotope or not isinstance(isotope, str)
            or not temperatures_K or len(temperatures_K) != len(labels)
            or len(set(labels)) != len(labels)
            or not all(isinstance(x, str) and x for x in labels)
            or not isinstance(energies_by_temperature, dict)):
        raise ValueError("invalid evaluated nuclide temperature energy-grid metadata")
    arr = np.asarray(temperatures_K, dtype=float)
    if (np.any(~np.isfinite(arr)) or np.any(arr <= 0)
            or np.any(np.diff(np.sort(arr)) <= 0)):
        raise ValueError("nonphysical or duplicate native evaluated temperatures")
    if any(label not in energies_by_temperature for label in labels):
        raise ValueError("missing evaluated neutron incident-energy grid for native temperature")
    rows = []
    for temp, label in zip(temperatures_K, labels):
        grid = np.asarray(energies_by_temperature[label], dtype=np.float64)
        if (grid.ndim != 1 or len(grid) < 2
                or not np.all(np.isfinite(grid))
                or grid[0] < 0. or grid[-1] <= 0.
                or np.any(np.diff(grid) <= 0.)):
            raise ValueError("malformed/unsorted evaluated neutron incident-energy grid: "
                             + isotope + " " + label)
        rows.append({
            "native_temperature_label":label,
            "evaluated_temperature_K":float(temp),
            "incident_neutron_grid_points":int(grid.size),
            "energy_min_eV":float(grid[0]),
            "energy_max_eV":float(grid[-1]),
            "strictly_increasing_finite_energy_grid":True,
        })
    low = max(x["energy_min_eV"] for x in rows)
    high = min(x["energy_max_eV"] for x in rows)
    if high <= low:
        raise ValueError("evaluated neutron temperature energy grids have no common domain")
    return {
        "nuclide":isotope,
        "evaluated_temperature_node_count":len(rows),
        "evaluated_temperature_min_K":float(arr.min()),
        "evaluated_temperature_max_K":float(arr.max()),
        "all_native_temperature_energy_grids_valid":True,
        "common_incident_energy_min_eV":float(low),
        "common_incident_energy_max_eV":float(high),
        "evaluated_multitemperature_library_for_this_nuclide":len(rows) >= 2,
        "temperatures":rows,
    }


def audit_executed_neutron_energy_grids(
    summary_path: Path, cross_sections: Path,
) -> dict:
    """Require actual four-seed XML/material/data identity before scanning."""
    summary_path = Path(summary_path)
    index = Path(cross_sections)
    prior = audit_actual_core_neutron_temperature(summary_path, index)
    if (prior["all_real_source_model_provenance_consistent"] is not True
            or prior["actual_geometry_reachability_and_evaluated_xs_reconciled"] is not True
            or prior["actual_openmc_seed_count"] < 4
            or prior["cross_sections_xml_sha256"] != _sha(index)):
        raise ValueError("unqualified executed OpenMC neutron transport provenance")

    physical = prior["physical_materials"]
    required = {nuclide for material in physical for nuclide in material["nuclides"]}
    if not required:
        raise ValueError("no executed physically reachable neutron nuclides")
    indexed = {}
    for entry in _library_entries(index):
        if entry.data_type != "neutron":
            continue
        for isotope in entry.materials:
            if isotope in required:
                if isotope in indexed:
                    raise ValueError("duplicate physically used evaluated neutron isotope")
                indexed[isotope] = Path(entry.resolved_path)
    if set(indexed) != required:
        raise ValueError("executed neutron nuclides missing from evaluated HDF5 inventory")

    evidence = {}
    verified_file_digests = {}
    for isotope in sorted(required):
        path = indexed[isotope]
        if not path.is_file():
            raise ValueError("missing physically executed neutron evaluation: " + isotope)
        data = openmc.data.IncidentNeutron.from_hdf5(path)
        if data.name != isotope:
            raise ValueError("native evaluated neutron HDF5 isotope identity mismatch")
        kelvin = [float(kT / openmc.data.K_BOLTZMANN) for kT in data.kTs]
        result = assess_evaluated_neutron_energy_grids(
            isotope, kelvin, data.energy, list(data.temperatures),
        )
        digest = _sha(path)
        verified_file_digests[str(path.resolve())] = digest
        result["evaluated_hdf5_sha256"] = digest
        result["evaluated_hdf5_file"] = str(path.resolve())
        evidence[isotope] = result

    material_rows = []
    for material in physical:
        names = list(material["nuclides"])
        all_multi = all(
            evidence[n]["evaluated_multitemperature_library_for_this_nuclide"]
            for n in names
        )
        common_lo = max(evidence[n]["evaluated_temperature_min_K"] for n in names)
        common_hi = min(evidence[n]["evaluated_temperature_max_K"] for n in names)
        material_rows.append({
            "material_id":material["material_id"],
            "material_name":material["material_name"],
            "physically_root_reachable":True,
            "explicit_executed_material_temperature_K":
                material["explicit_material_temperature_K"],
            "evaluated_neutron_nuclides":names,
            "all_nuclides_have_multiple_tabulated_temperatures":bool(all_multi),
            "common_evaluated_temperature_interval_K":(
                [common_lo, common_hi] if common_hi >= common_lo else None
            ),
            "executed_material_temperature_explicit":(
                material["explicit_material_temperature_K"] is not None
            ),
            "bound_scattering_phase_assignment_qualified":
                material.get("bound_phase_table_assignment_matches") is True,
        })
    # This does not make the original reactor hot, irrespective of the
    # candidate evaluated data's future temperature-grid availability.
    all_multitemperature = all(
        evidence[name]["evaluated_multitemperature_library_for_this_nuclide"]
        for name in required
    )
    return {
        "audit": "ACTUAL_OPENMC_EXECUTED_NUCLEAR_DATA_TEMPERATURE_ENERGY_GRID_INTEGRITY",
        "executed_transport_summary_sha256":_sha(summary_path),
        "evaluated_neutron_cross_sections_xml_sha256":_sha(index),
        "four_seed_provenance_verified":True,
        "executed_reachable_material_ids":
            prior["actual_reachable_material_ids"],
        "executed_reachable_neutron_nuclide_count":len(required),
        "all_actual_neutron_evaluations_have_valid_energy_grids":True,
        "all_actual_neutron_nuclides_have_multiple_temperature_tables":bool(
            all_multitemperature
        ),
        "all_executed_material_temperatures_explicit":
            prior["all_material_temperatures_explicit"],
        "hot_reactor_transport_physically_qualified":
            prior["hot_operating_temperature_neutron_physics_qualified"],
        "nuclear_data_sha256_manifest":verified_file_digests,
        "physically_instantiated_material_temperature_evidence":material_rows,
        "incident_neutron_temperature_energy_grid_evidence":evidence,
        "no_extrapolated_or_synthesized_nuclear_data":True,
        "fission_source_convergence_demonstrated":False,
        "reactor_energy_deposition_qualified_by_this_audit":False,
        "nasa_historical_nerva_benchmark_qualified":False,
        "interpretation": (
            "Read-only numerical energy-grid integrity of genuine neutron "
            "evaluation HDF5 for all root-reachable OpenMC materials. "
            "Temperature-node count and cross-material common intersection "
            "are inventory information, NOT proof of operating hot-core "
            "neutron scattering, energy deposition or reactor fidelity."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--cross-sections", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit_executed_neutron_energy_grids(a.summary, a.cross_sections)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n",
                        encoding="utf-8")
    print("Executed evaluated neutron isotope energy grids:",
          result["executed_reachable_neutron_nuclide_count"])
    print("Native HDF5 grid integrity: PASS")
    print("Multitemperature nuclear data for ALL physically used isotopes:",
          result["all_actual_neutron_nuclides_have_multiple_temperature_tables"])
    print("Hot operating NERVA physical qualification:",
          result["hot_reactor_transport_physically_qualified"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
