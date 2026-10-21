"""Audit *physical* evaluated bound thermal-scattering HDF5 kernels.

These are actual S(alpha,beta,T) nuclear-data evaluations, not free-atom
approximations. A cross_sections.xml declaration alone is not sufficient.
Fail closed if a required kernel is missing, malformed, or outside its
tabulated temperature range. Never extrapolate hot-core moderator physics.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import math

import openmc.data

from .config import NervaConfig
from .materials import required_bound_scattering_tables
from .nuclear_data import _library_entries, resolve_cross_sections


def audit_bound_thermal_scattering(
    cross_sections: str | Path,
    config: NervaConfig,
) -> dict:
    config.validate()
    requested = required_bound_scattering_tables(config)
    if not requested:
        return {
            "treatment": "free-gas",
            "ready": False,
            "physically_complete_bound_thermal_scattering": False,
            "required_tables": [],
            "reason": "Bound-atom S(alpha,beta) is not enabled; surrogate approximation",
            "tables": [],
        }

    index = resolve_cross_sections(cross_sections)
    entries = _library_entries(index)
    table_index = {}
    for entry in entries:
        if entry.data_type == "thermal":
            for name in entry.materials:
                if name in table_index:
                    raise ValueError(
                        "ambiguous duplicate evaluated bound thermal table: "+name
                    )
                table_index[name] = entry

    rows = []
    temp = config.bound_solid_temperature_k
    for name in requested:
        entry = table_index.get(name)
        row = {
            "table": name,
            "path": None if entry is None else entry.resolved_path,
            "requested_temperature_K": temp,
            "evaluated_temperatures_K": [],
            "sha256": None,
            "accepted": False,
            "error": None,
        }
        if entry is None:
            row["error"] = "kernel absent from evaluated cross_sections.xml"
        elif not entry.file_exists:
            row["error"] = "evaluated HDF5 kernel is missing"
        else:
            path = Path(entry.resolved_path)
            try:
                thermal = openmc.data.ThermalScattering.from_hdf5(path)
                if thermal.name != name:
                    raise ValueError(
                        f"evaluated kernel HDF5 name {thermal.name!r} "
                        f"does not match {name!r} in cross_sections.xml"
                    )
                # OpenMC kTs maps temperature identifiers to kT [eV].
                kts = thermal.kTs
                vals = kts.values() if isinstance(kts, dict) else kts
                temperatures = sorted(
                    float(v) / openmc.data.K_BOLTZMANN for v in vals
                )
                if (not temperatures or
                        not all(math.isfinite(t) and t > 0 for t in temperatures)):
                    raise ValueError("thermal kernel has no valid evaluated temperatures")
                row["evaluated_temperatures_K"] = temperatures
                if not (temperatures[0] - 1e-7 <= temp <= temperatures[-1] + 1e-7):
                    raise ValueError(
                        "requested temperature outside evaluated thermal-kernel range"
                    )
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for data in iter(lambda: stream.read(1024*1024), b""):
                        digest.update(data)
                row["sha256"] = digest.hexdigest()
                row["accepted"] = True
            except (OSError, KeyError, ValueError, TypeError, AttributeError) as error:
                row["error"] = str(error)
        rows.append(row)

    ready = all(row["accepted"] for row in rows)
    return {
        "treatment": "evaluated-bound",
        "ready": ready,
        "physically_complete_bound_thermal_scattering": ready,
        "requested_solid_temperature_K": temp,
        "required_tables": list(requested),
        "tables": rows,
        "interpretation": (
            "Kernel HDF5 loading, SHA-256, and tabulated temperature coverage "
            "are verified. This does not validate graphite phase fractions "
            "within the heterogeneous fuel, source convergence, full operating "
            "temperature fields, criticality or any historical reactor benchmark."
        ),
    }
