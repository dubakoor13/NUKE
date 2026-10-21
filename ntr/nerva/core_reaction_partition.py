"""Independent material/energy reaction partition QA on REAL OpenMC eigenvalue statepoints.

A correctly constructed neutron-only material tally must partition the
unfiltered whole-geometry absorption and nu-fission TRACKLENGTH rates.
The separate multiplicative-scattering balance uses OpenMC's ANALOG
estimator and must not be equated to tracklength tallies per history.
EnergyFilter reaction rates must also account for the same material totals
within its explicitly bounded incident-neutron energy range.

This is an independently filtered estimator consistency test; it does not
prove historical NASA materials, criticality, source stationarity or hot
reactor operation. All rates are per initial eigenvalue-generation source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import openmc


SCORES = ("absorption", "nu-fission")
FUEL_MATERIAL_NAME = "conceptual (U,Zr)C-graphite surrogate"


def _file_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def _material_names(materials_xml: Path) -> dict[int, str]:
    """Read exported OpenMC materials.xml as physical material provenance."""
    root = ET.parse(materials_xml).getroot()
    if root.tag != "materials":
        raise ValueError("expected genuine OpenMC materials XML")
    entries = root.findall("material")
    result: dict[int, str] = {}
    for entry in entries:
        ident = int(entry.attrib["id"])
        if ident in result or ident < 1:
            raise ValueError("duplicate or invalid OpenMC material ID")
        name = entry.attrib.get("name", "")
        if not name:
            raise ValueError("unnamed OpenMC physical material")
        result[ident] = name
    if not result or FUEL_MATERIAL_NAME not in result.values():
        raise ValueError("conceptual fuel material missing from materials.xml")
    return result


def assess_material_partition(
    material_ids: list[int],
    names: dict[int, str],
    global_rates: dict[str, float],
    material_rates: dict[str, np.ndarray],
    energy_rates: dict[str, np.ndarray],
    energy_bins_ev: np.ndarray,
    *,
    maximum_partition_absolute_mismatch: float = 1e-6,
    maximum_energy_unresolved_fraction: float = 0.02,
    maximum_nonfuel_fission_fraction: float = 0.01,
) -> dict:
    """Apply explicit whole-geometry and incident-energy reaction identities.

    Parameter tolerances are *QA screening choices*, NOT measurement
    uncertainties; the same collision histories underlie the tallies.
    """
    for key, tol in (
        ("maximum_partition_absolute_mismatch", maximum_partition_absolute_mismatch),
        ("maximum_energy_unresolved_fraction", maximum_energy_unresolved_fraction),
        ("maximum_nonfuel_fission_fraction", maximum_nonfuel_fission_fraction),
    ):
        if (not math.isfinite(tol) or tol <= 0 or tol >= 1):
            raise ValueError(f"{key} must be finite within (0,1)")
    if (not material_ids or len(set(material_ids)) != len(material_ids)
            or any(not isinstance(n, (int, np.integer)) or n < 1
                   for n in material_ids)):
        raise ValueError("invalid or duplicate MaterialFilter bins")
    if set(material_ids) != set(names):
        raise ValueError("material tally does not include every exported physical material")
    fuel_ids = [k for k in material_ids if names[k] == FUEL_MATERIAL_NAME]
    if len(fuel_ids) != 1:
        raise ValueError("fuel material must have one unique material ID")
    energy = np.asarray(energy_bins_ev, dtype=float)
    if (energy.ndim != 2 or energy.shape[1] != 2 or energy.shape[0] < 3
            or not np.isfinite(energy).all()
            or np.any(energy[:, 1] <= energy[:, 0])
            or np.any(energy[:, 0] < 0)
            or np.any(np.abs(energy[1:, 0] - energy[:-1, 1]) >
                      1e-9 * np.maximum(1.0, energy[1:, 0]))):
        raise ValueError("energy reaction bins must be finite contiguous intervals")
    nmaterials = len(material_ids)
    nbins = energy.shape[0]
    detail = {}
    every_partition_passes = True
    every_energy_coverage_passes = True
    for score in SCORES:
        if score not in global_rates or score not in material_rates or score not in energy_rates:
            raise ValueError(f"required core reaction score missing: {score}")
        global_mean = float(global_rates[score])
        mat = np.asarray(material_rates[score], dtype=float)
        filtered = np.asarray(energy_rates[score], dtype=float)
        if (mat.shape != (nmaterials,) or filtered.shape != (nmaterials, nbins)
                or not math.isfinite(global_mean) or global_mean < 0
                or not np.isfinite(mat).all() or np.any(mat < 0)
                or not np.isfinite(filtered).all() or np.any(filtered < 0)):
            raise ValueError(f"invalid or negative {score} reaction bins")
        sum_material = float(np.sum(mat, dtype=np.float64))
        sum_energy = float(np.sum(filtered, dtype=np.float64))
        delta = sum_material - global_mean
        energy_missing = sum_material - sum_energy
        partition_pass = abs(delta) <= maximum_partition_absolute_mismatch
        energy_pass = abs(energy_missing) <= max(
            maximum_partition_absolute_mismatch,
            maximum_energy_unresolved_fraction * max(sum_material, 1e-12),
        )
        every_partition_passes &= partition_pass
        every_energy_coverage_passes &= energy_pass
        detail[score] = {
            "global_reactions_per_source": global_mean,
            "summed_material_reactions_per_source": sum_material,
            "summed_energy_filtered_reactions_per_source": sum_energy,
            "material_minus_global": delta,
            "energy_window_unaccounted_reactions_per_source": energy_missing,
            "unaccounted_energy_fraction_of_material": (
                abs(energy_missing) / sum_material if sum_material else 0.0
            ),
            "material_partition_screen_passed": bool(partition_pass),
            "incident_energy_coverage_screen_passed": bool(energy_pass),
            "by_material": [
                {
                    "material_id": int(ident),
                    "material_name": names[ident],
                    "reaction_rate_per_source": float(mat[idx]),
                    "energy_binned_reaction_rate_per_source":
                        float(np.sum(filtered[idx], dtype=np.float64)),
                }
                for idx, ident in enumerate(material_ids)
            ],
        }
    fission = np.asarray(material_rates["nu-fission"], dtype=float)
    total = float(np.sum(fission, dtype=np.float64))
    fuel_prod = float(fission[material_ids.index(fuel_ids[0])])
    nonfuel_fraction = (total - fuel_prod) / total if total > 0 else None
    fission_screen = (nonfuel_fraction is not None
                      and nonfuel_fraction <= maximum_nonfuel_fission_fraction)
    return {
        "reaction_basis": "OpenMC eigenvalue per-source neutron reactions",
        "energy_window_eV": [float(energy[0, 0]), float(energy[-1, 1])],
        "incident_energy_groups": nbins,
        "number_of_materials": nmaterials,
        "partition_absolute_tolerance_per_source":
            maximum_partition_absolute_mismatch,
        "maximum_unaccounted_energy_fraction":
            maximum_energy_unresolved_fraction,
        "maximum_nonfuel_fission_fraction": maximum_nonfuel_fission_fraction,
        "scores": detail,
        "material_vs_global_screen_passed": bool(every_partition_passes),
        "material_vs_energy_screen_passed": bool(every_energy_coverage_passes),
        "nonfuel_nu_fission_fraction": nonfuel_fraction,
        "fuel_only_fission_origin_screen_passed": bool(fission_screen),
        "all_partition_screens_passed": bool(
            every_partition_passes and every_energy_coverage_passes and fission_screen
        ),
        "nasa_nerva_physical_core_validated": False,
        "source_stationarity_proven": False,
        "interpretation": (
            "Independently filtered reaction tallies agree only if the model "
            "materials and scored neutron-energy window account for the global "
            "reaction estimator. Differences are numerical or missing-coverage "
            "flags, not experimental benchmark error. The fuel-only fission "
            "screen is particular to this explicitly declared surrogate fuel "
            "composition and is not a criticality or historical-material claim."
        ),
    }


def _extract_tally(tally: openmc.Tally, score: str, shape: tuple[int, ...]) -> np.ndarray:
    if score not in tally.scores:
        raise ValueError(f"OpenMC required {score} tally score absent")
    values = np.asarray(tally.get_values(scores=[score], value="mean"), dtype=float)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"OpenMC {score} tally has unexpected filter dimensions")
    return values.reshape(shape)


def inspect_real_statepoint(statepoint: Path, materials_xml: Path) -> dict:
    statepoint = Path(statepoint)
    materials_xml = Path(materials_xml)
    if not statepoint.is_file() or not materials_xml.is_file():
        raise FileNotFoundError("real statepoint or materials.xml missing")
    names = _material_names(materials_xml)
    with openmc.StatePoint(statepoint, autolink=False) as sp:
        if sp.run_mode != "eigenvalue" or sp.photon_transport:
            raise ValueError("reaction partition requires neutron-only eigenvalue transport")
        if not sp.tallies_present or sp.n_realizations < 2:
            raise ValueError("no real active OpenMC neutron reaction realizations")
        global_tally = sp.get_tally(name="nerva_global_neutron_balance")
        global_tracklength = sp.get_tally(name="nerva_global_reaction_tracklength")
        material = sp.get_tally(name="nerva_material_transport")
        energy = sp.get_tally(name="nerva_material_energy_transport")
        if global_tally.filters or global_tracklength.filters:
            raise ValueError("whole-core reaction tallies must not have clipped filters")
        if global_tally.estimator != "analog":
            raise ValueError("multiplicative-scatter neutron balance must use analog estimator")
        if any(tally.estimator != "tracklength" for tally in
               (global_tracklength, material, energy)):
            raise ValueError("material and global reference reactions must use matching tracklength estimators")
        if len(material.filters) != 1 or not isinstance(
                material.filters[0], openmc.MaterialFilter):
            raise ValueError("whole-core material reactions must use only MaterialFilter")
        if (len(energy.filters) != 2
                or not isinstance(energy.filters[0], openmc.MaterialFilter)
                or not isinstance(energy.filters[1], openmc.EnergyFilter)):
            raise ValueError("energy reactions must use matching MaterialFilter and EnergyFilter")
        ids = list(map(int, material.filters[0].bins))
        if ids != list(map(int, energy.filters[0].bins)):
            raise ValueError("material and energy tally bins do not match")
        bins = np.asarray(energy.filters[1].bins, dtype=float)
        global_rates = {
            s: float(_extract_tally(global_tracklength, s, (1,))[0])
            for s in SCORES
        }
        analog_rates = {
            s: float(_extract_tally(global_tally, s, (1,))[0]) for s in SCORES
        }
        material_rates = {
            s: _extract_tally(material, s, (len(ids),)) for s in SCORES
        }
        energy_rates = {
            s: _extract_tally(energy, s, (len(ids), len(bins))) for s in SCORES
        }
        result = assess_material_partition(
            ids, names, global_rates, material_rates, energy_rates, bins
        )
        result.update({
            "estimator_consistency": {
                "partition_reference_estimator": "tracklength",
                "material_estimator": "tracklength",
                "material_energy_estimator": "tracklength",
                "separate_conservation_estimator": "analog",
                "analog_minus_tracklength_global": {
                    score: analog_rates[score] - global_rates[score]
                    for score in SCORES
                },
                "interpretation": (
                    "Analog neutron multiplicity accounting and tracklength "
                    "reaction-rate estimates are distinct statistical estimators. "
                    "Their nonzero sample-mean differences are not missing "
                    "materials. Partition identities require matching estimators."
                ),
            },
            "source": "GENUINE_OPENMC_EIGENVALUE_STATEPOINT",
            "statepoint_sha256": _file_sha256(statepoint),
            "materials_xml_sha256": _file_sha256(materials_xml),
            "completed_batch": int(sp.current_batch),
            "active_reaction_realizations": int(sp.n_realizations),
        })
        return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statepoint", required=True, type=Path)
    p.add_argument("--materials-xml", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--strict", action="store_true",
                   help="return nonzero when any physical reaction-partition screen fails")
    a = p.parse_args()
    result = inspect_real_statepoint(a.statepoint, a.materials_xml)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print("OpenMC global vs material reactions:",
          result["material_vs_global_screen_passed"])
    print("OpenMC material vs incident-energy reactions:",
          result["material_vs_energy_screen_passed"])
    print("OpenMC declared fuel origin of nu-fission:",
          result["fuel_only_fission_origin_screen_passed"])
    print("NASA physical reactor benchmark validated: NO")
    if a.strict and not result["all_partition_screens_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
