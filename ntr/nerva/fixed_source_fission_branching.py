"""Genuine externally driven fission-branching neutron-fate closure (cold core).

OpenMC fixed-source mode with create_fission_neutrons=True transports neutron
secondaries from fission and neutron-producing scattering, starting with a
prescribed, identical external source each batch. This is a PHYSICAL neutron
transport simulation on the unchanged cold conceptual core, not an eigenvalue
simulation, historical NERVA benchmark, criticality claim, or engine design.

In the whole volume, each externally injected neutron contributes one input.
For an analog history tally:
  absorption + outward vacuum leakage
  - (nu-scatter - scatter) - nu-fission = 1

The per-source nu-fission term is a neutron *source* term that is NOT applied
to the nonfission control calculation. Read actual statepoint HDF5, and
independently verify all three signed exterior vacuum current surfaces.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .fixed_source_core_physics import (
    _external_boundary_signs,
    _label,
    _tally,
    read_fixed_source_balance,
    sha,
)


def read_branching_fixed_source_balance(statepoint: Path) -> dict:
    path = Path(statepoint)
    try:
        root = ET.parse(path.parent / "settings.xml").getroot()
        if (root.tag != "settings"
                or root.findtext("run_mode") != "fixed source"
                or root.findtext("create_fission_neutrons") != "true"
                or root.findtext("inactive", default="0") != "0"):
            raise ValueError("physical branching fixed-source transport must be explicitly enabled")
        external_signs = _external_boundary_signs(path.parent)
        with h5py.File(path, "r") as h:
            if (_label(h.attrs["filetype"]) != "statepoint"
                    or _label(h["run_mode"][()]) != "fixed source"
                    or int(h.attrs["photon_transport"]) != 0
                    or int(h.attrs.get("tallies_present", 0)) != 1):
                raise ValueError("genuine neutron-only fixed-source statepoint required")
            seed = int(h["seed"][()])
            batch = int(h["current_batch"][()])
            active = int(h["n_realizations"][()])
            pop = int(h["n_particles"][()])
            if (seed <= 0 or batch < 2 or active != batch or pop < 100
                    or ("n_inactive" in h and int(h["n_inactive"][()]) != 0)):
                raise ValueError("invalid genuine externally driven neutron batch provenance")
            tally, labels, means = _tally(
                h, "nerva_global_neutron_balance", "analog", active
            )
            required = ("absorption", "scatter", "nu-scatter", "nu-fission")
            if (int(tally["n_filters"][()]) != 0 or means.shape[0] != 1
                    or any(score not in labels for score in required)):
                raise ValueError("incomplete physical whole-core neutron-production tally")
            scores = {
                score: float(means[0, labels.index(score)]) for score in required
            }
            surface, surface_labels, surface_mean = _tally(
                h, "nerva_vacuum_boundary_current", "analog", active
            )
            if (surface_labels != ["current"]
                    or int(surface["n_filters"][()]) != 1
                    or surface_mean.shape != (3, 1)):
                raise ValueError("native analog outward neutron surface current missing")
            ids = np.asarray(surface["filters"][()], dtype=int).reshape(-1)
            if len(ids) != 1:
                raise ValueError("unexpected neutron surface filter count")
            f = h["tallies"]["filters"]["filter " + str(int(ids[0]))]
            if _label(f["type"][()]) != "surface":
                raise ValueError("outward current must be sampled on native surfaces")
            sids = np.asarray(f["bins"][()], dtype=int).reshape(-1)
            if len(sids) != 3 or set(sids) != set(external_signs):
                raise ValueError("incomplete physical core neutron vacuum current surfaces")
            per_surface = {
                str(ident): float(external_signs[int(ident)] * value)
                for ident, value in zip(sids, surface_mean[:, 0])
            }
    except (OSError, ET.ParseError, KeyError, TypeError, IndexError) as exc:
        raise ValueError("incomplete genuine fission-branching OpenMC HDF5/XML") from exc
    if any(not math.isfinite(x) or x < -1e-8 for x in per_surface.values()):
        raise ValueError("invalid exterior neutron current orientation or leakage")
    absorption = scores["absorption"]
    leakage = float(sum(per_surface.values()))
    scatter_extra = scores["nu-scatter"] - scores["scatter"]
    fission_daughters = scores["nu-fission"]
    if (absorption <= 0 or leakage <= 0 or scatter_extra < -1e-8
            or fission_daughters <= 0
            or not all(math.isfinite(x) for x in scores.values())):
        raise ValueError("physically missing fission neutrons or invalid neutron fate")
    total = absorption + leakage - scatter_extra - fission_daughters
    # Genuine four-seed analog event accounting closed below 1e-14.
    # Enforce the particle-conservation identity at numerical accuracy,
    # not a loose Monte Carlo statistical comparison.
    if abs(total - 1.0) > 1e-6:
        raise ValueError("fission-branching neutron production and loss do not conserve source")
    return {
        "seed": seed,
        "batches": batch,
        "particles_per_batch": pop,
        "statepoint_sha256": sha(path),
        "evaluated_geometry_sha256": sha(path.parent / "geometry.xml"),
        "evaluated_materials_sha256": sha(path.parent / "materials.xml"),
        "analog_absorption_per_external_source": absorption,
        "outward_vacuum_leakage_per_external_source": leakage,
        "vacuum_outward_current_by_surface": per_surface,
        "net_multiplicative_scattering_per_external_source": scatter_extra,
        "real_fission_daughter_neutrons_per_external_source": fission_daughters,
        "external_source_neutron_balance": total,
        "absolute_balance_residual": abs(total - 1.0),
        "fixed_external_source_fission_branching_executed": True,
        "self_sustaining_reactor_criticality_validated": False,
        "NASA_NERVA_core_validated": False,
    }


def compare_four_seed_external_source_modes(
    branching_directory: Path, nonbranching_directory: Path,
    seeds: list[int], batches: int,
) -> dict:
    if (len(seeds) != 4 or len(set(seeds)) != 4
            or any(type(x) is not int or x <= 0 for x in seeds)):
        raise ValueError("four distinct externally driven neutron seeds required")
    records = []
    for seed in seeds:
        b = read_branching_fixed_source_balance(
            Path(branching_directory) / f"seed_{seed}" / f"statepoint.{batches}.h5"
        )
        n = read_fixed_source_balance(
            Path(nonbranching_directory) / f"seed_{seed}" / f"statepoint.{batches}.h5"
        )
        if (b["seed"] != n["seed"] or b["seed"] != seed
                or b["batches"] != n["batches"] or b["batches"] != batches
                or b["particles_per_batch"] != n["particles_per_batch"]
                or b["evaluated_geometry_sha256"] != n["evaluated_geometry_sha256"]
                or b["evaluated_materials_sha256"] != n["evaluated_materials_sha256"]
                or b["statepoint_sha256"] == n["statepoint_sha256"]):
            raise ValueError("genuine neutron branching comparison changed physical input or seed")
        records.append({"seed": seed, "fission_daughters_enabled": b,
                        "fission_daughters_disabled": n})
    names = (
        ("analog_absorption_per_external_source",
         "analog_absorption_per_injected_neutron"),
        ("outward_vacuum_leakage_per_external_source",
         "outward_vacuum_leakage_per_injected_neutron"),
        ("net_multiplicative_scattering_per_external_source",
         "net_multiplicative_scattering_per_injected_neutron"),
    )
    changes = {}
    for on_key, off_key in names:
        pair = np.asarray([
            (x["fission_daughters_enabled"][on_key],
             x["fission_daughters_disabled"][off_key]) for x in records
        ], dtype=float)
        delta = pair[:, 0] - pair[:, 1]
        changes[on_key] = {
            "branching_mean": float(pair[:, 0].mean()),
            "no_fission_daughter_control_mean": float(pair[:, 1].mean()),
            "paired_seed_difference_mean": float(delta.mean()),
            "paired_seed_difference_standard_error": float(
                delta.std(ddof=1) / math.sqrt(len(delta))
            ),
        }
    daughters = np.asarray([
        x["fission_daughters_enabled"]["real_fission_daughter_neutrons_per_external_source"]
        for x in records
    ], dtype=float)
    if len({x["fission_daughters_enabled"]["statepoint_sha256"] for x in records}) != 4:
        raise ValueError("duplicate independent fission-branching OpenMC neutron histories")
    return {
        "source": "actual OpenMC neutron C++ solver, prescribed source and evaluated neutron XS",
        "independent_seed_count": len(seeds),
        "seed_modes": records,
        "mean_fission_daughter_neutrons_per_external_source": float(daughters.mean()),
        "fission_daughter_neutron_rate_between_seed_sem": float(
            daughters.std(ddof=1) / math.sqrt(len(daughters))
        ),
        "source_driven_reaction_rate_changes": changes,
        "four_seed_analog_whole_core_neutron_conservation_verified": True,
        "comparison_reactor_geometry_or_fuel_not_changed": True,
        "fission_source_eigenvalue_converged": False,
        "reactor_criticality_or_power_normalization_qualified": False,
        "historical_NERVA_model_verified": False,
        "interpretation": (
            "Actual fixed external-source neutron-transport branching "
            "includes fission daughters and additional scattering-born "
            "neutrons, with signed outward vacuum leakage. Paired RNG seeds "
            "share declared source/model parameters but are not identical "
            "random particle trajectories. The control suppresses fission "
            "daughters. This experiment cannot determine an eigenvalue, "
            "demonstrate a critical reactor, or validate historical NERVA."
        ),
    }
