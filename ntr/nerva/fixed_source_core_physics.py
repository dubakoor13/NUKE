"""Independent physical neutron-balance audit for genuine OpenMC fixed-source transport.

OpenMC's cold, strongly subcritical three-ring demonstration is *not* a
historical NERVA reactor. Fixed-source histories start from a reproducible
prescribed source rather than cloning the fission bank every generation.

With create_fission_neutrons=False, fission is a terminal absorption:
    absorption + outward vacuum leakage - (nu-scatter - scatter) = 1
per injected neutron. Nonfission (n,xn) multiplication remains physical.

This is a source-driven *non-chain-multiplying* transport calculation; it
does not calculate reactor criticality, thermal-power deposition or NERVA
engine operating conditions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _label(v) -> str:
    if not isinstance(v, (bytes, np.bytes_)):
        raise ValueError("invalid genuine OpenMC neutron score label")
    return bytes(v).decode("utf-8").rstrip("\x00")


def _tally(handle: h5py.File, name: str, estimator: str, active: int):
    group = handle["tallies"]
    matches = [
        item for key, item in group.items()
        if key.startswith("tally ") and isinstance(item, h5py.Group)
        and "name" in item and _label(item["name"][()]) == name
    ]
    if len(matches) != 1:
        raise ValueError("absent or duplicate physical OpenMC tally: " + name)
    tally = matches[0]
    if (_label(tally["estimator"][()]) != estimator
            or int(tally["n_realizations"][()]) != active):
        raise ValueError("native fixed-source tally estimator or active batch mismatch")
    labels = [_label(x) for x in tally["score_bins"][()]]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate native neutron tally score")
    moments = np.asarray(tally["results"][()], dtype=float)
    if (moments.ndim != 3 or moments.shape[1:] != (len(labels), 2)
            or not np.isfinite(moments).all()):
        raise ValueError("invalid actual neutron score moments")
    return tally, labels, moments[:, :, 0] / active


def _external_boundary_signs(directory: Path) -> dict[int, int]:
    try:
        root = ET.parse(directory / "geometry.xml").getroot()
        external = [s for s in root.findall("surface")
                    if s.attrib.get("boundary") == "vacuum"]
        if root.tag != "geometry" or len(external) != 3:
            raise ValueError("requires the 3-ring cylindrical vacuum boundary")
        planes = [s for s in external if s.attrib.get("type") == "z-plane"]
        radial = [s for s in external if s.attrib.get("type") == "z-cylinder"]
        if len(planes) != 2 or len(radial) != 1:
            raise ValueError("unsupported fixed-source external vacuum geometry")
        z0, z1 = sorted((float(s.attrib["coeffs"]), int(s.attrib["id"]))
                        for s in planes)
        if not math.isfinite(z0[0]) or not math.isfinite(z1[0]) or z0[0] >= z1[0]:
            raise ValueError("nonphysical external axial boundaries")
        r = [float(x) for x in radial[0].attrib["coeffs"].split()]
        if len(r) != 3 or not np.isfinite(r).all() or r[2] <= 0:
            raise ValueError("invalid radial reactor boundary")
        return {z0[1]: -1, z1[1]: +1, int(radial[0].attrib["id"]): +1}
    except (OSError, ET.ParseError, KeyError, TypeError) as exc:
        raise ValueError("missing executed physical reactor boundary XML") from exc


def read_fixed_source_balance(statepoint: Path) -> dict:
    path = Path(statepoint)
    settings = ET.parse(path.parent / "settings.xml").getroot()
    if (settings.tag != "settings"
            or settings.findtext("run_mode") != "fixed source"
            or settings.findtext("create_fission_neutrons") != "false"
            or settings.findtext("inactive", default="0") != "0"):
        raise ValueError("not physically suppressed-fission fixed-source settings")
    signs = _external_boundary_signs(path.parent)
    with h5py.File(path, "r") as h:
        if (_label(h.attrs["filetype"]) != "statepoint"
                or _label(h["run_mode"][()]) != "fixed source"
                or int(h.attrs["photon_transport"]) != 0
                or int(h.attrs.get("tallies_present", 0)) != 1):
            raise ValueError("requires actual neutron-only fixed-source OpenMC statepoint")
        seed = int(h["seed"][()])
        batch = int(h["current_batch"][()])
        # Native fixed-source OpenMC omits n_inactive entirely because
        # there is no iterative fission-source equilibration. The exported
        # settings.xml has independently required inactive=0.
        inactive = int(h["n_inactive"][()]) if "n_inactive" in h else 0
        n = int(h["n_realizations"][()])
        pop = int(h["n_particles"][()])
        if seed < 1 or n < 2 or inactive != 0 or n != batch or pop < 100:
            raise ValueError("invalid genuine fixed-source random histories")
        tally, labels, means = _tally(
            h, "nerva_global_neutron_balance", "analog", n
        )
        if (int(tally["n_filters"][()]) != 0
                or means.shape[0] != 1
                or not {"absorption", "scatter", "nu-scatter", "nu-fission"}.issubset(labels)):
            raise ValueError("unfiltered analog physical neutron fate scores absent")
        scores = {score: float(means[0, labels.index(score)])
                  for score in ("absorption", "scatter", "nu-scatter", "nu-fission")}
        surface, names, surface_mean = _tally(
            h, "nerva_vacuum_boundary_current", "analog", n
        )
        if (names != ["current"] or int(surface["n_filters"][()]) != 1):
            raise ValueError("native vacuum outgoing neutron current unavailable")
        ids = np.asarray(surface["filters"][()]).reshape(-1)
        if len(ids) != 1:
            raise ValueError("unexpected physical neutron surface filters")
        f = h["tallies"]["filters"]["filter " + str(int(ids[0]))]
        if _label(f["type"][()]) != "surface":
            raise ValueError("neutron current must be scored on actual vacuum surfaces")
        mids = np.asarray(f["bins"][()], dtype=int).reshape(-1)
        if len(mids) != 3 or set(mids) != set(signs) or surface_mean.shape != (3, 1):
            raise ValueError("outgoing neutron current has incomplete vacuum coverage")
        outward = {str(k): float(signs[int(k)] * j)
                   for k, j in zip(mids, surface_mean[:, 0])}
        if not np.isfinite(list(outward.values())).all():
            raise ValueError("nonfinite fixed-source outward neutron current")
    if any(v < -1e-6 for v in outward.values()):
        raise ValueError("inward neutron leakage through an exterior vacuum boundary")
    leakage = float(sum(outward.values()))
    absorbed = scores["absorption"]
    multiply = scores["nu-scatter"] - scores["scatter"]
    if (absorbed <= 0 or leakage <= 0 or multiply < -1e-6
            or abs(scores["nu-fission"]) > 1e-10):
        raise ValueError("nonphysical source-driven neutron collision/escape rates")
    # This is a paired analog event-count identity using actual outward
    # currents and all collision histories; the four real 3000x30 runs
    # close to ~1e-14. Fail at numerical rather than loose MC tolerance.
    balance = absorbed + leakage - multiply
    if abs(balance - 1.0) > 1e-6:
        raise ValueError("actual nonfission neutron source balance violates conservation")
    return {
        "seed": seed, "batches": batch, "particles_per_batch": pop,
        "statepoint_sha256": sha(path),
        "evaluated_geometry_sha256": sha(path.parent / "geometry.xml"),
        "evaluated_materials_sha256": sha(path.parent / "materials.xml"),
        "analog_absorption_per_injected_neutron": absorbed,
        "outward_vacuum_leakage_per_injected_neutron": leakage,
        "outward_vacuum_current_by_surface": outward,
        "net_multiplicative_scattering_per_injected_neutron": multiply,
        "nu_fission_tally_per_injected_neutron_NOT_TRANSPORTED": scores["nu-fission"],
        "nonfission_source_neutron_balance": balance,
        "absolute_balance_error": abs(balance - 1.0),
        "source_driven_no_fission_secondaries": True,
        "reactor_eigenvalue_or_criticality_validated": False,
        "physical_energy_deposition_validated": False,
        "historical_nerva_core_validated": False,
    }


def audit_four_seed_fixed_source(output: Path, seeds: list[int], batches: int) -> dict:
    if len(seeds) != 4 or any(type(i) is not int or i <= 0 for i in seeds) or len(set(seeds)) != 4:
        raise ValueError("four distinct fixed-source transport seeds required")
    rows = []
    for seed in seeds:
        p = Path(output) / f"seed_{seed}" / f"statepoint.{batches}.h5"
        if not p.is_file():
            raise ValueError("missing genuine fixed-source OpenMC statepoint")
        row = read_fixed_source_balance(p)
        if row["seed"] != seed or row["batches"] != batches:
            raise ValueError("actual fixed-source seed/batch provenance inconsistent")
        rows.append(row)
    if (len({r["statepoint_sha256"] for r in rows}) != 4
            or len({r["evaluated_geometry_sha256"] for r in rows}) != 1
            or len({r["evaluated_materials_sha256"] for r in rows}) != 1):
        raise ValueError("changed physical core geometry/materials or duplicated neutron histories")
    keys = (
        "analog_absorption_per_injected_neutron",
        "outward_vacuum_leakage_per_injected_neutron",
        "net_multiplicative_scattering_per_injected_neutron",
        "nonfission_source_neutron_balance",
    )
    means = {}
    between_seed_sem = {}
    for k in keys:
        values = np.asarray([row[k] for row in rows], dtype=float)
        means[k] = float(values.mean())
        between_seed_sem[k] = float(values.std(ddof=1) / math.sqrt(len(values)))
    return {
        "audit": "GENUINE_OPENMC_NONMULTIPLYING_FIXED_SOURCE_CORE_NEUTRON_BALANCE",
        "seed_count": len(rows),
        "per_seed": rows,
        "mean_rates": means,
        "between_independent_seed_standard_errors": between_seed_sem,
        "fixed_external_source_replaces_cloned_eigenvalue_source": True,
        "fission_daughter_transport_disabled_NOT_PHYSICAL_REACTOR_OPERATION": True,
        "fission_source_stationarity_certified": False,
        "reactor_k_eff_inferred_from_fixed_source": False,
        "hot_nuclear_data_or_NASA_historical_fidelity_qualified": False,
    }
