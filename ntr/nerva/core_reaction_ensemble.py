"""Cross-seed physical neutron reaction-rate and incident-energy validation.

Run ONLY on genuine four-seed OpenMC statepoints of an unchanged core and
unchanged evaluated nuclear-data files. Reconcile global TRACKLENGTH reaction
rates with independent material and incident-energy filtered estimators.

Four independent initialization seeds do not imply that active fission
generations or fission descendants are independent. The resulting between-
seed variations are descriptive physical transport evidence, not a formal
confidence interval, source-stationarity proof, NASA benchmark, or criticality
optimization.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
import openmc

from .core_reaction_partition import (
    SCORES, inspect_real_statepoint as verify_material_reaction_partition,
)


def _sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _read_core_rates(path: Path) -> dict:
    """Read physically independent score filters from a completed statepoint."""
    with openmc.StatePoint(path, autolink=False) as sp:
        if sp.run_mode != "eigenvalue" or sp.photon_transport:
            raise ValueError("real core reaction audit requires neutron-only eigenvalue transport")
        if sp.n_realizations < 2 or not sp.tallies_present:
            raise ValueError("completed OpenMC active neutron reaction tallies missing")
        full = sp.get_tally(name="nerva_global_reaction_tracklength")
        material = sp.get_tally(name="nerva_material_transport")
        incident = sp.get_tally(name="nerva_material_energy_transport")
        if (full.filters or full.estimator != "tracklength"
                or material.estimator != "tracklength"
                or incident.estimator != "tracklength"):
            raise ValueError("all physical cross-seed rates require matching tracklength estimators")
        if (len(material.filters) != 1
                or not isinstance(material.filters[0], openmc.MaterialFilter)
                or len(incident.filters) != 2
                or not isinstance(incident.filters[0], openmc.MaterialFilter)
                or not isinstance(incident.filters[1], openmc.EnergyFilter)):
            raise ValueError("material/incident-energy filter identities differ")
        mids = [int(v) for v in material.filters[0].bins]
        incident_mids = [int(v) for v in incident.filters[0].bins]
        if mids != incident_mids or len(mids) < 2:
            raise ValueError("energy-filtered and material tally IDs differ")
        edges = np.asarray(incident.filters[1].bins, dtype=float)
        if (edges.ndim != 2 or len(edges) < 3 or edges.shape[1] != 2
                or not np.isfinite(edges).all()
                or np.any(edges[:, 1] <= edges[:, 0])
                or np.any(np.abs(edges[1:, 0] - edges[:-1, 1]) >
                          1e-9 * np.maximum(1., edges[1:, 0]))):
            raise ValueError("invalid actual incident neutron energy groups")
        nenergy = len(edges)
        nmaterials = len(mids)
        scores = {}
        for score in SCORES:
            def tally_values(obj, shape):
                if score not in obj.scores:
                    raise ValueError("missing neutron reaction: " + score)
                a = np.asarray(obj.get_values(scores=[score], value="mean"),
                               dtype=float)
                if a.size != np.prod(shape):
                    raise ValueError("OpenMC reaction tally dimension mismatch: " + score)
                a = a.reshape(shape)
                if not np.isfinite(a).all() or np.any(a < 0):
                    raise ValueError("nonphysical neutron reaction rate: " + score)
                return a
            global_total = float(tally_values(full, (1,))[0])
            material_values = tally_values(material, (nmaterials,))
            mat_energy = tally_values(incident, (nmaterials,nenergy))
            if (abs(float(material_values.sum()) - global_total) > 1e-6
                    or abs(float(mat_energy.sum()) - global_total) > max(1e-6, 0.02 * global_total)):
                raise ValueError("real core reaction partition does not conserve global rate")
            if global_total <= 0:
                raise ValueError("required neutron reaction rate is zero: " + score)
            scores[score] = {
                "global_per_source": global_total,
                "material_per_source": material_values.tolist(),
                "incident_energy_per_source": mat_energy.sum(axis=0).tolist(),
                "normalized_incident_energy_spectrum":
                    (mat_energy.sum(axis=0) / float(mat_energy.sum())).tolist(),
            }
        return {
            "material_ids": mids,
            "incident_energy_bins_eV": edges.tolist(),
            "completed_batch": int(sp.current_batch),
            "active_tally_realizations": int(sp.n_realizations),
            "scores": scores,
        }


def analyze_reaction_replica_statistics(data: list[dict]) -> dict:
    """Compare identical physical reactions over different OpenMC RNG seeds."""
    if len(data) < 4:
        raise ValueError("at least four independent real-core neutron seeds required")
    seeds = [item["seed"] for item in data]
    if (any(not isinstance(x, int) or x < 1 for x in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("neutron reaction ensemble seeds duplicate or invalid")
    ids = data[0]["material_ids"]
    bins = np.asarray(data[0]["incident_energy_bins_eV"], dtype=float)
    nenergy = len(bins)
    if not ids or nenergy < 3:
        raise ValueError("empty physical neutron reaction material/energy filters")
    for item in data:
        if (item["material_ids"] != ids
                or not np.array_equal(
                    np.asarray(item["incident_energy_bins_eV"], dtype=float), bins
                )):
            raise ValueError("cross-seed neutron material or energy filters changed")
    result = {}
    for score in SCORES:
        total = np.array([float(x["scores"][score]["global_per_source"])
                          for x in data], dtype=float)
        mat = np.array([x["scores"][score]["material_per_source"]
                        for x in data], dtype=float)
        energies = np.array([
            x["scores"][score]["incident_energy_per_source"] for x in data
        ], dtype=float)
        spectra = np.array([
            x["scores"][score]["normalized_incident_energy_spectrum"]
            for x in data
        ], dtype=float)
        if (total.shape != (len(data),)
                or mat.shape != (len(data), len(ids))
                or energies.shape != (len(data), nenergy)
                or spectra.shape != (len(data), nenergy)
                or not np.isfinite(total).all()
                or not np.isfinite(mat).all()
                or not np.isfinite(energies).all()
                or not np.isfinite(spectra).all()
                or np.any(total <= 0) or np.any(mat < 0)
                or np.any(energies < 0) or np.any(spectra < 0)
                or any(abs(float(row.sum())-1.) > 1e-10 for row in spectra)
                or np.any(np.abs(mat.sum(axis=1)-total) > 1e-6)
                or np.any(np.abs(energies.sum(axis=1)-total) >
                          np.maximum(1e-6, 0.02*total))):
            raise ValueError("nonconservative or mismatched physical neutron reaction samples")
        pairwise_spectral_tv = [
            0.5*float(np.abs(spectra[i]-spectra[j]).sum())
            for i,j in combinations(range(len(data)), 2)
        ]
        mean = float(total.mean())
        sd = float(total.std(ddof=1))
        mat_mean = mat.mean(axis=0)
        per_material = []
        for i,ident in enumerate(ids):
            m = float(mat_mean[i])
            per_material.append({
                "material_id": ident,
                "mean_rate_per_source": m,
                "independent_seed_standard_deviation": float(mat[:,i].std(ddof=1)),
                "mean_fraction_of_global_reaction":
                    float(m/mat_mean.sum()) if mat_mean.sum() > 0 else None,
            })
        result[score] = {
            "per_seed_global_rate_per_source": total.tolist(),
            "mean_global_reaction_per_source": mean,
            "between_seed_sample_standard_deviation": sd,
            "between_seed_coefficient_of_variation": sd / mean,
            "descriptive_between_seed_standard_error": sd/math.sqrt(len(data)),
            "global_rate_min_per_source": float(total.min()),
            "global_rate_max_per_source": float(total.max()),
            "per_seed_incident_energy_spectrum": spectra.tolist(),
            "mean_pairwise_incident_energy_spectrum_tv":
                float(np.mean(pairwise_spectral_tv)),
            "max_pairwise_incident_energy_spectrum_tv":
                float(np.max(pairwise_spectral_tv)),
            "material_reaction_partition": per_material,
        }
    return {
        "analysis": "GENUINE_OPENMC_CROSS_SEED_CORE_NEUTRON_REACTION_RATES",
        "independent_seed_count": len(data),
        "seeds": seeds,
        "material_ids": ids,
        "incident_energy_bins_eV": bins.tolist(),
        "reactions": result,
        "independent_fission_source_stationarity_demonstrated": False,
        "nasa_historical_nerva_neutron_physics_validated": False,
        "interpretation": (
            "Actual OpenMC neutron-only eigenvalue tracklength estimators. "
            "Global neutron absorption and nu-fission exactly reconcile "
            "to material and incident-neutron energy tallies. "
            "Cross-seed differences are DESCRIPTIVE: four distinct initial "
            "RNG seeds do not make correlated fission generations or "
            "descendant particle histories independent, and these are not "
            "experimental NASA reactor uncertainty bounds. The cold, "
            "strongly subcritical surrogate remains physically unvalidated."
        ),
    }


def audit_real_neutron_reactions(summary_file: Path) -> dict:
    """Prevent synthetic summary/statepoint substitution or model changes."""
    summary_file = Path(summary_file)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if (summary.get("execution") != "ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly") != "core"):
        raise ValueError("requires genuine executed OpenMC reactor core")
    runs = summary.get("runs")
    if not isinstance(runs, list) or len(runs) < 4:
        raise ValueError("real reaction ensemble must use >=4 genuine seeds")
    seeds_seen = set()
    hashes = set()
    geometry = set()
    tally_xml = set()
    material_xml = set()
    records = []
    for entry in runs:
        seed = entry.get("seed")
        statepoint = Path(entry.get("statepoint_path", ""))
        if (not isinstance(seed, int) or seed < 1 or seed in seeds_seen
                or not statepoint.is_file()
                or _sha256(statepoint) != entry.get("statepoint_sha256")):
            raise ValueError("duplicate source seed or genuine statepoint SHA mismatch")
        seeds_seen.add(seed)
        if entry["statepoint_sha256"] in hashes:
            raise ValueError("identical OpenMC statepoint used across seeds")
        hashes.add(entry["statepoint_sha256"])
        materials = statepoint.parent / "materials.xml"
        geom = statepoint.parent / "geometry.xml"
        tallies = statepoint.parent / "tallies.xml"
        provenance = statepoint.parent / "transport_provenance.json"
        if any(not f.is_file() for f in (materials,geom,tallies,provenance)):
            raise ValueError("missing physical transport XML/provenance")
        transport = json.loads(provenance.read_text(encoding="utf-8"))
        if (transport.get("config", {}).get("seed") != seed
                or transport.get("nuclear_data", {}).get("cross_sections_sha256")
                != summary.get("cross_sections_xml_sha256")):
            raise ValueError("reaction ensemble must share actual evaluated neutron data")
        for file in (materials,geom,tallies):
            if _sha256(file) != transport.get("xml_sha256", {}).get(file.name):
                raise ValueError("OpenMC physical model XML differs from seed provenance")
        material_xml.add(_sha256(materials))
        geometry.add(_sha256(geom))
        tally_xml.add(_sha256(tallies))
        verified = verify_material_reaction_partition(statepoint, materials)
        if not verified["all_partition_screens_passed"]:
            raise ValueError("real material reaction conservation failed")
        row = _read_core_rates(statepoint)
        if row["completed_batch"] != summary.get("batches"):
            raise ValueError("real OpenMC statepoint is not final active batch")
        row["seed"] = seed
        row["statepoint_sha256"] = entry["statepoint_sha256"]
        row["material_xml_sha256"] = _sha256(materials)
        records.append(row)
    if any(len(collection) != 1 for collection in (geometry,material_xml,tally_xml)):
        raise ValueError("reactor geometry/material/tallies differ across neutron replicas")
    result = analyze_reaction_replica_statistics(records)
    result["statepoint_sha256"] = [row["statepoint_sha256"] for row in records]
    result["materials_xml_sha256"] = next(iter(material_xml))
    result["geometry_xml_sha256"] = next(iter(geometry))
    result["tallies_xml_sha256"] = next(iter(tally_xml))
    result["nuclear_data_index_sha256"] = summary["cross_sections_xml_sha256"]
    result["real_transport_summary_sha256"] = _sha256(summary_file)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = audit_real_neutron_reactions(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print("Actual OpenMC global/material/energy neutron reactions: PASS")
    for score in SCORES:
        item = report["reactions"][score]
        print(score, "mean rate/source", item["mean_global_reaction_per_source"],
              "seed CV", item["between_seed_coefficient_of_variation"],
              "incident-energy cross-seed mean spectral TV",
              item["mean_pairwise_incident_energy_spectrum_tv"])
    print("Fission source and NASA historical reactor physical validation: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
