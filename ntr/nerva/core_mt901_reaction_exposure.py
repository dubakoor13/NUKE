"""Quantify actual core neutron reactions affected by unavailable local KERMA.

Read four genuine OpenMC eigenvalue HDF5 material-filtered neutron tallies and
the existing independently verified root-reachable, evaluated MT901 inventory.
The result is a fraction of *neutron tracklength/reactions* occurring in
materials lacking evaluated MT901 -- NEVER a fraction of missing reactor heat.
Do not substitute kappa-fission, MT301 or an invented heating coefficient.

This observational qualification changes NO fuel, geometry, XS, source, k,
control setting or thermal-power normalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

SCORES = ("flux", "absorption", "fission", "nu-fission")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _label(value) -> str:
    if not isinstance(value, (bytes, np.bytes_)):
        raise ValueError("native OpenMC tally label must be bytes")
    return bytes(value).decode("utf-8").rstrip("\x00")


def evaluate_reaction_exposure(
    material_ids: list[int], values: np.ndarray,
    mt901_ready_by_id: dict[int, bool], *,
    reachable_ids: set[int],
) -> dict:
    """Quantify observation support; no conversion from reactions to energy."""
    arr = np.asarray(values, dtype=float)
    if (not material_ids or len(set(material_ids)) != len(material_ids)
            or any(type(x) is not int or x <= 0 for x in material_ids)
            or arr.shape != (len(material_ids), len(SCORES))
            or not np.isfinite(arr).all() or np.any(arr < 0)
            or set(mt901_ready_by_id) != set(material_ids)
            or any(type(v) is not bool for v in mt901_ready_by_id.values())
            or not reachable_ids or not reachable_ids <= set(material_ids)):
        raise ValueError("invalid genuine material reaction/MT901 inventory")
    mapped = np.asarray([m in reachable_ids for m in material_ids])
    # Declared-but-unreachable OpenMC materials are not part of the reactor.
    # No epsilon tolerance is granted: these tallies must be exactly zero.
    if np.any(arr[~mapped] != 0):
        raise ValueError("unreachable OpenMC material has neutron reactions")
    supported = np.asarray([mt901_ready_by_id[m] for m in material_ids])
    per_score = {}
    for j, score in enumerate(SCORES):
        column = arr[:, j]
        total = float(math.fsum(column[mapped].tolist()))
        affected = float(math.fsum(column[mapped & ~supported].tolist()))
        covered = float(math.fsum(column[mapped & supported].tolist()))
        if not math.isclose(total, affected + covered,
                            rel_tol=5e-13, abs_tol=1e-16):
            raise ValueError("material neutron reaction partition does not conserve")
        # None means no scored observations, NOT evidence of physical zero.
        per_score[score] = {
            "total_per_fission_source": total,
            "in_materials_without_full_mt901_per_source": affected,
            "in_materials_with_complete_mt901_per_source": covered,
            "fraction_of_scored_activity_without_full_mt901": (
                affected / total if total > 0 else None
            ),
            "numerator_is_local_neutron_heating": False,
            "unobserved_reaction_is_not_a_zero_physical_rate": total == 0.0,
        }
    return {
        "scores": per_score,
        "material_ids": material_ids,
        "reachable_material_ids": sorted(reachable_ids),
        "mt901_missing_material_ids": sorted(
            m for m in reachable_ids if not mt901_ready_by_id[m]
        ),
        "per_material": [
            {
                "material_id": mid,
                "reachable_in_executed_core": mid in reachable_ids,
                "all_material_nuclides_have_evaluated_mt901": mt901_ready_by_id[mid],
                "scored_neutron_activity_per_fission_source": {
                    score: float(arr[i, j]) for j, score in enumerate(SCORES)
                },
            }
            for i, mid in enumerate(material_ids)
        ],
        "thermal_power_or_missing_heat_fraction_computed": False,
    }


def read_real_material_reactions(path: Path, declared_ids: set[int]) -> dict:
    """Get native HDF5 first moments for one actual OpenMC eigenvalue run."""
    path = Path(path)
    with h5py.File(path, "r") as state:
        if ( _label(state.attrs["filetype"]) != "statepoint"
                or _label(state["run_mode"][()]) != "eigenvalue"
                or int(state.attrs["photon_transport"]) != 0
                or tuple(map(int, state.attrs["openmc_version"])) != (0, 15, 3)):
            raise ValueError("requires actual neutron-only OpenMC 0.15.3 eigenvalue statepoint")
        seed = int(state["seed"][()])
        batch = int(state["current_batch"][()])
        inactive = int(state["n_inactive"][()])
        realizations = batch - inactive
        if seed <= 0 or realizations < 2:
            raise ValueError("real eigenvalue source batch provenance is invalid")
        group = state["tallies"]
        matches = [
            item for key, item in group.items()
            if key.startswith("tally ") and isinstance(item, h5py.Group)
            and "name" in item and _label(item["name"][()]) == "nerva_material_transport"
        ]
        if len(matches) != 1:
            raise ValueError("missing or duplicate native core material reaction tally")
        t = matches[0]
        if (_label(t["estimator"][()]) != "tracklength"
                or int(t["n_realizations"][()]) != realizations
                or int(t["n_filters"][()]) != 1
                or [_label(n) for n in t["nuclides"][()]] != ["total"]):
            raise ValueError("incorrect native neutron reaction estimator/active batches")
        filter_ids = np.asarray(t["filters"][()], dtype=int).reshape(-1)
        if filter_ids.size != 1:
            raise ValueError("native material reaction filter count mismatch")
        filt = group["filters"]["filter " + str(int(filter_ids[0]))]
        if _label(filt["type"][()]) != "material":
            raise ValueError("OpenMC reaction tally is not material-filtered")
        mids = list(map(int, np.asarray(filt["bins"][()]).reshape(-1)))
        if set(mids) != declared_ids or len(mids) != len(declared_ids):
            raise ValueError("native neutron score material inventory does not match geometry")
        labels = [_label(v) for v in t["score_bins"][()]]
        if len(labels) != len(set(labels)) or any(s not in labels for s in SCORES):
            raise ValueError("missing genuine neutron flux, absorption or fission reaction")
        moments = np.asarray(t["results"][()], dtype=float)
        if (moments.shape != (len(mids), len(labels), 2)
                or not np.isfinite(moments).all()
                or np.any(moments[:, :, 1] < 0)):
            raise ValueError("invalid native core neutron reaction moments")
        n = float(realizations)
        raw_sum, raw_sq = moments[:, :, 0], moments[:, :, 1]
        # Native scored second moments obey E[X^2] >= E[X]^2.
        # Do not misread raw sums as means, especially for rare fissions.
        if np.any(raw_sq - raw_sum**2/n <
                  -1e-10*np.maximum(raw_sq, 1e-30)):
            raise ValueError("native neutron reaction tally violates raw second-moment identity")
        values = raw_sum[:, [labels.index(s) for s in SCORES]]/n
        if np.any(values < 0):
            raise ValueError("negative actual neutron reaction/tracklength tally")
        return {
            "seed": seed,
            "completed_batch": batch,
            "inactive_batches": inactive,
            "active_realizations": realizations,
            "statepoint_sha256": _sha(path),
            "material_ids": mids,
            "values": values,
        }


def audit_actual_core_mt901_reaction_exposure(
    summary_file: Path, kerma_file: Path,
) -> dict:
    summary_file, kerma_file = Path(summary_file), Path(kerma_file)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    kerma = json.loads(kerma_file.read_text(encoding="utf-8"))
    if (summary.get("execution") != "ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly") != "core"
            or kerma.get("audit") != "INDEPENDENT_EXECUTED_OPENMC_MATERIAL_RESOLVED_MT901_PHYSICS"
            or kerma.get("transport_summary_sha256") != _sha(summary_file)
            or kerma.get("evaluated_neutron_cross_sections_xml_sha256") !=
                summary.get("cross_sections_xml_sha256")
            or kerma.get("real_independent_seed_count") != 4):
        raise ValueError("executed core neutron-transport/evaluated MT901 provenance mismatch")
    declared = kerma.get("physical_materials", []) + kerma.get(
        "uninstantiated_declared_materials", []
    )
    ids = [x.get("material_id") for x in declared]
    if (not ids or any(type(x) is not int or x <= 0 for x in ids)
            or len(set(ids)) != len(ids)
            or len(ids) != kerma.get("declared_material_count")):
        raise ValueError("invalid root-reachable evaluated material/MT901 report")
    reachable = set(kerma.get("executed_reachable_material_ids", []))
    ready = {
        int(row["material_id"]): row["all_material_nuclides_have_evaluated_mt901"]
        for row in kerma["physical_materials"]
    }
    ready.update({
        int(row["material_id"]): False
        for row in kerma["uninstantiated_declared_materials"]
    })
    if (set(ready) != set(ids) or set(ready) & reachable != reachable
            or len(reachable) != kerma.get("physical_material_count")
            or any(type(v) is not bool for v in ready.values())):
        raise ValueError("invalid MT901 support for physically instantiated materials")
    runs = summary.get("runs", [])
    if not isinstance(runs, list) or len(runs) != 4:
        raise ValueError("four independent actual OpenMC neutron seeds required")
    seen_seeds, state_hashes = set(), set()
    rows = []
    for run in runs:
        file = Path(run.get("statepoint_path", ""))
        if not file.is_file() or _sha(file) != run.get("statepoint_sha256"):
            raise ValueError("actual physical neutron HDF5 SHA256 mismatch")
        state = read_real_material_reactions(file, set(ids))
        seed = run.get("seed")
        if (type(seed) is not int or seed <= 0 or seed in seen_seeds
                or seed != state["seed"] or state["statepoint_sha256"] in state_hashes
                or state["completed_batch"] != summary.get("batches")
                or state["inactive_batches"] != summary.get("inactive_batches")):
            raise ValueError("duplicate or inconsistent OpenMC eigenvalue source history")
        seen_seeds.add(seed)
        state_hashes.add(state["statepoint_sha256"])
        row = evaluate_reaction_exposure(
            state["material_ids"], state["values"], ready, reachable_ids=reachable
        )
        rows.append({
            "seed": seed, "statepoint_sha256": state["statepoint_sha256"],
            "active_realizations": state["active_realizations"],
            **row,
        })
    ensemble = {}
    for score in SCORES:
        observed = [row["scores"][score][
            "fraction_of_scored_activity_without_full_mt901"] for row in rows]
        valid = [x for x in observed if x is not None]
        ensemble[score] = {
            "fraction_by_seed": observed,
            "between_seed_mean_missing_data_exposure_fraction": (
                float(np.mean(valid)) if valid else None
            ),
            "between_seed_descriptive_SEM": (
                float(np.std(valid, ddof=1)/math.sqrt(len(valid)))
                if len(valid) >= 2 else None
            ),
            "structural_kerma_coverage_is_not_monte_carlo_heating_uncertainty": True,
        }
    return {
        "audit": "ACTUAL_OPENMC_REACTION_WEIGHTED_MISSING_LOCAL_KERMA_COVERAGE",
        "transport_summary_sha256": _sha(summary_file),
        "evaluated_kerma_report_sha256": _sha(kerma_file),
        "evaluated_xs_index_sha256": summary["cross_sections_xml_sha256"],
        "actual_independent_seed_count": len(seen_seeds),
        "reachable_material_ids": sorted(reachable),
        "material_regions_without_verified_mt901": sorted(
            m for m in reachable if not ready[m]),
        "score_meanings": list(SCORES),
        "per_seed": rows,
        "ensemble": ensemble,
        "qualified_local_neutron_heating": False,
        "fraction_of_missing_reactor_heating_calculated": False,
        "thermal_power_normalization_allowed": False,
        "source_stationarity_or_nasa_core_validated": False,
        "interpretation": (
            "Actual native neutron flux/absorption/fission scores show where "
            "transport is sampled relative to evaluated local KERMA coverage. "
            "Absorption and nu-fission rates are NOT thermal energy deposition; "
            "reaction-weighted missing-data exposure is not the missing-heat "
            "fraction. No stand-in neutron heating or nuclear power supplied."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--material-kerma", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    out = audit_actual_core_mt901_reaction_exposure(a.summary, a.material_kerma)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    for score in SCORES:
        print("Executed cold core", score, "missing MT901 exposure fraction:",
              out["ensemble"][score]["between_seed_mean_missing_data_exposure_fraction"])
    print("Unqualified local-neutron heating is NOT a zero-power measurement")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
