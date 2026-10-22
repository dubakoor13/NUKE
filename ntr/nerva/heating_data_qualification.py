"""Real OpenMC neutron-only heating-data physical qualification.

Unlike an executable smoke test, reactor heat-deposition validation MUST
require a nonzero, usable deposited-energy score in the actual statepoint.
Some evaluated-data bundles permit neutron transport and fission scoring
while returning identically ZERO for 'heating-local'. Such calculations
must not be repurposed into reactor thermal power or coolant heat flux.

This checks saved OpenMC HDF5 bytes; no surrogate heating, replacement
KERMA values, reactor geometry changes, criticality tuning, or optimization.
A nonzero score is necessary, NOT sufficient for historical NERVA validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .power import heating_score_for_transport


REQUIRED_MESH = "nerva_3d_neutronics"
REQUIRED_MATERIAL = "nerva_material_transport"
REQUIRED_SCORES = ("heating", "heating-local", "kappa-fission")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def _name_bytes(value) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8").rstrip("\x00")
    raise ValueError("invalid OpenMC HDF5 byte string")


def _score_totals(tallies, tally_name: str, scores: tuple[str, ...], *,
                  expected_active_batches: int) -> tuple[dict, dict]:
    """Read actual OpenMC first and second moments for active-batch uncertainty."""
    candidates = [
        item for item in tallies.values()
        if isinstance(item, h5py.Group) and "name" in item
        and _name_bytes(item["name"][()]) == tally_name
    ]
    if len(candidates) != 1:
        raise ValueError("missing or duplicate genuine OpenMC tally " + tally_name)
    tally = candidates[0]
    n = int(tally["n_realizations"][()])
    if n < 2 or n != expected_active_batches:
        raise ValueError("OpenMC heating tally active realizations mismatch executed batches")
    labels = [_name_bytes(v) for v in tally["score_bins"][()]]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate OpenMC reaction score")
    results = np.asarray(tally["results"][()], dtype=float)
    if (results.ndim != 3 or results.shape[1:] != (len(labels), 2)
            or not np.isfinite(results).all()):
        raise ValueError("invalid OpenMC heating tally array")
    ret = {}
    sem_bounds = {}
    for score in scores:
        if score not in labels:
            raise ValueError("required OpenMC heating score absent: " + score)
        first = results[:, labels.index(score), 0]
        second = results[:, labels.index(score), 1]
        vals = first / n
        mean_square = second / n
        if not np.isfinite(vals).all() or np.any(second < 0):
            raise ValueError("nonfinite or negative deposited-energy tally moments")
        # E[X²] >= E[X]², allowing only IEEE-754 roundoff.
        variance_numerator = mean_square - vals * vals
        scale = np.maximum(1.0, np.maximum(np.abs(mean_square), vals * vals))
        if np.any(variance_numerator < -1e-8 * scale):
            raise ValueError("OpenMC heating tally second moments violate nonnegative variance")
        per_bin_sem = np.sqrt(np.maximum(0.0, variance_numerator) / (n - 1))
        ret[score] = float(np.sum(vals, dtype=np.float64))
        # Different spatial/material bins are correlated by neutron histories.
        # Summing marginal SEMs is a conservative covariance-agnostic bound.
        sem_bounds[score] = float(np.sum(per_bin_sem, dtype=np.float64))
    return ret, sem_bounds


def evaluate_heating_scores(mesh: dict[str, float],
                            material: dict[str, float],
                            *,
                            coherence_relative_tolerance: float = 1e-8,
                            mesh_local_sem_upper_bound: float | None = None,
                            material_local_sem_upper_bound: float | None = None,
                            minimum_signal_to_sem: float = 5.0) -> dict:
    """Assess data availability, without inventing a replacement estimator.

    For a mesh that covers the complete reactor, 'heating' and 'heating-local'
    must reconcile with their independently filtered material tallies.
    These are same-estimator reaction scores, so no Monte Carlo
    cross-estimator tolerance relaxation is appropriate.
    """
    if (not math.isfinite(coherence_relative_tolerance)
            or coherence_relative_tolerance <= 0
            or coherence_relative_tolerance >= 0.1):
        raise ValueError("invalid mesh/material energy closure tolerance")
    if (not math.isfinite(minimum_signal_to_sem)
            or minimum_signal_to_sem < 1 or minimum_signal_to_sem > 20):
        raise ValueError("invalid deposited-energy statistical-resolution threshold")
    for bound in (mesh_local_sem_upper_bound, material_local_sem_upper_bound):
        if bound is not None and (not math.isfinite(bound) or bound < 0):
            raise ValueError("nonfinite/negative deposited-energy standard-error bound")
    required = set(REQUIRED_SCORES)
    if not required.issubset(mesh) or not {"heating", "heating-local"}.issubset(material):
        raise ValueError("required evaluated OpenMC heating scores missing")
    if any(not math.isfinite(v) for v in list(mesh.values()) + list(material.values())):
        raise ValueError("nonfinite OpenMC energy deposition")
    reference = float(mesh["heating"])
    local = float(mesh["heating-local"])
    kappa = float(mesh["kappa-fission"])
    if reference <= 0 or kappa <= 0:
        raise ValueError("nonpositive evaluated standard heating or fission recoverable energy")
    material_heating = float(material["heating"])
    material_local = float(material["heating-local"])
    tol = coherence_relative_tolerance * max(1., abs(reference))
    ordinary_closed = abs(reference - material_heating) <= tol
    local_closed = abs(local - material_local) <= tol
    if not ordinary_closed or not local_closed:
        raise ValueError("whole-core deposited energy is not consistently scored by material")
    # A nonzero mean alone may be unresolved Monte Carlo noise. The
    # statepoint records per-batch moments, NOT independent neutron histories.
    nonzero = local > 1e-12 * max(reference, kappa)
    uncertainty_available = (
        mesh_local_sem_upper_bound is not None
        and material_local_sem_upper_bound is not None
    )
    sem_bound = (
        max(mesh_local_sem_upper_bound, material_local_sem_upper_bound)
        if uncertainty_available else None
    )
    resolved = bool(
        uncertainty_available and nonzero
        and (sem_bound == 0.0 or local / sem_bound >= minimum_signal_to_sem)
    )
    usable = resolved
    if not nonzero:
        status = "BLOCKED_ZERO_NEUTRON_LOCAL_HEATING"
    elif not uncertainty_available:
        status = "BLOCKED_MISSING_LOCAL_HEATING_STATISTICS"
    elif not resolved:
        status = "BLOCKED_UNRESOLVED_LOCAL_HEATING_STATISTICS"
    else:
        status = "LOCAL_HEATING_STATISTICALLY_RESOLVED_NOT_PHYSICALLY_VALIDATED"
    return {
        "neutron_only_required_deposition_score": heating_score_for_transport(False),
        "ordinary_heating_eV_per_source": reference,
        "local_heating_eV_per_source": local,
        "recoverable_fission_kappa_eV_per_source": kappa,
        "material_heating_eV_per_source": material_heating,
        "material_local_heating_eV_per_source": material_local,
        "local_to_ordinary_heating_ratio": local / reference,
        "mesh_material_energy_partition_consistent": True,
        "mesh_local_heating_conservative_sem_upper_bound":mesh_local_sem_upper_bound,
        "material_local_heating_conservative_sem_upper_bound":material_local_sem_upper_bound,
        "minimum_signal_to_sem_ratio":minimum_signal_to_sem,
        "local_heating_signal_to_conservative_sem_bound":(
            None if sem_bound is None or sem_bound == 0 else local / sem_bound
        ),
        "local_heating_statistically_resolved":resolved,
        "statistical_resolution_is_physical_validation":False,
        "neutron_only_local_heating_data_usable": bool(usable),
        "physical_heating_qualification_status":status,
        "thermal_power_normalization_allowed": bool(usable),
        "nasa_nerva_core_thermal_physics_validated": False,
        "interpretation": (
            "Standard heating and recoverable fission energy may be nonzero "
            "while evaluated heating-local is identically zero. In that case "
            "the neutron-only thermal normalization input is UNAVAILABLE. "
            "A tiny nonzero heating-local score without numerical "
            "resolution in executed batch moments is also inadequate. "
            "Summed marginal SEMs conservatively bound unknown bin covariance "
            "but do not prove fission-chain independence or historical fidelity. "
            "Do not substitute kappa-fission, distribute missing heat, "
            "or switch to ordinary heating without separately verified "
            "energy-deposition assumptions and nuclear-data coverage."
        ),
    }


def inspect_statepoint(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as state:
        if _name_bytes(state.attrs["filetype"]) != "statepoint":
            raise ValueError("not an actual OpenMC statepoint")
        if _name_bytes(state["run_mode"][()]) != "eigenvalue":
            raise ValueError("not genuine OpenMC eigenvalue transport")
        if int(state.attrs.get("photon_transport", -1)) != 0:
            raise ValueError("neutron-only local-heating audit cannot score photon transport")
        if int(state.attrs.get("tallies_present", 0)) != 1:
            raise ValueError("OpenMC heating tallies not saved")
        seed = int(state["seed"][()])
        batches = int(state["current_batch"][()])
        inactive = int(state["n_inactive"][()])
        if seed <= 0 or inactive < 0 or batches <= inactive + 1:
            raise ValueError("invalid real transport seed/batch count")
        tallies = state["tallies"]
        active = batches - inactive
        mesh, mesh_sem = _score_totals(
            tallies, REQUIRED_MESH, REQUIRED_SCORES,
            expected_active_batches=active,
        )
        material, material_sem = _score_totals(
            tallies, REQUIRED_MATERIAL, ("heating", "heating-local"),
            expected_active_batches=active,
        )
        report = evaluate_heating_scores(
            mesh, material,
            mesh_local_sem_upper_bound=mesh_sem["heating-local"],
            material_local_sem_upper_bound=material_sem["heating-local"],
        )
    report.update({
        "source": "OPENMC_FORMAT_EIGENVALUE_HDF5_CONTENT",
        "executable_origin_proven_by_hdf5_contents_alone": False,
        "statepoint_sha256": digest(path),
        "statepoint_path": str(path),
        "seed": seed,
        "completed_batches": batches,
        "inactive_batches": inactive,
        "active_tally_realizations": active,
    })
    return report


def audit_real_ensemble(summary: Path) -> dict:
    summary = Path(summary)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    if (payload.get("execution") != "ACTUAL_OPENMC_TRANSPORT"
            or payload.get("assembly") != "core"):
        raise ValueError("real OpenMC core summary is required")
    runs = payload.get("runs")
    if not isinstance(runs, list) or len(runs) < 4:
        raise ValueError("four or more independent OpenMC statepoints required")
    seen: set[int] = set()
    reports = []
    for item in runs:
        file = Path(item["statepoint_path"])
        observed = inspect_statepoint(file)
        if (observed["seed"] != item.get("seed")
                or observed["statepoint_sha256"] != item.get("statepoint_sha256")):
            raise ValueError("OpenMC source identity/hash does not match transport summary")
        if observed["seed"] in seen:
            raise ValueError("OpenMC seed duplicated in heating evidence")
        seen.add(observed["seed"])
        reports.append(observed)
    qualified = all(row["neutron_only_local_heating_data_usable"]
                    for row in reports)
    return {
        "audit": "INDEPENDENT_REAL_OPENMC_HEATING_DATA_QUALIFICATION",
        "transport_summary_sha256": digest(summary),
        "number_of_independent_seeds": len(seen),
        "all_real_statepoints_have_usable_neutron_local_heating": qualified,
        "thermal_power_normalization_allowed": qualified,
        "status": ("DATA_PRESENT_REQUIRES_MORE_PHYSICAL_VALIDATION"
                   if qualified else "BLOCKED_NO_VERIFIED_LOCAL_HEAT_DEPOSITION"),
        "runs": reports,
        "core_historical_nasa_validation": False,
        "fission_source_stationarity_validated": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-usable-heating", action="store_true",
                   help="fail closed if local heating data are unavailable")
    args = p.parse_args()
    report = audit_real_ensemble(args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Real OpenMC neutron-local heating status:", report["status"])
    for row in report["runs"]:
        print("seed", row["seed"], "heating", row["ordinary_heating_eV_per_source"],
              "heating-local", row["local_heating_eV_per_source"],
              "kappa-fission", row["recoverable_fission_kappa_eV_per_source"])
    print("NASA NERVA core thermal validation: NOT ESTABLISHED")
    if args.require_usable_heating and not report["thermal_power_normalization_allowed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
