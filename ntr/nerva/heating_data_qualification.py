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
import xml.etree.ElementTree as ET

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
        # Local tally resolution and whole-domain integration alone do not
        # certify evaluated hot-core KERMA, source stationarity or a NERVA
        # reactor heat-deposition benchmark. This individual-statepoint
        # record must never authorize physical power normalization.
        "thermal_power_normalization_allowed": False,
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



def _verify_complete_core_heating_tally_support(state, statepoint_path: Path) -> dict:
    """Verify actual tally filters span every material and the vacuum core.

    Identically clipped mesh and material tallies can agree numerically
    without representing whole-reactor heating. The executed statepoint
    filter/mesh HDF5 and exported geometry/material XML must independently
    identify a whole-core domain before any heat-deposition closure is used.
    """
    folder = statepoint_path.parent
    geometry_path = folder / "geometry.xml"
    materials_path = folder / "materials.xml"
    if not geometry_path.is_file() or not materials_path.is_file():
        raise ValueError("executed OpenMC geometry/material XML required for whole-core heating")

    materials_root = ET.parse(materials_path).getroot()
    if materials_root.tag != "materials":
        raise ValueError("invalid executed OpenMC materials XML")
    material_ids = [int(node.attrib["id"])
                    for node in materials_root.findall("material")]
    if (len(material_ids) < 2 or any(i <= 0 for i in material_ids)
            or len(set(material_ids)) != len(material_ids)):
        raise ValueError("invalid executed OpenMC physical material IDs")

    geometry_root = ET.parse(geometry_path).getroot()
    if geometry_root.tag != "geometry":
        raise ValueError("invalid executed OpenMC geometry XML")
    vacuum = [node for node in geometry_root.findall("surface")
              if node.attrib.get("boundary") == "vacuum"]
    radial = [node for node in vacuum
              if node.attrib.get("type") == "z-cylinder"]
    axial = [node for node in vacuum
             if node.attrib.get("type") == "z-plane"]
    if len(radial) != 1 or len(axial) != 2 or len(vacuum) != 3:
        raise ValueError("unsupported actual core vacuum boundary for whole-core mesh")
    cx, cy, radius = (float(x) for x in radial[0].attrib["coeffs"].split())
    zlimits = sorted(float(node.attrib["coeffs"]) for node in axial)
    if (not all(math.isfinite(v) for v in (cx, cy, radius, *zlimits))
            or radius <= 0 or zlimits[1] <= zlimits[0]):
        raise ValueError("invalid executed OpenMC core vacuum surfaces")

    tallies = state["tallies"]
    def named_tally(name: str):
        found = [g for g in tallies.values()
                 if isinstance(g, h5py.Group) and "name" in g
                 and _name_bytes(g["name"][()]) == name]
        if len(found) != 1:
            raise ValueError("missing or duplicate heating tally filter identity: " + name)
        return found[0]

    def resolved_filter(tally, expected_type: str):
        filters = np.asarray(tally["filters"][()], dtype=np.int64)
        if (filters.shape != (1,) or int(tally["n_filters"][()]) != 1):
            raise ValueError("heating tally must have exactly one complete-core filter")
        key = "filter " + str(int(filters[0]))
        f = tallies["filters"].get(key)
        if f is None or _name_bytes(f["type"][()]) != expected_type:
            raise ValueError("incorrect physical heating tally filter type")
        if _name_bytes(tally["estimator"][()]) != "tracklength":
            raise ValueError("inconsistent heating tally estimators")
        if [_name_bytes(x) for x in tally["nuclides"][()]] != ["total"]:
            raise ValueError("heating tallies must represent total nuclide population")
        return f

    mesh_tally = named_tally(REQUIRED_MESH)
    mesh_filter = resolved_filter(mesh_tally, "mesh")
    mesh_id = int(mesh_filter["bins"][()])
    mesh = tallies["meshes"].get("mesh " + str(mesh_id))
    if (mesh is None or _name_bytes(mesh["name"][()]) != "nerva_core_mesh"
            or _name_bytes(mesh["type"][()]) != "regular"):
        raise ValueError("heating mesh must reference the executed whole-core regular mesh")
    dims = np.asarray(mesh["dimension"][()], dtype=np.int64)
    lower = np.asarray(mesh["lower_left"][()], dtype=float)
    upper = np.asarray(mesh["upper_right"][()], dtype=float)
    if (dims.shape != (3,) or np.any(dims <= 0)
            or lower.shape != (3,) or upper.shape != (3,)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(upper <= lower)):
        raise ValueError("invalid executed whole-core tally mesh geometry")
    n_mesh = int(np.prod(dims))
    if (int(mesh_filter["n_bins"][()]) != n_mesh
            or int(mesh_tally["results"].shape[0]) != n_mesh):
        raise ValueError("heating mesh tally does not resolve its reported physical bins")
    # Full bounding box of the radial vacuum cylinder and axial end planes.
    eps = 1e-7
    if (lower[0] > cx - radius + eps or upper[0] < cx + radius - eps
            or lower[1] > cy - radius + eps or upper[1] < cy + radius - eps
            or lower[2] > zlimits[0] + eps or upper[2] < zlimits[1] - eps):
        raise ValueError("heating tally mesh clips the executed physical core")

    material_tally = named_tally(REQUIRED_MATERIAL)
    material_filter = resolved_filter(material_tally, "material")
    covered = [int(v) for v in np.asarray(material_filter["bins"][()]).reshape(-1)]
    if (len(covered) != len(set(covered)) or set(covered) != set(material_ids)
            or int(material_filter["n_bins"][()]) != len(material_ids)
            or int(material_tally["results"].shape[0]) != len(material_ids)):
        raise ValueError("material heating tally does not include every executed material")

    return {
        "whole_core_heating_tally_spatial_and_material_coverage_verified": True,
        "executed_mesh_dimensions": dims.tolist(),
        "executed_mesh_lower_left_cm": lower.tolist(),
        "executed_mesh_upper_right_cm": upper.tolist(),
        "executed_vacuum_radial_radius_cm": radius,
        "executed_vacuum_axial_bounds_cm": zlimits,
        "executed_material_ids_in_heating_tally": sorted(covered),
        "executed_geometry_xml_sha256": digest(geometry_path),
        "executed_materials_xml_sha256": digest(materials_path),
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
        particles = int(state["n_particles"][()])
        if seed <= 0 or inactive < 0 or batches <= inactive + 1 or particles < 2:
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
        whole_core_support = _verify_complete_core_heating_tally_support(
            state, path
        )
        report = evaluate_heating_scores(
            mesh, material,
            mesh_local_sem_upper_bound=mesh_sem["heating-local"],
            material_local_sem_upper_bound=material_sem["heating-local"],
        )
    report.update({
        **whole_core_support,
        "source": "OPENMC_FORMAT_EIGENVALUE_HDF5_CONTENT",
        "executable_origin_proven_by_hdf5_contents_alone": False,
        "statepoint_sha256": digest(path),
        "statepoint_path": str(path),
        "seed": seed,
        "completed_batches": batches,
        "inactive_batches": inactive,
        "active_tally_realizations": active,
        "transport_particles_per_generation": particles,
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
    # Compare recorded hardware-independent transport conditions with
    # the values actually persisted by OpenMC's eigenvalue statepoints.
    expected = (
        payload.get("batches"),
        payload.get("inactive_batches"),
        payload.get("particles_per_generation"),
    )
    if (
        any(isinstance(x,bool) or not isinstance(x,int) for x in expected)
        or expected[0] <= expected[1] + 1
        or expected[1] < 0 or expected[2] < 2
    ):
        raise ValueError("invalid physical eigenvalue source settings in summary")
    seen: set[int] = set()
    reports = []
    for item in runs:
        file = Path(item["statepoint_path"])
        observed = inspect_statepoint(file)
        if (observed["seed"] != item.get("seed")
                or observed["statepoint_sha256"] != item.get("statepoint_sha256")):
            raise ValueError("OpenMC source identity/hash does not match transport summary")
        actual = (
            observed["completed_batches"],
            observed["inactive_batches"],
            observed["transport_particles_per_generation"],
        )
        if actual != expected:
            raise ValueError("OpenMC active-source population/batches disagree across real seeds")
        if observed["seed"] in seen:
            raise ValueError("OpenMC seed duplicated in heating evidence")
        seen.add(observed["seed"])
        reports.append(observed)
    qualified = all(row["neutron_only_local_heating_data_usable"]
                    for row in reports)
    values = np.array(
        [row["local_heating_eV_per_source"] for row in reports], dtype=float
    )
    zero_moments = [
        row["local_heating_eV_per_source"] == 0.0
        and row["mesh_local_heating_conservative_sem_upper_bound"] == 0.0
        and row["material_local_heating_conservative_sem_upper_bound"] == 0.0
        for row in reports
    ]
    all_zero = all(zero_moments)
    mixed_zero = any(zero_moments) and not all_zero
    # Replicates are separate transport RNG seeds, but a small sample and
    # source genealogy do not establish independence of fission histories.
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    sem = sd / math.sqrt(len(values))
    if all_zero:
        evidence = "ZERO_LOCAL_HEAT_DEPOSITION_IN_ALL_REAL_SEEDS"
    elif mixed_zero:
        evidence = "INCONSISTENT_ZERO_AND_NONZERO_LOCAL_HEATING_ACROSS_SEEDS"
    elif qualified:
        evidence = "POSITIVE_LOCAL_HEATING_BUT_CORE_PHYSICS_UNQUALIFIED"
    else:
        evidence = "LOCAL_HEATING_UNRESOLVED_IN_ONE_OR_MORE_REAL_SEEDS"
    # A resolved *tally* alone cannot authorize thermal power calibration:
    # evaluated nuclear KERMA, neutron-source stationarity and the historical
    # core operating-temperature physics remain separate unpassed gates.
    allowed = False
    return {
        "audit": "INDEPENDENT_REAL_OPENMC_HEATING_DATA_QUALIFICATION",
        "transport_summary_sha256": digest(summary),
        "number_of_independent_seeds": len(seen),
        "all_real_statepoints_have_usable_neutron_local_heating": qualified,
        "ensemble_local_heat_deposition_evidence": evidence,
        "exactly_zero_local_deposition_moments_in_every_seed": all_zero,
        "inconsistent_zero_and_nonzero_local_deposition_across_seeds": mixed_zero,
        "independent_seed_local_heating_mean_eV_per_source": mean,
        "independent_seed_local_heating_sample_sd_eV_per_source": sd,
        "independent_seed_local_heating_sem_eV_per_source": sem,
        "independent_seed_replicates_are_neutron_history_independence_proof": False,
        "all_runs_have_identical_executed_source_settings": True,
        "executed_particles_per_generation": expected[2],
        "executed_active_batches": expected[0] - expected[1],
        "additional_physical_requirements_before_power_normalization": [
            "evaluated nonzero physical KERMA with provenance",
            "physically qualified material operating temperatures",
            "independently demonstrated fission-source stationarity",
            "independent physical heat-deposition benchmark",
        ],
        "thermal_power_normalization_allowed": allowed,
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
