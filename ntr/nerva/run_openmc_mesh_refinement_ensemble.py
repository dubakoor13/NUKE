"""Run uncertainty-aware OpenMC mesh-refinement comparison across REAL seeds.

Both mesh levels require >=4 separately seeded, provenance-verified OpenMC
statepoints. Synthetic mode is opt-in and cannot claim transport validation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .mesh_refinement import HeatingMesh
from .mesh_refinement_statistics import assess_independent_mesh_refinement
from .openmc_replica_convergence import (
    load_mesh_replica, _check_transport_identity,
)
from .summarize_convergence import _require_same_reactor_physics


def evaluate_mesh_refinement_directories(
    coarse_power_dirs: list[Path],
    fine_power_dirs: list[Path],
    *,
    coarse_provenance_files: list[Path] | None = None,
    fine_provenance_files: list[Path] | None = None,
    allow_synthetic: bool = False,
    maximum_sampling_sem_relative_l2: float = 0.01,
    maximum_screen_relative_l2: float = 0.05,
):
    """Require fixed reactor physics and unique independent statepoints.

    Different TALLY mesh dimensions and tallies.xml between mesh levels
    are permitted; geometry.xml, materials.xml and nuclear-data checksum
    must be identical. Within each resolution, transport identity must
    also match including tallies.xml (except independent seeds).
    """
    groups = (coarse_power_dirs, fine_power_dirs)
    provenance_groups = (coarse_provenance_files, fine_provenance_files)
    for paths, provenance in zip(groups, provenance_groups):
        if len(paths) < 4:
            raise ValueError(">=4 independent OpenMC replicas per mesh level required")
        if allow_synthetic and provenance is not None:
            raise ValueError("synthetic mode must not claim transport provenance")
        if not allow_synthetic and (provenance is None or len(provenance) != len(paths)):
            raise ValueError("each real OpenMC replica needs matching provenance")

    heating_groups: list[list[HeatingMesh]] = [[], []]
    identity_groups: list[list[dict]] = [[], []]
    seen_seeds: set[int] = set()
    seen_statepoints: set[str] = set()
    file_identities = []
    for level, paths in enumerate(groups):
        for index, directory in enumerate(paths):
            obj, prov = load_mesh_replica(
                Path(directory),
                provenance_path=(
                    None if allow_synthetic else Path(provenance_groups[level][index])
                ),
                allow_synthetic=allow_synthetic,
            )
            metadata = json.loads(
                (Path(directory) / "metadata.json").read_text(encoding="utf-8")
            )
            if not allow_synthetic and (
                "power_normalization_score" not in metadata
                or "photon_transport" not in metadata
            ):
                raise ValueError("real heating comparison needs explicit score/photon metadata")
            mode = metadata.get("photon_transport", False)
            score = metadata.get("power_normalization_score", "heating-local")
            heating_groups[level].append(HeatingMesh(
                density_w_cm3=obj.power_density_w_cm3,
                lower_left_cm=obj.lower_left_cm,
                upper_right_cm=obj.upper_right_cm,
                normalized_power_w=obj.reactor_power_w,
                score=score,
                photon_transport=mode,
            ))
            file_identities.append({
                "mesh_level": "coarse" if level == 0 else "fine",
                "power_directory": str(directory),
                "seed": obj.openmc_seed,
                "statepoint_sha256": obj.statepoint_sha256,
                "nuclear_data_sha256": obj.nuclear_data_sha256,
            })
            if not allow_synthetic:
                if obj.openmc_seed in seen_seeds or obj.statepoint_sha256 in seen_statepoints:
                    raise ValueError("same OpenMC seed/statepoint reused across mesh ensembles")
                seen_seeds.add(obj.openmc_seed)
                seen_statepoints.add(obj.statepoint_sha256)
                if not identity_groups[level]:
                    if level == 1:
                        _require_same_reactor_physics(identity_groups[0][0], prov)
                else:
                    _check_transport_identity(identity_groups[level][0], prov)
                identity_groups[level].append(prov)

    result, arrays = assess_independent_mesh_refinement(
        heating_groups[0], heating_groups[1],
        maximum_sampling_sem_relative_l2=maximum_sampling_sem_relative_l2,
        maximum_screen_relative_l2=maximum_screen_relative_l2,
    )
    return result, arrays, file_identities


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coarse-power-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--fine-power-dirs", nargs="+", type=Path, required=True)
    p.add_argument("--coarse-provenance-files", nargs="+", type=Path)
    p.add_argument("--fine-provenance-files", nargs="+", type=Path)
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--max-sampling-sem-relative-l2", type=float, default=0.01)
    p.add_argument("--max-screen-relative-l2", type=float, default=0.05)
    p.add_argument("--output", type=Path,
                   default=Path("build/nerva_mesh_refinement_statistics"))
    return p


def main() -> int:
    a = build_parser().parse_args()
    summary, fields, identities = evaluate_mesh_refinement_directories(
        a.coarse_power_dirs,
        a.fine_power_dirs,
        coarse_provenance_files=a.coarse_provenance_files,
        fine_provenance_files=a.fine_provenance_files,
        allow_synthetic=a.allow_synthetic,
        maximum_sampling_sem_relative_l2=a.max_sampling_sem_relative_l2,
        maximum_screen_relative_l2=a.max_screen_relative_l2,
    )
    out = a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"refinement_statistics.npz",**fields)
    report = {
        "source_mode": (
            "synthetic_unverified" if a.allow_synthetic
            else "verified_openmc_file_provenance"
        ),
        "historical_nerva_validation": False,
        "real_openmc_executed_by_this_command": False,
        "independent_seed_verified_statepoints": not a.allow_synthetic,
        "summary": asdict(summary),
        "replicas": identities,
        "arrays": {
            key:"refinement_statistics.npz::"+key for key in fields
        },
    }
    (out/"refinement_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    print("NERVA mesh-refinement Monte Carlo uncertainty: "+summary.screening_status)
    print(f"  coarse/fine realizations: {summary.coarse_replicas}/{summary.fine_replicas}")
    print(f"  common grid relative L2 difference: {summary.observed_common_grid_relative_l2:.6g}")
    print(f"  independent-seed noise relative L2: {summary.common_grid_sampling_sem_relative_l2:.6g}")
    print(f"  uncertainty margin (2x SEM): {summary.upper_screen_relative_l2:.6g}")
    print(f"  subvoxel mean structure/noise: "
          f"{summary.fine_subvoxel_structure_relative_l2:.6g}/"
          f"{summary.fine_subvoxel_sampling_sem_relative_l2:.6g}")
    print(f"  report: {out/'refinement_report.json'}")
    return 0 if summary.screening_status == "SCREEN_PASS_NOT_PHYSICS_VALIDATED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
