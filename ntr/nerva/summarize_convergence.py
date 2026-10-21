"""Summarize numerical convergence across completed NERVA OpenMC runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .mesh_refinement import HeatingMesh, compare_heating_meshes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a sequence of completed NERVA full-run roots as a "
            "numerical convergence study. Physics settings are not optimized."
        )
    )
    parser.add_argument(
        "runs",
        type=Path,
        nargs="+",
        help="Full-run roots containing transport/ and analysis/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_convergence.json"),
    )
    return parser.parse_args()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _analysis_dir(root: Path) -> Path:
    nested = root / "analysis"
    return nested if nested.is_dir() else root


def _transport_dir(root: Path) -> Path:
    nested = root / "transport"
    return nested if nested.is_dir() else root


_SAMPLING_ONLY_CONFIG = {
    "particles", "batches", "inactive", "seed",
    "heating_mesh_dimension", "diagnostic_energy_groups",
    "axial_mesh_bins", "tally_rel_err_trigger",
    "trigger_max_batches", "trigger_batch_interval",
    "entropy_mesh_xy", "entropy_mesh_z",
}


def _require_same_reactor_physics(first: dict, second: dict) -> None:
    """Reject comparisons that change reactor physics/nuclear provenance."""
    if first.get("assembly") != second.get("assembly"):
        raise ValueError("cannot compare different OpenMC assembly geometries")
    xs_a = first.get("nuclear_data", {}).get("cross_sections_sha256")
    xs_b = second.get("nuclear_data", {}).get("cross_sections_sha256")
    if not xs_a or xs_a != xs_b:
        raise ValueError("OpenMC nuclear-data checksums differ or are missing")
    for name in ("geometry.xml", "materials.xml"):
        a = first.get("xml_sha256", {}).get(name)
        b = second.get("xml_sha256", {}).get(name)
        if not a or a != b:
            raise ValueError(
                f"OpenMC {name} checksum differs or is missing; "
                "mesh convergence cannot compare different reactor physics"
            )
    configs = []
    for record in (first, second):
        cfg = record.get("config")
        if not isinstance(cfg, dict):
            raise ValueError("OpenMC transport configuration missing")
        configs.append({
            key: value for key, value in cfg.items()
            if key not in _SAMPLING_ONLY_CONFIG
        })
    if configs[0] != configs[1]:
        raise ValueError("non-sampling OpenMC transport physics settings differ")


def main() -> int:
    args = parse_args()
    if len(args.runs) < 2:
        raise ValueError("at least two completed runs are required")

    records = []
    fields: list[HeatingMesh] = []
    provenances: list[dict] = []
    for root in args.runs:
        analysis = _analysis_dir(root)
        transport = _transport_dir(root)

        provenance = _json(
            transport / "transport_provenance.json"
        )
        diagnostics = _json(
            analysis
            / "openmc_diagnostics"
            / "openmc_diagnostics.json"
        )
        power = _json(analysis / "power" / "metadata.json")

        with np.load(
            analysis / "power" / "nerva_mesh_fields.npz",
            allow_pickle=False,
        ) as field_data:
            power_field = np.asarray(
                field_data["power_density_w_cm3"], dtype=float,
            )
            dimension = np.asarray(field_data["dimension"], dtype=int)
            if dimension.shape != (3,) or tuple(dimension) != power_field.shape:
                raise ValueError("OpenMC power array and metadata dimensions differ")
            lower_left = np.asarray(field_data["lower_left_cm"], dtype=float)
            upper_right = np.asarray(field_data["upper_right_cm"], dtype=float)
        mesh = HeatingMesh(
            density_w_cm3=power_field,
            lower_left_cm=lower_left,
            upper_right_cm=upper_right,
            normalized_power_w=float(power["normalized_power_W"]),
            score=power.get("power_normalization_score", "heating-local"),
            photon_transport=power.get("photon_transport", False),
        )
        mesh.validate()
        fields.append(mesh)
        provenances.append(provenance)

        config = provenance["config"]
        records.append(
            {
                "root": str(root),
                "heating_mesh_dimension": list(power_field.shape),
                "heating_normalization_score": mesh.score,
                "voxel_volume_cm3": float(np.prod(
                    (upper_right - lower_left) / np.asarray(power_field.shape)
                )),
                "particles": int(config["particles"]),
                "batches": int(config["batches"]),
                "inactive": int(config["inactive"]),
                "total_active_histories_nominal": int(
                    config["particles"]
                    * (config["batches"] - config["inactive"])
                ),
                "requested_power_MW": power["requested_power_MW"],
                "keff_mean_reported_only": diagnostics[
                    "keff"
                ]["mean"],
                "keff_std_dev": diagnostics["keff"]["std_dev"],
                "mesh_power_peaking": diagnostics[
                    "mesh_peaking"
                ]["max_to_mean_active"],
                "axial_power_peaking": diagnostics[
                    "axial_power_peaking"
                ],
                "heating_median_relative_error": diagnostics[
                    "mesh_heating_relative_error"
                ]["median_relative_error"],
                "heating_p95_relative_error": diagnostics[
                    "mesh_heating_relative_error"
                ]["p95_relative_error"],
            }
        )

    final_field = fields[-1]
    for i, record in enumerate(records):
        _require_same_reactor_physics(provenances[i], provenances[-1])
        comparison = compare_heating_meshes(fields[i], final_field)
        record["mesh_comparison_to_last"] = comparison
        record["power_field_relative_l2_to_last"] = comparison[
            "relative_l2_power_density_difference"
        ]

    successive = []
    for i in range(1, len(records)):
        _require_same_reactor_physics(provenances[i - 1], provenances[i])
        mesh_comparison = compare_heating_meshes(fields[i - 1], fields[i])
        previous_keff = records[i - 1]["keff_mean_reported_only"]
        current_keff = records[i]["keff_mean_reported_only"]
        keff_delta = (
            None if previous_keff is None or current_keff is None
            else float(current_keff) - float(previous_keff)
        )
        successive.append(
            {
                "from": records[i - 1]["root"],
                "to": records[i]["root"],
                "power_field_relative_l2": mesh_comparison[
                    "relative_l2_power_density_difference"
                ],
                "mesh_comparison": mesh_comparison,
                "keff_delta_reported_only": keff_delta,
                "mesh_peaking_delta": (
                    records[i]["mesh_power_peaking"]
                    - records[i - 1]["mesh_power_peaking"]
                ),
                "p95_relative_error_delta": (
                    records[i]["heating_p95_relative_error"]
                    - records[i - 1]["heating_p95_relative_error"]
                ),
            }
        )

    result = {
        "interpretation": (
            "Numerical convergence study only. Runs should differ only in "
            "sampling/resolution controls when using this report to assess "
            "Monte Carlo convergence. All compared runs must have identical "
            "geometry, materials, nuclear data and non-sampling physics. "
            "Nested fine-grid power is coarsened conservatively, preserving "
            "the thermal-power integral; nonnested grids are rejected. "
            "Spatial agreement at the shared coarse grid does not certify "
            "fine-scale hotspot resolution, fission-source convergence or "
            "historical NERVA benchmark agreement. No reactor-design optimization "
            "is performed."
        ),
        "runs": records,
        "successive_differences": successive,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA numerical-convergence summary written:")
    print(f"  runs: {len(records)}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
