"""Synthetic two-resolution OpenMC convergence report regression.

No HDF5 neutron transport occurs. This verifies the CLI's data model,
physics-identity gates, and conservative mesh comparison against known fields.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from .summarize_convergence import main as report_main


def _write_case(path: Path, shape: tuple[int, int, int],
                normalized_field: np.ndarray, seed: int) -> None:
    transport = path / "transport"
    power = path / "analysis" / "power"
    diag = path / "analysis" / "openmc_diagnostics"
    for folder in (transport, power, diag):
        folder.mkdir(parents=True, exist_ok=True)
    watts = float(np.sum(normalized_field) * 64 / np.prod(shape))
    np.savez_compressed(
        power / "nerva_mesh_fields.npz",
        power_density_w_cm3=normalized_field,
        dimension=np.asarray(shape, dtype=int),
        lower_left_cm=np.array([0., 0., 0.]),
        upper_right_cm=np.array([4., 4., 4.]),
    )
    (power / "metadata.json").write_text(json.dumps({
        "normalized_power_W": watts,
        "requested_power_MW": watts / 1e6,
        "power_normalization_score": "heating-local",
        "photon_transport": False,
    }))
    (transport / "transport_provenance.json").write_text(json.dumps({
        "assembly": "reactor",
        "config": {
            "particles": 1000,
            "batches": 30,
            "inactive": 10,
            "seed": seed,
            "core_rings": 3,
            "photon_transport": False,
            "heating_mesh_dimension": list(shape),
        },
        "nuclear_data": {"cross_sections_sha256": "unit-test-xs"},
        "xml_sha256": {
            "geometry.xml": "unit-test-geometry",
            "materials.xml": "unit-test-material",
            "tallies.xml": "NOT-SAME-ACROSS-RESOLUTIONS",
        },
    }))
    (diag / "openmc_diagnostics.json").write_text(json.dumps({
        "keff": {"mean": 0.9, "std_dev": 0.01},
        "mesh_peaking": {"max_to_mean_active": 1.2},
        "axial_power_peaking": 1.15,
        "mesh_heating_relative_error": {
            "median_relative_error": 0.04,
            "p95_relative_error": 0.14,
        },
    }))


def _report(first: Path, second: Path, output: Path) -> dict:
    previous = sys.argv
    sys.argv = ["summarize_convergence", str(first), str(second),
                "--output", str(output)]
    try:
        assert report_main() == 0
    finally:
        sys.argv = previous
    return json.loads(output.read_text())


def reject(fn, phrase: str):
    try:
        fn()
    except ValueError as exc:
        assert phrase in str(exc), str(exc)
    else:
        raise AssertionError("expected rejection: " + phrase)


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        coarse_root, fine_root = root / "coarse", root / "fine"
        coarse = np.arange(64, dtype=float).reshape((4, 4, 4)) + 1.
        fine = np.repeat(np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1),
                         2, axis=2)
        _write_case(coarse_root, (4, 4, 4), coarse, seed=17)
        _write_case(fine_root, (8, 8, 8), fine, seed=31)

        result = _report(coarse_root, fine_root, root / "convergence.json")
        assert len(result["runs"]) == 2
        first = result["runs"][0]
        comparison = result["successive_differences"][0]["mesh_comparison"]
        assert first["power_field_relative_l2_to_last"] < 1e-12
        assert comparison["common_coarse_shape"] == [4, 4, 4]
        assert comparison["refinement_factors"] == [2, 2, 2]
        assert comparison["total_variation_power_fraction"] < 1e-12
        assert comparison["fine_to_coarse_power_integral_relative_error"] < 1e-12
        assert result["runs"][1]["heating_mesh_dimension"] == [8, 8, 8]
        assert result["runs"][1]["mesh_comparison_to_last"]["comparison_type"] == "same_grid"

        # Fail closed when a different nuclear/material model is compared.
        provenance_path = fine_root / "transport" / "transport_provenance.json"
        original = json.loads(provenance_path.read_text())
        changed = dict(original)
        changed["xml_sha256"] = {**original["xml_sha256"],
                                  "materials.xml": "different-material"}
        provenance_path.write_text(json.dumps(changed))
        reject(lambda: _report(coarse_root, fine_root, root/"invalid.json"),
               "materials.xml")
        provenance_path.write_text(json.dumps(original))

        # Non-nested meshes must not produce a misleading null/error metric.
        density = np.full((6, 8, 8), float(np.sum(coarse)) / 64.)
        power_file = fine_root / "analysis" / "power" / "nerva_mesh_fields.npz"
        np.savez_compressed(
            power_file,
            power_density_w_cm3=density,
            dimension=np.array([6, 8, 8]),
            lower_left_cm=np.array([0., 0., 0.]),
            upper_right_cm=np.array([4., 4., 4.]),
        )
        reject(lambda: _report(coarse_root, fine_root, root/"invalid_mesh.json"),
               "integer-multiple")

    print("NERVA OpenMC synthetic two-resolution report/physics-identity QA: PASS")
    print("  actual neutron transport executed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
