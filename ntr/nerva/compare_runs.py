"""Compare two completed NERVA OpenMC analysis packages numerically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two completed NERVA analysis roots. This reports "
            "numerical differences only and does not select/optimize designs."
        )
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_run_comparison.json"),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _field_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"field shape mismatch: {a.shape} vs {b.shape}")

    delta = b - a
    l2_a = float(np.linalg.norm(a.ravel()))
    l2_delta = float(np.linalg.norm(delta.ravel()))
    max_abs = float(np.max(np.abs(delta)))
    max_ref = float(np.max(np.abs(a)))

    return {
        "reference_l2": l2_a,
        "delta_l2": l2_delta,
        "relative_l2": (
            l2_delta / l2_a if l2_a > 0.0 else 0.0
        ),
        "max_abs_delta": max_abs,
        "max_abs_delta_over_reference_max": (
            max_abs / max_ref if max_ref > 0.0 else 0.0
        ),
    }


def _scalar_delta(a, b) -> dict[str, float | None]:
    if a is None or b is None:
        return {
            "reference": a,
            "candidate": b,
            "delta": None,
            "relative_delta": None,
        }
    av = float(a)
    bv = float(b)
    return {
        "reference": av,
        "candidate": bv,
        "delta": bv - av,
        "relative_delta": (
            (bv - av) / av if av != 0.0 else None
        ),
    }


def main() -> int:
    args = parse_args()

    ref_power = _load_json(
        args.reference / "power" / "metadata.json"
    )
    cand_power = _load_json(
        args.candidate / "power" / "metadata.json"
    )
    ref_diag = _load_json(
        args.reference
        / "openmc_diagnostics"
        / "openmc_diagnostics.json"
    )
    cand_diag = _load_json(
        args.candidate
        / "openmc_diagnostics"
        / "openmc_diagnostics.json"
    )

    ref_fields = np.load(
        args.reference / "power" / "nerva_mesh_fields.npz"
    )
    cand_fields = np.load(
        args.candidate / "power" / "nerva_mesh_fields.npz"
    )

    common_fields = [
        name
        for name in (
            "power_density_w_cm3",
            "fuel_power_density_w_cm3",
            "tie_power_density_w_cm3",
            "fuel_hydrogen_power_density_w_cm3",
            "flux_cm2_s",
            "absorption_rate_cm3_s",
            "fission_rate_cm3_s",
            "nu_fission_rate_cm3_s",
        )
        if name in ref_fields.files and name in cand_fields.files
    ]

    field_comparison = {
        name: _field_metrics(ref_fields[name], cand_fields[name])
        for name in common_fields
    }

    ref_keff = ref_diag.get("keff", {}).get("mean")
    cand_keff = cand_diag.get("keff", {}).get("mean")

    result = {
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "interpretation": (
            "Numerical comparison only. No design ranking, criticality "
            "targeting, or control/enrichment optimization is performed."
        ),
        "normalization": {
            "reference_requested_power_MW": ref_power[
                "requested_power_MW"
            ],
            "candidate_requested_power_MW": cand_power[
                "requested_power_MW"
            ],
        },
        "reported_keff_diagnostic_only": _scalar_delta(
            ref_keff,
            cand_keff,
        ),
        "mesh_power_peaking": _scalar_delta(
            ref_diag["mesh_peaking"]["max_to_mean_active"],
            cand_diag["mesh_peaking"]["max_to_mean_active"],
        ),
        "axial_power_peaking": _scalar_delta(
            ref_diag["axial_power_peaking"],
            cand_diag["axial_power_peaking"],
        ),
        "heating_p95_relative_error": _scalar_delta(
            ref_diag["mesh_heating_relative_error"][
                "p95_relative_error"
            ],
            cand_diag["mesh_heating_relative_error"][
                "p95_relative_error"
            ],
        ),
        "fields": field_comparison,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA analysis comparison written:")
    print(f"  compared fields: {len(common_fields)}")
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
