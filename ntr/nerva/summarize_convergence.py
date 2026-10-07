"""Summarize numerical convergence across completed NERVA OpenMC runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


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


def _relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a.ravel()))
    delta = float(np.linalg.norm((b - a).ravel()))
    return delta / denom if denom > 0.0 else 0.0


def main() -> int:
    args = parse_args()
    if len(args.runs) < 2:
        raise ValueError("at least two completed runs are required")

    records = []
    fields = []
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

        field_data = np.load(
            analysis / "power" / "nerva_mesh_fields.npz"
        )
        power_field = np.asarray(
            field_data["power_density_w_cm3"],
            dtype=float,
        )
        fields.append(power_field)

        config = provenance["config"]
        records.append(
            {
                "root": str(root),
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
        if fields[i].shape != final_field.shape:
            record["power_field_relative_l2_to_last"] = None
        else:
            record["power_field_relative_l2_to_last"] = (
                _relative_l2(final_field, fields[i])
            )

    successive = []
    for i in range(1, len(records)):
        if fields[i - 1].shape == fields[i].shape:
            field_delta = _relative_l2(
                fields[i - 1],
                fields[i],
            )
        else:
            field_delta = None

        previous_keff = records[i - 1]["keff_mean_reported_only"]
        current_keff = records[i]["keff_mean_reported_only"]
        keff_delta = (
            None
            if previous_keff is None or current_keff is None
            else float(current_keff) - float(previous_keff)
        )

        successive.append(
            {
                "from": records[i - 1]["root"],
                "to": records[i]["root"],
                "power_field_relative_l2": field_delta,
                "keff_delta_reported_only": keff_delta,
                "mesh_peaking_delta": (
                    records[i]["mesh_power_peaking"]
                    - records[i - 1]["mesh_power_peaking"]
                ),
                "p95_relative_error_delta": (
                    records[i]["heating_p95_relative_error"]
                    - records[i - 1][
                        "heating_p95_relative_error"
                    ]
                ),
            }
        )

    result = {
        "interpretation": (
            "Numerical convergence study only. Runs should differ only in "
            "sampling/resolution controls when using this report to assess "
            "Monte Carlo convergence. No reactor-design optimization is "
            "performed."
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
