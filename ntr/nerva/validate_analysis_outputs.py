"""Validate a completed NERVA OpenMC -> thermal analysis package."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate internal consistency of a completed NERVA analysis "
            "directory. This is a software/physics bookkeeping QA check, "
            "not a historical criticality acceptance test."
        )
    )
    parser.add_argument(
        "analysis_root",
        type=Path,
        help="Root produced by ntr.nerva.run_analysis.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional report path. Default: "
            "<analysis_root>/analysis_validation.json"
        ),
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_number(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite: {value!r}")
    return number


def _assert_close(
    actual: float,
    expected: float,
    name: str,
    rel_tol: float = 1.0e-6,
    abs_tol: float = 1.0e-6,
) -> None:
    if not math.isclose(
        actual,
        expected,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    ):
        raise ValueError(
            f"{name} mismatch: actual={actual}, expected={expected}"
        )


def _fraction_sum(mapping: dict[str, Any], name: str) -> float:
    values = []
    for key, value in mapping.items():
        number = _finite_number(value, f"{name}.{key}")
        if number < -1.0e-12:
            raise ValueError(f"{name}.{key} is negative")
        values.append(number)
    return float(sum(values))


def main() -> int:
    args = parse_args()
    root = args.analysis_root

    power_path = root / "power" / "metadata.json"
    fields_path = root / "power" / "nerva_mesh_fields.npz"
    diagnostics_path = (
        root / "openmc_diagnostics" / "openmc_diagnostics.json"
    )
    fuel_path = root / "fuel" / "thermal_summary.json"
    tie_path = root / "tie" / "tie_thermal_summary.json"
    element_channels_path = (
        root / "element_channels" / "element_channel_summary.json"
    )
    tie_instances_path = (
        root / "tie_instances" / "tie_instance_summary.json"
    )

    power = _load_json(power_path)
    diagnostics = _load_json(diagnostics_path)
    fuel = _load_json(fuel_path)
    tie = _load_json(tie_path)
    element_channels = (
        _load_json(element_channels_path)
        if element_channels_path.is_file()
        else None
    )
    tie_instances = (
        _load_json(tie_instances_path)
        if tie_instances_path.is_file()
        else None
    )

    if not fields_path.is_file():
        raise FileNotFoundError(fields_path)

    requested_power_mw = _finite_number(
        power["requested_power_MW"],
        "requested_power_MW",
    )
    normalized_power_w = _finite_number(
        power["normalized_power_W"],
        "normalized_power_W",
    )
    source_rate = _finite_number(
        power["source_rate_per_s"],
        "source_rate_per_s",
    )

    if requested_power_mw <= 0.0:
        raise ValueError("requested_power_MW must be positive")
    if normalized_power_w <= 0.0:
        raise ValueError("normalized_power_W must be positive")
    if source_rate <= 0.0:
        raise ValueError("source_rate_per_s must be positive")

    _assert_close(
        normalized_power_w,
        requested_power_mw * 1.0e6,
        "normalized total power",
        rel_tol=1.0e-8,
        abs_tol=1.0e-3,
    )

    diagnostics_source_rate = _finite_number(
        diagnostics["source_rate_per_s"],
        "diagnostics.source_rate_per_s",
    )
    _assert_close(
        diagnostics_source_rate,
        source_rate,
        "diagnostics source rate",
        rel_tol=1.0e-12,
        abs_tol=0.0,
    )

    fuel_power_w = _finite_number(
        power["fuel_power_W"],
        "fuel_power_W",
    )
    tie_power_w = _finite_number(
        power["tie_power_W"],
        "tie_power_W",
    )
    if fuel_power_w < 0.0 or tie_power_w < 0.0:
        raise ValueError("fuel/tie deposited power must be non-negative")
    if fuel_power_w + tie_power_w > normalized_power_w * 1.000001:
        raise ValueError(
            "fuel + tie deposited power exceeds normalized total power"
        )

    _assert_close(
        _finite_number(fuel["fuel_power_W"], "fuel.fuel_power_W"),
        fuel_power_w,
        "fuel power cross-check",
        rel_tol=1.0e-8,
        abs_tol=1.0e-3,
    )
    _assert_close(
        _finite_number(
            tie["total_tie_power_W"],
            "tie.total_tie_power_W",
        ),
        tie_power_w,
        "tie power cross-check",
        rel_tol=1.0e-8,
        abs_tol=1.0e-3,
    )

    mesh_peaking = _finite_number(
        diagnostics["mesh_peaking"]["max_to_mean_active"],
        "mesh_peaking.max_to_mean_active",
    )
    axial_peaking = _finite_number(
        diagnostics["axial_power_peaking"],
        "axial_power_peaking",
    )
    if mesh_peaking < 1.0:
        raise ValueError("mesh power peaking must be >= 1")
    if axial_peaking < 1.0:
        raise ValueError("axial power peaking must be >= 1")

    error_stats = diagnostics["mesh_heating_relative_error"]
    for key in (
        "median_relative_error",
        "p95_relative_error",
        "max_relative_error",
    ):
        if _finite_number(error_stats[key], f"uncertainty.{key}") < 0.0:
            raise ValueError(f"uncertainty.{key} must be non-negative")


    energy_deposition = diagnostics.get("energy_deposition", {})
    if energy_deposition:
        for key in (
            "mesh_heating_local_W",
            "mesh_heating_W",
            "heating_minus_heating_local_W",
            "heating_minus_local_fraction",
            "material_heating_local_sum_W",
            "material_heating_sum_W",
        ):
            _finite_number(
                energy_deposition[key],
                f"energy_deposition.{key}",
            )

        if energy_deposition["mesh_heating_local_W"] <= 0.0:
            raise ValueError(
                "energy_deposition.mesh_heating_local_W must be positive"
            )
        if energy_deposition["mesh_heating_W"] <= 0.0:
            raise ValueError(
                "energy_deposition.mesh_heating_W must be positive"
            )

    material_fraction_sum = _fraction_sum(
        diagnostics["material_power_fractions"],
        "material_power_fractions",
    )
    if diagnostics["material_power_fractions"]:
        _assert_close(
            material_fraction_sum,
            1.0,
            "material power fraction closure",
            rel_tol=1.0e-8,
            abs_tol=1.0e-8,
        )

    for spectrum_name in (
        "fuel_flux_spectral_fractions",
        "hydrogen_flux_spectral_fractions",
    ):
        spectrum_sum = _fraction_sum(
            diagnostics[spectrum_name],
            spectrum_name,
        )
        if spectrum_sum > 0.0:
            _assert_close(
                spectrum_sum,
                1.0,
                f"{spectrum_name} closure",
                rel_tol=1.0e-8,
                abs_tol=1.0e-8,
            )

    source_convergence = diagnostics.get(
        "source_convergence",
        {},
    )
    for group_name in (
        "active_k_batch_metrics",
        "active_entropy_metrics",
    ):
        metrics = source_convergence.get(group_name, {})
        for key in (
            "window_mean",
            "window_std",
            "window_range",
            "window_relative_range",
            "window_slope_per_index",
        ):
            value = metrics.get(key)
            if value is not None:
                _finite_number(
                    value,
                    f"source_convergence.{group_name}.{key}",
                )

    keff = diagnostics.get("keff", {})
    keff_mean = keff.get("mean")
    keff_std = keff.get("std_dev")
    if keff_mean is not None:
        _finite_number(keff_mean, "keff.mean")
    if keff_std is not None:
        if _finite_number(keff_std, "keff.std_dev") < 0.0:
            raise ValueError("keff.std_dev must be non-negative")

    fuel_t_in = _finite_number(
        fuel["inlet_temperature_K"],
        "fuel.inlet_temperature_K",
    )
    fuel_t_out = _finite_number(
        fuel["outlet_temperature_K"],
        "fuel.outlet_temperature_K",
    )
    fuel_p_in = _finite_number(
        fuel["inlet_pressure_Pa"],
        "fuel.inlet_pressure_Pa",
    )
    fuel_p_out = _finite_number(
        fuel["outlet_pressure_Pa"],
        "fuel.outlet_pressure_Pa",
    )
    if fuel_t_out < fuel_t_in:
        raise ValueError("fuel-channel outlet temperature is below inlet")
    if fuel_p_out > fuel_p_in:
        raise ValueError("fuel-channel outlet pressure exceeds inlet")

    max_coolant_wall = _finite_number(
        fuel["maximum_coolant_wall_temperature_K"],
        "fuel.maximum_coolant_wall_temperature_K",
    )
    max_fuel_surface = _finite_number(
        fuel["maximum_fuel_surface_temperature_K"],
        "fuel.maximum_fuel_surface_temperature_K",
    )
    max_peak_fuel = _finite_number(
        fuel["maximum_peak_fuel_temperature_K"],
        "fuel.maximum_peak_fuel_temperature_K",
    )
    if max_coolant_wall < fuel_t_out:
        raise ValueError(
            "maximum coolant wall temperature is below outlet bulk temperature"
        )
    if max_fuel_surface < max_coolant_wall:
        raise ValueError(
            "maximum fuel surface temperature is below coolant-wall maximum"
        )
    if max_peak_fuel < max_fuel_surface:
        raise ValueError(
            "maximum peak fuel temperature is below surface maximum"
        )

    tie_t_in = _finite_number(
        tie["inlet_temperature_K"],
        "tie.inlet_temperature_K",
    )
    tie_t_out = _finite_number(
        tie["outlet_temperature_K"],
        "tie.outlet_temperature_K",
    )
    tie_p_in = _finite_number(
        tie["inlet_pressure_Pa"],
        "tie.inlet_pressure_Pa",
    )
    tie_p_out = _finite_number(
        tie["outlet_pressure_Pa"],
        "tie.outlet_pressure_Pa",
    )
    if tie_t_out < tie_t_in:
        raise ValueError("tie-tube outlet temperature is below inlet")
    if tie_p_out > tie_p_in:
        raise ValueError("tie-tube outlet pressure exceeds inlet")


    if element_channels is not None:
        reconstructed_wall = _finite_number(
            element_channels["reconstructed_wall_power_W"],
            "element_channels.reconstructed_wall_power_W",
        )
        _assert_close(
            reconstructed_wall,
            fuel_power_w,
            "element/channel reconstructed fuel wall power",
            rel_tol=1.0e-6,
            abs_tol=1.0e-2,
        )

        reconstructed_direct = _finite_number(
            element_channels[
                "reconstructed_direct_hydrogen_power_W"
            ],
            "element_channels.reconstructed_direct_hydrogen_power_W",
        )
        expected_direct = _finite_number(
            power.get("fuel_hydrogen_direct_power_W", 0.0),
            "fuel_hydrogen_direct_power_W",
        )
        if expected_direct > 0.0:
            _assert_close(
                reconstructed_direct,
                expected_direct,
                "element/channel direct hydrogen power",
                rel_tol=1.0e-6,
                abs_tol=1.0e-2,
            )

        if _finite_number(
            element_channels["maximum_peak_fuel_temperature_K"],
            "element_channels.maximum_peak_fuel_temperature_K",
        ) <= 0.0:
            raise ValueError(
                "element/channel maximum peak fuel temperature must be positive"
            )


    if tie_instances is not None:
        _assert_close(
            _finite_number(
                tie_instances[
                    "reconstructed_tie_solid_power_W"
                ],
                "tie_instances.reconstructed_tie_solid_power_W",
            ),
            tie_power_w,
            "per-tie reconstructed solid power",
            rel_tol=1.0e-6,
            abs_tol=1.0e-2,
        )

        expected_supply_direct = _finite_number(
            power.get(
                "tie_supply_hydrogen_direct_power_W",
                0.0,
            ),
            "tie_supply_hydrogen_direct_power_W",
        )
        expected_return_direct = _finite_number(
            power.get(
                "tie_return_hydrogen_direct_power_W",
                0.0,
            ),
            "tie_return_hydrogen_direct_power_W",
        )

        if expected_supply_direct > 0.0:
            _assert_close(
                _finite_number(
                    tie_instances[
                        "reconstructed_direct_supply_hydrogen_power_W"
                    ],
                    "tie_instances.reconstructed_direct_supply_hydrogen_power_W",
                ),
                expected_supply_direct,
                "per-tie direct supply H2 power",
                rel_tol=1.0e-6,
                abs_tol=1.0e-2,
            )

        if expected_return_direct > 0.0:
            _assert_close(
                _finite_number(
                    tie_instances[
                        "reconstructed_direct_return_hydrogen_power_W"
                    ],
                    "tie_instances.reconstructed_direct_return_hydrogen_power_W",
                ),
                expected_return_direct,
                "per-tie direct return H2 power",
                rel_tol=1.0e-6,
                abs_tol=1.0e-2,
            )

        tie_instance_inlet = _finite_number(
            tie_instances["inlet_temperature_K"],
            "tie_instances.inlet_temperature_K",
        )
        tie_instance_max_outlet = _finite_number(
            tie_instances["maximum_outlet_temperature_K"],
            "tie_instances.maximum_outlet_temperature_K",
        )
        if tie_instance_max_outlet < tie_instance_inlet:
            raise ValueError(
                "all-tie maximum outlet temperature is below inlet"
            )

        if _finite_number(
            tie_instances["maximum_wall_temperature_K"],
            "tie_instances.maximum_wall_temperature_K",
        ) <= 0.0:
            raise ValueError(
                "all-tie maximum wall temperature must be positive"
            )

    with np.load(fields_path) as fields:
        required_arrays = {
            "power_density_w_cm3",
            "power_density_std_w_cm3",
            "flux_cm2_s",
            "absorption_rate_cm3_s",
            "fission_rate_cm3_s",
            "nu_fission_rate_cm3_s",
            "fuel_power_density_w_cm3",
            "tie_power_density_w_cm3",
            "dimension",
        }
        missing = required_arrays - set(fields.files)
        if missing:
            raise KeyError(
                "normalized field package is missing arrays: "
                + ", ".join(sorted(missing))
            )

        dimension = tuple(
            int(v) for v in np.asarray(fields["dimension"]).reshape(-1)
        )
        if len(dimension) != 3:
            raise ValueError("dimension must contain exactly three values")

        for name in required_arrays - {"dimension"}:
            array = np.asarray(fields[name], dtype=float)
            if array.shape != dimension:
                raise ValueError(
                    f"{name} shape {array.shape} != stored dimension {dimension}"
                )
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains non-finite values")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "analysis_root": str(root),
        "checks": {
            "normalized_power_closure": True,
            "source_rate_consistency": True,
            "fuel_power_consistency": True,
            "tie_power_consistency": True,
            "mesh_peaking_finite": True,
            "monte_carlo_uncertainty_finite": True,
            "material_power_fraction_closure": True,
            "energy_deposition_diagnostics_finite": True,
            "spectral_fraction_closure": True,
            "keff_report_finite_if_present": True,
            "source_convergence_metrics_finite": True,
            "fuel_temperature_pressure_trends": True,
            "tie_temperature_pressure_trends": True,
            "normalized_field_arrays_finite": True,
            "element_channel_power_closure": (
                element_channels is not None
            ),
            "tie_instance_power_closure": (
                tie_instances is not None
            ),
        },
        "diagnostic_values": {
            "requested_power_MW": requested_power_mw,
            "source_rate_per_s": source_rate,
            "mesh_power_peaking": mesh_peaking,
            "axial_power_peaking": axial_peaking,
            "heating_p95_relative_error": error_stats[
                "p95_relative_error"
            ],
            "keff_mean_reported_only": keff_mean,
            "keff_std_dev": keff_std,
        },
        "interpretation": (
            "Internal consistency PASS only. This does not constitute "
            "historical NERVA criticality or performance validation."
        ),
    }

    output = (
        args.output
        if args.output is not None
        else root / "analysis_validation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("NERVA analysis package validation: PASS")
    print(f"  requested power: {requested_power_mw:.6g} MW")
    print(f"  mesh peaking: {mesh_peaking:.6f}")
    print(f"  axial peaking: {axial_peaking:.6f}")
    print(
        f"  heating p95 relative error: "
        f"{error_stats['p95_relative_error']:.6f}"
    )
    print(f"  report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
