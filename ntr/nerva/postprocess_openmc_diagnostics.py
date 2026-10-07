"""Advanced OpenMC diagnostics postprocessor for the NERVA-derived model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import openmc

from .power import EV_TO_J


_MATERIAL_ORDER = (
    "fuel",
    "hydrogen",
    "zrc",
    "reflector",
    "graphite",
    "zrh",
    "inconel",
    "b4c",
    "aluminum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract advanced OpenMC diagnostics from a NERVA-derived "
            "statepoint using the source-rate normalization from metadata.json."
        )
    )
    parser.add_argument("statepoint", type=Path)
    parser.add_argument(
        "--power-metadata",
        type=Path,
        required=True,
        help="metadata.json produced by postprocess_statepoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/nerva_diagnostics"),
    )
    return parser.parse_args()


def _tally_values(
    tally: openmc.Tally,
    score: str,
    value: str = "mean",
) -> np.ndarray:
    sliced = tally.get_slice(scores=[score])
    data = np.asarray(
        sliced.get_reshaped_data(value=value, expand_dims=True),
        dtype=float,
    )
    if data.shape[-2:] != (1, 1):
        raise ValueError(
            f"unexpected trailing tally dimensions {data.shape[-2:]}"
        )
    return np.asarray(data[..., 0, 0], dtype=float)


def _mesh_score(
    tally: openmc.Tally,
    score: str,
    value: str = "mean",
) -> tuple[np.ndarray, openmc.RegularMesh]:
    mesh_filter = tally.find_filter(openmc.MeshFilter)
    mesh = mesh_filter.mesh
    if not isinstance(mesh, openmc.RegularMesh):
        raise TypeError(f"{tally.name} must use RegularMesh")

    mesh_shape = tuple(int(v) for v in mesh.dimension)
    data = _tally_values(tally, score, value=value)

    if data.shape[: len(mesh_shape)] != mesh_shape:
        raise ValueError(
            f"{tally.name} mesh prefix {data.shape} does not begin "
            f"with {mesh_shape}"
        )

    if data.ndim > len(mesh_shape):
        data = np.sum(
            data,
            axis=tuple(range(len(mesh_shape), data.ndim)),
        )

    return data, mesh


def _uncertain_pair(obj: Any) -> tuple[float | None, float | None]:
    if obj is None:
        return None, None

    for mean_name, std_name in (
        ("nominal_value", "std_dev"),
        ("n", "s"),
        ("mean", "std_dev"),
    ):
        if hasattr(obj, mean_name):
            mean = float(getattr(obj, mean_name))
            std_attr = getattr(obj, std_name, None)
            std = None if std_attr is None else float(std_attr)
            return mean, std

    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        return float(obj[0]), float(obj[1])

    try:
        return float(obj), None
    except (TypeError, ValueError):
        return None, None


def _keff_pair(statepoint: openmc.StatePoint) -> tuple[float | None, float | None]:
    value = getattr(statepoint, "keff", None)
    if value is None:
        value = getattr(statepoint, "k_combined", None)
    return _uncertain_pair(value)


def _entropy_history(statepoint: openmc.StatePoint) -> list[float]:
    entropy = getattr(statepoint, "entropy", None)
    if entropy is None:
        return []
    array = np.asarray(entropy, dtype=float).reshape(-1)
    return [float(v) for v in array]


def _spectral_fractions(
    energy_bins_ev: np.ndarray,
    spectrum: np.ndarray,
) -> dict[str, float]:
    values = np.asarray(spectrum, dtype=float).reshape(-1)
    bins = np.asarray(energy_bins_ev, dtype=float)

    if bins.ndim != 2 or bins.shape[1] != 2:
        raise ValueError("EnergyFilter bins must have shape (N, 2)")
    if len(values) != len(bins):
        raise ValueError("spectrum length does not match energy bins")

    total = float(np.sum(values))
    if total <= 0.0:
        return {
            "thermal_below_0p625_eV": 0.0,
            "epithermal_0p625eV_to_100keV": 0.0,
            "fast_above_100keV": 0.0,
        }

    upper = bins[:, 1]
    lower = bins[:, 0]

    thermal = float(np.sum(values[upper <= 0.625]))
    fast = float(np.sum(values[lower >= 1.0e5]))
    epi = max(0.0, total - thermal - fast)

    return {
        "thermal_below_0p625_eV": thermal / total,
        "epithermal_0p625eV_to_100keV": epi / total,
        "fast_above_100keV": fast / total,
    }


def _peaking_metrics(power_density: np.ndarray) -> dict[str, float]:
    values = np.asarray(power_density, dtype=float)
    maximum = float(np.max(values))
    if maximum <= 0.0:
        return {
            "maximum_W_cm3": 0.0,
            "mean_active_W_cm3": 0.0,
            "max_to_mean_active": 0.0,
        }

    active = values > 1.0e-9 * maximum
    mean_active = float(np.mean(values[active]))
    return {
        "maximum_W_cm3": maximum,
        "mean_active_W_cm3": mean_active,
        "max_to_mean_active": maximum / mean_active,
    }


def _relative_error_metrics(
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float]:
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)

    max_mean = float(np.max(np.abs(mean)))
    if max_mean <= 0.0:
        return {
            "median_relative_error": 0.0,
            "p95_relative_error": 0.0,
            "max_relative_error": 0.0,
        }

    mask = np.abs(mean) > 1.0e-8 * max_mean
    rel = np.zeros_like(mean, dtype=float)
    rel[mask] = np.abs(std[mask] / mean[mask])
    valid = rel[mask]

    return {
        "median_relative_error": float(np.median(valid)),
        "p95_relative_error": float(np.percentile(valid, 95.0)),
        "max_relative_error": float(np.max(valid)),
    }


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads(
        args.power_metadata.read_text(encoding="utf-8")
    )
    source_rate = float(metadata["source_rate_per_s"])
    if source_rate <= 0.0:
        raise ValueError("source_rate_per_s must be positive")

    with openmc.StatePoint(args.statepoint) as sp:
        mesh_tally = sp.get_tally(name="nerva_3d_neutronics")
        heating, mesh = _mesh_score(
            mesh_tally,
            "heating-local",
        )
        heating_std, _ = _mesh_score(
            mesh_tally,
            "heating-local",
            value="std_dev",
        )
        absorption, _ = _mesh_score(
            mesh_tally,
            "absorption",
        )
        nu_fission, _ = _mesh_score(
            mesh_tally,
            "nu-fission",
        )

        dimension = np.asarray(mesh.dimension, dtype=int)
        lower_left = np.asarray(mesh.lower_left, dtype=float)
        upper_right = np.asarray(mesh.upper_right, dtype=float)
        spacing = (upper_right - lower_left) / dimension
        cell_volume_cm3 = float(np.prod(spacing))

        power_density = (
            heating * EV_TO_J * source_rate / cell_volume_cm3
        )
        power_density_std = (
            heating_std * EV_TO_J * source_rate / cell_volume_cm3
        )
        absorption_rate = (
            absorption * source_rate / cell_volume_cm3
        )
        nu_fission_rate = (
            nu_fission * source_rate / cell_volume_cm3
        )

        material_tally = sp.get_tally(name="nerva_material_transport")
        material_filter = material_tally.find_filter(
            openmc.MaterialFilter
        )
        material_bins = list(material_filter.bins)
        material_names = list(_MATERIAL_ORDER[: len(material_bins)])

        material_rows: list[dict[str, float | str | int]] = []
        for score in (
            "flux",
            "absorption",
            "fission",
            "nu-fission",
            "heating",
            "heating-local",
        ):
            mean = _tally_values(
                material_tally,
                score,
            ).reshape(-1)
            std = _tally_values(
                material_tally,
                score,
                value="std_dev",
            ).reshape(-1)

            if len(mean) != len(material_names):
                raise ValueError(
                    f"material tally length {len(mean)} != "
                    f"expected {len(material_names)}"
                )

            for i, name in enumerate(material_names):
                if score in ("heating", "heating-local"):
                    value_scaled = (
                        float(mean[i]) * EV_TO_J * source_rate
                    )
                    std_scaled = (
                        float(std[i]) * EV_TO_J * source_rate
                    )
                    units = "W"
                else:
                    value_scaled = float(mean[i]) * source_rate
                    std_scaled = float(std[i]) * source_rate
                    units = (
                        "particles/s"
                        if score == "flux"
                        else "reactions/s"
                    )

                material_rows.append(
                    {
                        "material": name,
                        "material_bin": int(material_bins[i]),
                        "score": score,
                        "value": value_scaled,
                        "std_dev": std_scaled,
                        "units": units,
                    }
                )

        fuel_spec = sp.get_tally(name="nerva_fuel_spectrum")
        hydrogen_spec = sp.get_tally(
            name="nerva_hydrogen_spectrum"
        )

        energy_filter = fuel_spec.find_filter(openmc.EnergyFilter)
        energy_bins = np.asarray(energy_filter.bins, dtype=float)

        fuel_flux = _tally_values(
            fuel_spec,
            "flux",
        ).reshape(-1) * source_rate
        fuel_absorption = _tally_values(
            fuel_spec,
            "absorption",
        ).reshape(-1) * source_rate
        fuel_fission = _tally_values(
            fuel_spec,
            "fission",
        ).reshape(-1) * source_rate
        fuel_nu_fission = _tally_values(
            fuel_spec,
            "nu-fission",
        ).reshape(-1) * source_rate

        hydrogen_flux = _tally_values(
            hydrogen_spec,
            "flux",
        ).reshape(-1) * source_rate
        hydrogen_absorption = _tally_values(
            hydrogen_spec,
            "absorption",
        ).reshape(-1) * source_rate
        hydrogen_heating = (
            _tally_values(
                hydrogen_spec,
                "heating-local",
            ).reshape(-1)
            * EV_TO_J
            * source_rate
        )

        axial_tally = sp.get_tally(name="nerva_axial_transport")
        axial_heating, axial_mesh = _mesh_score(
            axial_tally,
            "heating-local",
        )
        axial_flux, _ = _mesh_score(
            axial_tally,
            "flux",
        )
        axial_fission, _ = _mesh_score(
            axial_tally,
            "fission",
        )
        axial_nu_fission, _ = _mesh_score(
            axial_tally,
            "nu-fission",
        )

        axial_dimension = np.asarray(
            axial_mesh.dimension,
            dtype=int,
        )
        axial_lower = np.asarray(axial_mesh.lower_left, dtype=float)
        axial_upper = np.asarray(axial_mesh.upper_right, dtype=float)
        axial_spacing = (
            axial_upper - axial_lower
        ) / axial_dimension
        axial_cell_volume = float(np.prod(axial_spacing))

        axial_power = (
            axial_heating.reshape(-1)
            * EV_TO_J
            * source_rate
        )
        axial_flux_rate = (
            axial_flux.reshape(-1)
            * source_rate
            / axial_cell_volume
        )
        axial_fission_rate = (
            axial_fission.reshape(-1)
            * source_rate
            / axial_cell_volume
        )
        axial_nu_fission_rate = (
            axial_nu_fission.reshape(-1)
            * source_rate
            / axial_cell_volume
        )

        z_centers = (
            axial_lower[2]
            + (
                np.arange(axial_dimension[2], dtype=float)
                + 0.5
            )
            * axial_spacing[2]
        )

        try:
            leakage = sp.get_tally(
                name="nerva_vacuum_boundary_current"
            )
        except LookupError:
            leakage_rows = []
        else:
            surface_filter = leakage.find_filter(
                openmc.SurfaceFilter
            )
            surface_bins = list(surface_filter.bins)
            current = _tally_values(
                leakage,
                "current",
            ).reshape(-1) * source_rate
            current_std = _tally_values(
                leakage,
                "current",
                value="std_dev",
            ).reshape(-1) * source_rate

            leakage_rows = [
                {
                    "surface_bin": int(surface_bins[i]),
                    "signed_current_per_s": float(current[i]),
                    "std_dev_per_s": float(current_std[i]),
                }
                for i in range(len(surface_bins))
            ]

        keff_mean, keff_std = _keff_pair(sp)
        entropy = _entropy_history(sp)

    with (args.output / "material_transport.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "material",
                "material_bin",
                "score",
                "value",
                "std_dev",
                "units",
            ],
        )
        writer.writeheader()
        writer.writerows(material_rows)

    with (args.output / "spectra.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "energy_low_eV",
                "energy_high_eV",
                "fuel_flux_per_s",
                "fuel_absorption_per_s",
                "fuel_fission_per_s",
                "fuel_nu_fission_per_s",
                "hydrogen_flux_per_s",
                "hydrogen_absorption_per_s",
                "hydrogen_heating_W",
            )
        )
        for i, (elow, ehigh) in enumerate(energy_bins):
            writer.writerow(
                (
                    elow,
                    ehigh,
                    fuel_flux[i],
                    fuel_absorption[i],
                    fuel_fission[i],
                    fuel_nu_fission[i],
                    hydrogen_flux[i],
                    hydrogen_absorption[i],
                    hydrogen_heating[i],
                )
            )

    with (args.output / "axial_profile.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "z_cm",
                "power_W",
                "flux_cm2_s",
                "fission_rate_cm3_s",
                "nu_fission_rate_cm3_s",
            )
        )
        for i, z_cm in enumerate(z_centers):
            writer.writerow(
                (
                    z_cm,
                    axial_power[i],
                    axial_flux_rate[i],
                    axial_fission_rate[i],
                    axial_nu_fission_rate[i],
                )
            )

    with (args.output / "leakage_current.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "surface_bin",
                "signed_current_per_s",
                "std_dev_per_s",
            ],
        )
        writer.writeheader()
        writer.writerows(leakage_rows)

    np.savez_compressed(
        args.output / "openmc_diagnostics.npz",
        power_density_w_cm3=power_density,
        power_density_std_w_cm3=power_density_std,
        absorption_rate_cm3_s=absorption_rate,
        nu_fission_rate_cm3_s=nu_fission_rate,
        energy_bins_ev=energy_bins,
        fuel_flux_spectrum_per_s=fuel_flux,
        fuel_absorption_spectrum_per_s=fuel_absorption,
        fuel_fission_spectrum_per_s=fuel_fission,
        fuel_nu_fission_spectrum_per_s=fuel_nu_fission,
        hydrogen_flux_spectrum_per_s=hydrogen_flux,
        hydrogen_absorption_spectrum_per_s=hydrogen_absorption,
        hydrogen_heating_spectrum_w=hydrogen_heating,
        z_centers_cm=z_centers,
        axial_power_w=axial_power,
        axial_flux_cm2_s=axial_flux_rate,
        axial_fission_rate_cm3_s=axial_fission_rate,
        axial_nu_fission_rate_cm3_s=axial_nu_fission_rate,
        entropy=np.asarray(entropy, dtype=float),
    )

    material_power = {
        row["material"]: float(row["value"])
        for row in material_rows
        if row["score"] == "heating-local"
    }
    material_power_total = sum(material_power.values())

    axial_positive = axial_power[axial_power > 0.0]
    if axial_positive.size:
        axial_peaking = float(
            np.max(axial_positive) / np.mean(axial_positive)
        )
    else:
        axial_peaking = 0.0

    summary = {
        "statepoint": str(args.statepoint),
        "power_metadata": str(args.power_metadata),
        "source_rate_per_s": source_rate,
        "keff": {
            "mean": keff_mean,
            "std_dev": keff_std,
            "interpretation": (
                "reported diagnostic only; no automatic tuning/calibration"
            ),
        },
        "entropy_history": entropy,
        "entropy_last": None if not entropy else entropy[-1],
        "mesh_peaking": _peaking_metrics(power_density),
        "axial_power_peaking": axial_peaking,
        "mesh_heating_relative_error": _relative_error_metrics(
            heating,
            heating_std,
        ),
        "material_power_W": material_power,
        "material_power_fractions": {
            key: (
                value / material_power_total
                if material_power_total > 0.0
                else 0.0
            )
            for key, value in material_power.items()
        },
        "fuel_flux_spectral_fractions": _spectral_fractions(
            energy_bins,
            fuel_flux,
        ),
        "hydrogen_flux_spectral_fractions": _spectral_fractions(
            energy_bins,
            hydrogen_flux,
        ),
        "vacuum_boundary_current": leakage_rows,
        "mesh": {
            "dimension": [int(v) for v in dimension],
            "lower_left_cm": [float(v) for v in lower_left],
            "upper_right_cm": [float(v) for v in upper_right],
            "cell_volume_cm3": cell_volume_cm3,
        },
        "model_status": (
            "OpenMC diagnostics for the non-calibrated low-enrichment "
            "NERVA-derived demonstrator; not a historical benchmark"
        ),
    }

    (args.output / "openmc_diagnostics.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Advanced NERVA OpenMC diagnostics written:")
    print(f"  k_eff: {keff_mean} +/- {keff_std}")
    print(
        f"  3-D max/mean-active peaking: "
        f"{summary['mesh_peaking']['max_to_mean_active']:.4f}"
    )
    print(f"  axial peaking: {axial_peaking:.4f}")
    print(
        f"  heating p95 relative error: "
        f"{summary['mesh_heating_relative_error']['p95_relative_error']:.4f}"
    )
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
