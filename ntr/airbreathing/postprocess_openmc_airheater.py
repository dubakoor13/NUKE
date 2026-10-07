"""Postprocess a generic OpenMC air-heater fixed-source statepoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openmc


EV_TO_J = 1.602176634e-19


def _mesh_score(tally: openmc.Tally, score: str, value: str = "mean") -> np.ndarray:
    mesh_filter = tally.find_filter(openmc.MeshFilter)
    mesh_shape = tuple(int(v) for v in mesh_filter.mesh.dimension)

    sliced = tally.get_slice(scores=[score])
    data = np.asarray(
        sliced.get_reshaped_data(value=value, expand_dims=True),
        dtype=float,
    )

    if data.shape[-2:] != (1, 1):
        raise ValueError(
            f"unexpected trailing tally dimensions {data.shape[-2:]}"
        )

    data = data[..., 0, 0]
    if data.shape[: len(mesh_shape)] != mesh_shape:
        raise ValueError(
            f"mesh prefix {data.shape[:len(mesh_shape)]} != {mesh_shape}"
        )

    if data.ndim > len(mesh_shape):
        data = np.sum(
            data,
            axis=tuple(range(len(mesh_shape), data.ndim)),
        )

    return data


def _region_score(
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
    return np.asarray(data[..., 0, 0], dtype=float).reshape(-1)


def _spectrum_score(
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
    return np.squeeze(data[..., 0, 0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize/postprocess a generic fixed-source OpenMC "
            "nuclear-air-heater statepoint."
        )
    )
    parser.add_argument("statepoint", type=Path)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--source-rate-s",
        type=float,
        help="External source intensity [particles/s].",
    )
    group.add_argument(
        "--target-deposited-power-mw",
        type=float,
        help=(
            "Normalize the total OpenMC deposited heating to this target "
            "power [MW]. This is a scaling operation, not a criticality solve."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/openmc_airheater_post"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.source_rate_s is not None and args.source_rate_s <= 0.0:
        raise ValueError("--source-rate-s must be positive")
    if (
        args.target_deposited_power_mw is not None
        and args.target_deposited_power_mw <= 0.0
    ):
        raise ValueError("--target-deposited-power-mw must be positive")

    with openmc.StatePoint(args.statepoint) as sp:
        mesh_tally = sp.get_tally(name="airheater_3d_transport")
        region_tally = sp.get_tally(name="airheater_region_transport")
        fuel_spec = sp.get_tally(name="airheater_fuel_spectrum")
        air_spec = sp.get_tally(name="airheater_air_spectrum")
        leakage = sp.get_tally(name="airheater_boundary_current")

        mesh_filter = mesh_tally.find_filter(openmc.MeshFilter)
        mesh = mesh_filter.mesh
        if not isinstance(mesh, openmc.RegularMesh):
            raise TypeError("airheater_3d_transport must use RegularMesh")

        dimension = np.asarray(mesh.dimension, dtype=int)
        lower_left = np.asarray(mesh.lower_left, dtype=float)
        upper_right = np.asarray(mesh.upper_right, dtype=float)
        spacing = (upper_right - lower_left) / dimension
        cell_volume_cm3 = float(np.prod(spacing))

        heating = _mesh_score(mesh_tally, "heating")
        heating_std = _mesh_score(mesh_tally, "heating", value="std_dev")
        flux = _mesh_score(mesh_tally, "flux")
        absorption = _mesh_score(mesh_tally, "absorption")
        fission = _mesh_score(mesh_tally, "fission")
        nu_fission = _mesh_score(mesh_tally, "nu-fission")

        total_heating_ev_per_source = float(np.sum(heating))
        if total_heating_ev_per_source <= 0.0:
            raise ValueError("statepoint has non-positive total heating")

        if args.source_rate_s is not None:
            source_rate_s = float(args.source_rate_s)
            deposited_power_w = (
                total_heating_ev_per_source * EV_TO_J * source_rate_s
            )
            normalization_mode = "source_rate"
        else:
            deposited_power_w = (
                float(args.target_deposited_power_mw) * 1.0e6
            )
            source_rate_s = deposited_power_w / (
                total_heating_ev_per_source * EV_TO_J
            )
            normalization_mode = "target_deposited_power"

        power_density = (
            heating * EV_TO_J * source_rate_s / cell_volume_cm3
        )
        power_density_std = (
            heating_std * EV_TO_J * source_rate_s / cell_volume_cm3
        )
        flux_rate = flux * source_rate_s / cell_volume_cm3
        absorption_rate = absorption * source_rate_s / cell_volume_cm3
        fission_rate = fission * source_rate_s / cell_volume_cm3
        nu_fission_rate = nu_fission * source_rate_s / cell_volume_cm3

        region_heating = _region_score(region_tally, "heating")
        region_heating_std = _region_score(
            region_tally,
            "heating",
            value="std_dev",
        )
        if region_heating.size != 3:
            raise ValueError(
                "expected fuel/air/vessel bins in region tally"
            )

        region_names = ("fuel", "air", "vessel")
        region_power = {
            name: float(region_heating[i] * EV_TO_J * source_rate_s)
            for i, name in enumerate(region_names)
        }
        region_power_std = {
            name: float(
                region_heating_std[i] * EV_TO_J * source_rate_s
            )
            for i, name in enumerate(region_names)
        }

        fuel_energy_filter = fuel_spec.find_filter(openmc.EnergyFilter)
        fuel_energy_bins = np.asarray(
            fuel_energy_filter.bins,
            dtype=float,
        )

        fuel_spectrum_flux = _spectrum_score(
            fuel_spec,
            "flux",
        ) * source_rate_s
        fuel_spectrum_absorption = _spectrum_score(
            fuel_spec,
            "absorption",
        ) * source_rate_s
        fuel_spectrum_fission = _spectrum_score(
            fuel_spec,
            "fission",
        ) * source_rate_s
        fuel_spectrum_nu_fission = _spectrum_score(
            fuel_spec,
            "nu-fission",
        ) * source_rate_s

        air_spectrum_flux = _spectrum_score(
            air_spec,
            "flux",
        ) * source_rate_s
        air_spectrum_absorption = _spectrum_score(
            air_spec,
            "absorption",
        ) * source_rate_s
        air_spectrum_heating = _spectrum_score(
            air_spec,
            "heating-local",
        ) * EV_TO_J * source_rate_s

        leakage_mean = np.asarray(
            leakage.get_reshaped_data(
                value="mean",
                expand_dims=True,
            ),
            dtype=float,
        )
        leakage_std = np.asarray(
            leakage.get_reshaped_data(
                value="std_dev",
                expand_dims=True,
            ),
            dtype=float,
        )
        leakage_rate = np.squeeze(leakage_mean) * source_rate_s
        leakage_rate_std = np.squeeze(leakage_std) * source_rate_s

    np.savez_compressed(
        args.output / "openmc_airheater_fields.npz",
        power_density_w_cm3=power_density,
        power_density_std_w_cm3=power_density_std,
        flux_cm2_s=flux_rate,
        absorption_rate_cm3_s=absorption_rate,
        fission_rate_cm3_s=fission_rate,
        nu_fission_rate_cm3_s=nu_fission_rate,
        lower_left_cm=lower_left,
        upper_right_cm=upper_right,
        dimension=dimension,
        energy_bins_ev=fuel_energy_bins,
        fuel_spectrum_flux_per_s=fuel_spectrum_flux,
        fuel_spectrum_absorption_per_s=fuel_spectrum_absorption,
        fuel_spectrum_fission_per_s=fuel_spectrum_fission,
        fuel_spectrum_nu_fission_per_s=fuel_spectrum_nu_fission,
        air_spectrum_flux_per_s=air_spectrum_flux,
        air_spectrum_absorption_per_s=air_spectrum_absorption,
        air_spectrum_heating_w=air_spectrum_heating,
        boundary_current_per_s=leakage_rate,
        boundary_current_std_per_s=leakage_rate_std,
    )

    region_sum = sum(region_power.values())
    metadata = {
        "statepoint": str(args.statepoint),
        "normalization_mode": normalization_mode,
        "source_rate_per_s": source_rate_s,
        "deposited_power_W": deposited_power_w,
        "mesh_cell_volume_cm3": cell_volume_cm3,
        "mesh_dimension": [int(v) for v in dimension],
        "mesh_lower_left_cm": [float(v) for v in lower_left],
        "mesh_upper_right_cm": [float(v) for v in upper_right],
        "region_power_W": region_power,
        "region_power_std_W": region_power_std,
        "region_power_sum_W": region_sum,
        "region_power_fractions": {
            name: value / region_sum
            for name, value in region_power.items()
        },
        "field_units": {
            "power_density": "W/cm3",
            "flux": "particles/cm2/s",
            "absorption_rate": "reactions/cm3/s",
            "fission_rate": "reactions/cm3/s",
            "nu_fission_rate": "neutrons/cm3/s",
            "energy_bins": "eV",
            "boundary_current": "particles/s",
        },
        "model_status": (
            "generic fixed-source OpenMC demonstrator; no k-effective "
            "or weapon-specific design inference"
        ),
    }

    (args.output / "openmc_airheater_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Generic OpenMC air-heater postprocessing complete:")
    print(f"  source rate: {source_rate_s:.6e} 1/s")
    print(f"  deposited power: {deposited_power_w / 1.0e6:.6f} MW")
    for name in region_names:
        print(
            f"  {name} heating: "
            f"{region_power[name] / 1.0e6:.6f} MW"
        )
    print(f"  output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
