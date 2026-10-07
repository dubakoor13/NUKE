"""Streamlit dashboard for NERVA-derived analysis results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .engine_performance import (
    combine_outlet_streams,
    ideal_nozzle_performance,
)


def _read_json_upload(uploaded) -> dict[str, Any] | None:
    if uploaded is None:
        return None
    return json.loads(uploaded.getvalue().decode("utf-8"))


def _synthetic_demo() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = {
        "target_power_W": 1.0e6,
        "fuel_power_W": 0.70e6,
        "fuel_power_fraction": 0.70,
        "tie_power_W": 0.05e6,
        "tie_power_fraction": 0.05,
        "source_rate_per_s": None,
        "model_note": "Synthetic CI coupling field; not an OpenMC transport result.",
    }
    fuel = {
        "fuel_elements": 42,
        "fuel_channels": 798,
        "fuel_mass_flow_kg_s": 2.0,
        "fuel_power_W": 0.70e6,
        "inlet_temperature_K": 500.0,
        "outlet_temperature_K": 524.475524,
        "inlet_pressure_Pa": 8.0e6,
        "outlet_pressure_Pa": 7.95e6,
        "maximum_coolant_wall_temperature_K": 540.0,
        "maximum_fuel_surface_temperature_K": 550.0,
        "maximum_peak_fuel_temperature_K": 565.0,
        "hydrogen_property_model": "constant",
    }
    tie = {
        "tie_tubes": 19,
        "tie_mass_flow_kg_s": 0.40,
        "total_tie_power_W": 0.05e6,
        "inlet_temperature_K": 500.0,
        "outlet_temperature_K": 508.741259,
        "inlet_pressure_Pa": 8.0e6,
        "outlet_pressure_Pa": 7.90e6,
        "max_supply_wall_temperature_K": 520.0,
        "max_return_wall_temperature_K": 525.0,
        "model": "constant",
    }
    return metadata, fuel, tie


def _pick(mapping: dict[str, Any], key: str, default: float) -> float:
    value = mapping.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _metric(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def main() -> None:
    st.set_page_config(
        page_title="NUKE / NERVA Demo",
        page_icon="🚀",
        layout="wide",
    )

    st.title("NUKE — NERVA-derived run dashboard")
    st.caption(
        "OpenMC → deposited power → H₂ thermal paths → ideal-nozzle performance estimate"
    )

    with st.sidebar:
        st.header("Result source")
        mode = st.radio(
            "Mode",
            ("Synthetic demo", "Upload run outputs"),
            help=(
                "Synthetic demo uses the repository's deterministic CI example. "
                "Upload mode reads your analysis JSON outputs."
            ),
        )

        if mode == "Upload run outputs":
            metadata_upload = st.file_uploader(
                "metadata.json",
                type=["json"],
                key="metadata",
            )
            fuel_upload = st.file_uploader(
                "thermal_summary.json",
                type=["json"],
                key="fuel",
            )
            tie_upload = st.file_uploader(
                "tie_thermal_summary.json",
                type=["json"],
                key="tie",
            )

            metadata = _read_json_upload(metadata_upload)
            fuel = _read_json_upload(fuel_upload)
            tie = _read_json_upload(tie_upload)

            ready = metadata is not None and fuel is not None and tie is not None
            if not ready:
                st.info("Upload all three JSON files to compute run metrics.")
                metadata, fuel, tie = _synthetic_demo()
                using_fallback = True
            else:
                using_fallback = False
        else:
            metadata, fuel, tie = _synthetic_demo()
            using_fallback = True

        st.header("Nozzle assumptions")
        gamma = st.slider(
            "γ",
            min_value=1.10,
            max_value=1.50,
            value=1.35,
            step=0.01,
        )
        nozzle_efficiency = st.slider(
            "Nozzle efficiency",
            min_value=0.70,
            max_value=1.00,
            value=0.95,
            step=0.01,
        )
        exit_pressure_kpa = st.number_input(
            "Exit pressure [kPa]",
            min_value=0.1,
            value=1.0,
            step=0.1,
        )
        ambient_pressure_kpa = st.number_input(
            "Ambient pressure [kPa]",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

    if using_fallback:
        st.warning(
            "DEMO MODE: these are synthetic CI values, not a completed OpenMC "
            "transport solution and not validated NERVA performance."
        )
    else:
        st.info(
            "Loaded-run mode: thermal values come from your analysis outputs. "
            "Thrust/Isp are still ideal-nozzle estimates, not a historical "
            "NERVA validation."
        )

    fuel_flow = _pick(fuel, "fuel_mass_flow_kg_s", 2.0)
    tie_flow = _pick(tie, "tie_mass_flow_kg_s", 0.4)
    fuel_tout = _pick(fuel, "outlet_temperature_K", 500.0)
    tie_tout = _pick(tie, "outlet_temperature_K", 500.0)
    fuel_pout = _pick(fuel, "outlet_pressure_Pa", 8.0e6)
    tie_pout = _pick(tie, "outlet_pressure_Pa", 8.0e6)

    mixed = combine_outlet_streams(
        fuel_mass_flow_kg_s=fuel_flow,
        fuel_temperature_k=fuel_tout,
        fuel_pressure_pa=fuel_pout,
        tie_mass_flow_kg_s=tie_flow,
        tie_temperature_k=tie_tout,
        tie_pressure_pa=tie_pout,
    )

    performance = ideal_nozzle_performance(
        mass_flow_kg_s=mixed.mass_flow_kg_s,
        chamber_temperature_k=mixed.temperature_k,
        chamber_pressure_pa=mixed.pressure_pa,
        exit_pressure_pa=exit_pressure_kpa * 1000.0,
        ambient_pressure_pa=ambient_pressure_kpa * 1000.0,
        gamma=gamma,
        nozzle_efficiency=nozzle_efficiency,
    )

    thermal_power_w = _pick(
        metadata,
        "target_power_W",
        _pick(fuel, "fuel_power_W", 0.0)
        + _pick(tie, "total_tie_power_W", 0.0),
    )
    fuel_power_w = _pick(
        metadata,
        "fuel_power_W",
        _pick(fuel, "fuel_power_W", 0.0),
    )
    tie_power_w = _pick(
        metadata,
        "tie_power_W",
        _pick(tie, "total_tie_power_W", 0.0),
    )

    st.subheader("Engine performance")
    row1 = st.columns(5)
    with row1[0]:
        _metric("Thermal power", f"{thermal_power_w / 1.0e6:,.3f} MW")
    with row1[1]:
        _metric("H₂ mass flow", f"{mixed.mass_flow_kg_s:,.3f} kg/s")
    with row1[2]:
        _metric("Mixed chamber T", f"{mixed.temperature_k:,.1f} K")
    with row1[3]:
        _metric("Chamber pressure", f"{mixed.pressure_pa / 1.0e6:,.3f} MPa")
    with row1[4]:
        _metric("Thrust", f"{performance.thrust_n / 1000.0:,.2f} kN")

    row2 = st.columns(5)
    with row2[0]:
        _metric("Vacuum-style Isp", f"{performance.isp_s:,.1f} s")
    with row2[1]:
        _metric(
            "Effective exhaust velocity",
            f"{performance.effective_exhaust_velocity_m_s / 1000.0:,.3f} km/s",
        )
    with row2[2]:
        _metric("Exit Mach", f"{performance.exit_mach:,.2f}")
    with row2[3]:
        _metric("Expansion ratio Ae/At", f"{performance.expansion_ratio:,.1f}")
    with row2[4]:
        exit_diameter = math.sqrt(4.0 * performance.exit_area_m2 / math.pi)
        _metric("Equivalent exit diameter", f"{exit_diameter:,.3f} m")

    st.subheader("Thermal split")
    split_df = pd.DataFrame(
        {
            "Path": ["Fuel elements", "Tie-tube solids", "Other deposition"],
            "Power MW": [
                fuel_power_w / 1.0e6,
                tie_power_w / 1.0e6,
                max(0.0, thermal_power_w - fuel_power_w - tie_power_w) / 1.0e6,
            ],
        }
    )
    st.bar_chart(split_df.set_index("Path"))

    st.subheader("Hydrogen outlet state")
    state_df = pd.DataFrame(
        {
            "Path": ["Fuel channels", "Tie-tube return", "Mixed"],
            "Mass flow kg/s": [fuel_flow, tie_flow, mixed.mass_flow_kg_s],
            "Temperature K": [fuel_tout, tie_tout, mixed.temperature_k],
            "Pressure MPa": [
                fuel_pout / 1.0e6,
                tie_pout / 1.0e6,
                mixed.pressure_pa / 1.0e6,
            ],
        }
    )
    st.dataframe(state_df, use_container_width=True, hide_index=True)

    st.subheader("Temperature limits / screening values")
    temperature_df = pd.DataFrame(
        {
            "Metric": [
                "Fuel-channel outlet H₂",
                "Maximum coolant wall",
                "Maximum fuel surface",
                "Estimated peak fuel",
                "Tie-tube return outlet H₂",
                "Maximum tie supply wall",
                "Maximum tie return wall",
            ],
            "Temperature K": [
                fuel_tout,
                _pick(fuel, "maximum_coolant_wall_temperature_K", fuel_tout),
                _pick(fuel, "maximum_fuel_surface_temperature_K", fuel_tout),
                _pick(fuel, "maximum_peak_fuel_temperature_K", fuel_tout),
                tie_tout,
                _pick(tie, "max_supply_wall_temperature_K", tie_tout),
                _pick(tie, "max_return_wall_temperature_K", tie_tout),
            ],
        }
    )
    st.bar_chart(temperature_df.set_index("Metric"))

    with st.expander("Nozzle details"):
        nozzle_df = pd.DataFrame(
            {
                "Quantity": [
                    "Momentum thrust",
                    "Pressure thrust",
                    "Throat area",
                    "Exit area",
                    "Exit velocity",
                    "Effective exhaust velocity",
                    "Exit pressure",
                    "Ambient pressure",
                    "Gamma",
                    "Nozzle efficiency",
                ],
                "Value": [
                    f"{performance.momentum_thrust_n:,.3f} N",
                    f"{performance.pressure_thrust_n:,.3f} N",
                    f"{performance.throat_area_m2:.6f} m²",
                    f"{performance.exit_area_m2:.6f} m²",
                    f"{performance.exit_velocity_m_s:,.3f} m/s",
                    f"{performance.effective_exhaust_velocity_m_s:,.3f} m/s",
                    f"{exit_pressure_kpa:,.3f} kPa",
                    f"{ambient_pressure_kpa:,.3f} kPa",
                    f"{gamma:.3f}",
                    f"{nozzle_efficiency:.3f}",
                ],
            }
        )
        st.dataframe(nozzle_df, use_container_width=True, hide_index=True)

    st.caption(
        "Thrust/Isp use a choked calorically-perfect-gas nozzle estimate. "
        "For production analysis replace this layer with equilibrium/frozen "
        "CEA/Cantera nozzle chemistry and validated reactor/thermal inputs."
    )


if __name__ == "__main__":
    main()
