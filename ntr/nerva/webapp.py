"""Streamlit dashboard for NERVA-derived and historical reference results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from .engine_performance import (
    G0_M_S2,
    combine_outlet_streams,
    ideal_nozzle_performance,
)
from .historical_presets import PRESETS, HistoricalNervaPreset
from .lantr import (
    BASE_CHAMBER_PRESSURE_PA,
    BASE_ISP_S,
    BASE_NOZZLE_AREA_RATIO,
    BASE_REACTOR_EXIT_TEMPERATURE_K,
    BASE_REACTOR_POWER_MW,
    BIMODAL_POWER,
    NASA_BIMODAL_SOURCE,
    NASA_BIMODAL_URL,
    NASA_LANTR_SOURCE,
    NASA_LANTR_URL,
    HIGH_PRESSURE_LANTR_CHAMBER_PRESSURE_PA,
    HIGH_PRESSURE_LANTR_NOZZLE_AREA_RATIO,
    HIGH_PRESSURE_LANTR_BASE_THRUST_N,
    HIGH_PRESSURE_LANTR_TEMPERATURE_RANGE_K,
    NASA_HIGH_PRESSURE_LANTR_SOURCE,
    NASA_HIGH_PRESSURE_LANTR_URL,
    coupled_high_pressure_lantr_point,
    coupled_lantr_point,
    high_pressure_lantr_isp,
    lantr_point,
    lantr_table,
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


def _format_range(values: tuple[float, float] | None, unit: str) -> str:
    if values is None:
        return "—"
    return f"{values[0]:,.0f}–{values[1]:,.0f} {unit}"


def _render_historical(preset: HistoricalNervaPreset) -> None:
    st.success(f"{preset.category}: {preset.name}")
    st.write(preset.description)

    row1 = st.columns(5)
    with row1[0]:
        _metric(
            "Thermal power",
            "—" if preset.thermal_power_mw is None else f"{preset.thermal_power_mw:,.0f} MW",
        )
    with row1[1]:
        _metric(
            "Thrust",
            "—" if preset.thrust_n is None else f"{preset.thrust_n / 1000.0:,.1f} kN",
        )
    with row1[2]:
        if preset.isp_s is not None:
            isp_text = f"{preset.isp_s:,.0f} s"
        elif preset.isp_range_s is not None:
            isp_text = _format_range(preset.isp_range_s, "s")
        else:
            isp_text = "—"
        _metric("Specific impulse", isp_text)
    with row1[3]:
        if preset.chamber_temperature_k is not None:
            t_text = f"{preset.chamber_temperature_k:,.0f} K"
        elif preset.chamber_temperature_range_k is not None:
            t_text = _format_range(preset.chamber_temperature_range_k, "K")
        else:
            t_text = "—"
        _metric("Chamber temperature", t_text)
    with row1[4]:
        _metric(
            "Chamber pressure",
            "—" if preset.chamber_pressure_pa is None else f"{preset.chamber_pressure_pa / 1.0e6:,.3f} MPa",
        )

    row2 = st.columns(5)
    with row2[0]:
        _metric(
            "H₂ mass flow",
            "—" if preset.hydrogen_mass_flow_kg_s is None else f"{preset.hydrogen_mass_flow_kg_s:,.2f} kg/s",
        )
    with row2[1]:
        _metric(
            "Nozzle Ae/At",
            "—" if preset.nozzle_expansion_ratio is None else f"{preset.nozzle_expansion_ratio:,.0f}",
        )
    with row2[2]:
        _metric(
            "Fuel elements",
            "—" if preset.fuel_element_count is None else f"{preset.fuel_element_count:,}",
        )
    with row2[3]:
        _metric(
            "Core diameter",
            "—" if preset.core_diameter_cm is None else f"{preset.core_diameter_cm / 100.0:,.3f} m",
        )
    with row2[4]:
        _metric(
            "Active length",
            "—" if preset.active_length_cm is None else f"{preset.active_length_cm / 100.0:,.3f} m",
        )

    if preset.isp_s is not None:
        effective_velocity = preset.isp_s * G0_M_S2
        st.caption(
            f"Reference effective exhaust velocity from published Isp: "
            f"{effective_velocity / 1000.0:.3f} km/s"
        )

    st.subheader("Published geometry / reactor values")
    rows: list[tuple[str, str]] = []
    if preset.fuel_element_flat_to_flat_cm is not None:
        rows.append(("Fuel element across flats", f"{preset.fuel_element_flat_to_flat_cm:.4f} cm"))
    if preset.coolant_channels_per_element is not None:
        rows.append(("Coolant channels / fuel element", str(preset.coolant_channels_per_element)))
    if preset.coolant_channel_diameter_cm is not None:
        rows.append(("Coolant channel diameter", f"{preset.coolant_channel_diameter_cm:.4f} cm"))
    if preset.core_diameter_cm is not None:
        rows.append(("Core diameter", f"{preset.core_diameter_cm:.2f} cm"))
    if preset.active_length_cm is not None:
        rows.append(("Active/core length", f"{preset.active_length_cm:.2f} cm"))
    if preset.tie_heating_fraction_range is not None:
        rows.append(
            (
                "Tie-tube heat deposition",
                f"{100*preset.tie_heating_fraction_range[0]:.0f}–"
                f"{100*preset.tie_heating_fraction_range[1]:.0f}%",
            )
        )
    if preset.reflector_heating_fraction_range is not None:
        rows.append(
            (
                "Reflector heat deposition",
                f"{100*preset.reflector_heating_fraction_range[0]:.0f}–"
                f"{100*preset.reflector_heating_fraction_range[1]:.0f}%",
            )
        )
    if rows:
        st.dataframe(
            pd.DataFrame(rows, columns=["Parameter", "Published value"]),
            hide_index=True,
            use_container_width=True,
        )

    st.subheader("Source")
    st.markdown(f"**{preset.source_title}**")
    st.markdown(f"[NASA NTRS source]({preset.source_url})")
    st.caption(preset.source_note)

    st.warning(
        "Historical reference mode displays published NASA system-level values. "
        "It does not supply fissile loading or an exact criticality recipe."
    )



def _render_lantr(
    oxygen_hydrogen_ratio: float,
    architecture: str,
    chamber_pressure_mpa: float,
    reactor_exit_temperature_k: float,
) -> None:
    high_pressure = architecture == "2000-psia trimodal LANTR"

    if high_pressure:
        point = coupled_high_pressure_lantr_point(
            oxygen_hydrogen_ratio,
            chamber_pressure_mpa * 1.0e6,
            reactor_exit_temperature_k,
        )
        nozzle_ratio = HIGH_PRESSURE_LANTR_NOZZLE_AREA_RATIO
        pressure_anchor_mpa = (
            HIGH_PRESSURE_LANTR_CHAMBER_PRESSURE_PA / 1.0e6
        )
        reference_name = "NASA 2000-psia trimodal LANTR"
    else:
        point = coupled_lantr_point(
            oxygen_hydrogen_ratio,
            chamber_pressure_mpa * 1.0e6,
            reactor_exit_temperature_k,
        )
        nozzle_ratio = BASE_NOZZLE_AREA_RATIO
        pressure_anchor_mpa = BASE_CHAMBER_PRESSURE_PA / 1.0e6
        reference_name = "NASA 1000-psia small-engine LANTR"

    st.success("Bimodal NTR + LOX-augmented LANTR")
    st.write(
        "Pressure is now a coupled operating variable, not a fixed label. "
        "For fixed throat geometry, choked H2 flow scales approximately as "
        "Pc/sqrt(Tc). LOX flow is MR times H2 flow, and thrust is computed "
        "from total mass flow times delivered Isp."
    )

    row1 = st.columns(5)
    with row1[0]:
        _metric("Chamber / nozzle-inlet pressure", f"{chamber_pressure_mpa:,.3f} MPa")
    with row1[1]:
        _metric("O/H mixture ratio", f"{oxygen_hydrogen_ratio:.2f}")
    with row1[2]:
        _metric("Thrust", f"{point.thrust_n / 1000.0:,.1f} kN")
    with row1[3]:
        _metric("Delivered Isp", f"{point.delivered_isp_s:,.0f} s")
    with row1[4]:
        _metric("Thrust augmentation", f"{point.thrust_augmentation_factor:.3f}x")

    row2 = st.columns(5)
    with row2[0]:
        _metric("H2 flow", f"{point.hydrogen_mass_flow_kg_s:,.2f} kg/s")
    with row2[1]:
        _metric("O2 flow", f"{point.oxygen_mass_flow_kg_s:,.2f} kg/s")
    with row2[2]:
        _metric("Total propellant flow", f"{point.total_mass_flow_kg_s:,.2f} kg/s")
    with row2[3]:
        _metric("Hot-H2 temperature", f"{reactor_exit_temperature_k:,.0f} K")
    with row2[4]:
        _metric("Nozzle Ae/At", f"{nozzle_ratio:,.0f}:1")

    st.caption(
        f"Reference anchor: {reference_name} at {pressure_anchor_mpa:.3f} MPa. "
        "Moving the pressure slider keeps the reference throat geometry fixed, "
        "so mass flow and thrust change with pressure."
    )

    st.subheader("Pressure -> flow -> thrust coupling")
    pressure_samples = [
        6.895,
        10.0,
        13.79,
        17.5,
        20.7,
        25.0,
    ]
    rows = []
    for pressure in pressure_samples:
        if high_pressure:
            sample = coupled_high_pressure_lantr_point(
                oxygen_hydrogen_ratio,
                pressure * 1.0e6,
                reactor_exit_temperature_k,
            )
        else:
            sample = coupled_lantr_point(
                min(oxygen_hydrogen_ratio, 5.0),
                pressure * 1.0e6,
                reactor_exit_temperature_k,
            )
        rows.append(
            {
                "Pc MPa": pressure,
                "H2 kg/s": sample.hydrogen_mass_flow_kg_s,
                "O2 kg/s": sample.oxygen_mass_flow_kg_s,
                "Total kg/s": sample.total_mass_flow_kg_s,
                "Thrust kN": sample.thrust_n / 1000.0,
                "Isp s": sample.delivered_isp_s,
            }
        )
    pressure_df = pd.DataFrame(rows)
    st.dataframe(pressure_df, hide_index=True, use_container_width=True)
    st.line_chart(
        pressure_df.set_index("Pc MPa")[["Thrust kN", "Total kg/s"]],
        height=320,
    )

    if high_pressure:
        high_table = pd.DataFrame(
            {
                "O/H MR": [0, 1, 3, 5, 7],
                "Isp s (5 h / 2900 K)": [941, 772, 647, 576, 514],
                "Reference engine T/W": [3.0, 4.8, 8.2, 11.0, 13.1],
            }
        )
        st.subheader("Published 2000-psia trimodal LANTR map")
        st.dataframe(high_table, hide_index=True, use_container_width=True)
    else:
        table = lantr_table()
        trade = pd.DataFrame(
            {
                "O/H MR": [p.oxygen_hydrogen_ratio for p in table],
                "Thrust kN at 6.895 MPa": [p.thrust_n / 1000.0 for p in table],
                "Isp s": [p.delivered_isp_s for p in table],
                "Thrust augmentation": [
                    p.thrust_augmentation_factor for p in table
                ],
            }
        )
        st.subheader("Published 1000-psia LANTR map")
        st.dataframe(trade, hide_index=True, use_container_width=True)

    st.subheader("Bimodal electrical-power mode")
    power_row = st.columns(4)
    with power_row[0]:
        _metric(
            "Electrical power / engine",
            f"{BIMODAL_POWER.electric_power_kwe_per_engine:.0f} kWe",
        )
    with power_row[1]:
        _metric(
            "Reference stage power",
            f"{BIMODAL_POWER.reference_stage_power_kwe:.0f} kWe",
        )
    with power_row[2]:
        _metric(
            "Idle reactor thermal power",
            f"{BIMODAL_POWER.idle_reactor_thermal_power_kwt:.0f} kWt",
        )
    with power_row[3]:
        _metric(
            "Reference conversion efficiency",
            f"{100.0 * BIMODAL_POWER.conversion_efficiency:.0f}%",
        )

    st.info(
        "Classic LANTR still injects oxygen downstream of the choked throat. "
        "This pressure sweep changes the upstream NTR stagnation/chamber state "
        "and therefore the choked hydrogen mass flow through a fixed throat."
    )

    st.subheader("Public NASA sources")
    st.markdown(f"**LANTR:** {NASA_LANTR_SOURCE}")
    st.markdown(f"[NASA LANTR source]({NASA_LANTR_URL})")
    st.markdown(f"**High-pressure LANTR:** {NASA_HIGH_PRESSURE_LANTR_SOURCE}")
    st.markdown(
        f"[NASA high-pressure LANTR source]({NASA_HIGH_PRESSURE_LANTR_URL})"
    )
    st.markdown(f"**Bimodal power:** {NASA_BIMODAL_SOURCE}")
    st.markdown(f"[NASA bimodal source]({NASA_BIMODAL_URL})")


def _render_calculated(
    metadata: dict[str, Any],
    fuel: dict[str, Any],
    tie: dict[str, Any],
    using_fallback: bool,
    gamma: float,
    nozzle_efficiency: float,
    exit_pressure_kpa: float,
    ambient_pressure_kpa: float,
) -> None:
    if using_fallback:
        st.warning(
            "DEMO MODE: these are synthetic CI values, not a completed OpenMC "
            "transport solution and not validated NERVA performance."
        )
    else:
        st.info(
            "Loaded-run mode: thermal values come from your analysis outputs. "
            "Thrust/Isp are ideal-nozzle estimates and should be compared with "
            "the historical-reference tab."
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

    st.subheader("Calculated engine performance")
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

    st.subheader("Historical comparison")
    comparison_rows = []
    for preset in PRESETS.values():
        if preset.thrust_n is None and preset.isp_s is None and preset.isp_range_s is None:
            continue
        if preset.isp_s is not None:
            isp_ref = f"{preset.isp_s:.0f}"
        elif preset.isp_range_s is not None:
            isp_ref = f"{preset.isp_range_s[0]:.0f}–{preset.isp_range_s[1]:.0f}"
        else:
            isp_ref = "—"
        comparison_rows.append(
            {
                "Case": preset.name,
                "Thrust kN": "—" if preset.thrust_n is None else f"{preset.thrust_n/1000.0:.1f}",
                "Isp s": isp_ref,
                "Thermal MW": "—" if preset.thermal_power_mw is None else f"{preset.thermal_power_mw:.0f}",
            }
        )
    comparison_rows.insert(
        0,
        {
            "Case": "Current calculated/demo result",
            "Thrust kN": f"{performance.thrust_n/1000.0:.1f}",
            "Isp s": f"{performance.isp_s:.1f}",
            "Thermal MW": f"{thermal_power_w/1.0e6:.3f}",
        },
    )
    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, use_container_width=True)

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


def main() -> None:
    st.set_page_config(
        page_title="NUKE / NERVA Dashboard",
        page_icon="🚀",
        layout="wide",
    )

    st.title("NUKE — NERVA reference and run dashboard")
    st.caption(
        "Historical NASA references alongside OpenMC → thermal → nozzle calculations"
    )

    with st.sidebar:
        st.header("Result source")
        mode = st.radio(
            "Mode",
            (
                "Historical NERVA reference",
                "Bimodal NTR + LANTR",
                "Synthetic demo",
                "Auto-load analysis directory",
                "Upload run outputs",
            ),
        )

        historical_preset = None
        lantr_mr = None
        lantr_architecture = None
        lantr_pressure_mpa = None
        lantr_temperature_k = None
        if mode == "Historical NERVA reference":
            preset_keys = list(PRESETS)
            selected = st.selectbox(
                "Historical configuration",
                preset_keys,
                format_func=lambda key: PRESETS[key].name,
            )
            historical_preset = PRESETS[selected]
            metadata = fuel = tie = {}
            using_fallback = False
        elif mode == "Bimodal NTR + LANTR":
            lantr_architecture = st.selectbox(
                "LANTR reference architecture",
                ("1000-psia small-engine LANTR", "2000-psia trimodal LANTR"),
                index=1,
            )
            high_pressure = lantr_architecture == "2000-psia trimodal LANTR"
            max_mr = 7.0 if high_pressure else 5.0
            default_pressure = 13.789515 if high_pressure else 6.894757
            default_temperature = 2900.0 if high_pressure else BASE_REACTOR_EXIT_TEMPERATURE_K
            lantr_pressure_mpa = st.slider(
                "Chamber / nozzle-inlet pressure [MPa]",
                min_value=5.0,
                max_value=25.0,
                value=float(default_pressure),
                step=0.25,
                key=f"lantr_pc_{lantr_architecture}",
            )
            lantr_temperature_k = st.slider(
                "Hot-H2 reactor outlet temperature [K]",
                min_value=2400.0,
                max_value=3100.0,
                value=float(default_temperature),
                step=25.0,
                key=f"lantr_tc_{lantr_architecture}",
            )
            lantr_mr = st.slider(
                "LOX augmentation O/H mixture ratio",
                min_value=0.0,
                max_value=max_mr,
                value=0.0,
                step=0.1,
                help=(
                    "NASA reference table spans MR 0 to 5. MR=0 is pure "
                    "LH2 NTR; increasing MR adds downstream oxygen afterburning."
                ),
            )
            metadata = fuel = tie = {}
            using_fallback = False
        elif mode == "Auto-load analysis directory":
            analysis_root = Path(
                st.text_input(
                    "Analysis directory",
                    value="build/nerva_analysis",
                )
            )
            metadata_path = analysis_root / "power" / "metadata.json"
            fuel_path = analysis_root / "fuel" / "thermal_summary.json"
            tie_path = analysis_root / "tie" / "tie_thermal_summary.json"
            if metadata_path.is_file() and fuel_path.is_file() and tie_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                fuel = json.loads(fuel_path.read_text(encoding="utf-8"))
                tie = json.loads(tie_path.read_text(encoding="utf-8"))
                using_fallback = False
                st.success(f"Loaded analysis from {analysis_root}")
            else:
                metadata, fuel, tie = _synthetic_demo()
                using_fallback = True
        elif mode == "Upload run outputs":
            metadata = _read_json_upload(
                st.file_uploader("metadata.json", type=["json"], key="metadata")
            )
            fuel = _read_json_upload(
                st.file_uploader("thermal_summary.json", type=["json"], key="fuel")
            )
            tie = _read_json_upload(
                st.file_uploader("tie_thermal_summary.json", type=["json"], key="tie")
            )
            if metadata is None or fuel is None or tie is None:
                metadata, fuel, tie = _synthetic_demo()
                using_fallback = True
            else:
                using_fallback = False
        else:
            metadata, fuel, tie = _synthetic_demo()
            using_fallback = True

        gamma = 1.35
        nozzle_efficiency = 0.95
        exit_pressure_kpa = 1.0
        ambient_pressure_kpa = 0.0

        if mode not in ("Historical NERVA reference", "Bimodal NTR + LANTR"):
            st.header("Nozzle assumptions")
            gamma = st.slider("γ", 1.10, 1.50, 1.35, 0.01)
            nozzle_efficiency = st.slider(
                "Nozzle efficiency",
                0.70,
                1.00,
                0.95,
                0.01,
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

    if historical_preset is not None:
        _render_historical(historical_preset)
    elif lantr_mr is not None:
        _render_lantr(
            lantr_mr,
            lantr_architecture,
            lantr_pressure_mpa,
            lantr_temperature_k,
        )
    else:
        _render_calculated(
            metadata=metadata,
            fuel=fuel,
            tie=tie,
            using_fallback=using_fallback,
            gamma=gamma,
            nozzle_efficiency=nozzle_efficiency,
            exit_pressure_kpa=exit_pressure_kpa,
            ambient_pressure_kpa=ambient_pressure_kpa,
        )

    st.caption(
        "Historical mode reproduces published NASA system-level reference values. "
        "Calculated mode derives thrust/Isp from the loaded thermal state using "
        "the repository's ideal-nozzle engineering model."
    )


if __name__ == "__main__":
    main()
