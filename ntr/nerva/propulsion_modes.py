"""Four NTP propulsion families requested for NUKE.

Source fidelity:
* LH2 NTR: source-backed reference from the uploaded NASA/AMA deck.
* LCH4 NTR: source-backed methane PBM Cases 1 and 2 from the deck.
* LH2 + LOX: source-backed NASA LANTR performance tables.
* LCH4 + LOX: exploratory architecture only. No calibrated Isp/thrust map is
  asserted because neither the uploaded deck nor the NASA LANTR references
  provide one.

This module deliberately separates source-backed performance from unvalidated
architecture so the UI cannot silently present invented methane+LOX numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .alternative_propellant_cases import (
    H_NTP,
    METHANE_CASE_1,
    METHANE_CASE_2,
)
from .lantr import lantr_point, high_pressure_lantr_isp


@dataclass(frozen=True)
class PropulsionMode:
    key: str
    name: str
    nuclear_propellant: str
    lox_augmented: bool
    fidelity: str
    source_basis: str
    notes: str


@dataclass(frozen=True)
class ModePerformance:
    mode_key: str
    operating_point: str
    nuclear_propellant_flow_kg_s: Optional[float]
    oxygen_flow_kg_s: Optional[float]
    total_flow_kg_s: Optional[float]
    reactor_power_mw: Optional[float]
    chamber_temperature_k: Optional[float]
    chamber_pressure_mpa: Optional[float]
    isp_s: Optional[float]
    thrust_n: Optional[float]
    performance_status: str


LH2_NTR = PropulsionMode(
    key="lh2_ntp",
    name="LH2 Nuclear Thermal Rocket",
    nuclear_propellant="LH2",
    lox_augmented=False,
    fidelity="source-backed reference",
    source_basis="Uploaded NASA/AMA methane-NTP comparison deck",
    notes=(
        "Deck reference is approximately 900 s Isp at 2700 K. The deck does "
        "not provide a complete numbered LH2 PBM state table."
    ),
)


LCH4_NTR = PropulsionMode(
    key="lch4_ntp",
    name="LCH4 Nuclear Thermal Rocket",
    nuclear_propellant="LCH4",
    lox_augmented=False,
    fidelity="source-backed PBM",
    source_basis="Uploaded NASA/AMA M-NTP PBM Cases 1 and 2",
    notes=(
        "Frozen-chemistry sizing baseline. Methane cracking/pyrolysis is not "
        "included in the source PBM performance numbers."
    ),
)


LH2_LOX_LANTR = PropulsionMode(
    key="lh2_lox_lantr",
    name="LH2 NTR + LOX augmentation (LANTR)",
    nuclear_propellant="LH2",
    lox_augmented=True,
    fidelity="source-backed NASA LANTR table",
    source_basis=(
        "NASA LOX-Augmented NTR / trimodal LANTR published performance tables"
    ),
    notes=(
        "Oxygen is injected downstream of the choked throat and combusts "
        "supersonically with reactor-heated hydrogen."
    ),
)


LCH4_LOX_AUGMENTED = PropulsionMode(
    key="lch4_lox_augmented",
    name="LCH4 NTR + LOX augmentation",
    nuclear_propellant="LCH4",
    lox_augmented=True,
    fidelity="exploratory / chemistry model required",
    source_basis=(
        "Architecture requested by user; no calibrated performance table in "
        "the uploaded deck or NASA LANTR references"
    ),
    notes=(
        "Mass-flow bookkeeping is supported. Isp and thrust are intentionally "
        "not produced until a validated reacting-flow CH4-pyrolysis + O2 "
        "chemistry/nozzle model is attached."
    ),
)


FOUR_PROPULSION_MODES = {
    mode.key: mode
    for mode in (
        LH2_NTR,
        LCH4_NTR,
        LH2_LOX_LANTR,
        LCH4_LOX_AUGMENTED,
    )
}


def lh2_ntp_reference() -> ModePerformance:
    return ModePerformance(
        mode_key=LH2_NTR.key,
        operating_point="2700 K deck reference",
        nuclear_propellant_flow_kg_s=None,
        oxygen_flow_kg_s=0.0,
        total_flow_kg_s=None,
        reactor_power_mw=None,
        chamber_temperature_k=H_NTP.reference_temperature_k,
        chamber_pressure_mpa=None,
        isp_s=H_NTP.reference_isp_s,
        thrust_n=None,
        performance_status=(
            "reference-only: deck gives Isp/temperature but no full LH2 PBM "
            "flow/thrust state table"
        ),
    )


def lch4_ntp_reference(case_number: int = 2) -> ModePerformance:
    if case_number == 1:
        case = METHANE_CASE_1
    elif case_number == 2:
        case = METHANE_CASE_2
    else:
        raise ValueError("case_number must be 1 or 2")

    return ModePerformance(
        mode_key=LCH4_NTR.key,
        operating_point=case.name,
        nuclear_propellant_flow_kg_s=case.total_mass_flow_kg_s,
        oxygen_flow_kg_s=0.0,
        total_flow_kg_s=case.total_mass_flow_kg_s,
        reactor_power_mw=case.reactor_power_mw,
        chamber_temperature_k=case.chamber_temperature_k,
        chamber_pressure_mpa=case.chamber_pressure_mpa,
        isp_s=case.isp_s,
        thrust_n=case.derived_thrust_n,
        performance_status=(
            "source-backed frozen-chemistry PBM sizing result"
        ),
    )


def lh2_lox_lantr_reference(
    oxygen_hydrogen_ratio: float,
    high_pressure: bool = False,
) -> ModePerformance:
    mr = float(oxygen_hydrogen_ratio)
    if mr < 0.0:
        raise ValueError("oxygen_hydrogen_ratio must be non-negative")

    if high_pressure:
        if mr > 7.0:
            raise ValueError("high-pressure NASA LANTR table spans MR 0 to 7")
        isp_s, _ = high_pressure_lantr_isp(mr)
        return ModePerformance(
            mode_key=LH2_LOX_LANTR.key,
            operating_point="NASA 2000-psia trimodal LANTR",
            nuclear_propellant_flow_kg_s=None,
            oxygen_flow_kg_s=None,
            total_flow_kg_s=None,
            reactor_power_mw=None,
            chamber_temperature_k=2900.0,
            chamber_pressure_mpa=13.789514586336,
            isp_s=isp_s,
            thrust_n=None,
            performance_status=(
                "source-backed Isp map; this table does not publish enough "
                "flow data to reconstruct arbitrary thrust without choosing "
                "a specific engine geometry"
            ),
        )

    if mr > 5.0:
        raise ValueError("NASA small-engine LANTR table spans MR 0 to 5")
    point = lantr_point(mr)
    return ModePerformance(
        mode_key=LH2_LOX_LANTR.key,
        operating_point="NASA 1000-psia LANTR",
        nuclear_propellant_flow_kg_s=None,
        oxygen_flow_kg_s=None,
        total_flow_kg_s=None,
        reactor_power_mw=None,
        chamber_temperature_k=2734.0,
        chamber_pressure_mpa=6.894757293168,
        isp_s=point.delivered_isp_s,
        thrust_n=point.thrust_n,
        performance_status="source-backed NASA LANTR table",
    )


def lch4_lox_mass_bookkeeping(
    oxygen_methane_ratio: float,
    methane_case_number: int = 2,
) -> ModePerformance:
    """Return only quantities supported without inventing reacting performance.

    O/F is interpreted here as oxygen mass flow divided by the source M-NTP
    methane mass flow. This function does NOT estimate combustion efficiency,
    afterburner temperature, nozzle chemistry, Isp, or thrust.
    """
    ratio = float(oxygen_methane_ratio)
    if ratio < 0.0:
        raise ValueError("oxygen_methane_ratio must be non-negative")

    base = lch4_ntp_reference(methane_case_number)
    methane_flow = base.nuclear_propellant_flow_kg_s
    if methane_flow is None:
        raise RuntimeError("methane source case is missing mass flow")

    oxygen_flow = ratio * methane_flow
    total_flow = methane_flow + oxygen_flow

    return ModePerformance(
        mode_key=LCH4_LOX_AUGMENTED.key,
        operating_point=(
            f"{base.operating_point} + exploratory downstream LOX, "
            f"O/CH4={ratio:g}"
        ),
        nuclear_propellant_flow_kg_s=methane_flow,
        oxygen_flow_kg_s=oxygen_flow,
        total_flow_kg_s=total_flow,
        reactor_power_mw=base.reactor_power_mw,
        chamber_temperature_k=base.chamber_temperature_k,
        chamber_pressure_mpa=base.chamber_pressure_mpa,
        isp_s=None,
        thrust_n=None,
        performance_status=(
            "mass-flow bookkeeping only; reacting CH4-pyrolysis/O2 chemistry "
            "and nozzle solution required for Isp/thrust"
        ),
    )
