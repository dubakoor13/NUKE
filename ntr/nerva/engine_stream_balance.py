"""Conservative NERVA engine system stream accounting.

An ideal nozzle is NOT the NERVA engine. Main nozzle and turbine/bleed exhaust
must be tracked separately to compute true propellant-average specific impulse.

This does not size turbine hardware, reconstruct a NESS hot-bleed network,
or model transient reactor dynamics. The inverse bleed-speed operation is
an algebraic constraint, NOT a validated prediction of turbine exhaust.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

G0 = 9.80665


@dataclass(frozen=True)
class EngineStreamBalance:
    main_flow_kg_s: float
    bleed_flow_kg_s: float
    total_flow_kg_s: float
    nozzle_effective_velocity_m_s: float
    bleed_effective_velocity_m_s: float
    main_thrust_n: float
    bleed_thrust_n: float
    total_thrust_n: float
    nozzle_isp_s: float
    engine_isp_s: float
    bleed_mass_fraction: float


def _positive_finite(x: float, name: str) -> float:
    value = float(x)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def engine_stream_balance(
    main_flow_kg_s: float,
    nozzle_effective_velocity_m_s: float,
    bleed_flow_kg_s: float = 0.0,
    bleed_effective_velocity_m_s: float = 0.0,
) -> EngineStreamBalance:
    """Sum stream propellant and momentum contribution in a common reference frame.

    Effective velocities represent *net thrust / flow* including exit-pressure
    thrust, with each stream's exhaust and pressure forces already evaluated.
    They may not be substituted for local nozzle jet speeds uncorrected.
    """
    main_flow = _positive_finite(main_flow_kg_s, "main mass flow")
    main_speed = _positive_finite(nozzle_effective_velocity_m_s, "main effective velocity")
    bleed = float(bleed_flow_kg_s)
    bleed_speed = float(bleed_effective_velocity_m_s)
    if not math.isfinite(bleed) or bleed < 0:
        raise ValueError("bleed flow must be nonnegative")
    if not math.isfinite(bleed_speed) or bleed_speed < 0:
        raise ValueError("bleed effective velocity must be nonnegative")
    if not bleed and bleed_speed:
        raise ValueError("bleed speed without bleed flow is invalid")
    total = main_flow + bleed
    main_force = main_flow * main_speed
    bleed_force = bleed * bleed_speed
    force = main_force + bleed_force
    return EngineStreamBalance(
        main_flow_kg_s=main_flow, bleed_flow_kg_s=bleed,
        total_flow_kg_s=total,
        nozzle_effective_velocity_m_s=main_speed,
        bleed_effective_velocity_m_s=bleed_speed,
        main_thrust_n=main_force, bleed_thrust_n=bleed_force,
        total_thrust_n=force,
        nozzle_isp_s=main_speed/G0,
        engine_isp_s=force/(total*G0),
        bleed_mass_fraction=bleed/total,
    )


@dataclass(frozen=True)
class InferredBleedConstraint:
    observed_engine_thrust_n: float
    observed_engine_isp_s: float
    observed_main_flow_kg_s: float
    inferred_total_flow_kg_s: float
    inferred_bleed_flow_kg_s: float
    assumed_nozzle_effective_velocity_m_s: float
    required_bleed_effective_velocity_m_s: float
    interpretation: str = (
        "Inverse algebra from reported overall thrust and Isp and assumed "
        "nozzle model, NOT measured bleed flow or validated cycle performance"
    )


def infer_bleed_from_engine_reference(
    observed_thrust_n: float, observed_engine_isp_s: float,
    main_flow_kg_s: float, nozzle_effective_velocity_m_s: float,
) -> InferredBleedConstraint:
    """Infer missing bleed mass flow and velocity from *two* engine observables.

    This is underdetermined if main flow or nozzle state are unknown; requiring
    them as independent inputs makes the assumptions explicit. A negative
    required velocity is returned to flag an incompatible nozzle assumption.
    """
    force = _positive_finite(observed_thrust_n, "observed thrust")
    isp = _positive_finite(observed_engine_isp_s, "observed Isp")
    main = _positive_finite(main_flow_kg_s, "observed main mass flow")
    vmain = _positive_finite(nozzle_effective_velocity_m_s, "assumed nozzle velocity")
    total = force/(isp*G0)
    bleed = total - main
    if bleed <= 0:
        raise ValueError("observed total flow leaves no positive bleed flow")
    vbleed = (force-main*vmain)/bleed
    return InferredBleedConstraint(
        observed_engine_thrust_n=force, observed_engine_isp_s=isp,
        observed_main_flow_kg_s=main, inferred_total_flow_kg_s=total,
        inferred_bleed_flow_kg_s=bleed,
        assumed_nozzle_effective_velocity_m_s=vmain,
        required_bleed_effective_velocity_m_s=vbleed,
    )
