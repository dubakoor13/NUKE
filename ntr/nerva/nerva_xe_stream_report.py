"""NASA XE-Prime inverse stream-accounting study using direct NASA CEA.

A numerical *identifiability demonstration*, not a reconstructed NERVA
bleed-cycle prediction: the turbine/bleed effective velocity is inferred
from historical net thrust and Isp rather than predicted.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .historical_presets import XE_PRIME
from .nasa_cea_rocket import nasa_cea_hydrogen_rocket
from .engine_stream_balance import (
    engine_stream_balance, infer_bleed_from_engine_reference,
)


def main() -> int:
    required = ("thrust_n", "isp_s", "hydrogen_mass_flow_kg_s",
                "chamber_temperature_k", "chamber_pressure_pa",
                "nozzle_expansion_ratio")
    if any(getattr(XE_PRIME, key) is None for key in required):
        raise RuntimeError("XE-Prime historical preset missing required fields")
    cases = []
    for nozzle_eta in (1.0, 0.95, 0.90):
        # Only specific impulse is taken from CEA. Area 0.02 m2 is an
        # arbitrary geometry because the historical main mass flow is
        # separately specified; do not infer historical throat area.
        rocket = nasa_cea_hydrogen_rocket(
            XE_PRIME.chamber_pressure_pa, XE_PRIME.chamber_temperature_k,
            0.02, XE_PRIME.nozzle_expansion_ratio,
            mode="equilibrium", nozzle_efficiency=nozzle_eta,
        )
        inferred = infer_bleed_from_engine_reference(
            XE_PRIME.thrust_n, XE_PRIME.isp_s,
            XE_PRIME.hydrogen_mass_flow_kg_s,
            rocket.delivered_isp_s * 9.80665,
        )
        if inferred.required_bleed_effective_velocity_m_s < 0:
            raise ValueError("CEA nozzle assumption exceeds target thrust")
        reproduced = engine_stream_balance(
            inferred.observed_main_flow_kg_s,
            inferred.assumed_nozzle_effective_velocity_m_s,
            inferred.inferred_bleed_flow_kg_s,
            inferred.required_bleed_effective_velocity_m_s,
        )
        assert abs(reproduced.total_thrust_n - XE_PRIME.thrust_n) < 1e-6
        assert abs(reproduced.engine_isp_s - XE_PRIME.isp_s) < 1e-9
        cases.append({
            "nozzle_kinetic_efficiency_assumption": nozzle_eta,
            "cea_nozzle_isp_s": rocket.delivered_isp_s,
            "inferred": asdict(inferred),
            "reconstructed_balance": asdict(reproduced),
        })
        print(f"eta={nozzle_eta:.2f} => CEA Isp={rocket.delivered_isp_s:.2f} s; "
              f"required effective bleed velocity "
              f"{inferred.required_bleed_effective_velocity_m_s:.1f} m/s")
    result = {
        "reference": XE_PRIME.source_title, "source_url": XE_PRIME.source_url,
        "status": "inverse mass/thrust accounting only, not validated cycle",
        "known_inputs": {
            "thrust_n": XE_PRIME.thrust_n,
            "net_isp_s": XE_PRIME.isp_s,
            "main_hydrogen_flow_kg_s": XE_PRIME.hydrogen_mass_flow_kg_s,
            "chamber_temperature_k": XE_PRIME.chamber_temperature_k,
            "chamber_pressure_pa": XE_PRIME.chamber_pressure_pa,
            "nozzle_area_ratio": XE_PRIME.nozzle_expansion_ratio,
        },
        "cases": cases,
    }
    outpath = Path("build/nerva/xe_prime_stream_balance.json")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print("XE-Prime inverse stream accounting: PASS (not a cycle prediction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
