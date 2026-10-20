"""Numerical comparisons between NASA CEA IAC rocket and NUKE NASA-7 nozzle.

Reference operating states are public heritage examples. The throat area is
an arbitrary, explicit geometric input: matching historical thrust is NOT
a validation claim.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from .nasa_cea_rocket import nasa_cea_hydrogen_rocket
from .nasa_hydrogen import hydrogen_nozzle

CASES = (
    ("1972-NERVA-design-state", 450 * 6894.757293168, 2425.0, 100.0),
    ("XE-Prime-comparison-state", 3.90e6, 2281.0, 10.0),
)


def main() -> int:
    output = {
        "reference_backend": "NASA CEA 3.3.4 RocketSolver",
        "scope": "hot-H2 IAC nozzle; NOT historical NERVA full-engine validation",
        "assumed_throat_area_m2": 0.02, "cases": [],
    }
    for name, pc, tc, ratio in CASES:
        nasa = nasa_cea_hydrogen_rocket(pc, tc, 0.02, ratio)
        custom = hydrogen_nozzle(pc, tc, 0.02, ratio, mode="equilibrium")
        throat_frozen = nasa_cea_hydrogen_rocket(
            pc, tc, 0.02, ratio, mode="frozen-throat"
        )
        twice = nasa_cea_hydrogen_rocket(pc, tc, 0.04, ratio)
        lossy = nasa_cea_hydrogen_rocket(
            pc, tc, 0.02, ratio, nozzle_efficiency=0.95
        )
        assert abs(nasa.throat_mach - 1.0) < 0.01
        assert nasa.exit_mach > 1
        assert nasa.throat_temperature_k > nasa.exit_temperature_k
        assert abs(nasa.mass_flow_closure_relative) < 1e-12
        assert abs(nasa.vacuum_identity_residual_n) < 1e-7
        assert math.isclose(twice.mass_flow_kg_s / nasa.mass_flow_kg_s, 2,
                            rel_tol=1e-8)
        assert math.isclose(twice.delivered_thrust_n / nasa.delivered_thrust_n,
                            2, rel_tol=1e-8)
        assert math.isclose(twice.delivered_isp_s, nasa.delivered_isp_s,
                            rel_tol=1e-8)
        assert lossy.delivered_isp_s < nasa.delivered_isp_s
        assert 400 < nasa.delivered_isp_s < 1200
        assert 400 < throat_frozen.delivered_isp_s < 1200
        mass_diff = custom.mass_flow_kg_s / nasa.mass_flow_kg_s - 1
        isp_diff = custom.isp_s / nasa.delivered_isp_s - 1
        pe_diff = custom.exit.pressure_pa / nasa.exit_pressure_pa - 1
        # GRI-Mech NASA-7 vs NASA Glenn data: do not demand exact parity.
        assert abs(mass_diff) < 0.05
        assert abs(isp_diff) < 0.05
        assert abs(pe_diff) < 0.10
        output["cases"].append({
            "case": name, "chamber_pressure_pa": pc,
            "chamber_temperature_k": tc, "area_ratio": ratio,
            "nasa_cea_equilibrium": asdict(nasa),
            "nasa_cea_frozen_at_throat": asdict(throat_frozen),
            "custom_nasa7": {
                "mass_flow_kg_s": custom.mass_flow_kg_s,
                "isp_s": custom.isp_s,
                "exit_pressure_pa": custom.exit.pressure_pa,
            },
            "relative_difference": {
                "mass_flow": mass_diff, "isp": isp_diff,
                "exit_pressure": pe_diff,
            },
        })
        print(f"{name}: NASA CEA Isp={nasa.delivered_isp_s:.3f} s; "
              f"NASA-7 Isp={custom.isp_s:.3f} s; delta={isp_diff:+.3%}; "
              f"CEA throat-frozen Isp={throat_frozen.delivered_isp_s:.3f} s")
    artifact = Path("build/nerva/cea_rocket_reference_comparison.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("NASA CEA hot-hydrogen rocket-nozzle validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
