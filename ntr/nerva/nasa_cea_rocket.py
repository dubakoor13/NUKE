"""NASA CEA 3.3.4 hot-H2 rocket nozzle backend, not a historical NERVA simulation.

One-dimensional, infinite-area combustor (IAC); NASA RP-1311 RocketSolver.
Equilibrium H2/H chemistry or frozen composition from the throat (n_frz=2).
No feed cycle, reactor geometry/criticality, multi-D or boundary-layer losses.

Reference: https://github.com/nasa/cea/blob/main/docs/source/examples/rocket/example12.rst
Install: python -m pip install cea==3.3.4 numpy
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from importlib.metadata import version
from pathlib import Path

G0 = 9.80665


@dataclass(frozen=True)
class CeaHydrogenRocket:
    mode: str
    cea_version: str
    chamber_pressure_pa: float
    chamber_temperature_k: float
    throat_area_m2: float
    area_ratio: float
    throat_temperature_k: float
    exit_temperature_k: float
    exit_pressure_pa: float
    throat_mach: float
    exit_mach: float
    chamber_h_atom_mole_fraction: float
    exit_h_atom_mole_fraction: float
    characteristic_velocity_m_s: float
    mass_flow_kg_s: float
    exit_velocity_ideal_m_s: float
    exit_velocity_delivered_m_s: float
    ideal_vacuum_effective_velocity_m_s: float
    vacuum_thrust_n: float
    delivered_thrust_n: float
    delivered_isp_s: float
    effective_thrust_coefficient: float
    nozzle_efficiency: float
    ambient_pressure_pa: float
    mass_flow_closure_relative: float
    vacuum_identity_residual_n: float


def nasa_cea_hydrogen_rocket(
    chamber_pressure_pa: float, chamber_temperature_k: float,
    throat_area_m2: float, area_ratio: float,
    mode: str = "equilibrium", ambient_pressure_pa: float = 0.0,
    nozzle_efficiency: float = 1.0,
) -> CeaHydrogenRocket:
    """Evaluate NASA CEA hot hydrogen nozzle from an imposed stagnation state.

    NASA CEA RocketSolution.Isp_vacuum is effective exhaust velocity in m/s,
    NOT seconds. Pressures in CEA's rocket interface use bar, not Pa.
    The efficiency scalar acts only on exit kinetic energy, not CEA chemistry.
    """
    values = (chamber_pressure_pa, chamber_temperature_k, throat_area_m2,
              area_ratio, ambient_pressure_pa, nozzle_efficiency)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("all CEA inputs must be finite")
    if chamber_pressure_pa <= 0 or throat_area_m2 <= 0 or area_ratio <= 1:
        raise ValueError("invalid chamber pressure or nozzle geometry")
    if not 200 <= chamber_temperature_k <= 3500:
        raise ValueError("study temperature must be within 200–3500 K")
    if ambient_pressure_pa < 0 or not 0 < nozzle_efficiency <= 1:
        raise ValueError("invalid ambient pressure or nozzle efficiency")
    if mode not in ("equilibrium", "frozen-throat"):
        raise ValueError("mode must be equilibrium or frozen-throat")
    try:
        import cea
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Requires NASA CEA: pip install cea==3.3.4 numpy") from exc

    # Hot H2 is a reactor outlet stream, not a hydrogen/oxygen combustion case.
    reactants = cea.Mixture(["H2"])
    products = cea.Mixture(["H2", "H"])
    solver = cea.RocketSolver(products, reactants=reactants)
    solution = cea.RocketSolution(solver)
    solver.solve(
        solution, np.array([1.0], dtype=np.float64),
        float(chamber_pressure_pa / 1e5), supar=[float(area_ratio)],
        tc=float(chamber_temperature_k), iac=True,
        n_frz=2 if mode == "frozen-throat" else None,
    )
    if not solution.converged or solution.num_pts < 3:
        raise RuntimeError(f"NASA CEA rocket failed; error={solution.last_error}")
    idx = solution.num_pts - 1
    if not math.isclose(float(solution.T[0]), chamber_temperature_k, rel_tol=2e-5):
        raise RuntimeError("NASA CEA chamber temperature differs from input")
    if not math.isclose(float(solution.P[0]) * 1e5, chamber_pressure_pa, rel_tol=2e-5):
        raise RuntimeError("NASA CEA chamber pressure differs from input")

    cstar = float(solution.c_star[idx])
    pe = float(solution.P[idx]) * 1e5
    if not math.isfinite(cstar) or cstar <= 0 or not 0 < pe < chamber_pressure_pa:
        raise RuntimeError("invalid CEA nozzle state")
    mdot = chamber_pressure_pa * throat_area_m2 / cstar
    ae = area_ratio * throat_area_m2
    v_vac = float(solution.Isp_vacuum[idx])
    # Ideal vacuum effective velocity contains the Pe*Ae/mdot thrust term.
    ve_ideal = v_vac - pe * ae / mdot
    if not math.isfinite(ve_ideal) or ve_ideal <= 0:
        raise RuntimeError("NASA CEA vacuum velocity decomposition invalid")
    ve_delivered = math.sqrt(nozzle_efficiency) * ve_ideal
    f_vac = mdot * v_vac
    f = mdot * ve_delivered + (pe - ambient_pressure_pa) * ae
    if f <= 0:
        raise ValueError("negative/zero net thrust for selected ambient pressure")
    h_mole = solution.mole_fractions["H"]
    return CeaHydrogenRocket(
        mode=mode, cea_version=version("cea"),
        chamber_pressure_pa=chamber_pressure_pa,
        chamber_temperature_k=float(solution.T[0]),
        throat_area_m2=throat_area_m2, area_ratio=area_ratio,
        throat_temperature_k=float(solution.T[1]),
        exit_temperature_k=float(solution.T[idx]),
        exit_pressure_pa=pe, throat_mach=float(solution.Mach[1]),
        exit_mach=float(solution.Mach[idx]),
        chamber_h_atom_mole_fraction=float(h_mole[0]),
        exit_h_atom_mole_fraction=float(h_mole[idx]),
        characteristic_velocity_m_s=cstar,
        mass_flow_kg_s=mdot,
        exit_velocity_ideal_m_s=ve_ideal,
        exit_velocity_delivered_m_s=ve_delivered,
        ideal_vacuum_effective_velocity_m_s=v_vac,
        vacuum_thrust_n=f_vac, delivered_thrust_n=f,
        delivered_isp_s=f / (mdot * G0),
        effective_thrust_coefficient=f / (chamber_pressure_pa * throat_area_m2),
        nozzle_efficiency=nozzle_efficiency,
        ambient_pressure_pa=ambient_pressure_pa,
        mass_flow_closure_relative=(mdot * cstar -
            chamber_pressure_pa * throat_area_m2) /
            (chamber_pressure_pa * throat_area_m2),
        vacuum_identity_residual_n=f_vac - (mdot * ve_ideal + pe * ae),
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pc-mpa", type=float, default=3.10264)
    p.add_argument("--tc-k", type=float, default=2425.0)
    p.add_argument("--at-m2", type=float, default=0.02)
    p.add_argument("--area-ratio", type=float, default=100.0)
    p.add_argument("--mode", choices=("equilibrium", "frozen-throat"),
                   default="equilibrium")
    p.add_argument("--nozzle-efficiency", type=float, default=1.0)
    p.add_argument("--ambient-pressure-pa", type=float, default=0.0)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = nasa_cea_hydrogen_rocket(
        args.pc_mpa * 1e6, args.tc_k, args.at_m2, args.area_ratio,
        mode=args.mode, nozzle_efficiency=args.nozzle_efficiency,
        ambient_pressure_pa=args.ambient_pressure_pa,
    )
    payload = json.dumps(asdict(result), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
