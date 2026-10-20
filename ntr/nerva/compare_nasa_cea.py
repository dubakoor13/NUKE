"""Independent NASA CEA reference comparison for the NERVA hot-H2 solver.

Uses NASA's *actual* CEA Python binding and thermochemical database, not
the polynomial implementation under test. This is a validation aid, not a
claim that NUKE reproduces historic XE Prime or NESS engine-system results.

Install separately with: python -m pip install cea numpy
Official upstream: https://github.com/nasa/cea
"""
from __future__ import annotations

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

from .nasa_hydrogen import hydrogen_state

NASA_CEA_SOURCE = "https://github.com/nasa/cea"
NASA_CEA_REFERENCE = "NASA RP-1311 and NASA TP-2002-211556"


def compare(p_c_pa: float, temperatures_k: list[float]) -> list[dict]:
    """Compare independent NASA-7 ideal-H2/H equilibrium against NASA CEA TP.

    Both codes use the same restricted H/H2 reaction set, but DIFFERENT
    thermochemical coefficients (GRI NASA-7 vs NASA Glenn CEA database).
    Differences are expected; the report records them explicitly.
    """
    try:
        import cea
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NASA CEA is optional. Install: python -m pip install cea numpy"
        ) from exc
    if not math.isfinite(p_c_pa) or p_c_pa <= 0:
        raise ValueError("pressure must be finite and positive")
    if not temperatures_k:
        raise ValueError("temperatures list is empty")
    if any(not math.isfinite(T) or not 200 <= T <= 3500 for T in temperatures_k):
        raise ValueError("comparison temperatures must be 200-3500 K")

    # Only H and H2 are allowed in the CEA product set. This matches exactly
    # the reaction species of our *reduced* equilibrium reference model.
    reactants = cea.Mixture(["H2"])
    products = cea.Mixture(["H", "H2"])
    solver = cea.EqSolver(products, reactants=reactants)
    solution = cea.EqSolution(solver)
    reactant_weights = np.array([1.0], dtype=float)
    rows = []

    for T in temperatures_k:
        # NASA CEA public Python API documents pressure in bar and T in K.
        solver.solve(solution, cea.TP, float(T), p_c_pa/1.0e5,
                     reactant_weights)
        if not solution.converged:
            raise RuntimeError(
                f"NASA CEA did not converge at {T} K, {p_c_pa} Pa"
            )
        y_h = float(solution.mole_fractions["H"])
        y_h2 = float(solution.mole_fractions["H2"])
        if not 0 <= y_h <= 1 or not math.isclose(y_h+y_h2, 1, abs_tol=1e-5):
            raise AssertionError("NASA CEA produced invalid H/H2 mole fractions")
        dissociation = y_h/(2.0-y_h)
        self_state = hydrogen_state(T, p_c_pa, "equilibrium")

        rho_cea = float(solution.density)  # CEA Python: kg/m3
        rho_rel = (self_state.density_kg_m3-rho_cea)/rho_cea
        diss_abs = self_state.dissociation_fraction-dissociation
        # CEA EqSolution enthalpy and frozen cp reported in kJ/kg and
        # kJ/(kg*K); avoid comparing absolute h offsets across databases.
        cp_cea = float(solution.cp_fr)*1e3
        cp_rel = (self_state.frozen_cp_j_kg_k-cp_cea)/cp_cea
        rows.append({
            "temperature_k": T,
            "pressure_pa": p_c_pa,
            "alpha_cea": dissociation,
            "alpha_nasa7": self_state.dissociation_fraction,
            "alpha_absolute_error": diss_abs,
            "density_cea_kg_m3": rho_cea,
            "density_nasa7_kg_m3": self_state.density_kg_m3,
            "density_relative_error": rho_rel,
            "frozen_cp_cea_j_kg_k": cp_cea,
            "frozen_cp_nasa7_j_kg_k": self_state.frozen_cp_j_kg_k,
            "frozen_cp_relative_error": cp_rel,
        })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pc-mpa", type=float, default=3.10264)
    p.add_argument("--temperatures-k", type=float, nargs="+",
                   default=[2350.0, 2500.0, 2700.0, 2900.0])
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    rows = compare(a.pc_mpa*1.0e6, a.temperatures_k)
    report = {
        "reference": NASA_CEA_REFERENCE,
        "cea_source": NASA_CEA_SOURCE,
        "cea_package_version": version("cea"),
        "test_model": "GRI-Mech 3.0 NASA-7 two-species H2/H",
        "validating_model": "NASA CEA full package, restricted to H2/H",
        "scope": "TP equilibrium thermodynamics only; not full NERVA",
        "rows": rows,
    }
    for row in rows:
        print(
            "T={temperature_k:.0f} K, alpha_CEA={alpha_cea:.6e}, "
            "alpha_NASA7={alpha_nasa7:.6e}, "
            "rho_err={density_relative_error:+.4%}, "
            "cp_err={frozen_cp_relative_error:+.4%}".format(**row)
        )
        # Broad engineering *sanity* checks; tight cross-database parity
        # requires a dedicated numerical tolerance/coefficients study.
        if abs(row["density_relative_error"]) > 0.005:
            raise AssertionError("density discrepancy > 0.5% vs NASA CEA")
        if abs(row["frozen_cp_relative_error"]) > 0.01:
            raise AssertionError("cp discrepancy > 1% vs NASA CEA")
        if abs(row["alpha_absolute_error"]) > max(1e-5, 0.03*row["alpha_cea"]):
            raise AssertionError("dissociation discrepancy > 3% vs NASA CEA")
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("NASA CEA independent hydrogen comparison: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
