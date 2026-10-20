"""Regression tests for timestep convergence of resolved fuel/tie transients.

Synthetic mathematically constructed fields only. No actual OpenMC reactor
or historical test data are used in these verification checks.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .resolved_transient import (
    ResolvedParameters, initial_resolved_state, read_resolved_heating,
)
from .temporal_convergence import (
    field_error, refine_resolved_timestep
)
from .validate_resolved_transient import write_synthetic_reconstructions


def reject(operation, error=ValueError):
    try:
        operation()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        mesh, fuel, tie = write_synthetic_reconstructions(Path(directory))
        heat = read_resolved_heating(fuel, tie, mesh)
        p = ResolvedParameters(
            fuel_solid_capacity_j_k=5000.,
            fuel_h2_capacity_j_k=500.,
            fuel_ua_w_k=100.,
            tie_solid_capacity_j_k=3000.,
            tie_supply_h2_capacity_j_k=600.,
            tie_return_h2_capacity_j_k=700.,
            tie_supply_ua_w_k=80.,
            tie_return_ua_w_k=90.,
            hydrogen_cp_reference_j_kg_k=14300.,
            enthalpy_model="nasa7-h2",
        )
        state = initial_resolved_state(
            heat, solid_temperature_k=500., gas_temperature_k=300.
        )
        old_fuel = state.fuel_solid_t_k.copy()
        old_tie = state.tie_return_t_k.copy()
        report = refine_resolved_timestep(
            heat, state, p, 300., 0.8,
            base_steps=2, levels=4, tolerance_k=0.5
        )
        assert report.status == "PASS", report
        assert report.energy_residual_passed
        assert report.tolerance_passed
        assert report.difference_reducing
        assert len(report.levels)==4
        assert [row.steps for row in report.levels]==[2,4,8,16]
        assert [round(row.dt_s,10) for row in report.levels]==[0.4,0.2,0.1,0.05]
        assert all(math.isclose(row.end_time_s,0.8,abs_tol=1e-13)
                   for row in report.levels)
        assert len(report.difference_infinity_k)==3
        assert len(report.difference_by_field_k)==3
        assert all(diff>0 for diff in report.difference_infinity_k)
        assert all(
            later < earlier for earlier, later in zip(
                report.difference_infinity_k,
                report.difference_infinity_k[1:],
            )
        )
        assert report.finest_difference_k <= 0.5
        assert report.levels[-1].maximum_step_energy_residual_j < 1e-4
        for d in report.difference_by_field_k:
            assert set(d) == {
                "fuel_solid_t_k", "fuel_h2_t_k",
                "tie_solid_t_k", "tie_supply_t_k", "tie_return_t_k"
            }
        assert np.array_equal(state.fuel_solid_t_k,old_fuel)
        assert np.array_equal(state.tie_return_t_k,old_tie)
        assert field_error(state, state)["fuel_h2_t_k"] == 0.

        # A tolerance stricter than the computed truncation error must
        # create an explicit failure, never a success by conservation alone.
        fail = refine_resolved_timestep(
            heat, state, p, 300., 0.8,
            base_steps=2, levels=3, tolerance_k=1e-12
        )
        assert not fail.tolerance_passed
        assert fail.status == "FAIL"
        assert fail.energy_residual_passed

        # The constant-cp solver must have its own independent
        # convergence series, without sharing any cached NASA-7 states.
        cp_report = refine_resolved_timestep(
            heat, state, replace(p, enthalpy_model="constant"), 300., 0.8,
            base_steps=2, levels=3, tolerance_k=0.5
        )
        assert cp_report.status == "PASS"
        assert cp_report.difference_infinity_k[-1] > 0

        reject(lambda: refine_resolved_timestep(
            heat, state, p, 300., 0.8, levels=2
        ))
        reject(lambda: refine_resolved_timestep(
            heat, state, p, 300., 0.8, base_steps=0
        ))
        reject(lambda: refine_resolved_timestep(
            heat, state, p, 300., -1
        ))
        reject(lambda: refine_resolved_timestep(
            heat, state, p, 300., 0.8, tolerance_k=float("nan")
        ))
        reject(lambda: refine_resolved_timestep(
            heat, state, p, 300., 0.8, base_steps=200, levels=7,
            max_steps=300
        ))
        reject(lambda: field_error(
            state, replace(state, tie_return_t_k=np.zeros((1,1)))
        ))
        print(
            "NERVA resolved temporal convergence: PASS; "
            f"dt-grid={','.join(str(r.dt_s) for r in report.levels)}; "
            f"last absolute difference={report.finest_difference_k:.7g} K; "
            "all state fields and energy closure checked"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
