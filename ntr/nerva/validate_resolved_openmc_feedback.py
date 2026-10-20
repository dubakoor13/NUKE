"""Numerical-only verification of channel-resolved OpenMC/thermal Picard.

NO real OpenMC is run here. Mock providers are explicitly tagged synthetic.
These tests check temperature projection, energy conservation, stale tally
rejection, immutable old-state integration, and convergence failure.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .resolved_openmc_feedback import (
    TransportSnapshot, advance_resolved_with_openmc,
    reduce_openmc_material_state, state_infinity_difference
)
from .resolved_transient import (
    ResolvedParameters, read_resolved_heating,
    initial_resolved_state, advance_resolved
)
from .validate_resolved_transient import write_synthetic_reconstructions


def expect(error, callable_):
    try:
        callable_()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        mesh, fuel, tie = write_synthetic_reconstructions(Path(tmp))
        base = read_resolved_heating(fuel, tie, mesh)
        params = ResolvedParameters(
            fuel_solid_capacity_j_k=5000,
            fuel_h2_capacity_j_k=500,
            fuel_ua_w_k=100,
            tie_solid_capacity_j_k=3000,
            tie_supply_h2_capacity_j_k=600,
            tie_return_h2_capacity_j_k=700,
            tie_supply_ua_w_k=80,
            tie_return_ua_w_k=90,
        )
        state = initial_resolved_state(
            base, solid_temperature_k=500., gas_temperature_k=300.
        )
        f,g = reduce_openmc_material_state(state, params)
        assert math.isclose(f,500)
        assert math.isclose(g,300)
        assert state_infinity_difference(state,state)==0
        calls=[]

        def mock(tf,th):
            n=len(calls)+1
            calls.append((tf,th))
            # Mild synthetic temperature sensitivity of the spatial shape,
            # NO physical neutronic feedback is implied.
            shift = (tf-500)*1e-4
            fuel_s = base.fuel_solid_w.copy()
            tie_s = base.tie_solid_w.copy()
            fuel_s[0,0,0] += shift
            tie_s[0,0] -= shift
            h=replace(base, fuel_solid_w=fuel_s, tie_solid_w=tie_s)
            h.validate()
            return TransportSnapshot(
                heating=h, source_kind="synthetic-test",
                statepoint_sha256=f"synthetic-iteration-{n}",
                nuclear_data_sha256="synthetic-library-not-real",
                fuel_material_temperature_k=tf,
                hydrogen_material_temperature_k=th,
                analysis_qa_passed=True,
            )

        result = advance_resolved_with_openmc(
            state,params,300.,0.1,mock,
            allow_synthetic_test_provider=True,
            convergence_temperature_k=0.5,
        )
        assert result.iterations>=2
        assert result.iterations==len(calls)
        assert len(set(x.snapshot_sha256 for x in result.diagnostics))==len(calls)
        assert abs(result.result.energy_residual_j)<1e-5
        assert result.result.state.unresolved_energy_j==26.4
        assert result.temperature_error_k<=0.5
        assert result.profile_error_k<=0.5
        # The FINAL converged solution is one single 0.1 s step from
        # original, not N times advancement within Picard.
        direct=advance_resolved(
            result.last_snapshot.heating,state,params,300.,0.1
        )
        assert state_infinity_difference(
            result.result.state,direct.state
        ) < 1e-11
        assert math.isclose(
            result.result.state.unresolved_energy_j,26.4,rel_tol=1e-12
        )

        # Real source must not silently accept simulated provider output.
        expect(ValueError,lambda:advance_resolved_with_openmc(
            state,params,300.,0.1,mock
        ))

        def stale(tf,th):
            return replace(mock(tf,th),statepoint_sha256="same-statepoint")
        expect(RuntimeError,lambda:advance_resolved_with_openmc(
            state,params,300.,0.1,stale,
            allow_synthetic_test_provider=True,max_iterations=3
        ))

        def reject_qa(tf,th):
            return replace(mock(tf,th),analysis_qa_passed=False)
        expect(RuntimeError,lambda:advance_resolved_with_openmc(
            state,params,300.,0.1,reject_qa,
            allow_synthetic_test_provider=True
        ))

        def changed_flows(tf,th):
            ss=mock(tf,th)
            if len(calls)%2==0:
                h=ss.heating
                f=h.fuel_flow_kg_s.copy()
                f[0,0]*=1.1
                return replace(ss,heating=replace(h,fuel_flow_kg_s=f))
            return ss
        # Ensure first call is odd for the second call to be even.
        calls.clear()
        expect(RuntimeError,lambda:advance_resolved_with_openmc(
            state,params,300.,0.1,changed_flows,
            allow_synthetic_test_provider=True,max_iterations=3
        ))

        def missing_xs(tf,th):
            return replace(mock(tf,th),nuclear_data_sha256="")
        expect(RuntimeError,lambda:advance_resolved_with_openmc(
            state,params,300.,0.1,missing_xs,
            allow_synthetic_test_provider=True
        ))

        def wrong_transport_temperature(tf,th):
            return replace(mock(tf,th),hydrogen_material_temperature_k=th+5)
        expect(RuntimeError,lambda:advance_resolved_with_openmc(
            state,params,300.,0.1,wrong_transport_temperature,
            allow_synthetic_test_provider=True
        ))

        def too_much_unresolved(tf,th):
            ss=mock(tf,th)
            return replace(ss,heating=ss.heating)
        expect(RuntimeError,lambda:advance_resolved_with_openmc(
            state,params,300.,0.1,too_much_unresolved,
            allow_synthetic_test_provider=True,
            max_unresolved_fraction=0.001
        ))

        expect(RuntimeError,lambda:advance_resolved_with_openmc(
            state,params,300.,0.1,mock,
            allow_synthetic_test_provider=True,
            max_iterations=2,convergence_temperature_k=1e-14
        ))
        expect(ValueError,lambda:advance_resolved_with_openmc(
            state,params,300.,-1,mock,
            allow_synthetic_test_provider=True
        ))
        # Test the actual equilibrium H2 mass-weighting rule:
        fstate=replace(state,
            tie_supply_t_k=np.full_like(state.tie_supply_t_k,350),
            tie_return_t_k=np.full_like(state.tie_return_t_k,400))
        f2,g2=reduce_openmc_material_state(fstate,params)
        expected=(
            500*params.fuel_h2_capacity_j_k*state.fuel_h2_t_k.size
            + 350*params.tie_supply_h2_capacity_j_k*state.tie_supply_t_k.size
            + 400*params.tie_return_h2_capacity_j_k*state.tie_return_t_k.size
        )/(
            params.fuel_h2_capacity_j_k*state.fuel_h2_t_k.size
            +params.tie_supply_h2_capacity_j_k*state.tie_supply_t_k.size
            +params.tie_return_h2_capacity_j_k*state.tie_return_t_k.size
        )
        # Fuel gas stays at 300 K; use corrected reference in expected.
        correct=expected-(
            200*params.fuel_h2_capacity_j_k*state.fuel_h2_t_k.size
        )/(
            params.fuel_h2_capacity_j_k*state.fuel_h2_t_k.size
            +params.tie_supply_h2_capacity_j_k*state.tie_supply_t_k.size
            +params.tie_return_h2_capacity_j_k*state.tie_return_t_k.size
        )
        assert math.isclose(f2,500)
        assert math.isclose(g2,correct,rel_tol=1e-12)
    print("Resolved OpenMC multi-iteration feedback and QA gates: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
