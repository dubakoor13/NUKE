"""Tests for implicit hydrogen enthalpy transport and OpenMC coupling.

No nuclear cross-section data are needed: these validate mathematics and
energy conservation, NOT historical reactor physics or criticality.
"""
from __future__ import annotations

from dataclasses import replace
import math

from .heritage_transient import (
    ThermalCell, ThermalState, advance_thermal,
    max_explicit_step_s
)
from .implicit_hydrogen import (
    advance_thermal_implicit, hydrogen_specific_enthalpy
)
from .openmc_transient import (
    Branch, OpenMCPowerMap, initial_state, advance_openmc_transient,
    advance_with_transport_feedback
)


def assert_raises(error: type, fn) -> None:
    try:
        fn()
    except error:
        return
    raise AssertionError(f"expected {error.__name__}")


def check_energy(step) -> None:
    scale = max(
        1., abs(step.stored_energy_change_j),
        abs(step.input_power_w*0.1)
    )
    assert abs(step.energy_balance_error_j) < 1e-9*scale, (
        f"energy residual = {step.energy_balance_error_j}"
    )


def main() -> int:
    # First-principles single-node, no advection and no heat exchange:
    # both sensible energies increase exactly by their own deposited heat.
    c = (ThermalCell(1000., 1000., 0., 100., 500.),)
    original = ThermalState((400.,), (300.,))
    result = advance_thermal_implicit(
        c, original, 300., 0.0, 14000., 1.0,
        enthalpy_model="constant"
    )
    assert math.isclose(result.state.solid_t_k[0], 400.1, abs_tol=1e-10)
    assert math.isclose(result.state.gas_t_k[0], 300.5, abs_tol=1e-10)
    assert abs(result.energy_balance_error_j) < 1e-7

    # Two lumped temperatures equilibrate without exchanging energy with
    # their surroundings. Implicit dt far exceeds explicit heat-exchange
    # positivity limit, with no instability or heat creation.
    isolated = (ThermalCell(1000., 1000., 1200., 0.),)
    old = ThermalState((900.,), (300.,))
    explicit_lim = max_explicit_step_s(isolated, 0., 14000.)
    huge = advance_thermal_implicit(isolated, old, 300.,
                                    0., 14000., 100.*explicit_lim,
                                    enthalpy_model="constant")
    assert 300 < huge.state.gas_t_k[0] < 600
    assert 600 < huge.state.solid_t_k[0] < 900
    assert math.isclose(
        huge.state.gas_t_k[0]+huge.state.solid_t_k[0], 1200.,
        rel_tol=1e-12
    )
    assert abs(huge.energy_balance_error_j) < 1e-6

    # Implicit axial advection must use the upstream NEW gas enthalpy,
    # without bypassing the first cell and without creating heat.
    chain = tuple(ThermalCell(1000., 14000., 0., 0.) for _ in range(4))
    initial = ThermalState((600.,)*4, (600.,)*4)
    adv = advance_thermal_implicit(chain, initial, 300., 1.0, 14000.,
                                   1., enthalpy_model="constant")
    vals = adv.state.gas_t_k
    assert 300 < vals[0] < vals[1] < vals[2] < vals[3] < 600
    assert abs(adv.energy_balance_error_j) < 1e-6

    # Variable H2 enthalpy must respond to variable NASA-7 cp(T) and
    # return the actual h(T) energy difference, not cp(T_old)*DeltaT.
    heat = (ThermalCell(3000., 14000., 0., 0., 1.5e6),)
    old = ThermalState((1200.,), (1200.,))
    nasa = advance_thermal_implicit(heat, old, 1200., 0., 14000., 1.,
                                    enthalpy_model="nasa7-h2")
    constant = advance_thermal_implicit(heat, old, 1200., 0., 14000., 1.,
                                        enthalpy_model="constant")
    assert nasa.state.gas_t_k[0] > 1200.
    assert abs(nasa.state.gas_t_k[0]-constant.state.gas_t_k[0]) > 0.1
    h_delta = (hydrogen_specific_enthalpy(nasa.state.gas_t_k[0], 14000.)
               - hydrogen_specific_enthalpy(1200., 14000.))
    assert math.isclose(h_delta, 1.5e6, rel_tol=2e-11)
    assert abs(nasa.energy_balance_error_j) < 1e-6

    # Combined OpenMC partition ledger: solid heat and DIRECT H2 heat
    # are distinct. Every axially integrated heating channel is included.
    m = OpenMCPowerMap(
        fuel_solid_w=(400., 300., 200.),
        fuel_h2_direct_w=(70., 60., 50.),
        tie_solid_w=(50., 40., 30.),
        tie_h2_direct_w=(20., 15., 10.),
        unresolved_w=155., prescribed_power_w=1400.,
        source_reported_power_w=1400.,
        source_sha256="synthetic-test-only",
        source_kind="synthetic"
    )
    # 900+180+120+45+155 = 1400 W.
    m.validate()
    fuel = Branch(5000., 14000., 100., 0.1, 14000.)
    tie = Branch(2000., 4000., 50., 0.02, 14000.)
    state = initial_state(3, 600., 350.)
    step = advance_openmc_transient(
        m, state, fuel, tie, 350., 5.,
        integrator="implicit", enthalpy_model="nasa7-h2"
    )
    assert step.state.unresolved_energy_j == 775.
    assert abs(step.energy_residual_j) < 5e-6
    assert math.isclose(step.gas_direct_power_w, 225.)
    assert step.outlet_fuel_k > 350.
    assert step.outlet_tie_k > 350.
    assert all(200 <= x <= 3500 for x in step.state.fuel.gas_t_k)

    # Feedback callback must restart from the SAME old state on each
    # Picard iteration, and reject nonconvergence.
    calls = []
    def tagged_test_callback(fuel_t, gas_t):
        calls.append((fuel_t, gas_t))
        return replace(m, source_kind="openmc-statepoint")
    result = advance_with_transport_feedback(
        tagged_test_callback, state, fuel, tie, 350., 5.,
        integrator="implicit", enthalpy_model="nasa7-h2",
        max_iterations=4, temperature_tolerance_k=1e-7,
    )
    assert result.converged and result.iterations == 2
    assert len(calls) == 2
    assert abs(result.step.energy_residual_j) < 5e-6
    assert result.step.state == step.state
    assert_raises(ValueError, lambda: advance_thermal_implicit(
        c, original, 300., 0., 14000., -0.1
    ))
    assert_raises(ValueError, lambda: advance_thermal_implicit(
        heat, old, 1200., 0., 14000., 0.1,
        enthalpy_model="unsupported"
    ))
    assert_raises(ValueError, lambda: advance_thermal_implicit(
        c, original, 180., 0., 14000., 0.1,
        enthalpy_model="nasa7-h2"
    ))
    assert_raises(ValueError, lambda: advance_openmc_transient(
        m, state, fuel, tie, 350., 1.0,
        integrator="explicit", enthalpy_model="nasa7-h2"
    ))
    assert_raises(ValueError, lambda: advance_thermal_implicit(
        (ThermalCell(1000., 14000., 0., 0., 100e6),),
        ThermalState((300.,), (300.,)), 300., 0., 14000.,
        1.0, enthalpy_model="nasa7-h2"
    ))
    print("NERVA implicit hydrogen enthalpy, advection and energy closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
