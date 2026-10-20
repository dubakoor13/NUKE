"""Independent thermal and shaft conservation/regression checks."""
from __future__ import annotations
import math

from .heritage_transient import (
    ThermalCell, ThermalState, advance_thermal, max_explicit_step_s,
    ShaftState, advance_shaft
)


def main() -> int:
    # Independent analytic heating check, zero advection/transfer.
    cell = (ThermalCell(200.0, 100.0, 0.0, 600.0),)
    state = ThermalState((500.0,), (300.0,))
    step = advance_thermal(cell, state, 300, 0, 14000, 0.2)
    assert math.isclose(step.state.solid_t_k[0], 500.6, abs_tol=1e-12)
    assert step.state.gas_t_k[0] == 300
    assert abs(step.energy_balance_error_j) < 1e-9

    # Heat exchange with no mass flow and no deposition must preserve
    # total solid+gas enthalpy. The final temperature is the weighted mean.
    closed_cell = (ThermalCell(3000.0, 1000.0, 25.0, 0.0),)
    old = ThermalState((1000.0,), (300.0,))
    total0 = 3000*1000+1000*300
    for _ in range(4000):
        res = advance_thermal(closed_cell, old, 300, 0, 14000, 0.05)
        assert abs(res.energy_balance_error_j) < 1e-8
        old = res.state
    total1 = 3000*old.solid_t_k[0]+1000*old.gas_t_k[0]
    assert math.isclose(total1, total0, rel_tol=1e-12)
    assert abs(old.solid_t_k[0]-825.0) < 0.05
    assert abs(old.gas_t_k[0]-825.0) < 0.05

    # Chain of three cells with simultaneous advection and conduction:
    # energy balance includes inlet/outlet enthalpy across old states.
    chain = tuple(ThermalCell(5000., 600., 200., q)
                  for q in (500., 700., 900.))
    s = ThermalState((800., 820., 840.), (310., 340., 380.))
    step = advance_thermal(chain, s, 290.0, 0.01, 14000., 0.1)
    assert abs(step.energy_balance_error_j) < 1e-8
    expected = 0.1*(sum(c.deposition_w for c in chain)
                    + 0.01*14000*(290-380))
    assert math.isclose(step.stored_energy_change_j, expected, abs_tol=1e-8)

    # Explicit stability/positivity gate.
    maxdt = max_explicit_step_s(chain, 0.01, 14000)
    try:
        advance_thermal(chain, s, 290., 0.01, 14000., maxdt*1.01)
    except ValueError:
        pass
    else:
        raise AssertionError("unstable timestep accepted")

    # Shaft inertia update from integrated work.
    shaft = ShaftState(inertia_kg_m2=10.0, omega_rad_s=100.0)
    result = advance_shaft(shaft, 3000., 1700., 300., 2.0)
    assert math.isclose(result.starting_kinetic_energy_j, 50000.)
    assert math.isclose(result.ending_kinetic_energy_j, 52000.)
    assert abs(result.energy_balance_error_j) < 1e-8
    assert math.isclose(result.state.omega_rad_s,
                        math.sqrt(10400.), rel_tol=1e-14)
    zero = advance_shaft(ShaftState(3., 0.), 150., 0., 0., 2.)
    assert math.isclose(zero.ending_kinetic_energy_j, 300.)
    try:
        advance_shaft(ShaftState(1., 0.), 0., 10., 0., 1.)
    except ValueError:
        pass
    else:
        raise AssertionError("stall/negative energy silently accepted")
    print("Conservative NERVA-inspired transient thermal/shaft checks: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
