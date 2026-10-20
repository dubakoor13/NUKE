"""Implicit finite-volume transient solid/H2 energy conservation.

For a well-mixed gas cell, energy is stored as m_h2*h(T), not as
C_g*T with a fixed cp, so enthalpy transport conserves energy even when
cp(T) varies strongly. Solids retain a prescribed constant heat capacity.

Local implicit backward-Euler equations:
  C_s*(Ts1-Ts0)/dt = Q_s - UA*(Ts1-Tg1)
  m_g*(h(Tg1)-h(Tg0))/dt =
       UA*(Ts1-Tg1) + Q_direct + mdot*(h(Tup1)-h(Tg1))

The upstream enthalpy is the new-time value in the preceding gas
cell, making the implicit axial advection causal. A scalar monotonic
nonlinear solve per cell is sufficient; it cannot produce spurious
negative temperatures. Every thermal step checks the first-law residual.

The NASA-7 H2 option uses the repository's pure-molecular frozen H2
thermo polynomials, 200..3500 K. It is NOT dissociating equilibrium
chemistry, a compressible pressure-wave solver, or a NERVA-certified
gas transient. Pressure and gas inventory are held fixed per cell.
"""
from __future__ import annotations

import math

from .heritage_transient import (
    ThermalCell, ThermalState, ThermalStep, max_explicit_step_s,
)
from .nasa_hydrogen import T_MIN, T_MAX, hydrogen_state


def hydrogen_specific_enthalpy(t_k: float, reference_cp_j_kg_k: float,
                               model: str = "nasa7-h2") -> float:
    """Return consistent SI mass-specific h(T) for a chosen gas model."""
    if not math.isfinite(t_k) or t_k <= 0:
        raise ValueError("hydrogen temperature must be finite and positive")
    if model == "constant":
        if not math.isfinite(reference_cp_j_kg_k) or reference_cp_j_kg_k <= 0:
            raise ValueError("constant gas cp must be positive")
        return reference_cp_j_kg_k*t_k
    if model == "nasa7-h2":
        return hydrogen_state(t_k, 101325.0, mode="frozen").enthalpy_j_kg
    raise ValueError("enthalpy model must be constant or nasa7-h2")


def advance_thermal_implicit(
    cells: tuple[ThermalCell, ...], state: ThermalState,
    inlet_t_k: float, mass_flow_kg_s: float,
    reference_cp_j_kg_k: float, dt_s: float,
    *, enthalpy_model: str = "nasa7-h2",
) -> ThermalStep:
    """Backward Euler with exact discrete energy closure at every node.

    The user-specified gas_heat_capacity_j_k is interpreted as the
    reference gas inventory m_g = C_g/cp_reference [kg]. This avoids
    creating/destroying gas mass as cp(T) changes. For NASA-7, the
    reference cp is used only for this inventory conversion and the
    reported explicit timestep bound; actual enthalpy fluxes use h(T).
    """
    if not cells:
        raise ValueError("at least one thermal cell is required")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("time step must be finite and positive")
    if not math.isfinite(reference_cp_j_kg_k) or reference_cp_j_kg_k <= 0:
        raise ValueError("reference cp must be positive")
    if not math.isfinite(mass_flow_kg_s) or mass_flow_kg_s < 0:
        raise ValueError("mass flow must be nonnegative")
    if (len(cells) != len(state.gas_t_k)
            or len(cells) != len(state.solid_t_k)):
        raise ValueError("state and cell dimensions differ")
    for c in cells:
        c.validate()
    temperatures = (*state.gas_t_k, *state.solid_t_k, inlet_t_k)
    if any(not math.isfinite(t) or t <= 0 for t in temperatures):
        raise ValueError("temperatures must be finite and positive")
    if enthalpy_model not in ("constant", "nasa7-h2"):
        raise ValueError("invalid gas enthalpy model")
    lower, upper = (T_MIN, T_MAX) if enthalpy_model == "nasa7-h2" else (1e-8, 1e7)
    if any(not lower <= t <= upper for t in temperatures):
        raise ValueError("gas state outside enthalpy model temperature domain")
    enthalpy = lambda t: hydrogen_specific_enthalpy(
        t, reference_cp_j_kg_k, enthalpy_model
    )
    h_up = enthalpy(inlet_t_k)
    h_in = h_up
    next_s: list[float] = []
    next_g: list[float] = []
    stored = 0.0
    deposited = 0.0
    direct = 0.0
    transferred = 0.0

    for i, cell in enumerate(cells):
        cs_rate = cell.solid_heat_capacity_j_k/dt_s
        ua = cell.conductance_w_k
        m_gas = cell.gas_heat_capacity_j_k/reference_cp_j_kg_k
        h_old = enthalpy(state.gas_t_k[i])

        def solid_t(tgas: float) -> float:
            return (cs_rate*state.solid_t_k[i]
                    + cell.deposition_w + ua*tgas)/(cs_rate + ua)

        def residual(tgas: float) -> float:
            wall = ua*(solid_t(tgas)-tgas)
            return (
                m_gas*(enthalpy(tgas)-h_old)/dt_s - wall
                - cell.direct_gas_deposition_w
                - mass_flow_kg_s*(h_up-enthalpy(tgas))
            )

        f_lower = residual(lower)
        f_upper = residual(upper)
        if f_lower > 0 or f_upper < 0:
            raise ValueError(
                f"implicit gas temperature in cell {i} outside "
                f"{lower:g}..{upper:g} K thermo range"
            )
        a, b = lower, upper
        for _ in range(61):
            mid = 0.5*(a+b)
            if residual(mid) < 0:
                a = mid
            else:
                b = mid
        tg = 0.5*(a+b)
        ts = solid_t(tg)
        if not math.isfinite(ts) or ts <= 0:
            raise ValueError(f"nonphysical solid temperature at cell {i}")
        h_new = enthalpy(tg)
        next_s.append(ts)
        next_g.append(tg)
        stored += (
            cell.solid_heat_capacity_j_k*(ts-state.solid_t_k[i])
            + m_gas*(h_new-h_old)
        )
        deposited += cell.deposition_w+cell.direct_gas_deposition_w
        direct += cell.direct_gas_deposition_w
        transferred += ua*(ts-tg)
        h_up = h_new

    inlet_w = mass_flow_kg_s*h_in
    outlet_w = mass_flow_kg_s*h_up
    balance = stored-dt_s*(deposited+inlet_w-outlet_w)
    scale = max(1.0, abs(stored), dt_s*deposited,
                abs(inlet_w-outlet_w)*dt_s)
    # Double-precision enthalpy subtraction can leave a few nanjoules
    # in an exactly closed lumped system; retain tight scale tolerance.
    if abs(balance) > 2e-9*scale + 1e-8:
        raise RuntimeError(
            f"implicit finite-volume energy audit failed: {balance:.6g} J"
        )
    return ThermalStep(
        state=ThermalState(tuple(next_s), tuple(next_g)),
        input_power_w=deposited, exchange_power_w=transferred,
        outflow_enthalpy_power_w=outlet_w,
        inflow_enthalpy_power_w=inlet_w,
        stored_energy_change_j=stored, energy_balance_error_j=balance,
        maximum_explicit_step_s=max_explicit_step_s(
            cells, mass_flow_kg_s, reference_cp_j_kg_k
        ),
        direct_gas_power_w=direct,
    )
