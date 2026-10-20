"""Conservative transient thermal/shaft kernel for NERVA heritage studies.

Provenance: inspired by the *architecture* described for RN-S-0114 in
Aerojet report RN-S-0150, NOT transcribed or calibrated from RN-S-0114.

This code is a minimal verification-first base for a transient engine
simulator. It is NOT an implementation of the historical Aerojet IBM 7094
code; in particular it omits neutron kinetics, temperature feedback,
pressure/momentum dynamics, valve maps and actual pump/turbine maps.

A row of well-mixed gas control volumes exchanges heat with separate
solid nodes and transports mass-specific sensible enthalpy downstream.
All state variables, energies and powers are SI.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ThermalCell:
    solid_heat_capacity_j_k: float
    gas_heat_capacity_j_k: float
    conductance_w_k: float
    deposition_w: float
    direct_gas_deposition_w: float = 0.0

    def validate(self) -> None:
        for name in ("solid_heat_capacity_j_k", "gas_heat_capacity_j_k"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("conductance_w_k", "deposition_w", "direct_gas_deposition_w"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class ThermalState:
    solid_t_k: tuple[float, ...]
    gas_t_k: tuple[float, ...]


@dataclass(frozen=True)
class ThermalStep:
    state: ThermalState
    input_power_w: float
    exchange_power_w: float
    outflow_enthalpy_power_w: float
    inflow_enthalpy_power_w: float
    stored_energy_change_j: float
    energy_balance_error_j: float
    maximum_explicit_step_s: float
    direct_gas_power_w: float = 0.0


def _allfinite_positive(values: tuple[float, ...], name: str) -> None:
    if any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError(f"{name} must be finite and positive")


def max_explicit_step_s(
    cells: tuple[ThermalCell, ...], mass_flow_kg_s: float, cp_j_kg_k: float
) -> float:
    """Conservative Euler positivity limit for a chain of thermal cells."""
    if not cells:
        raise ValueError("at least one cell required")
    if not math.isfinite(mass_flow_kg_s) or mass_flow_kg_s < 0:
        raise ValueError("mass flow must be finite and nonnegative")
    if not math.isfinite(cp_j_kg_k) or cp_j_kg_k <= 0:
        raise ValueError("cp must be finite and positive")
    limits: list[float] = []
    for c in cells:
        c.validate()
        if c.conductance_w_k > 0:
            limits.append(c.solid_heat_capacity_j_k / c.conductance_w_k)
        combined = c.conductance_w_k + mass_flow_kg_s*cp_j_kg_k
        if combined > 0:
            limits.append(c.gas_heat_capacity_j_k / combined)
    return min(limits) if limits else math.inf


def advance_thermal(
    cells: tuple[ThermalCell, ...], state: ThermalState,
    inlet_t_k: float, mass_flow_kg_s: float,
    cp_j_kg_k: float, dt_s: float
) -> ThermalStep:
    """Forward Euler; all interfaces use the OLD state to conserve energy.

    Cell i equations:
      Cs_i dTs_i/dt = Qi - Ui (Ts_i - Tg_i)
      Cg_i dTg_i/dt = Ui (Ts_i - Tg_i) + Qdirect_gas_i
                           + mdot * cp * (Tup_i - Tg_i)
      Tup_0 = inlet temperature; Tup_i = Tg_(i-1).
    The energy audit includes both solid and gas stored energy.
    """
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("positive finite timestep required")
    if len(cells) != len(state.solid_t_k) or len(cells) != len(state.gas_t_k):
        raise ValueError("thermal state dimension mismatch")
    _allfinite_positive(state.solid_t_k, "solid temperatures")
    _allfinite_positive(state.gas_t_k, "gas temperatures")
    if not math.isfinite(inlet_t_k) or inlet_t_k <= 0:
        raise ValueError("inlet temperature must be positive")
    dt_max = max_explicit_step_s(cells, mass_flow_kg_s, cp_j_kg_k)
    if dt_s > dt_max * (1 + 1e-12):
        raise ValueError("explicit timestep exceeds thermal positivity limit")
    mdotcp = mass_flow_kg_s*cp_j_kg_k
    new_solid: list[float] = []
    new_gas: list[float] = []
    q_dep = 0.0
    q_direct = 0.0
    q_transfer = 0.0
    stored = 0.0
    for i, cell in enumerate(cells):
        ts, tg = state.solid_t_k[i], state.gas_t_k[i]
        t_up = inlet_t_k if i == 0 else state.gas_t_k[i-1]
        q_wall = cell.conductance_w_k*(ts-tg)
        qs = cell.deposition_w-q_wall
        qg = q_wall+cell.direct_gas_deposition_w+mdotcp*(t_up-tg)
        ts1 = ts+dt_s*qs/cell.solid_heat_capacity_j_k
        tg1 = tg+dt_s*qg/cell.gas_heat_capacity_j_k
        if ts1 <= 0 or tg1 <= 0:
            raise ValueError("nonphysical temperature update")
        new_solid.append(ts1)
        new_gas.append(tg1)
        q_dep += cell.deposition_w+cell.direct_gas_deposition_w
        q_direct += cell.direct_gas_deposition_w
        q_transfer += q_wall
        stored += cell.solid_heat_capacity_j_k*(ts1-ts)
        stored += cell.gas_heat_capacity_j_k*(tg1-tg)
    incoming = mdotcp*inlet_t_k
    outgoing = mdotcp*state.gas_t_k[-1]
    expected = dt_s*(q_dep+incoming-outgoing)
    return ThermalStep(
        state=ThermalState(tuple(new_solid), tuple(new_gas)),
        input_power_w=q_dep, exchange_power_w=q_transfer,
        inflow_enthalpy_power_w=incoming,
        outflow_enthalpy_power_w=outgoing,
        stored_energy_change_j=stored,
        energy_balance_error_j=stored-expected,
        maximum_explicit_step_s=dt_max,
        direct_gas_power_w=q_direct,
    )


@dataclass(frozen=True)
class ShaftState:
    inertia_kg_m2: float
    omega_rad_s: float

    def validate(self) -> None:
        if not math.isfinite(self.inertia_kg_m2) or self.inertia_kg_m2 <= 0:
            raise ValueError("shaft inertia must be finite and positive")
        if not math.isfinite(self.omega_rad_s) or self.omega_rad_s < 0:
            raise ValueError("shaft angular speed must be finite and nonnegative")


@dataclass(frozen=True)
class ShaftStep:
    state: ShaftState
    starting_kinetic_energy_j: float
    ending_kinetic_energy_j: float
    net_power_w: float
    energy_balance_error_j: float


def advance_shaft(
    state: ShaftState, turbine_power_w: float, pump_power_w: float,
    loss_power_w: float, dt_s: float
) -> ShaftStep:
    """Exact finite-difference shaft energy update for piecewise constant P.

       d(0.5*I*omega^2)/dt = Pturbine - Ppump - Ploss.
    This is deliberately an energy integration, avoiding a 1/omega
    singularity at start-up. Power inputs must come from external maps.
    """
    state.validate()
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("positive timestep required")
    if any(not math.isfinite(p) or p < 0 for p in
           (turbine_power_w, pump_power_w, loss_power_w)):
        raise ValueError("shaft powers must be finite and nonnegative")
    pnet = turbine_power_w - pump_power_w - loss_power_w
    old = 0.5*state.inertia_kg_m2*state.omega_rad_s**2
    new = old+dt_s*pnet
    if new < -1e-10 * max(old, 1.0):
        raise ValueError("shaft would reverse; reduce timestep or model stall")
    new = max(0.0, new)
    omega = math.sqrt(2.0*new/state.inertia_kg_m2)
    return ShaftStep(
        state=ShaftState(state.inertia_kg_m2, omega),
        starting_kinetic_energy_j=old, ending_kinetic_energy_j=new,
        net_power_w=pnet,
        energy_balance_error_j=new-old-dt_s*pnet,
    )
