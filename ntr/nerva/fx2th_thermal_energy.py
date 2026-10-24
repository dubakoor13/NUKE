"""Energy-conservative lumped cell solid/hydrogen thermal boundary for NUKE.

This intentionally does NOT model actual NERVA fuel, hydrogen thermodynamics,
reactor design geometry, heat transfer coefficients, temperature-dependent
cross sections, or perform an OpenMC transport calculation. All thermal
parameters and deposited power are supplied by the caller.

Each cell's solid and coolant are two well-mixed thermal nodes. Heat
capacities [J/K], conductance [W/K], m_dot [kg/s], cp [J/(kg K)],
inlet temperature [K]. A constant-cp coolant enthalpy export is
m_dot * cp * (T_coolant - T_inlet) [W]; this is NOT a resolved channel PDE
or a substitute for hot/cryogenic H2 equations of state.

Implicit conservation (each cell, for dt):
  C_s (Ts1-Ts0) = dt*(Q_s - H*(Ts1-Tc1))
  C_c (Tc1-Tc0) = dt*(Q_c + H*(Ts1-Tc1) - mdot*cp*(Tc1-Tin))
  total deposition = sum(Q_s + Q_c + Q_unmodeled)

Unmodeled deposition is explicitly accounted outside these two nodes, not
silently deposited into coolant. No fission energy or reactor power is
inferred from arbitrary-normalized neutron flux.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _vector(values, count: int, name: str, *, positive: bool = False):
    a=np.asarray(values,dtype=float)
    if a.shape != (count,) or not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError(f"{name} must be finite nonnegative shape ({count},)")
    if positive and np.any(a<=0):
        raise ValueError(f"{name} must be strictly positive")
    return a


@dataclass(frozen=True)
class ThermalCells:
    solid_heat_capacity_j_k: np.ndarray
    hydrogen_heat_capacity_j_k: np.ndarray
    wall_conductance_w_k: np.ndarray
    hydrogen_mass_flow_kg_s: np.ndarray
    hydrogen_cp_j_kg_k: np.ndarray
    inlet_temperature_k: np.ndarray

    def __post_init__(self):
        c=np.asarray(self.solid_heat_capacity_j_k,dtype=float)
        if c.ndim!=1 or len(c)<1:
            raise ValueError("solid heat capacities must be cellwise arrays")
        n=len(c)
        for name in ("solid_heat_capacity_j_k","hydrogen_heat_capacity_j_k",
                     "hydrogen_cp_j_kg_k","inlet_temperature_k"):
            _vector(getattr(self,name),n,name,positive=True)
        for name in ("wall_conductance_w_k","hydrogen_mass_flow_kg_s"):
            _vector(getattr(self,name),n,name)

    @property
    def cells(self):
        return len(self.solid_heat_capacity_j_k)


@dataclass(frozen=True)
class ThermalState:
    time_s: float
    solid_temperature_k: np.ndarray
    hydrogen_temperature_k: np.ndarray


@dataclass(frozen=True)
class ThermalDeposition:
    source_id: str
    solid_power_w: np.ndarray
    hydrogen_power_w: np.ndarray
    other_power_w: np.ndarray
    declared_total_w: float
    synthetic: bool = True

    def check(self, cells: int, *, relative_tolerance: float = 1e-10):
        if not isinstance(self.source_id,str) or not self.source_id.strip():
            raise ValueError("deposition source ID required")
        if not isinstance(self.synthetic,(bool,np.bool_)):
            raise ValueError("synthetic flag must be explicit boolean")
        if (not math.isfinite(self.declared_total_w)
                or self.declared_total_w<0):
            raise ValueError("declared total power must be finite nonnegative")
        q_s=_vector(self.solid_power_w,cells,"solid power")
        q_c=_vector(self.hydrogen_power_w,cells,"hydrogen power")
        q_o=_vector(self.other_power_w,cells,"unmodeled power")
        calculated=float(np.sum(q_s+q_c+q_o))
        if abs(calculated-self.declared_total_w) > (
                relative_tolerance*max(1.,calculated,self.declared_total_w)):
            raise ValueError("deposition partition does not equal declared total power")
        return q_s,q_c,q_o


@dataclass(frozen=True)
class ThermalStep:
    state: ThermalState
    fuel_storage_change_j: float
    hydrogen_storage_change_j: float
    hydrogen_net_export_j: float
    unmodeled_energy_j: float
    declared_deposited_energy_j: float
    solid_to_hydrogen_transfer_j: float
    thermal_energy_residual_j: float
    relative_energy_residual: float
    synthetic: bool


def advance_thermal_cells(
    cells: ThermalCells,
    initial: ThermalState,
    deposited: ThermalDeposition,
    dt_s: float,
) -> ThermalStep:
    """Solve 2x2 backward-Euler energy system separately in each cell."""
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("positive finite timestep required")
    if not math.isfinite(initial.time_s) or initial.time_s<0:
        raise ValueError("invalid initial time")
    n=cells.cells
    ts=_vector(initial.solid_temperature_k,n,"solid temperature",positive=True)
    th=_vector(initial.hydrogen_temperature_k,n,"hydrogen temperature",positive=True)
    qs,qc,qo=deposited.check(n)
    Cs=np.asarray(cells.solid_heat_capacity_j_k,dtype=float)
    Ch=np.asarray(cells.hydrogen_heat_capacity_j_k,dtype=float)
    H=np.asarray(cells.wall_conductance_w_k,dtype=float)
    G=(np.asarray(cells.hydrogen_mass_flow_kg_s,dtype=float)*
       np.asarray(cells.hydrogen_cp_j_kg_k,dtype=float))
    tin=np.asarray(cells.inlet_temperature_k,dtype=float)
    a=Cs/dt_s+H
    b=Ch/dt_s+H+G
    det=a*b-H*H
    if np.any(det<=0) or not np.all(np.isfinite(det)):
        raise ArithmeticError("thermal step matrix singular/nonphysical")
    rhs_s=(Cs/dt_s)*ts+qs
    rhs_h=(Ch/dt_s)*th+qc+G*tin
    ts1=(rhs_s*b+H*rhs_h)/det
    th1=(a*rhs_h+H*rhs_s)/det
    if np.any(~np.isfinite(ts1)) or np.any(~np.isfinite(th1)) or np.any(ts1<=0) or np.any(th1<=0):
        raise ArithmeticError("thermal integration produced invalid temperatures")
    delta_s=float(np.sum(Cs*(ts1-ts)))
    delta_h=float(np.sum(Ch*(th1-th)))
    export=float(dt_s*np.sum(G*(th1-tin)))
    other=float(dt_s*np.sum(qo))
    deposition=float(dt_s*deposited.declared_total_w)
    wall=float(dt_s*np.sum(H*(ts1-th1)))
    residual=delta_s+delta_h+export+other-deposition
    scale=max(1.,abs(delta_s),abs(delta_h),abs(export),abs(deposition),abs(other))
    if abs(residual)/scale>1e-9:
        raise ArithmeticError("first-law energy balance failed")
    return ThermalStep(
        state=ThermalState(initial.time_s+dt_s,ts1,th1),
        fuel_storage_change_j=delta_s,
        hydrogen_storage_change_j=delta_h,
        hydrogen_net_export_j=export,
        unmodeled_energy_j=other,
        declared_deposited_energy_j=deposition,
        solid_to_hydrogen_transfer_j=wall,
        thermal_energy_residual_j=residual,
        relative_energy_residual=abs(residual)/scale,
        synthetic=bool(deposited.synthetic),
    )
