"""Channel-resolved fuel and per-tie counterflow NERVA transient energy solver.

Consumes NUKE's existing *reconstructed* OpenMC power fields; does not
reconstruct or infer nuclear fuel design, neutron kinetics, pumps or valves.
All temperature states are 1-D per channel/duct; pressure fixed. The
shared solid node in each tie axial bin exchanges heat with BOTH H2 paths.

Fuel heat maps: [element, 19 channels, axial bins].
Tie maps: [tie instance, axial bins], with H2 supply flowing +z and
return flowing -z. Both ducts share the same tie-solid thermal node.
Gas masses come from specified reference gas heat capacities / cp_ref.
NASA-7 pure H2 enthalpy or constant cp is supported by the shared helper.

Uniform per-cell heat capacities and conductances are user *assumptions*,
not recovered material masses or historical heat-transfer correlations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .heritage_transient import ThermalCell, ThermalState
from .implicit_hydrogen import (
    advance_thermal_implicit, hydrogen_specific_enthalpy
)
from .nasa_hydrogen import T_MIN, T_MAX


def _read_array(data, key: str, dims: int) -> np.ndarray:
    if key not in data.files:
        raise KeyError(f"reconstruction missing {key}")
    a = np.asarray(data[key], dtype=float)
    if a.ndim != dims or not np.all(np.isfinite(a)):
        raise ValueError(f"{key} requires finite {dims}-D values")
    return a.copy()


@dataclass(frozen=True)
class ResolvedHeating:
    fuel_solid_w: np.ndarray          # [element, 19, z]
    fuel_h2_w: np.ndarray             # [element, 19, z]
    fuel_flow_kg_s: np.ndarray        # [element, 19]
    tie_solid_w: np.ndarray           # [tie, z]
    tie_supply_h2_w: np.ndarray       # [tie, z]
    tie_return_h2_w: np.ndarray       # [tie, z]
    tie_flow_kg_s: np.ndarray         # [tie]
    total_openmc_power_w: float
    unresolved_power_w: float

    def validate(self) -> None:
        f, d = self.fuel_solid_w, self.fuel_h2_w
        t = self.tie_solid_w
        if f.ndim != 3 or f.shape[0] < 1 or f.shape[1] != 19 or f.shape[2] < 1:
            raise ValueError("fuel heat must be [element,19,axial]")
        if d.shape != f.shape:
            raise ValueError("fuel coolant power shape mismatch")
        if self.fuel_flow_kg_s.shape != f.shape[:2]:
            raise ValueError("fuel flow shape mismatch")
        if t.ndim != 2 or t.shape[0] < 1 or t.shape[1] < 1:
            raise ValueError("tie heat must be [tie,axial]")
        if (self.tie_supply_h2_w.shape != t.shape
                or self.tie_return_h2_w.shape != t.shape):
            raise ValueError("tie coolant power shape mismatch")
        if self.tie_flow_kg_s.shape != (t.shape[0],):
            raise ValueError("tie flow shape mismatch")
        for value in (f, d, t, self.tie_supply_h2_w,
                      self.tie_return_h2_w):
            if not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError("all heating powers must be finite and nonnegative")
        for value in (self.fuel_flow_kg_s, self.tie_flow_kg_s):
            if not np.all(np.isfinite(value)) or np.any(value <= 0):
                raise ValueError("all coolant flows must be finite and positive")
        if (not math.isfinite(self.total_openmc_power_w)
                or self.total_openmc_power_w <= 0
                or not math.isfinite(self.unresolved_power_w)
                or self.unresolved_power_w < 0):
            raise ValueError("invalid OpenMC total or unresolved power")
        resolved = sum(float(np.sum(x)) for x in
                       (f, d, t, self.tie_supply_h2_w,
                        self.tie_return_h2_w))
        if not math.isclose(
            resolved + self.unresolved_power_w,
            self.total_openmc_power_w,
            rel_tol=1e-9, abs_tol=1e-7,
        ):
            raise ValueError("resolved plus unresolved power not conserved")

    @property
    def total_fuel_flow_kg_s(self) -> float:
        return float(np.sum(self.fuel_flow_kg_s))

    @property
    def total_tie_flow_kg_s(self) -> float:
        return float(np.sum(self.tie_flow_kg_s))


def read_resolved_heating(
    fuel_reconstruction: Path, tie_reconstruction: Path,
    mesh_fields: Path, *,
    power_tolerance: float = 1e-4,
) -> ResolvedHeating:
    """Read existing NUKE reconstruction NPZ and audit them against mesh.

    The mesh power is externally prescribed and normalized by NUKE's
    OpenMC postprocessor; it is NOT inferred from keff.
    Do not rescale independently reconstructed material arrays.
    """
    if not 0 <= power_tolerance < 0.1:
        raise ValueError("invalid tolerance")
    with np.load(fuel_reconstruction, allow_pickle=False) as f:
        fw = _read_array(f, "channel_wall_axial_power_w", 3)
        fd = _read_array(f, "channel_direct_h2_axial_power_w", 3)
        fm = _read_array(f, "branch_mass_flow_kg_s", 2)
    with np.load(tie_reconstruction, allow_pickle=False) as t:
        tw = _read_array(t, "tie_solid_axial_power_w", 2)
        ts = _read_array(t, "tie_supply_direct_h2_axial_power_w", 2)
        tr = _read_array(t, "tie_return_direct_h2_axial_power_w", 2)
        tm = _read_array(t, "tie_mass_flow_kg_s", 1)
    with np.load(mesh_fields, allow_pickle=False) as m:
        dim = _read_array(m, "dimension", 1)
        lo = _read_array(m, "lower_left_cm", 1)
        hi = _read_array(m, "upper_right_cm", 1)
        if dim.shape != (3,) or lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("invalid 3-D mesh geometry")
        if np.any(dim < 1) or np.any(hi <= lo):
            raise ValueError("invalid mesh extent/dimension")
        voxel = float(np.prod((hi-lo)/dim))
        expected_shape = tuple(dim.astype(int))
        keys = (
            "power_density_w_cm3",
            "fuel_power_density_w_cm3",
            "fuel_hydrogen_power_density_w_cm3",
            "tie_power_density_w_cm3",
            "tie_supply_hydrogen_power_density_w_cm3",
            "tie_return_hydrogen_power_density_w_cm3",
        )
        values: dict[str, float] = {}
        for key in keys:
            a = _read_array(m, key, 3)
            if a.shape != expected_shape or np.any(a < 0):
                raise ValueError(f"invalid OpenMC mesh field {key}")
            values[key] = float(a.sum()*voxel)
    total = values["power_density_w_cm3"]
    if total <= 0:
        raise ValueError("mesh reports zero reactor power")
    expected = (
        ("fuel solid", float(fw.sum()), values["fuel_power_density_w_cm3"]),
        ("fuel hydrogen", float(fd.sum()),
         values["fuel_hydrogen_power_density_w_cm3"]),
        ("tie solid", float(tw.sum()), values["tie_power_density_w_cm3"]),
        ("tie supply", float(ts.sum()),
         values["tie_supply_hydrogen_power_density_w_cm3"]),
        ("tie return", float(tr.sum()),
         values["tie_return_hydrogen_power_density_w_cm3"]),
    )
    for label, measured, reference in expected:
        if abs(measured-reference) > max(total*power_tolerance, 1e-8):
            raise ValueError(
                f"{label} power mismatch: reconstructed {measured:g} W "
                f"vs OpenMC mesh {reference:g} W"
            )
    allocated = sum(row[1] for row in expected)
    unresolved = total-allocated
    if unresolved < -total*power_tolerance:
        raise ValueError("double-counted nuclear heat partitions")
    # Don't camouflage significant negative energy with a clip.
    if unresolved < 0:
        raise ValueError("negative unresolved deposited power")
    out = ResolvedHeating(
        fw, fd, fm, tw, ts, tr, tm,
        total_openmc_power_w=total,
        unresolved_power_w=unresolved,
    )
    out.validate()
    return out


@dataclass(frozen=True)
class ResolvedParameters:
    fuel_solid_capacity_j_k: float
    fuel_h2_capacity_j_k: float
    fuel_ua_w_k: float
    tie_solid_capacity_j_k: float
    tie_supply_h2_capacity_j_k: float
    tie_return_h2_capacity_j_k: float
    tie_supply_ua_w_k: float
    tie_return_ua_w_k: float
    hydrogen_cp_reference_j_kg_k: float = 14300.0
    enthalpy_model: str = "nasa7-h2"
    coupling_tol_k: float = 1e-7
    max_coupling_iterations: int = 100

    def validate(self) -> None:
        for name in (
            "fuel_solid_capacity_j_k", "fuel_h2_capacity_j_k",
            "tie_solid_capacity_j_k", "tie_supply_h2_capacity_j_k",
            "tie_return_h2_capacity_j_k",
            "hydrogen_cp_reference_j_kg_k"
        ):
            v = getattr(self, name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("fuel_ua_w_k", "tie_supply_ua_w_k",
                     "tie_return_ua_w_k"):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.enthalpy_model not in ("nasa7-h2", "constant"):
            raise ValueError("invalid enthalpy model")
        if self.max_coupling_iterations < 2 or not math.isfinite(self.coupling_tol_k) or self.coupling_tol_k <= 0:
            raise ValueError("invalid implicit counterflow solver tolerance")


@dataclass(frozen=True)
class ResolvedState:
    fuel_solid_t_k: np.ndarray
    fuel_h2_t_k: np.ndarray
    tie_solid_t_k: np.ndarray
    tie_supply_t_k: np.ndarray
    tie_return_t_k: np.ndarray
    unresolved_energy_j: float


def initial_resolved_state(
    heating: ResolvedHeating, *, solid_temperature_k: float,
    gas_temperature_k: float,
) -> ResolvedState:
    heating.validate()
    if (not math.isfinite(solid_temperature_k)
            or not math.isfinite(gas_temperature_k)
            or min(solid_temperature_k, gas_temperature_k) <= 0):
        raise ValueError("invalid initial temperatures")
    return ResolvedState(
        np.full_like(heating.fuel_solid_w, solid_temperature_k),
        np.full_like(heating.fuel_h2_w, gas_temperature_k),
        np.full_like(heating.tie_solid_w, solid_temperature_k),
        np.full_like(heating.tie_supply_h2_w, gas_temperature_k),
        np.full_like(heating.tie_return_h2_w, gas_temperature_k),
        unresolved_energy_j=0.0,
    )


def _gas_root(
    old_t: float, solid_t: float, upstream_h: float,
    gas_mass: float, ua: float, q_direct: float,
    flow: float, dt_s: float, cp_ref: float, model: str
) -> float:
    """Monotone one-variable enthalpy equation with fixed solid temperature."""
    h = lambda t: hydrogen_specific_enthalpy(t, cp_ref, model)
    lower, upper = (T_MIN, T_MAX) if model == "nasa7-h2" else (1e-8, 1e7)
    h_old = h(old_t)
    def residual(t: float) -> float:
        ht = h(t)
        return (gas_mass*(ht-h_old)/dt_s
                + flow*(ht-upstream_h)
                + ua*(t-solid_t)-q_direct)
    if residual(lower) > 0 or residual(upper) < 0:
        raise ValueError("gas temperature left thermochemical domain")
    a, b = lower, upper
    for _ in range(56):
        mid = (a+b)*0.5
        if residual(mid) < 0:
            a = mid
        else:
            b = mid
    return (a+b)*0.5


@dataclass(frozen=True)
class ResolvedStep:
    state: ResolvedState
    energy_residual_j: float
    energy_stored_delta_j: float
    inflow_enthalpy_w: float
    outflow_enthalpy_w: float
    unresolved_power_w: float
    fuel_outlet_t_k: np.ndarray
    tie_outlet_t_k: np.ndarray
    tie_coupling_max_iterations: int


def advance_resolved(
    heat: ResolvedHeating, state: ResolvedState,
    parameters: ResolvedParameters, inlet_t_k: float, dt_s: float
) -> ResolvedStep:
    """Independent fuel channel BE + shared-solid nonlinear tie counterflow.

    All heating is fixed for the timestep; no neutron feedback is inferred.
    In tie duct, supply H2 enters at physical z=0, returns from z=L to 0.
    Thus outward supply flow joins return flow at the far end, and ONLY
    the return path z=0 end is a net hydraulic outlet.
    """
    heat.validate()
    parameters.validate()
    if (not math.isfinite(dt_s) or dt_s <= 0
            or not math.isfinite(inlet_t_k) or inlet_t_k <= 0):
        raise ValueError("invalid timestep or inlet temperature")
    if (state.fuel_solid_t_k.shape != heat.fuel_solid_w.shape
            or state.fuel_h2_t_k.shape != heat.fuel_solid_w.shape
            or state.tie_solid_t_k.shape != heat.tie_solid_w.shape
            or state.tie_supply_t_k.shape != heat.tie_solid_w.shape
            or state.tie_return_t_k.shape != heat.tie_solid_w.shape):
        raise ValueError("state dimensions do not match OpenMC reconstruction")

    model = parameters.enthalpy_model
    cp = parameters.hydrogen_cp_reference_j_kg_k
    h = lambda t: hydrogen_specific_enthalpy(t, cp, model)
    hin = h(inlet_t_k)
    n_elem, n_channels, n_fuel_z = heat.fuel_solid_w.shape
    n_tie, n_tie_z = heat.tie_solid_w.shape
    fsolid = np.empty_like(state.fuel_solid_t_k)
    fgas = np.empty_like(state.fuel_h2_t_k)
    tsolid = np.empty_like(state.tie_solid_t_k)
    tsup = np.empty_like(state.tie_supply_t_k)
    tret = np.empty_like(state.tie_return_t_k)
    stored = 0.0
    incoming = 0.0
    outgoing = 0.0
    maxiter = 0

    for el in range(n_elem):
        for ch in range(n_channels):
            cells = tuple(
                ThermalCell(
                    parameters.fuel_solid_capacity_j_k,
                    parameters.fuel_h2_capacity_j_k,
                    parameters.fuel_ua_w_k,
                    float(heat.fuel_solid_w[el,ch,i]),
                    float(heat.fuel_h2_w[el,ch,i])
                )
                for i in range(n_fuel_z)
            )
            old = ThermalState(
                tuple(state.fuel_solid_t_k[el,ch,:]),
                tuple(state.fuel_h2_t_k[el,ch,:])
            )
            res = advance_thermal_implicit(
                cells, old, inlet_t_k,
                float(heat.fuel_flow_kg_s[el,ch]), cp, dt_s,
                enthalpy_model=model
            )
            fsolid[el,ch,:] = res.state.solid_t_k
            fgas[el,ch,:] = res.state.gas_t_k
            stored += res.stored_energy_change_j
            incoming += res.inflow_enthalpy_power_w
            outgoing += res.outflow_enthalpy_power_w

    # Per-tie counterflow: coupled new-time gas and shared-solid states.
    c_s = parameters.tie_solid_capacity_j_k
    m_sup = parameters.tie_supply_h2_capacity_j_k/cp
    m_ret = parameters.tie_return_h2_capacity_j_k/cp
    us = parameters.tie_supply_ua_w_k
    ur = parameters.tie_return_ua_w_k
    for j in range(n_tie):
        flow = float(heat.tie_flow_kg_s[j])
        solid_prev = state.tie_solid_t_k[j,:]
        sup_prev = state.tie_supply_t_k[j,:]
        ret_prev = state.tie_return_t_k[j,:]
        sg = solid_prev.copy()
        sup = sup_prev.copy()
        ret = ret_prev.copy()
        for it in range(1, parameters.max_coupling_iterations+1):
            # Supply flows from z=0 to L; use each new-time upstream value.
            for i in range(n_tie_z):
                hup = hin if i == 0 else h(float(sup[i-1]))
                sup[i] = _gas_root(
                    float(sup_prev[i]), float(sg[i]), hup, m_sup,
                    us, float(heat.tie_supply_h2_w[j,i]),
                    flow, dt_s, cp, model
                )
            # Turnaround: return inlet at far end is supply outlet.
            for i in range(n_tie_z-1, -1, -1):
                hup = h(float(sup[-1])) if i==n_tie_z-1 else h(float(ret[i+1]))
                ret[i] = _gas_root(
                    float(ret_prev[i]), float(sg[i]), hup, m_ret,
                    ur, float(heat.tie_return_h2_w[j,i]),
                    flow, dt_s, cp, model
                )
            new_solid = (
                (c_s/dt_s)*solid_prev
                + heat.tie_solid_w[j,:] + us*sup + ur*ret
            ) / (c_s/dt_s+us+ur)
            delta = float(np.max(np.abs(new_solid-sg)))
            if delta <= parameters.coupling_tol_k:
                # Correct the final gas with the converged solid for
                # simultaneous BE residual auditing.
                sg = new_solid
                break
            sg = 0.5*(sg+new_solid)
        else:
            raise RuntimeError(
                f"tie #{j}: shared-solid counterflow Picard not converged "
                f"({parameters.max_coupling_iterations} iterations)"
            )
        # Recalculate fresh gas at converged solid; material correction
        # is small under the configured convergence tolerance.
        for i in range(n_tie_z):
            hup = hin if i == 0 else h(float(sup[i-1]))
            sup[i] = _gas_root(float(sup_prev[i]),float(sg[i]),hup,m_sup,us,
                               float(heat.tie_supply_h2_w[j,i]),flow,dt_s,cp,model)
        for i in range(n_tie_z-1,-1,-1):
            hup = h(float(sup[-1])) if i==n_tie_z-1 else h(float(ret[i+1]))
            ret[i] = _gas_root(float(ret_prev[i]),float(sg[i]),hup,m_ret,ur,
                               float(heat.tie_return_h2_w[j,i]),flow,dt_s,cp,model)
        tsolid[j,:],tsup[j,:],tret[j,:] = sg,sup,ret
        # Separate gas accumulations, including both passes.
        for i in range(n_tie_z):
            stored += c_s*(float(sg[i])-float(solid_prev[i]))
            stored += m_sup*(h(float(sup[i]))-h(float(sup_prev[i])))
            stored += m_ret*(h(float(ret[i]))-h(float(ret_prev[i])))
        incoming += flow*hin
        outgoing += flow*h(float(ret[0]))
        maxiter = max(maxiter,it)

    # All unmodeled reactor heat is retained in an explicit unmodeled
    # reservoir. It must not spuriously heat the modeled coolant.
    other = heat.unresolved_power_w*dt_s
    stored += other
    residual = stored-dt_s*(heat.total_openmc_power_w+incoming-outgoing)
    audit_scale = max(1.0, abs(stored), abs(dt_s*heat.total_openmc_power_w))
    if abs(residual) > 1e-7*audit_scale + 1e-6:
        raise RuntimeError(
            f"resolved transient energy conservation failed: {residual:g} J"
        )
    next_state = ResolvedState(
        fsolid, fgas, tsolid, tsup, tret,
        unresolved_energy_j=state.unresolved_energy_j+other
    )
    return ResolvedStep(
        state=next_state,
        energy_residual_j=residual,
        energy_stored_delta_j=stored,
        inflow_enthalpy_w=incoming,
        outflow_enthalpy_w=outgoing,
        unresolved_power_w=heat.unresolved_power_w,
        fuel_outlet_t_k=fgas[:,:,-1].copy(),
        tie_outlet_t_k=tret[:,0].copy(),
        tie_coupling_max_iterations=maxiter,
    )
