"""OpenMC normalized-tally -> time-dependent fuel/tie heat ledger.

Supports (1) real OpenMC statepoint snapshots, and (2) explicit opt-in
synthetic NPZ fixtures. The thermal solver uses two aggregate coolant
branches with axial solid/gas states and a separate unresolved-power
ledger: it is NOT per-channel NERVA/NESS transient hydraulics.

Nuclear power is externally prescribed at every snapshot. OpenMC solves
quasi-static spatial neutron transport, not the time-dependent neutron
population. Feedback-capable transport adapters update material
temperatures and bulk coolant density *uniformly*, not by cell.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable

import numpy as np

from .heritage_transient import ThermalCell, ThermalState, advance_thermal

_FIELDS = (
    "power_density_w_cm3",
    "fuel_power_density_w_cm3",
    "tie_power_density_w_cm3",
    "fuel_hydrogen_power_density_w_cm3",
    "tie_hydrogen_power_density_w_cm3",
)


@dataclass(frozen=True)
class OpenMCPowerMap:
    fuel_solid_w: tuple[float, ...]
    fuel_h2_direct_w: tuple[float, ...]
    tie_solid_w: tuple[float, ...]
    tie_h2_direct_w: tuple[float, ...]
    unresolved_w: float
    prescribed_power_w: float
    source_reported_power_w: float
    source_sha256: str
    source_kind: str

    def validate(self) -> None:
        n = len(self.fuel_solid_w)
        if n == 0 or any(len(a) != n for a in (
            self.fuel_h2_direct_w, self.tie_solid_w, self.tie_h2_direct_w
        )):
            raise ValueError("inconsistent axial power-map shape")
        for a in (self.fuel_solid_w, self.fuel_h2_direct_w,
                  self.tie_solid_w, self.tie_h2_direct_w):
            if any(not math.isfinite(p) or p < 0 for p in a):
                raise ValueError("negative/nonfinite power in heating deposition")
        if not math.isfinite(self.unresolved_w) or self.unresolved_w < 0:
            raise ValueError("invalid unresolved energy compartment")
        if self.prescribed_power_w <= 0 or not math.isfinite(self.prescribed_power_w):
            raise ValueError("invalid prescribed reactor power")
        total = sum(map(sum, (self.fuel_solid_w, self.fuel_h2_direct_w,
                              self.tie_solid_w, self.tie_h2_direct_w))) + self.unresolved_w
        if not math.isclose(total, self.prescribed_power_w, rel_tol=1e-7):
            raise ValueError("normalized heating channels do not close global power")

    @property
    def modeled_power_w(self) -> float:
        return self.prescribed_power_w-self.unresolved_w


def power_map_from_npz(
    path: Path, power_w: float, *,
    source_kind: str = "synthetic",
    require_full_channels: bool = True,
) -> OpenMCPowerMap:
    """Integrate OpenMC mesh heating and normalize to prescribed watts.

    All arrays use W/cm³ and the shared OpenMC voxel mesh in cm. Direct
    nuclear hydrogen energy is allocated ONLY to gas, never to solid.
    The remainder accounts for reflector, vessel and unscored deposition.
    No silent renormalization of individual material partitions occurs.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not math.isfinite(power_w) or power_w <= 0:
        raise ValueError("reactor power must be finite and positive")
    if source_kind not in ("synthetic", "openmc-statepoint"):
        raise ValueError("invalid source kind")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as d:
        files = set(d.files)
        core = {"dimension", "lower_left_cm", "upper_right_cm",
                "power_density_w_cm3", "fuel_power_density_w_cm3",
                "tie_power_density_w_cm3"}
        if require_full_channels:
            core.update(("fuel_hydrogen_power_density_w_cm3",
                         "tie_hydrogen_power_density_w_cm3"))
        missing = core-files
        if missing:
            raise ValueError("power map missing OpenMC fields: "+", ".join(sorted(missing)))
        dim = np.asarray(d["dimension"], dtype=int)
        lo = np.asarray(d["lower_left_cm"], dtype=float)
        hi = np.asarray(d["upper_right_cm"], dtype=float)
        if dim.shape != (3,) or lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("OpenMC mesh must be a three-dimensional regular mesh")
        if np.any(dim < 1) or np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
            raise ValueError("invalid OpenMC mesh bounds")
        if np.any(hi <= lo):
            raise ValueError("mesh upper bounds must exceed lower bounds")
        voxel_cm3 = float(np.prod((hi-lo)/dim))
        shape = tuple(int(v) for v in dim)

        def axial(name: str) -> np.ndarray:
            if name not in files and not require_full_channels:
                return np.zeros(shape[2], dtype=float)
            a = np.asarray(d[name], dtype=float)
            if a.shape != shape or np.any(~np.isfinite(a)):
                raise ValueError(f"invalid mesh heating field: {name}")
            # Statistical noise may cause slightly negative bins. Do NOT clip:
            # negative heating is a failed tally/deposition integrity check.
            if np.min(a) < -1e-10 * max(float(np.max(a)), 1):
                raise ValueError(f"negative heating field {name}")
            return a.sum(axis=(0, 1)) * voxel_cm3
        full = axial("power_density_w_cm3")
        if full.sum() <= 0:
            raise ValueError("OpenMC total heating is zero")
        groups = [axial(name) for name in _FIELDS[1:]]
        scale = float(power_w)/float(full.sum())
        groups = [g*scale for g in groups]
        total_components = float(sum(float(g.sum()) for g in groups))
        remainder = power_w - total_components
        if remainder < -1e-6 * power_w:
            raise ValueError("OpenMC material heating exceeds normalized total "
                             "(possible double-counted heating / incompatible tally)")
        if remainder < 0:
            # Floating point accumulation only; preserve exact global closure.
            raise ValueError("OpenMC heating partitions exceed total")
        out = OpenMCPowerMap(
            fuel_solid_w=tuple(float(v) for v in groups[0]),
            fuel_h2_direct_w=tuple(float(v) for v in groups[2]),
            tie_solid_w=tuple(float(v) for v in groups[1]),
            tie_h2_direct_w=tuple(float(v) for v in groups[3]),
            unresolved_w=float(remainder),
            prescribed_power_w=float(power_w),
            source_reported_power_w=float(full.sum()),
            source_sha256=digest,
            source_kind=source_kind,
        )
    out.validate()
    return out


@dataclass(frozen=True)
class Branch:
    solid_heat_capacity_j_k: float
    gas_heat_capacity_j_k: float
    ua_w_k: float
    mass_flow_kg_s: float
    cp_j_kg_k: float

    def validate(self) -> None:
        for name in ("solid_heat_capacity_j_k", "gas_heat_capacity_j_k",
                     "mass_flow_kg_s", "cp_j_kg_k"):
            x = float(getattr(self, name))
            if not math.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be positive finite")
        if not math.isfinite(self.ua_w_k) or self.ua_w_k < 0:
            raise ValueError("UA must be nonnegative finite")


@dataclass(frozen=True)
class TransientState:
    fuel: ThermalState
    tie: ThermalState
    unresolved_energy_j: float = 0.0


@dataclass(frozen=True)
class TransientStep:
    state: TransientState
    reactor_power_w: float
    modeled_power_w: float
    unresolved_power_w: float
    gas_direct_power_w: float
    outlet_fuel_k: float
    outlet_tie_k: float
    stored_energy_change_j: float
    inlet_enthalpy_power_w: float
    outlet_enthalpy_power_w: float
    energy_residual_j: float


def initial_state(n: int, solid_temp_k: float, gas_temp_k: float) -> TransientState:
    if n <= 0 or min(solid_temp_k, gas_temp_k) <= 0:
        raise ValueError("positive number of bins and temperatures required")
    a = ThermalState((float(solid_temp_k),)*n, (float(gas_temp_k),)*n)
    return TransientState(fuel=a, tie=a)


def advance_openmc_transient(
    pm: OpenMCPowerMap, state: TransientState,
    fuel: Branch, tie: Branch, inlet_t_k: float, dt_s: float
) -> TransientStep:
    """One conservative timestep using old-state, power-normalized OpenMC heat.

    The external OpenMC source is held fixed during the step; if feedback is
    enabled, a caller re-runs OpenMC at new material conditions and repeats
    this step from the SAME old transient state (Picard iteration).
    """
    pm.validate()
    fuel.validate()
    tie.validate()
    if not math.isfinite(inlet_t_k) or inlet_t_k <= 0:
        raise ValueError("positive inlet temperature required")
    n = len(pm.fuel_solid_w)

    def advance_branch(name: str, branch: Branch):
        solid = getattr(pm, name+"_solid_w")
        direct = getattr(pm, name+"_h2_direct_w")
        cells = tuple(ThermalCell(
            solid_heat_capacity_j_k=branch.solid_heat_capacity_j_k,
            gas_heat_capacity_j_k=branch.gas_heat_capacity_j_k,
            conductance_w_k=branch.ua_w_k,
            deposition_w=solid[i],
            direct_gas_deposition_w=direct[i],
        ) for i in range(n))
        previous = getattr(state, name)
        return advance_thermal(cells, previous, inlet_t_k,
                               branch.mass_flow_kg_s, branch.cp_j_kg_k, dt_s)

    f = advance_branch("fuel", fuel)
    t = advance_branch("tie", tie)
    other = state.unresolved_energy_j+pm.unresolved_w*dt_s
    if not math.isfinite(other):
        raise ValueError("unresolved energy overflow")
    stored = (f.stored_energy_change_j+t.stored_energy_change_j
              +pm.unresolved_w*dt_s)
    in_h = f.inflow_enthalpy_power_w+t.inflow_enthalpy_power_w
    out_h = f.outflow_enthalpy_power_w+t.outflow_enthalpy_power_w
    residual = stored-dt_s*(pm.prescribed_power_w+in_h-out_h)
    if abs(residual) > 1e-8*max(abs(stored), abs(dt_s*pm.prescribed_power_w), 1):
        raise RuntimeError("coupled OpenMC transient energy balance failed")
    return TransientStep(
        state=TransientState(fuel=f.state, tie=t.state,
                             unresolved_energy_j=other),
        reactor_power_w=pm.prescribed_power_w,
        modeled_power_w=pm.modeled_power_w,
        unresolved_power_w=pm.unresolved_w,
        gas_direct_power_w=f.direct_gas_power_w+t.direct_gas_power_w,
        outlet_fuel_k=f.state.gas_t_k[-1],
        outlet_tie_k=t.state.gas_t_k[-1],
        stored_energy_change_j=stored,
        inlet_enthalpy_power_w=in_h, outlet_enthalpy_power_w=out_h,
        energy_residual_j=residual,
    )


@dataclass(frozen=True)
class FeedbackStatus:
    step: TransientStep
    iterations: int
    converged: bool
    max_temperature_change_k: float
    last_map: OpenMCPowerMap


def advance_with_transport_feedback(
    provider: Callable[[float, float], OpenMCPowerMap],
    state: TransientState, fuel: Branch, tie: Branch, inlet_t_k: float,
    dt_s: float, *, max_iterations: int = 4,
    temperature_tolerance_k: float = 0.5,
) -> FeedbackStatus:
    """Quasi-static OpenMC/thermal fixed-point iteration.

    provider(fuel_material_temperature_K, hydrogen_material_temperature_K)
    MUST execute/retrieve a fresh OpenMC transport snapshot; do not
    substitute a synthetic response and claim physics validation.
    Iterations restart from the SAME old thermal state. Nonconvergence
    raises an error instead of treating results as accepted.
    """
    if max_iterations < 2 or not math.isfinite(temperature_tolerance_k) or temperature_tolerance_k <= 0:
        raise ValueError("feedback requires >=2 iterations and positive tolerance")
    solid_guess = float(np.mean(state.fuel.solid_t_k))
    gas_guess = float(np.mean(state.fuel.gas_t_k))
    if min(solid_guess, gas_guess) <= 0:
        raise ValueError("invalid thermal feedback state")
    for iteration in range(1, max_iterations+1):
        power = provider(solid_guess, gas_guess)
        if power.source_kind != "openmc-statepoint":
            raise ValueError("feedback requires actual OpenMC statepoint snapshots")
        step = advance_openmc_transient(
            power, state, fuel, tie, inlet_t_k, dt_s
        )
        # Uniform-material feedback is a reduced-order proxy, not a full
        # axial/material-instance OpenMC temperature field.
        solid_new = float(np.mean(step.state.fuel.solid_t_k))
        gas_new = float(np.mean(step.state.fuel.gas_t_k))
        delta = max(abs(solid_new-solid_guess), abs(gas_new-gas_guess))
        if iteration > 1 and delta <= temperature_tolerance_k:
            return FeedbackStatus(step, iteration, True, delta, power)
        solid_guess, gas_guess = solid_new, gas_new
    raise RuntimeError(
        f"OpenMC-temperature Picard feedback did not converge within {max_iterations} "
        f"iterations; maximum temperature update {delta:.3g} K"
    )
