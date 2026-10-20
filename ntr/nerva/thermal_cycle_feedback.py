"""One-way, energy-conserving resolved solid-to-H2 wall-heat coupling.

Uses the existing backward-Euler fuel-channel and tie-counterflow solver.
The wall exchange is NOT prescribed as a fraction of nuclear deposition:
    Qwall_i = UA_i * (Tsolid_i^(n+1) - TH2_i^(n+1))
and the source is partitioned into heat delivered and solid inventory:
    Qdeposition_i = Qwall_i + Cs_i * (Ts_i^(n+1)-Ts_i^n)/dt.

The cycle operating point is still steady and independent of the transient
coolant-flow solution. That hydraulic mismatch must not be hidden or called
a fully converged conjugate NERVA solution. Uniform supplied Cs and UA are
engineering inputs, not experimental coefficients or geometry-derived maps.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .lh2_feed_cycle import LH2Cycle
from .regenerative_cooling import CoolingNetwork
from .resolved_transient import (
    ResolvedHeating, ResolvedParameters, ResolvedState, ResolvedStep,
)


@dataclass(frozen=True)
class ResolvedWallPowerLedger:
    reactor_power_w: float
    nuclear_heat_to_coolant_w: float
    solid_energy_storage_rate_w: float
    unresolved_power_w: float
    prescribed_nonopenmc_wall_heat_w: float
    engine_cycle_heating_power_w: float
    fuel_wall_to_h2_w: float
    tie_supply_wall_to_h2_w: float
    tie_return_wall_to_h2_w: float
    fuel_direct_h2_w: float
    tie_supply_direct_h2_w: float
    tie_return_direct_h2_w: float
    fuel_solid_storage_rate_w: float
    tie_solid_storage_rate_w: float
    nuclear_power_balance_error_w: float
    maximum_solid_balance_error_w: float
    resolved_thermal_step_energy_residual_j: float
    source_fidelity: str = "one-way backward-Euler thermal flux; not NERVA validated; engine flow not iterated"


def _valid_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be nonnegative finite watts")


def _arrays(heat: ResolvedHeating, old: ResolvedState, new: ResolvedState) -> None:
    for attr, shape in (
        ("fuel_solid_t_k", heat.fuel_solid_w.shape),
        ("fuel_h2_t_k", heat.fuel_h2_w.shape),
        ("tie_solid_t_k", heat.tie_solid_w.shape),
        ("tie_supply_t_k", heat.tie_solid_w.shape),
        ("tie_return_t_k", heat.tie_solid_w.shape),
    ):
        for name, state in (("old", old), ("new", new)):
            a = np.asarray(getattr(state, attr), dtype=float)
            if (a.shape != shape or not np.all(np.isfinite(a))
                    or np.any(a <= 0)):
                raise ValueError(f"invalid {name} {attr} thermal state")


def project_resolved_wall_power(
    heat: ResolvedHeating,
    old: ResolvedState,
    step: ResolvedStep,
    parameters: ResolvedParameters,
    core: LH2Cycle,
    cooling: CoolingNetwork,
    dt_s: float,
    *,
    nozzle_wall_heat_w: float,
    reflector_wall_heat_w: float,
) -> tuple[LH2Cycle, CoolingNetwork, ResolvedWallPowerLedger]:
    """Map solved implicit wall fluxes to existing steady cycle power inputs.

    Only wall-to-H2 transfer at the NEW timestep and direct nuclear H2
    deposition are forwarded. Increases/decreases of fuel/tie solid energy
    are reported separately, so transient stored energy is never counted
    as instantaneous coolant heat. No neutron kinetics is introduced.
    """
    heat.validate()
    parameters.validate()
    core.validate()
    cooling.validate()
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("positive timestep required for transient projection")
    for name, val in (
        ("nozzle_wall_heat_w", nozzle_wall_heat_w),
        ("reflector_wall_heat_w", reflector_wall_heat_w),
    ):
        _valid_nonnegative(name, val)
    if heat.fuel_solid_w.shape[2] != len(core.z_edges_m)-1:
        raise ValueError("fuel axial bins do not match engine core")
    if heat.tie_solid_w.shape[1] != len(core.z_edges_m)-1:
        raise ValueError("tie axial bins do not match engine core")
    new = step.state
    _arrays(heat, old, new)
    for name, value in (("old", old.unresolved_energy_j),
                        ("new", new.unresolved_energy_j)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid {name} unresolved stored energy")
    if not math.isfinite(step.energy_residual_j):
        raise ValueError("nonfinite resolved transient energy residual")
    scale = max(1., dt_s*heat.total_openmc_power_w)
    if abs(step.energy_residual_j) > 1e-7*scale + 1e-6:
        raise RuntimeError("upstream resolved thermal step energy balance failed")
    if (abs(new.unresolved_energy_j-old.unresolved_energy_j -
            dt_s*heat.unresolved_power_w) > 1e-8*scale):
        raise RuntimeError("unresolved OpenMC reservoir was not conserved")

    qfuel = parameters.fuel_ua_w_k * (
        new.fuel_solid_t_k - new.fuel_h2_t_k
    )
    qs = parameters.tie_supply_ua_w_k * (
        new.tie_solid_t_k - new.tie_supply_t_k
    )
    qr = parameters.tie_return_ua_w_k * (
        new.tie_solid_t_k - new.tie_return_t_k
    )
    for label, arr in (("fuel",qfuel),("tie supply",qs),("tie return",qr)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"nonfinite {label} wall flux")
        if np.any(arr < -1e-12):
            raise ValueError(
                f"{label} reverse H2-to-solid heat flux is not supported "
                "by the positive-heat engine-cycle reduction"
            )

    dfuel = parameters.fuel_solid_capacity_j_k * (
        new.fuel_solid_t_k-old.fuel_solid_t_k
    ) / dt_s
    dtie = parameters.tie_solid_capacity_j_k * (
        new.tie_solid_t_k-old.tie_solid_t_k
    ) / dt_s
    rf = heat.fuel_solid_w - qfuel - dfuel
    rt = heat.tie_solid_w - qs - qr - dtie
    local_scale = max(
        1., float(np.max(heat.fuel_solid_w)),
        float(np.max(heat.tie_solid_w)),
        float(np.max(np.abs(dfuel))),float(np.max(np.abs(dtie))),
    )
    max_error = max(float(np.max(np.abs(rf))),float(np.max(np.abs(rt))))
    if max_error > 1e-7*local_scale+1e-6:
        raise RuntimeError(
            "resolved solid heat source != wall coolant flux + stored energy"
        )

    fuel_axial_wall = tuple(
        float(v) for v in np.sum(qfuel,axis=(0,1))
    )
    fuel_axial_direct = tuple(
        float(v) for v in np.sum(heat.fuel_h2_w,axis=(0,1))
    )
    tie_supply = float(np.sum(qs)+np.sum(heat.tie_supply_h2_w))
    tie_return = float(np.sum(qr)+np.sum(heat.tie_return_h2_w))
    mapped = replace(
        cooling,
        nozzle_jacket=replace(cooling.nozzle_jacket,
                              heat_power_w=nozzle_wall_heat_w),
        reflector=replace(cooling.reflector,
                          heat_power_w=reflector_wall_heat_w),
        tie_supply=replace(cooling.tie_supply, heat_power_w=tie_supply),
        tie_return=replace(cooling.tie_return, heat_power_w=tie_return),
    )
    mapped_core = replace(
        core, channel_wall_power_w=fuel_axial_wall,
        channel_direct_hydrogen_power_w=fuel_axial_direct,
        preheater_thermal_power_w=mapped.total_heat_w,
    )
    mapped.validate()
    mapped_core.validate()

    qfuel_wall = float(np.sum(qfuel))
    qs_wall = float(np.sum(qs))
    qr_wall = float(np.sum(qr))
    qfuel_direct = float(np.sum(heat.fuel_h2_w))
    qs_direct = float(np.sum(heat.tie_supply_h2_w))
    qr_direct = float(np.sum(heat.tie_return_h2_w))
    qgas = math.fsum((
        qfuel_wall,qs_wall,qr_wall,qfuel_direct,qs_direct,qr_direct
    ))
    solid_storage = float(np.sum(dfuel)+np.sum(dtie))
    power_error = (heat.total_openmc_power_w-qgas-solid_storage-
                   heat.unresolved_power_w)
    tol = 1e-8 * max(1.,heat.total_openmc_power_w,abs(qgas),abs(solid_storage))
    if abs(power_error)>tol:
        raise RuntimeError("transient OpenMC source-to-cycle power ledger failed")
    external = nozzle_wall_heat_w + reflector_wall_heat_w
    qcycle = math.fsum((
        *fuel_axial_wall,*fuel_axial_direct,tie_supply,tie_return,external
    ))
    if not math.isclose(qcycle,qgas+external,rel_tol=1e-11,abs_tol=1e-6):
        raise RuntimeError("cycle source power double-counted")
    return mapped_core, mapped, ResolvedWallPowerLedger(
        reactor_power_w=heat.total_openmc_power_w,
        nuclear_heat_to_coolant_w=qgas,
        solid_energy_storage_rate_w=solid_storage,
        unresolved_power_w=heat.unresolved_power_w,
        prescribed_nonopenmc_wall_heat_w=external,
        engine_cycle_heating_power_w=qcycle,
        fuel_wall_to_h2_w=qfuel_wall,
        tie_supply_wall_to_h2_w=qs_wall,
        tie_return_wall_to_h2_w=qr_wall,
        fuel_direct_h2_w=qfuel_direct,
        tie_supply_direct_h2_w=qs_direct,
        tie_return_direct_h2_w=qr_direct,
        fuel_solid_storage_rate_w=float(np.sum(dfuel)),
        tie_solid_storage_rate_w=float(np.sum(dtie)),
        nuclear_power_balance_error_w=power_error,
        maximum_solid_balance_error_w=max_error,
        resolved_thermal_step_energy_residual_j=step.energy_residual_j,
    )
