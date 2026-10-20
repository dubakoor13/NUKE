"""Conservative handoff from reconstructed OpenMC heating to NERVA cycle inputs.

This is a *steady power mapping*, not solid/fluid conjugate heat transfer,
a transient solution, or a new transport calculation. OpenMC provides
nuclear deposition in fuel solid, fuel H2, tie solid, and two tie H2 ducts.
A user must state how much deposited solid heat actually reaches coolant.
Unresolved heating is NEVER silently assigned to reflector or nozzle jacket.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import numpy as np

from .lh2_feed_cycle import LH2Cycle
from .regenerative_cooling import CoolingNetwork
from .resolved_transient import ResolvedHeating


@dataclass(frozen=True)
class CyclePowerLedger:
    reactor_power_w: float
    routed_openmc_power_w: float
    retained_or_unmodeled_openmc_power_w: float
    prescribed_nonopenmc_wall_heat_w: float
    cycle_heating_power_w: float
    core_heating_power_w: float
    cooling_heating_power_w: float
    tie_solid_untransferred_w: float
    fuel_solid_untransferred_w: float
    power_balance_error_w: float
    axial_fuel_elements: int
    axial_fuel_channels_per_element: int
    source_fidelity: str = "externally supplied normalized OpenMC reconstructions; transport lineage unverified"


def _fraction(label: str, value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must lie within [0,1]")


def _nonnegative(label: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")


def check_reconstruction_axial_edges(
    fuel_reconstruction: Path, tie_reconstruction: Path,
    core: LH2Cycle,
) -> None:
    """Reject power-bin index mapping when axial reconstruction grids disagree.

    Both instance reconstructions must carry the same physical axial bin
    edges as the reduced-order core model. Rebinning is not silently done.
    """
    reference = np.asarray(core.z_edges_m, dtype=float)
    for label, path in (("fuel", fuel_reconstruction),
                        ("tie", tie_reconstruction)):
        with np.load(path, allow_pickle=False) as data:
            if "z_edges_m" not in data.files:
                raise ValueError(f"{label} reconstruction lacks z_edges_m")
            edges = np.asarray(data["z_edges_m"], dtype=float)
        if (edges.shape != reference.shape
                or not np.all(np.isfinite(edges))
                or not np.allclose(edges, reference, rtol=0., atol=1e-9)):
            raise ValueError(
                f"{label} reconstruction axial grid differs from LH2Cycle"
            )


def map_openmc_heating_to_cycle(
    heating: ResolvedHeating,
    core: LH2Cycle,
    cooling: CoolingNetwork,
    *,
    fuel_solid_transfer_fraction: float,
    tie_solid_transfer_fraction: float,
    tie_solid_supply_fraction: float,
    nozzle_wall_heat_w: float,
    reflector_wall_heat_w: float,
) -> tuple[LH2Cycle, CoolingNetwork, CyclePowerLedger]:
    """Project *partitioned* source power, retaining energy not delivered.

    fuel_solid_transfer_fraction and tie_solid_transfer_fraction are
    explicitly user-prescribed heat-to-gas efficiencies (screening-level).
    The tie solid-to-gas heat is split between supply and return using an
    independently prescribed fraction. The total fuel deposition is
    projected axially, not assumed identical across 19 local channels.

    Nozzle/reflector wall heat is externally supplied, and is NOT inferred
    from the OpenMC unresolved reservoir. Core/fuel and tie deposition
    never enter the coolant network twice.
    """
    heating.validate()
    core.validate()
    cooling.validate()
    for label, value in (
        ("fuel_solid_transfer_fraction", fuel_solid_transfer_fraction),
        ("tie_solid_transfer_fraction", tie_solid_transfer_fraction),
        ("tie_solid_supply_fraction", tie_solid_supply_fraction),
    ):
        _fraction(label, value)
    for label, value in (
        ("nozzle_wall_heat_w", nozzle_wall_heat_w),
        ("reflector_wall_heat_w", reflector_wall_heat_w),
    ):
        _nonnegative(label, value)
    if heating.fuel_solid_w.shape[2] != len(core.z_edges_m)-1:
        raise ValueError("OpenMC fuel axial bins differ from core channel model")
    if heating.tie_solid_w.shape[1] != len(core.z_edges_m)-1:
        raise ValueError("OpenMC tie axial bins differ from core channel model")

    fuel_solid = float(np.sum(heating.fuel_solid_w))
    tie_solid = float(np.sum(heating.tie_solid_w))
    fwall = tuple(float(x) for x in (
        np.sum(heating.fuel_solid_w, axis=(0,1)) * fuel_solid_transfer_fraction
    ))
    fgas = tuple(float(x) for x in (
        np.sum(heating.fuel_h2_w, axis=(0,1))
    ))
    tie_solid_to_gas = tie_solid * tie_solid_transfer_fraction
    tie_supply_heat = (float(np.sum(heating.tie_supply_h2_w))
                       + tie_solid_to_gas*tie_solid_supply_fraction)
    tie_return_heat = (float(np.sum(heating.tie_return_h2_w))
                       + tie_solid_to_gas*(1.-tie_solid_supply_fraction))

    mapped_network = replace(
        cooling,
        nozzle_jacket=replace(cooling.nozzle_jacket,
                              heat_power_w=nozzle_wall_heat_w),
        reflector=replace(cooling.reflector, heat_power_w=reflector_wall_heat_w),
        tie_supply=replace(cooling.tie_supply, heat_power_w=tie_supply_heat),
        tie_return=replace(cooling.tie_return, heat_power_w=tie_return_heat),
    )
    qcool = mapped_network.total_heat_w
    mapped_core = replace(
        core,
        channel_wall_power_w=fwall,
        channel_direct_hydrogen_power_w=fgas,
        preheater_thermal_power_w=qcool,
    )
    mapped_core.validate()
    mapped_network.validate()

    qcore = math.fsum(fwall) + math.fsum(fgas)
    qopenmc = qcore + tie_supply_heat + tie_return_heat
    qretained = (heating.unresolved_power_w
                 + fuel_solid*(1.-fuel_solid_transfer_fraction)
                 + tie_solid*(1.-tie_solid_transfer_fraction))
    qexternal = nozzle_wall_heat_w + reflector_wall_heat_w
    qengine = qcore + qcool
    residual = heating.total_openmc_power_w - qopenmc - qretained
    tol = max(1e-7, 1e-9*heating.total_openmc_power_w)
    if (abs(residual) > tol
            or abs(qengine-qopenmc-qexternal) > tol):
        raise RuntimeError("OpenMC-to-engine power ledger violates conservation")
    return mapped_core, mapped_network, CyclePowerLedger(
        reactor_power_w=heating.total_openmc_power_w,
        routed_openmc_power_w=qopenmc,
        retained_or_unmodeled_openmc_power_w=qretained,
        prescribed_nonopenmc_wall_heat_w=qexternal,
        cycle_heating_power_w=qengine,
        core_heating_power_w=qcore,
        cooling_heating_power_w=qcool,
        tie_solid_untransferred_w=tie_solid*(1.-tie_solid_transfer_fraction),
        fuel_solid_untransferred_w=fuel_solid*(1.-fuel_solid_transfer_fraction),
        power_balance_error_w=residual,
        axial_fuel_elements=int(heating.fuel_solid_w.shape[0]),
        axial_fuel_channels_per_element=int(heating.fuel_solid_w.shape[1]),
    )
