"""Explicitly supplied neutronic-thermal fixed-point interface for Stage 7.

Stage 7 neutron kinetics is advanced ONCE per physical timestep. Every
feedback iteration is a trial beginning at the SAME old flux, precursor and
thermal states. Temperature-to-material coefficients and neutron-power-to-
deposition normalization are required EXTERNAL callbacks. No reactor
coefficients, energy per fission, OpenMC deposition, control law, or hydrogen
equation of state is fabricated here.

The closure is still a two-node-per-cell *surrogate*, not the existing
NUKE resolved fuel/conduction/flow model. Use only for mathematical
verification until those physically justified components are coupled.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable
import numpy as np

from .fx2th_family_spacetime import (
    SpatialFamilyData, SpatialFamilyState, SpatialFamilyStep,
    advance_spatial_families,
)
from .fx2th_thermal_energy import (
    ThermalCells, ThermalDeposition, ThermalState, ThermalStep,
    advance_thermal_cells,
)


@dataclass(frozen=True)
class CoupledBoundaryStep:
    kinetic: SpatialFamilyStep
    thermal: ThermalStep
    iterations: int
    fixed_point_temperature_residual_k: float
    neutronic_model_id: str
    deposition_id: str
    synthetic: bool


def _same_geometry(a: SpatialFamilyData, b: SpatialFamilyData) -> bool:
    return (a.cells==b.cells and a.groups==b.groups and
            a.families==b.families and
            np.array_equal(a.volumes_m3,b.volumes_m3) and
            np.array_equal(a.group_speeds_m_s,b.group_speeds_m_s) and
            np.array_equal(a.decay_constants_s_inv,b.decay_constants_s_inv))


def advance_coupled_boundary(
    old_kinetics: SpatialFamilyState,
    old_thermal: ThermalState,
    thermal_cells: ThermalCells,
    dt_s: float,
    materials_from_temperature: Callable[[ThermalState],SpatialFamilyData],
    deposition_from_kinetics: Callable[[SpatialFamilyState,SpatialFamilyData],
                                         ThermalDeposition],
    *,
    external_neutron_source_per_m3: np.ndarray | None = None,
    temperature_tolerance_k: float = 1e-8,
    relaxation: float = 0.7,
    max_iterations: int = 80,
) -> CoupledBoundaryStep:
    """Fully re-advance neutronics/thermal from old state for each trial.

    The resulting temperature-to-neutronics dependence comes exclusively
    from the caller's externally supplied material-state function.
    A missing or nonconvergent coefficient callback is NOT silently replaced
    by a Doppler coefficient. Deposition must be in real W or explicitly
    synthetic W of consistent amplitude; no flux-to-W conversion occurs here.
    """
    if not callable(materials_from_temperature) or not callable(deposition_from_kinetics):
        raise TypeError("explicit material and deposition provider functions required")
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("dt must be positive finite")
    if (not math.isfinite(temperature_tolerance_k) or
            temperature_tolerance_k<=0 or temperature_tolerance_k>1):
        raise ValueError("temperature tolerance must be finite in (0,1] K")
    if not math.isfinite(relaxation) or not 0<relaxation<=1:
        raise ValueError("relaxation must be in (0,1]")
    if not isinstance(max_iterations,int) or max_iterations<1:
        raise ValueError("max_iterations must be positive integer")
    if abs(old_kinetics.time_s-old_thermal.time_s)>1e-10:
        raise ValueError("old thermal and neutron times must match")
    reference=materials_from_temperature(old_thermal)
    if not isinstance(reference,SpatialFamilyData):
        raise TypeError("material callback must return SpatialFamilyData")
    if thermal_cells.cells!=reference.cells:
        raise ValueError("thermal and neutronic cell meshes differ")
    if external_neutron_source_per_m3 is not None:
        source=np.asarray(external_neutron_source_per_m3,dtype=float)
        if (source.shape!=(reference.cells,reference.groups) or
                not np.all(np.isfinite(source)) or np.any(source<0)):
            raise ValueError("neutron source must be finite nonnegative [cell,group]")
    else:
        source=None

    trial=old_thermal
    for iteration in range(1,max_iterations+1):
        data=materials_from_temperature(trial)
        if not isinstance(data,SpatialFamilyData) or not _same_geometry(reference,data):
            raise ValueError("material callback changed mesh, groups, speeds or delayed families")
        kinetics=advance_spatial_families(data,old_kinetics,dt_s,source)
        deposited=deposition_from_kinetics(kinetics.state,data)
        if not isinstance(deposited,ThermalDeposition):
            raise TypeError("power callback must provide ThermalDeposition and a power budget")
        thermal=advance_thermal_cells(thermal_cells,old_thermal,deposited,dt_s)
        residual=max(
            float(np.max(np.abs(thermal.state.solid_temperature_k-trial.solid_temperature_k))),
            float(np.max(np.abs(thermal.state.hydrogen_temperature_k-trial.hydrogen_temperature_k)))
        )
        if residual<=temperature_tolerance_k:
            return CoupledBoundaryStep(
                kinetic=kinetics,thermal=thermal,iterations=iteration,
                fixed_point_temperature_residual_k=residual,
                neutronic_model_id=data.source_id,deposition_id=deposited.source_id,
                synthetic=bool(deposited.synthetic)
            )
        trial=ThermalState(
            old_thermal.time_s,
            (1-relaxation)*trial.solid_temperature_k+
            relaxation*thermal.state.solid_temperature_k,
            (1-relaxation)*trial.hydrogen_temperature_k+
            relaxation*thermal.state.hydrogen_temperature_k
        )
    raise RuntimeError("thermoneutronic boundary did not converge; no step accepted")
