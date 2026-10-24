"""Temperature-dependent r-z neutron kinetics + resolved conjugate fuel FV.

This combines the previously verified Stage-17 multigroup temperature
tables, Stage-7 spatial neutron/precursor backward Euler, and Stage-19
radial/axial conduction + upwind channel-energy solver.

At each Picard trial, neutrons and fuel/hydrogen are each advanced ONCE
from the SAME old states with coefficients evaluated at the current trial
volume-weighted fuel temperature. There is NO arbitrary scalar reactivity
feedback and NO assumed neutron-to-thermal watts conversion.

Every trial requires externally supplied, fully partitioned energy
deposition in W. The effective fuel temperature is a simple solid-volume
homogenization (not qualified for a real NERVA fuel element or evaluated
multigroup cross sections). The resolved finite-volume thermal equations
use constant cp, prescribed conductances, and fixed coolant mass flows;
they do not determine pressures/density, H2 dissociation or choking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math
import numpy as np

from .fx2th_family_spacetime import (
    SpatialFamilyData, SpatialFamilyState, SpatialFamilyStep,
    advance_spatial_families,
)
from .fx2th_rz_temperature_tables import (
    RZTemperatureTables, sample_rz_temperature_tables,
)
from .fx2th_conjugate_fuel_transport import (
    ConjugateCoreGeometry, ConjugateCoreState,
    ConjugateCoreDeposition, ConjugateCoreStep,
    advance_conjugate_core, effective_solid_temperature_for_xs,
)


@dataclass(frozen=True)
class ResolvedFeedbackStep:
    neutron: SpatialFamilyStep
    thermal: ConjugateCoreStep
    iterations: int
    maximum_temperature_residual_k: float
    material_source_id: str
    deposition_source_id: str
    synthetic: bool


def advance_resolved_rz_feedback(
    table: RZTemperatureTables,
    geometry: ConjugateCoreGeometry,
    old_neutrons: SpatialFamilyState,
    old_thermal: ConjugateCoreState,
    dt_s: float,
    heating_from_neutrons: Callable[
        [SpatialFamilyState,SpatialFamilyData],ConjugateCoreDeposition
    ],
    *, external_source_per_m3: np.ndarray | None=None,
    temperature_tolerance_k: float=1e-8,
    relaxation: float=.75,
    max_iterations: int=100,
    thermal_stepper:Callable[
        [ConjugateCoreGeometry,ConjugateCoreState,ConjugateCoreDeposition,float],
        ConjugateCoreStep
    ] | None=None,
) -> ResolvedFeedbackStep:
    """Simultaneously converge one physical step, never cascade trial states."""
    if not callable(heating_from_neutrons):
        raise TypeError("explicit partitioned nuclear heating callback required")
    if thermal_stepper is not None and not callable(thermal_stepper):
        raise TypeError("thermal_stepper must be callable")
    run_thermal=advance_conjugate_core if thermal_stepper is None else thermal_stepper
    if (not math.isfinite(dt_s) or dt_s<=0 or
        not math.isfinite(temperature_tolerance_k) or
        not 0<temperature_tolerance_k<=1 or
        not math.isfinite(relaxation) or not 0<relaxation<=1 or
        not isinstance(max_iterations,int) or max_iterations<1):
        raise ValueError("invalid resolved feedback timestep or solver controls")
    nr,nz,nring=geometry.shape
    if (nr,nz)!=table.spatial_shape:
        raise ValueError("resolved solid/fuel and r-z neutron cells do not match")
    ng=int(table.reference_grid.diffusion_m.shape[-1])
    nf=len(table.decay_constants_s_inv)
    if (np.asarray(old_neutrons.flux).shape!=(nr*nz,ng) or
        np.asarray(old_neutrons.precursors).shape!=(nr*nz,nf) or
        np.asarray(old_thermal.fuel_temperature_k).shape!=(nr,nz,nring) or
        np.asarray(old_thermal.hydrogen_temperature_k).shape!=(nr,nz)):
        raise ValueError("initial resolved thermal and neutron mesh/group shapes mismatch")
    if (not math.isfinite(old_neutrons.time_s) or
        not math.isfinite(old_thermal.time_s) or
        old_neutrons.time_s<0 or old_thermal.time_s<0 or
        abs(old_neutrons.time_s-old_thermal.time_s)>1e-10):
        raise ValueError("old neutron and fuel/coolant times must match")
    if external_source_per_m3 is not None:
        source=np.asarray(external_source_per_m3,dtype=float)
        if (source.shape!=(nr*nz,ng) or
            np.any(~np.isfinite(source)) or np.any(source<0)):
            raise ValueError("external nuclear source must be nonnegative [cell,group]")
    else:
        source=None
    # Effective homogenization must not conceal a ring temperature
    # outside the table's evaluated T range. Every physical fuel ring
    # whose properties contribute to the average must be covered.
    knots=np.asarray(table.temperature_knots_k,dtype=float)
    def check_fuel_coverage(state):
        t=np.asarray(state.fuel_temperature_k,dtype=float)
        if np.any(t<knots[0]) or np.any(t>knots[-1]):
            raise ValueError("individual fuel ring temperature outside nuclear-data table domain")
    check_fuel_coverage(old_thermal)
    sample_rz_temperature_tables(table,effective_solid_temperature_for_xs(
        geometry,old_thermal))
    trial=old_thermal
    for iteration in range(1,max_iterations+1):
        check_fuel_coverage(trial)
        fuel_average=effective_solid_temperature_for_xs(geometry,trial)
        sampled=sample_rz_temperature_tables(table,fuel_average)
        data=sampled.material
        neutron=advance_spatial_families(data,old_neutrons,dt_s,source)
        deposition=heating_from_neutrons(neutron.state,data)
        if not isinstance(deposition,ConjugateCoreDeposition):
            raise TypeError("nuclear callback must return ConjugateCoreDeposition")
        thermal=run_thermal(geometry,old_thermal,deposition,dt_s)
        check_fuel_coverage(thermal.state)
        diff=max(
            float(np.max(np.abs(thermal.state.fuel_temperature_k-
                                trial.fuel_temperature_k))),
            float(np.max(np.abs(thermal.state.hydrogen_temperature_k-
                                trial.hydrogen_temperature_k))),
        )
        if diff<=temperature_tolerance_k:
            return ResolvedFeedbackStep(
                neutron=neutron,thermal=thermal,
                iterations=iteration,
                maximum_temperature_residual_k=diff,
                material_source_id=data.source_id,
                deposition_source_id=deposition.source_id,
                synthetic=bool(deposition.synthetic or
                               table.source_kind=="synthetic"),
            )
        trial=ConjugateCoreState(
            old_thermal.time_s,
            (1-relaxation)*trial.fuel_temperature_k+
             relaxation*thermal.state.fuel_temperature_k,
            (1-relaxation)*trial.hydrogen_temperature_k+
             relaxation*thermal.state.hydrogen_temperature_k,
        )
    raise RuntimeError("resolved fuel/neutron feedback did not converge; no step accepted")
