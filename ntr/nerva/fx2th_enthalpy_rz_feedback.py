"""Temperature-dependent r-z neutron feedback with NASA-7 enthalpy H2.

This Stage20 integration delegates the actual coupled Picard equations
to the independently tested Stage19 resolved neutron–thermal feedback,
but replaces its constant-cp conjugate fuel–H2 operator with Stage20's
nonlinear molecular-hydrogen enthalpy FV solve.

The externally supplied nuclear deposition partitions, temperature
table guards, old-state time semantics and neutron/precursor balance
checks remain unchanged. New fuel/H2 energy uses frozen H2 NASA-7
h(T) and enforces independent first-law storage/outlet enthalpy closure.

Not a high-fidelity NERVA reactor design, pressure/density coupled
gas flow, dissociating H2 equilibrium or evaluated nuclear-data run.
"""
from __future__ import annotations

from .fx2th_family_spacetime import SpatialFamilyState
from .fx2th_rz_temperature_tables import RZTemperatureTables
from .fx2th_conjugate_fuel_transport import (
    ConjugateCoreGeometry,ConjugateCoreState,ConjugateCoreDeposition,
)
from .fx2th_resolved_rz_feedback import (
    ResolvedFeedbackStep,advance_resolved_rz_feedback,
)
from .fx2th_enthalpy_conjugate_core import (
    EnthalpyCoreControls,advance_enthalpy_conjugate_core,
)
from .fx2th_family_spacetime import SpatialFamilyData

from typing import Callable
import numpy as np


def advance_enthalpy_resolved_rz_feedback(
    table:RZTemperatureTables,
    geometry:ConjugateCoreGeometry,
    old_neutrons:SpatialFamilyState,
    old_thermal:ConjugateCoreState,
    dt_s:float,
    heating_from_neutrons:Callable[
        [SpatialFamilyState,SpatialFamilyData],ConjugateCoreDeposition
    ],
    *,
    enthalpy_controls:EnthalpyCoreControls|None=None,
    external_source_per_m3:np.ndarray|None=None,
    temperature_tolerance_k:float=1e-8,
    relaxation:float=.75,
    max_iterations:int=100,
)->ResolvedFeedbackStep:
    """Advance one coupled physical timestep with real H2 molecular h(T).

    Entropy and pressure/momentum are NOT advanced. All gas cells and
    inlets must be within NASA-7 200..3500 K (unless constant mode used).
    For temperature-dependent nuclear coefficients the fuel-ring
    temperature must also stay within supplied evaluated table domain.
    """
    controls=(EnthalpyCoreControls() if enthalpy_controls is None
              else enthalpy_controls)
    if not isinstance(controls,EnthalpyCoreControls):
        raise TypeError("enthalpy_controls must be EnthalpyCoreControls")
    def thermal_solver(geo,old,power,dt):
        return advance_enthalpy_conjugate_core(
            geo,old,power,dt,controls=controls
        )
    return advance_resolved_rz_feedback(
        table,geometry,old_neutrons,old_thermal,dt_s,
        heating_from_neutrons,
        external_source_per_m3=external_source_per_m3,
        temperature_tolerance_k=temperature_tolerance_k,
        relaxation=relaxation,
        max_iterations=max_iterations,
        thermal_stepper=thermal_solver,
    )
