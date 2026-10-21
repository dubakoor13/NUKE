"""Inertial interface-momentum surrogate for cryogenic NERVA tie H2.

Adds finite face-fluid inertance to the PR #31 conservative fixed-volume
mass/internal-energy model. A pressure difference does not instantly
produce the quasi-steady orifice flow. For a face with effective length
L, area A and homogeneous upwind density rho, a lumped momentum law is

    (L/A)*d(mdot)/dt = p_left - p_right
                      - K*mdot*abs(mdot)/(2*rho*A*A),   K=Cd^-2.

Pressure and density on the RHS are frozen over one timestep. The
nonlinear drag term is evaluated at the NEW signed face flux, and
has a closed-form backward-Euler root. This is an IMEX momentum step
feeding the existing conservative explicit fluid/solid energy and
mass finite volumes. One flux is shared across adjacent CVs including
the physical supply-to-return turnaround.

This is a lumped inertance/restriction SURROGATE, not the full 1D
compressible two-phase Euler/Navier-Stokes momentum equation: no
convective velocity flux, momentum storage inside fluid CVs, choking,
slip, wave propagation, liquid flashing/CHF or interface kinetic
energy in the transient energy ledger. Resulting mass and INTERNAL
energy closure is valid for the implemented screening equations,
NOT a total-energy kinetic balance or hardware-certified NERVA model.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .isochoric_tie import IsochoricTieState, _properties
from .lh2_feed_cycle import HydrogenEOS
from .pressure_driven_tie import (
    PressureTieFlowParameters, PressureTieFlowStep, advance_pressure_driven_tie,
)
from .resolved_transient import ResolvedHeating


@dataclass(frozen=True)
class InertialTieParameters:
    thermal_hydraulic: PressureTieFlowParameters
    interface_inertial_length_m: float

    def validate(self) -> None:
        self.thermal_hydraulic.validate()
        if (not math.isfinite(self.interface_inertial_length_m)
                or self.interface_inertial_length_m <= 0):
            raise ValueError("interface inertial length must be positive and finite")


@dataclass(frozen=True)
class InertialTieState:
    thermal: IsochoricTieState
    interface_mass_flow_kg_s: np.ndarray

    def validate(self, heating: ResolvedHeating) -> None:
        self.thermal.validate(heating)
        shape=heating.tie_solid_w.shape
        flux=np.asarray(self.interface_mass_flow_kg_s)
        if (flux.shape!=(shape[0],2*shape[1]+1)
                or not np.all(np.isfinite(flux))):
            raise ValueError("inertial mass-flow state must be finite [tie,2*Nz+1]")


@dataclass(frozen=True)
class InertialTieStep:
    state: InertialTieState
    fluid: PressureTieFlowStep
    driving_pressure_pa: np.ndarray
    drag_pressure_pa: np.ndarray
    acceleration_pressure_pa: np.ndarray
    maximum_momentum_residual_pa: float
    maximum_fractional_pressure_jump: float
    momentum_law: str = "semi-implicit finite face inertance + quadratic drag"
    full_transient_momentum_equations: bool = False
    kinetic_energy_accounted_for: bool = False


def advance_inertial_tie(
    heating: ResolvedHeating,
    old: InertialTieState,
    parameters: InertialTieParameters,
    inlet_pressure_pa: float,
    inlet_enthalpy_j_kg: float,
    outlet_pressure_pa: float,
    outlet_backflow_enthalpy_j_kg: float,
    dt_s: float,
    *,
    eos: HydrogenEOS | None = None,
) -> InertialTieStep:
    """One finite-inertia momentum step followed by conservative FV step.

    This code uses the old H2 EOS state for each interface pressure
    difference and upwind density, then transfers the NEW face flux
    to the SAME existing cell mass/internal-energy integrator.
    """
    heating.validate()
    parameters.validate()
    old.validate(heating)
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("inertial interface timestep must be positive")
    for label,p in (("inlet",inlet_pressure_pa),("outlet",outlet_pressure_pa)):
        if not math.isfinite(p) or p <= 0:
            raise ValueError(f"invalid {label} pressure")
    for label,h in (("inlet",inlet_enthalpy_j_kg),
                    ("outlet backflow",outlet_backflow_enthalpy_j_kg)):
        if not math.isfinite(h):
            raise ValueError(f"invalid {label} enthalpy")
    eos=eos or HydrogenEOS()
    pr=parameters.thermal_hydraulic
    area=pr.interface_area_m2
    length=parameters.interface_inertial_length_m
    k_loss=1./(pr.discharge_coefficient**2)
    old_cv=old.thermal
    nz=heating.tie_solid_w.shape[1]
    mass=np.concatenate((old_cv.supply_mass_kg,
                         old_cv.return_mass_kg[:,::-1]),axis=1)
    u=np.concatenate((old_cv.supply_internal_energy_j_kg,
                      old_cv.return_internal_energy_j_kg[:,::-1]),axis=1)
    vol=np.array(
        [pr.thermal.supply_cell_volume_m3]*nz+
        [pr.thermal.return_cell_volume_m3]*nz,
    )[None,:]
    density=mass/vol
    p_cells=np.empty_like(u)
    for idx in np.ndindex(u.shape):
        p_cells[idx]=_properties(
            eos,float(density[idx]),float(u[idx])
        )[0]
    in_rho=eos.hp("Dmass",inlet_enthalpy_j_kg,inlet_pressure_pa)
    out_rho=eos.hp(
        "Dmass",outlet_backflow_enthalpy_j_kg,outlet_pressure_pa
    )
    if not all(math.isfinite(x) and x>0 for x in (in_rho,out_rho)):
        raise ValueError("reservoir H2 density invalid")

    dp_field=np.empty_like(old.interface_mass_flow_kg_s)
    dp_loss=np.empty_like(dp_field)
    dp_accel=np.empty_like(dp_field)
    newflux=np.empty_like(dp_field)
    frac_max=0.
    for j in range(dp_field.shape[0]):
        for i in range(dp_field.shape[1]):
            pleft=float(inlet_pressure_pa if i==0 else p_cells[j,i-1])
            pright=float(outlet_pressure_pa
                         if i==dp_field.shape[1]-1 else p_cells[j,i])
            dp=pleft-pright
            # Mirror the pressure-driven solver's EOS roundoff band.
            if abs(dp)<=1e-8*max(1.,pleft,pright):
                dp=0.
            jump=abs(dp)/max(pleft,pright)
            frac_max=max(frac_max,jump)
            if jump>pr.max_fractional_pressure_drop:
                raise ValueError(
                    "face pressure jump exceeds nonchoked lumped-momentum scope"
                )
            old_flow=float(old.interface_mass_flow_kg_s[j,i])
            driving=old_flow+dt_s*(area/length)*dp
            # Flow can persist AGAINST dp because of momentum. Use
            # the implicit flux sign, not dp sign, for upwind rho.
            if driving>=0:
                rho=float(in_rho if i==0 else density[j,i-1])
            else:
                rho=float(out_rho
                          if i==dp_field.shape[1]-1 else density[j,i])
            drag_coefficient=(
                dt_s*k_loss/(2.*rho*area*length)
            )
            # Backward-Euler positive quadratic drag exact root.
            # Flux new + b*flux new*abs(flux new) = driving.
            value=2.*driving/(
                1.+math.sqrt(1.+4.*drag_coefficient*abs(driving))
            )
            newflux[j,i]=value
            dp_field[j,i]=dp
            dp_loss[j,i]=k_loss*value*abs(value)/(2.*rho*area*area)
            dp_accel[j,i]=(length/area)*(value-old_flow)/dt_s
    residual=dp_field-dp_loss-dp_accel
    maximum=float(np.max(np.abs(residual)))
    scale=max(1.,float(np.max(abs(dp_field))),
              float(np.max(abs(dp_loss))),float(np.max(abs(dp_accel))))
    if maximum>1e-9*scale+1e-6:
        raise RuntimeError("inertial interface momentum pressure closure failure")

    fluid=advance_pressure_driven_tie(
        heating,old_cv,pr,
        inlet_pressure_pa,inlet_enthalpy_j_kg,
        outlet_pressure_pa,outlet_backflow_enthalpy_j_kg,dt_s,
        eos=eos,interface_mass_flow_kg_s=newflux,
    )
    if not np.array_equal(fluid.interface_mass_flow_kg_s,newflux):
        raise RuntimeError("momentum and fluid CV interface fluxes differ")
    return InertialTieStep(
        state=InertialTieState(
            thermal=fluid.state,
            interface_mass_flow_kg_s=newflux.copy(),
        ),
        fluid=fluid,
        driving_pressure_pa=dp_field,
        drag_pressure_pa=dp_loss,
        acceleration_pressure_pa=dp_accel,
        maximum_momentum_residual_pa=maximum,
        maximum_fractional_pressure_jump=frac_max,
    )
