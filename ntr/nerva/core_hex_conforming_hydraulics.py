"""Steady coupled core fuel/coating FEM + quasi-1D compressible H2 hydraulics.

This unifies existing NUKE components instead of making a second unrelated
thermal solver. A Picard outer iteration drives the conforming 3-D
fuel/coating conduction with film coefficients computed by the actual
quasi-steady H2 mass/momentum/stagnation-enthalpy channel solver.

Flow mass rates and inlet pressures are prescribed (no manifold solver).
The full thermal FEM simultaneously updates the 19 H2 gas enthalpy
temperatures. The channel hydraulic solver then updates pressures,
accelerative kinetic energy and local heat-transfer coefficients from
exactly those wall heat rates. The kinetic-energy change is returned
as a signed H2 energy correction, so BOTH solvers conserve stagnation
enthalpy at the coupled steady-state solution.

Only single-phase, subsonic NASA-7 FROZEN H2 is admissible; property
viscosity/conductivity and Nu/f correlations are screening assumptions.
No hydrotransient, entrance losses, pump/manifold, radiation,
historical NERVA properties, qualified evaluated KERMA or neutronic
temperature feedback is represented here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .config import NervaConfig
from .nasa_hydrogen import hydrogen_state, T_MIN, T_MAX
from .thermal import HydrogenProperties, nusselt_number
from .compressible_channel import solve_fuel_channel_compressible
from .core_hex_conforming_mesh import HexTriangleSection
from .core_hex_conforming_fem import (
    ConformingCoreState, ConformingCoreStep, solve_conforming_core_thermal,
)
from .core_hex_lateral_fv import HexCoolantBoundary
from .core_conjugate_fv import CoreSolidLaw
from .core_solid_material_tables import TabulatedCoreSolidLaw


@dataclass(frozen=True)
class CoupledCoreHydraulicStep:
    thermal: ConformingCoreStep
    pressure_pa: np.ndarray             # [z,19]
    mach_number: np.ndarray             # [z,19]
    film_w_m2_k: np.ndarray             # [z,19]
    kinetic_correction_w: np.ndarray     # [z,19]
    hydraulic_temperature_k: np.ndarray # [z,19]
    pressure_friction_loss_pa: np.ndarray
    pressure_acceleration_loss_pa: np.ndarray
    thermal_hydraulic_temperature_error_k: float
    iterations: int
    physical_nerva_validation: bool = False


def solve_conforming_core_with_hydraulics(
    section: HexTriangleSection,
    z_edges_m: np.ndarray,
    initial: ConformingCoreState,
    fuel: CoreSolidLaw|TabulatedCoreSolidLaw,
    coating: CoreSolidLaw|TabulatedCoreSolidLaw,
    solid_power_w: np.ndarray,
    *,
    mass_flow_kg_s: np.ndarray,
    inlet_temperature_k: np.ndarray,
    inlet_pressure_pa: np.ndarray,
    direct_hydrogen_power_w: np.ndarray,
    channel_diameter_m: float|None = None,
    roughness_m: float = 1e-6,
    props: HydrogenProperties|None = None,
    max_mach: float = 0.85,
    relaxation: float = 0.55,
    max_outer_iterations: int = 30,
    temperature_tolerance_k: float = 2e-3,
    film_relative_tolerance: float = 2e-4,
    kinetic_relative_tolerance: float = 2e-4,
)->CoupledCoreHydraulicStep:
    """Find fixed point of 3-D fuel conduction and steady H2 hydraulics.

    This is a screening-scale fully coupled *steady* energy + hydraulic
    closure. It does not claim a converged spatial nuclear-thermal
    OpenMC feedback problem, and it cannot solve transonic flow.
    """
    section.validate()
    fuel.validate();coating.validate()
    z=np.asarray(z_edges_m,dtype=float)
    if (z.ndim!=1 or len(z)<2 or not np.isfinite(z).all()
            or np.any(np.diff(z)<=0)):
        raise ValueError("invalid fuel axial grid")
    nz=len(z)-1
    flow=np.asarray(mass_flow_kg_s,dtype=float)
    inlet_t=np.asarray(inlet_temperature_k,dtype=float)
    inlet_p=np.asarray(inlet_pressure_pa,dtype=float)
    direct=np.asarray(direct_hydrogen_power_w,dtype=float)
    if (flow.shape!=(19,) or inlet_t.shape!=(19,) or inlet_p.shape!=(19,)
            or direct.shape!=(nz,19)
            or any(not np.isfinite(a).all() for a in
                   (flow,inlet_t,inlet_p,direct))
            or np.any(flow<=0) or np.any(inlet_p<=0)
            or np.any(direct<0)
            or np.any(inlet_t<T_MIN) or np.any(inlet_t>T_MAX)):
        raise ValueError("invalid 19-channel flow/inlet/source specification")
    D=(float(channel_diameter_m) if channel_diameter_m is not None
       else 2*NervaConfig().channel_radius_cm*0.01)
    if (not math.isfinite(D) or D<=0 or not math.isfinite(roughness_m)
            or roughness_m<0 or max_outer_iterations<2
            or not 0<relaxation<=1
            or not 0<max_mach<1
            or any(not math.isfinite(t) or t<=0 for t in
                   (temperature_tolerance_k,film_relative_tolerance,
                    kinetic_relative_tolerance))):
        raise ValueError("invalid hydraulic/outer solver controls")
    props=props or HydrogenProperties()
    for val in (
        props.dynamic_viscosity_pa_s,props.thermal_conductivity_w_m_k,
    ):
        if not math.isfinite(val) or val<=0:
            raise ValueError("positive viscosity and conductivity required")
    area=math.pi*D*D/4
    # Initial film coefficient follows the SAME screening correlation
    # used by the existing NUKE compressible channel solver.
    htc=np.empty((nz,19),dtype=float)
    gas_mass=np.empty((nz,19),dtype=float)
    for c in range(19):
        state=hydrogen_state(float(inlet_t[c]),float(inlet_p[c]),mode="frozen")
        Re=4*flow[c]/(math.pi*D*props.dynamic_viscosity_pa_s)
        Pr=(state.frozen_cp_j_kg_k*props.dynamic_viscosity_pa_s/
            props.thermal_conductivity_w_m_k)
        h0=nusselt_number(float(Re),float(Pr))*props.thermal_conductivity_w_m_k/D
        htc[:,c]=h0
        gas_mass[:,c]=state.density_kg_m3*area*np.diff(z)
    correction=np.zeros((nz,19),dtype=float)
    prev_solid=np.asarray(initial.solid_temperature_k,dtype=float)
    prev_gas=np.asarray(initial.hydrogen_temperature_k,dtype=float)
    if prev_solid.shape!=(nz+1,len(section.xy_m)) or prev_gas.shape!=(nz,19):
        raise ValueError("initial conforming state dimensions invalid")
    last_error=math.inf

    for outer in range(1,max_outer_iterations+1):
        boundary=HexCoolantBoundary(
            mdot_kg_s=flow,inlet_temperature_k=inlet_t,
            hydrogen_mass_kg=gas_mass,
            film_w_m2_k=htc,
            direct_hydrogen_power_w=direct,
            hydrogen_model="nasa7-h2")
        # Initial is an iteration GUESS; dt_s=None avoids artificial
        # physical thermal energy accumulation between outer passes.
        thermal=solve_conforming_core_thermal(
            section,z,initial,boundary,fuel,coating,solid_power_w,
            dt_s=None,hydrogen_kinetic_power_w=correction)
        new_film=np.empty_like(htc)
        new_correction=np.empty_like(correction)
        new_p=np.empty_like(htc)
        new_mach=np.empty_like(htc)
        new_T=np.empty_like(htc)
        new_dp_f=np.empty_like(htc)
        new_dp_acc=np.empty_like(htc)
        for c in range(19):
            wall=np.asarray(thermal.wall_to_hydrogen_w[:,c],dtype=float)
            if np.min(wall)<-1e-10:
                raise ValueError(
                    "coolant wall heating became negative; the existing "
                    "compressible channel model requires nonnegative input")
            result=solve_fuel_channel_compressible(
                axial_total_fuel_power_w=np.maximum(wall,0.),
                z_edges_m=z,
                fuel_channel_count=1,
                mass_flow_per_channel_kg_s=float(flow[c]),
                inlet_temperature_k=float(inlet_t[c]),
                inlet_pressure_pa=float(inlet_p[c]),
                channel_diameter_m=D,
                properties=props,
                roughness_m=roughness_m,
                axial_direct_coolant_power_w=direct[:,c],
                max_mach=max_mach,
                chemistry_mode="frozen")
            new_p[:,c]=result.pressure_pa
            new_T[:,c]=result.bulk_temperature_k
            new_mach[:,c]=result.mach_number
            new_film[:,c]=result.heat_transfer_coefficient_w_m2_k
            new_dp_f[:,c]=result.friction_pressure_loss_pa
            new_dp_acc[:,c]=result.acceleration_pressure_loss_pa
            previous_p=float(inlet_p[c])
            previous_t=float(inlet_t[c])
            mass_flux=float(flow[c]/area)
            for i in range(nz):
                state_in=hydrogen_state(previous_t,previous_p,mode="frozen")
                state_out=hydrogen_state(
                    float(new_T[i,c]),float(new_p[i,c]),mode="frozen")
                v_in=mass_flux/state_in.density_kg_m3
                v_out=mass_flux/state_out.density_kg_m3
                new_correction[i,c]=(
                    -0.5*flow[c]*(v_out*v_out-v_in*v_in))
                previous_p=float(new_p[i,c])
                previous_t=float(new_T[i,c])
        temp_error=float(np.max(np.abs(new_T-thermal.state.hydrogen_temperature_k)))
        film_error=float(np.max(np.abs(new_film-htc)/np.maximum(1.,new_film)))
        qscale=np.maximum(1.,np.abs(thermal.wall_to_hydrogen_w)+direct)
        kinetic_error=float(np.max(np.abs(new_correction-correction)/qscale))
        last_error=max(temp_error,film_error,kinetic_error)
        if (temp_error<=temperature_tolerance_k
                and film_error<=film_relative_tolerance
                and kinetic_error<=kinetic_relative_tolerance):
            return CoupledCoreHydraulicStep(
                thermal=thermal,
                pressure_pa=new_p,
                mach_number=new_mach,
                film_w_m2_k=new_film,
                kinetic_correction_w=new_correction,
                hydraulic_temperature_k=new_T,
                pressure_friction_loss_pa=new_dp_f,
                pressure_acceleration_loss_pa=new_dp_acc,
                thermal_hydraulic_temperature_error_k=temp_error,
                iterations=outer)
        htc=relaxation*new_film+(1-relaxation)*htc
        correction=relaxation*new_correction+(1-relaxation)*correction
        # No silent fallback to prescribed flow/constant h.
    raise RuntimeError(
        "conforming core H2 pressure/thermal outer fixed point did not "
        f"converge after {max_outer_iterations} passes; error={last_error:g}")
