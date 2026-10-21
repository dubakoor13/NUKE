"""Two-way tie-Euler <-> finite-volume common-header thermodynamic feedback.

The existing shaft-closed NERVA engine and regenerative network set
REFERENCE pump/tie/nozzle branch conditions and core mass demand.
Unlike one-way reservoir forcing, the finite-volume header carries
H2 MASS and INTERNAL ENERGY in a rigid, stationary mixing volume.
Its CoolProp p(rho,u), h and T become the NEXT timestep's right
Euler pressure/backflow enthalpy boundary. The Euler return uses a
single conservative Rusanov mass and TOTAL-energy face flux, and
the header consumes those EXACT same fluxes (no mdot*h substitution).

The other nozzle+reflector branch contributes its independent solved
steady flow and outlet total enthalpy. Header discharge toward the
core/turbine has the independently solved steady tank mass demand
rather than the dynamic nozzle/shaft mass constraint. This is
therefore REAL two-way transient tie/header pressure and enthalpy
feedback, NOT full two-way pump/turbine/core/nozzle closure.

The model neglects mixing-volume bulk velocity, wall heat and
kinetic energy, does not resolve turbulent mixing, shock reflection,
characteristic junction geometry, nozzle transient flow or reactor
neutronic feedback. Stationary-header outflow uses h from p(rho,u).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .distributed_tie_euler import (
    DistributedTieParameters,DistributedTieState,DistributedTieStep,
    advance_distributed_tie,
)
from .distributed_tie_engine_boundary import (
    EngineTieReservoirs,engine_tie_reservoirs,
)
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import ResolvedHeating
from .turbine_bleed_cycle import ShaftClosedPoint


@dataclass(frozen=True)
class DynamicHeaderParameters:
    volume_m3: float
    minimum_mass_kg: float = 1e-9
    minimum_pressure_pa: float = 100.
    maximum_pressure_pa: float = 50e6
    minimum_temperature_k: float = 17.
    maximum_temperature_k: float = 900.

    def validate(self):
        for name in ("volume_m3","minimum_mass_kg",
                     "minimum_pressure_pa","maximum_pressure_pa",
                     "minimum_temperature_k","maximum_temperature_k"):
            val=getattr(self,name)
            if not math.isfinite(val) or val<=0:
                raise ValueError(f"dynamic header {name} must be positive finite")
        if self.minimum_pressure_pa>=self.maximum_pressure_pa:
            raise ValueError("invalid dynamic header pressure range")
        if self.minimum_temperature_k>=self.maximum_temperature_k:
            raise ValueError("invalid dynamic header temperature range")


@dataclass(frozen=True)
class DynamicHeaderState:
    mass_kg: float
    internal_energy_j: float
    def validate(self):
        if not math.isfinite(self.mass_kg) or self.mass_kg<=0:
            raise ValueError("dynamic header inventory must be positive")
        if not math.isfinite(self.internal_energy_j):
            raise ValueError("dynamic header internal energy must be finite")


@dataclass(frozen=True)
class DynamicHeaderTieState:
    tie: DistributedTieState
    header: DynamicHeaderState


@dataclass(frozen=True)
class DynamicHeaderTieStep:
    state: DynamicHeaderTieState
    tie_step: DistributedTieStep
    header_old_pressure_pa: float
    header_new_pressure_pa: float
    header_old_enthalpy_j_kg: float
    header_new_enthalpy_j_kg: float
    header_new_temperature_k: float
    header_new_density_kg_m3: float
    nozzle_branch_inlet_flow_kg_s: float
    nozzle_branch_inlet_enthalpy_j_kg: float
    tie_inlet_flow_kg_s: float
    tie_header_interface_flow_kg_s: float
    tie_header_interface_total_energy_power_w: float
    fixed_engine_discharge_flow_kg_s: float
    header_inventory_rate_kg_s: float
    header_internal_energy_rate_w: float
    header_mass_residual_kg_s: float
    header_internal_energy_residual_j: float
    combined_tie_header_mass_residual_kg_s: float
    combined_tie_solid_header_energy_residual_j: float
    combined_tie_header_momentum_residual_n: float
    full_shaft_cycle_feedback_converged: bool = False
    tie_header_pressure_feedback_active: bool = True
    dynamic_core_flow_computed: bool = False
    actual_openmc_transport_executed: bool = False


def header_properties(
    state: DynamicHeaderState,
    params: DynamicHeaderParameters,
    *,
    eos: HydrogenEOS | None = None,
) -> tuple[float,float,float,float]:
    """Return (p,h,T,rho) at rigid volume with CoolProp H2 EOS."""
    state.validate()
    params.validate()
    if state.mass_kg<=params.minimum_mass_kg:
        raise ValueError("header hydrogen inventory depleted")
    eos=eos or HydrogenEOS()
    rho=state.mass_kg/params.volume_m3
    u=state.internal_energy_j/state.mass_kg
    p=eos.get("P","Dmass",rho,"Umass",u)
    t=eos.get("T","Dmass",rho,"Umass",u)
    h=u+p/rho
    if not all(math.isfinite(x) for x in (p,h,t,rho)):
        raise ValueError("invalid dynamic header H2 EOS result")
    if not (params.minimum_pressure_pa<p<params.maximum_pressure_pa
            and params.minimum_temperature_k<t<params.maximum_temperature_k):
        raise ValueError("header H2 pressure/temperature outside declared domain")
    return p,h,t,rho


def initial_dynamic_header(
    shaft: ShaftClosedPoint,
    params: DynamicHeaderParameters,
    *,
    eos: HydrogenEOS | None = None,
) -> DynamicHeaderState:
    """Fill fixed header volume from the solved warm steady H2 state."""
    params.validate()
    eos=eos or HydrogenEOS()
    boundary=engine_tie_reservoirs(shaft)
    p=boundary.steady_common_header_pressure_pa
    h=boundary.steady_common_header_stagnation_enthalpy_j_kg
    rho=eos.hp("Dmass",h,p)
    u=eos.hp("Umass",h,p)
    state=DynamicHeaderState(
        mass_kg=rho*params.volume_m3,
        internal_energy_j=rho*params.volume_m3*u,
    )
    header_properties(state,params,eos=eos)
    return state


def advance_dynamic_header_tie(
    heating: ResolvedHeating,
    old: DynamicHeaderTieState,
    distributed_params: DistributedTieParameters,
    header_params: DynamicHeaderParameters,
    shaft: ShaftClosedPoint,
    dt_s: float,
    *,
    eos: HydrogenEOS | None = None,
) -> DynamicHeaderTieStep:
    """Explicit split-step tie + dynamic header with exact SHARED face fluxes.

    All face/source fluxes use old-time thermodynamic conditions.
    Header new EOS p/h drive the following tie Euler timestep.
    """
    heating.validate()
    distributed_params.validate()
    header_params.validate()
    old.tie.validate(heating)
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("header/tie timestep must be finite positive")
    eos=eos or HydrogenEOS()
    bc=engine_tie_reservoirs(shaft)
    p_old,h_old,_,_=header_properties(old.header,header_params,eos=eos)
    hydraulic=shaft.regenerative_cooling
    assert hydraulic is not None
    nozzle_leg=hydraulic.legs[1]
    m_nozzle=hydraulic.nozzle_reflector_mass_flow_kg_s
    h_nozzle=nozzle_leg.outlet_stagnation_enthalpy_j_kg
    m_core=shaft.tank_mass_flow_kg_s
    if (not math.isfinite(m_nozzle) or m_nozzle<=0 or
            not math.isfinite(h_nozzle) or m_core<=0 or
            not math.isclose(
                m_nozzle+bc.steady_tie_branch_mass_flow_kg_s,m_core,
                rel_tol=1e-9,abs_tol=1e-10
            )):
        raise ValueError("steady nozzle / tie / header pump split not closed")

    tie=advance_distributed_tie(
        heating,old.tie,distributed_params,
        bc.upstream_pressure_pa,
        bc.upstream_stagnation_enthalpy_j_kg,
        p_old,h_old,dt_s,eos=eos,
    )
    # The only tie->header interface carries these conservative Euler
    # fluxes, with signed backflow handled by the same numerical flux.
    tie_m=float(np.sum(tie.interface_mass_flux_kg_s[:,-1]))
    tie_energy=float(np.sum(tie.interface_total_energy_flux_w[:,-1]))
    tie_in=float(np.sum(tie.interface_mass_flux_kg_s[:,0]))
    tie_in_energy=float(np.sum(tie.interface_total_energy_flux_w[:,0]))

    dm=dt_s*(m_nozzle+tie_m-m_core)
    dE=dt_s*(m_nozzle*h_nozzle+tie_energy-m_core*h_old)
    new_header=DynamicHeaderState(
        mass_kg=old.header.mass_kg+dm,
        internal_energy_j=old.header.internal_energy_j+dE,
    )
    p_new,h_new,t_new,rho_new=header_properties(
        new_header,header_params,eos=eos,
    )
    hdr_rate=(new_header.mass_kg-old.header.mass_kg)/dt_s
    hdr_e_rate=(new_header.internal_energy_j-
                old.header.internal_energy_j)/dt_s
    hdr_mass_res=hdr_rate-(m_nozzle+tie_m-m_core)
    hdr_energy_res=(
        new_header.internal_energy_j-old.header.internal_energy_j-dE
    )
    tie_old_mass=float(np.sum(old.tie.mass_kg))
    tie_new_mass=float(np.sum(tie.state.mass_kg))
    combined_m=(tie_new_mass-tie_old_mass+
                new_header.mass_kg-old.header.mass_kg)/dt_s-(
                    tie_in+m_nozzle-m_core
                )
    # Solid sensible heat, fluid TOTAL energy including its kinetic
    # share, and header fluid INTERNAL energy; shared tie face cancels.
    Cs=distributed_params.thermal.solid_capacity_j_k
    ds=float(np.sum(Cs*(
        tie.state.solid_temperature_k-old.tie.solid_temperature_k
    )))
    de=float(np.sum(
        tie.state.total_energy_j-old.tie.total_energy_j
    ))
    combined_energy=(
        ds+de+new_header.internal_energy_j-old.header.internal_energy_j
    )-dt_s*(
        tie.nuclear_deposition_w+
        tie_in_energy+m_nozzle*h_nozzle-m_core*h_old
    )
    combined_mom=tie.momentum_residual_n
    # Bound roundoff by actual stored inventories, not unrelated
    # nuclear heating powers; guards never replace conservative flux.
    eps=np.finfo(float).eps
    mass_tolerance=(
        1e-9+8.*eps*(
            abs(tie_old_mass)+abs(tie_new_mass)+
            old.header.mass_kg+new_header.mass_kg
        )/dt_s
    )
    energy_scale=(
        abs(old.header.internal_energy_j)+abs(new_header.internal_energy_j)+
        float(np.sum(abs(old.tie.total_energy_j)+
                     abs(tie.state.total_energy_j)))+
        float(np.sum(
            Cs*(abs(old.tie.solid_temperature_k)+
                abs(tie.state.solid_temperature_k))
        ))
    )
    energy_tolerance=1e-6+8.*eps*energy_scale
    if (abs(hdr_mass_res)>mass_tolerance
            or abs(combined_m)>mass_tolerance
            or abs(hdr_energy_res)>energy_tolerance
            or abs(combined_energy)>energy_tolerance
            or abs(tie.total_energy_residual_j)>energy_tolerance):
        raise RuntimeError(
            "dynamic header/tie conservative first-law or mass audit FAILED: "
            f"header dm={hdr_mass_res:.6g} kg/s, "
            f"combined dm={combined_m:.6g} kg/s, "
            f"header dE={hdr_energy_res:.6g} J, "
            f"combined dE={combined_energy:.6g} J"
        )
    return DynamicHeaderTieStep(
        state=DynamicHeaderTieState(tie=tie.state,header=new_header),
        tie_step=tie,
        header_old_pressure_pa=p_old,
        header_new_pressure_pa=p_new,
        header_old_enthalpy_j_kg=h_old,
        header_new_enthalpy_j_kg=h_new,
        header_new_temperature_k=t_new,
        header_new_density_kg_m3=rho_new,
        nozzle_branch_inlet_flow_kg_s=m_nozzle,
        nozzle_branch_inlet_enthalpy_j_kg=h_nozzle,
        tie_inlet_flow_kg_s=tie_in,
        tie_header_interface_flow_kg_s=tie_m,
        tie_header_interface_total_energy_power_w=tie_energy,
        fixed_engine_discharge_flow_kg_s=m_core,
        header_inventory_rate_kg_s=hdr_rate,
        header_internal_energy_rate_w=hdr_e_rate,
        header_mass_residual_kg_s=hdr_mass_res,
        header_internal_energy_residual_j=hdr_energy_res,
        combined_tie_header_mass_residual_kg_s=combined_m,
        combined_tie_solid_header_energy_residual_j=combined_energy,
        combined_tie_header_momentum_residual_n=combined_mom,
    )
