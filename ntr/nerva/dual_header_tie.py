"""Two dynamic H2 reservoirs coupled to the NERVA tie Euler and nozzle branch.

Conservative open system:
  pump manifold:       dM/dt = mdot_pump_ref - mdot_nozzle - Fm_tie_in
                       d(Mu)/dt = mdot_pump_ref*h_pump_ref
                                  -mdot_nozzle*h_pump_header - FE_tie_in
  distributed tie:     mass, axial momentum, total energy (incl kinetic)
  nozzle+reflector:    mdot conserved; dH = Q_nozzle + Q_reflector
  common header:       dM/dt = mdot_nozzle + Fm_tie_out - mdot_core_ref
                       d(Mu)/dt = mdot_nozzle*h_branch_out + FE_tie_out
                                  -mdot_core_ref*h_common_header

The same signed Rusanov Fm,FE at tie inlet/outlet appears with
opposite signs in adjacent header/tie CVs. Pump-manifold EOS pressure
and enthalpy feed BOTH hydraulic branches on the NEXT step.

The pump flow, pump delivered stagnation enthalpy and engine/core
outflow demand are STILL externally prescribed from a shaft-closed
steady reference. Dynamic pump head/shaft/turbine/nozzle closure is
not claimed. Headers have no kinetic inventory, wall heating or
resolved momentum; the nozzle branch is quasi-steady.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

import numpy as np

from .distributed_tie_euler import DistributedTieParameters
from .dynamic_header_tie import (
    DynamicHeaderParameters,DynamicHeaderState,DynamicHeaderTieState,
    header_properties,
)
from .dynamic_parallel_header import (
    DynamicParallelTieHeaderStep,advance_dynamic_parallel_tie_header,
)
from .distributed_tie_engine_boundary import engine_tie_reservoirs
from .lh2_feed_cycle import HydrogenEOS
from .regenerative_cooling import CoolingNetwork
from .resolved_transient import ResolvedHeating
from .turbine_bleed_cycle import ShaftClosedPoint


@dataclass(frozen=True)
class DualHeaderTieState:
    pump: DynamicHeaderState
    downstream: DynamicHeaderTieState


@dataclass(frozen=True)
class DualHeaderTieStep:
    state: DualHeaderTieState
    parallel: DynamicParallelTieHeaderStep
    pump_old_pressure_pa: float
    pump_new_pressure_pa: float
    pump_old_enthalpy_j_kg: float
    pump_new_enthalpy_j_kg: float
    pump_new_temperature_k: float
    pump_delivered_reference_flow_kg_s: float
    pump_delivered_reference_enthalpy_j_kg: float
    tie_supply_interface_mass_flow_kg_s: float
    tie_supply_interface_total_energy_power_w: float
    pump_manifold_inventory_rate_kg_s: float
    pump_manifold_internal_energy_rate_w: float
    pump_manifold_mass_residual_kg_s: float
    pump_manifold_energy_residual_j: float
    combined_system_mass_residual_kg_s: float
    combined_system_total_energy_residual_j: float
    combined_net_mass_inventory_rate_kg_s: float
    conserved_tie_supply_face: bool = True
    dynamic_pump_manifold_active: bool = True
    pump_flow_and_head_are_reference_values: bool = True
    shaft_turbine_core_nozzle_feedback_converged: bool = False
    actual_openmc_transport_executed: bool = False
    pump_flow_and_head_dynamic: bool = False


def initial_pump_manifold(
    shaft: ShaftClosedPoint,
    parameters: DynamicHeaderParameters,
    *, eos: HydrogenEOS | None = None,
) -> DynamicHeaderState:
    """Fill a rigid upstream manifold with shaft-reference pump H2."""
    eos=eos or HydrogenEOS()
    reservoirs=engine_tie_reservoirs(shaft)
    p=reservoirs.upstream_pressure_pa
    h=reservoirs.upstream_stagnation_enthalpy_j_kg
    rho=eos.hp("Dmass",h,p)
    u=eos.hp("Umass",h,p)
    state=DynamicHeaderState(
        mass_kg=rho*parameters.volume_m3,
        internal_energy_j=rho*parameters.volume_m3*u,
    )
    header_properties(state,parameters,eos=eos)
    return state


def advance_dual_header_tie(
    heating: ResolvedHeating,
    old: DualHeaderTieState,
    distributed_params: DistributedTieParameters,
    upstream_params: DynamicHeaderParameters,
    downstream_params: DynamicHeaderParameters,
    network: CoolingNetwork,
    shaft: ShaftClosedPoint,
    dt_s: float,
    *, eos: HydrogenEOS | None = None,
    pump_delivery_override: tuple[float,float] | None = None,
) -> DualHeaderTieStep:
    """One explicit two-manifold/two-branch conservative timestep."""
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("dual header time step must be positive finite")
    eos=eos or HydrogenEOS()
    pup,hup,_,_=header_properties(old.pump,upstream_params,eos=eos)
    bc=engine_tie_reservoirs(shaft)
    # The reference pump *source* is not the instantaneous sum of the
    # dynamically varying nozzle branch and tie supply Euler flux.
    if pump_delivery_override is None:
        m_pump=shaft.tank_mass_flow_kg_s
        h_pump=bc.upstream_stagnation_enthalpy_j_kg
    else:
        if (not isinstance(pump_delivery_override,tuple)
                or len(pump_delivery_override)!=2):
            raise ValueError("pump delivery must be (mass flow, enthalpy)")
        m_pump,h_pump=pump_delivery_override
    if not math.isfinite(m_pump) or m_pump<=0 or not math.isfinite(h_pump):
        raise ValueError("invalid pump delivery source")

    parallel=advance_dynamic_parallel_tie_header(
        heating,old.downstream,distributed_params,downstream_params,
        network,shaft,dt_s,eos=eos,
        pump_reservoir_override=(pup,hup),
    )
    combined=parallel.combined
    tie=combined.tie_step
    fm=float(np.sum(tie.interface_mass_flux_kg_s[:,0]))
    fe=float(np.sum(tie.interface_total_energy_flux_w[:,0]))
    m_noz=parallel.nozzle_branch.mass_flow_kg_s
    h_noz_out=parallel.nozzle_branch.outlet_stagnation_enthalpy_j_kg
    heat_branch=network.nozzle_jacket.heat_power_w+network.reflector.heat_power_w
    if not math.isclose(
        m_noz*(h_noz_out-hup),heat_branch,
        rel_tol=1e-9,abs_tol=1e-4,
    ):
        raise RuntimeError("nozzle branch stagnation enthalpy first law failed")

    change_mass=dt_s*(m_pump-m_noz-fm)
    change_energy=dt_s*(m_pump*h_pump-m_noz*hup-fe)
    updated=DynamicHeaderState(
        mass_kg=old.pump.mass_kg+change_mass,
        internal_energy_j=old.pump.internal_energy_j+change_energy,
    )
    pnew,hnew,tnew,_=header_properties(
        updated,upstream_params,eos=eos,
    )
    dm=(updated.mass_kg-old.pump.mass_kg)/dt_s
    de=(updated.internal_energy_j-old.pump.internal_energy_j)/dt_s
    mass_error=dm-(m_pump-m_noz-fm)
    energy_error=updated.internal_energy_j-old.pump.internal_energy_j-change_energy

    # Combined audit: the numerical upstream and downstream tie
    # energy/mass fluxes cancel EXACTLY. The nozzle path is no fluid
    # inventory, and its known heat source must appear once.
    downstream_new=combined.state
    old_tie=old.downstream.tie
    new_tie=downstream_new.tie
    old_header=old.downstream.header
    new_header=downstream_new.header
    before_mass=(old.pump.mass_kg+old_header.mass_kg+
                 float(np.sum(old_tie.mass_kg)))
    after_mass=(updated.mass_kg+new_header.mass_kg+
                float(np.sum(new_tie.mass_kg)))
    engine_out=combined.fixed_engine_discharge_flow_kg_s
    total_mass_rate=(after_mass-before_mass)/dt_s
    total_mass_residual=total_mass_rate-(m_pump-engine_out)
    Cs=distributed_params.thermal.solid_capacity_j_k
    solid_delta=Cs*float(np.sum(
        new_tie.solid_temperature_k-old_tie.solid_temperature_k
    ))
    delta_energy=(
        updated.internal_energy_j-old.pump.internal_energy_j+
        new_header.internal_energy_j-old_header.internal_energy_j+
        float(np.sum(new_tie.total_energy_j-old_tie.total_energy_j))+
        solid_delta
    )
    total_heat=heat_branch+tie.nuclear_deposition_w
    expected_energy=dt_s*(
        m_pump*h_pump-engine_out*combined.header_old_enthalpy_j_kg+
        total_heat
    )
    energy_residual=delta_energy-expected_energy
    # Independent inventory-subtraction roundoff cap. No invented
    # physical source is used to hide a non-conservative numerical flux.
    eps=np.finfo(float).eps
    storage_m=(
        old.pump.mass_kg+updated.mass_kg+
        old_header.mass_kg+new_header.mass_kg+
        float(np.sum(abs(old_tie.mass_kg)+abs(new_tie.mass_kg)))
    )
    storage_e=(
        abs(updated.internal_energy_j)+abs(old.pump.internal_energy_j)+
        abs(old_header.internal_energy_j)+abs(new_header.internal_energy_j)+
        float(np.sum(abs(old_tie.total_energy_j)+
                     abs(new_tie.total_energy_j)))+
        Cs*float(np.sum(abs(old_tie.solid_temperature_k)+
                        abs(new_tie.solid_temperature_k)))
    )
    mass_tol=1e-8+16.*eps*storage_m/dt_s
    energy_tol=1e-6+16.*eps*storage_e
    if (abs(mass_error)>mass_tol
            or abs(total_mass_residual)>mass_tol
            or abs(energy_error)>energy_tol
            or abs(energy_residual)>energy_tol
            or abs(combined.combined_tie_solid_header_energy_residual_j)>energy_tol):
        raise RuntimeError(
            "dual-header mass/total-energy first law failed: "
            f"pump mass={mass_error:.9g} kg/s; "
            f"system mass={total_mass_residual:.9g} kg/s; "
            f"pump energy={energy_error:.9g} J; "
            f"system energy={energy_residual:.9g} J"
        )
    return DualHeaderTieStep(
        state=DualHeaderTieState(
            pump=updated,downstream=downstream_new,
        ),
        parallel=parallel,
        pump_old_pressure_pa=pup,pump_new_pressure_pa=pnew,
        pump_old_enthalpy_j_kg=hup,pump_new_enthalpy_j_kg=hnew,
        pump_new_temperature_k=tnew,
        pump_delivered_reference_flow_kg_s=m_pump,
        pump_delivered_reference_enthalpy_j_kg=h_pump,
        tie_supply_interface_mass_flow_kg_s=fm,
        tie_supply_interface_total_energy_power_w=fe,
        pump_manifold_inventory_rate_kg_s=dm,
        pump_manifold_internal_energy_rate_w=de,
        pump_manifold_mass_residual_kg_s=mass_error,
        pump_manifold_energy_residual_j=energy_error,
        combined_system_mass_residual_kg_s=total_mass_residual,
        combined_system_total_energy_residual_j=energy_residual,
        combined_net_mass_inventory_rate_kg_s=total_mass_rate,
        pump_flow_and_head_are_reference_values=(
            pump_delivery_override is None
        ),
        pump_flow_and_head_dynamic=(pump_delivery_override is not None),
    )
