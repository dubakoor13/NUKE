"""Warm-H2 turbine admission and rotor coupling, with real-fluid EOS.

The finite-volume common header provides REAL pressure and specific
enthalpy at every timestep. A configurable effective turbine admission
area determines mass flow using a quasi-steady *isentropic CoolProp*
real-H2 nozzle. At low back pressure, flux is limited by a computed
critical throat state, not multiplied linearly by pressure. This is
a prescribed effective flow area, NOT an inferred authentic NERVA
turbine nozzle design. Reference-area calibration is optional and
must be labeled synthetic unless backed by NASA hardware evidence.

Each timestep:
  mdot_bleed = A_eff * G_isen(p_header,h_header,p_exhaust)
  w_isentropic = h_header - h(s_header,p_exhaust)
  w_gas = eta_turb * w_isentropic
  P_turbine_shaft = mdot_bleed * w_gas * eta_mechanical
  m_core = m_engine_reference - mdot_bleed
  E_rot,new-E_rot,old = dt*(P_turbine_shaft - P_pump)

The H2 header loses *total* m_engine_reference*h_header
(core + turbine intake). The turbine exhaust and mechanical losses
are accounted for independently in a combined fluid+rotor energy
audit. Thus turbine work transferred to rotor is not an additional
unjustified external power source.

LIMITS: total core+bleed demand remains prescribed. This is NOT a
closed core/nozzle throat flow solver, a dynamic turbine inertia/
efficiency map, verified historical pump/bleed characteristics,
or an actual reactor neutron-transport solution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .dynamic_pump_shaft import (
    DynamicPumpParameters, DynamicPumpState, DynamicPumpStep,
    advance_dynamic_pump_tie,
)
from .dynamic_header_tie import DynamicHeaderParameters, header_properties
from .distributed_tie_euler import DistributedTieParameters
from .lh2_feed_cycle import HydrogenEOS, LH2Cycle
from .regenerative_cooling import CoolingNetwork
from .resolved_transient import ResolvedHeating
from .turbine_bleed_cycle import ShaftClosedPoint, TurbineBleed


@dataclass(frozen=True)
class TurbineAdmission:
    effective_admission_area_m2: float
    exhaust_pressure_pa: float
    isentropic_efficiency: float
    shaft_mechanical_efficiency: float
    maximum_bleed_fraction: float

    def validate(self):
        for name in ("effective_admission_area_m2","exhaust_pressure_pa",
                     "isentropic_efficiency","shaft_mechanical_efficiency",
                     "maximum_bleed_fraction"):
            x=getattr(self,name)
            if not math.isfinite(x) or x<=0:
                raise ValueError(f"turbine admission {name} must be positive")
        if (not 0<self.isentropic_efficiency<=1
                or not 0<self.shaft_mechanical_efficiency<=1
                or not 0<self.maximum_bleed_fraction<1):
            raise ValueError("turbine bleed efficiencies/fraction outside domain")


@dataclass(frozen=True)
class TurbineFlowPoint:
    inlet_pressure_pa: float
    inlet_enthalpy_j_kg: float
    inlet_temperature_k: float
    exhaust_pressure_pa: float
    admission_throat_pressure_pa: float
    admission_mass_flux_kg_m2_s: float
    is_admission_choked: bool
    bleed_mass_flow_kg_s: float
    core_reference_remainder_mass_flow_kg_s: float
    specific_isentropic_work_j_kg: float
    specific_actual_work_j_kg: float
    exhaust_enthalpy_j_kg: float
    exhaust_temperature_k: float
    turbine_gas_power_w: float
    delivered_rotor_power_w: float
    mechanical_loss_w: float
    turbine_energy_residual_w: float


@dataclass(frozen=True)
class DynamicTurbinePumpStep:
    state: DynamicPumpState
    pump_and_fluid: DynamicPumpStep
    turbine: TurbineFlowPoint
    engine_reference_total_discharge_flow_kg_s: float
    core_reference_nozzle_flow_mismatch_kg_s: float
    combined_open_system_energy_residual_j: float
    moving_turbine_bleed_and_shaft_power: bool = True
    pump_and_header_flow_pressure_feedback_active: bool = True
    core_nozzle_throat_mass_demand_dynamic: bool = False
    nasa_turbine_admission_area_calibrated: bool = False
    real_openmc_transport_executed: bool = False


def isentropic_hydrogen_admission_mass_flux(
    upstream_pressure_pa: float,
    upstream_enthalpy_j_kg: float,
    exhaust_pressure_pa: float,
    *,
    eos: HydrogenEOS | None = None,
) -> tuple[float,float,bool]:
    """Real-EOS isentropic nozzle mass flux and effective throat pressure.

    Uses stagnation inlet h,s; static outlet rho(p,s) and
    sqrt(2*(h_stag-h_static)), seeking the maximum mass flux for
    p in [p_exhaust,p_in]. The maximum represents the choked sonic
    mass flux if its pressure exceeds the imposed exhaust pressure;
    otherwise expansion reaches exhaust unchoked. Homogeneous
    equilibrium EOS; turbine blade passage losses omitted.
    """
    if (not math.isfinite(upstream_pressure_pa) or
            not math.isfinite(upstream_enthalpy_j_kg) or
            not math.isfinite(exhaust_pressure_pa) or
            not 0<exhaust_pressure_pa<upstream_pressure_pa):
        raise ValueError("invalid H2 turbine nozzle inlet/exhaust")
    eos=eos or HydrogenEOS()
    t0=eos.hp("T",upstream_enthalpy_j_kg,upstream_pressure_pa)
    if not 200.<t0<eos.maximum_temperature_k:
        raise ValueError("turbine admission must use warm single-phase H2")
    s0=eos.hp("Smass",upstream_enthalpy_j_kg,upstream_pressure_pa)

    def mass_flux(p:float)->float:
        h=eos.sp("Hmass",s0,p)
        delta=upstream_enthalpy_j_kg-h
        if delta<=0:
            if abs(delta)<1e-8*max(1.,abs(upstream_enthalpy_j_kg)):
                return 0.
            raise ValueError("inadmissible reverse turbine isentropic work")
        rho=eos.sp("Dmass",s0,p)
        if not math.isfinite(rho) or rho<=0:
            raise ValueError("nonpositive turbine isentropic density")
        return rho*math.sqrt(2.*delta)

    left,right=exhaust_pressure_pa,upstream_pressure_pa
    # Golden-section search of a single peak for a warm gas.
    golden=(math.sqrt(5.)-1.)/2.
    x1=right-golden*(right-left)
    x2=left+golden*(right-left)
    f1,f2=mass_flux(x1),mass_flux(x2)
    for _ in range(43):
        if f1 < f2:
            left=x1
            x1,f1=x2,f2
            x2=left+golden*(right-left)
            f2=mass_flux(x2)
        else:
            right=x2
            x2,f2=x1,f1
            x1=right-golden*(right-left)
            f1=mass_flux(x1)
    pstar=0.5*(left+right)
    gstar=mass_flux(pstar)
    gback=mass_flux(exhaust_pressure_pa)
    if gback>=gstar*(1.-1e-10):
        return gback,exhaust_pressure_pa,False
    return gstar,pstar,True


def calibrate_effective_admission(
    reference: ShaftClosedPoint,
    turbine: TurbineBleed,
    *,
    eos: HydrogenEOS | None = None,
) -> TurbineAdmission:
    """Fit equivalent admission area to an EXISTING solved reference.

    This is a *synthetic effective* area; it is NOT a NASA hardware
    dimension or validated turbine nozzle passage area.
    """
    turbine.validate()
    hydraulic=reference.regenerative_cooling
    if hydraulic is None:
        raise ValueError("cannot calibrate admission without cooling shaft reference")
    g,_,_=isentropic_hydrogen_admission_mass_flux(
        hydraulic.common_header_pressure_pa,
        hydraulic.common_header_stagnation_enthalpy_j_kg,
        turbine.exhaust_pressure_pa,eos=eos,
    )
    if g<=0 or not math.isfinite(g):
        raise ValueError("reference turbine cannot sustain admission flow")
    p=TurbineAdmission(
        effective_admission_area_m2=
            reference.turbine_bleed_mass_flow_kg_s/g,
        exhaust_pressure_pa=turbine.exhaust_pressure_pa,
        isentropic_efficiency=turbine.isentropic_efficiency,
        shaft_mechanical_efficiency=turbine.shaft_mechanical_efficiency,
        maximum_bleed_fraction=turbine.maximum_bleed_fraction,
    )
    p.validate()
    return p


def evaluate_dynamic_turbine(
    parameters: TurbineAdmission,
    upstream_pressure_pa:float,
    upstream_enthalpy_j_kg:float,
    total_engine_discharge_mass_flow_kg_s:float,
    *,
    eos: HydrogenEOS | None = None,
) -> TurbineFlowPoint:
    parameters.validate()
    if (not math.isfinite(total_engine_discharge_mass_flow_kg_s)
            or total_engine_discharge_mass_flow_kg_s<=0):
        raise ValueError("invalid total engine discharge reference")
    eos=eos or HydrogenEOS()
    g,pcrit,choked=isentropic_hydrogen_admission_mass_flux(
        upstream_pressure_pa,upstream_enthalpy_j_kg,
        parameters.exhaust_pressure_pa,eos=eos,
    )
    bleed=g*parameters.effective_admission_area_m2
    core=total_engine_discharge_mass_flow_kg_s-bleed
    if (not math.isfinite(bleed) or bleed<=0 or core<=0
            or bleed>=parameters.maximum_bleed_fraction*
            total_engine_discharge_mass_flow_kg_s):
        raise ValueError("turbine bleed exceeds physical core-flow constraints")
    entropy=eos.hp("Smass",upstream_enthalpy_j_kg,upstream_pressure_pa)
    h_isen=eos.sp("Hmass",entropy,parameters.exhaust_pressure_pa)
    work_isen=upstream_enthalpy_j_kg-h_isen
    work=parameters.isentropic_efficiency*work_isen
    if work<=0 or not math.isfinite(work):
        raise ValueError("turbine work is not physically positive")
    h_exit=upstream_enthalpy_j_kg-work
    temp_exit=eos.hp("T",h_exit,parameters.exhaust_pressure_pa)
    if not 120.<temp_exit<eos.maximum_temperature_k:
        raise ValueError("turbine exhaust outside warm-H2 domain")
    gas_power=bleed*work
    delivered=gas_power*parameters.shaft_mechanical_efficiency
    mechanical_loss=gas_power-delivered
    energy_residual=(
        bleed*(upstream_enthalpy_j_kg-h_exit)
        -delivered-mechanical_loss
    )
    if abs(energy_residual)>1e-7*max(1.,gas_power):
        raise RuntimeError("turbine gas->rotor/exhaust first law failed")
    return TurbineFlowPoint(
        inlet_pressure_pa=upstream_pressure_pa,
        inlet_enthalpy_j_kg=upstream_enthalpy_j_kg,
        inlet_temperature_k=eos.hp(
            "T",upstream_enthalpy_j_kg,upstream_pressure_pa
        ),
        exhaust_pressure_pa=parameters.exhaust_pressure_pa,
        admission_throat_pressure_pa=pcrit,
        admission_mass_flux_kg_m2_s=g,
        is_admission_choked=choked,
        bleed_mass_flow_kg_s=bleed,
        core_reference_remainder_mass_flow_kg_s=core,
        specific_isentropic_work_j_kg=work_isen,
        specific_actual_work_j_kg=work,
        exhaust_enthalpy_j_kg=h_exit,
        exhaust_temperature_k=temp_exit,
        turbine_gas_power_w=gas_power,
        delivered_rotor_power_w=delivered,
        mechanical_loss_w=mechanical_loss,
        turbine_energy_residual_w=energy_residual,
    )


def advance_dynamic_turbine_pump(
    heating: ResolvedHeating,
    old: DynamicPumpState,
    core: LH2Cycle,
    pump_parameters: DynamicPumpParameters,
    admission: TurbineAdmission,
    distributed_parameters: DistributedTieParameters,
    upstream_parameters: DynamicHeaderParameters,
    downstream_parameters: DynamicHeaderParameters,
    network: CoolingNetwork,
    reference_shaft: ShaftClosedPoint,
    dt_s:float,
    *,
    eos: HydrogenEOS | None = None,
) -> DynamicTurbinePumpStep:
    """One conservative dual-manifold + dynamic turbine/rotor timestep."""
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("dynamic turbine timestep must be positive finite")
    eos=eos or HydrogenEOS()
    p_header,h_header,_,_=header_properties(
        old.hydraulic.downstream.header,downstream_parameters,eos=eos,
    )
    turbine=evaluate_dynamic_turbine(
        admission,p_header,h_header,
        reference_shaft.tank_mass_flow_kg_s,eos=eos,
    )
    pump=advance_dynamic_pump_tie(
        heating,old,core,pump_parameters,distributed_parameters,
        upstream_parameters,downstream_parameters,
        network,reference_shaft,dt_s,eos=eos,
        turbine_shaft_power_override_w=turbine.delivered_rotor_power_w,
    )
    mengine=reference_shaft.tank_mass_flow_kg_s
    energy=pump.fluid
    next_hyd=pump.state.hydraulic
    old_hyd=old.hydraulic
    cs=distributed_parameters.thermal.solid_capacity_j_k
    delta_fluids=(
        next_hyd.pump.internal_energy_j-old_hyd.pump.internal_energy_j+
        next_hyd.downstream.header.internal_energy_j-
        old_hyd.downstream.header.internal_energy_j+
        float(np.sum(
            next_hyd.downstream.tie.total_energy_j-
            old_hyd.downstream.tie.total_energy_j
        ))+
        cs*float(np.sum(
            next_hyd.downstream.tie.solid_temperature_k-
            old_hyd.downstream.tie.solid_temperature_k
        ))
    )
    total_delta=delta_fluids+pump.rotor_final_energy_j-pump.rotor_initial_energy_j
    q_heat=(
        network.nozzle_jacket.heat_power_w+
        network.reflector.heat_power_w+
        energy.parallel.combined.tie_step.nuclear_deposition_w
    )
    expected=dt_s*(
        pump.pump.mass_flow_kg_s*pump.pump.tank_enthalpy_j_kg+
        q_heat-
        turbine.core_reference_remainder_mass_flow_kg_s*h_header-
        turbine.bleed_mass_flow_kg_s*turbine.exhaust_enthalpy_j_kg-
        turbine.mechanical_loss_w
    )
    residual=total_delta-expected
    eps=np.finfo(float).eps
    roundoff=1e-6+16.*eps*(
        abs(old_hyd.pump.internal_energy_j)+
        abs(next_hyd.pump.internal_energy_j)+
        abs(old_hyd.downstream.header.internal_energy_j)+
        abs(next_hyd.downstream.header.internal_energy_j)+
        float(np.sum(abs(old_hyd.downstream.tie.total_energy_j)+
                     abs(next_hyd.downstream.tie.total_energy_j)))+
        cs*float(np.sum(
            abs(old_hyd.downstream.tie.solid_temperature_k)+
            abs(next_hyd.downstream.tie.solid_temperature_k)
        ))+
        pump.rotor_final_energy_j+pump.rotor_initial_energy_j
    )
    if abs(residual)>roundoff:
        raise RuntimeError(
            f"dynamic turbine/pump total boundary energy mismatch: {residual:.8g} J"
        )
    return DynamicTurbinePumpStep(
        state=pump.state,
        pump_and_fluid=pump,
        turbine=turbine,
        engine_reference_total_discharge_flow_kg_s=mengine,
        core_reference_nozzle_flow_mismatch_kg_s=(
            turbine.core_reference_remainder_mass_flow_kg_s-
            reference_shaft.core_mass_flow_kg_s
        ),
        combined_open_system_energy_residual_j=residual,
    )
