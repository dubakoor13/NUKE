"""Pressure-responsive nozzle/reflector regenerative leg -> dynamic tie/header.

A pressure-conditioned SHARED pump-outlet to common-header branch is
solved using the existing validated Cryogenic CoolProp _leg solver
(nozzle jacket + reflector in series). The branch flow is NOT
held equal to the previously solved steady split: it is the unique
admissible positive flow whose reflector outlet matches the current
evolving header pressure. This updates mass and enthalpy entering the
dynamic finite-volume common header, which already exchanges exact
conservative Rusanov face mass/total-energy flux with the tie Euler.

The pump header p,h and total engine discharge remain prescribed by
an independently shaft-closed reference point, NOT solved in a new
pump/turbine/nozzle transient loop. The inlet mass mismatch is openly
reported. No friction law beyond the provided leg models, unsteady
branch inertia or reactor benchmark is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .dynamic_header_tie import (
    DynamicHeaderParameters,DynamicHeaderTieState,DynamicHeaderTieStep,
    advance_dynamic_header_tie,header_properties,
)
from .distributed_tie_euler import DistributedTieParameters
from .lh2_feed_cycle import HydrogenEOS, CryogenicCycleError
from .regenerative_cooling import CoolingNetwork,CoolingLegState,_leg
from .resolved_transient import ResolvedHeating
from .turbine_bleed_cycle import ShaftClosedPoint
from .distributed_tie_engine_boundary import engine_tie_reservoirs


@dataclass(frozen=True)
class DynamicNozzleBranch:
    mass_flow_kg_s: float
    outlet_stagnation_enthalpy_j_kg: float
    desired_header_pressure_pa: float
    outlet_pressure_pa: float
    outlet_pressure_residual_pa: float
    jacket: CoolingLegState
    reflector: CoolingLegState
    evaluations: int


@dataclass(frozen=True)
class DynamicParallelTieHeaderStep:
    combined: DynamicHeaderTieStep
    nozzle_branch: DynamicNozzleBranch
    pump_inflow_mismatch_kg_s: float
    steady_shaft_reference_mass_flow_kg_s: float
    nozzle_branch_pressure_feedback_active: bool = True
    upstream_pump_pressure_held_fixed: bool = True
    downstream_engine_mass_demand_held_fixed: bool = True
    full_shaft_flow_and_pressure_coupling_converged: bool = False


def solve_nozzle_reflector_to_header(
    network: CoolingNetwork,
    shaft: ShaftClosedPoint,
    header_pressure_pa: float,
    *,
    eos: HydrogenEOS | None = None,
    minimum_reference_fraction: float = 0.25,
    maximum_reference_fraction: float = 1.75,
) -> DynamicNozzleBranch:
    """Find series jacket/reflector flow matching the live header p.

    The actual compressible cryogenic momentum/energy leg solvers are
    called, rather than a linearized d(mdot)/dP surrogate. Search is
    constrained to a declared neighborhood of the reference flow:
    outside this hydraulic regime fail explicitly, not extrapolate.
    """
    network.validate()
    eos=eos or HydrogenEOS()
    bc=engine_tie_reservoirs(shaft)
    cooling=shaft.regenerative_cooling
    if cooling is None:
        raise ValueError("regenerative network absent")
    p_in=bc.upstream_pressure_pa
    h_in=bc.upstream_stagnation_enthalpy_j_kg
    m_ref=cooling.nozzle_reflector_mass_flow_kg_s
    if (not math.isfinite(header_pressure_pa)
            or header_pressure_pa<=0 or header_pressure_pa>=p_in):
        raise ValueError("dynamic header must be positive and below pump discharge")
    if (not math.isfinite(minimum_reference_fraction)
            or not math.isfinite(maximum_reference_fraction)
            or not 0<minimum_reference_fraction<1<maximum_reference_fraction<=5):
        raise ValueError("invalid hydraulic root flow bounds")
    cache={}
    def evaluate(m:float):
        if m in cache:
            return cache[m]
        if m<=0 or not math.isfinite(m):
            raise ValueError("invalid nozzle coolant flow")
        jacket=_leg(eos,"nozzle_jacket",network.nozzle_jacket,m,p_in,h_in)
        reflector=_leg(
            eos,"reflector",network.reflector,m,
            jacket.outlet_pressure_pa,jacket.outlet_stagnation_enthalpy_j_kg
        )
        val=reflector.outlet_pressure_pa-header_pressure_pa
        if not math.isfinite(val):
            raise ValueError("nozzle/reflector pressure residual nonfinite")
        cache[m]=(val,jacket,reflector)
        return cache[m]

    # The shaft reference solution is the most robust starting point.
    p_tol=max(0.10,2e-8*p_in)
    near=evaluate(m_ref)
    if abs(near[0])<=p_tol:
        mdot=m_ref;solution=near
    else:
        bracket=None
        samples=(1.,0.98,1.02,0.95,1.05,0.90,1.10,
                 0.80,1.20,0.65,1.35,0.50,1.50,0.35,1.65,0.25,1.75)
        valid=[]
        for x in samples:
            if not minimum_reference_fraction<=x<=maximum_reference_fraction:
                continue
            m=m_ref*x
            try:
                entry=evaluate(m)
            except (ValueError,RuntimeError,CryogenicCycleError):
                continue
            valid.append((m,entry[0]))
            ordered=sorted(valid,key=lambda z:z[0])
            for left,right in zip(ordered,ordered[1:]):
                if left[1]*right[1]<=0:
                    bracket=(left[0],right[0])
                    break
            if bracket is not None:
                break
        if bracket is None:
            raise ValueError(
                "dynamic header pressure outside validated nozzle/reflector "
                "branch flow range or no admissible subsonic hydraulic root"
            )
        lo,hi=bracket
        flo=evaluate(lo)[0]
        # For a dissipative branch p_out decreases as mass flow rises.
        if flo<0:
            raise RuntimeError("nonmonotone nozzle/reflector flow-pressure branch")
        mdot=None
        for _ in range(54):
            mid=0.5*(lo+hi)
            fm=evaluate(mid)[0]
            if abs(fm)<=p_tol or (hi-lo)<=1e-11*max(1.,m_ref):
                mdot=mid;break
            if fm>0:
                lo=mid
            else:
                hi=mid
        if mdot is None:
            mdot=0.5*(lo+hi)
        solution=evaluate(mdot)
    residual,jacket,reflector=solution
    if (abs(residual)>p_tol+1e-6
            or not math.isclose(
                jacket.outlet_stagnation_enthalpy_j_kg,
                reflector.inlet_stagnation_enthalpy_j_kg,
                rel_tol=1e-12
            )
            or not math.isclose(
                reflector.outlet_stagnation_enthalpy_j_kg,
                h_in+(network.nozzle_jacket.heat_power_w+
                      network.reflector.heat_power_w)/mdot,
                rel_tol=1e-10,abs_tol=1e-6
            )):
        raise RuntimeError("nozzle/reflector header root or enthalpy closure failed")
    return DynamicNozzleBranch(
        mass_flow_kg_s=mdot,
        outlet_stagnation_enthalpy_j_kg=
            reflector.outlet_stagnation_enthalpy_j_kg,
        desired_header_pressure_pa=header_pressure_pa,
        outlet_pressure_pa=reflector.outlet_pressure_pa,
        outlet_pressure_residual_pa=residual,
        jacket=jacket,reflector=reflector,evaluations=len(cache),
    )


def advance_dynamic_parallel_tie_header(
    heating: ResolvedHeating,
    old: DynamicHeaderTieState,
    distributed_params: DistributedTieParameters,
    header_params: DynamicHeaderParameters,
    network: CoolingNetwork,
    shaft: ShaftClosedPoint,
    dt_s:float,
    *,
    eos: HydrogenEOS | None = None,
) -> DynamicParallelTieHeaderStep:
    """Live nozzle mass/enthalpy + live Euler tie outlet -> same header."""
    eos=eos or HydrogenEOS()
    p_old,_,_,_=header_properties(old.header,header_params,eos=eos)
    solution=solve_nozzle_reflector_to_header(
        network,shaft,p_old,eos=eos
    )
    combined=advance_dynamic_header_tie(
        heating,old,distributed_params,header_params,shaft,dt_s,
        eos=eos,nozzle_branch_override=(
            solution.mass_flow_kg_s,
            solution.outlet_stagnation_enthalpy_j_kg,
        )
    )
    if not combined.dynamic_nozzle_branch_pressure_response_active:
        raise RuntimeError("dynamic pressure-conditioned branch not applied")
    mismatch=combined.upstream_pump_header_mass_flow_mismatch_kg_s
    # This records the unresolved *upstream* mass-inventory source:
    # the prescribed fixed pump header cannot supply a different
    # sum of transient branch flows without evolving its own mass.
    if not math.isfinite(mismatch):
        raise RuntimeError("nonfinite upstream pump header mass mismatch")
    return DynamicParallelTieHeaderStep(
        combined=combined,nozzle_branch=solution,
        pump_inflow_mismatch_kg_s=mismatch,
        steady_shaft_reference_mass_flow_kg_s=shaft.tank_mass_flow_kg_s,
    )
