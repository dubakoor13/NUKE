"""Pressure-driven compressible H2 tie-cell mass and energy transport.

Distinct from the isochoric uniform-flow solver, this screening model
calculates a DIFFERENT mass flux at every supply / turnaround / return
interface using a declared quasisteady orifice momentum LOSS:

    mdot = sign(p_left-p_right)*Cd*A*sqrt(2*rho_upwind*abs(dp)).

It conserves mass in every rigid cell:
    (m_new-m_old)/dt = mdot_in - mdot_out.
It conserves INTERNAL fluid energy, advecting ENTHALPY:
    ((m*u)_new-(m*u)_old)/dt =
      mdot_in*h_upwind,in - mdot_out*h_upwind,out + Q_wall + Q_nuclear,H2.
Cell pressures, h and T are evaluated with CoolProp from rho=m/V,u.

The one interface at the far-end tie turnaround is shared by supply
and return; it cannot create or destroy hydrogen. The two branches
share one solid cell at each physical axial station. Mass/energy are
updated with explicit first-order Euler from the SAME old states and
old-pressure hydraulic fluxes; a strict outflow/Courant gate prevents
silently stepping beyond explicit resolution.

LIMITS: this is a quasisteady pressure-loss relation, NOT a transient
momentum equation, choked nozzle or a calibrated two-phase flow law.
Homogeneous-equilibrium mixture density is used without phase slip,
gravity, wall flashing correlations, CHF, or hardware-specific losses.
Pump/outlet reservoir pressures and enthalpies are prescribed; neither
reactor power kinetics nor full pump/turbine/shaft flow is solved.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .isochoric_tie import IsochoricTieState, IsochoricTieParameters, _properties
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import ResolvedHeating


@dataclass(frozen=True)
class PressureTieFlowParameters:
    thermal: IsochoricTieParameters
    interface_area_m2: float
    discharge_coefficient: float = 0.70
    max_fractional_mass_exchange: float = 0.20
    max_fractional_pressure_drop: float = 0.45

    def validate(self) -> None:
        self.thermal.validate()
        if (not math.isfinite(self.interface_area_m2)
                or self.interface_area_m2<=0):
            raise ValueError("tie interface flow area must be positive")
        if (not math.isfinite(self.discharge_coefficient)
                or not 0<self.discharge_coefficient<=1):
            raise ValueError("tie discharge coefficient outside (0,1]")
        if (not math.isfinite(self.max_fractional_mass_exchange)
                or not 0<self.max_fractional_mass_exchange<=0.25):
            raise ValueError("invalid explicit mass exchange stability bound")
        if (not math.isfinite(self.max_fractional_pressure_drop)
                or not 0<self.max_fractional_pressure_drop<=0.5):
            raise ValueError("invalid subsonic orifice pressure-drop bound")


@dataclass(frozen=True)
class PressureTieFlowStep:
    state: IsochoricTieState
    supply_pressure_pa: np.ndarray
    return_pressure_pa: np.ndarray
    supply_temperature_k: np.ndarray
    return_temperature_k: np.ndarray
    supply_density_kg_m3: np.ndarray
    return_density_kg_m3: np.ndarray
    supply_enthalpy_j_kg: np.ndarray
    return_enthalpy_j_kg: np.ndarray
    supply_quality: np.ndarray
    return_quality: np.ndarray
    # shape [tie, 2*Nz+1], directed inlet -> supply -> return -> outlet
    interface_mass_flow_kg_s: np.ndarray
    supply_wall_power_w: np.ndarray
    return_wall_power_w: np.ndarray
    inlet_mass_flow_kg_s: float
    outlet_mass_flow_kg_s: float
    total_mass_storage_rate_kg_s: float
    solid_storage_rate_w: float
    hydrogen_internal_energy_storage_rate_w: float
    nuclear_deposition_w: float
    inlet_enthalpy_power_w: float
    outlet_enthalpy_power_w: float
    mass_balance_error_kg_s: float
    energy_balance_error_j: float
    max_local_mass_balance_error_kg_s: float
    max_local_energy_balance_error_w: float
    maximum_fractional_mass_exchange: float
    momentum_law_kind: str = "quasisteady homogeneous-mixture orifice, no inertia"


def advance_pressure_driven_tie(
    heating: ResolvedHeating,
    old: IsochoricTieState,
    params: PressureTieFlowParameters,
    inlet_pressure_pa: float,
    inlet_enthalpy_j_kg: float,
    outlet_pressure_pa: float,
    outlet_backflow_enthalpy_j_kg: float,
    dt_s: float,
    *,
    eos: HydrogenEOS | None = None,
) -> PressureTieFlowStep:
    """One conservative explicit step with computed interface fluxes.

    Each flow is calculated ONCE and reused with opposite sign for both
    neighboring cells, ensuring global mass/energy telescoping.
    """
    heating.validate()
    params.validate()
    old.validate(heating)
    for label,p in (("inlet",inlet_pressure_pa),("outlet",outlet_pressure_pa)):
        if not math.isfinite(p) or p<=0:
            raise ValueError(f"invalid {label} boundary pressure")
    for label,h in (("inlet",inlet_enthalpy_j_kg),
                    ("backflow",outlet_backflow_enthalpy_j_kg)):
        if not math.isfinite(h):
            raise ValueError(f"nonfinite {label} boundary enthalpy")
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("explicit tie timestep must be positive")
    eos=eos or HydrogenEOS()
    th=params.thermal
    shape=heating.tie_solid_w.shape
    nz=shape[1]

    # Orient the return duct in its real hydraulic flow direction
    # (z=L -> z=0), not in the OpenMC spatial array convention.
    mass=np.concatenate((old.supply_mass_kg,old.return_mass_kg[:,::-1]),axis=1)
    internal=np.concatenate((old.supply_internal_energy_j_kg,
                             old.return_internal_energy_j_kg[:,::-1]),axis=1)
    volume=np.array([th.supply_cell_volume_m3]*nz+
                    [th.return_cell_volume_m3]*nz)[None,:]
    rho=mass/volume
    p=np.empty_like(internal)
    temp=np.empty_like(internal)
    h=np.empty_like(internal)
    for j,i in np.ndindex(internal.shape):
        p[j,i],temp[j,i],h[j,i]=_properties(
            eos,float(rho[j,i]),float(internal[j,i])
        )
    if np.any((p<th.min_pressure_pa)|(p>th.max_pressure_pa)):
        raise ValueError("initial EOS state exceeds pressure limits")
    if np.any((temp<th.min_temperature_k)|(temp>th.max_temperature_k)):
        raise ValueError("initial EOS state exceeds temperature limits")

    inlet_rho=eos.hp("Dmass",inlet_enthalpy_j_kg,inlet_pressure_pa)
    outlet_rho=eos.hp("Dmass",outlet_backflow_enthalpy_j_kg,outlet_pressure_pa)
    if not (math.isfinite(inlet_rho) and inlet_rho>0 and
            math.isfinite(outlet_rho) and outlet_rho>0):
        raise ValueError("invalid reservoir H2 densities")
    edge=np.empty((shape[0],2*nz+1))
    edge_h=np.empty_like(edge)
    max_dp_ratio=0.
    for j in range(shape[0]):
        for k in range(2*nz+1):
            pleft=float(inlet_pressure_pa if k==0 else p[j,k-1])
            pright=float(outlet_pressure_pa if k==2*nz else p[j,k])
            dp=pleft-pright
            # CoolProp's P(rho,u) inverse is not bitwise identical to
            # the input P used for initialising rho/u. Without a small
            # relative equilibrium band a sqrt(dp) restriction creates
            # artificial flux under physically identical reservoirs.
            if abs(dp)<=1e-8*max(1.,pleft,pright):
                dp=0.
            frac=abs(dp)/max(pleft,pright)
            max_dp_ratio=max(max_dp_ratio,frac)
            if frac>params.max_fractional_pressure_drop:
                raise ValueError("tie orifice pressure ratio outside screening model")
            if dp>=0:
                density=float(inlet_rho if k==0 else rho[j,k-1])
                upstream_h=float(inlet_enthalpy_j_kg if k==0 else h[j,k-1])
            else:
                density=float(outlet_rho if k==2*nz else rho[j,k])
                upstream_h=float(outlet_backflow_enthalpy_j_kg
                                 if k==2*nz else h[j,k])
            edge[j,k]=(1. if dp>=0 else -1.)*(
                params.discharge_coefficient*params.interface_area_m2*
                math.sqrt(2.*density*abs(dp))
            )
            edge_h[j,k]=upstream_h

    # For each CV, use OLD-time temperature and power with its OWN
    # solid station: return cell i=Nz-1-k and supply cell i=k.
    q_s=th.supply_ua_w_k*(old.solid_temperature_k-temp[:,:nz])
    q_r=th.return_ua_w_k*(old.solid_temperature_k-
                           temp[:,nz:][:,::-1])
    wall=np.concatenate((q_s,q_r[:,::-1]),axis=1)
    deposition=np.concatenate((heating.tie_supply_h2_w,
                                 heating.tie_return_h2_w[:,::-1]),axis=1)
    rate_in=edge[:,:-1]
    rate_out=edge[:,1:]
    fractional=dt_s*(np.abs(rate_in)+np.abs(rate_out))/mass
    frac_max=float(np.max(fractional))
    if frac_max>params.max_fractional_mass_exchange:
        raise ValueError(
            f"explicit tie mass Courant bound exceeded: {frac_max:.6g}"
        )
    mnew=mass+dt_s*(rate_in-rate_out)
    if np.any(mnew<=0) or not np.all(np.isfinite(mnew)):
        raise ValueError("tie fluid depleted or mass update nonfinite")
    total_old=mass*internal
    e_new=total_old+dt_s*(
        rate_in*edge_h[:,:-1]-rate_out*edge_h[:,1:]+wall+deposition
    )
    unew=e_new/mnew
    if not np.all(np.isfinite(unew)):
        raise ValueError("tie fluid internal energy nonfinite")
    # Solve the shared solid explicit CV once per physical station.
    q_solid=heating.tie_solid_w-q_s-q_r
    solid_new=old.solid_temperature_k+dt_s*q_solid/th.solid_capacity_j_k
    if np.any(solid_new<=0):
        raise ValueError("nonphysical shared tie solid temperature")
    pnew=np.empty_like(unew)
    tnew=np.empty_like(unew)
    hnew=np.empty_like(unew)
    quality=np.empty_like(unew)
    for j,i in np.ndindex(unew.shape):
        density=float(mnew[j,i]/volume[0,i])
        pnew[j,i],tnew[j,i],hnew[j,i]=_properties(
            eos,density,float(unew[j,i])
        )
        if not (th.min_pressure_pa<=pnew[j,i]<=th.max_pressure_pa
                and th.min_temperature_k<=tnew[j,i]<=th.max_temperature_k):
            raise ValueError("updated EOS state outside tie screening range")
        quality[j,i]=eos.get("Q","Dmass",density,"Umass",float(unew[j,i]))

    local_mass=(mnew-mass)/dt_s-(rate_in-rate_out)
    local_energy=(e_new-total_old)/dt_s-(
        rate_in*edge_h[:,:-1]-rate_out*edge_h[:,1:]+wall+deposition
    )
    local_solid=(solid_new-old.solid_temperature_k)*(
        th.solid_capacity_j_k/dt_s
    )-q_solid
    mass_old=float(np.sum(mass))
    mass_new=float(np.sum(mnew))
    fin=float(np.sum(edge[:,0]))
    fout=float(np.sum(edge[:,-1]))
    mass_err=(mass_new-mass_old)/dt_s-(fin-fout)
    d_e_h=float(np.sum(e_new-total_old)/dt_s)
    d_e_s=float(np.sum(
        th.solid_capacity_j_k*(solid_new-old.solid_temperature_k)
    )/dt_s)
    nuclear=float(np.sum(heating.tie_solid_w)+
                  np.sum(heating.tie_supply_h2_w)+
                  np.sum(heating.tie_return_h2_w))
    enth_in=float(np.sum(edge[:,0]*edge_h[:,0]))
    enth_out=float(np.sum(edge[:,-1]*edge_h[:,-1]))
    e_err=dt_s*(d_e_h+d_e_s-nuclear-enth_in+enth_out)
    max_local_mass=float(np.max(np.abs(local_mass)))
    max_local_energy=max(float(np.max(np.abs(local_energy))),
                         float(np.max(np.abs(local_solid))))
    scale=max(1.,abs(d_e_h),abs(d_e_s),abs(nuclear),
              abs(enth_in),abs(enth_out))
    if (abs(e_err)>1e-8*scale*dt_s+1e-7
            or abs(mass_err)>1e-8*max(1.,abs(fin),abs(fout))+1e-9
            or max_local_mass>1e-8*max(1.,abs(fin),abs(fout))+1e-9
            or max_local_energy>1e-8*scale+1e-6):
        raise RuntimeError("pressure-driven tie mass/energy conservation failure")

    return PressureTieFlowStep(
        state=IsochoricTieState(
            solid_temperature_k=solid_new,
            supply_mass_kg=mnew[:,:nz].copy(),
            return_mass_kg=mnew[:,nz:][:,::-1].copy(),
            supply_internal_energy_j_kg=unew[:,:nz].copy(),
            return_internal_energy_j_kg=unew[:,nz:][:,::-1].copy(),
        ),
        supply_pressure_pa=pnew[:,:nz].copy(),
        return_pressure_pa=pnew[:,nz:][:,::-1].copy(),
        supply_temperature_k=tnew[:,:nz].copy(),
        return_temperature_k=tnew[:,nz:][:,::-1].copy(),
        supply_density_kg_m3=(mnew/volume)[:,:nz].copy(),
        return_density_kg_m3=(mnew/volume)[:,nz:][:,::-1].copy(),
        supply_enthalpy_j_kg=hnew[:,:nz].copy(),
        return_enthalpy_j_kg=hnew[:,nz:][:,::-1].copy(),
        supply_quality=quality[:,:nz].copy(),
        return_quality=quality[:,nz:][:,::-1].copy(),
        interface_mass_flow_kg_s=edge,
        supply_wall_power_w=q_s,return_wall_power_w=q_r,
        inlet_mass_flow_kg_s=fin,outlet_mass_flow_kg_s=fout,
        total_mass_storage_rate_kg_s=(mass_new-mass_old)/dt_s,
        solid_storage_rate_w=d_e_s,
        hydrogen_internal_energy_storage_rate_w=d_e_h,
        nuclear_deposition_w=nuclear,
        inlet_enthalpy_power_w=enth_in,
        outlet_enthalpy_power_w=enth_out,
        mass_balance_error_kg_s=mass_err,
        energy_balance_error_j=e_err,
        max_local_mass_balance_error_kg_s=max_local_mass,
        max_local_energy_balance_error_w=max_local_energy,
        maximum_fractional_mass_exchange=frac_max,
    )
