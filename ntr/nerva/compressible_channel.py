"""Quasi-steady, subsonic compressible H2 fuel-channel thermal hydraulics.

For each constant-area axial control volume, conserve mass, total
enthalpy and streamwise momentum with a Darcy wall-friction loss:

 G = mdot/A ; v = G/rho
 mdot * [(h_out + v_out²/2) - (h_in + v_in²/2)] = Q_wall + Q_direct
 p_in - p_out = G*(v_out-v_in) + f_D*(dz/D)*G*(v_in+v_out)/4

This is a discrete quasi-1D, constant-area, single-phase *ideal*
pure-H2 model using GRI-Mech NASA-7 frozen molecular H2 enthalpy.
Thermal transport coefficients and Dittus-Boelter Nu reuse the existing
simple screening correlations. No coolant inventory transient,
compressible gas surge, mixture dissociation, heat-transfer entrance
effect, curvature, wall blowing, variable area or loss coefficient.

The iterative pressure root is on the low-Mach branch and rejects high
Mach solutions. Passing a CoolProp property model is explicitly
unsupported: mixing ideal-gas energy with real-fluid density would
violate energy/momentum consistency.
"""
from __future__ import annotations

import math
import numpy as np

from .nasa_hydrogen import hydrogen_state, T_MIN, T_MAX, RU, M_H2
from .thermal import (
    ChannelSolution, HydrogenProperties, friction_factor, nusselt_number
)

R_H2 = RU/M_H2


class SubsonicFlowError(ValueError):
    """Supplied pressure/flow/heating cannot be resolved subsonically."""


def _state(t: float, p: float):
    return hydrogen_state(t, p, mode="frozen")


def _temperature_from_total_enthalpy(
    target_j_kg: float, pressure_pa: float, mass_flux: float,
    guess_k: float
) -> float:
    """Invert h(T)+[G*R*T/p]^2/2 with safeguarded Newton iteration."""
    if not math.isfinite(pressure_pa) or pressure_pa <= 0:
        raise SubsonicFlowError("nonpositive pressure during energy solve")
    def f(t: float):
        hs = _state(t,pressure_pa)
        speed=mass_flux/hs.density_kg_m3
        return hs.enthalpy_j_kg+0.5*speed**2-target_j_kg, hs, speed

    low=f(T_MIN)[0]
    high=f(T_MAX)[0]
    if low > 1e-7*max(abs(target_j_kg),1.) or high < -1e-7*max(abs(target_j_kg),1.):
        raise SubsonicFlowError(
            "gas enthalpy solution outside NASA-7 200-3500 K domain"
        )
    lower,upper=T_MIN,T_MAX
    t=max(lower,min(upper,guess_k))
    for _ in range(28):
        val,hs,speed=f(t)
        if abs(val)<=1e-10*max(1.,abs(target_j_kg),abs(hs.enthalpy_j_kg)):
            return t
        if val>0:
            upper=t
        else:
            lower=t
        derivative=hs.frozen_cp_j_kg_k+speed*speed/t
        trial=t-val/derivative
        if not math.isfinite(trial) or trial<=lower or trial>=upper:
            trial=0.5*(lower+upper)
        t=trial
    val,hs,_=f(t)
    if abs(val)>2e-8*max(1.,abs(target_j_kg),abs(hs.enthalpy_j_kg)):
        raise RuntimeError("hydrogen stagnation enthalpy inversion not converged")
    return t


def _mach(state, mass_flux: float) -> float:
    cp=state.frozen_cp_j_kg_k
    if cp<=R_H2:
        raise ValueError("invalid pure hydrogen heat-capacity ratio")
    gamma=cp/(cp-R_H2)
    a=math.sqrt(gamma*R_H2*state.temperature_k)
    return (mass_flux/state.density_kg_m3)/a


def solve_fuel_channel_compressible(
    axial_total_fuel_power_w: np.ndarray,
    z_edges_m: np.ndarray,
    fuel_channel_count: int,
    mass_flow_per_channel_kg_s: float,
    inlet_temperature_k: float,
    inlet_pressure_pa: float,
    channel_diameter_m: float,
    properties: HydrogenProperties | None = None,
    property_model=None,
    roughness_m: float = 1e-6,
    axial_direct_coolant_power_w: np.ndarray | None = None,
    max_mach: float = 0.85,
) -> ChannelSolution:
    """Quasi-steady mass/momentum/total-enthalpy for an H2 channel.

    Supports only NASA-7 molecular-hydrogen gas in 200-3500 K, and a
    fixed viscosity/conductivity surrogate for Reynolds/Nusselt/f_D.
    A conservative momentum and total-enthalpy audit is returned
    explicitly for each axial segment.
    """
    if property_model is not None:
        raise ValueError(
            "compressible NASA-7 solver does not support CoolProp "
            "without a consistent real-fluid stagnation-enthalpy backend"
        )
    props=properties or HydrogenProperties()
    power=np.asarray(axial_total_fuel_power_w,dtype=float)
    z=np.asarray(z_edges_m,dtype=float)
    direct=np.zeros_like(power) if axial_direct_coolant_power_w is None else np.asarray(
        axial_direct_coolant_power_w,dtype=float
    )
    if (power.ndim!=1 or power.size<1
            or direct.shape!=power.shape or z.shape!=(power.size+1,)
            or np.any(~np.isfinite(power)) or np.any(~np.isfinite(direct))
            or np.any(power<0) or np.any(direct<0)
            or np.any(~np.isfinite(z)) or np.any(np.diff(z)<=0)):
        raise ValueError("nonnegative 1D heating and increasing z edges required")
    if not isinstance(fuel_channel_count,int) or fuel_channel_count<1:
        raise ValueError("fuel_channel_count must be positive")
    for label,val in (
        ("flow",mass_flow_per_channel_kg_s),
        ("inlet pressure",inlet_pressure_pa),
        ("diameter",channel_diameter_m),
        ("viscosity",props.dynamic_viscosity_pa_s),
        ("conductivity",props.thermal_conductivity_w_m_k),
    ):
        if not math.isfinite(val) or val<=0:
            raise ValueError(f"{label} must be finite and positive")
    if (not math.isfinite(roughness_m) or roughness_m<0
            or not math.isfinite(max_mach) or not 0<max_mach<1):
        raise ValueError("invalid roughness or allowable subsonic Mach limit")
    if not math.isfinite(inlet_temperature_k) or not T_MIN<=inlet_temperature_k<=T_MAX:
        raise ValueError("pure H2 gas inlet outside NASA-7 200-3500 K")

    mdot=float(mass_flow_per_channel_kg_s)
    d=float(channel_diameter_m)
    area=math.pi*d*d/4
    G=mdot/area
    rel_rough=roughness_m/d
    reynolds=G*d/props.dynamic_viscosity_pa_s
    f=friction_factor(reynolds,rel_rough)
    t_in=float(inlet_temperature_k)
    p_in=float(inlet_pressure_pa)
    wall=power/fuel_channel_count
    gas=direct/fuel_channel_count
    n=power.size
    outlet_t=np.empty(n)
    outlet_p=np.empty(n)
    wall_t=np.empty(n)
    htc=np.empty(n)
    heat_flux=np.empty(n)
    mach=np.empty(n)
    dp_f=np.empty(n)
    dp_acc=np.empty(n)
    energy_residual=np.empty(n)
    momentum_residual=np.empty(n)
    re=np.full(n,reynolds)

    for i in range(n):
        q_wall=float(wall[i])
        q_direct=float(gas[i])
        q=q_wall+q_direct
        dz=float(z[i+1]-z[i])
        gas_in=_state(t_in,p_in)
        v_in=G/gas_in.density_kg_m3
        m_in=_mach(gas_in,G)
        if m_in>=max_mach:
            raise SubsonicFlowError(
                f"cell {i} inlet Mach {m_in:.3g} exceeds allowed "
                f"{max_mach:.3g}; choking/momentum model requires "
                "a compressible transonic solver"
            )
        target=gas_in.enthalpy_j_kg+0.5*v_in*v_in+q/mdot
        guess=min(T_MAX,max(T_MIN,t_in+q/(mdot*gas_in.frozen_cp_j_kg_k)))
        def evaluate(p: float):
            temperature=_temperature_from_total_enthalpy(target,p,G,guess)
            gas_out=_state(temperature,p)
            v_out=G/gas_out.density_kg_m3
            # Darcy friction using midpoint of endpoint specific volumes.
            friction_dp=f*(dz/d)*G*(v_in+v_out)/4
            acceleration_dp=G*(v_out-v_in)
            residual=p+friction_dp+acceleration_dp-p_in
            return residual,temperature,gas_out,v_out,friction_dp,acceleration_dp

        p=p_in
        for iteration in range(45):
            residual,t,gas_out,v_out,fric,acc=evaluate(p)
            m_out=_mach(gas_out,G)
            if m_out>=max_mach:
                raise SubsonicFlowError(
                    f"cell {i} predicted Mach {m_out:.3g} exceeds "
                    f"{max_mach:.3g}; no validated subsonic solution"
                )
            if abs(residual)<=1e-9*max(p_in,1):
                break
            eps=max(p*1e-6,1e-4)
            # A negative pressure perturbation keeps the evaluation
            # on the physical pressure-drop side of the inlet.
            p_pert=max(p-eps,p*0.9999)
            # Choose a step that is small compared with p, but nonzero.
            p_pert=p-eps
            if p_pert<=0:
                raise SubsonicFlowError("pressure would become nonpositive")
            f_pert=evaluate(p_pert)[0]
            derivative=(residual-f_pert)/(p-p_pert)
            if not math.isfinite(derivative) or derivative<=0:
                raise SubsonicFlowError(
                    "momentum equation lost the subsonic pressure root"
                )
            new_p=p-residual/derivative
            # Subsonic Newton damping avoids shooting below the low-Mach root.
            delta=max(-0.1*p_in,min(0.1*p_in,new_p-p))
            p+=delta
            if p<=0 or not math.isfinite(p):
                raise SubsonicFlowError("nonpositive pressure during momentum solve")
        else:
            raise RuntimeError(f"cell {i} compressible momentum not converged")

        energy_err=mdot*(
            gas_out.enthalpy_j_kg+0.5*v_out*v_out
            -gas_in.enthalpy_j_kg-0.5*v_in*v_in
        )-q
        momentum_err=(p_in-p)-(fric+acc)
        if abs(energy_err)>1e-7*max(1.,q,abs(mdot*target)):
            raise RuntimeError(f"cell {i} stagnation energy audit failed")
        if abs(momentum_err)>1e-8*max(1.,p_in):
            raise RuntimeError(f"cell {i} momentum balance audit failed")
        mean_t=0.5*(t_in+t)
        cp_mean=_state(mean_t,0.5*(p_in+p)).frozen_cp_j_kg_k
        pr=cp_mean*props.dynamic_viscosity_pa_s/props.thermal_conductivity_w_m_k
        nu=nusselt_number(reynolds,pr)
        h=nu*props.thermal_conductivity_w_m_k/d
        surface=math.pi*d*dz
        flux=q_wall/surface
        t_wall=mean_t+flux/h

        outlet_t[i]=t
        outlet_p[i]=p
        wall_t[i]=t_wall
        heat_flux[i]=flux
        htc[i]=h
        mach[i]=m_out
        dp_f[i]=fric
        dp_acc[i]=acc
        energy_residual[i]=energy_err
        momentum_residual[i]=momentum_err
        t_in,p_in=t,p

    return ChannelSolution(
        z_center_m=0.5*(z[:-1]+z[1:]),
        bulk_temperature_k=outlet_t,
        wall_temperature_k=wall_t,
        pressure_pa=outlet_p,
        reynolds=re,
        heat_transfer_coefficient_w_m2_k=htc,
        heat_flux_w_m2=heat_flux,
        channel_power_w=wall+gas,
        wall_heat_power_w=wall,
        direct_coolant_power_w=gas,
        mass_flow_kg_s=mdot,
        channel_diameter_m=d,
        mach_number=mach,
        friction_pressure_loss_pa=dp_f,
        acceleration_pressure_loss_pa=dp_acc,
        energy_residual_w=energy_residual,
        momentum_residual_pa=momentum_residual,
    )
