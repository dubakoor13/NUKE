"""Quasi-steady GAS-PHASE hydrogen feed -> fuel channel -> nozzle closure.

This reduced-order subsystem solves a single unknown total hydrogen flow
against a prescribed pressure-source characteristic, an isentropic-
efficiency GAS COMPRESSOR, line losses, repeated compressible fuel channels,
and a choked H2 nozzle. Nuclear heating enters ONLY as externally supplied
thermal power in the modeled fuel channels.

It does NOT represent cryogenic LH2 pumping, tie-tube/reflector return paths,
transient turbomachinery dynamics, variable pump shaft speed, spatial
neutronics, reactor kinetics, structural stresses or historical NERVA
operating conditions. This deliberately small scope permits exact mass,
total enthalpy and pressure conservation checks, without invented pump maps.

Pressure-source closure:
  p_discharge = p_supply + (head_shutoff - k_source*mdot²)
  p_core_in   = p_discharge - k_feed*mdot²
  p_chamber   = p_core_out - k_exit*mdot²
  mdot        = At * choked_mass_flux(p_chamber,T_chamber,gamma(T_chamber))

Gas compressor entropy/enthalpy:
  s(T_2s,p_discharge) = s(T_supply,p_supply)
  h(T_2,p_discharge) = h(T_supply,p_supply)
                         + [h(T_2s)-h(T_supply)] / eta_isen

Channel energy includes fluid kinetic energy and independently tracked
friction/acceleration pressure losses. A large chamber diffuser is
assumed to convert channel-exit kinetic energy to chamber stagnation
enthalpy; its net pressure losses are represented by k_exit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .compressible_channel import (
    SubsonicFlowError, R_H2, solve_fuel_channel_compressible,
)
from .nasa_hydrogen import T_MAX, T_MIN, hydrogen_state
from .nerva_engine import (
    NervaGasModel, NervaNozzleResult, choked_mass_flux_kg_m2_s,
    evaluate_nerva_nozzle,
)
from .thermal import ChannelSolution, HydrogenProperties


class NoSubsonicNetworkSolution(ValueError):
    """No consistent, physically admissible gas network root exists."""


@dataclass(frozen=True)
class GasFeedNetwork:
    supply_pressure_pa: float
    supply_temperature_k: float
    pressure_source_shutoff_pa: float
    pressure_source_quadratic_pa_s2_kg2: float
    feed_quadratic_pa_s2_kg2: float
    exit_quadratic_pa_s2_kg2: float
    isentropic_compression_efficiency: float
    throat_area_m2: float
    expansion_ratio: float
    channel_count: int
    channel_diameter_m: float
    z_edges_m: tuple[float, ...]
    wall_heating_w: tuple[float, ...]
    direct_hydrogen_heating_w: tuple[float, ...]
    nozzle_efficiency: float = 0.95
    ambient_pressure_pa: float = 0.
    maximum_total_flow_kg_s: float = 5.
    maximum_mach: float = 0.85

    def validate(self) -> None:
        positive = (
            "supply_pressure_pa", "supply_temperature_k",
            "pressure_source_shutoff_pa", "throat_area_m2",
            "channel_diameter_m", "maximum_total_flow_kg_s",
        )
        for name in positive:
            x = getattr(self,name)
            if not math.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in (
            "pressure_source_quadratic_pa_s2_kg2",
            "feed_quadratic_pa_s2_kg2",
            "exit_quadratic_pa_s2_kg2", "ambient_pressure_pa",
        ):
            x = getattr(self,name)
            if not math.isfinite(x) or x < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if not T_MIN <= self.supply_temperature_k < T_MAX:
            raise ValueError("supply H2 must be molecular gas in NASA7 200-3500 K")
        if not (0 < self.isentropic_compression_efficiency <= 1):
            raise ValueError("isentropic compression efficiency outside (0,1]")
        if not (0 < self.nozzle_efficiency <= 1):
            raise ValueError("nozzle efficiency outside (0,1]")
        if not 0 < self.maximum_mach < 1:
            raise ValueError("subsonic Mach limit must lie in (0,1)")
        if not self.expansion_ratio >= 1:
            raise ValueError("expansion ratio must be >= 1")
        if not isinstance(self.channel_count,int) or self.channel_count <= 0:
            raise ValueError("channel count must be positive")
        z=np.asarray(self.z_edges_m)
        w=np.asarray(self.wall_heating_w)
        g=np.asarray(self.direct_hydrogen_heating_w)
        if (w.ndim!=1 or w.size<1 or g.shape!=w.shape
                or z.shape!=(w.size+1,) or np.any(np.diff(z)<=0)
                or not np.all(np.isfinite(z)) or not np.all(np.isfinite(w))
                or not np.all(np.isfinite(g)) or np.any(w<0) or np.any(g<0)):
            raise ValueError("invalid axial heat deposition/z-grid")


@dataclass(frozen=True)
class NetworkOperatingPoint:
    total_mass_flow_kg_s: float
    supply_pressure_pa: float
    pressure_source_discharge_pa: float
    fuel_channel_inlet_pa: float
    fuel_channel_exit_pa: float
    chamber_pressure_pa: float
    supply_temperature_k: float
    compressor_outlet_temperature_k: float
    chamber_temperature_k: float
    pressure_source_rise_pa: float
    feed_loss_pa: float
    exit_loss_pa: float
    channel_friction_loss_pa: float
    channel_acceleration_loss_pa: float
    compressor_shaft_power_w: float
    deposited_reactor_heat_w: float
    compressed_gas_enthalpy_rise_w: float
    nozzle_mass_flow_residual_kg_s: float
    total_energy_residual_w: float
    pressure_balance_residual_pa: float
    channel: ChannelSolution
    nozzle: NervaNozzleResult


def _isentropic_temperature(t1: float, p1: float, p2: float) -> float:
    s_in = hydrogen_state(t1,p1,mode="frozen").entropy_j_kg_k
    if p2 < p1:
        raise ValueError("compressor cannot reduce gas pressure")
    lo,hi=t1,T_MAX
    if hydrogen_state(hi,p2,mode="frozen").entropy_j_kg_k < s_in:
        raise NoSubsonicNetworkSolution("compressor isentrope above NASA7 range")
    for _ in range(58):
        mid=(lo+hi)/2.
        if hydrogen_state(mid,p2,mode="frozen").entropy_j_kg_k < s_in:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2.


def _temperature_at_enthalpy(h_j_kg: float, p: float) -> float:
    low=hydrogen_state(T_MIN,p,mode="frozen").enthalpy_j_kg
    high=hydrogen_state(T_MAX,p,mode="frozen").enthalpy_j_kg
    if not low<=h_j_kg<=high:
        raise NoSubsonicNetworkSolution(
            "stagnation enthalpy outside pure H2 NASA7 temperature domain"
        )
    lo,hi=T_MIN,T_MAX
    for _ in range(58):
        mid=(lo+hi)/2.
        if hydrogen_state(mid,p,mode="frozen").enthalpy_j_kg < h_j_kg:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2.


def evaluate_gas_network(
    config: GasFeedNetwork, total_mass_flow_kg_s: float,
    *, hydrogen_transport: HydrogenProperties | None = None,
) -> tuple[NetworkOperatingPoint,float]:
    """Evaluate complete physics at a proposed mass flow.

    Returns operating point and signed nozzle minus feed mass residual.
    Solver-only candidates are not necessarily closed; accept only a
    small root residual from solve_gas_network.
    """
    config.validate()
    m=float(total_mass_flow_kg_s)
    if not math.isfinite(m) or m<=0 or m>config.maximum_total_flow_kg_s:
        raise ValueError("invalid trial feed mass flow")
    head=(config.pressure_source_shutoff_pa
          -config.pressure_source_quadratic_pa_s2_kg2*m*m)
    if head <= 0:
        raise NoSubsonicNetworkSolution("pressure source curve exhausted")
    p_dis=config.supply_pressure_pa+head
    feed=config.feed_quadratic_pa_s2_kg2*m*m
    core_p=p_dis-feed
    if core_p<=config.supply_pressure_pa*0.1:
        raise NoSubsonicNetworkSolution("feed line loss exceeds available pressure")
    inlet=hydrogen_state(config.supply_temperature_k,
                         config.supply_pressure_pa,mode="frozen")
    t_ideal=_isentropic_temperature(
        config.supply_temperature_k,config.supply_pressure_pa,p_dis
    )
    h_ideal=hydrogen_state(t_ideal,p_dis,mode="frozen").enthalpy_j_kg
    h2=inlet.enthalpy_j_kg+(
        h_ideal-inlet.enthalpy_j_kg
    )/config.isentropic_compression_efficiency
    t_comp=_temperature_at_enthalpy(h2,p_dis)
    q=sum(config.wall_heating_w)+sum(config.direct_hydrogen_heating_w)
    core=solve_fuel_channel_compressible(
        np.asarray(config.wall_heating_w,dtype=float),
        np.asarray(config.z_edges_m,dtype=float),
        config.channel_count,
        m/config.channel_count,
        t_comp,core_p,config.channel_diameter_m,
        properties=hydrogen_transport or HydrogenProperties(),
        axial_direct_coolant_power_w=np.asarray(
            config.direct_hydrogen_heating_w,dtype=float
        ),
        max_mach=config.maximum_mach,
    )
    p_exit=core.outlet_pressure_pa
    exit_loss=config.exit_quadratic_pa_s2_kg2*m*m
    pc=p_exit-exit_loss
    if pc<=0:
        raise NoSubsonicNetworkSolution("core cannot supply positive chamber pressure")
    h_exit=hydrogen_state(
        core.outlet_temperature_k,p_exit,mode="frozen"
    ).enthalpy_j_kg
    last_rho=hydrogen_state(
        core.outlet_temperature_k,p_exit,mode="frozen"
    ).density_kg_m3
    area=math.pi*config.channel_diameter_m**2/4
    v_out=(m/config.channel_count)/(last_rho*area)
    h_stag=h_exit+0.5*v_out*v_out
    tc=_temperature_at_enthalpy(h_stag,pc)
    chamber_state=hydrogen_state(tc,pc,mode="frozen")
    cp=chamber_state.frozen_cp_j_kg_k
    gamma=cp/(cp-R_H2)
    model=NervaGasModel(
        gamma=gamma,gas_constant_j_kg_k=R_H2,cp_j_kg_k=cp
    )
    nozzle=evaluate_nerva_nozzle(
        pc,tc,config.throat_area_m2,config.expansion_ratio,
        ambient_pressure_pa=config.ambient_pressure_pa,
        nozzle_efficiency=config.nozzle_efficiency,
        gas=model,
    )
    # Energy equation for entire network: compressor shaft work and
    # reactor deposition equal stagnation enthalpy delivered downstream.
    compression_power=m*(h2-inlet.enthalpy_j_kg)
    balance=m*(h_stag-inlet.enthalpy_j_kg)-(compression_power+q)
    dp_core=p_exit-core_p
    pressure_balance=(
        (config.supply_pressure_pa+head)-(feed+(core_p-p_exit)+exit_loss+pc)
    )
    result=NetworkOperatingPoint(
        total_mass_flow_kg_s=m,
        supply_pressure_pa=config.supply_pressure_pa,
        pressure_source_discharge_pa=p_dis,
        fuel_channel_inlet_pa=core_p,
        fuel_channel_exit_pa=p_exit,
        chamber_pressure_pa=pc,
        supply_temperature_k=config.supply_temperature_k,
        compressor_outlet_temperature_k=t_comp,
        chamber_temperature_k=tc,
        pressure_source_rise_pa=head,
        feed_loss_pa=feed,exit_loss_pa=exit_loss,
        channel_friction_loss_pa=float(np.sum(core.friction_pressure_loss_pa)),
        channel_acceleration_loss_pa=float(np.sum(core.acceleration_pressure_loss_pa)),
        compressor_shaft_power_w=compression_power,
        deposited_reactor_heat_w=q,
        compressed_gas_enthalpy_rise_w=compression_power,
        nozzle_mass_flow_residual_kg_s=m-nozzle.mass_flow_kg_s,
        total_energy_residual_w=balance,
        pressure_balance_residual_pa=pressure_balance,
        channel=core,nozzle=nozzle,
    )
    return result, result.nozzle_mass_flow_residual_kg_s


def solve_gas_network(
    config: GasFeedNetwork, *,
    bracket_samples: int = 80,
    relative_mass_tolerance: float = 1e-7,
    hydrogen_transport: HydrogenProperties | None = None,
) -> NetworkOperatingPoint:
    """Bracket and solve unique low-Mach operating branch by mass balance.

    A sparse initial log-grid excludes invalid source head, overheating,
    choking and unstable pressure roots. Physical solutions require an
    attainable sign change; failure is explicit, never fabricated.
    """
    config.validate()
    if bracket_samples<16 or bracket_samples>1000:
        raise ValueError("bracket_samples must lie in [16,1000]")
    if not math.isfinite(relative_mass_tolerance) or not 0<relative_mass_tolerance<1e-3:
        raise ValueError("invalid mass-flow closure tolerance")
    limit=config.maximum_total_flow_kg_s
    k=config.pressure_source_quadratic_pa_s2_kg2
    if k>0:
        limit=min(limit,math.sqrt(config.pressure_source_shutoff_pa/k)*(1-1e-8))
    if limit<=0:
        raise NoSubsonicNetworkSolution("no positive source-head operating range")
    candidates=np.geomspace(
        max(limit*1e-4,1e-8),limit,bracket_samples
    )
    previous=None
    bracket=None
    for m in candidates:
        try:
            point,fx=evaluate_gas_network(
                config,float(m),hydrogen_transport=hydrogen_transport
            )
        except (ValueError,RuntimeError,OverflowError):
            continue
        if previous is not None and fx*previous[1] <= 0:
            bracket=(previous[0],float(m),previous[1],fx)
            break
        previous=(float(m),fx)
    if bracket is None:
        raise NoSubsonicNetworkSolution(
            "no mass-flow root bracket in admissible single-phase "
            "subsonic regime; verify source curve, heating and throat area"
        )
    lo,hi,flo,fhi=bracket
    if flo>fhi:
        raise NoSubsonicNetworkSolution("nonmonotone source/nozzle flow closure")
    result=None
    for _ in range(65):
        m=(lo+hi)/2.
        result,fx=evaluate_gas_network(
            config,m,hydrogen_transport=hydrogen_transport
        )
        if abs(fx)<=relative_mass_tolerance*max(
            m,result.nozzle.mass_flow_kg_s
        ):
            break
        if fx>0:
            hi=m
        else:
            lo=m
    if result is None or abs(result.nozzle_mass_flow_residual_kg_s)>relative_mass_tolerance*max(result.total_mass_flow_kg_s,result.nozzle.mass_flow_kg_s):
        raise NoSubsonicNetworkSolution("nozzle/feed mass flow did not converge")
    # Conservation checks independent of bisection stop condition.
    scale=max(
        1.0,result.compressor_shaft_power_w,
        result.deposited_reactor_heat_w
    )
    if abs(result.total_energy_residual_w)>1e-7*scale:
        raise RuntimeError("gas network total energy audit did not close")
    if abs(result.pressure_balance_residual_pa)>1e-8*max(1.,result.pressure_source_discharge_pa):
        raise RuntimeError("gas network pressure audit did not close")
    return result
