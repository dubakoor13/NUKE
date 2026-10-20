"""Parallel nozzle-jacket/reflector and counterflow tie-tube cooling.

Topology is a reduced-order approximation of NASA/TM-2013-216553,
"Small NERVA Derived Engine" (2013), p.6:
    LH2 pump -> splitter -> [nozzle jacket -> reflector] --+
                          [tie supply -> tie return] -------+ -> common header
The common header then drives the existing turbine-bleed/core cycle.

The coolant H2 uses ONE CoolProp enthalpy reference from LH2 through
gasification; every path conserves total (stagnation) enthalpy, and
the outlet pressure is obtained from streamwise momentum:
    p_in - p_out = G*(v_out-v_in)
                   + K*G*(v_in+v_out)/4.
K is an externally supplied Darcy+minor-loss COEFFICIENT, not a
calibrated channel geometry/roughness prediction. Density is the
CoolProp homogeneous equilibrium density including two-phase states;
the model does NOT resolve two-phase slip, critical heat flux,
wall boiling, onset of choking or heat-transfer coefficients.
The tie supply and annular return are TWO serial coolant legs flowing
in opposite reactor-axis directions; their heat loads are imposed,
not computed from conjugate ZrH wall conduction.

Only the pressure split at the pump and mixture at the common header
are modeled. Downstream flow division/turbine bypass/recombination in
the NASA derived expander architecture is still approximated by the
existing separate turbine-exhaust bleed system.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .lh2_feed_cycle import HydrogenEOS, LH2Cycle, CryogenicCycleError
from .thermal import friction_factor


@dataclass(frozen=True)
class CoolingLeg:
    heat_power_w: float
    flow_area_m2: float
    # Legacy aggregate resistance. For geometry-based mode leave at zero.
    pressure_loss_coefficient: float = 0.
    length_m: float | None = None
    hydraulic_diameter_m: float | None = None
    roughness_m: float = 0.
    minor_loss_coefficient: float = 0.

    @property
    def geometry_based(self) -> bool:
        return self.length_m is not None

    def validate(self):
        for name in (
            "heat_power_w","flow_area_m2","pressure_loss_coefficient",
            "roughness_m","minor_loss_coefficient",
        ):
            val=getattr(self,name)
            if not math.isfinite(val) or val<0:
                raise ValueError(f"invalid cooling leg {name}")
        if self.flow_area_m2<=0:
            raise ValueError("cooling leg flow area must be positive")
        if (self.length_m is None) != (self.hydraulic_diameter_m is None):
            raise ValueError("length and hydraulic diameter must both be specified")
        if self.geometry_based:
            if (
                not math.isfinite(self.length_m) or self.length_m <= 0
                or not math.isfinite(self.hydraulic_diameter_m)
                or self.hydraulic_diameter_m <= 0
            ):
                raise ValueError("geometry length and hydraulic diameter must be positive")
            if self.pressure_loss_coefficient != 0:
                raise ValueError("geometry-based Darcy K cannot be combined with legacy K")
        elif self.roughness_m != 0 or self.minor_loss_coefficient != 0:
            raise ValueError("roughness/minor losses require length and diameter")


@dataclass(frozen=True)
class CoolingNetwork:
    nozzle_jacket: CoolingLeg
    reflector: CoolingLeg
    tie_supply: CoolingLeg
    tie_return: CoolingLeg
    min_branch_fraction: float = 0.02

    def validate(self):
        for section in (self.nozzle_jacket,self.reflector,self.tie_supply,self.tie_return):
            section.validate()
        if not math.isfinite(self.min_branch_fraction) or not 0<self.min_branch_fraction<0.45:
            raise ValueError("invalid minimum hydraulic branch flow")
        if all(
            x.pressure_loss_coefficient==0 and not x.geometry_based
            for x in (
                self.nozzle_jacket,self.reflector,self.tie_supply,self.tie_return
            )
        ):
            raise ValueError("at least one hydraulic resistance is required")

    @property
    def total_heat_w(self) -> float:
        return sum(x.heat_power_w for x in (
            self.nozzle_jacket,self.reflector,self.tie_supply,self.tie_return
        ))


@dataclass(frozen=True)
class CoolingLegState:
    name: str
    mass_flow_kg_s: float
    inlet_pressure_pa: float
    outlet_pressure_pa: float
    inlet_stagnation_enthalpy_j_kg: float
    outlet_stagnation_enthalpy_j_kg: float
    inlet_static_temperature_k: float
    outlet_static_temperature_k: float
    inlet_quality: float
    outlet_quality: float
    heat_power_w: float
    irreversible_pressure_drop_pa: float
    acceleration_pressure_drop_pa: float
    energy_residual_w: float
    momentum_residual_pa: float
    effective_pressure_loss_coefficient: float = 0.
    reynolds: float | None = None
    darcy_friction_factor: float | None = None
    darcy_pressure_drop_pa: float | None = None
    minor_pressure_drop_pa: float | None = None
    outlet_mach: float | None = None


@dataclass(frozen=True)
class CoolingNetworkPoint:
    nozzle_reflector_mass_flow_kg_s: float
    tie_tube_mass_flow_kg_s: float
    tie_tube_fraction: float
    common_header_pressure_pa: float
    common_header_temperature_k: float
    common_header_stagnation_enthalpy_j_kg: float
    total_heat_power_w: float
    split_pressure_residual_pa: float
    total_energy_residual_w: float
    maximum_leg_energy_residual_w: float
    maximum_leg_momentum_residual_pa: float
    max_two_phase_leg_count: int
    legs: tuple[CoolingLegState,...]


@dataclass(frozen=True)
class PumpConditionedHeader:
    pump_pressure_pa: float
    pump_temperature_k: float
    pump_shaft_power_w: float
    header_pressure_pa: float
    header_temperature_k: float
    header_stagnation_enthalpy_j_kg: float
    tank_enthalpy_j_kg: float
    inlet_npsh_available_m: float
    cooling: CoolingNetworkPoint


def _static_state(
    eos: HydrogenEOS, p: float, h0: float, flux: float
) -> tuple[float,float,float,float]:
    """Given stagnation enthalpy, return (h_static,rho,T,Q).

    The homogeneous-equilibrium quality Q is -1 for single phase.
    This retains kinetic energy exactly even for large density changes.
    """
    if p<=0 or not math.isfinite(p):
        raise CryogenicCycleError("coolant pressure must remain positive")
    h=h0
    for _ in range(12):
        rho=eos.hp("Dmass",h,p)
        if rho<=0:
            raise CryogenicCycleError("invalid coolant mixture density")
        v=flux/rho
        updated=h0-0.5*v*v
        if not math.isfinite(updated):
            raise CryogenicCycleError("invalid coolant stagnation-to-static conversion")
        if abs(updated-h)<=1e-10*max(1.,abs(h0)):
            h=updated
            break
        # Kinetic correction is small on the model's admissible
        # subsonic branch. A full thermodynamic fixed-point update
        # converges much faster than artificial 50% damping.
        h=updated
    else:
        raise CryogenicCycleError("coolant stagnation/static enthalpy iteration diverged")
    rho=eos.hp("Dmass",h,p)
    t=eos.hp("T",h,p)
    quality=eos.hp("Q",h,p)
    if t<14 or t>=900:
        raise CryogenicCycleError("coolant state outside CoolProp 14-900 K screening range")
    if not math.isfinite(quality):
        quality=-1.
    return h,rho,t,quality



def _geometry_resistance(
    eos: HydrogenEOS, leg: CoolingLeg,
    mass_flux: float, pin: float, pout: float,
    hin: float, hout: float,
) -> tuple[float, float, float, float]:
    """Return (K_total, K_Darcy, Reynolds, Darcy f).

    Darcy K=f_D*L/D, evaluated at midpoint thermodynamic state.
    For parallel channels, area is the total open flow area and D_h
    is the hydraulic diameter of an individual passage.

    The fluid property closure is SINGLE PHASE or SUPERCRITICAL only.
    Two-phase friction needs a separate validated multiplier/slip
    model and cannot be inferred from a one-phase Haaland relation.
    """
    if not leg.geometry_based:
        raise ValueError("geometry resistance requested on aggregate-K leg")
    p_mid=0.5*(pin+pout)
    h_mid=0.5*(hin+hout)
    quality=eos.hp("Q",h_mid,p_mid)
    if 0. <= quality <= 1.:
        raise CryogenicCycleError(
            "two-phase midpoint cannot use single-phase Darcy geometry; "
            "requires validated two-phase friction model"
        )
    viscosity=eos.hp("VISCOSITY",h_mid,p_mid)
    if viscosity<=0 or not math.isfinite(viscosity):
        raise CryogenicCycleError("nonphysical hydrogen viscosity")
    re=abs(mass_flux)*leg.hydraulic_diameter_m/viscosity
    if not math.isfinite(re) or re<=0:
        raise CryogenicCycleError("invalid hydraulic Reynolds number")
    f=friction_factor(re,leg.roughness_m/leg.hydraulic_diameter_m)
    kd=float(f)*leg.length_m/leg.hydraulic_diameter_m
    return kd+leg.minor_loss_coefficient,kd,re,float(f)


def _leg(
    eos: HydrogenEOS, name: str, leg: CoolingLeg,
    mdot: float, pin: float, h0in: float,
) -> CoolingLegState:
    if mdot<=0 or pin<=0:
        raise CryogenicCycleError("invalid cooling-branch inlet flow/pressure")
    G=mdot/leg.flow_area_m2
    hin,rhoin,tin,qin=_static_state(eos,pin,h0in,G)
    vin=G/rhoin
    h0out=h0in+leg.heat_power_w/mdot
    if not math.isfinite(h0out):
        raise CryogenicCycleError("invalid cooling heat input")

    def f(p: float):
        hout,rhoout,tout,qout=_static_state(eos,p,h0out,G)
        vout=G/rhoout
        if leg.geometry_based:
            ktot,kdarcy,reynolds,darcy_f=_geometry_resistance(
                eos,leg,G,pin,p,hin,hout
            )
            sound=eos.hp("A",hout,p)
            mach=vout/sound
            if not math.isfinite(mach) or mach>=0.85:
                raise CryogenicCycleError(
                    "geometry coolant passage requires subsonic Mach below 0.85"
                )
            darcy_loss=kdarcy*G*(vin+vout)/4.
            minor_loss=leg.minor_loss_coefficient*G*(vin+vout)/4.
        else:
            ktot=leg.pressure_loss_coefficient
            reynolds=darcy_f=mach=None
            darcy_loss=minor_loss=None
        irrev=ktot*G*(vin+vout)/4.
        acc=G*(vout-vin)
        return p+irrev+acc-pin,(
            hout,tout,qout,irrev,acc,ktot,reynolds,darcy_f,
            darcy_loss,minor_loss,mach
        )

    # Fixed-point momentum solve with damping. In the subsonic regime,
    # loss coefficients depend on outlet density, which itself changes
    # with pressure. Refuse divergence rather than invent a root.
    p=pin
    for _ in range(15):
        residual,out=f(p)
        irrev,acc=out[3],out[4]
        estimated=pin-irrev-acc
        if estimated<=0 or estimated>pin:
            raise CryogenicCycleError(
                "cooling leg has no admissible positive-pressure loss root"
            )
        if abs(residual)<1e-8*max(1.,pin):
            break
        # For a subsonic channel the fixed-point derivative is well
        # below unity; full updates converge faster than half-steps.
        # If density/phase feedback prevents convergence, fail closed.
        p=estimated
    else:
        raise CryogenicCycleError(
            "cooling leg momentum failed to converge; possible HEM choking"
        )
    pout=p
    residual,out=f(pout)
    (hout,tout,qout,irrev,acc,ktot,reynolds,darcy_f,
     darcy_loss,minor_loss,mach)=out
    energy=mdot*((hout+0.5*(G/eos.hp("Dmass",hout,pout))**2)-
                 (hin+0.5*vin**2))-leg.heat_power_w
    if abs(residual)>1e-7*pin:
        raise RuntimeError("cooling leg momentum did not close")
    if abs(energy)>1e-7*max(1.,leg.heat_power_w):
        raise RuntimeError("cooling leg total enthalpy did not close")
    return CoolingLegState(
        name=name,mass_flow_kg_s=mdot,
        inlet_pressure_pa=pin,outlet_pressure_pa=pout,
        inlet_stagnation_enthalpy_j_kg=h0in,
        outlet_stagnation_enthalpy_j_kg=h0out,
        inlet_static_temperature_k=tin,
        outlet_static_temperature_k=tout,
        inlet_quality=qin,outlet_quality=qout,
        heat_power_w=leg.heat_power_w,
        irreversible_pressure_drop_pa=irrev,
        acceleration_pressure_drop_pa=acc,
        energy_residual_w=energy,momentum_residual_pa=residual,
        effective_pressure_loss_coefficient=ktot,
        reynolds=reynolds,darcy_friction_factor=darcy_f,
        darcy_pressure_drop_pa=darcy_loss,
        minor_pressure_drop_pa=minor_loss,
        outlet_mach=mach,
    )


def solve_cooling_network(
    network: CoolingNetwork, mdot: float, pin: float, hin: float,
    *, eos: HydrogenEOS | None = None,
) -> CoolingNetworkPoint:
    """Two parallel branches; solve equal end pressure, mix total enthalpy."""
    network.validate()
    eos=eos or HydrogenEOS()
    if not math.isfinite(mdot) or mdot<=0:
        raise ValueError("positive total coolant mass flow required")

    def trial(fraction: float):
        mt=mdot*fraction
        mr=mdot-mt
        nozzle=_leg(eos,"nozzle_jacket",network.nozzle_jacket,mr,pin,hin)
        reflector=_leg(
            eos,"reflector",network.reflector,mr,
            nozzle.outlet_pressure_pa,nozzle.outlet_stagnation_enthalpy_j_kg
        )
        tie_down=_leg(eos,"tie_supply",network.tie_supply,mt,pin,hin)
        tie_up=_leg(
            eos,"tie_return",network.tie_return,mt,
            tie_down.outlet_pressure_pa,tie_down.outlet_stagnation_enthalpy_j_kg
        )
        return (
            reflector.outlet_pressure_pa-tie_up.outlet_pressure_pa,
            (nozzle,reflector,tie_down,tie_up)
        )

    last=None
    bracket=None
    lo_lim=network.min_branch_fraction
    for beta in np.linspace(lo_lim,1.-lo_lim,9):
        try:
            residual, legs=trial(float(beta))
        except (CryogenicCycleError,RuntimeError):
            continue
        if last is not None and last[1]*residual<=0:
            bracket=(last[0],float(beta),last[1])
            break
        last=(float(beta),residual)
    if bracket is None:
        raise CryogenicCycleError(
            "no hydraulic flow split equalizes reflector and tie-tube "
            "return pressures for supplied resistances, heat and total flow"
        )
    lo,hi,flow_lo=bracket
    for _ in range(22):
        mid=(lo+hi)/2
        difference,_=trial(mid)
        if abs(difference)<1e-7*pin:
            lo=hi=mid
            break
        if (difference>0)==(flow_lo>0):
            lo,flow_lo=mid,difference
        else:
            hi=mid
    beta=(lo+hi)/2
    mismatch, legs=trial(beta)
    nozzle,reflector,supply,return_leg=legs
    pressure=0.5*(reflector.outlet_pressure_pa+return_leg.outlet_pressure_pa)
    if pressure<=0 or abs(mismatch)>1e-7*pin:
        raise CryogenicCycleError("cooling branches did not close common pressure")
    mt=mdot*beta
    mr=mdot-mt
    h=(mr*reflector.outlet_stagnation_enthalpy_j_kg+
       mt*return_leg.outlet_stagnation_enthalpy_j_kg)/mdot
    t=eos.hp("T",h,pressure)
    if t<=200 or t>=900:
        raise CryogenicCycleError(
            "mixed coolant header outside single-phase warm H2 bridge 200-900 K"
        )
    q=network.total_heat_w
    net=mdot*(h-hin)-q
    if abs(net)>1e-8*max(1.,q):
        raise RuntimeError("mixed coolant first-law closure FAILED")
    maxe=max(abs(x.energy_residual_w) for x in legs)
    maxp=max(abs(x.momentum_residual_pa) for x in legs)
    return CoolingNetworkPoint(
        nozzle_reflector_mass_flow_kg_s=mr,tie_tube_mass_flow_kg_s=mt,
        tie_tube_fraction=beta,common_header_pressure_pa=pressure,
        common_header_temperature_k=t,
        common_header_stagnation_enthalpy_j_kg=h,
        total_heat_power_w=q,split_pressure_residual_pa=mismatch,
        total_energy_residual_w=net,
        maximum_leg_energy_residual_w=maxe,
        maximum_leg_momentum_residual_pa=maxp,
        max_two_phase_leg_count=sum(
            0<=x.inlet_quality<=1 or 0<=x.outlet_quality<=1 for x in legs
        ),legs=legs,
    )


def pump_and_regenerative_cooling(
    config: LH2Cycle, network: CoolingNetwork, mdot: float,
    *, eos: HydrogenEOS | None = None
) -> PumpConditionedHeader:
    """Cryogenic LH2 pump and two-branched regeneratively heated header."""
    config.validate()
    network.validate()
    eos=eos or HydrogenEOS()
    if not math.isclose(config.preheater_thermal_power_w,
                        network.total_heat_w,rel_tol=1e-10,abs_tol=1e-7):
        raise ValueError(
            "preheater_thermal_power_w must equal sum of jacket, reflector "
            "and tie supply/return nuclear heat; no double-counting allowed"
        )
    t1,p1=config.tank_temperature_k,config.tank_pressure_pa
    pvap=eos.get("P","T",t1,"Q",0.)
    if p1<=pvap:
        raise CryogenicCycleError("tank H2 is not subcooled liquid")
    rho1=eos.tp("Dmass",t1,p1)
    if rho1<=20:
        raise CryogenicCycleError("tank hydrogen not dense liquid")
    head=config.pump_shutoff_pressure_rise_pa-config.pump_head_quadratic_pa_s2_kg2*mdot**2
    if head<=0:
        raise CryogenicCycleError("pump pressure-source curve exhausted")
    p2=p1+head
    h1=eos.tp("Hmass",t1,p1)
    s1=eos.tp("Smass",t1,p1)
    h2s=eos.sp("Hmass",s1,p2)
    if h2s<=h1:
        raise CryogenicCycleError("nonpositive isentropic pump work")
    h2=h1+(h2s-h1)/config.pump_isentropic_efficiency
    t2=eos.hp("T",h2,p2)
    rho2=eos.hp("Dmass",h2,p2)
    if t2>=34 or rho2<=20:
        raise CryogenicCycleError("pump outlet cannot be treated as dense LH2")
    qpump=mdot*(h2-h1)
    # Any remaining external feed valve/line resistance is upstream
    # of the explicit cooling network; never duplicate cooling K.
    feed_drop=config.feed_loss_quadratic_pa_s2_kg2*mdot**2
    p_cool_in=p2-feed_drop
    if p_cool_in<=p1:
        raise CryogenicCycleError("feed line loss eliminates pump head")
    result=solve_cooling_network(network,mdot,p_cool_in,h2,eos=eos)
    return PumpConditionedHeader(
        pump_pressure_pa=p2,pump_temperature_k=t2,pump_shaft_power_w=qpump,
        header_pressure_pa=result.common_header_pressure_pa,
        header_temperature_k=result.common_header_temperature_k,
        header_stagnation_enthalpy_j_kg=result.common_header_stagnation_enthalpy_j_kg,
        tank_enthalpy_j_kg=h1,
        inlet_npsh_available_m=(p1-pvap)/(rho1*9.80665),
        cooling=result,
    )
