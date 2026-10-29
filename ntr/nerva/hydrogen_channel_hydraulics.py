"""Stage56: compressible hydrogen-channel hydraulics, independent of reactor design.

Solves a positive-flow, single-phase, constant-diameter, quasi-steady 1D
momentum balance for the existing 19 coolant channels with *prescribed*
temperature profiles and mass flows. The equation includes acceleration
and Darcy wall friction, with an explicit Mach-number validity guard.

This module intentionally starts with a transparent **ideal H2 gas**
baseline, NOT a validated real-gas high-pressure equation of state.
Specific gas constants are dimensional; temperature-dependent viscosity
and cp are supplied via bounded tables. Real-gas nonideality, hydrogen
dissociation, conjugate pressure feedback, changes of channel mass over
time, and choked flow are not silently approximated.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import brentq

from .config import NervaConfig
from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable

R_UNIVERSAL_J_MOL_K = 8.31446261815324
H2_MOLAR_MASS_KG_MOL = 0.00201588


def _vector(name: str, data: object, n: int, *, allow_zero: bool=False
            ) -> np.ndarray:
    a=np.asarray(data,dtype=float)
    if a.ndim==0:
        a=np.full(n,float(a))
    if a.shape!=(n,) or not np.all(np.isfinite(a)) or (
        np.any(a<0.) if allow_zero else np.any(a<=0.)
    ):
        raise ValueError(f"{name} must be finite {'nonnegative' if allow_zero else 'positive'} scalar/{n}-vector")
    return a


@dataclass(frozen=True)
class IdealHydrogenProperties:
    """Pressure-dependent ideal-gas density and bounded T property tables.

    c_p > R guarantees gamma=c_p/(c_p-R)>1 and positive ideal-gas sound
    speed. The model is only appropriate for an independently qualified
    single-phase, non-dissociating gas state.
    """
    cp_j_kg_k: TemperatureTable
    viscosity_pa_s: TemperatureTable
    molar_mass_kg_mol: float = H2_MOLAR_MASS_KG_MOL

    def __post_init__(self) -> None:
        if not (math.isfinite(self.molar_mass_kg_mol)
                and self.molar_mass_kg_mol>0):
            raise ValueError("hydrogen molecular mass must be finite positive")
        rg=self.gas_constant
        if np.any(self.cp_j_kg_k.values<=rg):
            raise ValueError("gas cp(T) must exceed R for stable ideal-gas gamma")
        if self.viscosity_pa_s.allow_zero or self.cp_j_kg_k.allow_zero:
            raise ValueError("gas viscosity and cp must both be strictly positive")

    @property
    def gas_constant(self) -> float:
        return R_UNIVERSAL_J_MOL_K/self.molar_mass_kg_mol

    def density_kg_m3(self, p_pa: object, t_k: object) -> np.ndarray:
        p=np.asarray(p_pa,dtype=float)
        t=np.asarray(t_k,dtype=float)
        if not np.all(np.isfinite(p)) or np.any(p<=0):
            raise ValueError("absolute hydrogen pressure must be positive finite")
        self.cp_j_kg_k.at(t)
        return p/(self.gas_constant*t)

    def gamma(self,t_k: object) -> np.ndarray:
        cp=self.cp_j_kg_k.at(t_k)
        return cp/(cp-self.gas_constant)

    def sound_speed_m_s(self,t_k: object) -> np.ndarray:
        t=np.asarray(t_k,dtype=float)
        return np.sqrt(self.gamma(t)*self.gas_constant*t)

    def enthalpy_j_kg(self,t_k: object) -> np.ndarray:
        return self.cp_j_kg_k.integral(t_k)


def darcy_friction_factor(reynolds: object, relative_roughness: float
                          ) -> np.ndarray:
    """Laminar 64/Re; Haaland turbulent; explicit transition interpolation.

    The transition band 2300--4000 is a numerical engineering closure,
    NOT a universally validated roughness/transition correlation.
    It must be treated as an uncertainty for predictive reactor operation.
    """
    if not (math.isfinite(relative_roughness)
            and 0<=relative_roughness<0.05):
        raise ValueError("relative roughness must be in [0,0.05)")
    re=np.asarray(reynolds,dtype=float)
    if not np.all(np.isfinite(re)) or np.any(re<0):
        raise ValueError("Reynolds number must be finite nonnegative")
    out=np.zeros_like(re)
    active=re>0
    if not np.any(active):
        return out
    r=re[active]
    laminar=64./r
    turbulent=(-1.8*np.log10(
        (relative_roughness/3.7)**1.11+6.9/r))**(-2)
    t=np.clip((r-2300.)/1700.,0.,1.)
    out[active]=(1.-t)*laminar+t*turbulent
    if not np.all(np.isfinite(out)) or np.any(out[active]<=0):
        raise ArithmeticError("nonphysical Darcy friction closure")
    return out


@dataclass(frozen=True)
class ChannelHydraulicSolution:
    """Arrays use [axial_face, channel] and [axial_segment, channel]."""
    axial_edges_m: np.ndarray
    channel_diameter_m: np.ndarray
    mass_flow_kg_s: np.ndarray
    pressure_faces_pa: np.ndarray
    temperature_faces_k: np.ndarray
    density_faces_kg_m3: np.ndarray
    velocity_faces_m_s: np.ndarray
    mach_faces: np.ndarray
    reynolds_segments: np.ndarray
    darcy_friction_segments: np.ndarray
    friction_loss_pa: np.ndarray
    momentum_residual_pa: np.ndarray
    channel_gas_mass_kg: np.ndarray
    flow_enthalpy_faces_w: np.ndarray

    @property
    def maximum_mach(self) -> float:
        return float(np.max(self.mach_faces))

    @property
    def pressure_drop_pa(self) -> np.ndarray:
        return self.pressure_faces_pa[0]-self.pressure_faces_pa[-1]

    @property
    def total_fluid_inventory_kg(self) -> float:
        return float(np.sum(self.channel_gas_mass_kg))


def solve_ideal_hydrogen_hydraulics(
    config: NervaConfig,
    eos: IdealHydrogenProperties,
    *,
    axial_edges_m: object,
    hydrogen_temperature_k: object,
    inlet_temperature_k: object,
    inlet_pressure_pa: object,
    mass_flow_kg_s: object,
    absolute_roughness_m: float=0.,
    minimum_pressure_pa: float=1.e3,
    maximum_mach: float=0.75,
) -> ChannelHydraulicSolution:
    """Solve the steady compressible momentum balance in 19 *fixed* bores.

    For each axial control volume, integrated steady momentum is

      p_out - p_in + G**2*(1/rho_out-1/rho_in)
      + f_D*(dz/D)*G**2/(2*rho_mid) = 0,

    G=mdot/A, density=p/(R*T), with film/viscosity evaluated at bounded
    temperature and midpoint pressure for the friction term.

    Temperature inputs are zone-OUTLET temperatures, consistent with the
    upwind Stage55 axial H2 enthalpy nodes. For fixed flow and geometry,
    EOS inventory is a quasi-steady diagnostic and must NOT be inserted
    into a transient equation without a continuity equation.
    """
    config.validate()
    z=np.asarray(axial_edges_m,dtype=float)
    if (z.ndim!=1 or len(z)<2 or len(z)>257
            or not np.all(np.isfinite(z)) or np.any(np.diff(z)<=0.)
            or abs(z[0])>1e-12
            or abs(z[-1]-config.active_length_cm/100)>1e-10):
        raise ValueError("axial edges must strictly partition original active length")
    nz=len(z)-1
    temp=np.asarray(hydrogen_temperature_k,dtype=float)
    if temp.shape!=(nz,19) or not np.all(np.isfinite(temp)):
        raise ValueError("hydrogen outlet node temperatures require finite [nz,19] array")
    inlet_t=_vector("hydrogen inlet T",inlet_temperature_k,19)
    pressure=_vector("hydrogen inlet absolute p",inlet_pressure_pa,19)
    flow=_vector("channel mass flow",mass_flow_kg_s,19,allow_zero=True)
    if not (math.isfinite(absolute_roughness_m) and absolute_roughness_m>=0):
        raise ValueError("wall absolute roughness must be finite nonnegative")
    if (not math.isfinite(minimum_pressure_pa) or minimum_pressure_pa<=0
            or not math.isfinite(maximum_mach) or not 0<maximum_mach<=0.75):
        raise ValueError("positive pressure floor and subsonic Mach limit <=0.75 required")
    diameter=float(config.coolant_bore_diameter_cm/100)
    D=np.full(19,diameter)
    if absolute_roughness_m>=0.05*diameter:
        raise ValueError("unqualified high wall relative roughness")
    area=np.pi*D**2/4
    G=flow/area
    face_t=np.vstack((inlet_t[None,:],temp))
    eos.cp_j_kg_k.at(face_t)
    eos.viscosity_pa_s.at(face_t)
    faces=np.zeros((nz+1,19))
    faces[0,:]=pressure
    density=np.zeros_like(faces)
    velocity=np.zeros_like(faces)
    mach=np.zeros_like(faces)
    re=np.zeros((nz,19))
    darcy=np.zeros_like(re)
    loss=np.zeros_like(re)
    residual=np.zeros_like(re)
    inventory=np.zeros_like(re)
    rho0=eos.density_kg_m3(pressure,inlet_t)
    density[0,:]=rho0
    velocity[0,:]=G/rho0
    mach[0,:]=velocity[0,:]/eos.sound_speed_m_s(inlet_t)
    if np.any(mach[0,:]>=maximum_mach):
        raise ValueError("inlet already exceeds qualified subsonic Mach limit")

    for zz,dz in enumerate(np.diff(z)):
        for ch in range(19):
            tin=float(face_t[zz,ch])
            tout=float(face_t[zz+1,ch])
            pin=float(faces[zz,ch])
            gi=float(G[ch])
            din=float(density[zz,ch])
            mean_t=.5*(tin+tout)
            mu=float(eos.viscosity_pa_s.at(mean_t))
            rei=gi*diameter/mu
            fri=float(darcy_friction_factor(rei,absolute_roughness_m/diameter))
            re[zz,ch]=rei
            darcy[zz,ch]=fri
            if gi==0:
                pout=pin
            else:
                gamma_out=float(eos.gamma(tout))
                # Reject roots in the unqualified compressible/near-choked
                # region rather than choosing an unphysical high-speed branch.
                p_mach=gi*math.sqrt(eos.gas_constant*tout/gamma_out)/maximum_mach
                p_low=max(float(minimum_pressure_pa),p_mach*(1.+1e-10))
                def momentum(pout_candidate: float) -> float:
                    rhoout=pout_candidate/(eos.gas_constant*tout)
                    rhomid=(pin+pout_candidate)/(
                        2*eos.gas_constant*mean_t)
                    drag=fri*dz/diameter*gi*gi/(2*rhomid)
                    return (pout_candidate-pin
                            +gi*gi*(1./rhoout-1./din)+drag)
                if p_low>=pin*10:
                    raise ValueError("no qualified subsonic outlet branch")
                f_low=momentum(p_low)
                upper=max(pin*1.01,p_low*1.01)
                for _ in range(35):
                    if momentum(upper)>0:
                        break
                    upper*=1.4
                else:
                    raise ArithmeticError("could not bracket outlet pressure")
                if f_low>=0:
                    raise ValueError("no safe subsonic pressure root; flow may choke")
                pout=brentq(momentum,p_low,upper,xtol=1.e-8,
                            rtol=1.e-12,maxiter=100)
            faces[zz+1,ch]=pout
            rhoout=float(eos.density_kg_m3(pout,tout))
            density[zz+1,ch]=rhoout
            velocity[zz+1,ch]=gi/rhoout
            mach[zz+1,ch]=velocity[zz+1,ch]/float(eos.sound_speed_m_s(tout))
            if not (pout>=minimum_pressure_pa and
                    mach[zz+1,ch]<maximum_mach and pout>0):
                raise ValueError("hydrogen profile outside subsonic pressure qualification")
            rhomid=(pin+pout)/(2*eos.gas_constant*mean_t)
            loss[zz,ch]=fri*dz/diameter*gi*gi/(2*rhomid)
            residual[zz,ch]=(
                pout-pin+gi*gi*(1./rhoout-1./din)+loss[zz,ch])
            # Midpoint ideal-gas fluid inventory is *not* a transient
            # mass-conserving continuity closure; it is diagnostic only.
            inventory[zz,ch]=rhomid*area[ch]*dz
    if np.max(np.abs(residual))>max(
        1e-5,1e-10*float(np.max(faces))):
        raise ArithmeticError("steady compressible momentum residual failed")
    flow_enthalpy=flow[None,:]*eos.enthalpy_j_kg(face_t)
    return ChannelHydraulicSolution(
        axial_edges_m=z.copy(),channel_diameter_m=D,
        mass_flow_kg_s=flow,pressure_faces_pa=faces,
        temperature_faces_k=face_t,density_faces_kg_m3=density,
        velocity_faces_m_s=velocity,mach_faces=mach,
        reynolds_segments=re,darcy_friction_segments=darcy,
        friction_loss_pa=loss,momentum_residual_pa=residual,
        channel_gas_mass_kg=inventory,
        flow_enthalpy_faces_w=flow_enthalpy,
    )
