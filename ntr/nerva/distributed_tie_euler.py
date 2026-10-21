"""Conservative one-dimensional H2 tie supply/return Euler finite volumes.

The supply + physical turnaround + return are represented as ONE
hydraulic path with 2*Nz rigid fluid cells in route order. Every face
uses a single Rusanov approximate Riemann flux for all three conserved
quantities, (mass, axial momentum, TOTAL energy):

    U = [rho, rho*v, rho*(u + v*v/2)]
    F = [rho*v, rho*v*v + p, rho*v*(h + v*v/2)]
    F* = (F_L + F_R)/2 - a_max*(U_R-U_L)/2 .

CoolProp HEOS::Hydrogen gives p(rho,u), speed of sound and vapor quality.
Each rigid cell is advanced conservatively using face fluxes, with an
optional Darcy wall-friction momentum sink and an explicit shared tie
solid energy balance. Wall friction converts kinetic into internal
energy because cell TOTAL energy is not removed by mechanical drag.

The boundary ghost cells are prescribed stationary thermodynamic
reservoirs. This is a numerical screening first-order homogeneous-
equilibrium Euler model, NOT a NASA-calibrated two-phase channel:
no two-fluid slip, non-equilibrium flashing, boiling HTC/CHF, 3D
turnaround force, characteristic boundary matching or real engine
shaft/neutronic feedback. The Rusanov solver adds numerical viscosity
and the fixed channel area ignores detailed tie-tube geometry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .isochoric_tie import IsochoricTieParameters
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import ResolvedHeating


@dataclass(frozen=True)
class DistributedTieParameters:
    thermal: IsochoricTieParameters
    flow_area_m2: float
    darcy_friction_factor: float = 0.0
    turnaround_loss_coefficient: float = 0.0
    max_cfl: float = 0.45
    maximum_velocity_m_s: float = 1500.0

    def validate(self) -> None:
        self.thermal.validate()
        if not math.isfinite(self.flow_area_m2) or self.flow_area_m2<=0:
            raise ValueError("tie effective flow area must be positive")
        if (not math.isfinite(self.darcy_friction_factor)
                or not 0<=self.darcy_friction_factor<=10):
            raise ValueError("invalid tie Darcy friction factor")
        if (not math.isfinite(self.turnaround_loss_coefficient)
                or not 0<=self.turnaround_loss_coefficient<=50):
            raise ValueError("invalid tie turnaround loss coefficient")
        if not math.isfinite(self.max_cfl) or not 0<self.max_cfl<=0.9:
            raise ValueError("invalid Euler CFL ceiling")
        if (not math.isfinite(self.maximum_velocity_m_s)
                or self.maximum_velocity_m_s<=0):
            raise ValueError("invalid Euler velocity ceiling")


@dataclass(frozen=True)
class DistributedTieState:
    solid_temperature_k: np.ndarray  # [tie, physical Nz]
    mass_kg: np.ndarray              # [tie, 2*Nz] in routed flow order
    momentum_kg_m_s: np.ndarray      # [tie, 2*Nz]
    total_energy_j: np.ndarray       # [tie, 2*Nz] INCLUDING kinetic energy

    def validate(self, heating: ResolvedHeating) -> None:
        n_tie,nz=heating.tie_solid_w.shape
        for key,expected in (
            ("solid_temperature_k",(n_tie,nz)),
            ("mass_kg",(n_tie,2*nz)),
            ("momentum_kg_m_s",(n_tie,2*nz)),
            ("total_energy_j",(n_tie,2*nz)),
        ):
            a=np.asarray(getattr(self,key))
            if a.shape!=expected or not np.all(np.isfinite(a)):
                raise ValueError(f"invalid distributed tie {key}")
        if np.any(self.mass_kg<=0) or np.any(self.solid_temperature_k<=0):
            raise ValueError("distributed tie initial masses and solids must be positive")


@dataclass(frozen=True)
class DistributedTieStep:
    state: DistributedTieState
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    density_kg_m3: np.ndarray
    velocity_m_s: np.ndarray
    quality: np.ndarray
    internal_energy_j_kg: np.ndarray
    specific_enthalpy_j_kg: np.ndarray
    interface_mass_flux_kg_s: np.ndarray       # [tie,2*Nz+1]
    interface_momentum_flux_n: np.ndarray
    interface_total_energy_flux_w: np.ndarray
    wall_friction_force_n: np.ndarray
    solid_storage_rate_w: float
    fluid_internal_energy_storage_rate_w: float
    fluid_kinetic_energy_storage_rate_w: float
    nuclear_deposition_w: float
    inlet_total_energy_power_w: float
    outlet_total_energy_power_w: float
    mass_residual_kg_s: float
    momentum_residual_n: float
    total_energy_residual_j: float
    maximum_local_mass_residual_kg_s: float
    maximum_local_momentum_residual_n: float
    maximum_local_total_energy_residual_w: float
    maximum_cfl: float
    numerical_flux: str = "local Lax-Friedrichs / Rusanov, one shared flux per interface"
    momentum_transport_distributed: bool = True
    total_energy_including_kinetic_conserved: bool = True
    homogeneous_equilibrium_only: bool = True


def _state_properties(
    eos: HydrogenEOS, rho: float, u: float, velocity: float,
) -> tuple[float,float,float,float,float,float]:
    if not (math.isfinite(rho) and rho>0 and math.isfinite(u)
            and math.isfinite(velocity)):
        raise ValueError("invalid compressible hydrogen state")
    p=eos.get("P","Dmass",rho,"Umass",u)
    t=eos.get("T","Dmass",rho,"Umass",u)
    quality=eos.get("Q","Dmass",rho,"Umass",u)
    if 0.<quality<1.:
        # CoolProp intentionally leaves two-phase sound speed
        # undefined because it depends on phase dynamics. For THIS
        # explicitly homogeneous-equilibrium Euler model use the
        # isentropic EQUILIBRIUM derivative c^2=(dP/dRho)|s.
        # This is not a slip, frozen-phase or measured sound speed.
        entropy=eos.get("Smass","Dmass",rho,"Umass",u)
        c=None
        for fractional in (1e-4,5e-4,2e-3):
            dr=fractional*rho
            try:
                pplus=eos.get("P","Dmass",rho+dr,"Smass",entropy)
                pminus=eos.get("P","Dmass",rho-dr,"Smass",entropy)
            except (RuntimeError,ValueError):
                continue
            csq=(pplus-pminus)/(2.*dr)
            if math.isfinite(csq) and csq>0:
                c=math.sqrt(csq)
                break
        if c is None:
            raise ValueError(
                "two-phase H2 equilibrium sound speed cannot be "
                "constructed from positive isentropic EOS derivative"
            )
    else:
        c=eos.get("A","Dmass",rho,"Umass",u)
    if not (math.isfinite(p) and math.isfinite(t) and math.isfinite(c)
            and p>0 and t>0 and c>0):
        raise ValueError("invalid H2 thermodynamic/sound-speed state")
    return p,t,c,quality,u+p/rho,u+0.5*velocity*velocity


def _reservoir(
    eos: HydrogenEOS, pressure: float, h: float,
) -> tuple[float,float,float,float,float,float,float]:
    if not math.isfinite(pressure) or pressure<=0 or not math.isfinite(h):
        raise ValueError("invalid H2 reservoir")
    rho=eos.hp("Dmass",h,pressure)
    u=eos.hp("Umass",h,pressure)
    p,t,c,q,_,_= _state_properties(eos,rho,u,0.)
    # Preserve the specified reservoir pressure in the Riemann flux,
    # rather than a slightly rounded EOS inversion of (rho,u).
    return rho,0.,u,pressure,t,c,q


def initial_distributed_tie_state(
    heating: ResolvedHeating, params: DistributedTieParameters,
    initial_pressure_pa: float, initial_specific_enthalpy_j_kg: float,
    *, solid_temperature_k: float, velocity_m_s: float = 0.,
    eos: HydrogenEOS | None = None,
) -> DistributedTieState:
    """Construct real-fluid mass, MOMENTUM and kinetic-inclusive energy."""
    heating.validate()
    params.validate()
    if not math.isfinite(solid_temperature_k) or solid_temperature_k<=0:
        raise ValueError("invalid initial tie solid temperature")
    if (not math.isfinite(velocity_m_s)
            or abs(velocity_m_s)>params.maximum_velocity_m_s):
        raise ValueError("invalid initial tie fluid velocity")
    eos=eos or HydrogenEOS()
    rho=eos.hp("Dmass",initial_specific_enthalpy_j_kg,initial_pressure_pa)
    u=eos.hp("Umass",initial_specific_enthalpy_j_kg,initial_pressure_pa)
    nt,nz=heating.tie_solid_w.shape
    vs=params.thermal.supply_cell_volume_m3
    vr=params.thermal.return_cell_volume_m3
    volume=np.array([vs]*nz+[vr]*nz)
    masses=np.tile(rho*volume,(nt,1))
    return DistributedTieState(
        solid_temperature_k=np.full((nt,nz),solid_temperature_k),
        mass_kg=masses,
        momentum_kg_m_s=masses*velocity_m_s,
        total_energy_j=masses*(u+0.5*velocity_m_s**2),
    )


def advance_distributed_tie(
    heating: ResolvedHeating,
    old: DistributedTieState,
    params: DistributedTieParameters,
    inlet_pressure_pa: float,
    inlet_enthalpy_j_kg: float,
    outlet_pressure_pa: float,
    outlet_backflow_enthalpy_j_kg: float,
    dt_s: float,
    *, eos: HydrogenEOS | None = None,
) -> DistributedTieStep:
    """One first-order conservative Euler+solid step with strict CFL guard."""
    heating.validate()
    old.validate(heating)
    params.validate()
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("Euler timestep must be positive")
    eos=eos or HydrogenEOS()
    nt,nz=heating.tie_solid_w.shape
    nc=2*nz
    area=params.flow_area_m2
    vol=np.asarray(
        [params.thermal.supply_cell_volume_m3]*nz+
        [params.thermal.return_cell_volume_m3]*nz,dtype=float
    )
    dx=vol/area
    diameter=math.sqrt(4.*area/math.pi)
    rho=old.mass_kg/vol[None,:]
    vel=old.momentum_kg_m_s/old.mass_kg
    internal=old.total_energy_j/old.mass_kg-0.5*vel**2
    if np.any(abs(vel)>params.maximum_velocity_m_s):
        raise ValueError("initial velocity outside model range")
    if not np.all(np.isfinite(internal)):
        raise ValueError("nonfinite H2 internal energy")
    p=np.empty((nt,nc));t=np.empty((nt,nc))
    c=np.empty((nt,nc));q=np.empty((nt,nc))
    h=np.empty((nt,nc))
    for j,i in np.ndindex(rho.shape):
        p[j,i],t[j,i],c[j,i],q[j,i],h[j,i],_= _state_properties(
            eos,float(rho[j,i]),float(internal[j,i]),float(vel[j,i])
        )
    th=params.thermal
    if (np.any((p<th.min_pressure_pa)|(p>th.max_pressure_pa))
            or np.any((t<th.min_temperature_k)|(t>th.max_temperature_k))):
        raise ValueError("old H2 field outside declared EOS validity")

    # Old-state thermal exchange: supply and physical-return slices
    # address the SAME solid at a given z, despite reverse path order.
    q_s=th.supply_ua_w_k*(old.solid_temperature_k-t[:,:nz])
    q_r=th.return_ua_w_k*(old.solid_temperature_k-t[:,nz:][:,::-1])
    wall=np.concatenate((q_s,q_r[:,::-1]),axis=1)
    direct=np.concatenate((
        heating.tie_supply_h2_w,heating.tie_return_h2_w[:,::-1]
    ),axis=1)
    fluid_heat=wall+direct
    solid_heat=heating.tie_solid_w-q_s-q_r

    left_res=_reservoir(eos,inlet_pressure_pa,inlet_enthalpy_j_kg)
    right_res=_reservoir(eos,outlet_pressure_pa,outlet_backflow_enthalpy_j_kg)
    face=np.empty((nt,nc+1,3))
    max_cfl=0.
    for j in range(nt):
        # FV conservative state and physical Euler flux per cell.
        ucv=np.stack(
            (rho[j],rho[j]*vel[j],
             old.total_energy_j[j]/vol),axis=1
        )
        physical=np.stack(
            (rho[j]*vel[j],
             rho[j]*vel[j]**2+p[j],
             rho[j]*vel[j]*(h[j]+0.5*vel[j]**2)),
            axis=1,
        )
        for i in range(nc+1):
            if i==0:
                rl,vl,ul,pl,tl,cl,ql=left_res
                L=np.array((rl,rl*vl,rl*(ul+0.5*vl*vl)))
                FL=np.array((rl*vl,rl*vl*vl+pl,
                             rl*vl*(ul+pl/rl+0.5*vl*vl)))
                cr=abs(float(vel[j,0]))+float(c[j,0])
                amax=max(abs(vl)+cl,cr)
                R=ucv[0];FR=physical[0]
            elif i==nc:
                rr,vr,ur,pr,tr,cr,qr=right_res
                R=np.array((rr,rr*vr,rr*(ur+0.5*vr*vr)))
                FR=np.array((rr*vr,rr*vr*vr+pr,
                             rr*vr*(ur+pr/rr+0.5*vr*vr)))
                amax=max(abs(vr)+cr,
                         abs(float(vel[j,-1]))+float(c[j,-1]))
                L=ucv[-1];FL=physical[-1]
            else:
                L=ucv[i-1];R=ucv[i]
                FL=physical[i-1];FR=physical[i]
                amax=max(abs(float(vel[j,i-1]))+float(c[j,i-1]),
                         abs(float(vel[j,i]))+float(c[j,i]))
            face[j,i]=area*(0.5*(FL+FR)-0.5*amax*(R-L))
        max_cfl=max(max_cfl,float(np.max(
            dt_s*(abs(vel[j])+c[j])/dx
        )))
        # Boundary-face waves apply to adjacent real control volumes.
        max_cfl=max(
            max_cfl,
            dt_s*(abs(left_res[1])+left_res[5])/dx[0],
            dt_s*(abs(right_res[1])+right_res[5])/dx[-1],
        )
    if max_cfl>params.max_cfl:
        raise ValueError(
            f"compressible tie Euler acoustic CFL exceeded: {max_cfl:.6g}"
        )

    # Distributed viscous wall force, resolved in each routed cell.
    friction=0.5*params.darcy_friction_factor*(
        dx[None,:]/diameter
    )*rho*vel*abs(vel)*area
    if params.turnaround_loss_coefficient:
        # Half the total elbow loss at each of two neighboring
        # cells. All mechanical dissipation remains in TOTAL energy
        # and converts to internal heat as momentum decreases.
        for index in (nz-1,nz):
            friction[:,index]+=0.25*(
                params.turnaround_loss_coefficient*
                rho[:,index]*vel[:,index]*abs(vel[:,index])*area
            )
    old_flux=face[:,:-1,:]-face[:,1:,:]
    old_conserved=np.stack((
        old.mass_kg,old.momentum_kg_m_s,old.total_energy_j
    ),axis=2)
    new_conserved=old_conserved+dt_s*old_flux
    new_conserved[:,:,1]-=dt_s*friction
    new_conserved[:,:,2]+=dt_s*fluid_heat
    mnew=new_conserved[:,:,0]
    momnew=new_conserved[:,:,1]
    etnew=new_conserved[:,:,2]
    if np.any(mnew<=0) or not np.all(np.isfinite(new_conserved)):
        raise ValueError("compressible tie Euler update depleted mass")
    vnew=momnew/mnew
    enew=etnew/mnew-0.5*vnew**2
    if not np.all(np.isfinite(enew)):
        raise ValueError("compressible tie new internal energy nonfinite")
    solid_new=old.solid_temperature_k+dt_s*(
        solid_heat/th.solid_capacity_j_k
    )
    if np.any(solid_new<=0):
        raise ValueError("nonphysical tie solid temperature")

    pn=np.empty((nt,nc));tn=np.empty((nt,nc))
    qn=np.empty((nt,nc));hn=np.empty((nt,nc))
    for j,i in np.ndindex(mnew.shape):
        pn[j,i],tn[j,i],_,qn[j,i],hn[j,i],_= _state_properties(
            eos,float(mnew[j,i]/vol[i]),
            float(enew[j,i]),float(vnew[j,i])
        )
    if (np.any((pn<th.min_pressure_pa)|(pn>th.max_pressure_pa))
            or np.any((tn<th.min_temperature_k)|(tn>th.max_temperature_k))
            or np.any(abs(vnew)>params.maximum_velocity_m_s)):
        raise ValueError("new H2 EOS state exceeds declared domain")

    rates=(new_conserved-old_conserved)/dt_s
    lm=rates[:,:,0]-old_flux[:,:,0]
    lp=rates[:,:,1]-old_flux[:,:,1]+friction
    le=rates[:,:,2]-old_flux[:,:,2]-fluid_heat
    ls=(solid_new-old.solid_temperature_k)*(
        th.solid_capacity_j_k/dt_s
    )-solid_heat
    mtotal=(float(np.sum(mnew-old.mass_kg))/dt_s-
            float(np.sum(face[:,0,0]-face[:,-1,0])))
    ptotal=(float(np.sum(momnew-old.momentum_kg_m_s))/dt_s-
            float(np.sum(face[:,0,1]-face[:,-1,1]))+
            float(np.sum(friction)))
    sstore=float(np.sum(
        th.solid_capacity_j_k*(solid_new-old.solid_temperature_k)
    )/dt_s)
    kinold=0.5*old.momentum_kg_m_s**2/old.mass_kg
    kinnew=0.5*momnew**2/mnew
    kstore=float(np.sum(kinnew-kinold)/dt_s)
    internal_store=float(np.sum(
        (etnew-kinnew)-(old.total_energy_j-kinold)
    )/dt_s)
    nuclear=float(np.sum(heating.tie_solid_w)+
                  np.sum(heating.tie_supply_h2_w)+
                  np.sum(heating.tie_return_h2_w))
    inpower=float(np.sum(face[:,0,2]))
    outpower=float(np.sum(face[:,-1,2]))
    erest=dt_s*(
        sstore+internal_store+kstore-nuclear-inpower+outpower
    )
    scale=max(1.,abs(sstore),abs(internal_store),abs(kstore),
              abs(nuclear),abs(inpower),abs(outpower))
    if (abs(mtotal)>1e-9+
            1e-9*max(abs(float(np.sum(face[:,0,0]))),
                     abs(float(np.sum(face[:,-1,0]))))
            or abs(ptotal)>1e-7+1e-9*max(
                abs(float(np.sum(face[:,0,1]))),
                abs(float(np.sum(face[:,-1,1])))
            )
            or abs(erest)>1e-7+1e-8*dt_s*scale
            or np.max(abs(lm))>1e-8
            or np.max(abs(lp))>1e-5
            or max(float(np.max(abs(le))),float(np.max(abs(ls))))
                   >1e-7+1e-8*scale):
        raise RuntimeError("compressible tie conservative Euler residual failed")
    return DistributedTieStep(
        state=DistributedTieState(
            solid_temperature_k=solid_new,
            mass_kg=mnew,momentum_kg_m_s=momnew,total_energy_j=etnew,
        ),
        pressure_pa=pn,temperature_k=tn,density_kg_m3=mnew/vol[None,:],
        velocity_m_s=vnew,quality=qn,internal_energy_j_kg=enew,
        specific_enthalpy_j_kg=hn,
        interface_mass_flux_kg_s=face[:,:,0],
        interface_momentum_flux_n=face[:,:,1],
        interface_total_energy_flux_w=face[:,:,2],
        wall_friction_force_n=friction,
        solid_storage_rate_w=sstore,
        fluid_internal_energy_storage_rate_w=internal_store,
        fluid_kinetic_energy_storage_rate_w=kstore,
        nuclear_deposition_w=nuclear,
        inlet_total_energy_power_w=inpower,
        outlet_total_energy_power_w=outpower,
        mass_residual_kg_s=mtotal,momentum_residual_n=ptotal,
        total_energy_residual_j=erest,
        maximum_local_mass_residual_kg_s=float(np.max(abs(lm))),
        maximum_local_momentum_residual_n=float(np.max(abs(lp))),
        maximum_local_total_energy_residual_w=max(
            float(np.max(abs(le))),float(np.max(abs(ls)))
        ),
        maximum_cfl=max_cfl,
    )
