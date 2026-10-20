"""Phase-aware real-fluid tie-tube counterflow energy model (CoolProp H2).

Every axial cell contains one shared tie solid and two independent fluid
enthalpy inventories. The supply flows +z, turns around, and the return
flows -z. The backward-Euler balances are

 Cs*(Ts1-Ts0)/dt = Q_solid - UA_s*(Ts1-T_s1) - UA_r*(Ts1-T_r1)
 Ms*(hs1-hs0)/dt = mdot*(hup_s1-hs1) + UA_s*(Ts1-T_s1) + Qdirect_s
 Mr*(hr1-hr0)/dt = mdot*(hup_r1-hr1) + UA_r*(Ts1-T_r1) + Qdirect_r.

T(h,p), density and equilibrium vapor quality are obtained directly from
CoolProp's hydrogen equation of state, including saturation. Unlike a
temperature-only gas solver, the latent heat is in stored and advected h.

This is a SOURCE/ENERGY step, not a NERVA-qualified full tie model:
 - Axial pressure is fixed and equal in both ducts (NO momentum closure);
 - prescribed, constant H2 inventory per cell represents a fixed-mass
   Lagrangian/mixed cell, not a rigid constant-volume pressure solve;
 - prescribed UA/Cs (no boiling heat transfer/CHF, slip or chilldown);
 - positive nuclear deposition, prescribed mass flow, no neutronic feedback.
Solid/coolant first-law residuals are independently audited. No OpenMC is
launched, and this does NOT establish an operable LH2 feed-loop solution.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import ResolvedHeating


@dataclass(frozen=True)
class CryogenicTieParameters:
    pressure_pa: float
    solid_capacity_j_k: float
    supply_inventory_kg: float
    return_inventory_kg: float
    supply_ua_w_k: float
    return_ua_w_k: float
    min_temperature_k: float = 16.
    max_temperature_k: float = 850.
    coupling_tolerance_k: float = 1e-8
    max_iterations: int = 80

    def validate(self) -> None:
        for name in ("pressure_pa", "solid_capacity_j_k",
                     "supply_inventory_kg", "return_inventory_kg"):
            v=getattr(self,name)
            if not math.isfinite(v) or v <= 0:
                raise ValueError(f"{name} must be positive finite")
        for name in ("supply_ua_w_k", "return_ua_w_k"):
            v=getattr(self,name)
            if not math.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be nonnegative finite")
        if (not math.isfinite(self.min_temperature_k) or
                not math.isfinite(self.max_temperature_k) or
                not 14.1 < self.min_temperature_k < self.max_temperature_k <= 900.):
            raise ValueError("unsupported real-fluid hydrogen temperature bracket")
        if (not isinstance(self.max_iterations,int)
                or not 2 <= self.max_iterations <= 500
                or not math.isfinite(self.coupling_tolerance_k)
                or self.coupling_tolerance_k <= 0):
            raise ValueError("invalid shared-solid Picard convergence settings")


@dataclass(frozen=True)
class CryogenicTieState:
    solid_temperature_k: np.ndarray      # [tie, z]
    supply_enthalpy_j_kg: np.ndarray      # [tie, z]
    return_enthalpy_j_kg: np.ndarray      # [tie, z]

    def validate(self, heating: ResolvedHeating, lo: float, hi: float) -> None:
        for label, value in (
            ("solid temperature",self.solid_temperature_k),
            ("supply enthalpy",self.supply_enthalpy_j_kg),
            ("return enthalpy",self.return_enthalpy_j_kg),
        ):
            if (value.shape != heating.tie_solid_w.shape
                    or not np.all(np.isfinite(value))):
                raise ValueError(f"invalid cryogenic tie {label} field")
        if np.any(self.solid_temperature_k<=0):
            raise ValueError("tie solid temperature must be positive")
        for name,values in (
            ("supply",self.supply_enthalpy_j_kg),
            ("return",self.return_enthalpy_j_kg),
        ):
            if np.any(values<lo) or np.any(values>hi):
                raise ValueError(f"{name} H2 initial enthalpy outside bounded EOS range")


@dataclass(frozen=True)
class CryogenicTieStep:
    state: CryogenicTieState
    supply_temperature_k: np.ndarray
    return_temperature_k: np.ndarray
    supply_quality: np.ndarray
    return_quality: np.ndarray
    supply_density_kg_m3: np.ndarray
    return_density_kg_m3: np.ndarray
    supply_wall_power_w: np.ndarray
    return_wall_power_w: np.ndarray
    solid_storage_rate_w: float
    hydrogen_storage_rate_w: float
    nuclear_deposition_w: float
    inlet_enthalpy_power_w: float
    outlet_enthalpy_power_w: float
    energy_residual_j: float
    max_local_solid_residual_w: float
    max_local_hydrogen_residual_w: float
    max_picard_iterations: int


def _cell_root(
    eos: HydrogenEOS, pressure: float, h_min: float, h_max: float,
    old_h: float, upstream_h: float, solid_t: float,
    inventory: float, mass_flow: float, ua: float,
    q_direct: float, dt: float,
) -> float:
    """Monotone implicit enthalpy root even on the saturation plateau."""
    storage=inventory/dt
    def residual(h: float) -> float:
        temp=eos.hp("T",h,pressure)
        return ((storage+mass_flow)*h
                - storage*old_h - mass_flow*upstream_h
                - ua*(solid_t-temp) - q_direct)
    lo,hi=h_min,h_max
    if residual(lo)>1e-7 or residual(hi)<-1e-7:
        raise ValueError("cryogenic tie coolant enthalpy left bounded EOS domain")
    for _ in range(48):
        mid=(lo+hi)/2.
        if residual(mid)<0:
            lo=mid
        else:
            hi=mid
    return (lo+hi)/2.


def advance_cryogenic_tie(
    heating: ResolvedHeating,
    old: CryogenicTieState,
    params: CryogenicTieParameters,
    inlet_enthalpy_j_kg: float,
    dt_s: float,
    *,
    eos: HydrogenEOS | None = None,
) -> CryogenicTieStep:
    """Advance every tie instance one simultaneous implicit H2/solid step.

    The OpenMC-style direct H2 and tie-solid source arrays are distinct;
    no source is assigned twice. Upstream return enthalpy at z=L equals
    NEW-time supply outlet, preserving counterflow turnaround mixing.
    """
    heating.validate()
    params.validate()
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("invalid real-fluid tie timestep")
    if not math.isfinite(inlet_enthalpy_j_kg):
        raise ValueError("nonfinite cryogenic tie inlet enthalpy")
    eos=eos or HydrogenEOS()
    p=params.pressure_pa
    hmin=eos.tp("Hmass",params.min_temperature_k,p)
    hmax=eos.tp("Hmass",params.max_temperature_k,p)
    if not hmin<hmax:
        raise ValueError("invalid real-fluid enthalpy domain")
    old.validate(heating,hmin,hmax)
    if not hmin<=inlet_enthalpy_j_kg<=hmax:
        raise ValueError("inlet enthalpy outside real-fluid domain")

    shape=heating.tie_solid_w.shape
    ts1=np.empty(shape,dtype=float)
    hs1=np.empty(shape,dtype=float)
    hr1=np.empty(shape,dtype=float)
    qs=np.empty(shape,dtype=float)
    qr=np.empty(shape,dtype=float)
    max_picard=0
    c_s=params.solid_capacity_j_k
    ms=params.supply_inventory_kg
    mr=params.return_inventory_kg
    ua_s=params.supply_ua_w_k
    ua_r=params.return_ua_w_k
    rate=c_s/dt_s

    for j in range(shape[0]):
        flow=float(heating.tie_flow_kg_s[j])
        prev_s=old.solid_temperature_k[j,:]
        prev_hs=old.supply_enthalpy_j_kg[j,:]
        prev_hr=old.return_enthalpy_j_kg[j,:]
        sg=prev_s.copy()

        def fluids(solids: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
            supplied=np.empty(shape[1],dtype=float)
            returned=np.empty(shape[1],dtype=float)
            for i in range(shape[1]):
                up=float(inlet_enthalpy_j_kg if i==0 else supplied[i-1])
                supplied[i]=_cell_root(
                    eos,p,hmin,hmax,float(prev_hs[i]),up,float(solids[i]),
                    ms,flow,ua_s,float(heating.tie_supply_h2_w[j,i]),dt_s,
                )
            for i in range(shape[1]-1,-1,-1):
                up=float(supplied[-1] if i==shape[1]-1 else returned[i+1])
                returned[i]=_cell_root(
                    eos,p,hmin,hmax,float(prev_hr[i]),up,float(solids[i]),
                    mr,flow,ua_r,float(heating.tie_return_h2_w[j,i]),dt_s,
                )
            return supplied,returned

        for k in range(1,params.max_iterations+1):
            a,b=fluids(sg)
            ta=np.array([eos.hp("T",float(v),p) for v in a])
            tb=np.array([eos.hp("T",float(v),p) for v in b])
            target=(
                rate*prev_s+heating.tie_solid_w[j,:]+ua_s*ta+ua_r*tb
            )/(rate+ua_s+ua_r)
            delta=float(np.max(np.abs(target-sg)))
            if delta<=params.coupling_tolerance_k:
                sg=target
                break
            sg=0.5*(sg+target)
        else:
            raise RuntimeError(
                f"tie {j}: real-fluid solid/H2 counterflow failed Picard "
                f"convergence in {params.max_iterations} iterations"
            )
        # Re-solve H2 with converged solid; the residual audit below
        # rejects inconsistent final coupled temperatures/fluxes.
        a,b=fluids(sg)
        ts1[j,:]=sg
        hs1[j,:]=a
        hr1[j,:]=b
        max_picard=max(max_picard,k)

    ta=np.empty(shape)
    tb=np.empty(shape)
    qa=np.empty(shape)
    qb=np.empty(shape)
    rhoa=np.empty(shape)
    rhob=np.empty(shape)
    for j,i in np.ndindex(shape):
        for hfield,temp,quality,rho in (
            (hs1,ta,qa,rhoa),(hr1,tb,qb,rhob),
        ):
            h=float(hfield[j,i])
            temp[j,i]=eos.hp("T",h,p)
            quality[j,i]=eos.hp("Q",h,p)
            rho[j,i]=eos.hp("Dmass",h,p)
            if not (math.isfinite(rho[j,i]) and rho[j,i]>0
                    and math.isfinite(temp[j,i])):
                raise RuntimeError("invalid real-fluid cryogenic tie state")

    qs=ua_s*(ts1-ta)
    qr=ua_r*(ts1-tb)
    solidrate=rate*(ts1-old.solid_temperature_k)
    local_s=heating.tie_solid_w-qs-qr-solidrate
    local_h=[]
    for j,i in np.ndindex(shape):
        flow=float(heating.tie_flow_kg_s[j])
        sup_up=float(inlet_enthalpy_j_kg if i==0 else hs1[j,i-1])
        ret_up=float(hs1[j,-1] if i==shape[1]-1 else hr1[j,i+1])
        local_h.append(
            ms*(hs1[j,i]-old.supply_enthalpy_j_kg[j,i])/dt_s
            - flow*(sup_up-hs1[j,i])
            - qs[j,i] - heating.tie_supply_h2_w[j,i]
        )
        local_h.append(
            mr*(hr1[j,i]-old.return_enthalpy_j_kg[j,i])/dt_s
            - flow*(ret_up-hr1[j,i])
            - qr[j,i] - heating.tie_return_h2_w[j,i]
        )
    solid_residual=float(np.max(np.abs(local_s)))
    gas_residual=float(max(abs(v) for v in local_h))
    local_power_scale=max(
        1.,float(np.max(heating.tie_solid_w)),
        float(np.max(np.abs(solidrate))),float(np.max(np.abs(qs+qr))),
        float(np.max(np.abs(np.asarray(local_h))))
    )
    if max(solid_residual,gas_residual)>1e-7*local_power_scale+1e-5:
        raise RuntimeError("real-fluid tie local energy conservation failure")

    h_storage=(ms*np.sum(hs1-old.supply_enthalpy_j_kg)
               +mr*np.sum(hr1-old.return_enthalpy_j_kg))/dt_s
    solid_storage=float(np.sum(solidrate))
    q_deposition=float(sum(np.sum(field) for field in (
        heating.tie_solid_w,heating.tie_supply_h2_w,
        heating.tie_return_h2_w,
    )))
    incoming=float(np.sum(heating.tie_flow_kg_s)*inlet_enthalpy_j_kg)
    outgoing=float(np.dot(heating.tie_flow_kg_s,hr1[:,0]))
    residual=(solid_storage+h_storage-q_deposition-incoming+outgoing)*dt_s
    scale=max(1.,dt_s*abs(q_deposition),dt_s*abs(solid_storage),
              dt_s*abs(h_storage),dt_s*abs(incoming),dt_s*abs(outgoing))
    if abs(residual)>1e-8*scale+1e-5:
        raise RuntimeError("real-fluid tie global first-law closure failure")
    return CryogenicTieStep(
        state=CryogenicTieState(ts1,hs1,hr1),
        supply_temperature_k=ta,return_temperature_k=tb,
        supply_quality=qa,return_quality=qb,
        supply_density_kg_m3=rhoa,return_density_kg_m3=rhob,
        supply_wall_power_w=qs,return_wall_power_w=qr,
        solid_storage_rate_w=solid_storage,
        hydrogen_storage_rate_w=float(h_storage),
        nuclear_deposition_w=q_deposition,
        inlet_enthalpy_power_w=incoming,
        outlet_enthalpy_power_w=outgoing,
        energy_residual_j=float(residual),
        max_local_solid_residual_w=solid_residual,
        max_local_hydrogen_residual_w=gas_residual,
        max_picard_iterations=max_picard,
    )
