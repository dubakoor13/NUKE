"""Implicit resolved fuel conduction with molecular-H2 enthalpy transport.

Reuses the Stage-19 conservative cylindrical fuel-ring/axial conduction
operator, but REPLACES the constant-cp coolant balance with the actual
NUKE implicit_hydrogen.hydrogen_specific_enthalpy(T) and NASA-7 frozen
molecular H2 cp(T). Hydrogen mass per cell stays fixed; mass flow
is externally specified and pressure/density are NOT solved.

Energy equations, end-of-step implicit in ALL unknowns:

 Cs_i*(Ts_i-Ts_i_old)/dt =
        Qs_i + sum_internal G_ij*(Ts_j-Ts_i) - UA_i*(Ts_i-Th_i)
 m_i*(h(Th_i)-h(Th_i_old))/dt =
        Qh_i + UA_i*(Ts_inner_i-Th_i)
        + mdot*(h(Tup_i)-h(Th_i))

Tup is the inlet for the first axial segment, else the PRECEDING segment
at NEW time. The exact global ledger is

 Delta Usolid + sum(m*(hnew-hold)) +
    dt*sum(mdot*(h(Tout)-h(Tin))) + dt*Qother = dt*Qtotal.

For pure frozen NASA-7 H2, h(T) and cp(T)=dh/dT come from
NUKE's existing nasa_hydrogen.py, valid only 200-3500 K. It is
NOT equilibrium H2/H dissociation, cryogenic real-fluid H2, channel
pressure/momentum dynamics, temperature-dependent fuel k/cp, or
qualified actual NERVA nuclear energy deposition. Geometry, mass
flows, wall UA, solid properties and heat loads are caller supplied.
The optional 'constant' mode must algebraically reduce to Stage 19.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from .fx2th_conjugate_fuel_transport import (
    ConjugateCoreGeometry, ConjugateCoreState, ConjugateCoreDeposition,
    assemble_conjugate_core,
)
from .implicit_hydrogen import hydrogen_specific_enthalpy
from .nasa_hydrogen import T_MIN,T_MAX,hydrogen_state


@dataclass(frozen=True)
class EnthalpyCoreControls:
    enthalpy_model: str = "nasa7-h2"
    max_iterations: int = 24
    residual_tolerance_w: float = 2e-8
    min_line_search_fraction: float = 1./4096
    max_unknowns: int = 2000

    def __post_init__(self):
        if self.enthalpy_model not in ("nasa7-h2", "constant"):
            raise ValueError("enthalpy_model must be nasa7-h2 or constant")
        if (not isinstance(self.max_iterations,int) or
            self.max_iterations<1 or
            not isinstance(self.max_unknowns,int) or self.max_unknowns<1):
            raise ValueError("positive Newton and system budgets required")
        if (not math.isfinite(self.residual_tolerance_w) or
            not 0<self.residual_tolerance_w<=1e-3 or
            not math.isfinite(self.min_line_search_fraction) or
            not 0<self.min_line_search_fraction<=1):
            raise ValueError("invalid nonlinear tolerance/line-search bounds")


@dataclass(frozen=True)
class EnthalpyCoreStep:
    state: ConjugateCoreState
    fuel_storage_j: float
    hydrogen_enthalpy_storage_j: float
    outlet_net_enthalpy_j: float
    other_deposition_j: float
    deposited_total_j: float
    wall_transfer_j: float
    thermal_balance_residual_j: float
    thermal_relative_residual: float
    maximum_equation_residual_w: float
    nonlinear_iterations: int
    source_id: str
    synthetic: bool
    enthalpy_model: str


class _EnthalpySystem:
    def __init__(self,geo,old,power,dt,control):
        self.geo=geo
        self.old=old
        self.power=power
        self.dt=dt
        self.control=control
        self.base=assemble_conjugate_core(geo,old,power,dt)
        p,z,r=geo.shape
        self.p,self.z,self.r=p,z,r
        self.ns=p*z*r
        self.size=self.ns+p*z
        self.mass=np.asarray(geo.hydrogen_mass_kg,dtype=float)
        self.flow=np.asarray(geo.hydrogen_mass_flow_kg_s,dtype=float)
        self.cp_ref=np.asarray(geo.hydrogen_cp_j_kg_k,dtype=float)
        self.inlet=np.asarray(geo.inlet_temperature_k,dtype=float)
        self.ua=np.asarray(geo.inner_wall_conductance_w_k,dtype=float)
        self.qh=np.asarray(power.direct_hydrogen_w,dtype=float)
        self.hold=np.empty((p,z))
        self.hin=np.empty(p)
        for ip in range(p):
            self.hin[ip]=self.h(self.inlet[ip],ip)
            for iz in range(z):
                self.hold[ip,iz]=self.h(old.hydrogen_temperature_k[ip,iz],ip)

    def h(self,T,ip):
        return hydrogen_specific_enthalpy(float(T),
            float(self.cp_ref[ip]),self.control.enthalpy_model)

    def cp(self,T,ip):
        if self.control.enthalpy_model=="constant":
            return self.cp_ref[ip]
        return hydrogen_state(float(T),101325.,mode="frozen").frozen_cp_j_kg_k

    def valid(self,x):
        if (np.shape(x)!=(self.size,) or
            np.any(~np.isfinite(x)) or np.any(x<=0)):
            return False
        if self.control.enthalpy_model=="nasa7-h2":
            t=x[self.ns:]
            return bool(np.all(t>=T_MIN) and np.all(t<=T_MAX))
        return True

    def equation(self,x,*,jacobian=False):
        if not self.valid(x):
            raise ValueError("nonphysical or out-of-range hydrogen trial")
        # The Stage19 solid rows are EXACT same conservative Fourier+wall
        # equations: only the gas storage/advection constitutive law changes.
        residual=np.asarray(self.base.matrix@x-self.base.rhs)
        J=self.base.matrix.tolil() if jacobian else None
        fuel=x[:self.ns].reshape((self.p,self.z,self.r))
        gas=x[self.ns:].reshape((self.p,self.z))
        for ip in range(self.p):
            for iz in range(self.z):
                idx=self.ns+ip*self.z+iz
                T=float(gas[ip,iz])
                h=self.h(T,ip)
                hup=(self.hin[ip] if iz==0 else self.h(gas[ip,iz-1],ip))
                R=self.mass[ip,iz]/self.dt*(h-self.hold[ip,iz])
                R+=self.flow[ip]*(h-hup)
                R-=self.qh[ip,iz]+self.ua[ip,iz]*(fuel[ip,iz,0]-T)
                residual[idx]=R
                if jacobian:
                    J[idx,idx]=(self.mass[ip,iz]/self.dt+self.flow[ip])*(
                         self.cp(T,ip))+self.ua[ip,iz]
                    if iz>0:
                        J[idx,idx-1]=-self.flow[ip]*self.cp(gas[ip,iz-1],ip)
                    # J[idx,first-solid] is the existing -UA from Stage19.
        return residual, (J.tocsc() if jacobian else None)


def advance_enthalpy_conjugate_core(
    geo:ConjugateCoreGeometry,
    old:ConjugateCoreState,
    power:ConjugateCoreDeposition,
    dt_s:float,
    *,
    controls:EnthalpyCoreControls | None=None,
)->EnthalpyCoreStep:
    """Fully coupled implicit fuel Fourier conduction and nonlinear H2 h(T).

    Sparse Newton uses the TRUE NASA-7 cp=dh/dT derivative, not an
    arbitrary constant cp or a finite-difference material fit.
    No iteration result is accepted without a local equation residual
    and independent global deposited-energy/enthalpy balance.
    """
    cfg=EnthalpyCoreControls() if controls is None else controls
    if not isinstance(cfg,EnthalpyCoreControls):
        raise TypeError("controls must be EnthalpyCoreControls")
    problem=_EnthalpySystem(geo,old,power,dt_s,cfg)
    if problem.size>cfg.max_unknowns:
        raise ValueError("enthalpy core exceeds maximum allowed unknowns")
    x=np.r_[np.asarray(old.fuel_temperature_k,dtype=float).ravel(),
            np.asarray(old.hydrogen_temperature_k,dtype=float).ravel()]
    if not problem.valid(x):
        raise ValueError("initial hydrogen temperatures outside enthalpy model domain")
    nonlinear_iterations=0
    for iteration in range(cfg.max_iterations+1):
        R,J=problem.equation(x,jacobian=True)
        err=float(np.max(np.abs(R)))
        if err<=cfg.residual_tolerance_w:
            nonlinear_iterations=iteration
            break
        if iteration==cfg.max_iterations:
            raise RuntimeError("nonlinear enthalpy fuel/H2 equations did not converge")
        with warnings.catch_warnings():
            warnings.simplefilter("error",MatrixRankWarning)
            try:
                delta=spsolve(J,-R)
            except (MatrixRankWarning,ValueError,RuntimeError) as exc:
                raise RuntimeError("singular/nonfinite enthalpy Newton Jacobian") from exc
        if not np.all(np.isfinite(delta)):
            raise ArithmeticError("nonfinite fuel/H2 Newton direction")
        alpha=1.
        accepted=False
        while alpha>=cfg.min_line_search_fraction:
            proposal=x+alpha*delta
            if problem.valid(proposal):
                candidate,_=problem.equation(proposal)
                if np.max(np.abs(candidate)) < err*(1.-1e-4*alpha):
                    x=proposal
                    accepted=True
                    break
            alpha/=2.
        if not accepted:
            raise RuntimeError("enthalpy fuel/H2 line search rejected all valid states")
    else:
        raise RuntimeError("enthalpy core unexpectedly left Newton without convergence")

    solid=x[:problem.ns].reshape(geo.shape)
    gas=x[problem.ns:].reshape((problem.p,problem.z))
    result=ConjugateCoreState(old.time_s+dt_s,solid,gas)
    dsolid=float(np.sum(problem.base.solid_storage_j_k*(
             solid-np.asarray(old.fuel_temperature_k))))
    dhydrogen=0.
    outlet=0.
    for ip in range(problem.p):
        for iz in range(problem.z):
            dhydrogen+=problem.mass[ip,iz]*(problem.h(gas[ip,iz],ip)-
                                             problem.hold[ip,iz])
        outlet+=dt_s*problem.flow[ip]*(
            problem.h(gas[ip,-1],ip)-problem.hin[ip])
    deposited=dt_s*power.declared_total_w
    other=dt_s*float(np.sum(power.other_w))
    residual=dsolid+dhydrogen+outlet+other-deposited
    scale=max(1.,abs(dsolid),abs(dhydrogen),abs(outlet),abs(other),abs(deposited))
    relative=abs(residual)/scale
    if relative>1e-9:
        raise ArithmeticError("nonlinear H2 enthalpy first-law audit failed")
    wall=dt_s*float(np.sum(problem.ua*(solid[:,:,0]-gas)))
    return EnthalpyCoreStep(
        state=result,
        fuel_storage_j=dsolid,
        hydrogen_enthalpy_storage_j=dhydrogen,
        outlet_net_enthalpy_j=outlet,
        other_deposition_j=other,
        deposited_total_j=deposited,
        wall_transfer_j=wall,
        thermal_balance_residual_j=residual,
        thermal_relative_residual=relative,
        maximum_equation_residual_w=err,
        nonlinear_iterations=nonlinear_iterations,
        source_id=power.source_id,
        synthetic=bool(power.synthetic),
        enthalpy_model=cfg.enthalpy_model,
    )
