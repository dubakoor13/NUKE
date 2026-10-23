"""Independent IQS-inspired two-rate, spatially evolving neutron diffusion.

Reference: Shober, Daly, Ferguson, ANL-78-97 Chapter II.C (1978).
This is NOT the original FX2-TH FORTRAN implementation and NOT a NERVA
nuclear-transport or reactor-thermal validation.

M d(phi)/dt = H phi + E(C) + Q; H=-A+(1-beta)F
dC_l/dt = beta_l P(phi)-lambda_l C_l
phi=n*s, n=w.T@M@phi, w.T@M@s=1.

Macro shape endpoints are solved from the integral space-time equation,
with linear shape interpolation and implicit amplitude/precursor microsteps.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .fx2th_diffusion import DiffusionGrid, assemble_diffusion
from .fx2th_spacetime import DelayedData, SpaceTimeState, mass_diagonal


@dataclass(frozen=True)
class ShapeMacroStep:
    state: SpaceTimeState
    initial_amplitude: float
    final_amplitude: float
    shape_iterations: int
    shape_normalization_residual: float
    spatial_integral_residual: float
    global_neutron_balance_residual: float
    maximum_precursor_residual: float
    micro_amplitudes: tuple[float, ...]
    micro_states: tuple[SpaceTimeState, ...]


def _validate_inputs(grid, delayed, initial, weight, q, dt, microsteps,
                     max_shape_iterations, tolerance, relaxation):
    m=assemble_diffusion(grid)
    if (initial.flux.shape != m.shape or
            not np.all(np.isfinite(initial.flux)) or np.any(initial.flux<0)):
        raise ValueError('invalid initial flux [ny,nx,g]')
    if (initial.precursors.shape != (*m.shape[:2],len(delayed.beta))
            or not np.all(np.isfinite(initial.precursors))
            or np.any(initial.precursors<0)):
        raise ValueError('invalid initial local precursor inventories')
    if len(delayed.speeds_m_s)!=m.shape[-1]:
        raise ValueError('energy speed count differs from group count')
    if not math.isfinite(initial.time_s) or initial.time_s<0:
        raise ValueError('invalid initial time')
    w=np.asarray(weight,dtype=float)
    if (w.shape!=m.shape or not np.all(np.isfinite(w))
            or np.any(w<0) or not np.any(w>0)):
        raise ValueError('valid nonnegative adjoint weighting required')
    q=np.asarray(q,dtype=float)
    if q.shape!=m.shape or not np.all(np.isfinite(q)) or np.any(q<0):
        raise ValueError('invalid external source [ny,nx,g]')
    if not math.isfinite(dt) or dt<=0:
        raise ValueError('reactivity timestep must be positive finite')
    if not isinstance(microsteps,int) or microsteps<1:
        raise ValueError('microsteps must be positive integer')
    if not isinstance(max_shape_iterations,int) or max_shape_iterations<1:
        raise ValueError('shape iteration limit must be positive integer')
    if not math.isfinite(tolerance) or not 0<tolerance<0.01:
        raise ValueError('invalid tolerance')
    if not math.isfinite(relaxation) or not 0<relaxation<=1:
        raise ValueError('relaxation outside (0,1]')
    return m,w.flatten(),q


def advance_iqs_shape(grid: DiffusionGrid, delayed: DelayedData,
                      initial: SpaceTimeState, dt: float,
                      adjoint_weight: np.ndarray, *, microsteps:int=4,
                      source: np.ndarray|None=None,
                      max_shape_iterations:int=70, tolerance:float=1e-9,
                      relaxation:float=0.8) -> ShapeMacroStep:
    """Solve micro amplitude/precursors and a macro evolving spatial shape.

    Coefficients, boundary conditions, source and mesh are constant across
    this macro interval; a single macro comprises dt*microsteps seconds.
    The original FX2-TH adaptive shape controller is NOT implemented.
    """
    if source is None: source=np.zeros_like(initial.flux)
    m,w,q=_validate_inputs(grid,delayed,initial,adjoint_weight,source,dt,
                          microsteps,max_shape_iterations,tolerance,relaxation)
    vol=m.cell_volume_m3
    M=mass_diagonal(grid,delayed)
    wm=w*M
    n0=float(wm @ np.asarray(initial.flux,dtype=float).flatten())
    if not math.isfinite(n0) or n0<=0:
        raise ValueError('nonpositive initial adjoint-weighted amplitude')
    s0=np.asarray(initial.flux,dtype=float).flatten()/n0
    H=-m.loss+(1-sum(delayed.beta))*m.fission
    chi=np.asarray(grid.fission_spectrum,dtype=float)
    nu=np.asarray(grid.nu_fission_inv_m,dtype=float)
    beta=np.asarray(delayed.beta,dtype=float)
    lam=np.asarray(delayed.decay_s_inv,dtype=float)
    qv=(q*vol[:,:,None]).flatten()
    implicit_coeff=1-sum(beta)+float(np.sum(dt*beta*lam/(1+dt*lam)))
    Heff=-m.loss+implicit_coeff*m.fission

    def emission(precursors):
        if not len(beta): return np.zeros_like(qv)
        return (chi*np.sum(precursors*lam[None,None,:],axis=-1)[:,:,None]).flatten()

    def integrate(endpoint):
        n=n0
        C=np.asarray(initial.precursors,dtype=float).copy()
        N=[]; Cs=[]; shapes=[]; states=[]; precres=0.
        for j in range(1,microsteps+1):
            frac=j/microsteps
            sj=(1-frac)*s0+frac*endpoint
            known=emission(C/(1+dt*lam)[None,None,:]) if len(beta) else np.zeros_like(qv)
            denominator=1-dt*(w@(Heff@sj))
            if not math.isfinite(denominator) or denominator<=0:
                raise ArithmeticError('IQS amplitude denominator nonpositive: reduce macro interval')
            nn=(n+dt*(w@(known+qv)))/denominator
            if not math.isfinite(nn) or nn<=0:
                raise ArithmeticError('nonphysical IQS amplitude')
            next_flux=(nn*sj).reshape(m.shape)
            cell_prod=np.sum(nu*next_flux,axis=-1)*vol
            next_C=(C+dt*cell_prod[:,:,None]*beta[None,None,:])/(1+dt*lam)[None,None,:] if len(beta) else C
            if np.any(next_C<0) or not np.all(np.isfinite(next_C)):
                raise ArithmeticError('nonphysical local precursor inventory')
            if len(beta):
                balance=((next_C-C)/dt-beta[None,None,:]*cell_prod[:,:,None]
                         +lam[None,None,:]*next_C)
                precres=max(precres,float(np.max(np.abs(balance))))
            N.append(nn); Cs.append(next_C.copy()); shapes.append(sj)
            states.append(SpaceTimeState(initial.time_s+j*dt,
                                         next_flux.copy(),next_C.copy()))
            n=nn; C=next_C
        return np.asarray(N),Cs,shapes,states,precres

    s=s0.copy()
    for iteration in range(1,max_shape_iterations+1):
        ns,Cs,shapes,states,precres=integrate(s)
        n_end=float(ns[-1])
        fractions=np.arange(1,microsteps+1,dtype=float)/microsteps
        a=float(dt*np.dot(ns,fractions))
        b=float(dt*np.dot(ns,1-fractions))
        emitted=np.zeros_like(qv)
        for C in Cs: emitted+=dt*emission(C)
        rhs=n0*M*s0+b*(H@s0)+emitted+dt*microsteps*qv
        system=diags(n_end*M,format='csc')-a*H
        new=np.asarray(spsolve(system,rhs),dtype=float)
        if not np.all(np.isfinite(new)) or np.min(new)<-1e-11:
            raise ArithmeticError('IQS shape nonphysical: reduce macro interval')
        new=np.maximum(new,0)
        normalization=float(wm@new)
        if normalization<=0 or not math.isfinite(normalization):
            raise ArithmeticError('IQS normalization invalid')
        normalized=new/normalization
        changed=(np.linalg.norm(normalized-s,ord=np.inf)/
                 max(np.linalg.norm(normalized,ord=np.inf),1e-16))
        s=(1-relaxation)*s+relaxation*normalized
        if changed<tolerance:
            ns,Cs,shapes,states,precres=integrate(s)
            n_end=float(ns[-1])
            spatial_rhs=np.zeros_like(qv)
            for n,C,sj in zip(ns,Cs,shapes):
                spatial_rhs+=dt*(H@(n*sj)+emission(C)+qv)
            residual=M*(n_end*s-n0*s0)-spatial_rhs
            denom=max(np.linalg.norm(M*(n_end*s),ord=np.inf),
                      np.linalg.norm(M*(n0*s0),ord=np.inf),
                      np.linalg.norm(spatial_rhs,ord=np.inf),1e-14)
            rel=float(np.linalg.norm(residual,ord=np.inf)/denom)
            if rel<tolerance*15 and abs(normalization-1)<tolerance*15:
                final=SpaceTimeState(initial.time_s+microsteps*dt,
                                     (n_end*s).reshape(m.shape),Cs[-1])
                inventory=(float(np.sum(M*(n_end*s-n0*s0)))+
                           float(np.sum(Cs[-1]-initial.precursors)))
                total=inventory
                for state in states:
                    phi=state.flux.flatten()
                    total+=dt*(float(np.sum(np.asarray(grid.absorption_inv_m)
                                             *state.flux*vol[:,:,None]))
                               +float(np.dot(m.boundary_current_coefficient,phi))
                               -float(np.sum(m.fission@phi)))
                total-=float(dt*microsteps*np.sum(qv))
                return ShapeMacroStep(final,n0,n_end,iteration,
                                      abs(float(wm@s)-1),rel,float(total),
                                      precres,tuple(float(x) for x in ns),
                                      tuple(states))
    raise RuntimeError('IQS shape iteration failed; reduce macro interval')


def run_iqs_shape(grid:DiffusionGrid,delayed:DelayedData,
                  initial:SpaceTimeState,dt:float,number_of_steps:int,
                  adjoint_weight:np.ndarray,*,shape_interval:int=4,
                  source:np.ndarray|None=None,tolerance:float=1e-9
                  )->tuple[ShapeMacroStep,...]:
    """Evolve shape at macro intervals; micro amplitude and precursors per dt."""
    if not isinstance(number_of_steps,int) or number_of_steps<1:
        raise ValueError('number_of_steps must be positive integer')
    if not isinstance(shape_interval,int) or shape_interval<1:
        raise ValueError('shape_interval must be positive integer')
    records=[];state=initial
    for first in range(0,number_of_steps,shape_interval):
        count=min(shape_interval,number_of_steps-first)
        result=advance_iqs_shape(grid,delayed,state,dt,adjoint_weight,
                                 microsteps=count,source=source,tolerance=tolerance)
        records.append(result)
        state=result.state
    return tuple(records)
