"""Independent FX2-TH-inspired space-time multigroup diffusion benchmark.

Cell-integrated balance:
    M d(phi)/dt = -A phi + (1-beta)F phi + B C + Q
    d(C_i)/dt  = beta_i P(phi) - lambda_i C_i
where M=V/v, P=V*sum(nuSigmaF*phi), and
    (B C)[cell,g] = chi[cell,g]*sum_i(lambda_i*C[cell,i]).

Backward Euler uses exact algebraic precursor elimination. Spatial flux
and precursor inventories retain independent cell-by-cell degrees of freedom.
This is NOT original FX2-TH FORTRAN, NERVA physical data, or OpenMC transport.

A separate frozen-shape/adjoint-projected model provides a deliberately
approximate two-rate predictor, NOT historical improved quasi-static IQS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import math
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .fx2th_diffusion import DiffusionGrid, assemble_diffusion


@dataclass(frozen=True)
class DelayedData:
    beta: tuple[float, ...]
    decay_s_inv: tuple[float, ...]
    speeds_m_s: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.beta) != len(self.decay_s_inv):
            raise ValueError('beta/decay group mismatch')
        if any(not math.isfinite(b) or b < 0 for b in self.beta) or sum(self.beta) >= 1:
            raise ValueError('invalid beta fractions')
        if any(not math.isfinite(a) or a <= 0 for a in self.decay_s_inv):
            raise ValueError('invalid precursor decays')
        if not self.speeds_m_s or any(not math.isfinite(v) or v<=0 for v in self.speeds_m_s):
            raise ValueError('positive finite energy-group speeds required')


@dataclass(frozen=True)
class SpaceTimeState:
    time_s: float
    flux: np.ndarray
    precursors: np.ndarray  # CELL-INTEGRATED precursor inventory per cell/group


@dataclass(frozen=True)
class SpaceTimeStep:
    state: SpaceTimeState
    flux_equation_residual: float
    precursor_equation_residual: float
    total_neutron_balance_residual: float
    normalized_amplitude: float | None = None
    spatial_refreshed: bool = True


def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    v=np.asarray(value,dtype=float)
    if v.shape!=shape or not np.all(np.isfinite(v)) or np.any(v<0):
        raise ValueError(f'{name} must have shape {shape}, finite and nonnegative')
    return v


def _check(grid: DiffusionGrid, delayed: DelayedData, state: SpaceTimeState,
           source: np.ndarray, dt: float):
    m=assemble_diffusion(grid)
    if not math.isfinite(dt) or dt<=0:
        raise ValueError('dt must be finite positive')
    if not math.isfinite(state.time_s) or state.time_s<0:
        raise ValueError('invalid initial time')
    ny,nx,ng=m.shape
    if len(delayed.speeds_m_s)!=ng:
        raise ValueError('energy speed count mismatch')
    _array(state.flux,m.shape,'flux')
    _array(state.precursors,(ny,nx,len(delayed.beta)),'precursors')
    q=_array(source,m.shape,'external source')
    return m,q


def mass_diagonal(grid: DiffusionGrid, delayed: DelayedData) -> np.ndarray:
    m=assemble_diffusion(grid)
    if len(delayed.speeds_m_s)!=m.shape[-1]:
        raise ValueError('speed mismatch')
    return (m.cell_volume_m3[:,:,None]/np.asarray(delayed.speeds_m_s)[None,None,:]).flatten()


def critical_precursors_spatial(grid: DiffusionGrid, delayed: DelayedData,
                                flux: np.ndarray) -> np.ndarray:
    """Precursors stationary for a FIXED supplied flux; not criticality."""
    m=assemble_diffusion(grid)
    phi=_array(flux,m.shape,'flux')
    prod=np.sum(np.asarray(grid.nu_fission_inv_m)*phi,axis=2)*m.cell_volume_m3
    return (np.stack([prod*beta/lam for beta,lam in zip(delayed.beta,delayed.decay_s_inv)],axis=-1)
            if delayed.beta else np.zeros((*m.shape[:2],0)))


def _quantities(grid: DiffusionGrid, delayed: DelayedData,
                state: SpaceTimeState, source: np.ndarray, dt: float):
    m,q=_check(grid,delayed,state,source,dt)
    vol=m.cell_volume_m3
    mass=(vol[:,:,None]/np.asarray(delayed.speeds_m_s)[None,None,:]).flatten()
    chis=np.asarray(grid.fission_spectrum)
    beta=np.asarray(delayed.beta,dtype=float)
    lam=np.asarray(delayed.decay_s_inv,dtype=float)
    return m,q,vol,mass,chis,beta,lam


def _delayed_known(state: SpaceTimeState, chis: np.ndarray,
                   beta: np.ndarray, lam: np.ndarray, dt: float)->np.ndarray:
    if not len(beta):
        return np.zeros_like(state.flux,dtype=float)
    emitted=np.sum(state.precursors*lam[None,None,:]/(1+dt*lam)[None,None,:],axis=2)
    return chis*emitted[:,:,None]


def _source_total(q: np.ndarray,vol: np.ndarray) -> np.ndarray:
    return (q*vol[:,:,None]).flatten()


def _coeff(beta: np.ndarray,lam:np.ndarray,dt:float)->float:
    return float(1-beta.sum()+np.sum(dt*beta*lam/(1+dt*lam)))


def _new_precursors(grid:DiffusionGrid, flux:np.ndarray, old:SpaceTimeState,
                    vol:np.ndarray,beta:np.ndarray,lam:np.ndarray,dt:float)->np.ndarray:
    if len(beta)==0:
        return np.zeros((*vol.shape,0),dtype=float)
    p=np.sum(np.asarray(grid.nu_fission_inv_m)*flux,axis=2)*vol
    return ((old.precursors+dt*p[:,:,None]*beta[None,None,:])/
            (1+dt*lam)[None,None,:])


def _verify(grid:DiffusionGrid, delayed:DelayedData, old:SpaceTimeState,
            new:SpaceTimeState, q:np.ndarray, dt:float, m, mass,
            chis:np.ndarray, beta:np.ndarray,lam:np.ndarray,
            *,refreshed:bool,amplitude:float|None)->SpaceTimeStep:
    prev=old.flux.flatten()
    phi=new.flux.flatten()
    delayed_emission=(chis*np.sum(new.precursors*lam[None,None,:],axis=2)[:,:,None]).flatten() if len(beta) else np.zeros_like(phi)
    flow=(mass*(phi-prev)/dt+m.loss@phi-(1-beta.sum())*(m.fission@phi)
          -delayed_emission-_source_total(q,m.cell_volume_m3))
    p=np.sum(np.asarray(grid.nu_fission_inv_m)*new.flux,axis=2)*m.cell_volume_m3
    pres=(new.precursors-old.precursors)/dt-beta[None,None,:]*p[:,:,None]+lam[None,None,:]*new.precursors
    total_balance=(np.sum(mass*(phi-prev))/dt
        +np.sum((new.precursors-old.precursors)/dt)
        +np.sum(np.asarray(grid.absorption_inv_m)*new.flux*m.cell_volume_m3[:,:,None])
        +np.dot(m.boundary_current_coefficient,phi)-np.sum(m.fission@phi)
        -np.sum(_source_total(q,m.cell_volume_m3)))
    return SpaceTimeStep(new,float(np.max(np.abs(flow))),
                         float(np.max(np.abs(pres))) if pres.size else 0.,
                         float(total_balance),amplitude,refreshed)


def advance_space_time(grid:DiffusionGrid, delayed:DelayedData, old:SpaceTimeState,
                       dt:float, source:np.ndarray|None=None)->SpaceTimeStep:
    """Full backward-Euler implicit multigroup flux and precursor step."""
    if source is None: source=np.zeros_like(old.flux)
    m,q,vol,mass,chis,beta,lam=_quantities(grid,delayed,old,source,dt)
    effective=_coeff(beta,lam,dt)
    H=diags(mass/dt,format='csc')+m.loss-effective*m.fission
    rhs=(mass*old.flux.flatten()/dt+_delayed_known(old,chis,beta,lam,dt).flatten()
         +_source_total(q,vol))
    phi=np.asarray(spsolve(H,rhs),dtype=float)
    if np.any(~np.isfinite(phi)) or np.min(phi)<-1e-9:
        raise ArithmeticError('implicit spatial kinetics step produced invalid flux; reduce dt')
    phi=np.maximum(phi,0).reshape(m.shape)
    prec=_new_precursors(grid,phi,old,vol,beta,lam,dt)
    new=SpaceTimeState(old.time_s+dt,phi,prec)
    return _verify(grid,delayed,old,new,q,dt,m,mass,chis,beta,lam,
                   refreshed=True,amplitude=None)


def project_amplitude(grid:DiffusionGrid,delayed:DelayedData,state:SpaceTimeState,
                      adjoint_weight:np.ndarray)->tuple[float,np.ndarray]:
    m,_=_check(grid,delayed,state,np.zeros_like(state.flux),1.)
    w=np.asarray(adjoint_weight,dtype=float)
    if w.shape!=m.shape or np.any(~np.isfinite(w)) or np.any(w<0) or not np.any(w>0):
        raise ValueError('adjoint weight must be nonnegative and correctly shaped')
    mass=mass_diagonal(grid,delayed)
    amp=float(np.dot(w.flatten()*mass,state.flux.flatten()))
    if amp<=0 or not math.isfinite(amp):
        raise ValueError('nonpositive weighted amplitude')
    shape=state.flux/amp
    return amp,shape


def advance_frozen_shape(grid:DiffusionGrid,delayed:DelayedData,old:SpaceTimeState,
                         dt:float,adjoint_weight:np.ndarray,
                         source:np.ndarray|None=None)->SpaceTimeStep:
    """Adjoint-projected, precursor-aware *fixed-shape* reduced model.

    Does NOT solve the improved-quasistatic shape PDE; exposes its residual.
    """
    if source is None: source=np.zeros_like(old.flux)
    m,q,vol,mass,chis,beta,lam=_quantities(grid,delayed,old,source,dt)
    n,shape=project_amplitude(grid,delayed,old,adjoint_weight)
    w=np.asarray(adjoint_weight).flatten()
    s=shape.flatten()
    known=_delayed_known(old,chis,beta,lam,dt).flatten()
    k=_coeff(beta,lam,dt)
    growth=float(w@((-m.loss+k*m.fission)@s))
    num=n+dt*float(w@(known+_source_total(q,vol)))
    den=1-dt*growth
    if den<=0 or not math.isfinite(den):
        raise ArithmeticError('projected kinetics time step unstable; reduce dt')
    new_n=num/den
    if not math.isfinite(new_n) or new_n<0:
        raise ArithmeticError('nonphysical projected amplitude')
    new_phi=new_n*shape
    new_precs=_new_precursors(grid,new_phi,old,vol,beta,lam,dt)
    new=SpaceTimeState(old.time_s+dt,new_phi,new_precs)
    return _verify(grid,delayed,old,new,q,dt,m,mass,chis,beta,lam,
                   refreshed=False,amplitude=new_n)


def run_space_time(grids:Sequence[DiffusionGrid], delayed:DelayedData,
                   initial:SpaceTimeState, dt:float,
                   sources:Sequence[np.ndarray]|None=None,
                   *,refresh_every:int=1,adjoint_weight:np.ndarray|None=None
                   )->tuple[SpaceTimeStep,...]:
    """Externally specified material coefficient sequence on a FIXED mesh.

    refresh_every=1 solves spatial equations every step. Larger values use
    a reduced, frozen-shape prediction on intermediate steps. No adaptive
    shape PDE is implied, and no OpenMC data is generated.
    """
    if not isinstance(refresh_every,int) or refresh_every<1:
        raise ValueError('refresh_every must be positive')
    if not grids: raise ValueError('at least one grid required')
    if sources is None: sources=[np.zeros_like(initial.flux) for _ in grids]
    if len(sources)!=len(grids): raise ValueError('source sequence length mismatch')
    if refresh_every>1 and adjoint_weight is None:
        raise ValueError('reduced stepping needs a reference adjoint')
    reference=grids[0]
    for step_grid in grids[1:]:
        if (not np.array_equal(step_grid.x_edges_m,reference.x_edges_m)
                or not np.array_equal(step_grid.y_edges_m,reference.y_edges_m)):
            raise ValueError('time-varying meshes need explicit conservative remapping')
    state=initial
    history=[]
    for index,(grid,q) in enumerate(zip(grids,sources)):
        if refresh_every==1 or (index+1)%refresh_every==0:
            result=advance_space_time(grid,delayed,state,dt,q)
        else:
            result=advance_frozen_shape(grid,delayed,state,dt,adjoint_weight,q)
        history.append(result)
        state=result.state
    return tuple(history)
