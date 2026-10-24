"""Adaptive timestep control for spatial multigroup neutron / precursor equations.

This is a verified *numerical* controller around the NUKE Stage-7 implicit
cell-integrated family solver, and can be used with the Stage-12 conservative
axisymmetric r-z diffusion loss matrix. No nuclear material parameters,
reactor geometry, temperatures or actual NERVA transients are inferred.

Embedded estimate: one backward Euler step of size h (coarse) versus two
backward Euler steps of h/2 (fine). The accepted solution is the TWO HALF
STEPS. Error is the max of elementwise scaled neutron flux and precursor
inventory differences and their mass-weighted global inventory differences.
These are LOCAL error estimates, not rigorous bounds on the GLOBAL error.
Full-state synthetic dense matrix-exponential references are tested
separately. The method is first order in global time accuracy; step-doubling
is NOT a higher-order integrator or improved quasistatic FX2-TH port.

The diffusion coefficients, neutron-group structure, precursors, and fixed
external source remain unchanged over one solve. If material state changes,
split the integration at that externally prescribed discontinuity; do not
interpolate data or integrate across the jump unknowingly.

No step is accepted if the estimated scaled local error exceeds 1, and
underresolved minimum timestep or solver failure is reported instead of
being silently accepted. Spatial global balance and cellwise precursor
equation residuals are recorded for each accepted half step.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .fx2th_family_spacetime import (
    SpatialFamilyData, SpatialFamilyState, SpatialFamilyStep,
    advance_spatial_families,
)


@dataclass(frozen=True)
class AdaptiveTolerances:
    """Local error controls, with separate flux and precursor absolute scales.

    rtol is dimensionless; atol_flux has units of flux, and atol_precursors
    has units of the Stage-7 *integrated* family inventory. In real units,
    user MUST supply physically consistent thresholds for both variables.
    """
    rtol: float = 2e-3
    atol_flux: float = 1e-9
    atol_precursors: float = 1e-9
    min_dt_s: float = 1e-7
    max_dt_s: float = 0.1
    max_steps: int = 20000
    max_rejected: int = 5000
    max_balance_residual: float = 1e-8

    def __post_init__(self):
        for name in ("rtol","atol_flux","atol_precursors","min_dt_s",
                     "max_dt_s","max_balance_residual"):
            x=getattr(self,name)
            if not math.isfinite(x) or x<=0:
                raise ValueError(f"{name} must be finite positive")
        if self.min_dt_s>self.max_dt_s:
            raise ValueError("min_dt_s cannot exceed max_dt_s")
        if (not isinstance(self.max_steps,int) or self.max_steps<1 or
            not isinstance(self.max_rejected,int) or self.max_rejected<0):
            raise ValueError("max_steps and max_rejected must be nonnegative limits")


@dataclass(frozen=True)
class AcceptedCoreStep:
    time_s: float
    dt_s: float
    scaled_local_error: float
    flux_scaled_error: float
    precursor_scaled_error: float
    integrated_neutron_scaled_error: float
    integrated_precursor_scaled_error: float
    first_half_neutron_balance_residual: float
    second_half_neutron_balance_residual: float
    first_half_precursor_residual: float
    second_half_precursor_residual: float
    neutron_inventory: float
    precursor_inventory: float
    cumulative_rejections: int


@dataclass(frozen=True)
class AdaptiveCoreResult:
    state: SpatialFamilyState
    steps: tuple[AcceptedCoreStep, ...]
    attempted_steps: int
    rejected_steps: int
    linear_implicit_solves: int
    max_local_error_accepted: float
    source_id: str


def _relative(a: np.ndarray | float, b: np.ndarray | float,
              atol: float, rtol: float) -> float:
    x=np.asarray(a,dtype=float)
    y=np.asarray(b,dtype=float)
    if x.shape!=y.shape:
        raise ValueError("adaptive error comparison changed array shape")
    scale=atol+rtol*np.maximum(np.abs(x),np.abs(y))
    return float(np.max(np.abs(x-y)/scale))


def _error_estimate(
    data: SpatialFamilyData,
    coarse: SpatialFamilyState,
    fine: SpatialFamilyState,
    cfg: AdaptiveTolerances,
) -> tuple[float,float,float,float,float]:
    ef=_relative(coarse.flux,fine.flux,cfg.atol_flux,cfg.rtol)
    ep=_relative(coarse.precursors,fine.precursors,cfg.atol_precursors,cfg.rtol)
    # The neutron population is M phi; the precursor inventory is already
    # integrated over the cell. The group speed and cell volumes count once.
    mass=data.mass_diagonal
    a=float(np.dot(mass,coarse.flux.flatten()))
    b=float(np.dot(mass,fine.flux.flatten()))
    # Consistent absolute inventory scales: at least atol per flux element,
    # weighted with its own cell/group inverse-speed mass.
    neutron_abs=cfg.atol_flux*float(np.sum(mass))
    en=_relative(a,b,neutron_abs,cfg.rtol)
    a=float(np.sum(coarse.precursors))
    b=float(np.sum(fine.precursors))
    epop=_relative(a,b,
        cfg.atol_precursors*coarse.precursors.size,cfg.rtol)
    return max(ef,ep,en,epop),ef,ep,en,epop


def _checked_step(step: SpatialFamilyStep, cfg: AdaptiveTolerances) -> None:
    residuals=(
        step.flux_equation_residual,
        step.precursor_equation_residual,
        step.global_balance_residual
    )
    if any(not math.isfinite(x) or x>cfg.max_balance_residual for x in residuals):
        raise ArithmeticError(
            "implicit step does not satisfy conservative flux/precursor balance"
        )


def advance_adaptive_core(
    data: SpatialFamilyData,
    initial: SpatialFamilyState,
    end_time_s: float,
    *,
    initial_dt_s: float,
    tolerances: AdaptiveTolerances | None = None,
    external_source_per_m3: np.ndarray | None = None,
) -> AdaptiveCoreResult:
    """Advance a fixed-material Stage-7 spatial-family system to end_time_s.

    End_time must strictly exceed the initial time. The source may be any
    finite nonnegative fixed cell/group array matching the Stage-7 contract;
    it is never rescaled or interpreted as reactor power.
    """
    cfg=tolerances if tolerances is not None else AdaptiveTolerances()
    if not isinstance(cfg,AdaptiveTolerances):
        raise TypeError("tolerances must be AdaptiveTolerances")
    if (not math.isfinite(end_time_s) or end_time_s<=initial.time_s or
        not math.isfinite(initial_dt_s) or initial_dt_s<=0):
        raise ValueError("finite end_time > initial.time and positive initial_dt required")
    if (not np.all(np.isfinite(initial.flux)) or
        initial.flux.shape!=(data.cells,data.groups) or
        np.any(initial.flux<0) or
        not np.all(np.isfinite(initial.precursors)) or
        initial.precursors.shape!=(data.cells,data.families) or
        np.any(initial.precursors<0)):
        raise ValueError("initial flux/precursor state invalid")
    if external_source_per_m3 is not None:
        q=np.asarray(external_source_per_m3,dtype=float)
        if (q.shape!=(data.cells,data.groups) or
            not np.all(np.isfinite(q)) or np.any(q<0)):
            raise ValueError("external source must be nonnegative [cell,group]")
    else:
        q=None

    steps=[]
    accepted=attempted=rejected=solves=0
    state=initial
    h=min(cfg.max_dt_s,initial_dt_s)
    eps_time=16*np.finfo(float).eps*max(abs(end_time_s),1.)
    while end_time_s-state.time_s>eps_time:
        if attempted >= cfg.max_steps:
            raise RuntimeError("adaptive core integration exceeded maximum attempted steps")
        remaining=end_time_s-state.time_s
        h=min(h,cfg.max_dt_s,remaining)
        # Final short interval is allowed only when the requested endpoint
        # itself is less than the configured minimum remaining dt.
        if h<cfg.min_dt_s and remaining>cfg.min_dt_s*(1+1e-12):
            raise RuntimeError("adaptive timestep below minimum without reaching endpoint")
        attempted+=1
        try:
            big=advance_spatial_families(data,state,h,q)
            first=advance_spatial_families(data,state,h/2,q)
            second=advance_spatial_families(data,first.state,h/2,q)
            solves+=3
            _checked_step(big,cfg)
            _checked_step(first,cfg)
            _checked_step(second,cfg)
            estimate=_error_estimate(data,big.state,second.state,cfg)
            err=estimate[0]
            if not math.isfinite(err):
                raise ArithmeticError("nonfinite embedded local error")
        except ArithmeticError as exc:
            # E.g. nonphysical implicit flux: reduce h, but do not treat it
            # as a converged step or swallow unrelated programming errors.
            err=math.inf
            estimate=(math.inf,)*5
            reason=str(exc)
        else:
            reason="local error estimate above 1"

        if err<=1.:
            # The refined two-half-step result is the accepted first-order
            # result. No Richardson extrapolation that might violate
            # positivity or the conservative implicit state equations.
            state=second.state
            accepted+=1
            steps.append(AcceptedCoreStep(
                time_s=state.time_s,dt_s=h,
                scaled_local_error=estimate[0],
                flux_scaled_error=estimate[1],
                precursor_scaled_error=estimate[2],
                integrated_neutron_scaled_error=estimate[3],
                integrated_precursor_scaled_error=estimate[4],
                first_half_neutron_balance_residual=first.global_balance_residual,
                second_half_neutron_balance_residual=second.global_balance_residual,
                first_half_precursor_residual=first.precursor_equation_residual,
                second_half_precursor_residual=second.precursor_equation_residual,
                neutron_inventory=second.neutron_inventory,
                precursor_inventory=second.total_precursor_inventory,
                cumulative_rejections=rejected,
            ))
            # Backward Euler coarse-fine discrepancy scales ~h**2 locally.
            factor=(2. if err<=1e-16 else
                    float(np.clip(.90/math.sqrt(err),.5,2.)))
            h=min(cfg.max_dt_s,h*factor)
        else:
            rejected+=1
            if rejected>cfg.max_rejected:
                raise RuntimeError("adaptive core exceeded rejected step budget")
            factor=(.2 if not math.isfinite(err) else
                    float(np.clip(.8/math.sqrt(err),.1,.5)))
            candidate=h*factor
            if candidate<cfg.min_dt_s:
                raise RuntimeError(
                    f"adaptive core failed at min_dt_s; step rejected: {reason}"
                )
            h=candidate
    if abs(end_time_s-state.time_s)<=eps_time:
        # Remove accumulated time sum roundoff, without changing physical
        # fields or hiding an actual missed endpoint.
        state=SpatialFamilyState(end_time_s,state.flux,state.precursors)
    elif state.time_s!=end_time_s:
        raise RuntimeError("adaptive core did not reach the requested endpoint")
    return AdaptiveCoreResult(
        state=state,steps=tuple(steps),attempted_steps=attempted,
        rejected_steps=rejected,linear_implicit_solves=solves,
        max_local_error_accepted=max((s.scaled_local_error for s in steps),
                                     default=0.),
        source_id=data.source_id
    )
