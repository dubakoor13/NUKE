"""Second-order L-stable implicit spatial neutron / precursor time integration.

Uses the Stage 7 cell-integrated multigroup diffusion and precursor
contracts without altering the Stage 12 cylindrical diffusion operator.
The coupled linear ODE is assembled independently as y' = G*y + b,
where y = [phi(c,g), C_integrated(c,f)]. The kinetic operator is fixed
during one integration interval. The two-stage singly diagonally implicit
Runge-Kutta (SDIRK2) method uses gamma=1-1/sqrt(2):

  (I-gamma*h*G) Y1 = y0 + gamma*h*b
  (I-gamma*h*G) Y2 = y0 + (1-gamma)*h*G*Y1 + h*b
  y_next=Y2

The two stages share ONE sparse LU factorization. The method is
second-order, A-stable and L-stable, but NOT unconditionally positivity
preserving. Reject nonphysical negative population states; never clip
them or silently advance with unconverged spatial kinetics.

For adaptivity, compare one step h with two steps h/2, estimating the
accepted fine solution's local error as |fine-coarse|/(2**2-1).
This is a LOCAL truncation-error estimate, not a rigorous global bound.
Whole-cell integrated precursor inventory and M=V/v neutron population
are included in the error check and conservation audit.

The material data, fission/delayed spectra, fixed nuclear source, and
mesh are supplied by the caller. Nothing here invents NERVA nuclear
cross sections, power, geometry, thermal feedback, OpenMC tallies, or
historical physical validation. This is an independent modern scheme,
NOT an exact port of the historical FX2-TH program.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import bmat, csc_matrix, coo_matrix, diags, identity
from scipy.sparse.linalg import splu

from .fx2th_family_spacetime import SpatialFamilyData, SpatialFamilyState


GAMMA = 1.0 - 1.0 / math.sqrt(2.0)


@dataclass(frozen=True)
class CoreOperator:
    generator: csc_matrix
    external_source: np.ndarray
    neutron_mass: np.ndarray
    source_id: str
    cells: int
    groups: int
    families: int


@dataclass(frozen=True)
class SDIRK2Step:
    state: SpatialFamilyState
    stage1_relative_residual: float
    stage2_relative_residual: float
    global_balance_relative_residual: float
    global_balance_absolute_residual: float
    neutron_inventory: float
    precursor_inventory: float


@dataclass(frozen=True)
class SDIRK2Tolerances:
    relative: float = 1e-4
    flux_absolute: float = 1e-9
    precursor_absolute: float = 1e-9
    min_dt_s: float = 1e-8
    max_dt_s: float = 0.2
    max_attempts: int = 20000
    max_rejections: int = 5000
    balance_tolerance: float = 1e-9

    def __post_init__(self):
        for name in ("relative", "flux_absolute", "precursor_absolute",
                     "min_dt_s", "max_dt_s", "balance_tolerance"):
            x = getattr(self, name)
            if not math.isfinite(x) or x <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_dt_s > self.max_dt_s:
            raise ValueError("minimum timestep exceeds maximum")
        if (not isinstance(self.max_attempts, int) or self.max_attempts < 1
                or not isinstance(self.max_rejections, int)
                or self.max_rejections < 0):
            raise ValueError("invalid attempt or rejection budget")


@dataclass(frozen=True)
class SDIRK2AcceptedStep:
    time_s: float
    duration_s: float
    estimated_local_error: float
    flux_error: float
    precursor_error: float
    integrated_neutron_error: float
    integrated_precursor_error: float
    global_balance_relative_residual: float
    cumulative_rejections: int


@dataclass(frozen=True)
class SDIRK2Result:
    state: SpatialFamilyState
    steps: tuple[SDIRK2AcceptedStep, ...]
    attempted_steps: int
    rejected_steps: int
    implicit_stage_solves: int
    max_accepted_estimated_error: float
    source_id: str


def _valid_state(data: SpatialFamilyData, state: SpatialFamilyState) -> np.ndarray:
    if not math.isfinite(state.time_s) or state.time_s < 0:
        raise ValueError("invalid neutron/precursor initial time")
    phi = np.asarray(state.flux, dtype=float)
    C = np.asarray(state.precursors, dtype=float)
    if (phi.shape != (data.cells, data.groups)
            or C.shape != (data.cells, data.families)
            or not np.all(np.isfinite(phi))
            or not np.all(np.isfinite(C))
            or np.any(phi < 0) or np.any(C < 0)):
        raise ValueError("invalid spatial neutron or precursor state")
    return np.r_[phi.ravel(), C.ravel()]


def build_core_operator(
    data: SpatialFamilyData,
    external_source_per_m3: np.ndarray | None = None,
) -> CoreOperator:
    """Sparse full-state generator for fixed externally supplied material data.

    Independent assembly prevents hiding consistency mistakes in the
    Stage-7 Schur elimination. For realistic meshes direct full-state
    sparse LU may require more memory than a specialized Schur solver.
    """
    nc, ng, nf = data.cells, data.groups, data.families
    N = nc * ng
    L = nc * nf
    v = np.asarray(data.volumes_m3, dtype=float)
    xs = np.asarray(data.nu_sigma_f_inv_m, dtype=float)
    beta = np.asarray(data.beta_by_incident_group_and_family, dtype=float)
    chip = np.asarray(data.prompt_spectrum, dtype=float)
    chid = np.asarray(data.delayed_spectra_by_family, dtype=float)
    lam = np.asarray(data.decay_constants_s_inv, dtype=float)
    M = data.mass_diagonal
    if external_source_per_m3 is None:
        q = np.zeros((nc, ng), dtype=float)
    else:
        q = np.asarray(external_source_per_m3, dtype=float)
        if (q.shape != (nc, ng) or np.any(~np.isfinite(q))
                or np.any(q < 0)):
            raise ValueError("external source must be nonnegative finite [cell,group]")

    pr, pc, pv = [], [], []
    br, bc, bv = [], [], []
    er, ec, ev = [], [], []
    for c in range(nc):
        for gi in range(ng):
            fissions = v[c] * xs[c, gi]
            prompt = fissions * (1.0 - float(beta[c, gi, :].sum()))
            for go in range(ng):
                if chip[c, go] != 0 and prompt != 0:
                    pr.append(c*ng+go)
                    pc.append(c*ng+gi)
                    pv.append(chip[c, go]*prompt)
            for f in range(nf):
                b = fissions*beta[c, gi, f]
                if b != 0:
                    br.append(c*nf+f)
                    bc.append(c*ng+gi)
                    bv.append(b)
        for f in range(nf):
            for go in range(ng):
                e = chid[c, f, go]*lam[f]
                if e != 0:
                    er.append(c*ng+go)
                    ec.append(c*nf+f)
                    ev.append(e)

    P = coo_matrix((pv, (pr, pc)), shape=(N, N)).tocsc()
    B = coo_matrix((bv, (br, bc)), shape=(L, N)).tocsc()
    E = coo_matrix((ev, (er, ec)), shape=(N, L)).tocsc()
    D = diags(np.tile(lam, nc), format="csc")
    Minv = diags(1.0/M, format="csc")
    top_left = Minv @ (-data.integrated_loss.tocsc() + P)
    top_right = Minv @ E
    G = csc_matrix(bmat([[top_left, top_right], [B, -D]], format="csc"))
    b = np.r_[np.asarray((q*v[:, None])).ravel()/M, np.zeros(L)]
    if not np.all(np.isfinite(G.data)) or not np.all(np.isfinite(b)):
        raise ArithmeticError("nonfinite coupled neutron/precursor generator")
    return CoreOperator(G, b, M, data.source_id, nc, ng, nf)


def _population(y: np.ndarray, op: CoreOperator) -> float:
    N = op.cells*op.groups
    return float(np.dot(op.neutron_mass, y[:N]) + y[N:].sum())


def _population_rhs(y: np.ndarray, op: CoreOperator) -> float:
    """Independently audited total-neutron-plus-precursor population rate."""
    dy = op.generator @ y + op.external_source
    return float(np.dot(op.neutron_mass, dy[:op.cells*op.groups])
                 + np.sum(dy[op.cells*op.groups:]))


def _step_with_factorization(
    op: CoreOperator,
    initial_vector: np.ndarray,
    duration_s: float,
    *,
    lu=None,
) -> tuple[np.ndarray, float, float, float, float]:
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("positive finite duration required")
    g = op.generator
    rhs_scale = max(1.0, float(np.linalg.norm(initial_vector, ord=np.inf)))
    matrix = identity(g.shape[0], format="csc") - (GAMMA*duration_s)*g
    factor = splu(matrix) if lu is None else lu
    b = op.external_source
    rhs1 = initial_vector + GAMMA*duration_s*b
    y1 = factor.solve(rhs1)
    rhs2 = initial_vector + duration_s*((1.-GAMMA)*(g @ y1) + b)
    y2 = factor.solve(rhs2)
    if (not np.all(np.isfinite(y1)) or not np.all(np.isfinite(y2))
            or np.any(y1 < 0) or np.any(y2 < 0)):
        raise ArithmeticError("implicit SDIRK2 stage has negative/nonfinite neutron or precursor population")
    r1 = float(np.linalg.norm(matrix@y1-rhs1, ord=np.inf)/rhs_scale)
    r2 = float(np.linalg.norm(matrix@y2-rhs2, ord=np.inf)/rhs_scale)
    delta = _population(y2, op)-_population(initial_vector, op)
    rhs_pop = duration_s*((1.-GAMMA)*_population_rhs(y1, op)
                          + GAMMA*_population_rhs(y2, op))
    balance = delta-rhs_pop
    scale = max(1., abs(delta), abs(rhs_pop),
                abs(_population(y2, op)), abs(_population(initial_vector, op)))
    return y2, r1, r2, balance, abs(balance)/scale


def advance_sdirk2(
    op: CoreOperator,
    data: SpatialFamilyData,
    initial: SpatialFamilyState,
    duration_s: float,
) -> SDIRK2Step:
    """One fully coupled two-stage implicit step with conservation audit."""
    if op.cells != data.cells or op.groups != data.groups or op.families != data.families:
        raise ValueError("operator and nuclear data dimensions do not match")
    if op.source_id != data.source_id:
        raise ValueError("operator and nuclear-data provenance mismatch")
    y0 = _valid_state(data, initial)
    y1, r1, r2, balance, relative = _step_with_factorization(
        op, y0, duration_s)
    N = op.cells*op.groups
    state = SpatialFamilyState(
        initial.time_s+duration_s,
        y1[:N].reshape(op.cells, op.groups),
        y1[N:].reshape(op.cells, op.families),
    )
    return SDIRK2Step(
        state, r1, r2, relative, balance,
        float(np.dot(op.neutron_mass, y1[:N])),
        float(np.sum(y1[N:])),
    )


def _scaled(x: np.ndarray | float, y: np.ndarray | float,
            absolute: float, relative: float) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.max(np.abs(a-b) /
               (absolute+relative*np.maximum(np.abs(a), np.abs(b)))))


def _error(
    op: CoreOperator, coarse: np.ndarray, fine: np.ndarray,
    cfg: SDIRK2Tolerances,
) -> tuple[float, float, float, float, float]:
    N = op.cells*op.groups
    factor = 1.0 / 3.0 # fine local truncation error: coarse-fine/(2**p-1), p=2
    ef = factor*_scaled(coarse[:N], fine[:N], cfg.flux_absolute, cfg.relative)
    ec = factor*_scaled(coarse[N:], fine[N:], cfg.precursor_absolute, cfg.relative)
    mn = float(np.dot(op.neutron_mass,coarse[:N]))
    mf = float(np.dot(op.neutron_mass,fine[:N]))
    en = factor*_scaled(mn,mf,cfg.flux_absolute*op.neutron_mass.sum(),cfg.relative)
    pc = float(np.sum(coarse[N:]))
    pf = float(np.sum(fine[N:]))
    ep = factor*_scaled(pc,pf,cfg.precursor_absolute*(len(coarse)-N),cfg.relative)
    return max(ef,ec,en,ep),ef,ec,en,ep


def advance_sdirk2_adaptive(
    op: CoreOperator, data: SpatialFamilyData,
    initial: SpatialFamilyState,
    end_time_s: float, *,
    initial_dt_s: float,
    tolerances: SDIRK2Tolerances | None = None,
) -> SDIRK2Result:
    """Adaptive second-order L-stable core kinetics for fixed external data."""
    cfg = SDIRK2Tolerances() if tolerances is None else tolerances
    if not isinstance(cfg, SDIRK2Tolerances):
        raise TypeError("tolerances must be SDIRK2Tolerances")
    y = _valid_state(data, initial)
    if (op.source_id != data.source_id or
        op.cells != data.cells or op.groups != data.groups or op.families != data.families):
        raise ValueError("operator/kinetic parameter contract mismatch")
    if (not math.isfinite(end_time_s) or end_time_s<=initial.time_s
            or not math.isfinite(initial_dt_s) or initial_dt_s<=0):
        raise ValueError("finite end_time and positive starting timestep required")
    eps_time=16*np.finfo(float).eps*max(1.,abs(initial.time_s),abs(end_time_s))
    if end_time_s-initial.time_s<=eps_time:
        raise ValueError("target interval smaller than representable timestep")
    t = initial.time_s
    h = min(cfg.max_dt_s, initial_dt_s)
    attempt = rejected = solves = 0
    results=[]
    while end_time_s-t > eps_time:
        if attempt >= cfg.max_attempts:
            raise RuntimeError("adaptive SDIRK2 exhausted attempt budget")
        remaining=end_time_s-t
        h=min(h,cfg.max_dt_s,remaining)
        if h<cfg.min_dt_s and remaining>cfg.min_dt_s*(1+1e-12):
            raise RuntimeError("adaptive SDIRK2 timestep below configured minimum")
        attempt+=1
        try:
            # One factorization for a full step; one half-step factorization
            # is reused for BOTH consecutive half steps.
            big,r1,r2,bal_b,rel_b=_step_with_factorization(op,y,h)
            half_matrix=identity(op.generator.shape[0],format="csc")-(
                GAMMA*h/2.)*op.generator
            half_lu=splu(half_matrix)
            half,hr1,hr2,bal_h,rel_h=_step_with_factorization(
                op,y,h/2.,lu=half_lu)
            fine,fr1,fr2,bal_f,rel_f=_step_with_factorization(
                op,half,h/2.,lu=half_lu)
            solves+=6
            if max(r1,r2,hr1,hr2,fr1,fr2,rel_b,rel_h,rel_f)>cfg.balance_tolerance:
                raise ArithmeticError("implicit neutron-balance/linear stage residual exceeded limit")
            errors=_error(op,big,fine,cfg)
            error=errors[0]
            if not math.isfinite(error):
                raise ArithmeticError("nonfinite local truncation estimate")
        except ArithmeticError as exc:
            error=math.inf
            reason=str(exc)
            errors=(math.inf,)*5
        else:
            reason="local error exceeded requested tolerance"

        if error<=1.:
            # The accepted state is the positivity-preserving fine step
            # (if both implicit stages are nonnegative); no extrapolation.
            y=fine
            t=min(end_time_s,t+h)
            results.append(SDIRK2AcceptedStep(
                time_s=t,duration_s=h,estimated_local_error=errors[0],
                flux_error=errors[1],precursor_error=errors[2],
                integrated_neutron_error=errors[3],
                integrated_precursor_error=errors[4],
                global_balance_relative_residual=max(rel_h,rel_f),
                cumulative_rejections=rejected))
            factor=(2.5 if error<1e-15 else
                    float(np.clip(.9*error**(-1./3.),.5,2.5)))
            h=min(cfg.max_dt_s,h*factor)
        else:
            rejected+=1
            if rejected>cfg.max_rejections:
                raise RuntimeError("adaptive SDIRK2 exceeded rejected-step budget")
            factor=(.2 if not math.isfinite(error) else
                    float(np.clip(.8*error**(-1./3.),.1,.5)))
            if h*factor<cfg.min_dt_s:
                raise RuntimeError("adaptive SDIRK2 rejected step at minimum timestep: "+reason)
            h*=factor
    if abs(end_time_s-t)>eps_time:
        raise RuntimeError("adaptive SDIRK2 did not reach requested final time")
    N=op.cells*op.groups
    return SDIRK2Result(
        state=SpatialFamilyState(end_time_s,
            y[:N].reshape(op.cells,op.groups),
            y[N:].reshape(op.cells,op.families)),
        steps=tuple(results),attempted_steps=attempt,
        rejected_steps=rejected,implicit_stage_solves=solves,
        max_accepted_estimated_error=max(
            (v.estimated_local_error for v in results),default=0.),
        source_id=op.source_id,
    )
