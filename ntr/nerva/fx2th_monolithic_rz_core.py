"""Small-system MONOLITHIC implicit neutron/precursor/thermal core benchmark.

Couples the actual Stage-12 cylindrical conservative spatial diffusion,
Stage-15 prompt + family-delayed fission kinetics, and Stage-17 externally
supplied spatial temperature-indexed nuclear material coefficients to
Stage-8 fuel-solid/hydrogen two-node energy storage.

At end of a single physical backward Euler timestep the nonlinear unknown
is x=[phi_new(cell,group), C_new(cell,family), Ts_new(cell), Th_new(cell)].
The equations are solved SIMULTANEOUSLY; a temperature change actually
rebuilds the nuclear loss/scattering/fission/precursor generator, rather
than freezing a static reactivity coefficient or normalizing neutron flux
to arbitrary reactor power. Each solve always starts from the SAME old
physical state, not from the preceding nonlinear trial.

  y1 - y0 - dt * (G(Ts1) y1 + b) = 0
  Cs*(Ts1-Ts0) - dt*(Qs(y1,T1)-H*(Ts1-Th1)) = 0
  Ch*(Th1-Th0) - dt*(Qh(y1,T1)+H*(Ts1-Th1)
                                 -mdot*cp*(Th1-Tin)) = 0

G is the complete Stage-14 coupled linear neutron/precursor generator,
with Stage-17 temperature tables. Heating Qs,Qh,Qother in W is supplied
by an EXTERNAL, provenance-bearing callback, not manufactured from eigen
flux, nuSigmaF or nuclear fission Q values. Qother is explicitly tracked
outside the modeled solid and hydrogen cells, and first-law conservation
is checked using the existing Stage-8 energy-ledger oracle.

For reproducible small mathematical verification, use damped Newton with
a finite-difference Jacobian and a DENSE linear algebra solve. This is
NOT a scalable reactor production nonlinear solver; do not treat it as
an implicit validated transient for actual NERVA core materials/H2 EOS,
resolved fuel conduction, flow, or neutron/photon energy deposition.

Strict no-extrapolation on the externally supplied temperature tables;
positivity and convergence guards; no silent Newton failure acceptance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.linalg import solve as dense_solve, LinAlgWarning
import warnings

from .fx2th_family_spacetime import SpatialFamilyData, SpatialFamilyState
from .fx2th_sdirk2_neutron_kinetics import build_core_operator
from .fx2th_thermal_energy import (
    ThermalCells, ThermalState, ThermalDeposition,
    ThermalStep, advance_thermal_cells,
)
from .fx2th_rz_temperature_tables import (
    RZTemperatureTables, RZTemperatureSample, sample_rz_temperature_tables,
)


@dataclass(frozen=True)
class MonolithicControls:
    scaled_residual_tolerance: float = 1e-10
    finite_difference_relative_step: float = 1e-6
    max_iterations: int = 20
    min_line_search_fraction: float = 1/4096
    max_unknowns: int = 120
    time_consistency_tolerance_s: float = 1e-10

    def __post_init__(self):
        for name in (
            "scaled_residual_tolerance",
            "finite_difference_relative_step",
            "min_line_search_fraction",
            "time_consistency_tolerance_s",
        ):
            x=getattr(self,name)
            if not math.isfinite(x) or x<=0:
                raise ValueError(f"{name} must be finite positive")
        if self.scaled_residual_tolerance>1e-3:
            raise ValueError("nonlinear residual tolerance is too lax")
        if self.min_line_search_fraction>1:
            raise ValueError("line search minimum fraction exceeds one")
        if (not isinstance(self.max_iterations,int) or self.max_iterations<1 or
            not isinstance(self.max_unknowns,int) or self.max_unknowns<1):
            raise ValueError("positive integer Newton budgets required")


@dataclass(frozen=True)
class MonolithicResidual:
    scaled: np.ndarray
    raw: np.ndarray
    sample: RZTemperatureSample
    deposition: ThermalDeposition
    state: SpatialFamilyState
    thermal_state: ThermalState
    neutron_population_balance: float


@dataclass(frozen=True)
class MonolithicCoreStep:
    neutron_state: SpatialFamilyState
    thermal_state: ThermalState
    thermal_ledger: ThermalStep
    material: SpatialFamilyData
    iterations: int
    evaluated_states: int
    scaled_residual: float
    neutron_population_balance: float
    material_source_id: str
    deposition_source_id: str
    synthetic: bool


class _ResidualEvaluator:
    def __init__(
        self,
        table:RZTemperatureTables,
        initial_neutrons:SpatialFamilyState,
        initial_thermal:ThermalState,
        cells:ThermalCells,
        dt_s:float,
        power:Callable[[SpatialFamilyState,SpatialFamilyData],ThermalDeposition],
        source:np.ndarray | None
    ):
        self.table=table
        self.initial_neutrons=initial_neutrons
        self.initial_thermal=initial_thermal
        self.cells=cells
        self.dt_s=dt_s
        self.power=power
        self.source=source
        self.nr,self.nz=table.spatial_shape
        self.ncell=self.nr*self.nz
        self.ng=int(table.reference_grid.diffusion_m.shape[-1])
        self.nf=len(table.decay_constants_s_inv)
        self.nneutron=self.ncell*self.ng
        self.nprecursor=self.ncell*self.nf
        self.ny=self.nneutron+self.nprecursor
        self.calls=0
        self.y0=np.r_[
            initial_neutrons.flux.ravel(),
            initial_neutrons.precursors.ravel()
        ]
        self.x0=np.r_[
            self.y0,initial_thermal.solid_temperature_k,
            initial_thermal.hydrogen_temperature_k
        ]
        Cs=np.asarray(cells.solid_heat_capacity_j_k,dtype=float)
        Ch=np.asarray(cells.hydrogen_heat_capacity_j_k,dtype=float)
        self.Cs=Cs
        self.Ch=Ch
        self.H=np.asarray(cells.wall_conductance_w_k,dtype=float)
        self.flow=np.asarray(cells.hydrogen_mass_flow_kg_s,dtype=float)*(
            np.asarray(cells.hydrogen_cp_j_kg_k,dtype=float))
        self.Tin=np.asarray(cells.inlet_temperature_k,dtype=float)
        # Scaling uses OLD physical state, and is held constant for Newton.
        self.scales=np.r_[
            np.maximum(1.,np.abs(self.y0)),
            np.maximum(1.,Cs*np.asarray(initial_thermal.solid_temperature_k)),
            np.maximum(1.,Ch*np.asarray(initial_thermal.hydrogen_temperature_k))
        ]

    def unpack(self,x):
        y=x[:self.ny]
        solid=x[self.ny:self.ny+self.ncell]
        hydrogen=x[self.ny+self.ncell:]
        return y,solid,hydrogen

    def valid(self,x):
        if x.shape != self.x0.shape or not np.all(np.isfinite(x)):
            return False
        y,solid,hydrogen=self.unpack(x)
        knots=np.asarray(self.table.temperature_knots_k,dtype=float)
        return bool(np.all(y>=0) and np.all(solid>0) and
                    np.all(hydrogen>0) and
                    np.all(solid>=knots[0]) and np.all(solid<=knots[-1]))

    def evaluate(self,x):
        if not self.valid(x):
            raise ValueError("invalid trial: neutron/precursor positivity or temperature table range")
        self.calls+=1
        y,solid,hydrogen=self.unpack(x)
        sample=sample_rz_temperature_tables(
            self.table,solid.reshape(self.nr,self.nz))
        data=sample.material
        operator=build_core_operator(data,self.source)
        dt=self.dt_s
        kinetic=SpatialFamilyState(
            self.initial_neutrons.time_s+dt,
            y[:self.nneutron].reshape(self.ncell,self.ng),
            y[self.nneutron:].reshape(self.ncell,self.nf))
        thermal=ThermalState(
            self.initial_thermal.time_s+dt,
            solid.copy(),hydrogen.copy())
        deposited=self.power(kinetic,data)
        if not isinstance(deposited,ThermalDeposition):
            raise TypeError("explicit power callback must return ThermalDeposition")
        qs,qh,_=deposited.check(self.ncell)
        rn=y-self.y0-dt*(operator.generator@y+operator.external_source)
        rs=self.Cs*(solid-self.initial_thermal.solid_temperature_k)-(
            dt*(qs-self.H*(solid-hydrogen)))
        rh=self.Ch*(hydrogen-self.initial_thermal.hydrogen_temperature_k)-(
            dt*(qh+self.H*(solid-hydrogen)-self.flow*(hydrogen-self.Tin)))
        raw=np.r_[rn,rs,rh]
        mass=operator.neutron_mass
        total_old=float(np.dot(mass,self.y0[:self.nneutron])+
                        np.sum(self.y0[self.nneutron:]))
        total_new=float(np.dot(mass,y[:self.nneutron])+
                        np.sum(y[self.nneutron:]))
        dy=operator.generator@y+operator.external_source
        neutron_balance=(total_new-total_old)-dt*float(
            np.dot(mass,dy[:self.nneutron])+np.sum(dy[self.nneutron:]))
        return MonolithicResidual(raw/self.scales,raw,
                                  sample,deposited,kinetic,thermal,
                                  neutron_balance)


def _finite_difference_jacobian(
    evaluator:_ResidualEvaluator, x:np.ndarray,
    baseline:np.ndarray, cfg:MonolithicControls,
) -> np.ndarray:
    n=len(x)
    J=np.empty((n,n),dtype=float)
    for j in range(n):
        h=cfg.finite_difference_relative_step*max(1.,abs(x[j]))
        plus=x.copy()
        minus=x.copy()
        plus[j]+=h
        minus[j]-=h
        vp=evaluator.valid(plus)
        vm=evaluator.valid(minus)
        if vp and vm:
            ap=evaluator.evaluate(plus).scaled
            am=evaluator.evaluate(minus).scaled
            J[:,j]=(ap-am)/(2*h)
        elif vp:
            ap=evaluator.evaluate(plus).scaled
            J[:,j]=(ap-baseline)/h
        elif vm:
            am=evaluator.evaluate(minus).scaled
            J[:,j]=(baseline-am)/h
        else:
            raise RuntimeError("cannot construct positivity-preserving numerical Jacobian")
    if not np.all(np.isfinite(J)):
        raise ArithmeticError("nonfinite numerical coupled Jacobian")
    return J


def advance_monolithic_rz_core(
    table:RZTemperatureTables,
    initial_neutrons:SpatialFamilyState,
    initial_thermal:ThermalState,
    cells:ThermalCells,
    dt_s:float,
    deposition_from_kinetics:Callable[
        [SpatialFamilyState,SpatialFamilyData],ThermalDeposition
    ],
    *,
    external_source_per_m3:np.ndarray | None=None,
    controls:MonolithicControls | None=None,
) -> MonolithicCoreStep:
    """Fully implicit coupled neutron/precursor/solid/H2 step, small meshes.

    Does NOT support numerical nuclear heating inference or material
    coefficients outside the qualified knot range supplied by the caller.
    """
    cfg=MonolithicControls() if controls is None else controls
    if not isinstance(cfg,MonolithicControls):
        raise TypeError("controls must be MonolithicControls")
    if not callable(deposition_from_kinetics):
        raise TypeError("deposited power must be an explicit callback")
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("positive finite physical timestep required")
    if (not math.isfinite(initial_neutrons.time_s) or
        not math.isfinite(initial_thermal.time_s) or
        initial_neutrons.time_s<0 or initial_thermal.time_s<0 or
        abs(initial_neutrons.time_s-initial_thermal.time_s)>
            cfg.time_consistency_tolerance_s):
        raise ValueError("old thermal/neutronic times must match")
    nr,nz=table.spatial_shape
    nc=nr*nz
    ng=int(table.reference_grid.diffusion_m.shape[-1])
    nf=len(table.decay_constants_s_inv)
    if cells.cells!=nc:
        raise ValueError("thermal cells must match cylindrical neutronic cells")
    if (np.asarray(initial_neutrons.flux).shape!=(nc,ng) or
        np.asarray(initial_neutrons.precursors).shape!=(nc,nf) or
        np.asarray(initial_thermal.solid_temperature_k).shape!=(nc,) or
        np.asarray(initial_thermal.hydrogen_temperature_k).shape!=(nc,)):
        raise ValueError("initial neutron/thermal state shapes inconsistent")
    if external_source_per_m3 is not None:
        source=np.asarray(external_source_per_m3,dtype=float)
        if (source.shape!=(nc,ng) or not np.all(np.isfinite(source))
            or np.any(source<0)):
            raise ValueError("neutron source must be nonnegative finite [cell,group]")
    else:
        source=None
    if not (np.all(np.isfinite(initial_neutrons.flux)) and
            np.all(np.isfinite(initial_neutrons.precursors)) and
            np.all(np.asarray(initial_neutrons.flux)>=0) and
            np.all(np.asarray(initial_neutrons.precursors)>=0)):
        raise ValueError("old neutron/precursor populations must be nonnegative")
    if (not np.all(np.isfinite(initial_thermal.solid_temperature_k)) or
        not np.all(np.isfinite(initial_thermal.hydrogen_temperature_k)) or
        np.any(np.asarray(initial_thermal.hydrogen_temperature_k)<=0)):
        raise ValueError("old material/coolant temperatures must be positive finite")
    sample_rz_temperature_tables(
        table,np.asarray(initial_thermal.solid_temperature_k).reshape(nr,nz))
    problem=_ResidualEvaluator(table,initial_neutrons,initial_thermal,cells,
                              dt_s,deposition_from_kinetics,source)
    if len(problem.x0)>cfg.max_unknowns:
        raise ValueError("dense Newton benchmark exceeds configured unknown limit")
    x=problem.x0.copy()
    if not problem.valid(x):
        raise ValueError("invalid initial coupled state")
    current=problem.evaluate(x)
    for iteration in range(cfg.max_iterations+1):
        max_error=float(np.linalg.norm(current.scaled,ord=np.inf))
        if max_error<=cfg.scaled_residual_tolerance:
            ledger=advance_thermal_cells(cells,initial_thermal,
                                         current.deposition,dt_s)
            discrepancy=max(
                float(np.max(np.abs(
                    ledger.state.solid_temperature_k-current.thermal_state.solid_temperature_k))),
                float(np.max(np.abs(
                    ledger.state.hydrogen_temperature_k-current.thermal_state.hydrogen_temperature_k)))
            )
            if discrepancy>1e-7:
                raise ArithmeticError("monolithic thermal result disagrees with conservative energy solver")
            if ledger.relative_energy_residual>1e-9:
                raise ArithmeticError("monolithic thermal first-law residual failed")
            balance_scale=max(1.,float(np.sum(np.abs(problem.y0))))
            if abs(current.neutron_population_balance)>1e-8*balance_scale:
                raise ArithmeticError("neutron and precursor inventory not conserved")
            return MonolithicCoreStep(
                neutron_state=current.state,
                thermal_state=current.thermal_state,
                thermal_ledger=ledger,
                material=current.sample.material,
                iterations=iteration,
                evaluated_states=problem.calls,
                scaled_residual=max_error,
                neutron_population_balance=current.neutron_population_balance,
                material_source_id=current.sample.material.source_id,
                deposition_source_id=current.deposition.source_id,
                synthetic=bool(current.deposition.synthetic or
                               current.sample.source_kind=="synthetic"),
            )
        if iteration==cfg.max_iterations:
            break
        J=_finite_difference_jacobian(problem,x,current.scaled,cfg)
        with warnings.catch_warnings():
            warnings.simplefilter("error",LinAlgWarning)
            try:
                delta=dense_solve(J,-current.scaled,assume_a="gen")
            except (np.linalg.LinAlgError,LinAlgWarning) as exc:
                raise RuntimeError("singular Newton core/thermal Jacobian") from exc
        if not np.all(np.isfinite(delta)):
            raise ArithmeticError("nonfinite nonlinear core Newton direction")
        alpha=1.
        accepted=False
        while alpha>=cfg.min_line_search_fraction:
            candidate=x+alpha*delta
            if problem.valid(candidate):
                residual=problem.evaluate(candidate)
                test=float(np.linalg.norm(residual.scaled,ord=np.inf))
                if test < max_error*(1.-1e-4*alpha):
                    x=candidate
                    current=residual
                    accepted=True
                    break
            alpha/=2.
        if not accepted:
            raise RuntimeError("nonlinear core/thermal line search did not decrease residual")
    raise RuntimeError("monolithic core/thermal solver did not converge")
