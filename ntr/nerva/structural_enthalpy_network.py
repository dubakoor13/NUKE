"""Stage69: conservative enthalpy evolution of WHOLE-CORE nonfuel structures.

The Stage68 independent native-OpenMC material ledger explicitly accounts
for deposited energy in seven nonfuel/non-H2 material categories. This
module integrates their actual *separate* finite heat capacities, strictly
bounded tabulated cp(T), inter-material interface heat transfer, and
optional externally declared boundary-reservoir heat flux.

A node is a WHOLE-CORE aggregate of one material category. Its mass and
conductances are REQUIRED inputs, not inferred from the OpenMC tally or
invented NERVA geometry. Inter-material conductance entries MUST be
symmetric and nonnegative; heat moved across an internal edge is added to
one node and subtracted from the other exactly. The solver performs an
implicit nonlinearly enthalpy-conservative solve, not constant-cp dT.

This is an independently testable STRUCTURAL thermal subsystem, not a
replacement for the existing Stage58 fuel/H2 solver. No unknown contact
conductance is invented between these whole-core aggregates and Stage58's
SINGLE fuel element. A complete physical reactor energy model must later
use resolved, identity-matched element/structure interfaces; source
allocation here cannot establish that coupling or qualify hot materials.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import math

import numpy as np
from scipy.linalg import solve

from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .openmc_full_core_energy_ledger import (
    CoreDepositedEnergyLedger, MATERIAL_ORDER,
)

STRUCTURAL_NAMES = tuple(
    name for name in MATERIAL_ORDER if name not in ("fuel", "hydrogen")
)
N_STRUCT = len(STRUCTURAL_NAMES)


@dataclass(frozen=True)
class StructuralThermalNetwork:
    """Seven exact material-identity aggregates with explicit properties.

    `mass_kg`: one total material mass per whole-core material category.
    `intermaterial_w_k[i,j]`: symmetric pairwise heat conductance.
    `bath_w_k[i]`: optional W/K conductance to separately specified bath.
    No default contact with the Stage58 fuel element or H2 is implied.
    """
    names: tuple[str, ...]
    mass_kg: np.ndarray
    cp_j_kg_k: tuple[TemperatureTable, ...]
    intermaterial_w_k: np.ndarray
    bath_w_k: np.ndarray
    laplacian_w_k: np.ndarray


@dataclass(frozen=True)
class StructuralEnergyAllocation:
    """Whole-core material ledger partition; NOT a per-element power map."""
    structural_power_w: np.ndarray
    fuel_power_w: float
    hydrogen_power_w: float
    whole_core_deposition_w: float
    allocation_residual_w: float
    source_statepoint_sha256: str
    source_rate_basis: str


@dataclass(frozen=True)
class StructuralThermalStep:
    """Exact integral-cp storage and antisymmetric intermaterial heat."""
    temperature_k: np.ndarray
    stored_energy_change_j: np.ndarray
    deposited_energy_j: np.ndarray
    internal_net_outflow_j: np.ndarray
    bath_net_outflow_j: np.ndarray
    per_material_first_law_residual_j: np.ndarray
    total_stored_energy_change_j: float
    total_deposited_energy_j: float
    total_bath_export_j: float
    total_first_law_residual_j: float
    relative_first_law_residual: float
    iterations: int


def build_structural_thermal_network(
    *,
    mass_kg: object,
    cp_j_kg_k: Mapping[str, TemperatureTable],
    intermaterial_w_k: object,
    bath_w_k: object = 0.,
) -> StructuralThermalNetwork:
    """Construct exactly seven material nodes and conservative contacts."""
    masses=np.asarray(mass_kg,dtype=float)
    if (masses.shape!=(N_STRUCT,) or not np.all(np.isfinite(masses))
            or np.any(masses<=0)):
        raise ValueError("seven positive whole-core structural masses [kg] required")
    if set(cp_j_kg_k)!=set(STRUCTURAL_NAMES):
        raise ValueError("must provide exact seven structural cp tables by material identity")
    props=tuple(cp_j_kg_k[k] for k in STRUCTURAL_NAMES)
    if any(not isinstance(cp,TemperatureTable) for cp in props):
        raise ValueError("structural cp properties must be bounded TemperatureTables")
    g=np.asarray(intermaterial_w_k,dtype=float)
    if (g.shape!=(N_STRUCT,N_STRUCT)
            or not np.all(np.isfinite(g)) or np.any(g<0)
            or not np.allclose(g,g.T,rtol=0,atol=1e-12)
            or not np.allclose(np.diag(g),0.,rtol=0,atol=1e-12)):
        raise ValueError(
            "structural interfaces need symmetric nonnegative W/K and zero diagonal")
    # Exact symmetric topology rather than near-symmetric input rounding.
    g=(g+g.T)*.5
    np.fill_diagonal(g,0.)
    bath=np.asarray(bath_w_k,dtype=float)
    if bath.ndim==0:
        bath=np.full(N_STRUCT,float(bath))
    if (bath.shape!=(N_STRUCT,) or not np.all(np.isfinite(bath))
            or np.any(bath<0)):
        raise ValueError("structural bath conductance requires nonnegative W/K")
    lap=np.diag(np.sum(g,axis=1))-g
    if not np.allclose(lap@np.ones(N_STRUCT),0.,rtol=0,atol=2e-12):
        raise ArithmeticError("internal structural heat exchange is not conservative")
    return StructuralThermalNetwork(
        names=STRUCTURAL_NAMES,mass_kg=masses.copy(),
        cp_j_kg_k=props,intermaterial_w_k=g.copy(),
        bath_w_k=bath.copy(),laplacian_w_k=lap,
    )


def allocate_full_core_structural_heating(
    ledger: CoreDepositedEnergyLedger,
) -> StructuralEnergyAllocation:
    """Partition Stage68 whole-core watts without scaling to ONE fuel hex.

    The NERVA whole-core tally categories and physical normalization must
    have been independently qualified before this allocation can represent
    a physical reactor calculation. Allocation itself only conserves
    explicitly provided watts.
    """
    watts=dict(ledger.material_power_w)
    if set(watts)!=set(MATERIAL_ORDER):
        raise ValueError("complete Stage68 whole-core material heating identities required")
    values=np.array([watts[name] for name in MATERIAL_ORDER],dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values<0):
        raise ValueError("whole-core material deposited watts must be nonnegative")
    if (not math.isfinite(ledger.material_total_w)
            or ledger.material_total_w<0):
        raise ValueError("whole-core deposited-energy total invalid")
    structural=np.array([watts[name] for name in STRUCTURAL_NAMES])
    total=float(np.sum(values))
    err=total-ledger.material_total_w
    if abs(err)>max(1e-11,1e-12*max(1.,total)):
        raise ArithmeticError("structural + fuel + H2 allocation does not close")
    return StructuralEnergyAllocation(
        structural_power_w=structural,
        fuel_power_w=float(watts["fuel"]),
        hydrogen_power_w=float(watts["hydrogen"]),
        whole_core_deposition_w=ledger.material_total_w,
        allocation_residual_w=err,
        source_statepoint_sha256=ledger.statepoint_sha256,
        source_rate_basis=ledger.source_rate_basis,
    )


def advance_structural_enthalpy(
    model: StructuralThermalNetwork,
    old_temperature_k: object,
    *,
    dt_s: float,
    deposited_power_w: object,
    bath_temperature_k: object | None = None,
    max_iterations: int = 36,
    relative_tolerance: float = 2e-11,
) -> StructuralThermalStep:
    """Implicit Euler conservative enthalpy update, seven materials.

    For node i:
      m_i (h_i(Tnew)-h_i(Told))/dt
       + sum_j G_ij*(Tnew_i - Tnew_j)
       + G_bath_i*(Tnew_i-Tbath_i) = deposited_i

    h_i(T) is the exact integral of the caller's piecewise-linear cp(T).
    The seven-node symmetric conduction graph is dissipative; positive
    heat capacity makes the Newton Jacobian SPD. Every failed/missing
    property, nonphysical state and unconverged step is rejected.
    """
    if (not math.isfinite(dt_s) or dt_s<=0
            or not isinstance(max_iterations,int) or max_iterations<1
            or not math.isfinite(relative_tolerance) or relative_tolerance<=0):
        raise ValueError("invalid structural step size or numerical tolerances")
    t0=np.asarray(old_temperature_k,dtype=float)
    power=np.asarray(deposited_power_w,dtype=float)
    if (t0.shape!=(N_STRUCT,) or not np.all(np.isfinite(t0)) or
            power.shape!=(N_STRUCT,) or not np.all(np.isfinite(power))
            or np.any(power<0)):
        raise ValueError("structural temperature [K] or deposited watts [7] invalid")
    if bath_temperature_k is None:
        if np.any(model.bath_w_k>0):
            raise ValueError("bath temperature required when bath conductance is nonzero")
        bath=t0.copy()
    else:
        bath=np.asarray(bath_temperature_k,dtype=float)
        if bath.ndim==0:
            bath=np.full(N_STRUCT,float(bath))
        if bath.shape!=(N_STRUCT,) or not np.all(np.isfinite(bath)):
            raise ValueError("bath temperatures require finite K [7]")
    # Bounds are table-specific, including bath temperature where used.
    h0=np.empty(N_STRUCT)
    for i,cp in enumerate(model.cp_j_kg_k):
        h0[i]=float(cp.integral(t0[i]))
        if model.bath_w_k[i]>0:
            cp.at(bath[i])

    mass=model.mass_kg
    lap=model.laplacian_w_k
    diag_bath=np.diag(model.bath_w_k)
    conductance=lap+diag_bath
    rhs=power+model.bath_w_k*bath
    def evaluate(t: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        h=np.array([
            float(p.integral(float(t[i])))
            for i,p in enumerate(model.cp_j_kg_k)
        ])
        cp=np.array([
            float(p.at(float(t[i])))
            for i,p in enumerate(model.cp_j_kg_k)
        ])
        residual=mass*(h-h0)/dt_s+conductance@t-rhs
        jac=conductance+np.diag(mass*cp/dt_s)
        return residual,jac

    t=t0.copy()
    tolerance=1e-9+relative_tolerance*max(
        1.,float(np.max(np.abs(power))),
        float(np.max(np.abs(conductance@t0))),
        float(np.max(np.abs(model.bath_w_k*bath))),
    )
    count=0
    for iteration in range(max_iterations):
        residual,jac=evaluate(t)
        norm=float(np.max(np.abs(residual)))
        if norm<tolerance:
            count=iteration
            break
        step=solve(jac,-residual,assume_a="pos",check_finite=True)
        if not np.all(np.isfinite(step)):
            raise ArithmeticError("invalid structural nonlinear correction")
        damping=1.
        for _ in range(45):
            test=t+damping*step
            try:
                testres,_=evaluate(test)
            except ValueError:
                damping*=.5
                continue
            if (np.max(np.abs(testres))<norm*(1-1e-5*damping)
                    or np.max(np.abs(testres))<tolerance):
                t=test
                break
            damping*=.5
        else:
            raise ArithmeticError("structural enthalpy nonlinear line search failed")
    else:
        raise ArithmeticError("structural enthalpy iteration did not converge")

    storage=np.array([
        mass[i]*(float(model.cp_j_kg_k[i].integral(t[i]))-h0[i])
        for i in range(N_STRUCT)
    ])
    deposited=dt_s*power
    internal=dt_s*(lap@t)
    bath_flux=dt_s*model.bath_w_k*(t-bath)
    rowres=storage+internal+bath_flux-deposited
    totalres=float(np.sum(storage)+np.sum(bath_flux)-np.sum(deposited))
    scale=max(1.,float(np.sum(np.abs(storage))),
              float(np.sum(np.abs(bath_flux))),
              float(np.sum(deposited)))
    if abs(float(np.sum(internal)))>2e-10*scale:
        raise ArithmeticError("structural internal heat network lost energy")
    if float(np.max(np.abs(rowres)))>tolerance*dt_s*2:
        raise ArithmeticError("structural material energy first law failed")
    return StructuralThermalStep(
        temperature_k=t,
        stored_energy_change_j=storage,
        deposited_energy_j=deposited,
        internal_net_outflow_j=internal,
        bath_net_outflow_j=bath_flux,
        per_material_first_law_residual_j=rowres,
        total_stored_energy_change_j=float(np.sum(storage)),
        total_deposited_energy_j=float(np.sum(deposited)),
        total_bath_export_j=float(np.sum(bath_flux)),
        total_first_law_residual_j=totalres,
        relative_first_law_residual=abs(totalres)/scale,
        iterations=count,
    )
