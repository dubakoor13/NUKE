"""Stage58: one conservative transient fuel / hydrogen thermal-hydraulic state.

Couples the Stage53/55 exact fuel cut-cell geometry and temperature-dependent
solid conduction to Stage57 compressible 19-channel H2 mass/momentum/TOTAL
energy. H2 is advanced ONLY by Stage57; Stage55 H2 enthalpy unknowns are
never integrated here. Each converged substep transfers the identical wall
power, with opposite signs, into the solid and fluid finite volumes.

Partitioned implicit-solid / SSPRK2-fluid fixed-point midpoint wall exchange.
Synthetic material tables remain synthetic: NO NERVA neutronic power,
historical hot material qualification, real-gas EOS or validated film law is
inferred from this computational conservation demonstration.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from .config import NervaConfig
from .fuel_cutcell_axial_transport import AxialCutCellModel
from .fuel_cutcell_nonlinear_enthalpy import (
    NonlinearThermalProperties, _NonlinearSystem,
)
from .hydrogen_transient_fvm import (
    ConservativeHydrogenState, HydrogenBoundary, TransientHydrogenModel,
    advance_transient_hydrogen, primitive,
)


@dataclass(frozen=True)
class CoupledCoreState:
    """Fuel temperatures [z,physical fuel cut cell]; fluid conserved [z,19,3]."""
    fuel_temperature_k: np.ndarray
    hydrogen: ConservativeHydrogenState


@dataclass(frozen=True)
class CoupledCoreResult:
    state: CoupledCoreState
    substeps: int
    max_wall_iterations: int
    wall_heat_to_hydrogen_j: np.ndarray
    fuel_deposition_j: float
    fuel_storage_change_j: float
    hydrogen_total_energy_change_j: float
    hydrogen_net_outlet_energy_j: float
    hydrogen_mass_change_kg: float
    first_law_residual_j: float
    relative_first_law_residual: float
    maximum_hydrogen_mach: float


@dataclass(frozen=True)
class CoupledCoreModel:
    config: NervaConfig
    solid: AxialCutCellModel
    fluid: TransientHydrogenModel
    properties: NonlinearThermalProperties
    solid_density_kg_m3: np.ndarray

    @property
    def nzone(self) -> int:
        return self.fluid.nzone


def build_coupled_core_model(
    config: NervaConfig,
    solid: AxialCutCellModel,
    fluid: TransientHydrogenModel,
    properties: NonlinearThermalProperties,
    *, solid_density_kg_m3: object,
) -> CoupledCoreModel:
    """Fail closed on geometry, H2 thermodynamics and mesh inconsistencies."""
    config.validate()
    ns=solid.axial_zones
    nf=solid.transverse.n_fuel
    if ns!=fluid.nzone or ns<1:
        raise ValueError("fuel/H2 axial volume count mismatch")
    if (not np.allclose(solid.axial_edges_m,fluid.axial_edges_m,
                        rtol=0,atol=2e-12)
            or abs(solid.transverse.axial_length_m-
                   config.active_length_cm/100)>2e-12
            or abs(fluid.diameter_m-
                   config.coolant_bore_diameter_cm/100)>2e-12):
        raise ValueError("fuel/H2 geometry or axial edges incompatible")
    # Must not advance H2 with one cp table and invert the fluid total
    # energy with another. Fluid EOS is the only authoritative H2 cp.
    gas=fluid.eos.cp_j_kg_k
    hcp=properties.hydrogen_cp_j_kg_k
    if (not np.array_equal(gas.temperature_k,hcp.temperature_k)
            or not np.array_equal(gas.values,hcp.values)):
        raise ValueError("Stage55/Stage57 hydrogen cp tables disagree")
    rho=np.asarray(solid_density_kg_m3,dtype=float)
    if rho.ndim==0:
        rho=np.full((ns,nf),float(rho))
    if rho.shape!=(ns,nf) or not np.all(np.isfinite(rho)) or np.any(rho<=0):
        raise ValueError("solid density requires positive [z,fuel] values")
    if (solid.fuel_hydrogen_exchange_w_k.shape != (ns*nf,ns*19)
            or fluid.volume_m3.shape!=(ns,19)):
        raise ValueError("invalid H2 wall coupling layout")
    return CoupledCoreModel(config,solid,fluid,properties,rho.copy())


def _wall_heat(
    system: _NonlinearSystem,
    tsolid: np.ndarray,
    thydrogen: np.ndarray,
) -> tuple[np.ndarray,np.ndarray]:
    """Identical interface powers on both sides, in watts per volume."""
    _,w=system._operators(tsolid,thydrogen)
    sf=tsolid.ravel()
    hf=thydrogen.ravel()
    fuel=(np.asarray(w.sum(axis=1)).ravel()*sf-w@hf)
    fluid=(w.T@sf-np.asarray(w.sum(axis=0)).ravel()*hf)
    return fuel.reshape(tsolid.shape),fluid.reshape(thydrogen.shape)


def _advance_solid_only(
    system: _NonlinearSystem,
    old_temperature: np.ndarray,
    qsolid: np.ndarray,
    dt: float,
    *,
    max_iterations: int,
) -> np.ndarray:
    """Implicit nonlinear solid enthalpy, with externally fixed wall watts."""
    p=system.p
    mass=system.mf
    h_old=p.fuel_cp_j_kg_k.integral(old_temperature)
    target=np.asarray(qsolid,dtype=float).ravel()
    x=old_temperature.copy().ravel()
    shape=old_temperature.shape
    tolerance=2e-8+1e-10*max(1.,float(np.max(np.abs(target))))

    def evaluate(v: np.ndarray, jacobian: bool):
        temp=v.reshape(shape)
        h=p.fuel_cp_j_kg_k.integral(temp)
        lap,_=system._operators(temp,system.old_h)
        f=(mass*(h-h_old)/dt).ravel()+lap@v-target
        if not jacobian:
            return f,None
        cp=p.fuel_cp_j_kg_k.at(temp)
        j=lap+diags((mass*cp/dt).ravel(),format="csr")
        return f,j

    for _ in range(max_iterations):
        r,j=evaluate(x,True)
        norm=float(np.max(np.abs(r)))
        if norm<=tolerance:
            return x.reshape(shape)
        dx=np.asarray(spsolve(j.tocsc(),-r),dtype=float)
        if dx.shape!=x.shape or not np.all(np.isfinite(dx)):
            raise ArithmeticError("singular/nonfinite nonlinear solid solve")
        step=1.
        for _ in range(30):
            trial=x+step*dx
            try:
                rt,_=evaluate(trial,False)
                new_norm=float(np.max(np.abs(rt)))
            except ValueError:
                step*=0.5
                continue
            if new_norm<norm*(1-1e-5*step) or new_norm<=tolerance:
                x=trial
                break
            step*=0.5
        else:
            raise ArithmeticError("fuel solid nonlinear line search failed")
    r,_=evaluate(x,False)
    if np.max(np.abs(r))>tolerance:
        raise ArithmeticError("fuel solid nonlinear step did not converge")
    return x.reshape(shape)


def advance_coupled_core(
    model: CoupledCoreModel,
    initial: CoupledCoreState,
    *, dt_s: float,
    fuel_deposition_w: object,
    boundary: HydrogenBoundary = HydrogenBoundary(),
    max_wall_iterations: int = 32,
    wall_relative_tolerance: float = 1e-9,
    max_solid_iterations: int = 16,
    max_substeps: int = 10000,
    friction_enabled: bool = True,
) -> CoupledCoreResult:
    """One common advanced fuel/H2 state with equal-and-opposite wall energy.

    SSPRK2 Stage57 H2 is the *only* fluid integrator. Each acoustic-limited
    substep iterates one midpoint interface power with a fully implicit
    solid enthalpy solve. The H2 and solid are always solved from the SAME
    substep initial state until the shared wall power converges.
    """
    if (not math.isfinite(dt_s) or dt_s<=0
            or not math.isfinite(wall_relative_tolerance)
            or not 0<wall_relative_tolerance<=1e-3
            or not 1<=max_wall_iterations<=200
            or not 1<=max_solid_iterations<=100
            or not 1<=max_substeps<=100000):
        raise ValueError("invalid coupled thermal time/iteration controls")
    nz=model.nzone
    nf=model.solid.transverse.n_fuel
    t0=np.asarray(initial.fuel_temperature_k,dtype=float)
    if t0.shape!=(nz,nf) or not np.all(np.isfinite(t0)):
        raise ValueError("fuel temperature must be finite [z,cutcell]")
    model.properties.fuel_cp_j_kg_k.at(t0)
    fsrc=np.asarray(fuel_deposition_w,dtype=float)
    if fsrc.shape!=(nz,nf) or not np.all(np.isfinite(fsrc)):
        raise ValueError("fuel deposition must be finite W [z,cutcell]")
    gas0=primitive(model.fluid,initial.hydrogen)
    pgas=gas0
    speed=float(np.max(
        (np.abs(gas0.velocity_m_s)+gas0.sound_speed_m_s)
        /model.fluid.dz_m[:,None]))
    # Keep a margin below Stage57 CFL at both RK stages; explicit advective
    # substeps are intentionally limited, not an invented implicit flow solve.
    count=max(1,int(math.ceil(dt_s*speed/(.70*model.fluid.maximum_cfl))))
    if count>max_substeps:
        raise ValueError("fluid acoustic subcycling count exceeds limit")
    sub_dt=dt_s/count
    t=t0.copy()
    hydrogen=ConservativeHydrogenState(
        np.asarray(initial.hydrogen.conserved,dtype=float).copy())
    rho=model.solid_density_kg_m3
    mf=rho*model.solid.solid_volume_m3
    area=np.zeros((nz,19))
    qtotal=0.
    outflow_net=0.
    max_iters=0
    initial_fluid_energy=float(np.sum(
        initial.hydrogen.conserved[...,2]*model.fluid.volume_m3))
    initial_fluid_mass=float(np.sum(
        initial.hydrogen.conserved[...,0]*model.fluid.volume_m3))
    initial_solid_h=model.properties.fuel_cp_j_kg_k.integral(t0)
    old_solid_energy=float(np.sum(mf*initial_solid_h))

    for _sub in range(count):
        old_t=t.copy()
        old_h=hydrogen
        old_gas=primitive(model.fluid,old_h)
        system=_NonlinearSystem(
            model.solid,model.config,model.properties,rho,
            old_gas.density_kg_m3*model.fluid.volume_m3,
            old_t,old_gas.temperature_k,
            old_gas.temperature_k[0,:],fsrc,
            np.zeros((nz,19)),sub_dt)
        old_wall_s,old_wall_h=_wall_heat(
            system,old_t,old_gas.temperature_k)
        candidate_h=old_wall_h.copy()
        candidate_s=old_wall_s.copy()
        converged=False
        final_solid=None
        final_hydrogen=None
        final_step=None
        for iteration in range(1,max_wall_iterations+1):
            trial_fluid=advance_transient_hydrogen(
                model.fluid,old_h,dt_s=sub_dt,
                wall_heat_w=candidate_h,
                boundary=boundary,friction_enabled=friction_enabled)
            trial_solid=_advance_solid_only(
                system,old_t,fsrc-candidate_s,sub_dt,
                max_iterations=max_solid_iterations)
            trial_gas=primitive(model.fluid,trial_fluid.state)
            tm=.5*(old_t+trial_solid)
            hm=.5*(old_gas.temperature_k+trial_gas.temperature_k)
            target_s,target_h=_wall_heat(system,tm,hm)
            difference=max(float(np.max(np.abs(target_s-candidate_s))),
                           float(np.max(np.abs(target_h-candidate_h))))
            magnitude=max(1.,float(np.max(np.abs(target_s))),
                          float(np.max(np.abs(target_h))))
            if difference<=wall_relative_tolerance*magnitude:
                converged=True
                final_solid=trial_solid
                final_hydrogen=trial_fluid.state
                final_step=trial_fluid
                max_iters=max(max_iters,iteration)
                break
            # Global heat stays antisymmetric under a common relaxation.
            candidate_s=.5*(candidate_s+target_s)
            candidate_h=.5*(candidate_h+target_h)
        if not converged:
            raise ArithmeticError(
                f"Stage58 interface heat did not converge in "
                f"{max_wall_iterations} iterations at substep {_sub}")
        t=final_solid
        hydrogen=final_hydrogen
        area+=sub_dt*candidate_h
        qtotal+=sub_dt*float(np.sum(fsrc))
        outflow_net+=sub_dt*float(np.sum(
            final_step.outlet_energy_flux_w-final_step.inlet_energy_flux_w))

    solid_energy=float(np.sum(
        mf*model.properties.fuel_cp_j_kg_k.integral(t)))
    gas_energy=float(np.sum(
        hydrogen.conserved[...,2]*model.fluid.volume_m3))
    gas_mass=float(np.sum(
        hydrogen.conserved[...,0]*model.fluid.volume_m3))
    ds=solid_energy-old_solid_energy
    dh=gas_energy-initial_fluid_energy
    residual=ds+dh+outflow_net-qtotal
    scale=max(1.,abs(ds),abs(dh),abs(outflow_net),abs(qtotal))
    # Energy is obtained by subtracting two large state inventories;
    # permit only a separately tracked floating-point cancellation floor.
    scale_old=(abs(solid_energy)+abs(old_solid_energy)
               +abs(gas_energy)+abs(initial_fluid_energy))
    bound=max(4e-8*scale,2e-11*scale_old)
    if abs(residual)>bound:
        raise ArithmeticError(
            f"combined fuel/H2 first-law residual={residual:.8g} J "
            f"exceeds numerical bound {bound:.8g} J")
    return CoupledCoreResult(
        state=CoupledCoreState(t,hydrogen),
        substeps=count,max_wall_iterations=max_iters,
        wall_heat_to_hydrogen_j=area,
        fuel_deposition_j=qtotal,
        fuel_storage_change_j=ds,
        hydrogen_total_energy_change_j=dh,
        hydrogen_net_outlet_energy_j=outflow_net,
        hydrogen_mass_change_kg=gas_mass-initial_fluid_mass,
        first_law_residual_j=residual,
        relative_first_law_residual=abs(residual)/scale,
        maximum_hydrogen_mach=float(np.max(
            primitive(model.fluid,hydrogen).mach)),
    )
