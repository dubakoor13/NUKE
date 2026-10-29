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
from .coating_angular_fvm import (
    AngularCoatingMesh, build_angular_coating_mesh,
    advance_angular_coating_enthalpy,
)
from .coating_angular_interfaces import (
    AngularCoatingInterfaces, build_angular_coating_interfaces,
    angular_interface_power, distribute_channel_coating_power,
)
from .resolved_coating_thermal import (
    ResolvedCoatingModel, build_resolved_coating,
    coating_interface_power, advance_coating_enthalpy,
    advance_coating_axial_enthalpy,
)
from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .hydrogen_transient_fvm import (
    ConservativeHydrogenState, HydrogenBoundary, TransientHydrogenModel,
    advance_transient_hydrogen, primitive,
)


@dataclass(frozen=True)
class CoupledCoreState:
    """Fuel temperatures [z,physical fuel cut cell]; fluid conserved [z,19,3]."""
    fuel_temperature_k: np.ndarray
    hydrogen: ConservativeHydrogenState
    coating_temperature_k: np.ndarray | None = None


@dataclass(frozen=True)
class CoupledCoreResult:
    state: CoupledCoreState
    substeps: int
    max_wall_iterations: int
    wall_heat_to_hydrogen_j: np.ndarray
    fuel_deposition_j: float
    hydrogen_deposition_j: float
    hydrogen_deposition_per_channel_j: np.ndarray
    fuel_storage_change_j: float
    hydrogen_total_energy_change_j: float
    hydrogen_net_outlet_energy_j: float
    hydrogen_mass_change_kg: float
    first_law_residual_j: float
    relative_first_law_residual: float
    maximum_hydrogen_mach: float
    coating_storage_change_j: float = 0.
    coating_deposition_j: float = 0.
    fuel_to_coating_j: float = 0.
    coating_to_hydrogen_j: float = 0.
    coating_first_law_residual_j: float = 0.
    coating_axial_interface_j: np.ndarray | None = None
    coating_axial_max_node_residual_j: float = 0.
    coating_angular_face_transfer_j: np.ndarray | None = None
    coating_angular_max_node_residual_j: float = 0.


@dataclass(frozen=True)
class CoupledCoreModel:
    config: NervaConfig
    solid: AxialCutCellModel
    fluid: TransientHydrogenModel
    properties: NonlinearThermalProperties
    solid_density_kg_m3: np.ndarray
    coating: ResolvedCoatingModel | None = None
    angular_coating: AngularCoatingMesh | None = None
    angular_contacts: AngularCoatingInterfaces | None = None

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


def with_resolved_coating(
    model: CoupledCoreModel,
    *,
    density_kg_m3: float,
    heat_capacity: TemperatureTable,
    contact_resistance_m2_k_w: object = 0.,
    axial_conduction_enabled: bool = False,
    angular_sectors: int | None = None,
) -> CoupledCoreModel:
    """Enable coating mass and thermal contact within existing Stage58."""
    coating=build_resolved_coating(
        model.config,model.solid,density_kg_m3=density_kg_m3,
        heat_capacity=heat_capacity,
        contact_resistance_m2_k_w=contact_resistance_m2_k_w,
        axial_conduction_enabled=axial_conduction_enabled,
    )
    if angular_sectors is None:
        angular=None
        contacts=None
    else:
        if not axial_conduction_enabled:
            raise ValueError(
                "angular coating transport includes axial conduction: enable axial_conduction_enabled"
            )
        angular=build_angular_coating_mesh(
            model.config,coating,angular_sectors=angular_sectors,
            density_kg_m3=density_kg_m3)
        contacts=build_angular_coating_interfaces(
            model.config,model.solid,coating,angular)
    return CoupledCoreModel(
        model.config,model.solid,model.fluid,model.properties,
        model.solid_density_kg_m3,coating,angular,contacts,
    )


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
    hydrogen_deposition_w: object = 0.,
    coating_deposition_w: object = 0.,
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

    fuel_deposition_w and hydrogen_deposition_w are TWO DISTINCT nuclear
    energy source ledgers. The latter contributes directly to hydrogen
    total energy, never to wall heat or solid energy, and is evaluated
    exactly once per physical substep. These source arrays must not
    contain energy transferred across the fuel/H2 wall: the wall flux is
    computed by the coupled solver and is internal to the first law.
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
    if np.any(fsrc<0):
        raise ValueError("nuclear fuel deposition must be nonnegative W")
    hsrc=np.asarray(hydrogen_deposition_w,dtype=float)
    if hsrc.ndim==0:
        hsrc=np.full((nz,19),float(hsrc))
    if (hsrc.shape!=(nz,19) or not np.all(np.isfinite(hsrc))
            or np.any(hsrc<0)):
        raise ValueError("hydrogen nuclear deposition must be nonnegative finite W [z,19]")
    coat_model=model.coating
    angular=model.angular_coating
    contacts=model.angular_contacts
    if (angular is None)!=(contacts is None):
        raise ValueError("angular geometry and physical contact must both be present")
    coating_src=np.asarray(coating_deposition_w,dtype=float)
    if coating_src.ndim==0:
        coating_src=np.full((nz,19),float(coating_src))
    if angular is not None and coating_src.shape==(nz,19):
        coating_src=distribute_channel_coating_power(coating_src,angular)
    desired=(nz,19) if angular is None else angular.sector_mass_kg.shape
    if (coating_src.shape!=desired or
            not np.all(np.isfinite(coating_src)) or
            np.any(coating_src<0)):
        raise ValueError("coating nuclear deposition must be nonnegative finite W in physical [z,19,angle] geometry")
    if coat_model is None:
        if initial.coating_temperature_k is not None or np.any(coating_src>0):
            raise ValueError("cannot advance coating heat without resolved coating model")
        tcoat=None
        initial_coating_energy=0.
    else:
        if initial.coating_temperature_k is None:
            raise ValueError("resolved coating temperature state required [z,19]")
        tcoat=np.asarray(initial.coating_temperature_k,dtype=float).copy()
        expected_coating_shape=(
            (nz,19) if angular is None else angular.sector_mass_kg.shape)
        if (tcoat.shape!=expected_coating_shape
                or not np.all(np.isfinite(tcoat))):
            raise ValueError(
                "coating state must have finite temperatures matching resolved [z,19,angle] geometry")
        coat_mass=(
            coat_model.mass_kg if angular is None
            else angular.sector_mass_kg)
        initial_coating_energy=float(np.sum(
            coat_mass*coat_model.heat_capacity.integral(tcoat)))
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
    hydrogen_qtotal=np.zeros((nz,19),dtype=float)
    coating_qtotal=0.
    coating_in_j=0.
    coating_out_j=0.
    coating_axial_j=(np.zeros((nz-1,19)) if coat_model is not None
                     and coat_model.axial_conduction_enabled else None)
    coating_axial_max_node_residual_j=0.
    coating_angular_j=(
        np.zeros(len(angular.edge_left)) if angular is not None else None)
    coating_angular_max_node_residual_j=0.
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
        old_coating=None if coat_model is None else tcoat.copy()
        system=_NonlinearSystem(
            model.solid,model.config,model.properties,rho,
            old_gas.density_kg_m3*model.fluid.volume_m3,
            old_t,old_gas.temperature_k,
            old_gas.temperature_k[0,:],fsrc,
            np.zeros((nz,19)),sub_dt)
        if coat_model is None:
            old_wall_s,old_wall_h=_wall_heat(
                system,old_t,old_gas.temperature_k)
            candidate_h=old_wall_h.copy()
            candidate_s=old_wall_s.copy()
            candidate_coating_in=None
        elif angular is None:
            candidate_s,candidate_coating_in,candidate_h=coating_interface_power(
                coat_model,model.solid,model.properties,
                old_t,old_coating,old_gas.temperature_k)
        else:
            contact_flux=angular_interface_power(
                coat_model,model.solid,model.properties,contacts,
                old_t,old_coating,old_gas.temperature_k)
            candidate_s=contact_flux.fuel_to_coating_by_fuel_w
            candidate_coating_in=contact_flux.fuel_to_coating_by_sector_w
            candidate_angular_out=contact_flux.coating_to_hydrogen_by_sector_w
            candidate_h=contact_flux.coating_to_hydrogen_by_channel_w
        converged=False
        final_solid=None
        final_hydrogen=None
        final_coating=None
        final_axial_diffusion=None
        final_angular_diffusion=None
        final_step=None
        for iteration in range(1,max_wall_iterations+1):
            trial_fluid=advance_transient_hydrogen(
                model.fluid,old_h,dt_s=sub_dt,
                # The Stage57 fluid total-energy equation receives the
                # sum of direct nuclear heat and the *separate* internal
                # wall flux. The solid uses only the wall source opposite.
                wall_heat_w=candidate_h+hsrc,
                boundary=boundary,friction_enabled=friction_enabled)
            trial_solid=_advance_solid_only(
                system,old_t,fsrc-candidate_s,sub_dt,
                max_iterations=max_solid_iterations)
            trial_gas=primitive(model.fluid,trial_fluid.state)
            tm=.5*(old_t+trial_solid)
            hm=.5*(old_gas.temperature_k+trial_gas.temperature_k)
            if coat_model is None:
                trial_coating=None
                target_s,target_h=_wall_heat(system,tm,hm)
                difference=max(
                    float(np.max(np.abs(target_s-candidate_s))),
                    float(np.max(np.abs(target_h-candidate_h))))
            else:
                coating_net_w=(coating_src+candidate_coating_in-candidate_h)
                if coat_model.axial_conduction_enabled:
                    trial_axial_diffusion=advance_coating_axial_enthalpy(
                        coat_model,old_coating,
                        conductivity_w_m_k=model.properties.coating_k_w_m_k,
                        dt_s=sub_dt,net_w=coating_net_w)
                    trial_coating=trial_axial_diffusion.temperature_k
                else:
                    trial_axial_diffusion=None
                    trial_coating=advance_coating_enthalpy(
                        coat_model,old_coating,dt_s=sub_dt,
                        net_w=coating_net_w)
                cm=.5*(old_coating+trial_coating)
                target_s,target_ci,target_h=coating_interface_power(
                    coat_model,model.solid,model.properties,tm,cm,hm)
                difference=max(
                    float(np.max(np.abs(target_s-candidate_s))),
                    float(np.max(np.abs(target_ci-candidate_coating_in))),
                    float(np.max(np.abs(target_h-candidate_h))))
            magnitude=max(1.,float(np.max(np.abs(target_s))),
                          float(np.max(np.abs(target_h))))
            if difference<=wall_relative_tolerance*magnitude:
                converged=True
                final_solid=trial_solid
                final_hydrogen=trial_fluid.state
                final_coating=trial_coating
                final_axial_diffusion=(
                    trial_axial_diffusion if coat_model is not None else None
                )
                final_step=trial_fluid
                max_iters=max(max_iters,iteration)
                break
            # Global heat stays antisymmetric under a common relaxation.
            candidate_s=.5*(candidate_s+target_s)
            candidate_h=.5*(candidate_h+target_h)
            if coat_model is not None:
                candidate_coating_in=.5*(candidate_coating_in+target_ci)
        if not converged:
            raise ArithmeticError(
                f"Stage58 interface heat did not converge in "
                f"{max_wall_iterations} iterations at substep {_sub}")
        t=final_solid
        hydrogen=final_hydrogen
        area+=sub_dt*candidate_h
        if coat_model is not None:
            tcoat=final_coating
            coating_qtotal+=sub_dt*float(np.sum(coating_src))
            coating_in_j+=sub_dt*float(np.sum(candidate_coating_in))
            coating_out_j+=sub_dt*float(np.sum(candidate_h))
            if final_axial_diffusion is not None:
                coating_axial_j+=final_axial_diffusion.axial_interface_j
                coating_axial_max_node_residual_j=max(
                    coating_axial_max_node_residual_j,
                    float(np.max(np.abs(
                        final_axial_diffusion.per_node_first_law_residual_j))))
        qtotal+=sub_dt*float(np.sum(fsrc))
        hydrogen_qtotal+=sub_dt*hsrc
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
    hydrogen_source_j=float(np.sum(hydrogen_qtotal))
    if coat_model is None:
        dcoat=0.
        coat_balance=0.
    else:
        coating_final_energy=float(np.sum(
            coat_model.mass_kg*coat_model.heat_capacity.integral(tcoat)))
        dcoat=coating_final_energy-initial_coating_energy
        coat_balance=dcoat+coating_out_j-coating_in_j-coating_qtotal
    residual=ds+dh+dcoat+outflow_net-qtotal-hydrogen_source_j-coating_qtotal
    scale=max(1.,abs(ds),abs(dh),abs(dcoat),abs(outflow_net),abs(qtotal),
              abs(hydrogen_source_j),abs(coating_qtotal))
    # Energy is obtained by subtracting two large state inventories;
    # permit only a separately tracked floating-point cancellation floor.
    scale_old=(abs(solid_energy)+abs(old_solid_energy)
               +abs(gas_energy)+abs(initial_fluid_energy))
    if coat_model is not None:
        scale_old+=abs(initial_coating_energy)+abs(coating_final_energy)
    bound=max(4e-8*scale,2e-11*scale_old)
    if abs(residual)>bound or abs(coat_balance)>bound:
        raise ArithmeticError(
            f"combined fuel/H2 first-law residual={residual:.8g} J "
            f"exceeds numerical bound {bound:.8g} J")
    return CoupledCoreResult(
        state=CoupledCoreState(t,hydrogen,tcoat),
        substeps=count,max_wall_iterations=max_iters,
        wall_heat_to_hydrogen_j=area,
        fuel_deposition_j=qtotal,
        hydrogen_deposition_j=hydrogen_source_j,
        hydrogen_deposition_per_channel_j=hydrogen_qtotal,
        fuel_storage_change_j=ds,
        hydrogen_total_energy_change_j=dh,
        hydrogen_net_outlet_energy_j=outflow_net,
        hydrogen_mass_change_kg=gas_mass-initial_fluid_mass,
        first_law_residual_j=residual,
        relative_first_law_residual=abs(residual)/scale,
        maximum_hydrogen_mach=float(np.max(
            primitive(model.fluid,hydrogen).mach)),
        coating_storage_change_j=dcoat,
        coating_deposition_j=coating_qtotal,
        fuel_to_coating_j=coating_in_j,
        coating_to_hydrogen_j=coating_out_j,
        coating_first_law_residual_j=coat_balance,
        coating_axial_interface_j=coating_axial_j,
        coating_axial_max_node_residual_j=coating_axial_max_node_residual_j,
    )
