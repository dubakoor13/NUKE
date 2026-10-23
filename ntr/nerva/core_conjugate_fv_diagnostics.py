"""Independent per-control-volume audit for NUKE's conjugate core FV solver.

Reconstruct solid radial/axial heat fluxes and gas enthalpy changes directly
from the final temperature field, not the sparse solver's assembled matrix.

This verifies *discretized* energy conservation and does NOT independently
verify material-property correlations, fuel geometry, evaluated nuclear KERMA,
reactor criticality, neutron statistics or NASA reactor operating data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .core_conjugate_fv import (
    CoreSectorBoundary, CoreSectorGrid, CoreSectorResult,
    CoreSectorState, CoreSolidLaw, _gas_enthalpy, _nuclear_cells,
)


@dataclass(frozen=True)
class CoreCellEnergyAudit:
    radial_face_flux_outward_w: np.ndarray      # [z, nr+1], +r
    axial_face_flux_upward_w: np.ndarray        # [nz+1, r], +z
    wall_heat_to_hydrogen_w: np.ndarray         # [z]
    solid_storage_rate_w: np.ndarray            # [z, r]
    gas_storage_rate_w: np.ndarray              # [z]
    solid_energy_residual_w: np.ndarray         # [z, r]
    gas_energy_residual_w: np.ndarray           # [z]
    global_energy_residual_w: float
    max_absolute_local_residual_w: float
    max_local_residual_scaled: float
    global_residual_scaled: float
    source_is_physics_qualified: bool = False


def audit_core_sector_discrete_energy(
    grid: CoreSectorGrid,
    previous: CoreSectorState,
    result: CoreSectorResult,
    boundary: CoreSectorBoundary,
    fuel: CoreSolidLaw,
    coating: CoreSolidLaw,
    nuclear_solid_power_w: np.ndarray,
    *,
    dt_s: float | None,
    fuel_coating_contact_resistance_m2_k_w: float = 0.0,
) -> CoreCellEnergyAudit:
    """Audit every solid and gas cell using signed physical heat fluxes.

    For each solid cell: storage + outward_face_fluxes = nuclear_heat.
    For each H2 cell: storage + advected_enthalpy_difference =
                      direct_nuclear_heat + solid_to_hydrogen_heat.

    The resulting residuals have watts as units; time stepping produces
    storage-energy increments divided by dt, not a mislabeled joule rate.
    """
    grid.validate()
    fuel.validate()
    coating.validate()
    nz,nr=grid.shape
    boundary.validate(nz)
    if (dt_s is not None and (not math.isfinite(dt_s) or dt_s<=0)
            or not math.isfinite(fuel_coating_contact_resistance_m2_k_w)
            or fuel_coating_contact_resistance_m2_k_w<0):
        raise ValueError("invalid FV discrete audit parameters")
    before_s=np.asarray(previous.solid_temperature_k,dtype=float)
    after_s=np.asarray(result.state.solid_temperature_k,dtype=float)
    before_g=np.asarray(previous.hydrogen_temperature_k,dtype=float)
    after_g=np.asarray(result.state.hydrogen_temperature_k,dtype=float)
    if (before_s.shape!=(nz,nr) or after_s.shape!=(nz,nr)
            or before_g.shape!=(nz,) or after_g.shape!=(nz,)
            or any(not np.isfinite(a).all() or np.any(a<=0) for a in
                   (before_s,after_s,before_g,after_g))):
        raise ValueError("nonphysical recorded core thermal states")

    qsol=_nuclear_cells(grid,nuclear_solid_power_w)
    qh2=np.asarray(boundary.direct_hydrogen_heating_w,dtype=float)
    r=np.asarray(grid.r_edges_m,dtype=float)
    z=np.asarray(grid.z_edges_m,dtype=float)
    dz=np.diff(z)
    rc=np.sqrt(0.5*(r[1:]**2+r[:-1]**2))
    area=np.pi*np.diff(r*r)
    v=grid.volumes_m3
    laws=[coating if j<grid.n_coating_cells else fuel for j in range(nr)]
    k=np.empty((nz,nr))
    for j,law in enumerate(laws):
        k[:,j]=[law.conductivity(float(t)) for t in after_s[:,j]]

    # Radial face flux is positive from inner cell into outer cell. At the
    # coolant bore this means NEGATIVE of solid-to-hydrogen heat flow.
    radial=np.zeros((nz,nr+1),dtype=float)
    wall=np.zeros(nz,dtype=float)
    h=np.asarray(boundary.convection_w_m2_k,dtype=float)
    for i in range(nz):
        wall_resistance=(math.log(rc[0]/r[0]) /
                         (2*math.pi*k[i,0]*dz[i])+
                         1/(h[i]*2*math.pi*r[0]*dz[i]))
        wall[i]=(after_s[i,0]-after_g[i])/wall_resistance
        radial[i,0]=-wall[i]
        for j in range(1,nr):
            face=r[j]
            resistance=(math.log(face/rc[j-1])/k[i,j-1]
                        +math.log(rc[j]/face)/k[i,j])
            if j==grid.n_coating_cells:
                resistance+=fuel_coating_contact_resistance_m2_k_w/face
            radial[i,j]=((after_s[i,j-1]-after_s[i,j])*
                         2*math.pi*dz[i]/resistance)

    axial=np.zeros((nz+1,nr),dtype=float)
    for i in range(1,nz):
        for j in range(nr):
            resistance=(0.5*dz[i-1]/k[i-1,j]+0.5*dz[i]/k[i,j])
            axial[i,j]=(after_s[i-1,j]-after_s[i,j])*area[j]/resistance

    store_s=np.zeros((nz,nr),dtype=float)
    store_g=np.zeros(nz,dtype=float)
    if dt_s is not None:
        for j,law in enumerate(laws):
            for i in range(nz):
                store_s[i,j]=(v[i,j]*law.density_kg_m3*
                    law.enthalpy_change(
                        float(before_s[i,j]),float(after_s[i,j]))/dt_s)
        mass=np.asarray(boundary.gas_mass_kg,dtype=float)
        for i in range(nz):
            store_g[i]=mass[i]*(
                _gas_enthalpy(float(after_g[i]),boundary)-
                _gas_enthalpy(float(before_g[i]),boundary))/dt_s

    solid_net=(radial[:,1:]-radial[:,:-1]
              +axial[1:,:]-axial[:-1,:])
    solid_res=store_s+solid_net-qsol
    adv=np.empty(nz,dtype=float)
    for i in range(nz):
        t_up=(boundary.inlet_temperature_k if i==0 else float(after_g[i-1]))
        adv[i]=boundary.mass_flow_kg_s*(
            _gas_enthalpy(float(after_g[i]),boundary)-
            _gas_enthalpy(t_up,boundary))
    gas_res=store_g+adv-wall-qh2
    glob=float(np.sum(solid_res,dtype=np.float64)+np.sum(gas_res,dtype=np.float64))
    # Avoid scale based on the huge absolute enthalpy reference; it could
    # make a wrong deposited-power partition look deceptively accurate.
    per_s=np.maximum.reduce((np.ones_like(qsol),np.abs(qsol),
                              np.abs(store_s),
                              np.abs(radial[:,1:]),np.abs(radial[:,:-1]),
                              np.abs(axial[1:,:]),np.abs(axial[:-1,:])))
    per_g=np.maximum.reduce((np.ones_like(qh2),np.abs(qh2),
                              np.abs(store_g),np.abs(adv),np.abs(wall)))
    scaled=max(float(np.max(np.abs(solid_res)/per_s)),
               float(np.max(np.abs(gas_res)/per_g)))
    gross=(float(np.sum(np.abs(qsol)))+float(np.sum(np.abs(qh2)))+
           float(np.sum(np.abs(store_s)))+float(np.sum(np.abs(store_g)))+
           abs(boundary.mass_flow_kg_s*
               (_gas_enthalpy(float(after_g[-1]),boundary)-
                _gas_enthalpy(boundary.inlet_temperature_k,boundary))))
    global_scaled=abs(glob)/max(1.0,gross)
    wall_original=np.asarray(result.convected_wall_heating_w,dtype=float)
    if (wall_original.shape!=(nz,) or not np.allclose(
            wall,wall_original,rtol=1e-7,atol=1e-7)):
        raise ValueError("independent convection flux disagrees with solver ledger")

    return CoreCellEnergyAudit(
        radial_face_flux_outward_w=radial,
        axial_face_flux_upward_w=axial,
        wall_heat_to_hydrogen_w=wall,
        solid_storage_rate_w=store_s,
        gas_storage_rate_w=store_g,
        solid_energy_residual_w=solid_res,
        gas_energy_residual_w=gas_res,
        global_energy_residual_w=glob,
        max_absolute_local_residual_w=max(
            float(np.max(np.abs(solid_res))),float(np.max(np.abs(gas_res)))),
        max_local_residual_scaled=scaled,
        global_residual_scaled=global_scaled,
    )
