"""Solve an externally supplied NUKE channel-heating reconstruction with 2-D FV.

Consumes element_channel_power_reconstruction.npz from the existing
OpenMC-to-channel pipeline. The file's wall axial powers are treated as
prescribed inputs, NOT proof of physically qualified neutron deposition.
This CLI does not run OpenMC or substitute missing MT901 heating data.

One equivalent-annulus solid field and a coupled 1-D gas enthalpy field
are solved per requested fuel channel. Each sector uses its own actual
Voronoi area, while retaining the explicit cylindrical approximation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import NervaConfig
from .core_conjugate_fv import (
    CoreSectorBoundary,CoreSectorGrid,CoreSectorState,
    CoreSolidLaw,solve_core_sector_conjugate,
)
from .fuel_sector_geometry import fuel_sector_fuel_areas_cm2
from .core_conjugate_fv_diagnostics import audit_core_sector_discrete_energy


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reconstruction",type=Path)
    p.add_argument("--inlet-temperature-k",type=float,required=True)
    p.add_argument("--initial-solid-temperature-k",type=float,required=True)
    p.add_argument("--initial-gas-temperature-k",type=float,required=True)
    p.add_argument("--convection-w-m2-k",type=float,required=True)
    p.add_argument("--coolant-density-kg-m3",type=float,required=True)
    p.add_argument("--gas-cp-j-kg-k",type=float,default=14300.0)
    p.add_argument("--gas-enthalpy-model",choices=("constant","nasa7-h2"),
                   default="nasa7-h2")
    p.add_argument("--fuel-density-kg-m3",type=float,required=True)
    p.add_argument("--fuel-cp-j-kg-k",type=float,required=True)
    p.add_argument("--fuel-conductivity-w-m-k",type=float,required=True)
    p.add_argument("--fuel-cp-slope-per-k",type=float,default=0.0)
    p.add_argument("--fuel-conductivity-slope-per-k",type=float,default=0.0)
    p.add_argument("--coating-density-kg-m3",type=float,required=True)
    p.add_argument("--coating-cp-j-kg-k",type=float,required=True)
    p.add_argument("--coating-conductivity-w-m-k",type=float,required=True)
    p.add_argument("--coating-cp-slope-per-k",type=float,default=0.0)
    p.add_argument("--coating-conductivity-slope-per-k",type=float,default=0.0)
    p.add_argument("--dt-s",type=float,default=None,
                   help="Transient backward-Euler timestep; omit for steady.")
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--fuel-coating-contact-resistance-m2-k-w",type=float,default=0.0)
    p.add_argument("--fuel-radial-cells",type=int,default=8)
    p.add_argument("--coating-radial-cells",type=int,default=2)
    p.add_argument("--element-index",type=int,default=None,
                   help="Optional zero-based index; omit for all elements.")
    p.add_argument("--channel-index",type=int,default=None,
                   help="Optional zero-based 0..18 channel; omit for all 19.")
    p.add_argument("--output",type=Path,default=Path("build/core_conjugate_fv"))
    return p.parse_args()


def main()->int:
    a=parse_args()
    if a.dt_s is None and a.steps!=1:
        raise ValueError("steady state accepts only one solve")
    if a.steps<1:
        raise ValueError("steps must be positive")
    if a.coolant_density_kg_m3<=0:
        raise ValueError("positive external H2 density required")
    cfg=NervaConfig()
    cfg.validate()
    with np.load(a.reconstruction,allow_pickle=False) as data:
        keys=("channel_wall_axial_power_w",
              "channel_direct_h2_axial_power_w",
              "branch_mass_flow_kg_s","z_edges_m")
        missing=set(keys)-set(data.files)
        if missing:
            raise KeyError("missing reconstruction fields: "+", ".join(sorted(missing)))
        qsolid=np.asarray(data[keys[0]],dtype=float)
        qgas=np.asarray(data[keys[1]],dtype=float)
        flows=np.asarray(data[keys[2]],dtype=float)
        z=np.asarray(data[keys[3]],dtype=float)
    if (qsolid.ndim!=3 or qsolid.shape[1]!=19 or qsolid.shape!=qgas.shape
            or flows.shape!=qsolid.shape[:2] or z.shape!=(qsolid.shape[2]+1,)
            or not all(np.isfinite(x).all() for x in (qsolid,qgas,flows,z))
            or np.any(np.diff(z)<=0) or np.any(flows<=0)):
        raise ValueError("invalid repeated-element heating/flow/axial geometry")
    area=fuel_sector_fuel_areas_cm2(cfg)*1e-4
    fuel=CoreSolidLaw(
        a.fuel_density_kg_m3,a.fuel_cp_j_kg_k,a.fuel_conductivity_w_m_k,
        cp_slope_per_k=a.fuel_cp_slope_per_k,
        conductivity_slope_per_k=a.fuel_conductivity_slope_per_k)
    coat=CoreSolidLaw(
        a.coating_density_kg_m3,a.coating_cp_j_kg_k,
        a.coating_conductivity_w_m_k,
        cp_slope_per_k=a.coating_cp_slope_per_k,
        conductivity_slope_per_k=a.coating_conductivity_slope_per_k)
    ne=qsolid.shape[0]
    elems=range(ne) if a.element_index is None else [a.element_index]
    channels=range(19) if a.channel_index is None else [a.channel_index]
    if any(not 0<=i<ne for i in elems) or any(not 0<=j<19 for j in channels):
        raise ValueError("element/channel selection outside repeated-core range")
    if a.fuel_radial_cells<1 or a.coating_radial_cells<1:
        raise ValueError("radial mesh count must be positive")
    results=[]
    solids=[]
    gases=[]
    radii=[]
    walls=[]
    for e in elems:
        for c in channels:
            grid=CoreSectorGrid.equivalent_annulus(
                z,bore_radius_m=cfg.channel_radius_cm/100,
                coating_thickness_m=cfg.zrc_coating_thickness_cm/100,
                fuel_sector_area_m2=float(area[c]),
                coating_cells=a.coating_radial_cells,
                fuel_cells=a.fuel_radial_cells)
            nz,nr=grid.shape
            gas_mass=(a.coolant_density_kg_m3 *
                      np.pi*(cfg.channel_radius_cm/100)**2*np.diff(z))
            boundary=CoreSectorBoundary(
                mass_flow_kg_s=float(flows[e,c]),
                inlet_temperature_k=a.inlet_temperature_k,
                convection_w_m2_k=np.full(nz,a.convection_w_m2_k),
                direct_hydrogen_heating_w=qgas[e,c],
                gas_mass_kg=gas_mass,
                constant_gas_cp_j_kg_k=a.gas_cp_j_kg_k,
                gas_enthalpy_model=a.gas_enthalpy_model)
            state=CoreSectorState(
                np.full((nz,nr),a.initial_solid_temperature_k),
                np.full(nz,a.initial_gas_temperature_k))
            run=[]
            for step in range(a.steps):
                old_state=state
                result=solve_core_sector_conjugate(
                    grid,state,boundary,fuel,coat,qsolid[e,c],
                    dt_s=a.dt_s,
                    fuel_coating_contact_resistance_m2_k_w=(
                        a.fuel_coating_contact_resistance_m2_k_w))
                audit=audit_core_sector_discrete_energy(
                    grid,old_state,result,boundary,fuel,coat,qsolid[e,c],
                    dt_s=a.dt_s,
                    fuel_coating_contact_resistance_m2_k_w=(
                        a.fuel_coating_contact_resistance_m2_k_w))
                if audit.max_local_residual_scaled>1e-5 or audit.global_residual_scaled>1e-7:
                    raise RuntimeError("cellwise core FV heat/enthalpy conservation not closed")
                state=result.state
                run.append({
                    "step":step+1,
                    "iterations":result.iterations,
                    "max_solid_temperature_K":float(np.max(
                        result.state.solid_temperature_k)),
                    "outlet_gas_temperature_K":float(
                        result.state.hydrogen_temperature_k[-1]),
                    "solid_heating_W":result.solid_nuclear_heating_w,
                    "direct_hydrogen_heating_W":result.hydrogen_direct_heating_w,
                    "wall_to_hydrogen_W":float(result.convected_wall_heating_w.sum()),
                    "solid_stored_energy_J":result.solid_energy_change_j,
                    "gas_stored_energy_J":result.hydrogen_energy_change_j,
                    "combined_first_law_residual":result.energy_balance_residual_j,
                    "independent_cellwise_max_scaled_residual":audit.max_local_residual_scaled,
                    "independent_global_scaled_residual":audit.global_residual_scaled,
                    "residual_units":"J" if result.transient else "W",
                })
            solids.append(state.solid_temperature_k)
            gases.append(state.hydrogen_temperature_k)
            radii.append(grid.r_edges_m)
            walls.append(result.convected_wall_heating_w)
            results.append({"element_instance":e,"channel_index":c,
                            "radial_geometry":"equal-area annulus",
                            "steps":run})
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        a.output/"core_conjugate_fields.npz",
        solid_temperature_k=np.asarray(solids),
        hydrogen_temperature_k=np.asarray(gases),
        radial_edges_m=np.asarray(radii),
        axial_edges_m=z,
        wall_heat_to_hydrogen_w=np.asarray(walls))
    report={
        "solver":"monolithic implicit radial-axial solid/axial H2 enthalpy finite-volume",
        "source":"externally provided NUKE instance-sector reconstruction",
        "source_sha256":hashlib.sha256(a.reconstruction.read_bytes()).hexdigest(),
        "thermal_material_properties":"user-supplied, uncalibrated",
        "neutron_heating_data_physically_qualified":False,
        "nasa_nerva_core_historical_validation":False,
        "steady":a.dt_s is None,"dt_s":a.dt_s,
        "radial_mesh_coating":a.coating_radial_cells,
        "radial_mesh_fuel":a.fuel_radial_cells,
        "fuel_coating_contact_resistance_m2_k_w":a.fuel_coating_contact_resistance_m2_k_w,
        "discrete_energy_audit":"independent, per cell",
        "channel_results":results,
    }
    (a.output/"core_conjugate_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Solved core conjugate sectors:",len(results))
    print("Results:",a.output.resolve())
    print("NOTE: prescribed neutron deposition is not physically qualified.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
