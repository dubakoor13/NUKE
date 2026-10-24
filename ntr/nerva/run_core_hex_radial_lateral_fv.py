"""NUKE fuel-element runner: unified radial, axial and lateral heat physics.

Accepts the real shape/schema of NUKE's element-channel power reconstruction
[repeated-element, 19-channel, axial-bin] and a shared axial edge array.
The supplied power is NOT declared qualified neutron MT901 deposition.
Thermophysical material, coolant mass, H2 film and flow inputs must be
independently specified. Historical core calibration is not asserted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .core_conjugate_fv import CoreSolidLaw
from .core_hex_lateral_fv import HexSectorNetwork, HexCoolantBoundary
from .core_hex_radial_lateral_fv import (
    RadialHexMesh, RadialHexState, solve_radial_hex_core,
)
from .core_hex_radial_lateral_balance import audit_radial_hex_energy


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reconstruction",type=Path)
    p.add_argument("--inlet-temperature-k",type=float,required=True)
    p.add_argument("--initial-temperature-k",type=float,required=True)
    p.add_argument("--hydrogen-density-kg-m3",type=float,required=True)
    p.add_argument("--film-w-m2-k",type=float,required=True)
    p.add_argument("--hydrogen-model",choices=("constant","nasa7-h2"),
                   default="nasa7-h2")
    p.add_argument("--hydrogen-cp-reference-j-kg-k",type=float,default=14300.)
    p.add_argument("--fuel-density-kg-m3",type=float,required=True)
    p.add_argument("--fuel-cp-j-kg-k",type=float,required=True)
    p.add_argument("--fuel-conductivity-w-m-k",type=float,required=True)
    p.add_argument("--coating-density-kg-m3",type=float,required=True)
    p.add_argument("--coating-cp-j-kg-k",type=float,required=True)
    p.add_argument("--coating-conductivity-w-m-k",type=float,required=True)
    p.add_argument("--coating-radial-cells",type=int,default=2)
    p.add_argument("--fuel-radial-cells",type=int,default=8)
    p.add_argument("--contact-resistance-m2-k-w",type=float,default=0.0)
    p.add_argument("--dt-s",type=float,default=None)
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--element-index",type=int,default=None)
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_core_hex_radial_lateral"))
    return p.parse_args()


def main()->int:
    a=parse_args()
    if a.steps<1 or a.dt_s is None and a.steps!=1:
        raise ValueError("steady solve requires one step; transient steps>=1")
    if (not np.isfinite(a.hydrogen_density_kg_m3)
            or a.hydrogen_density_kg_m3<=0):
        raise ValueError("positive external H2 density required")
    with np.load(a.reconstruction,allow_pickle=False) as data:
        keys=("channel_wall_axial_power_w",
              "channel_direct_h2_axial_power_w",
              "branch_mass_flow_kg_s","z_edges_m")
        missing=set(keys).difference(data.files)
        if missing:
            raise KeyError("reconstruction missing "+", ".join(sorted(missing)))
        nuclear=np.asarray(data[keys[0]],dtype=float)
        h2_power=np.asarray(data[keys[1]],dtype=float)
        flows=np.asarray(data[keys[2]],dtype=float)
        z=np.asarray(data[keys[3]],dtype=float)
    if (nuclear.ndim!=3 or nuclear.shape[1]!=19 or nuclear.shape!=h2_power.shape
            or flows.shape!=nuclear.shape[:2] or z.shape!=(nuclear.shape[2]+1,)
            or any(not np.isfinite(x).all() for x in
                   (nuclear,h2_power,flows,z))
            or np.any(nuclear<0) or np.any(h2_power<0)
            or np.any(flows<=0) or np.any(np.diff(z)<=0)):
        raise ValueError("invalid NUKE reconstructed heat/flow topology")
    geom=HexSectorNetwork.from_nerva_config()
    mesh=RadialHexMesh.from_network(
        geom,z,coating_cells=a.coating_radial_cells,
        fuel_cells=a.fuel_radial_cells)
    fuel=CoreSolidLaw(a.fuel_density_kg_m3,a.fuel_cp_j_kg_k,
                      a.fuel_conductivity_w_m_k)
    coat=CoreSolidLaw(a.coating_density_kg_m3,a.coating_cp_j_kg_k,
                      a.coating_conductivity_w_m_k)
    nz,ns,nr=mesh.shape
    gas_mass=(a.hydrogen_density_kg_m3 *
              np.pi*geom.coolant_bore_radius_m**2 *
              np.diff(z)[:,None]*np.ones((1,ns)))
    selections=(range(nuclear.shape[0]) if a.element_index is None
                else (a.element_index,))
    if any(not 0<=i<nuclear.shape[0] for i in selections):
        raise ValueError("element-index exceeds available instances")
    all_s=[];all_h=[];all_l=[];all_w=[];records=[]
    for elem in selections:
        boundary=HexCoolantBoundary(
            mdot_kg_s=flows[elem,:],
            inlet_temperature_k=np.full(ns,a.inlet_temperature_k),
            hydrogen_mass_kg=gas_mass,
            film_w_m2_k=np.full((nz,ns),a.film_w_m2_k),
            direct_hydrogen_power_w=h2_power[elem,:,:].T,
            hydrogen_model=a.hydrogen_model,
            cp_reference_j_kg_k=a.hydrogen_cp_reference_j_kg_k)
        state=RadialHexState(
            np.full((nz,ns,nr),a.initial_temperature_k),
            np.full((nz,ns),a.initial_temperature_k))
        hist=[]
        for timestep in range(a.steps):
            prev=state
            solved=solve_radial_hex_core(
                mesh,prev,boundary,fuel,coat,nuclear[elem,:,:].T,
                dt_s=a.dt_s,
                contact_resistance_m2_k_w=a.contact_resistance_m2_k_w)
            audit=audit_radial_hex_energy(
                mesh,prev,solved,boundary,fuel,coat,nuclear[elem,:,:].T,
                dt_s=a.dt_s,
                contact_resistance_m2_k_w=a.contact_resistance_m2_k_w)
            if (audit.max_relative_cell_residual>1e-6
                    or audit.global_relative_residual>1e-7):
                raise RuntimeError("unified core FV independent energy audit failed")
            state=solved.state
            hist.append({
                "step":timestep+1,"iterations":solved.iterations,
                "max_solid_temperature_K":float(np.max(state.solid_temperature_k)),
                "max_gas_outlet_temperature_K":float(
                    np.max(state.hydrogen_temperature_k[-1,:])),
                "max_lateral_transfer_W":float(
                    np.max(np.abs(solved.lateral_face_heat_w))),
                "nuclear_solid_power_W":float(np.sum(nuclear[elem])),
                "direct_H2_power_W":float(np.sum(h2_power[elem])),
                "energy_residual":solved.first_law_residual,
                "energy_residual_units":"J" if solved.transient else "W",
                "independent_cell_scaled_residual":audit.max_relative_cell_residual,
                "independent_global_scaled_residual":audit.global_relative_residual,
            })
        records.append({"element_instance":int(elem),"steps":hist})
        all_s.append(state.solid_temperature_k)
        all_h.append(state.hydrogen_temperature_k)
        all_l.append(solved.lateral_face_heat_w)
        all_w.append(solved.wall_to_hydrogen_w)
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        a.output/"unified_radial_lateral_core_fields.npz",
        solid_temperature_k=np.stack(all_s),
        hydrogen_temperature_k=np.stack(all_h),
        lateral_face_heat_w=np.stack(all_l),
        wall_to_hydrogen_w=np.stack(all_w),
        channel_radial_edges_m=mesh.radial_edges_m,
        shared_face_pair_indices=geom.shared_pair_indices,
        axial_edges_m=z)
    report={
        "model":"radial+axial+19-sector-lateral reduced FV conjugate solver",
        "radial_coating_cells":a.coating_radial_cells,
        "radial_fuel_cells":a.fuel_radial_cells,
        "true_3d_hex_mesh":False,
        "qualified_hot_neutron_kerma":False,
        "nuclear_sources_are_prescribed":True,
        "nasa_nerva_reference_validation":False,
        "source_sha256":hashlib.sha256(a.reconstruction.read_bytes()).hexdigest(),
        "results":records,
    }
    (a.output/"unified_radial_lateral_core_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Unified radial/lateral fuel elements solved:",len(records))
    print("Output:",a.output.resolve())
    return 0


if __name__=="__main__":
    raise SystemExit(main())
