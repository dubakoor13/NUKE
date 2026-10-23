"""Run laterally coupled 19-channel NERVA-derived fuel thermal physics.

Consumes NUKE's element_channel_power_reconstruction.npz. Its source is a
*prescribed* heating field; historical transport KERMA and hot-core nuclear
physics are not qualified by this CLI or its output. User must supply all
thermal transport constants and coolant inventory assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .core_conjugate_fv import CoreSolidLaw
from .core_hex_lateral_balance import audit_hex_core_cell_energy
from .core_hex_lateral_fv import (
    HexSectorNetwork,HexCoolantBoundary,HexCoreState,solve_hex_core_conjugate
)


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reconstruction",type=Path)
    p.add_argument("--inlet-temperature-k",type=float,required=True)
    p.add_argument("--initial-temperature-k",type=float,required=True)
    p.add_argument("--film-w-m2-k",type=float,required=True)
    p.add_argument("--hydrogen-density-kg-m3",type=float,required=True)
    p.add_argument("--gas-enthalpy-model",choices=("constant","nasa7-h2"),
                   default="nasa7-h2")
    p.add_argument("--gas-reference-cp-j-kg-k",type=float,default=14300.)
    p.add_argument("--fuel-density-kg-m3",type=float,required=True)
    p.add_argument("--fuel-cp-j-kg-k",type=float,required=True)
    p.add_argument("--fuel-conductivity-w-m-k",type=float,required=True)
    p.add_argument("--coating-density-kg-m3",type=float,required=True)
    p.add_argument("--coating-cp-j-kg-k",type=float,required=True)
    p.add_argument("--coating-conductivity-w-m-k",type=float,required=True)
    p.add_argument("--contact-resistance-m2-k-w",type=float,default=0.0)
    p.add_argument("--dt-s",type=float,default=None)
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--element-index",type=int,default=None)
    p.add_argument("--output",type=Path,default=Path("build/nerva_hex_19_channel"))
    return p.parse_args()


def main()->int:
    a=parse_args()
    if a.steps<1 or (a.dt_s is None and a.steps!=1):
        raise ValueError("steady solve needs one step; transient steps >=1")
    if not np.isfinite(a.hydrogen_density_kg_m3) or a.hydrogen_density_kg_m3<=0:
        raise ValueError("provide a positive hydrogen gas density")
    with np.load(a.reconstruction,allow_pickle=False) as data:
        names=("channel_wall_axial_power_w",
               "channel_direct_h2_axial_power_w",
               "branch_mass_flow_kg_s","z_edges_m")
        if not set(names).issubset(data.files):
            raise ValueError("reconstruction missing required source powers/flows")
        qsolid=np.asarray(data[names[0]],dtype=float)
        qhyd=np.asarray(data[names[1]],dtype=float)
        flow=np.asarray(data[names[2]],dtype=float)
        z=np.asarray(data[names[3]],dtype=float)
    if (qsolid.ndim!=3 or qsolid.shape[1]!=19 or qhyd.shape!=qsolid.shape
            or flow.shape!=qsolid.shape[:2] or z.shape!=(qsolid.shape[2]+1,)
            or not all(np.isfinite(v).all() for v in (qsolid,qhyd,flow,z))
            or np.any(qsolid<0) or np.any(qhyd<0)
            or np.any(flow<=0) or np.any(np.diff(z)<=0)):
        raise ValueError("reconstruction fields have wrong shapes/physical domains")
    geometry=HexSectorNetwork.from_nerva_config()
    fuel=CoreSolidLaw(a.fuel_density_kg_m3,a.fuel_cp_j_kg_k,
                      a.fuel_conductivity_w_m_k)
    coat=CoreSolidLaw(a.coating_density_kg_m3,a.coating_cp_j_kg_k,
                      a.coating_conductivity_w_m_k)
    nz=qsolid.shape[2]
    gasmass=(a.hydrogen_density_kg_m3 *
             np.pi*geometry.coolant_bore_radius_m**2*np.diff(z)[:,None]*
             np.ones((1,19)))
    selected=(range(qsolid.shape[0]) if a.element_index is None else
              (a.element_index,))
    if any(not 0<=idx<qsolid.shape[0] for idx in selected):
        raise ValueError("invalid fuel element instance selected")
    rows=[]; fs=[]; cs=[]; gs=[]; lateral=[];wall=[]
    for el in selected:
        boundary=HexCoolantBoundary(
            flow[el],np.full(19,a.inlet_temperature_k),
            gasmass,np.full((nz,19),a.film_w_m2_k),
            qhyd[el].T,hydrogen_model=a.gas_enthalpy_model,
            cp_reference_j_kg_k=a.gas_reference_cp_j_kg_k)
        initial=HexCoreState(
            np.full((nz,19),a.initial_temperature_k),
            np.full((nz,19),a.initial_temperature_k),
            np.full((nz,19),a.initial_temperature_k))
        history=[]
        for step in range(a.steps):
            preceding=initial
            solve=solve_hex_core_conjugate(
                geometry,z,initial,boundary,fuel,coat,qsolid[el].T,
                dt_s=a.dt_s,
                fuel_coating_contact_resistance_m2_k_w=(
                    a.contact_resistance_m2_k_w))
            audit=audit_hex_core_cell_energy(
                geometry,z,preceding,solve,boundary,fuel,coat,qsolid[el].T,
                dt_s=a.dt_s,contact_resistance_m2_k_w=(
                    a.contact_resistance_m2_k_w))
            if audit.maximum_scaled_cell_residual>1e-6 or audit.global_scaled_residual>1e-7:
                raise RuntimeError("independent lateral fuel/coating/H2 first-law failure")
            initial=solve.state
            history.append({
                "step":step+1,"iterations":solve.iterations,
                "max_fuel_temperature_K":float(np.max(solve.state.fuel_temperature_k)),
                "max_coating_temperature_K":float(np.max(solve.state.coating_temperature_k)),
                "max_hydrogen_outlet_temperature_K":float(
                    np.max(solve.state.hydrogen_temperature_k[-1])),
                "maximum_lateral_face_flux_W":float(
                    np.max(np.abs(solve.lateral_face_heat_w))),
                "total_solid_nuclear_power_W":float(qsolid[el].sum()),
                "total_direct_H2_nuclear_power_W":float(qhyd[el].sum()),
                "first_law_residual":solve.energy_residual,
                "independent_max_scaled_cell_residual":audit.maximum_scaled_cell_residual,
                "independent_global_scaled_residual":audit.global_scaled_residual,
                "residual_units":"J" if solve.transient else "W",
            })
        rows.append({"element_instance":int(el),"steps":history})
        fs.append(initial.fuel_temperature_k)
        cs.append(initial.coating_temperature_k)
        gs.append(initial.hydrogen_temperature_k)
        lateral.append(solve.lateral_face_heat_w)
        wall.append(solve.coating_to_hydrogen_w)

    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output/"core_lateral_19_channel_fields.npz",
        fuel_temperature_k=np.asarray(fs),
        coating_temperature_k=np.asarray(cs),
        hydrogen_temperature_k=np.asarray(gs),
        lateral_face_heat_w=np.asarray(lateral),
        coating_to_hydrogen_w=np.asarray(wall),
        sector_pair_indices=geometry.shared_pair_indices,
        sector_face_lengths_m=geometry.shared_face_lengths_m,
        z_edges_m=z)
    report={
        "physics_model":"implicit 19-sector Voronoi-lateral energy network",
        "source_sha256":hashlib.sha256(a.reconstruction.read_bytes()).hexdigest(),
        "lateral_face_count":len(geometry.shared_pair_indices),
        "nuclear_power_is_prescribed":True,
        "material_properties_are_uncalibrated_assumptions":True,
        "physical_hot_openmc_kerma_and_nasa_nerva_validation":False,
        "results":rows,
    }
    (a.output/"core_lateral_19_channel_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("19-channel laterally coupled elements solved:",len(rows))
    print("Output:",a.output)
    print("Note: externally specified heat fields are NOT qualified hot OpenMC results")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
