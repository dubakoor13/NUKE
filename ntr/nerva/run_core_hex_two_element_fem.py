"""Run TWO separately meshed conforming 3-D NERVA-derived fuel elements.

Two 19-bore fuel elements are coupled by a conservative nonmatching FE
contact mortar, along with both 19-channel H2 enthalpy networks (38 total).
They are solved in ONE nonlinear sparse system. Current spatial power
inputs can be existing NUKE [element,19,z] reconstructions redistributed
to fuel triangles, or two explicit user-provided [z,triangle] power NPZs.

User must supply actual element locations/orientation, contact
conductance, material properties, coolant inventory and heat transfer.
This runner does not extrapolate to a full-core element packing,
qualified local nuclear KERMA, or historical NASA validation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import NervaConfig
from .core_conjugate_fv import CoreSolidLaw
from .core_solid_material_tables import TabulatedCoreSolidLaw
from .core_hex_lateral_fv import HexCoolantBoundary
from .core_hex_conforming_fem import ConformingCoreState
from .core_hex_interelement_mortar import assemble_two_element_contact_mortar
from .core_hex_two_element_fem import solve_two_element_conforming_core
from .run_core_hex_conforming_fem import (
    load_section,channel_power_to_fuel_triangles,
)


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reconstruction",type=Path)
    p.add_argument("--mesh-a",type=Path,required=True)
    p.add_argument("--mesh-b",type=Path,required=True)
    p.add_argument("--element-a-index",type=int,required=True)
    p.add_argument("--element-b-index",type=int,required=True)
    p.add_argument("--origin-b-x-m",type=float,required=True)
    p.add_argument("--origin-b-y-m",type=float,required=True)
    p.add_argument("--normal-x",type=float,required=True)
    p.add_argument("--normal-y",type=float,required=True)
    p.add_argument("--contact-conductance-w-m2-k",type=float,required=True)
    p.add_argument("--power-a-npz",type=Path,
                   help="Optional NPZ solid_prism_power_w [z,triangle_A].")
    p.add_argument("--power-b-npz",type=Path,
                   help="Optional NPZ solid_prism_power_w [z,triangle_B].")
    p.add_argument("--inlet-temperature-a-k",type=float,required=True)
    p.add_argument("--inlet-temperature-b-k",type=float,required=True)
    p.add_argument("--initial-temperature-a-k",type=float,required=True)
    p.add_argument("--initial-temperature-b-k",type=float,required=True)
    p.add_argument("--gas-density-kg-m3",type=float,required=True)
    p.add_argument("--gas-film-w-m2-k",type=float,required=True)
    p.add_argument("--bore-diameter-m",type=float,required=True)
    p.add_argument("--hydrogen-model",choices=("constant","nasa7-h2"),
                   default="nasa7-h2")
    p.add_argument("--fuel-density-kg-m3",type=float,required=True)
    p.add_argument("--fuel-cp-j-kg-k",type=float,default=None)
    p.add_argument("--fuel-conductivity-w-m-k",type=float,default=None)
    p.add_argument("--fuel-material-npz",type=Path)
    p.add_argument("--coating-density-kg-m3",type=float,required=True)
    p.add_argument("--coating-cp-j-kg-k",type=float,default=None)
    p.add_argument("--coating-conductivity-w-m-k",type=float,default=None)
    p.add_argument("--coating-material-npz",type=Path)
    p.add_argument("--dt-s",type=float,default=None)
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--output",type=Path,
                   default=Path("build/two_element_conforming_core"))
    return p.parse_args()


def _law(table:Path|None,rho:float,cp:float|None,k:float|None,label:str):
    if table is not None:
        if cp is not None or k is not None:
            raise ValueError(f"{label} table cannot mix with constant k/cp")
        return TabulatedCoreSolidLaw.from_npz(table,density_kg_m3=rho)
    if cp is None or k is None:
        raise ValueError(f"{label} needs k/cp or a material NPZ")
    law=CoreSolidLaw(rho,cp,k)
    law.validate()
    return law


def _detailed_power(path:Path,nz:int,nelem:int)->np.ndarray:
    with np.load(path,allow_pickle=False) as source:
        out=np.asarray(source["solid_prism_power_w"],dtype=float)
    if (out.shape!=(nz,nelem) or not np.isfinite(out).all()
            or np.any(out<0)):
        raise ValueError("detailed power source must be nonnegative [z,triangle]")
    return out


def main()->int:
    a=parse_args()
    if a.steps<1 or (a.dt_s is None and a.steps!=1):
        raise ValueError("steady operation supports one step only")
    if any(not np.isfinite(t) or t<=0 for t in (
        a.inlet_temperature_a_k,a.inlet_temperature_b_k,
        a.initial_temperature_a_k,a.initial_temperature_b_k,
        a.gas_density_kg_m3,a.gas_film_w_m2_k,a.bore_diameter_m)):
        raise ValueError("nonphysical initial or coolant properties")
    with np.load(a.reconstruction,allow_pickle=False) as data:
        keys=("channel_wall_axial_power_w",
              "channel_direct_h2_axial_power_w",
              "branch_mass_flow_kg_s","z_edges_m")
        if not set(keys).issubset(data.files):
            raise KeyError("NUKE reconstruction missing 19-channel heating/flow")
        nuclear=np.asarray(data[keys[0]],dtype=float)
        direct=np.asarray(data[keys[1]],dtype=float)
        flow=np.asarray(data[keys[2]],dtype=float)
        z=np.asarray(data[keys[3]],dtype=float)
    if (nuclear.ndim!=3 or nuclear.shape[1]!=19
            or direct.shape!=nuclear.shape
            or flow.shape!=nuclear.shape[:2]
            or z.shape!=(nuclear.shape[2]+1,)
            or any(not np.isfinite(x).all() for x in (nuclear,direct,flow,z))
            or np.any(nuclear<0) or np.any(direct<0) or np.any(flow<=0)
            or np.any(np.diff(z)<=0)):
        raise ValueError("invalid NUKE two-element reconstruction")
    for index in (a.element_a_index,a.element_b_index):
        if index<0 or index>=nuclear.shape[0]:
            raise ValueError("selected element index not present")
    sa=load_section(a.mesh_a)
    sb=load_section(a.mesh_b)
    cfg=NervaConfig()
    qa=(_detailed_power(a.power_a_npz,len(z)-1,len(sa.triangles))
        if a.power_a_npz else channel_power_to_fuel_triangles(
            sa,nuclear[a.element_a_index].T,cfg)[0])
    qb=(_detailed_power(a.power_b_npz,len(z)-1,len(sb.triangles))
        if a.power_b_npz else channel_power_to_fuel_triangles(
            sb,nuclear[a.element_b_index].T,cfg)[0])
    mortar=assemble_two_element_contact_mortar(
        sa,sb,z,
        origin_b_m=(a.origin_b_x_m,a.origin_b_y_m),
        normal_a_to_b=(a.normal_x,a.normal_y),
        contact_conductance_w_m2_k=a.contact_conductance_w_m2_k)
    fuel=_law(a.fuel_material_npz,a.fuel_density_kg_m3,
              a.fuel_cp_j_kg_k,a.fuel_conductivity_w_m_k,"fuel")
    coat=_law(a.coating_material_npz,a.coating_density_kg_m3,
              a.coating_cp_j_kg_k,a.coating_conductivity_w_m_k,"coating")
    dz=np.diff(z)
    gas_mass=(a.gas_density_kg_m3*np.pi*(a.bore_diameter_m/2)**2*
              dz[:,None]*np.ones((1,19)))
    def boundary(index:int,inlet:float)->HexCoolantBoundary:
        return HexCoolantBoundary(
            flow[index],np.full(19,inlet),gas_mass,
            np.full((len(dz),19),a.gas_film_w_m2_k),
            direct[index].T,
            hydrogen_model=a.hydrogen_model)
    ba=boundary(a.element_a_index,a.inlet_temperature_a_k)
    bb=boundary(a.element_b_index,a.inlet_temperature_b_k)
    state_a=ConformingCoreState(
        np.full((len(z),len(sa.xy_m)),a.initial_temperature_a_k),
        np.full((len(dz),19),a.initial_temperature_a_k))
    state_b=ConformingCoreState(
        np.full((len(z),len(sb.xy_m)),a.initial_temperature_b_k),
        np.full((len(dz),19),a.initial_temperature_b_k))
    ledger=[]
    for step in range(a.steps):
        result=solve_two_element_conforming_core(
            sa,sb,z,state_a,state_b,ba,bb,
            fuel,coat,qa,qb,mortar,dt_s=a.dt_s)
        state_a=result.element_a
        state_b=result.element_b
        ledger.append({
            "step":step+1,"iterations":result.iterations,
            "contact_heat_a_to_b_w":result.contact_a_to_b_w,
            "element_a_energy_residual":result.element_a_first_law_residual,
            "element_b_energy_residual":result.element_b_first_law_residual,
            "combined_energy_residual":result.combined_first_law_residual,
            "energy_residual_units":"J" if result.transient else "W",
            "hottest_solid_a_k":float(np.max(state_a.solid_temperature_k)),
            "hottest_solid_b_k":float(np.max(state_b.solid_temperature_k)),
        })
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        a.output/"two_element_conforming_fields.npz",
        solid_a_temperature_k=state_a.solid_temperature_k,
        solid_b_temperature_k=state_b.solid_temperature_k,
        hydrogen_a_temperature_k=state_a.hydrogen_temperature_k,
        hydrogen_b_temperature_k=state_b.hydrogen_temperature_k,
        z_edges_m=z)
    report={
        "model":"monolithic two 3D conforming solid fuel elements / 38 H2 channels",
        "nuclear_power_a_w":float(np.sum(qa)),
        "nuclear_power_b_w":float(np.sum(qb)),
        "contact_area_m2":mortar.interface_area_m2,
        "contact_overlap_pairs":mortar.overlap_pairs,
        "prescribed_contact_conductance_w_m2_k":
            a.contact_conductance_w_m2_k,
        "qualified_hot_local_nuclear_deposition":False,
        "historical_nerva_validation":False,
        "source_sha256":hashlib.sha256(a.reconstruction.read_bytes()).hexdigest(),
        "steps":ledger,
    }
    (a.output/"two_element_conforming_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Two 3D fuel elements and 38 H2 channels solved")
    print("Contact area m2:",mortar.interface_area_m2)
    print("Last heat A->B W:",ledger[-1]["contact_heat_a_to_b_w"])
    print("Output:",a.output)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
