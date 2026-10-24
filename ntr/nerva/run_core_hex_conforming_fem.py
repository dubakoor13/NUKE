"""Run conforming XY/axial 3-D fuel element thermal physics using NUKE heat maps.

Produces a real triangle-based 19-bore coated hexagonal fuel cross section
with optional Gmsh mesh caching, then solves a 3-D tensor-product Galerkin
fuel/coating heat equation coupled to 19 implicit 1-D H2 channels.

Two source modes are deliberately distinguished:
1. detailed: user-provided spatial [element,z,triangle] deposition array.
2. reconstructed: existing NUKE [element,channel,z] solid power is
   conservatively partitioned among fuel triangles by nearest channel.
   This is a spatial PROJECTION, not a measured triangle-level OpenMC tally.

Historical NERVA properties, hot evaluated data, local neutron KERMA and
coolant momentum remain unqualified/not implemented in this thermal runner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .config import NervaConfig
from .layout import coolant_channel_positions
from .core_conjugate_fv import CoreSolidLaw
from .core_solid_material_tables import TabulatedCoreSolidLaw
from .core_hex_conforming_hydraulics import solve_conforming_core_with_hydraulics
from .core_hex_exterior_surface import CoreExteriorHeatBoundary
from .thermal import HydrogenProperties
from .core_hex_conforming_mesh import (
    HexTriangleSection,make_nerva_hex_triangle_section,
)
from .core_hex_conforming_fem import (
    ConformingCoreState,solve_conforming_core_thermal,
)
from .core_hex_lateral_fv import HexCoolantBoundary
from .core_hex_conforming_power import (
    CartesianSolidPower,project_cartesian_deposition_to_prisms,
)


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("reconstruction",type=Path)
    p.add_argument("--mesh-cache",type=Path,default=None,
                   help="Load existing conforming mesh NPZ, or generate with Gmsh.")
    p.add_argument("--save-mesh",type=Path,default=None)
    p.add_argument("--fuel-mesh-size-m",type=float,default=6e-4)
    p.add_argument("--coating-mesh-size-m",type=float,default=8e-5)
    p.add_argument("--detailed-source",type=Path,default=None,
                   help="Optional NPZ with solid_prism_power_w [element,z,triangle].")
    p.add_argument("--material-separated-mesh-heating",type=Path,default=None,
                   help="NPZ with explicit FUEL/ZrC material W/cm3 fields and "
                        "Cartesian dimension/bounds; requires --element-index.")
    p.add_argument("--element-index",type=int,default=None)
    p.add_argument("--inlet-temperature-k",type=float,required=True)
    p.add_argument("--initial-temperature-k",type=float,required=True)
    p.add_argument("--film-w-m2-k",type=float,required=True)
    p.add_argument("--hydrogen-density-kg-m3",type=float,required=True)
    p.add_argument("--hydrogen-model",choices=("constant","nasa7-h2"),
                   default="nasa7-h2")
    p.add_argument("--hydrogen-cp-reference-j-kg-k",type=float,default=14300.)
    p.add_argument("--fuel-density-kg-m3",type=float,required=True)
    p.add_argument("--fuel-cp-j-kg-k",type=float,default=None)
    p.add_argument("--fuel-conductivity-w-m-k",type=float,default=None)
    p.add_argument("--fuel-material-npz",type=Path,default=None,
                   help="temperature_k / conductivity_w_m_k / cp_j_kg_k"
                        " table. Replaces constant fuel k and cp.")
    p.add_argument("--coating-density-kg-m3",type=float,required=True)
    p.add_argument("--coating-cp-j-kg-k",type=float,default=None)
    p.add_argument("--coating-conductivity-w-m-k",type=float,default=None)
    p.add_argument("--coating-material-npz",type=Path,default=None,
                   help="temperature_k / conductivity_w_m_k / cp_j_kg_k"
                        " table. Replaces constant coating k and cp.")
    p.add_argument("--couple-hydraulics",action="store_true",
                   help="Steady 3-D FEM + NASA-7 frozen-H2 compressible "
                        "channel momentum/heat-transfer fixed point.")
    p.add_argument("--inlet-pressure-pa",type=float,default=None)
    p.add_argument("--hydrogen-viscosity-pa-s",type=float,default=1.2e-5)
    p.add_argument("--hydrogen-conductivity-w-m-k",type=float,default=0.18)
    p.add_argument("--hydraulic-roughness-m",type=float,default=1e-6)
    p.add_argument("--outer-contact-w-m2-k",type=float,default=0.,
                   help="Prescribed external perimeter contact conductance.")
    p.add_argument("--outer-contact-temperature-k",type=float,default=None,
                   help="Prescribed external contact reservoir temperature.")
    p.add_argument("--outer-emissivity",type=float,default=0.,
                   help="Greybody emissivity of the exposed fuel-element perimeter.")
    p.add_argument("--outer-radiation-temperature-k",type=float,default=None,
                   help="Absolute exterior radiative environment temperature.")
    p.add_argument("--dt-s",type=float,default=None)
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_conforming_core"))
    return p.parse_args()


def channel_power_to_fuel_triangles(
    section:HexTriangleSection,
    channel_axial_power_w:np.ndarray,
    config:NervaConfig|None=None,
)->tuple[np.ndarray,np.ndarray]:
    """Conservative channel -> triangle source projection, not spatial tally.

    Nearest Voronoi channel-site assignment by centroid is an approximation
    for triangles straddling a sector boundary. Assigned sum of triangle
    power is exactly equal to each channel/axial input (up to rounding).
    No heat is silently inserted into a coating triangle.
    """
    cfg=config or NervaConfig()
    section.validate()
    power=np.asarray(channel_axial_power_w,dtype=float)
    if (power.ndim!=2 or power.shape[1]!=19 or power.shape[0]<1
            or not np.isfinite(power).all() or np.any(power<0)):
        raise ValueError("channel nuclear powers must be finite [z,19] W")
    sites=np.asarray(coolant_channel_positions(cfg),dtype=float)*0.01
    xy=section.triangle_centroid_m
    sector=np.argmin(
        np.sum((xy[:,None,:]-sites[None,:,:])**2,axis=2),axis=1)
    volumes=section.triangle_area_m2
    fuel=np.asarray(section.material)==0
    output=np.zeros((power.shape[0],len(sector)),dtype=float)
    for channel in range(19):
        mask=fuel&(sector==channel)
        if not np.any(mask):
            raise ValueError(f"fuel mesh lacks channel sector {channel}")
        weights=volumes[mask]/float(np.sum(volumes[mask]))
        output[:,mask]=power[:,channel,None]*weights[None,:]
    if not np.allclose(
        np.sum(output,axis=1),np.sum(power,axis=1),rtol=1e-12,atol=1e-9):
        raise RuntimeError("nuclear source lost during triangle projection")
    return output,sector


def load_section(path:Path)->HexTriangleSection:
    with np.load(path,allow_pickle=False) as d:
        section=HexTriangleSection(
            xy_m=np.asarray(d["xy_m"],dtype=float),
            triangles=np.asarray(d["triangles"],dtype=np.int64),
            material=np.asarray(d["material"],dtype=int),
            wall_edges=np.asarray(d["wall_edges"],dtype=np.int64),
            wall_channel=np.asarray(d["wall_channel"],dtype=int))
    section.validate()
    return section


def save_section(path:Path|str,mesh:HexTriangleSection)->None:
    mesh.validate()
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        path,xy_m=mesh.xy_m,triangles=mesh.triangles,
        material=mesh.material,wall_edges=mesh.wall_edges,
        wall_channel=mesh.wall_channel)


def main()->int:
    args=parse_args()
    if args.steps<1 or (args.dt_s is None and args.steps!=1):
        raise ValueError("steady solves require one step, transient steps>=1")
    if args.couple_hydraulics:
        if args.dt_s is not None or args.steps!=1:
            raise ValueError("quasi-steady hydraulic outer coupling cannot "
                             "be used for transient H2 inventory")
        if args.hydrogen_model!="nasa7-h2":
            raise ValueError("hydraulic coupling requires frozen NASA-7 H2")
        if (args.inlet_pressure_pa is None
                or not np.isfinite(args.inlet_pressure_pa)
                or args.inlet_pressure_pa<=0):
            raise ValueError("positive inlet pressure required")
    if args.detailed_source and args.material_separated_mesh_heating:
        raise ValueError("select exactly one spatial deposition source")
    if args.material_separated_mesh_heating and args.element_index is None:
        raise ValueError("material-separated Cartesian source needs a "
                         "specific element instance")
    if (not np.isfinite(args.hydrogen_density_kg_m3)
            or args.hydrogen_density_kg_m3<=0):
        raise ValueError("positive H2 inventory density required")
    with np.load(args.reconstruction,allow_pickle=False) as d:
        required=("channel_wall_axial_power_w",
                  "channel_direct_h2_axial_power_w",
                  "branch_mass_flow_kg_s","z_edges_m")
        if not set(required).issubset(d.files):
            raise KeyError("NUKE reconstruction lacks heat/flow fields")
        fuel_power=np.asarray(d[required[0]],dtype=float)
        direct=np.asarray(d[required[1]],dtype=float)
        flows=np.asarray(d[required[2]],dtype=float)
        z=np.asarray(d[required[3]],dtype=float)
    if (fuel_power.ndim!=3 or fuel_power.shape[1]!=19
            or fuel_power.shape!=direct.shape
            or flows.shape!=fuel_power.shape[:2]
            or z.shape!=(fuel_power.shape[2]+1,)
            or any(not np.isfinite(a).all() for a in (fuel_power,direct,flows,z))
            or np.any(fuel_power<0) or np.any(direct<0)
            or np.any(flows<=0) or np.any(np.diff(z)<=0)):
        raise ValueError("invalid NUKE input reconstruction")
    cfg=NervaConfig()
    section=(load_section(args.mesh_cache) if args.mesh_cache
             else make_nerva_hex_triangle_section(
                 config=cfg,fuel_mesh_size_m=args.fuel_mesh_size_m,
                 coating_mesh_size_m=args.coating_mesh_size_m))
    if args.save_mesh:
        save_section(args.save_mesh,section)
    nz=fuel_power.shape[2]
    detail=None
    if args.detailed_source:
        with np.load(args.detailed_source,allow_pickle=False) as d:
            detail=np.asarray(d["solid_prism_power_w"],dtype=float)
        if (detail.shape!=(fuel_power.shape[0],nz,len(section.triangles))
                or not np.isfinite(detail).all() or np.any(detail<0)):
            raise ValueError("detailed spatial deposition shape mismatch")
    voxel_source=None
    if args.material_separated_mesh_heating:
        with np.load(args.material_separated_mesh_heating,allow_pickle=False) as d:
            needed=("dimension","lower_left_cm","upper_right_cm",
                    "fuel_material_power_density_w_cm3",
                    "coating_material_power_density_w_cm3")
            if not set(needed).issubset(d.files):
                raise KeyError("material-specific evaluated deposition fields "
                               "are required; generic heating cannot substitute")
            dim=np.asarray(d["dimension"],dtype=int)
            low=np.asarray(d["lower_left_cm"],dtype=float)
            high=np.asarray(d["upper_right_cm"],dtype=float)
            if (dim.shape!=(3,) or low.shape!=(3,) or high.shape!=(3,)
                    or np.any(dim<1) or np.any(high<=low)):
                raise ValueError("invalid energy mesh geometry")
            edges=[np.linspace(low[i],high[i],int(dim[i])+1)
                   for i in range(3)]
            voxel_source=CartesianSolidPower(
                *edges,
                np.asarray(d["fuel_material_power_density_w_cm3"],dtype=float),
                np.asarray(d["coating_material_power_density_w_cm3"],dtype=float))
        voxel_source.validate()
    def material_law(table, rho, cp, k, label):
        if table is not None:
            if cp is not None or k is not None:
                raise ValueError(
                    f"{label}: material NPZ is exclusive with constant cp/k")
            return TabulatedCoreSolidLaw.from_npz(
                table,density_kg_m3=rho)
        if cp is None or k is None:
            raise ValueError(
                f"{label}: provide a material table OR both constant cp and k")
        law=CoreSolidLaw(rho,cp,k)
        law.validate()
        return law
    fuel=material_law(args.fuel_material_npz,args.fuel_density_kg_m3,
                      args.fuel_cp_j_kg_k,args.fuel_conductivity_w_m_k,"fuel")
    coat=material_law(args.coating_material_npz,
                      args.coating_density_kg_m3,
                      args.coating_cp_j_kg_k,
                      args.coating_conductivity_w_m_k,"coating")
    if ((args.outer_contact_w_m2_k>0 and
            args.outer_contact_temperature_k is None)
            or (args.outer_emissivity>0 and
                args.outer_radiation_temperature_k is None)):
        raise ValueError(
            "external heat-transfer temperature must be explicitly provided; "
            "no hidden NERVA contact/bath temperature assumptions")
    contact_bath=(args.outer_contact_temperature_k if
                  args.outer_contact_temperature_k is not None else
                  args.initial_temperature_k)
    radiation_bath=(args.outer_radiation_temperature_k if
                    args.outer_radiation_temperature_k is not None else
                    args.initial_temperature_k)
    external=None
    if args.outer_contact_w_m2_k or args.outer_emissivity:
        external=CoreExteriorHeatBoundary(
            contact_conductance_w_m2_k=args.outer_contact_w_m2_k,
            contact_reservoir_temperature_k=contact_bath,
            emissivity=args.outer_emissivity,
            radiation_environment_temperature_k=radiation_bath)
    if (not np.isfinite(args.outer_contact_w_m2_k) or
            not np.isfinite(args.outer_emissivity) or
            args.outer_contact_w_m2_k<0 or
            not 0<=args.outer_emissivity<=1):
        raise ValueError("invalid external film coefficient/emissivity")
    selections=(range(fuel_power.shape[0]) if args.element_index is None
                else (args.element_index,))
    if any(not 0<=i<fuel_power.shape[0] for i in selections):
        raise ValueError("invalid fuel element index")
    all_s=[];all_g=[];all_wall=[];reports=[]
    hydraulic_pressure=[];hydraulic_mach=[];hydraulic_film=[]
    hydraulic_friction=[];hydraulic_acceleration=[];hydraulic_kinetic=[]
    for elem in selections:
        if detail is not None:
            source=detail[elem]
        elif voxel_source is not None:
            source=project_cartesian_deposition_to_prisms(
                section,z,voxel_source).solid_prism_power_w
        else:
            source,_=channel_power_to_fuel_triangles(
                section,fuel_power[elem,:,:].T,cfg)
        mass=(args.hydrogen_density_kg_m3*
              np.pi*(cfg.channel_radius_cm*0.01)**2*
              np.diff(z)[:,None]*np.ones((1,19)))
        boundary=HexCoolantBoundary(
            mdot_kg_s=flows[elem],
            inlet_temperature_k=np.full(19,args.inlet_temperature_k),
            hydrogen_mass_kg=mass,
            film_w_m2_k=np.full((nz,19),args.film_w_m2_k),
            direct_hydrogen_power_w=direct[elem,:,:].T,
            hydrogen_model=args.hydrogen_model,
            cp_reference_j_kg_k=args.hydrogen_cp_reference_j_kg_k)
        old=ConformingCoreState(
            np.full((nz+1,len(section.xy_m)),args.initial_temperature_k),
            np.full((nz,19),args.initial_temperature_k))
        history=[]
        for step in range(args.steps):
            hydraulic=None
            if args.couple_hydraulics:
                hydrogen_props=HydrogenProperties(
                    dynamic_viscosity_pa_s=args.hydrogen_viscosity_pa_s,
                    thermal_conductivity_w_m_k=(
                        args.hydrogen_conductivity_w_m_k))
                hydraulic=solve_conforming_core_with_hydraulics(
                    section,z,old,fuel,coat,source,
                    mass_flow_kg_s=flows[elem],
                    inlet_temperature_k=np.full(19,args.inlet_temperature_k),
                    inlet_pressure_pa=np.full(19,args.inlet_pressure_pa),
                    direct_hydrogen_power_w=direct[elem,:,:].T,
                    channel_diameter_m=cfg.coolant_bore_diameter_cm*0.01,
                    roughness_m=args.hydraulic_roughness_m,
                    props=hydrogen_props,
                    exterior_boundary=external)
                sol=hydraulic.thermal
            else:
                sol=solve_conforming_core_thermal(
                    section,z,old,boundary,fuel,coat,source,dt_s=args.dt_s,
                    exterior_boundary=external)
            old=sol.state
            history.append({
                "step":step+1,"iterations":sol.iterations,
                "max_solid_temperature_K":float(
                    np.max(sol.state.solid_temperature_k)),
                "max_gas_outlet_temperature_K":float(
                    np.max(sol.state.hydrogen_temperature_k[-1])),
                "solid_source_power_W":float(source.sum()),
                "direct_H2_power_W":float(direct[elem].sum()),
                "wall_to_gas_power_W":float(sol.wall_to_hydrogen_w.sum()),
                "external_perimeter_heat_loss_W":sol.exterior_heat_loss_w,
                "first_law_residual":sol.first_law_residual,
                "residual_units":"J" if sol.transient else "W",
                "hydraulic_outer_iterations":(
                    hydraulic.iterations if hydraulic is not None else None),
                "max_stagnation_energy_residual_W":(
                    float(np.max(np.abs(hydraulic.stagnation_energy_residual_w)))
                    if hydraulic is not None else None),
                "max_momentum_residual_Pa":(
                    float(np.max(np.abs(hydraulic.momentum_residual_pa)))
                    if hydraulic is not None else None),
                "max_hydraulic_mach":(
                    float(np.max(hydraulic.mach_number))
                    if hydraulic is not None else None),
                "max_hydraulic_thermal_temperature_disagreement_K":(
                    hydraulic.thermal_hydraulic_temperature_error_k
                    if hydraulic is not None else None),
            })
        reports.append({"element_instance":int(elem),"steps":history})
        all_s.append(old.solid_temperature_k)
        all_g.append(old.hydrogen_temperature_k)
        all_wall.append(sol.wall_to_hydrogen_w)
        if args.couple_hydraulics:
            hydraulic_pressure.append(hydraulic.pressure_pa)
            hydraulic_mach.append(hydraulic.mach_number)
            hydraulic_film.append(hydraulic.film_w_m2_k)
            hydraulic_friction.append(hydraulic.pressure_friction_loss_pa)
            hydraulic_acceleration.append(hydraulic.pressure_acceleration_loss_pa)
            hydraulic_kinetic.append(hydraulic.kinetic_correction_w)
    args.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        args.output/"conforming_core_3d_fields.npz",
        solid_temperature_k=np.stack(all_s),
        hydrogen_temperature_k=np.stack(all_g),
        wall_to_hydrogen_w=np.stack(all_wall),
        xy_m=section.xy_m,triangles=section.triangles,
        material=section.material,wall_edges=section.wall_edges,
        wall_channel=section.wall_channel,z_edges_m=z)
    if args.couple_hydraulics:
        np.savez_compressed(
            args.output/"conforming_core_hydraulics.npz",
            pressure_pa=np.stack(hydraulic_pressure),
            mach_number=np.stack(hydraulic_mach),
            heat_transfer_coefficient_w_m2_k=np.stack(hydraulic_film),
            friction_pressure_loss_pa=np.stack(hydraulic_friction),
            acceleration_pressure_loss_pa=np.stack(hydraulic_acceleration),
            kinetic_power_correction_w=np.stack(hydraulic_kinetic))
    report={
        "model":"conforming XY triangle x axial P1 element thermal FEM",
        "geometry_is_original_nuke_demonstration":True,
        "source_kind":(
            "provided triangle-prism deposition" if detail is not None
            else "material-separated exact voxel-to-prism deposition"
            if voxel_source is not None
            else "channel_to_triangle_nearest-site projection"),
        "local_kerma_data_qualified":False,
        "external_surface_model":("prescribed contact + greybody radiation"
                                  if external is not None else "adiabatic"),
        "hydraulic_model":("quasi-steady subsonic NASA7 frozen H2"
                            if args.couple_hydraulics else "prescribed"),
        "fuel_material_model":("tabulated" if args.fuel_material_npz
                               else "constant"),
        "coating_material_model":("tabulated" if args.coating_material_npz
                                  else "constant"),
        "historical_NERVA_reference_validated":False,
        "mesh_nodes":int(len(section.xy_m)),
        "mesh_triangles":int(len(section.triangles)),
        "source_sha256":hashlib.sha256(
            args.reconstruction.read_bytes()).hexdigest(),
        "results":reports,
    }
    (args.output/"conforming_core_3d_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Conforming 3-D thermal elements solved: {len(reports)}")
    print(f"Mesh nodes: {len(section.xy_m)} triangles: {len(section.triangles)}")
    print(f"Output: {args.output}")
    print("Physical hot nuclear deposition and NASA verification NOT established")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
