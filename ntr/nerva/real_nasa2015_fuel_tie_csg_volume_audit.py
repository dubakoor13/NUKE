"""Stage50 — source-linked NERVA-derived fuel and coaxial tie-element geometry.

NASA NTRS 20150021278, Fittje/Borowski/Schnitzler, 2015, PDF
pp. 6–8, Figs. 4–6, and Sec. III.A:
  https://ntrs.nasa.gov/api/citations/20150021278/downloads/20150021278.pdf

The nominal 111-kN class fuel ELEMENT has 19 coolant bores, 0.2565 cm
diameter, 0.40894 cm center pitch, 50 micrometre ZrC exterior coating,
132 cm length. Figures 5 and 6 print differing 1.913/1.905 cm across-flat
values. Use the model's 1.905 cm nominal baseline, REPORT this source
ambiguity, and do not claim a historical reactor/loaded-fuel reconstruction.

Check analytic cross-sectional area and volume closure AND the material
classification in the SAME OpenMC CSG universes used for actual transport.
Tie-tube coaxial rings are audited as described in the same source, and all
geometric interfaces are checked for nonoverlap. No geometry, fuel,
enrichment, material density, or control changes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import openmc

from .config import NervaConfig
from .geometry import build_fuel_element_universe, build_tie_tube_universe
from .layout import (
    coolant_channel_positions,
    minimum_channel_ligament_cm,
    outer_channel_margin_cm,
)
from .materials import build_materials
from .physical_core_reference_validation import assess_core_fidelity


NASA_SOURCE = "https://ntrs.nasa.gov/api/citations/20150021278/downloads/20150021278.pdf"
NASA_ID = "NASA NTRS 20150021278"
NORMALS = ((0.0, 1.0),
           (math.sqrt(3.0)/2.0, 0.5),
           (math.sqrt(3.0)/2.0,-0.5))


def hex_area(flat_to_flat_cm:float)->float:
    if not math.isfinite(flat_to_flat_cm) or flat_to_flat_cm<=0:
        raise ValueError("hex flat-to-flat dimension must be positive and finite")
    return math.sqrt(3.0)*flat_to_flat_cm**2/2.0


def support(x:float,y:float)->float:
    return max(abs(nx*x+ny*y) for nx,ny in NORMALS)


def fuel_analytical_areas(c:NervaConfig)->dict:
    """Exact cross-sectional partition for the 19-channel CAD/CSG model."""
    c.validate()
    centers=coolant_channel_positions(c)
    if len(centers)!=19:
        raise ValueError("exactly 19 actual fuel-element coolant centers required")
    if (minimum_channel_ligament_cm(c)<=0 or
            outer_channel_margin_cm(c)<=0):
        raise ValueError("OpenMC 19 channel coating circles are overlapping or out of hex")
    outer_area=hex_area(c.fuel_flat_to_flat_cm)
    inner_area=hex_area(c.inner_fuel_flat_to_flat_cm)
    bore=19.0*math.pi*c.channel_radius_cm**2
    coatings=19.0*math.pi*(
        c.coated_channel_radius_cm**2-c.channel_radius_cm**2)
    exterior=outer_area-inner_area
    fuel=inner_area-bore-coatings
    areas={
        "fuel_matrix":fuel,
        "hydrogen_coolant_bores":bore,
        "ZrC_channel_coatings":coatings,
        "ZrC_external_hex_coating":exterior,
    }
    if any(not math.isfinite(x) or x<=0 for x in areas.values()):
        raise ValueError("actual fuel partition has nonphysical area")
    if not math.isclose(sum(areas.values()),outer_area,abs_tol=1e-12,rel_tol=1e-12):
        raise ValueError("analytical fuel element CSG material partitions do not close")
    return {
        "source":NASA_ID,
        "source_pdf":NASA_SOURCE,
        "source_pdf_page_and_figure":"PDF page 6–7, Figs. 5–6; Sec. III.A",
        "actual_config_element_across_flats_cm":c.fuel_flat_to_flat_cm,
        "NASA_fig5_drawn_element_width_cm":1.913,
        "NASA_fig6_element_width_cm":1.905,
        "NASA_two_figure_nominal_width_discrepancy_cm":0.008,
        "model_uses_fig6_nominal_width_not_fabrication_tolerance":True,
        "source_fuel_active_length_cm":132.0,
        "actual_config_fuel_active_length_cm":c.active_length_cm,
        "fuel_coolant_channel_count":len(centers),
        "channel_bore_diameter_cm":c.coolant_bore_diameter_cm,
        "channel_center_pitch_cm":c.coolant_bore_pitch_cm,
        "ZrC_outer_nominal_thickness_cm":c.zrc_coating_thickness_cm,
        "hex_external_total_cross_section_cm2":outer_area,
        "inner_hex_cross_section_cm2":inner_area,
        "exact_2d_partition_areas_cm2":areas,
        "exact_active_length_volumes_cm3":{
            key:val*c.active_length_cm for key,val in areas.items()
        },
        "total_element_volume_cm3":outer_area*c.active_length_cm,
        "coolant_bore_area_fraction":bore/outer_area,
        "fuel_matrix_area_fraction":fuel/outer_area,
        "hydrogen_wetted_bore_perimeter_cm":
            19*2*math.pi*c.channel_radius_cm,
        "channel_coating_fuel_interface_perimeter_cm":
            19*2*math.pi*c.coated_channel_radius_cm,
        "hydrogen_channel_wetted_surface_cm2":
            19*2*math.pi*c.channel_radius_cm*c.active_length_cm,
        "minimum_2d_channel_to_channel_fuel_ligament_cm":
            minimum_channel_ligament_cm(c),
        "minimum_coated_channel_to_inner_hex_margin_cm":
            outer_channel_margin_cm(c),
        "analytical_area_conservation":True,
    }


def tie_analytical_areas(c:NervaConfig)->dict:
    """Compute all 11 radial tie-element zones from *actual* OpenMC radii."""
    c.validate()
    radii=(
        0.0,
        c.tie_inner_tube_inner_radius_cm,
        c.tie_inner_tube_outer_radius_cm,
        c.tie_zrh_inner_radius_cm,
        c.tie_zrh_outer_radius_cm,
        c.tie_outer_tube_inner_radius_cm,
        c.tie_outer_tube_outer_radius_cm,
        c.tie_zrc_inner_radius_cm,
        c.tie_zrc_outer_radius_cm,
        c.tie_graphite_inner_radius_cm,
    )
    names=(
        "hydrogen_supply",
        "inconel_inner_tube",
        "void_inner_cold_gap",
        "ZrH_moderator",
        "hydrogen_return",
        "inconel_outer_tube",
        "void_outer_cold_gap",
        "ZrC_sleeve",
        "void_liner_graphite_cold_gap",
    )
    if len(names)!=len(radii)-1 or not all(hi>lo>=0
                                             for lo,hi in zip(radii,radii[1:])):
        raise ValueError("tie tube CSG radial shells overlap or have negative thickness")
    if c.tie_graphite_inner_radius_cm>=c.inner_fuel_flat_to_flat_cm/2:
        raise ValueError("tie tube outer radial graphite does not fit into inner hex")
    areas={
        name:math.pi*(hi*hi-lo*lo)
        for name,lo,hi in zip(names,radii,radii[1:])
    }
    areas["graphite_filler"]=(
        hex_area(c.inner_fuel_flat_to_flat_cm)
        -math.pi*c.tie_graphite_inner_radius_cm**2
    )
    areas["ZrC_outer_hex_coating"]=(
        hex_area(c.fuel_flat_to_flat_cm)
        -hex_area(c.inner_fuel_flat_to_flat_cm)
    )
    outer=hex_area(c.fuel_flat_to_flat_cm)
    if (any(value<=0 for value in areas.values())
            or not math.isclose(sum(areas.values()),outer,abs_tol=1e-12,rel_tol=1e-12)):
        raise ValueError("coaxial tie support element area conservation failed")
    return {
        "actual_layer_radii_cm":list(radii),
        "radial_layer_order":list(names),
        "actual_tie_shell_areas_cm2":areas,
        "actual_tie_shell_volumes_cm3":{
            name:area*c.active_length_cm for name,area in areas.items()
        },
        "hydrogen_supply_plus_return_flow_area_cm2":
            areas["hydrogen_supply"]+areas["hydrogen_return"],
        "cold_condition_void_gap_area_cm2":
            sum(v for k,v in areas.items() if k.startswith("void_")),
        "tie_element_outer_hex_area_cm2":outer,
        "coaxial_shell_area_conservation":True,
        "original_nasa_figure_cold_gaps_are_not_hot_thermal_expansion":True,
    }


def _actual_material_key(path:list,materials:dict[str,openmc.Material])->str:
    cells=[node for node in path if isinstance(node,openmc.Cell)]
    if not cells or not isinstance(cells[-1].fill,openmc.Material):
        raise ValueError("OpenMC geometric ray did not reach a material cell")
    material=cells[-1].fill
    for key,value in materials.items():
        if material is value or material.id==value.id:
            return key
    raise ValueError("OpenMC CSG cell contains an unknown physical material")


def _fuel_expected(x:float,y:float,c:NervaConfig)->str:
    if support(x,y)>c.fuel_flat_to_flat_cm/2:
        raise ValueError("point is outside source-reference fuel element")
    for cx,cy in coolant_channel_positions(c):
        d=math.hypot(x-cx,y-cy)
        if d<c.channel_radius_cm:
            return "hydrogen"
        if d<c.coated_channel_radius_cm:
            return "zrc"
    if support(x,y)>c.inner_fuel_flat_to_flat_cm/2:
        return "zrc"
    return "fuel"


def _tie_expected(x:float,y:float,c:NervaConfig)->str:
    if support(x,y)>c.fuel_flat_to_flat_cm/2:
        raise ValueError("point is outside source-reference tie element")
    if support(x,y)>c.inner_fuel_flat_to_flat_cm/2:
        return "zrc"
    r=math.hypot(x,y)
    thresholds=(
        (c.tie_inner_tube_inner_radius_cm,"hydrogen"),
        (c.tie_inner_tube_outer_radius_cm,"inconel"),
        (c.tie_zrh_inner_radius_cm,None),
        (c.tie_zrh_outer_radius_cm,"zrh"),
        (c.tie_outer_tube_inner_radius_cm,"hydrogen"),
        (c.tie_outer_tube_outer_radius_cm,"inconel"),
        (c.tie_zrc_inner_radius_cm,None),
        (c.tie_zrc_outer_radius_cm,"zrc"),
        (c.tie_graphite_inner_radius_cm,None),
    )
    for bound,material in thresholds:
        if r<bound:
            return "void" if material is None else material
    return "graphite"


def _actual_fill_key(path:list,materials:dict[str,openmc.Material])->str:
    cells=[node for node in path if isinstance(node,openmc.Cell)]
    if not cells:
        raise ValueError("no CSG OpenMC cell found")
    fill=cells[-1].fill
    if fill is None:
        return "void"
    if not isinstance(fill,openmc.Material):
        raise ValueError("CSG did not finish at an actual material or physical void")
    for key,material in materials.items():
        if fill is material or fill.id==material.id:
            return key
    raise ValueError("unexpected physical OpenMC material in source reference geometry")


def actual_openmc_csg_classification(c:NervaConfig, *,
                                      random_sample_count:int=1800)->dict:
    """Directly query actual transport Universe.find() at controlled 2D sites."""
    if random_sample_count<100 or random_sample_count>20000:
        raise ValueError("finite cross-sectional verification point count required")
    materials=build_materials(c)
    fuel=build_fuel_element_universe(c,materials)
    tie=build_tie_tube_universe(c,materials)
    generators=np.random.default_rng(50492026)
    match={"fuel_element":0,"coaxial_tie_element":0}
    counts={"fuel_element":{},"coaxial_tie_element":{}}

    def check(univ:openmc.Universe,kind:str,x:float,y:float)->None:
        predicted=(_fuel_expected(x,y,c) if kind=="fuel_element"
                   else _tie_expected(x,y,c))
        path=univ.find((x,y,0.0))
        observed=_actual_fill_key(path,materials)
        if predicted!=observed:
            raise ValueError(
                f"OpenMC CSG {kind} material mismatch at ({x:.10g},{y:.10g}): "
                f"analytic={predicted} actual={observed}"
            )
        match[kind]+=1
        group=counts[kind]
        group[predicted]=group.get(predicted,0)+1

    for cx,cy in coolant_channel_positions(c):
        for azimuth in (0.02,0.87,2.31,4.29):
            ux,uy=math.cos(azimuth),math.sin(azimuth)
            for rad in (0.5*c.channel_radius_cm,
                        c.channel_radius_cm+0.50*c.zrc_coating_thickness_cm,
                        c.coated_channel_radius_cm+0.03):
                check(fuel,"fuel_element",cx+rad*ux,cy+rad*uy)
    check(fuel,"fuel_element",0.0,c.fuel_flat_to_flat_cm/2-0.5*c.zrc_coating_thickness_cm)
    for ang in (0.12,1.05,2.43,4.72):
        r=[0.0,c.tie_inner_tube_inner_radius_cm,
           c.tie_inner_tube_outer_radius_cm,c.tie_zrh_inner_radius_cm,
           c.tie_zrh_outer_radius_cm,c.tie_outer_tube_inner_radius_cm,
           c.tie_outer_tube_outer_radius_cm,c.tie_zrc_inner_radius_cm,
           c.tie_zrc_outer_radius_cm,c.tie_graphite_inner_radius_cm]
        for lo,hi in zip(r,r[1:]):
            radius=(lo+hi)/2
            check(tie,"coaxial_tie_element",radius*math.cos(ang),radius*math.sin(ang))
        radius=c.tie_graphite_inner_radius_cm+(
            c.inner_fuel_flat_to_flat_cm/2-c.tie_graphite_inner_radius_cm
        )*0.3
        # For four azimuths the graphite ring remains inside the inner hex
        # only if its hex support is within the inner apothem.
        if support(radius*math.cos(ang),radius*math.sin(ang)) < c.inner_fuel_flat_to_flat_cm/2:
            check(tie,"coaxial_tie_element",radius*math.cos(ang),radius*math.sin(ang))
    check(tie,"coaxial_tie_element",0.0,c.fuel_flat_to_flat_cm/2-0.5*c.zrc_coating_thickness_cm)

    # Exact analytic vs OpenMC CSG material classification on uniform area
    # points in the bounding rectangle, accepted only inside the hex.
    # The sample does not replace analytic area closure, which is exact.
    accepted=0
    while accepted<random_sample_count:
        x=float(generators.uniform(-c.fuel_flat_to_flat_cm/math.sqrt(3),
                                   c.fuel_flat_to_flat_cm/math.sqrt(3)))
        y=float(generators.uniform(-c.fuel_flat_to_flat_cm/2,
                                   c.fuel_flat_to_flat_cm/2))
        if support(x,y)>=c.fuel_flat_to_flat_cm/2-1e-12:
            continue
        check(fuel,"fuel_element",x,y)
        check(tie,"coaxial_tie_element",x,y)
        accepted+=1
    return {
        "actual_openmc_fuel_csg_samples_passed":match["fuel_element"],
        "actual_openmc_tie_csg_samples_passed":match["coaxial_tie_element"],
        "openmc_fuel_material_cell_hit_counts":counts["fuel_element"],
        "openmc_tie_material_and_void_cell_hit_counts":counts["coaxial_tie_element"],
        "original_openmc_csg_matches_independent_analytic_membership_at_all_points":True,
        "geometry_sample_generator_seed":50492026,
        "sampled_areas_not_used_to_infer_historical_loading":True,
    }


def audit_one_historical_element_reference(*,
                                           c:NervaConfig|None=None,
                                           sample_count:int=1800)->dict:
    c=c or NervaConfig(core_rings=3)
    fuel=fuel_analytical_areas(c)
    tie=tie_analytical_areas(c)
    csg=actual_openmc_csg_classification(c,random_sample_count=sample_count)
    hist=assess_core_fidelity(c,assembly="core")
    if (not hist["heritage_element_geometry_matches"]
            or hist["proposed_snre_bulk_geometry_matches"]
            or hist["overall_historical_core_validation"]!="NOT_VALIDATED"):
        raise ValueError("NASA reference model identification is inconsistent")
    return {
        "schema":"nuke-stage50-nasa2015-original-openmc-csg-element-material-volume-v1",
        "single_NASA_reference_id":NASA_ID,
        "single_NASA_reference_url":NASA_SOURCE,
        "original_model_assembly":"core",
        "original_three_ring_19_fuel_element_surrogate":True,
        "element_level_nasa_2015_dimensions":{"fuel":fuel,"tie":tie},
        "original_OpenMC_CSG_analytic_material_membership":csg,
        "separate_heritage_element_vs_proposed_SNRE_reference":hist,
        "material_atom_number_fractions_are_historical_NASA_verified":False,
        "fuel_element_geometry_entails_reactor_k_or_watts":False,
        "physical_coupled_hot_NERVA_reactor_validated":False,
        "limitations":[
            "The geometric source dimensions alone do not validate historical fuel loading or full reactor inventory.",
            "The 1.905/1.913 cm two-figure discrepancy is recorded, not silently reconciled.",
            "Analytical areas are exactly for the *current model*; no hot expansion or historical manufacturing tolerances.",
            "Coaxial tie gaps are described as COLD in the source; no hot gap closure is inferred.",
            "No tuning of fuel enrichment, control drum settings or k-effective performed.",
        ],
    }


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--random-csg-points",type=int,default=1800)
    args=p.parse_args()
    result=audit_one_historical_element_reference(
        sample_count=args.random_csg_points
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    a=result["element_level_nasa_2015_dimensions"]["fuel"]
    print("Original CSG fuel area cm2:",a["exact_2d_partition_areas_cm2"])
    print("Original CSG classification:",result[
        "original_OpenMC_CSG_analytic_material_membership"])
    print("Historical full NERVA reactor and fuel loading: NOT QUALIFIED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
