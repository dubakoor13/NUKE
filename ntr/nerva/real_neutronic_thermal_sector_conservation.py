"""Stage51: actual OpenMC CSG-to-thermal 19-sector volume conservation.

Stage50 discovered six outer ZrC coating annuli intersect the inner y-hex
by 0.00363 cm. Stage51 uses the exact *intersection* area to reconcile the
OpenMC fuel material with its pre-existing equivalent-annulus thermal
sector discretization. It ALSO removes material-cell double ownership
in the true OpenMC CSG, not merely in a diagnostic geometry formula.

No uranium loading, fuel material, reactor dimensions, reflector,
cross-section data, criticality or controls are changed.

The thermal step is SYNTHETIC energy-conservation verification; it
cannot infer reactor heating rates or physical NERVA engine power.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
import openmc

from .config import NervaConfig
from .fuel_sector_geometry import (
    fuel_sector_coated_channel_areas_cm2,
    fuel_sector_fuel_areas_cm2,
    fuel_sector_total_areas_cm2,
    fuel_sector_equivalent_outer_radii_m,
)
from .fx2th_conjugate_fuel_transport import (
    ConjugateCoreGeometry, ConjugateCoreState,
    ConjugateCoreDeposition, advance_conjugate_core,
)
from .geometry import build_fuel_element_universe
from .layout import coolant_channel_positions
from .materials import build_materials
from .real_nasa2015_fuel_tie_csg_volume_audit import (
    fuel_analytical_areas, hex_area, support, NORMALS,
)

SOURCE_GEOLOGY="NUKE-Stage51-actual-OpenMC-CSG-not-historical-NERVA"


def verify_true_openmc_csg_ownership(c:NervaConfig, *, random_points:int=2500)->dict:
    """Query every real Cell.region, not just first match in Universe.find."""
    if not 100<=random_points<=25000:
        raise ValueError("need finite deterministic OpenMC CSG membership sampling")
    materials=build_materials(c)
    universe=build_fuel_element_universe(c,materials)
    cells=list(universe.cells.values())
    exterior=[x for x in cells if x.name=="outer ZrC coating"]
    channels=[x for x in cells if x.name.startswith("ZrC channel coating ")]
    fuel_sectors=[x for x in cells if x.name.startswith("fuel matrix channel sector ")]
    if len(exterior)!=1 or len(channels)!=19 or len(fuel_sectors)!=19:
        raise ValueError("original 19-channel OpenMC material CSG cell topology altered")
    normals=NORMALS + tuple((-a,-b) for a,b in NORMALS)
    inner=c.inner_fuel_flat_to_flat_cm/2
    outer=c.fuel_flat_to_flat_cm/2
    positions=coolant_channel_positions(c)
    clipped_sites=[]
    controlled_points=0
    hits={"fuel":0,"hydrogen":0,"zrc":0}
    def check_point(x:float,y:float, *, expected_cell:str|None=None)->None:
        nonlocal controlled_points
        p=(x,y,0.0)
        matching=[cell for cell in cells if p in cell.region]
        if len(matching)!=1:
            raise ValueError(
                f"OpenMC actual overlapping/uncovered cells at {p}: "
                + ", ".join(z.name for z in matching)
            )
        match=matching[0]
        if expected_cell is not None and match.name!=expected_cell:
            raise ValueError(
                f"incorrect physical CSG cell ownership at {p}: "
                f"expected={expected_cell}, actual={match.name}"
            )
        fill=match.fill
        if not isinstance(fill,openmc.Material):
            raise ValueError("unexpected void in real fuel matrix or H2/ZrC")
        name=next((key for key,val in materials.items() if val.id==fill.id),None)
        if name not in hits:
            raise ValueError("unexpected OpenMC physical material inside fuel element")
        determined=universe.find(p)
        real=[x for x in determined if isinstance(x,openmc.Cell)]
        if not real or real[-1].id!=match.id:
            raise ValueError("native OpenMC Universe.find differs from cell set ownership")
        controlled_points+=1
        hits[name]+=1

    for i,(x,y) in enumerate(positions):
        distances=[(inner-nx*x-ny*y,nx,ny) for nx,ny in normals]
        distance,nx,ny=min(distances)
        if distance<c.coated_channel_radius_cm:
            clipped_sites.append(i+1)
            # The same point was in BOTH the original ZrC channel ring and
            # the exterior ZrC cell. It MUST now be owned by exterior only.
            r=(distance+c.coated_channel_radius_cm)/2
            px=x+r*nx
            py=y+r*ny
            if not inner<support(px,py)<outer:
                raise ValueError("analytically constructed clipped cap site invalid")
            check_point(px,py,expected_cell="outer ZrC coating")
            # Immediately inward of the inner flat, the channel coating
            # is the sole owner of the full annulus remainder.
            r=(distance+c.channel_radius_cm)/2
            check_point(x+r*nx,y+r*ny,
                        expected_cell=f"ZrC channel coating {i+1}")
        for angle in (0.0,1.1,2.2,3.5,4.8):
            ux,uy=math.cos(angle),math.sin(angle)
            r=c.channel_radius_cm*0.6
            px,py=x+r*ux,y+r*uy
            if support(px,py)<outer:
                check_point(px,py,expected_cell=f"hydrogen channel {i+1}")

    rng=np.random.default_rng(5192026)
    accepted=0
    while accepted<random_points:
        px=float(rng.uniform(-outer,outer))
        py=float(rng.uniform(-c.fuel_edge_length_cm,c.fuel_edge_length_cm))
        if support(px,py)>=outer-1e-10:
            continue
        check_point(px,py)
        accepted+=1
    if len(clipped_sites)!=6:
        raise ValueError("six historical clipped channel rings not located")
    if not all(hits.values()):
        raise ValueError("fuel, H2 and ZrC must all be sampled in genuine OpenMC")
    return {
        "genuine_openmc_csg_cell_ownership_points_checked":controlled_points,
        "random_independent_points_inside_element":random_points,
        "all_actual_cells_unique_for_sampled_real_fuel_and_coating_points":True,
        "six_analytically_proved_outer_channel_cap_sites":clipped_sites,
        "true_OpenMC_material_hit_counts":hits,
        "physical_material_identity_and_geometry_unchanged":True,
    }


def fuel_thermal_volume_balance(c:NervaConfig)->dict:
    """Exact sector area check plus an executable conservative synthetic step."""
    sectors=fuel_sector_fuel_areas_cm2(c)
    coated=fuel_sector_coated_channel_areas_cm2(c)
    total=fuel_sector_total_areas_cm2(c)
    radii=fuel_sector_equivalent_outer_radii_m(c)
    reference=fuel_analytical_areas(c)
    exact_area=reference["exact_2d_partition_areas_cm2"]["fuel_matrix"]
    inner_area=hex_area(c.inner_fuel_flat_to_flat_cm)
    old_fuel_area=inner_area-19*math.pi*c.coated_channel_radius_cm**2
    corrected_area=float(sectors.sum())
    delta=corrected_area-old_fuel_area
    if (sectors.shape!=(19,) or radii.shape!=(19,)
            or not np.isfinite(sectors).all() or np.any(sectors<=0)
            or abs(float(total.sum())-inner_area)>1e-11
            or abs(corrected_area-exact_area)>1e-12
            or abs(delta-reference[
                "area_of_ring_outside_inner_hex_but_inside_exterior_ZrC_cm2"]
                )>1e-12):
        raise ValueError("actual neutronic fuel geometry and 19-sector thermal areas differ")
    # Check each existing equivalent annulus preserves its individual
    # sector's correct fuel material mass/heat capacity, not its exact
    # perimeter or noncircular local heat-conduction topology.
    annulus_area=np.pi*(radii**2-(c.coated_channel_radius_cm/100)**2)*1e4
    if not np.allclose(annulus_area,sectors,rtol=0,atol=2e-13):
        raise ValueError("fuel thermal equivalent annuli no longer conserve volume")

    nr=19
    shape=(nr,1,1)
    fuel_edges=np.column_stack(
        (np.full(nr,c.coated_channel_radius_cm/100),radii)
    )
    axial=np.array([0.,c.active_length_cm/100])
    # Entirely synthetic constant-properties thermal conservation test.
    cp=1000.
    rho=1000.
    geo=ConjugateCoreGeometry(
        axial_edges_m=axial,
        fuel_radial_edges_m=fuel_edges,
        channels_per_radial_zone=np.ones(nr,dtype=int),
        solid_density_kg_m3=np.full(shape,rho),
        solid_heat_capacity_j_kg_k=np.full(shape,cp),
        solid_conductivity_w_m_k=np.full(shape,12.),
        hydrogen_mass_kg=np.full((nr,1),0.01),
        hydrogen_cp_j_kg_k=np.full(nr,1000.),
        hydrogen_mass_flow_kg_s=np.zeros(nr),
        inlet_temperature_k=np.full(nr,300.),
        inner_wall_conductance_w_k=np.zeros((nr,1)),
    )
    solid_volumes=geo.solid_volumes_m3
    expected_volume=exact_area*c.active_length_cm*1e-6
    if abs(float(solid_volumes.sum())-expected_volume)>1e-15:
        raise ValueError("thermal solver fuel mass differs from OpenMC physical fuel CSG")
    old_volume=old_fuel_area*c.active_length_cm*1e-6
    if not float(solid_volumes.sum())>old_volume:
        raise ValueError("previous missing thermal fuel mass not recovered")

    initial=ConjugateCoreState(
        time_s=0.,
        fuel_temperature_k=np.full(shape,300.),
        hydrogen_temperature_k=np.full((nr,1),300.),
    )
    # 1e6 W/m³ exactly offsets rho cp 1e6 J/(m³ K) to produce a 1 K
    # rise in 1 second; this is a SYNTHETIC algebraic conservation test.
    synthetic_source_density=1e6
    imposed=synthetic_source_density*solid_volumes
    depo=ConjugateCoreDeposition(
        source_id="SYNTHETIC-unit-volume-energy-closure-NO-reactor-power",
        solid_w=imposed,
        direct_hydrogen_w=np.zeros((nr,1)),
        other_w=np.zeros((nr,1)),
        declared_total_w=float(imposed.sum()),
        synthetic=True,
    )
    step=advance_conjugate_core(geo,initial,depo,1.)
    if (not np.allclose(step.state.fuel_temperature_k,301.,
                        rtol=0,atol=1e-9)
            or not np.allclose(step.state.hydrogen_temperature_k,300.,
                               rtol=0,atol=1e-9)
            or abs(step.thermal_balance_residual_j)>1e-7):
        raise ValueError("synthetic thermal mass/energy conservation failed")
    return {
        "openmc_inner_hex_area_cm2":inner_area,
        "total_voronoi_sector_area_cm2":float(total.sum()),
        "actual_openmc_fuel_material_area_cm2":exact_area,
        "exact_sum_of_19_thermal_sector_fuel_areas_cm2":corrected_area,
        "legacy_full_coated_disk_fuel_area_cm2":old_fuel_area,
        "actual_missing_legacy_thermal_fuel_area_cm2":delta,
        "actual_missing_legacy_thermal_volume_cm3":delta*c.active_length_cm,
        "corrected_material_fuel_volume_cm3":exact_area*c.active_length_cm,
        "corrected_fuel_volume_m3":float(solid_volumes.sum()),
        "old_fuel_volume_m3":old_volume,
        "corrected_vs_original_fuel_mass_relative_change":delta/old_fuel_area,
        "exact_coated_circle_areas_in_true_inner_hex_cm2":[float(x) for x in coated],
        "exact_19_fuel_sector_areas_cm2":[float(x) for x in sectors],
        "exact_19_equivalent_annulus_outer_radii_m":[float(x) for x in radii],
        "synthetic_thermal_energy_conservation_test":True,
        "synthetic_uniform_heat_density_W_m3":synthetic_source_density,
        "synthetic_one_second_thermal_temperature_rise_K":float(
            np.mean(step.state.fuel_temperature_k-300.)
        ),
        "synthetic_thermal_energy_residual_j":float(step.thermal_balance_residual_j),
        "native_NERVA_reactor_thermal_power_or_temperature_verified":False,
        "equivalent_annulus_preserves_mass_not_2D_hex_conduction_operator":True,
    }


def audit_neutron_thermal_geometry_consistency(*,
                                               random_points:int=2500)->dict:
    c=NervaConfig(core_rings=3)
    return {
        "schema":"nuke-stage51-true-openmc-csg-and-19-sector-thermal-volume-v1",
        "original_openmc_fuel_and_coating_cell_overlap_removed":True,
        "actual_nuclear_CSG_material_cell_ownership":
            verify_true_openmc_csg_ownership(c,random_points=random_points),
        "true_geometry_to_thermal_19_sector_conservation":
            fuel_thermal_volume_balance(c),
        "original_historical_nasa_core_fidelity_established":False,
        "historical_hot_NERVA_reactor_watts_qualified":False,
        "physical_geometry_or_fuel_chemistry_adjusted":False,
        "limitations":[
            "No neutron or photon source-normalized heating score drives the synthetic thermal test.",
            "Equivalent annular sectors preserve fuel thermal mass, not noncircular real local conduction.",
            "No historical NASA composition, criticality, real gas EOS or power is inferred.",
            "True CSG cell boundaries now separate inner ZrC annulus and exterior ZrC shell.",
        ],
    }


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--random-csg-points",type=int,default=2500)
    args=p.parse_args()
    data=audit_neutron_thermal_geometry_consistency(
        random_points=args.random_csg_points
    )
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    area=data["true_geometry_to_thermal_19_sector_conservation"]
    print("Actual OpenMC and thermal matrix fuel area (cm2):",
          area["actual_openmc_fuel_material_area_cm2"])
    print("Legacy thermal fuel-area undercount cm2:",
          area["actual_missing_legacy_thermal_fuel_area_cm2"])
    print("Checked true nonoverlapping CSG cell ownership at:",
          data["actual_nuclear_CSG_material_cell_ownership"][
              "genuine_openmc_csg_cell_ownership_points_checked"],"positions")
    print("Synthetic thermal energy closure verified; physical reactor watts NOT inferred")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
