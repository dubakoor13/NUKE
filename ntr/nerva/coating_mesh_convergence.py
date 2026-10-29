"""Stage76: measured 3-axis mesh refinement of FULL fuel–ZrC–H2 physics.

Runs the SAME physical source densities [W/m3], initial temperatures,
hydrogen boundary and integration time across independently refined
coating radial shells, angular sectors, and the common axial mesh.
Every case uses the authoritative Stage75 nonlinear 4D coating solver,
Stage58 cut-cell fuel and Stage57 compressible hydrogen equations.

Report differences in physical peak temperatures, area-normalized
local wall heat flux, total wall energy, and the first-law residual.
These are finite-resolution *observed* differences, NOT automatically a
Richardson convergence proof or uncertainty bound. Differences may be
nonmonotone, grid refinement does not qualify missing nuclear data, and
the fuel transverse grid must be held fixed for each coating/axial sweep.

The caller builds and labels each physical mesh and its initial state.
This deliberately prevents production numerics from importing tests
or inventing material properties, reactor source rates or pressures.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .fuel_hydrogen_coupled_transient import (
    CoupledCoreModel, CoupledCoreState, advance_coupled_core,
)
from .coating_radial_interfaces import radial_interface_power
from .hydrogen_transient_fvm import primitive


METRICS=(
    "fuel_peak_k",
    "fuel_volume_mean_k",
    "coating_peak_k",
    "coating_volume_mean_k",
    "hydrogen_peak_k",
    "fuel_outer_wall_peak_abs_w_m2",
    "hydrogen_inner_wall_peak_abs_w_m2",
    "fuel_to_coating_energy_j",
    "coating_to_hydrogen_energy_j",
    "coating_storage_energy_j",
)


@dataclass(frozen=True)
class CoupledMeshResolution:
    """Fuel 2D cutcell grid, H2 axial zones, ZrC azimuthal and radial nodes."""
    fuel_nx: int
    fuel_ny: int
    axial_zones: int
    angular_sectors: int
    radial_layers: int

    def __post_init__(self):
        for name,limit in (("fuel_nx",6),("fuel_ny",6),
                           ("axial_zones",1),("angular_sectors",8),
                           ("radial_layers",2)):
            value=getattr(self,name)
            if not isinstance(value,int) or value<limit:
                raise ValueError(f"{name} mesh resolution must be integer >= {limit}")
        if self.angular_sectors>96 or self.radial_layers>12:
            raise ValueError("coating resolution beyond declared solver limits")


@dataclass(frozen=True)
class UniformVolumetricHeating:
    """Exactly comparable SYNTHETIC uniform volumetric powers in W/m3."""
    fuel_w_m3: float
    coating_w_m3: float
    hydrogen_w_m3: float
    dt_s: float

    def __post_init__(self):
        for name in ("fuel_w_m3","coating_w_m3","hydrogen_w_m3"):
            value=getattr(self,name)
            if not math.isfinite(value) or value<0:
                raise ValueError(f"{name} must be finite nonnegative W/m3")
        if not math.isfinite(self.dt_s) or self.dt_s<=0:
            raise ValueError("positive physical thermal comparison dt required")


@dataclass(frozen=True)
class CoupledMeshObservation:
    mesh: CoupledMeshResolution
    measurements: dict[str,float]
    fuel_volume_m3: float
    coating_volume_m3: float
    hydrogen_volume_m3: float
    total_nuclear_energy_j: float
    source_energy_from_ledger_j: float
    absolute_first_law_residual_j: float
    relative_first_law_residual: float
    coating_first_law_residual_j: float
    radial_node_max_residual_j: float
    source_preservation_residual_j: float
    wall_flux_face_count: int


@dataclass(frozen=True)
class RefinementDifference:
    metric: str
    coarse_to_mid_abs: float
    mid_to_fine_abs: float
    relative_mid_to_fine: float
    adjacent_difference_ratio: float | None
    decreasing_successive_differences: bool | None
    # NOT a claim of formal Richardson convergence.
    interpretation: str


@dataclass(frozen=True)
class RefinementAxis:
    name: str
    counts: tuple[int,int,int]
    observations: tuple[CoupledMeshObservation,...]
    differences: dict[str,RefinementDifference]


@dataclass(frozen=True)
class CoupledMeshConvergenceStudy:
    base_mesh: CoupledMeshResolution
    heating: UniformVolumetricHeating
    axes: dict[str,RefinementAxis]
    unique_cases_run: int
    max_relative_first_law_residual: float
    maximum_nuclear_source_imbalance_j: float
    limitations: str=(
        "Empirical comparisons against finest RUN mesh only; no proven "
        "grid-independent hot-core solution or extrapolation, "
        "all power/material tables externally supplied; OpenMC validation pending."
    )


def _peak_positive_area_flux(watts:np.ndarray,area:np.ndarray)->tuple[float,int]:
    if watts.shape!=area.shape or not np.all(np.isfinite(watts)):
        raise ValueError("wall flux and true physical surface area mismatch")
    valid=area>1e-18
    if not np.any(valid):
        raise ArithmeticError("no physical wall areas to normalize heat flux")
    if np.any(np.abs(watts[~valid])>1e-10):
        raise ArithmeticError("heat transfer through nonexistent wall area")
    return float(np.max(np.abs(watts[valid])/area[valid])),int(np.count_nonzero(valid))


def run_coupled_mesh_observation(
    model:CoupledCoreModel,
    initial:CoupledCoreState,
    mesh:CoupledMeshResolution,
    heating:UniformVolumetricHeating,
    *,
    max_relative_energy_residual:float=5e-8,
) -> CoupledMeshObservation:
    """Solve one identical physical test using volume-normalized sources.

    q_fuel_i = density_fuel * EXACT physical cutcell volume_i
    q_ZrC_i  = density_ZrC  * HEX-clipped angular/radial shell volume_i
    q_H2_i   = density_H2   * actual gas control volume_i

    This distinction matters: prescribing a fixed number of watts PER CELL
    would artificially change nuclear input power under grid refinement.
    """
    if not 0<max_relative_energy_residual<=1e-3:
        raise ValueError("energy residual tolerance outside physical audit limits")
    r=model.radial_coating
    a=model.angular_coating
    c=model.coating
    if r is None or a is None or c is None or model.angular_contacts is None:
        raise ValueError("full radial/azimuthal ZrC coupling required for convergence")
    if (model.nzone!=mesh.axial_zones
            or len(model.solid.transverse.x_edges_cm)-1!=mesh.fuel_nx
            or len(model.solid.transverse.y_edges_cm)-1!=mesh.fuel_ny
            or a.sector_area_m2.shape[1]!=mesh.angular_sectors
            or r.sector_mass_kg.shape!=(mesh.axial_zones,19,
                                        mesh.angular_sectors,mesh.radial_layers)):
        raise ValueError("requested convergence mesh not equal to actual solver geometry")
    if (initial.coating_temperature_k is None
            or initial.coating_temperature_k.shape!=r.sector_mass_kg.shape):
        raise ValueError("physical initial coating state incompatible with mesh")

    vf=np.asarray(model.solid.solid_volume_m3,dtype=float)
    vc=np.asarray(r.sector_volume_m3,dtype=float)
    vh=np.asarray(model.fluid.volume_m3,dtype=float)
    if (not all(np.all(np.isfinite(x)) and np.all(x>=0) for x in (vf,vc,vh))
            or np.any(vc[r.sector_mass_kg>0]<=0)):
        raise ArithmeticError("nonphysical fuel/coating/hydrogen volume in refinement")
    fsrc=heating.fuel_w_m3*vf
    csrc=heating.coating_w_m3*vc
    hsrc=heating.hydrogen_w_m3*vh
    result=advance_coupled_core(
        model,initial,dt_s=heating.dt_s,
        fuel_deposition_w=fsrc,coating_deposition_w=csrc,
        hydrogen_deposition_w=hsrc)
    nuclear=(float(np.sum(fsrc)+np.sum(csrc)+np.sum(hsrc))*
             heating.dt_s)
    accounted=(result.fuel_deposition_j+result.coating_deposition_j+
               result.hydrogen_deposition_j)
    preservation=accounted-nuclear
    if abs(preservation)>max(2e-11,1e-10*max(1.,nuclear)):
        raise ArithmeticError("nuclear source watts changed under FV time integration")
    if result.relative_first_law_residual>max_relative_energy_residual:
        raise ArithmeticError("full-system first law failed under grid refinement")
    if abs(result.coating_first_law_residual_j)>max(
            2e-7,2e-7*max(1.,nuclear)):
        raise ArithmeticError("independent radial ZrC first law fails under refinement")

    end=result.state
    gas=primitive(model.fluid,end.hydrogen)
    contact=radial_interface_power(
        c,model.solid,model.properties,model.angular_contacts,r,
        end.fuel_temperature_k,end.coating_temperature_k,
        gas.temperature_k)
    length=float(model.solid.axial_edges_m[-1]-model.solid.axial_edges_m[0])
    dz=np.diff(model.solid.axial_edges_m)
    outer_full=np.asarray(
        model.angular_contacts.fuel_outer_full_m2.sum(axis=0)).reshape(
        (19,mesh.angular_sectors))
    outer_area=outer_full[None,:,:]*dz[:,None,None]/length
    outer_power=contact.fuel_to_coating_by_radial_w[:,:,:,-1]
    outer_q,outer_count=_peak_positive_area_flux(outer_power,outer_area)
    inner_area=(model.angular_contacts.h2_inner_full_m2[None,:,:]*
                dz[:,None,None]/length)
    inner_power=contact.coating_to_hydrogen_by_radial_w[:,:,:,0]
    inner_q,inner_count=_peak_positive_area_flux(inner_power,inner_area)
    mf=np.sum(vf)
    mc=np.sum(vc)
    mh=np.sum(vh)
    mass_avg_coating=float(np.sum(end.coating_temperature_k*vc)/mc)
    fuel_avg=float(np.sum(end.fuel_temperature_k*vf)/mf)
    measures={
        "fuel_peak_k":float(np.max(end.fuel_temperature_k)),
        "fuel_volume_mean_k":fuel_avg,
        "coating_peak_k":float(np.max(end.coating_temperature_k)),
        "coating_volume_mean_k":mass_avg_coating,
        "hydrogen_peak_k":float(np.max(gas.temperature_k)),
        "fuel_outer_wall_peak_abs_w_m2":outer_q,
        "hydrogen_inner_wall_peak_abs_w_m2":inner_q,
        "fuel_to_coating_energy_j":float(result.fuel_to_coating_j),
        "coating_to_hydrogen_energy_j":float(result.coating_to_hydrogen_j),
        "coating_storage_energy_j":float(result.coating_storage_change_j),
    }
    if any(not math.isfinite(v) for v in measures.values()):
        raise ArithmeticError("nonfinite peak temperature/flux/energy in refinement")
    return CoupledMeshObservation(
        mesh=mesh,measurements=measures,
        fuel_volume_m3=float(mf),coating_volume_m3=float(mc),
        hydrogen_volume_m3=float(mh),
        total_nuclear_energy_j=nuclear,
        source_energy_from_ledger_j=float(accounted),
        absolute_first_law_residual_j=abs(result.first_law_residual_j),
        relative_first_law_residual=float(result.relative_first_law_residual),
        coating_first_law_residual_j=float(result.coating_first_law_residual_j),
        radial_node_max_residual_j=float(result.coating_radial_max_node_residual_j),
        source_preservation_residual_j=float(preservation),
        wall_flux_face_count=outer_count+inner_count,
    )


def _compare(
    metric:str,values:tuple[float,float,float],
) -> RefinementDifference:
    c,m,f=values
    d0=abs(m-c)
    d1=abs(f-m)
    floor=1e-11*max(1.,abs(c),abs(m),abs(f))
    if d0<=floor and d1<=floor:
        ratio=None
        decreasing=None
        status="below absolute numerical reporting floor; not proof of convergence"
    elif d0<=floor:
        ratio=None
        decreasing=False
        status="not demonstrably converging: first step smaller than reporting floor"
    else:
        ratio=d1/d0
        decreasing=bool(d1<d0 and (m-c)*(f-m)>0)
        status=(
            "successive differences decreasing monotonically; not proof of asymptotic order"
            if decreasing else
            "successive differences not monotonically decreasing; refinement remains unresolved"
        )
    return RefinementDifference(
        metric=metric,coarse_to_mid_abs=float(d0),
        mid_to_fine_abs=float(d1),
        relative_mid_to_fine=float(d1/max(1.,abs(f))),
        adjacent_difference_ratio=(None if ratio is None else float(ratio)),
        decreasing_successive_differences=decreasing,
        interpretation=status,
    )


def run_coupled_mesh_convergence(
    factory:Callable[[CoupledMeshResolution],tuple[CoupledCoreModel,CoupledCoreState]],
    *,
    base_mesh:CoupledMeshResolution,
    heating:UniformVolumetricHeating,
    radial_counts:tuple[int,int,int]=(2,3,4),
    angular_counts:tuple[int,int,int]=(8,12,16),
    axial_counts:tuple[int,int,int]=(2,3,4),
) -> CoupledMeshConvergenceStudy:
    """Run 3 levels on each independent axis, sharing the base physical run.

    This returns measured finest-level deltas and observed difference ratios;
    it never silently declares a mesh-independent solution. Fuel transverse
    mesh is held fixed; independent XY refinement is outside this study.
    """
    sequences={
        "radial":radial_counts,
        "angular":angular_counts,
        "axial":axial_counts,
    }
    for axis,levels in sequences.items():
        if (len(levels)!=3 or any(not isinstance(x,int) for x in levels)
                or not levels[0]<levels[1]<levels[2]):
            raise ValueError(f"{axis} refinement must have three increasing integers")
    if (radial_counts[0]!=base_mesh.radial_layers
            or angular_counts[0]!=base_mesh.angular_sectors
            or axial_counts[0]!=base_mesh.axial_zones):
        raise ValueError("all axis sequences must start from the same physical base mesh")
    cache:dict[CoupledMeshResolution,CoupledMeshObservation]={}
    reference_volumes=None
    for axis,counts in sequences.items():
        for count in counts:
            fields=dict(
                fuel_nx=base_mesh.fuel_nx,fuel_ny=base_mesh.fuel_ny,
                axial_zones=base_mesh.axial_zones,
                angular_sectors=base_mesh.angular_sectors,
                radial_layers=base_mesh.radial_layers,
            )
            field={"radial":"radial_layers","angular":"angular_sectors",
                   "axial":"axial_zones"}[axis]
            fields[field]=count
            mesh=CoupledMeshResolution(**fields)
            if mesh in cache:
                continue
            model,initial=factory(mesh)
            obs=run_coupled_mesh_observation(model,initial,mesh,heating)
            v=np.array([obs.fuel_volume_m3,obs.coating_volume_m3,
                        obs.hydrogen_volume_m3])
            if reference_volumes is None:
                reference_volumes=v
            elif not np.allclose(v,reference_volumes,rtol=5e-7,atol=5e-13):
                raise ArithmeticError(
                    "refinement changed physical material inventory/energy volume")
            cache[mesh]=obs
    axes={}
    for axis,counts in sequences.items():
        observations=[]
        for count in counts:
            kwargs={
                "fuel_nx":base_mesh.fuel_nx,
                "fuel_ny":base_mesh.fuel_ny,
                "axial_zones":base_mesh.axial_zones,
                "angular_sectors":base_mesh.angular_sectors,
                "radial_layers":base_mesh.radial_layers,
            }
            kwargs[{"radial":"radial_layers","angular":"angular_sectors",
                    "axial":"axial_zones"}[axis]]=count
            observations.append(cache[CoupledMeshResolution(**kwargs)])
        differences={
            metric:_compare(metric,tuple(
                obs.measurements[metric] for obs in observations))
            for metric in METRICS
        }
        axes[axis]=RefinementAxis(
            name=axis,counts=counts,
            observations=tuple(observations),differences=differences)
    return CoupledMeshConvergenceStudy(
        base_mesh=base_mesh,heating=heating,axes=axes,
        unique_cases_run=len(cache),
        max_relative_first_law_residual=max(
            x.relative_first_law_residual for x in cache.values()),
        maximum_nuclear_source_imbalance_j=max(
            abs(x.source_preservation_residual_j) for x in cache.values()),
    )
