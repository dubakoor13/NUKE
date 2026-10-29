"""Stage77: fuel transverse Cartesian cut-cell refinement of the FULL
fuel–ZrC–hydrogen transient, with one physical fuel-element inventory.

Stage76 independently refined radial, angular and axial ZrC/H2 meshes,
but held its embedded-boundary fuel x-y grid fixed. This module varies
only that missing fuel-grid direction, while keeping the Stage75 4D
coating, axial mesh, H2 thermodynamics and physical nuclear W/m3 fixed.

Fuel temperatures are initialized by a common CONTINUOUS analytical
T(x,y,z) field sampled at the actual physical cut-cell centroids.
Physical curved fuel/ZrC interface areas, fuel volume, and 19 channel
identities are independently audited on every mesh.

This is measured numerical mesh sensitivity, NOT proof of asymptotic
convergence, NERVA model qualification or a genuine OpenMC heat source.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

import numpy as np

from .coating_mesh_convergence import (
    CoupledMeshResolution, CoupledMeshObservation,
    UniformVolumetricHeating, RefinementDifference, METRICS,
    _compare, run_coupled_mesh_observation,
)
from .fuel_hydrogen_coupled_transient import CoupledCoreModel,CoupledCoreState


@dataclass(frozen=True)
class FuelXyRefinementCase:
    mesh: CoupledMeshResolution
    observation: CoupledMeshObservation
    physical_fuel_cutcell_count: int
    fuel_volume_m3: float
    fuel_coating_contact_area_by_channel_m2: np.ndarray
    initial_fuel_temperature_min_k: float
    initial_fuel_temperature_max_k: float
    minimum_fuel_cutcell_volume_m3: float


@dataclass(frozen=True)
class FuelXyConvergenceStudy:
    sizes: tuple[tuple[int,int],tuple[int,int],tuple[int,int]]
    cases: tuple[FuelXyRefinementCase,...]
    metrics: dict[str,RefinementDifference]
    max_relative_energy_residual: float
    max_nuclear_source_rebin_difference_j: float
    max_fuel_inventory_discrepancy_m3: float
    max_wall_contact_discrepancy_m2: float
    interpretation: str=(
        "Observed three-grid fuel cut-cell XY sensitivity only. "
        "Peak value, cut-face flux, and wall-energy sequences can be "
        "nonmonotone and do NOT imply continuum/grid-independent reactor "
        "temperatures. No physical NERVA/OpenMC validation is claimed."
    )


def _continuous_fuel_temperature(
    model:CoupledCoreModel,
    initial:CoupledCoreState,
) -> CoupledCoreState:
    """Apply mesh-INDEPENDENT continuous transverse temperature function.

    The model's cutcell centre (x,y) is used, not the integer cell index.
    A smooth nonsymmetric field avoids cancellation of both XY gradients.
    This test perturbation must stay within qualified temperature tables.
    """
    xy=np.asarray(model.solid.transverse.cell_centers_cm,dtype=float)
    nf=model.solid.transverse.n_fuel
    if xy.shape!=(nf,2) or not np.all(np.isfinite(xy)):
        raise ValueError("fuel cutcell physical transverse centroids missing")
    ap=float(model.config.inner_fuel_flat_to_flat_cm)*.5
    if ap<=0 or not math.isfinite(ap):
        raise ValueError("invalid fuel physical hexagon size")
    x=xy[:,0]/ap
    y=xy[:,1]/ap
    # Smooth, independent transverse gradients in both directions.
    physical_delta_k=13.*x + 9.*y + 4.*x*y
    fuel=np.asarray(initial.fuel_temperature_k,dtype=float)
    if fuel.shape!=(model.nzone,nf):
        raise ValueError("fuel initial temperature and cutcell centres disagree")
    profile=fuel+physical_delta_k[None,:]
    model.properties.fuel_cp_j_kg_k.at(profile)
    model.properties.fuel_k_w_m_k.at(profile)
    return replace(initial,fuel_temperature_k=profile)


def _audit_geometry(
    model:CoupledCoreModel,
) -> tuple[int,float,np.ndarray,float]:
    """Independently audit exact curved contacts and fuel-only inventory."""
    cut=model.solid.transverse
    nf=int(cut.n_fuel)
    vol=np.asarray(model.solid.solid_volume_m3,dtype=float)
    if (vol.shape!=(model.nzone,nf)
            or not np.all(np.isfinite(vol)) or np.any(vol<=0)):
        raise ArithmeticError("fuel cutcell refinement has nonphysical mass volume")
    contact=np.asarray(cut.fuel_zrc_contact_m2.sum(axis=0)).ravel()
    if contact.shape!=(19,) or np.any(contact<=0):
        raise ArithmeticError("missing a physical coated channel perimeter")
    reference=np.asarray(model.coating.fuel_arc_m2.sum(axis=0)).ravel()
    if reference.shape!=(19,):
        raise ArithmeticError("coating fuel-wall reference has wrong shape")
    arc_error=float(np.max(np.abs(contact-reference)))
    if arc_error>4e-11:
        raise ArithmeticError("fuel-wall contact does not close on refinement grid")
    # Every fuel cut cell carries its own volume, not a uniform
    # rectangular occupancy. Note that clipped cells can be very small.
    return nf,float(np.sum(vol)),contact,float(np.min(vol))


def run_fuel_xy_convergence(
    factory:Callable[
        [CoupledMeshResolution],
        tuple[CoupledCoreModel,CoupledCoreState],
    ],
    *,
    base_mesh:CoupledMeshResolution,
    heating:UniformVolumetricHeating,
    xy_sizes:tuple[tuple[int,int],tuple[int,int],tuple[int,int]]=(
        (20,24),(28,32),(36,40)),
    fuel_profile:Callable[
        [CoupledCoreModel,CoupledCoreState],CoupledCoreState
    ]=_continuous_fuel_temperature,
) -> FuelXyConvergenceStudy:
    """Three nested-in-resolution physical grids; identical W/m3 inputs.

    XY resolutions need not be mathematically nested in cell-edge
    location. Actual material inventory and per-channel contact area
    are checked before claiming a comparable discretization study.
    """
    if (len(xy_sizes)!=3 or any(
            not isinstance(pair,tuple) or len(pair)!=2 or
            any(not isinstance(v,int) or v<6 for v in pair)
            for pair in xy_sizes)):
        raise ValueError("three fuel XY grid sizes with positive integers required")
    if xy_sizes[0]!=(base_mesh.fuel_nx,base_mesh.fuel_ny):
        raise ValueError("fuel XY refinement must start from declared base grid")
    for coarse,fine in zip(xy_sizes,xy_sizes[1:]):
        if not(fine[0]>coarse[0] and fine[1]>coarse[1]):
            raise ValueError("fuel XY refinement must increase both grid directions")
    cases=[]
    for nx,ny in xy_sizes:
        mesh=CoupledMeshResolution(
            fuel_nx=nx,fuel_ny=ny,
            axial_zones=base_mesh.axial_zones,
            angular_sectors=base_mesh.angular_sectors,
            radial_layers=base_mesh.radial_layers,
        )
        model,initial=factory(mesh)
        initial=fuel_profile(model,initial)
        nf,fuel_volume,contact,min_vol=_audit_geometry(model)
        observation=run_coupled_mesh_observation(
            model,initial,mesh,heating)
        if abs(observation.fuel_volume_m3-fuel_volume)>1e-12:
            raise ArithmeticError("physical fuel source volume differs from cutcell geometry")
        cases.append(FuelXyRefinementCase(
            mesh=mesh,observation=observation,
            physical_fuel_cutcell_count=nf,
            fuel_volume_m3=fuel_volume,
            fuel_coating_contact_area_by_channel_m2=contact.copy(),
            initial_fuel_temperature_min_k=float(
                np.min(initial.fuel_temperature_k)),
            initial_fuel_temperature_max_k=float(
                np.max(initial.fuel_temperature_k)),
            minimum_fuel_cutcell_volume_m3=min_vol,
        ))
    baseline=cases[0]
    vol_error=max(abs(case.fuel_volume_m3-baseline.fuel_volume_m3)
                  for case in cases)
    arc_error=max(float(np.max(np.abs(
        case.fuel_coating_contact_area_by_channel_m2-
        baseline.fuel_coating_contact_area_by_channel_m2)))
        for case in cases)
    if vol_error>max(5e-12,5e-7*baseline.fuel_volume_m3):
        raise ArithmeticError("fuel transverse mesh changed actual physical fuel inventory")
    if arc_error>4e-11:
        raise ArithmeticError("fuel XY mesh changed actual 19-channel wall area")
    # Coating and H2 geometry remain unchanged, by construction.
    for case in cases:
        if (abs(case.observation.coating_volume_m3-
                baseline.observation.coating_volume_m3)>3e-11
                or abs(case.observation.hydrogen_volume_m3-
                       baseline.observation.hydrogen_volume_m3)>3e-11):
            raise ArithmeticError("fuel-only XY study changed coating or H2 geometry")
    obs=[case.observation for case in cases]
    return FuelXyConvergenceStudy(
        sizes=xy_sizes,cases=tuple(cases),
        metrics={name:_compare(name,tuple(
            item.measurements[name] for item in obs)) for name in METRICS},
        max_relative_energy_residual=max(
            item.relative_first_law_residual for item in obs),
        max_nuclear_source_rebin_difference_j=max(
            abs(item.source_preservation_residual_j) for item in obs),
        max_fuel_inventory_discrepancy_m3=vol_error,
        max_wall_contact_discrepancy_m2=arc_error,
    )
