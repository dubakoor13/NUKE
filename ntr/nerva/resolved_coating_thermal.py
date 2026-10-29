"""Stage70 resolved 19-channel coating energy state and two-sided thermal contact.

Physical scope: 19 circular ZrC liners within ONE fuel-element inner hex.
The six peripheral channels are clipped by the same inner-hex boundary
as Stage53. Actual coating volumes use analytic circle-polygon
intersections; fuel contacts use Stage53's exact cut-cell arcs. Inner
bore heat-exchange arcs are independently clipped against the hex.
Neither tie coatings nor the outer element jacket are represented.

No OpenMC power is inferred here. Coating watts, material density, bounded
cp(T), contact resistance and heat-transfer coefficients are explicit data.
A coating channel is one thermal node per axial zone (19*NZ states);
tangential and axial conduction inside ZrC remain out of scope. Distinct
fuel and coolant heat paths share the *same coating enthalpy* and are
applied once with opposite signs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .config import NervaConfig
from .layout import coolant_channel_positions
from .fuel_cutcell_conduction_2d import (
    _NORMALS, _disk_polygon_intersection_area,
)
from .fuel_cutcell_axial_transport import AxialCutCellModel
from .fuel_cutcell_nonlinear_enthalpy import (
    NonlinearThermalProperties, TemperatureTable,
)


@dataclass(frozen=True)
class ResolvedCoatingModel:
    volume_m3: np.ndarray
    mass_kg: np.ndarray
    heat_capacity: TemperatureTable
    half_fuel_distance_m: float
    fuel_arc_m2: csr_matrix
    inner_bore_area_m2: np.ndarray
    contact_resistance_m2_k_w: np.ndarray
    outer_radius_m: float
    inner_radius_m: float
    geometric_mid_radius_m: float


def _hex_polygon(ap: float) -> list[tuple[float, float]]:
    q=ap/math.sqrt(3.)
    return [(ap,q),(0.,2*q),(-ap,q),
            (-ap,-q),(0.,-2*q),(ap,-q)]


def _circumference_inside_hex_cm(
    cx: float, cy: float, radius: float, ap: float,
) -> float:
    """True circle arc within six clipping half planes (no polygons)."""
    events=[0.,2*math.pi]
    for nx,ny in _NORMALS:
        distance=(ap-nx*cx-ny*cy)/radius
        if -1.<distance<1.:
            direction=math.atan2(ny,nx)
            half=math.acos(distance)
            events.extend(((direction+half)%(2*math.pi),
                           (direction-half)%(2*math.pi)))
    events=sorted(set(events))
    angle=0.
    for a,b in zip(events,events[1:]):
        m=(a+b)*.5
        x=cx+radius*math.cos(m)
        y=cy+radius*math.sin(m)
        if all(nx*x+ny*y<=ap+1e-12 for nx,ny in _NORMALS):
            angle+=b-a
    return radius*angle


def build_resolved_coating(
    config: NervaConfig,
    solid: AxialCutCellModel,
    *,
    density_kg_m3: float,
    heat_capacity: TemperatureTable,
    contact_resistance_m2_k_w: object=0.,
) -> ResolvedCoatingModel:
    """Preserve 19 annular volumes and both distinct physical interfaces."""
    config.validate()
    if (not math.isfinite(density_kg_m3) or density_kg_m3<=0
            or not isinstance(heat_capacity,TemperatureTable)):
        raise ValueError("coating requires positive density and bounded cp table")
    nz=solid.axial_zones
    if abs(solid.axial_edges_m[-1]-config.active_length_cm*.01)>1e-12:
        raise ValueError("coating axial geometry differs from fuel")
    contact=np.asarray(contact_resistance_m2_k_w,dtype=float)
    if contact.ndim==0:
        contact=np.full((nz,19),float(contact))
    if (contact.shape!=(nz,19) or not np.all(np.isfinite(contact))
            or np.any(contact<0)):
        raise ValueError("coating contact resistance requires nonnegative m2 K/W")
    ap=config.inner_fuel_flat_to_flat_cm/2.
    outer=config.coated_channel_radius_cm
    bore=config.channel_radius_cm
    poly=_hex_polygon(ap)
    dz=np.diff(solid.axial_edges_m)
    vol=np.zeros((nz,19))
    inner=np.zeros((nz,19))
    outer_contact=np.asarray(
        solid.transverse.fuel_zrc_contact_m2.sum(axis=0)).ravel()
    if np.any(outer_contact<=0):
        raise ValueError("every coating needs a physical fuel contact arc")
    for j,(cx,cy) in enumerate(coolant_channel_positions(config)):
        oa=_disk_polygon_intersection_area(poly,cx,cy,outer)
        ba=_disk_polygon_intersection_area(poly,cx,cy,bore)
        annulus=oa-ba
        if annulus<=0:
            raise ValueError("coating annulus has nonpositive physical volume")
        arc=_circumference_inside_hex_cm(cx,cy,bore,ap)
        if arc<=0:
            raise ValueError("coating has no physically resolved H2 bore")
        vol[:,j]=annulus*1e-4*dz
        inner[:,j]=arc*.01*dz
        outer_arc=_circumference_inside_hex_cm(cx,cy,outer,ap)
        if not np.isclose(
            outer_arc*.01*solid.axial_edges_m[-1],
            outer_contact[j],rtol=0,atol=3e-11):
            raise ArithmeticError("coating/fuel curved interface area mismatch")
    if not np.all(np.isfinite(vol)) or np.any(vol<=0):
        raise ArithmeticError("nonphysical resolved coating volume")
    return ResolvedCoatingModel(
        volume_m3=vol,mass_kg=vol*density_kg_m3,
        heat_capacity=heat_capacity,
        half_fuel_distance_m=.5*min(
            float(np.min(np.diff(solid.transverse.x_edges_cm))),
            float(np.min(np.diff(solid.transverse.y_edges_cm))))*.01,
        fuel_arc_m2=solid.transverse.fuel_zrc_contact_m2.copy(),
        inner_bore_area_m2=inner,contact_resistance_m2_k_w=contact.copy(),
        outer_radius_m=outer*.01,inner_radius_m=bore*.01,
        geometric_mid_radius_m=math.sqrt(bore*outer)*.01,
    )


def coating_interface_power(
    coating: ResolvedCoatingModel,
    solid: AxialCutCellModel,
    laws: NonlinearThermalProperties,
    fuel_temperature_k: np.ndarray,
    coating_temperature_k: np.ndarray,
    hydrogen_temperature_k: np.ndarray,
) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Return (fuel-to-coating by fuel cell, inlet to coating, outlet to H2).

    A positive fuel-to-coating quantity leaves the solid. A positive
    coating-to-hydrogen quantity enters the Stage57 gas energy equation.
    Both fuel and coating contact sides use one identical sparse arc flux.
    """
    nz,nf=fuel_temperature_k.shape
    if (coating_temperature_k.shape!=(nz,19)
            or hydrogen_temperature_k.shape!=(nz,19)
            or nz!=solid.axial_zones or nf!=solid.transverse.n_fuel):
        raise ValueError("coating interface temperature shape mismatch")
    kf=laws.fuel_k_w_m_k.at(fuel_temperature_k)
    kc=laws.coating_k_w_m_k.at(coating_temperature_k)
    film=laws.hydrogen_film_w_m2_k.at(hydrogen_temperature_k)
    contact=coating.fuel_arc_m2.tocoo()
    row=contact.row.astype(int)
    channel=contact.col.astype(int)
    arc_area=contact.data.astype(float)
    scale=np.diff(solid.axial_edges_m)/solid.axial_edges_m[-1]
    area=arc_area[None,:]*scale[:,None]
    outer=coating.outer_radius_m
    middle=coating.geometric_mid_radius_m
    bore=coating.inner_radius_m
    resistance=(coating.half_fuel_distance_m/kf[:,row]
                +outer*math.log(outer/middle)/kc[:,channel]
                +coating.contact_resistance_m2_k_w[:,channel])
    if np.any(resistance<=0):
        raise ArithmeticError("unphysical fuel-coating resistance")
    flux=area/resistance*(fuel_temperature_k[:,row]
                          -coating_temperature_k[:,channel])
    fuel=np.zeros((nz,nf))
    incoming=np.zeros((nz,19))
    for z in range(nz):
        np.add.at(fuel[z,:],row,flux[z,:])
        np.add.at(incoming[z,:],channel,flux[z,:])
    # Through the INNER half of the annulus plus H2 film; zero film
    # heat transfer is treated as a true insulating surface.
    out=np.zeros((nz,19))
    positive=film>0
    rcoat=bore*math.log(middle/bore)/kc
    denom=np.full_like(film,np.inf)
    denom[positive]=rcoat[positive]+1./film[positive]
    out[positive]=coating.inner_bore_area_m2[positive]/denom[positive]*(
        coating_temperature_k[positive]-hydrogen_temperature_k[positive])
    if abs(float(fuel.sum()-incoming.sum()))>1e-11*max(1.,abs(float(fuel.sum()))):
        raise ArithmeticError("coating arc flux was not transferred conservatively")
    return fuel,incoming,out


def advance_coating_enthalpy(
    coating: ResolvedCoatingModel,
    old_temperature_k: np.ndarray,
    *,
    dt_s: float,
    net_w: np.ndarray,
) -> np.ndarray:
    """Invert exact piecewise-linear cp enthalpy for each coating state.

    Valid for arbitrary heat source signs; every result must lie inside the
    declared property-table domain. No extrapolation or negative cp.
    """
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("positive coating step size required")
    old=np.asarray(old_temperature_k,dtype=float)
    power=np.asarray(net_w,dtype=float)
    if (old.shape!=coating.mass_kg.shape or power.shape!=old.shape
            or not np.all(np.isfinite(power))):
        raise ValueError("coating power/temperature shape mismatch")
    cp=coating.heat_capacity
    target=cp.integral(old)+dt_s*power/coating.mass_kg
    lo=float(cp.integral(cp.temperature_k[0]))
    hi=float(cp.integral(cp.temperature_k[-1]))
    if np.any(target<lo-1e-12) or np.any(target>hi+1e-12):
        raise ValueError("coating enthalpy exceeds qualified temperature table")
    a=np.full_like(old,cp.temperature_k[0])
    b=np.full_like(old,cp.temperature_k[-1])
    for _ in range(57):
        m=(a+b)*.5
        enthalpy=cp.integral(m)
        a=np.where(enthalpy<target,m,a)
        b=np.where(enthalpy>=target,m,b)
    return (a+b)*.5
