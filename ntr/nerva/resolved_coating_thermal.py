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
Stage71 optionally resolves conservative AXIAL conduction along each sleeve.
Tangential conduction within a sleeve requires angular DOFs and remains
out of scope. Distinct
fuel and coolant heat paths share the *same coating enthalpy* and are
applied once with opposite signs.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

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
    axial_conduction_enabled: bool = False
    axial_edges_m: np.ndarray | None = None


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
    axial_conduction_enabled: bool=False,
) -> ResolvedCoatingModel:
    """Preserve 19 annular volumes and both distinct physical interfaces.

    Axial thermal diffusion can be enabled separately. Circumferential
    diffusion must NOT connect the 19 separate sleeves to one another.
    """
    config.validate()
    if not isinstance(axial_conduction_enabled,bool):
        raise ValueError("axial coating conduction must be explicitly enabled/disabled")
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
        axial_conduction_enabled=axial_conduction_enabled,
        axial_edges_m=np.asarray(solid.axial_edges_m,dtype=float).copy(),
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


@dataclass(frozen=True)
class AxialCoatingDiffusionStep:
    """Nonlinearly implicit axial-coating thermal solve with local energy test."""
    temperature_k: np.ndarray
    axial_interface_j: np.ndarray
    per_node_first_law_residual_j: np.ndarray
    net_axial_energy_j: float
    global_first_law_residual_j: float
    iterations: int


def advance_coating_axial_enthalpy(
    coating: ResolvedCoatingModel,
    old_temperature_k: np.ndarray,
    *,
    conductivity_w_m_k: TemperatureTable,
    dt_s: float,
    net_w: np.ndarray,
    max_iterations: int=40,
    relative_tolerance: float=1e-11,
) -> AxialCoatingDiffusionStep:
    """Implicit finite-volume axial conduction on 19 independent ZrC sleeves.

    G_{z+1/2,h} = A_h /
        [dz_z/(2*k(T_z,h)) + dz_{z+1}/(2*k(T_{z+1},h))].

    The annular cross-section A_h is derived from Stage70 exact coating
    volume divided by physical axial slab length. Each axial interface
    delivers ONE equal/opposite heat flux. All boundaries are insulating,
    so no unexplained inter-sleeve, structural, or external source exists.

    The nonlinear k(T) is re-evaluated each iteration, with positive
    capacity and a symmetric positive *frozen-conductance* Jacobian.
    Line search uses the ACTUAL nonlinear residual and rejects
    unqualified temperatures instead of extrapolating the property table.
    """
    if (not math.isfinite(dt_s) or dt_s<=0 or
            not isinstance(max_iterations,int) or max_iterations<1 or
            not math.isfinite(relative_tolerance) or relative_tolerance<=0):
        raise ValueError("invalid axial coating time step or iteration tolerance")
    if not isinstance(conductivity_w_m_k,TemperatureTable):
        raise ValueError("axial coating conductivity requires bounded table")
    temp0=np.asarray(old_temperature_k,dtype=float)
    power=np.asarray(net_w,dtype=float)
    if (temp0.shape!=coating.mass_kg.shape or power.shape!=temp0.shape
            or not np.all(np.isfinite(temp0)) or
            not np.all(np.isfinite(power))):
        raise ValueError("axial coating states and watts must have shape [z,19]")
    nz,nc=temp0.shape
    if nc!=19 or coating.axial_edges_m is None:
        raise ValueError("missing 19-channel axial coating geometry")
    edges=np.asarray(coating.axial_edges_m,dtype=float)
    dz=np.diff(edges)
    if (edges.shape!=(nz+1,) or not np.all(np.isfinite(edges))
            or not np.all(dz>0) or not np.all(coating.mass_kg>0)):
        raise ValueError("invalid coating axial slab geometry/masses")
    area=coating.volume_m3/dz[:,None]
    if not np.allclose(area,area[0:1,:],rtol=5e-11,atol=1e-15):
        raise ValueError("axial coating area must be geometrically consistent")
    coating.heat_capacity.at(temp0)
    conductivity_w_m_k.at(temp0)
    start_h=coating.heat_capacity.integral(temp0)
    mass=coating.mass_kg
    nn=nz*nc
    columns=np.arange((nz-1)*nc).reshape((nz-1,nc))
    left=(columns//nc)*nc+columns%nc
    right=left+nc
    def residual_jacobian(t: np.ndarray):
        current=t.reshape((nz,nc))
        cap=coating.heat_capacity.at(current)
        energy=coating.heat_capacity.integral(current)
        k=conductivity_w_m_k.at(current)
        f=np.zeros((nz-1,nc))
        conductance=np.zeros_like(f)
        if nz>1:
            conductance=area[:-1,:]/(
                dz[:-1,None]/(2*k[:-1,:])+
                dz[1:,None]/(2*k[1:,:]))
            f=conductance*(current[:-1,:]-current[1:,:])
        net=np.zeros_like(current)
        if nz>1:
            net[:-1,:]+=f
            net[1:,:]-=f
        residual=mass*(energy-start_h)/dt_s+net-power
        main=(mass*cap/dt_s).ravel()
        if nz>1:
            np.add.at(main,left.ravel(),conductance.ravel())
            np.add.at(main,right.ravel(),conductance.ravel())
            jac=diags(main,format="csr")
            from scipy.sparse import coo_matrix
            I=np.concatenate((left.ravel(),right.ravel()))
            J=np.concatenate((right.ravel(),left.ravel()))
            V=np.concatenate((-conductance.ravel(),-conductance.ravel()))
            jac=jac+coo_matrix((V,(I,J)),shape=(nn,nn)).tocsr()
        else:
            jac=diags(main,format="csr")
        return residual, jac, f
    x=temp0.copy().ravel()
    # Do not scale solver convergence by mass*cp/dt: for short
    # acoustic substeps that term may dwarf REAL thermal fluxes and
    # would accept essentially an unadvanced coating state.
    tscale=max(1.,float(np.max(np.abs(power))))
    tolerance=2e-8+relative_tolerance*tscale
    iterations=0
    for count in range(max_iterations):
        res,jac,flux=residual_jacobian(x)
        error=float(np.max(np.abs(res)))
        if error<=tolerance:
            iterations=count
            break
        direction=np.asarray(spsolve(jac,-res.ravel()),dtype=float)
        if not np.all(np.isfinite(direction)):
            raise ArithmeticError("singular/nonfinite axial coating Newton step")
        alpha=1.
        for _ in range(45):
            trial=x+alpha*direction
            try:
                rtrial,_,_=residual_jacobian(trial)
                new_error=float(np.max(np.abs(rtrial)))
            except ValueError:
                alpha*=.5
                continue
            if new_error<error*(1.-1e-5*alpha) or new_error<=tolerance:
                x=trial
                break
            alpha*=.5
        else:
            raise ArithmeticError("coating axial nonlinear line search failed")
    else:
        raise ArithmeticError("axial coating enthalpy did not converge")
    final=x.reshape((nz,nc))
    res,_,flux=residual_jacobian(x)
    node_residual_j=res*dt_s
    hchange=mass*(coating.heat_capacity.integral(final)-start_h)
    all_flux=flux*dt_s
    closure=float(np.sum(hchange)-dt_s*np.sum(power))
    scale=max(1.,float(np.sum(np.abs(hchange))),
              float(dt_s*np.sum(np.abs(power))))
    axial_node_outflow=np.zeros_like(final)
    if nz>1:
        axial_node_outflow[:-1,:]+=all_flux
        axial_node_outflow[1:,:]-=all_flux
    axial_total=float(np.sum(axial_node_outflow))
    if (np.max(np.abs(node_residual_j))>2.*tolerance*dt_s or
            abs(closure)>3e-9*scale or
            abs(axial_total)>1e-11*scale):
        raise ArithmeticError("coating axial finite-volume energy is not conserved")
    return AxialCoatingDiffusionStep(
        temperature_k=final,
        axial_interface_j=all_flux,
        per_node_first_law_residual_j=node_residual_j,
        net_axial_energy_j=axial_total,
        global_first_law_residual_j=closure,
        iterations=iterations,
    )
