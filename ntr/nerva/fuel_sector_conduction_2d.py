"""Conservative 19-volume planar heat-conduction operator for NERVA fuel.

The geometry is the *actual y-oriented inner hex* with 19 circular ZrC
exclusions. Each volume is a clipped Voronoi fuel sector; volumes, shared
internal face lengths, clipped fuel/ZrC channel arcs and outside-hex contact
are computed from the same geometry used in the OpenMC material model.

This is a COARSE two-dimensional finite-volume reduction, not a resolved
temperature distribution around the 19 coolant holes. The channel-axis
positions are geometric flux-point surrogates, not physical solid nodes.
Numerical synthetic thermophysical data must never be represented as
qualified operating NERVA temperatures or watts.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

from .config import NervaConfig
from .fuel_sector_geometry import (
    fuel_sector_coated_channel_areas_cm2,
    fuel_sector_fuel_areas_cm2,
    fuel_sector_polygons_cm,
)
from .layout import coolant_channel_positions


@dataclass(frozen=True)
class FuelSectorConduction2D:
    """Thermal capacity and symmetric FV conductances for 19 fuel volumes.

    L has units W/K and obeys:
      (L T)[i] = sum_j G[i,j]*(T[i]-T[j]).
    L @ ones = 0; L is symmetric and positive semidefinite.
    Adiabatic exterior is assumed. Wall conductance is through
    ZrC and an independently specified coolant-side film coefficient.
    """

    fuel_area_cm2: np.ndarray
    fuel_volume_m3: np.ndarray
    fuel_heat_capacity_j_k: np.ndarray
    fuel_zrc_contact_area_m2: np.ndarray
    h2_bore_area_m2: np.ndarray
    exterior_zrc_contact_area_m2: float
    channel_wall_conductance_w_k: np.ndarray
    face_pairs: np.ndarray
    face_lengths_m: np.ndarray
    face_conductance_w_k: np.ndarray
    laplacian_w_k: csr_matrix


@dataclass(frozen=True)
class FuelSectorStep:
    fuel_temperature_k: np.ndarray
    hydrogen_temperature_k: np.ndarray
    fuel_storage_j: float
    hydrogen_storage_j: float
    coolant_export_j: float
    source_energy_j: float
    wall_exchange_j: float
    conservation_residual_j: float


def _positive_finite(label: str, data: object, n: int = 19) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim == 0:
        values = np.full(n, float(values))
    if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{label} must be positive finite scalar or 19-vector")
    return values


def _nonnegative_finite(label: str, data: object, n: int = 19) -> np.ndarray:
    values = np.asarray(data, dtype=float)
    if values.ndim == 0:
        values = np.full(n, float(values))
    if values.shape != (n,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{label} must be nonnegative finite scalar or 19-vector")
    return values


def _finite_vector(label: str, data: object) -> np.ndarray:
    a = np.asarray(data, dtype=float)
    if a.shape != (19,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{label} must be a finite 19-vector")
    return a


def _actual_shared_face_lengths_cm(config: NervaConfig) -> dict[tuple[int, int], float]:
    """Use the exact clipped-Voronoi polygon edges, not a hex-grid guess.

    A segment is an i/j internal face only if *both endpoints* lie on the
    sites' perpendicular bisector; otherwise it must be an exterior hex edge.
    Each interface is counted from the lower-index sector exactly once.
    """
    polygons = fuel_sector_polygons_cm(config)
    sites = coolant_channel_positions(config)
    apothem = 0.5 * config.inner_fuel_flat_to_flat_cm
    sqrt3 = math.sqrt(3)
    normals = ((1., 0.), (.5, sqrt3 / 2), (.5, -sqrt3 / 2))
    shared: dict[tuple[int, int], float] = {}
    exterior = 0.0
    for i, polygon in enumerate(polygons):
        xi, yi = sites[i]
        for p, q in zip(polygon, polygon[1:] + polygon[:1]):
            length = math.hypot(q[0] - p[0], q[1] - p[1])
            if length <= 1e-11:
                continue
            match = None
            for j, (xj, yj) in enumerate(sites):
                if i == j:
                    continue
                a, b = xj-xi, yj-yi
                d = 0.5*(xj*xj+yj*yj-xi*xi-yi*yi)
                if (abs(a*p[0]+b*p[1]-d) < 2e-10 and
                        abs(a*q[0]+b*q[1]-d) < 2e-10):
                    if match is not None:
                        raise ArithmeticError("nonunique Voronoi internal face")
                    match = j
            if match is None:
                for point in (p, q):
                    support = max(abs(nx*point[0]+ny*point[1]) for nx, ny in normals)
                    if abs(support-apothem) > 2e-9:
                        raise ArithmeticError("unclassified internal geometry boundary")
                exterior += length
            elif i < match:
                pair = (i, match)
                shared[pair] = shared.get(pair, 0.0) + length
    expected_perimeter = 6.0 * config.inner_fuel_edge_length_cm
    if abs(exterior-expected_perimeter) > 2e-8:
        raise ArithmeticError("Voronoi polygons do not close the exact inner hex")
    if len(shared) < 30:
        raise ArithmeticError("fuel-sector lateral conduction graph is disconnected")
    return shared


def _channel_interface_lengths_cm(
    config: NervaConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Exact fuel-to-ZrC arcs, full H2 bore arcs, and fuel-to-exterior ZrC.

    For the six partially clipped outer channels, the circular segment
    outside the inner fuel hex belongs to *external ZrC*. A full
    fuel/channel contact circle would overestimate its lateral heat path.
    The bore is entirely inside the hex at the qualified geometry.
    """
    # Reuse Stage51 geometry guard for overlapping holes and two-flat clips.
    footprints = fuel_sector_coated_channel_areas_cm2(config)
    if footprints.shape != (19,):
        raise ArithmeticError("not a 19-channel fuel element")
    r = config.coated_channel_radius_cm
    rb = config.channel_radius_cm
    apothem = 0.5 * config.inner_fuel_flat_to_flat_cm
    s = math.sqrt(3.0) / 2.0
    normals = ((1.,0.),(.5,s),(.5,-s),(-1.,0.),(-.5,-s),(-.5,s))
    active_perimeter = []
    exterior_removed = 0.0
    for x, y in coolant_channel_positions(config):
        distance = min(apothem-nx*x-ny*y for nx,ny in normals)
        if distance <= rb:
            raise ValueError("hydrogen bore crosses the actual inner fuel hex")
        if distance < r:
            theta = 2.0 * math.acos(distance / r)
            exterior_removed += 2.0 * math.sqrt(r*r - distance*distance)
        else:
            theta = 0.0
        active_perimeter.append(r * (2.0*math.pi-theta))
    ext = 6.0*config.inner_fuel_edge_length_cm-exterior_removed
    if ext <= 0:
        raise ArithmeticError("external fuel/ZrC boundary vanished")
    return (np.asarray(active_perimeter),
            np.full(19, 2.0*math.pi*rb),
            ext)


def build_fuel_sector_conduction_2d(
    config: NervaConfig,
    *,
    conductivity_w_m_k: object,
    density_kg_m3: object,
    heat_capacity_j_kg_k: object,
    zrc_conductivity_w_m_k: object,
    hydrogen_film_w_m2_k: object,
) -> FuelSectorConduction2D:
    """Build a conservative coarse 2D physical fuel-sector heat operator.

    Material properties are *explicit required inputs*: no fictional NERVA
    fuel/film properties are hidden as defaults. External fuel/ZrC boundary
    is adiabatic until an independently qualified outer-coating boundary is
    supplied. Flow/pressure feedback is NOT inferred here.
    """
    config.validate()
    k = _positive_finite("fuel conductivity", conductivity_w_m_k)
    rho = _positive_finite("fuel density", density_kg_m3)
    cp = _positive_finite("fuel heat capacity", heat_capacity_j_kg_k)
    kz = _positive_finite("ZrC conductivity", zrc_conductivity_w_m_k)
    film = _nonnegative_finite("hydrogen film coefficient", hydrogen_film_w_m2_k)
    length = config.active_length_cm * 0.01
    sites = coolant_channel_positions(config)
    areas = fuel_sector_fuel_areas_cm2(config)
    volume = areas * 1e-4 * length
    heat_capacity = rho*cp*volume
    arc_cm, bore_cm, ext_cm = _channel_interface_lengths_cm(config)
    contact = arc_cm * 0.01 * length
    bore_area = bore_cm * 0.01 * length
    ext_area = ext_cm * 0.01 * length
    shared = _actual_shared_face_lengths_cm(config)
    pairs = sorted(shared)
    conductance = []
    lengths = []
    rr = []
    cc = []
    vv = []
    diagonal = np.zeros(19)
    for i, j in pairs:
        dist = math.dist(sites[i], sites[j]) * 0.01
        face_len = shared[(i,j)] * 0.01
        # Series half-cell resistance on the orthogonal Voronoi dual.
        # Effective k is harmonic for heterogenous per-sector properties.
        keff = 2.0*k[i]*k[j]/(k[i]+k[j])
        g = keff * face_len * length / dist
        if not (math.isfinite(g) and g > 0):
            raise ArithmeticError("nonphysical lateral conduction coefficient")
        lengths.append(face_len)
        conductance.append(g)
        diagonal[i] += g
        diagonal[j] += g
        rr.extend((i,j))
        cc.extend((j,i))
        vv.extend((-g,-g))
    for i, d in enumerate(diagonal):
        rr.append(i)
        cc.append(i)
        vv.append(float(d))
    L = coo_matrix((vv,(rr,cc)),shape=(19,19)).tocsr()
    L.sum_duplicates()

    bore_r = config.channel_radius_cm * 0.01
    coat_r = config.coated_channel_radius_cm * 0.01
    active_fraction = arc_cm / (2.0 * math.pi * config.coated_channel_radius_cm)
    wall_g = np.zeros(19)
    active = film > 0
    wall_g[active] = (
        2.0 * math.pi * length * active_fraction[active]
        / (np.log(coat_r/bore_r)/kz[active] + 1.0/(bore_r*film[active]))
    )
    if not np.all(np.isfinite(wall_g)) or np.any(wall_g < 0):
        raise ArithmeticError("nonphysical fuel/ZrC/H2 heat-transfer resistance")
    if not np.allclose(L.toarray(), L.toarray().T, rtol=0, atol=2e-12):
        raise ArithmeticError("two-dimensional conductance matrix not symmetric")
    if np.max(np.abs(np.asarray(L.sum(axis=1)))) > 2e-11:
        raise ArithmeticError("lateral conduction lacks discrete first-law closure")
    return FuelSectorConduction2D(
        fuel_area_cm2=areas,
        fuel_volume_m3=volume,
        fuel_heat_capacity_j_k=heat_capacity,
        fuel_zrc_contact_area_m2=contact,
        h2_bore_area_m2=bore_area,
        exterior_zrc_contact_area_m2=ext_area,
        channel_wall_conductance_w_k=wall_g,
        face_pairs=np.asarray(pairs,dtype=int).reshape(-1,2),
        face_lengths_m=np.asarray(lengths,dtype=float),
        face_conductance_w_k=np.asarray(conductance,dtype=float),
        laplacian_w_k=L,
    )


def advance_fuel_sector_2d(
    op: FuelSectorConduction2D,
    old_fuel_temperature_k: object,
    old_hydrogen_temperature_k: object,
    *,
    dt_s: float,
    fuel_heating_w: object,
    hydrogen_heating_w: object,
    hydrogen_heat_capacity_j_k: object,
    hydrogen_flow_heat_capacity_w_k: object,
    inlet_hydrogen_temperature_k: object,
) -> FuelSectorStep:
    """Implicit coupled conservative fuel/H2 step with prescribed properties.

    C_f dT_f/dt = Q_f - L T_f - G_wall (T_f-T_h)
    C_h dT_h/dt = Q_h + G_wall (T_f-T_h) - m_dot cp (T_h-T_in).

    All 38 degrees of freedom are solved at once; no staggered feedback
    or artificial energy residual adjustment. No OpenMC normalization is
    performed by this function.
    """
    if not (math.isfinite(dt_s) and dt_s > 0):
        raise ValueError("dt_s must be positive finite")
    tf = _positive_finite("old fuel temperature", old_fuel_temperature_k)
    th = _positive_finite("old hydrogen temperature", old_hydrogen_temperature_k)
    cf = _positive_finite("fuel heat capacity J/K", op.fuel_heat_capacity_j_k)
    ch = _positive_finite("hydrogen heat capacity J/K", hydrogen_heat_capacity_j_k)
    flow = _nonnegative_finite("hydrogen flow heat capacity W/K",
                               hydrogen_flow_heat_capacity_w_k)
    inlet = _positive_finite("hydrogen inlet temperature", inlet_hydrogen_temperature_k)
    qf = _finite_vector("fuel heating W", fuel_heating_w)
    qh = _finite_vector("hydrogen heating W", hydrogen_heating_w)
    wall = _nonnegative_finite("wall conductance W/K", op.channel_wall_conductance_w_k)

    A = bmat([
        [op.laplacian_w_k+diags(cf/dt_s+wall), -diags(wall)],
        [-diags(wall), diags(ch/dt_s+wall+flow)],
    ],format="csc")
    rhs = np.concatenate((cf*tf/dt_s+qf,
                          ch*th/dt_s+qh+flow*inlet))
    result = np.asarray(spsolve(A,rhs),dtype=float)
    if (result.shape!=(38,) or not np.all(np.isfinite(result)) or
            np.any(result<=0)):
        raise ArithmeticError("invalid coupled implicit fuel/H2 state")
    new_f, new_h = result[:19], result[19:]
    delta_f = float(np.dot(cf,new_f-tf))
    delta_h = float(np.dot(ch,new_h-th))
    export = float(dt_s*np.dot(flow,new_h-inlet))
    injected = float(dt_s*np.sum(qf+qh))
    wall_exchange = float(dt_s*np.dot(wall,new_f-new_h))
    residual = delta_f+delta_h+export-injected
    scale = max(1.,abs(delta_f),abs(delta_h),abs(export),abs(injected))
    if abs(residual)/scale>2e-10:
        raise ArithmeticError("2D fuel/H2 implicit solver violates energy balance")
    return FuelSectorStep(new_f,new_h,delta_f,delta_h,export,injected,
                          wall_exchange,residual)
