"""Stage72: conservative azimuthal and axial finite volumes in ZrC sleeves.

Each of the 19 cooling passages has its OWN periodic angular coordinate.
No artificial interface is created between different channel sleeves.
A clipped peripheral sleeve is resolved by the intersection of each
annular polar sector with Stage53's real inner fuel hexagon. The sector
cross-sectional area is obtained by radial integration with adaptive
angular quadrature, not by assigning a full annulus to every passage.

Thermal DOFs are [axial slab, channel 0..18, angular sector]. Angular
heat flows within a coating annulus; axial heat flows within each
individual angular segment. Both use nonnegative, temperature-dependent,
harmonic face conductances and exact piecewise-linear cp enthalpy.

This is an independent *internal conduction operator*. Thermal boundary
flux disaggregation (fuel cutcell -> angular ZrC and angular ZrC -> H2),
physical nuclear coating-source angular shape, real OpenMC transport,
and thermomechanical qualification remain OUTSTANDING. Applying this
operator alone must not be presented as fully coupled reactor physics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import quad
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from .config import NervaConfig
from .layout import coolant_channel_positions
from .fuel_cutcell_conduction_2d import _NORMALS
from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .resolved_coating_thermal import ResolvedCoatingModel


@dataclass(frozen=True)
class AngularCoatingMesh:
    """Independent per-sleeve angular sectors and finite-volume geometry."""
    angular_edges_rad: np.ndarray
    sector_area_m2: np.ndarray
    sector_mass_kg: np.ndarray
    sector_active: np.ndarray
    edge_left: np.ndarray
    edge_right: np.ndarray
    face_resistance_left_geom: np.ndarray
    face_resistance_right_geom: np.ndarray
    face_kind: np.ndarray
    channel_count: int = 19
    geometry_provenance: str = (
        "Clipped hex-annulus sector integration matched against Stage70 "
        "exact coating volumes; no inter-sleeve contacts"
    )


@dataclass(frozen=True)
class AngularCoatingStep:
    """Positive face-J means left-node -> right-node thermal transfer."""
    temperature_k: np.ndarray
    deposited_energy_j: float
    enthalpy_storage_change_j: float
    face_transfer_j: np.ndarray
    face_kind: np.ndarray
    maximum_node_residual_j: float
    total_first_law_residual_j: float
    internal_flux_cancellation_j: float
    iterations: int


def _radial_limit_cm(
    cx: float, cy: float, theta: float,
    *, apothem: float, outer_radius: float,
) -> float:
    """Furthest radius permitted by the *actual* oriented hex halfplanes."""
    ux,uy=math.cos(theta),math.sin(theta)
    upper=outer_radius
    for nx,ny in _NORMALS:
        margin=apothem-nx*cx-ny*cy
        if margin < -5e-12:
            raise ValueError("hydrogen channel centre lies outside inner fuel hex")
        outward=nx*ux+ny*uy
        if outward>1e-15:
            upper=min(upper,max(0.,margin)/outward)
    return max(0.,upper)


def _coating_sector_area_cm2(
    center: tuple[float,float], lo: float,hi: float,
    *, apothem:float, inner:float,outer:float,
) -> float:
    """Integral 0.5*(r_outer(theta)^2-r_bore^2) over active angles."""
    cx,cy=center
    def area(theta):
        radius=_radial_limit_cm(
            cx,cy,theta,apothem=apothem,outer_radius=outer)
        return .5*max(0.,radius*radius-inner*inner)
    value,error=quad(area,lo,hi,epsabs=8e-13,epsrel=2e-11,limit=200)
    if not math.isfinite(value) or value < -2e-13:
        raise ArithmeticError("nonphysical clipped coating angular area")
    return max(0.,value)


def build_angular_coating_mesh(
    config: NervaConfig,
    coating: ResolvedCoatingModel,
    *,
    angular_sectors: int=24,
    density_kg_m3: float,
    max_area_error_fraction: float=2e-7,
) -> AngularCoatingMesh:
    """Discretize all 19 *disjoint* annuli without inventing cross contacts.

    Each angular cell uses exact angular integration of a hex-clipped
    radial annulus. Circumferential faces use the integrated cylindrical
    conduction factor log(r_outer/r_inner), not r*delta_theta as for a
    Cartesian plate. An angular sector cut away by the hex has ZERO mass.
    Such an inactive cell must receive zero energy, and cannot conduct.
    """
    if not isinstance(angular_sectors,int) or not 8<=angular_sectors<=96:
        raise ValueError("coating angular sectors must lie in [8,96]")
    if not math.isfinite(density_kg_m3) or density_kg_m3<=0:
        raise ValueError("positive ZrC density is required")
    if (not math.isfinite(max_area_error_fraction)
            or not 0<max_area_error_fraction<=1e-4):
        raise ValueError("clipped annulus area consistency tolerance invalid")
    edges=np.linspace(0.,2.*math.pi,angular_sectors+1)
    step=edges[1]-edges[0]
    center=coolant_channel_positions(config)
    ap=config.inner_fuel_flat_to_flat_cm*.5
    inner=config.channel_radius_cm
    outer=config.coated_channel_radius_cm
    a=np.zeros((19,angular_sectors),dtype=float)
    radius_face=np.zeros((19,angular_sectors),dtype=float)
    for channel,(cx,cy) in enumerate(center):
        for angle in range(angular_sectors):
            a[channel,angle]=_coating_sector_area_cm2(
                (cx,cy),float(edges[angle]),float(edges[angle+1]),
                apothem=ap,inner=inner,outer=outer)*1e-4
            radius_face[channel,angle]=_radial_limit_cm(
                cx,cy,float(edges[angle+1]),
                apothem=ap,outer_radius=outer)
    if coating.axial_edges_m is None:
        raise ValueError("coating axial geometry required")
    z=np.asarray(coating.axial_edges_m,dtype=float)
    dz=np.diff(z)
    nz=len(dz)
    if (nz<1 or coating.volume_m3.shape!=(nz,19)
            or np.any(dz<=0) or not np.all(np.isfinite(dz))):
        raise ValueError("coating axial extent invalid")
    actual_area=coating.volume_m3/dz[:,None]
    if not np.allclose(
            a.sum(axis=1)[None,:],actual_area,
            atol=1e-13,rtol=max_area_error_fraction):
        relative=np.max(np.abs(
            a.sum(axis=1)[None,:]-actual_area)/
            np.maximum(actual_area,1e-30))
        raise ArithmeticError(
            f"clipped angular ZrC sectors lose physical coating area: {relative}")
    # A sector can be geometrically absent on a clipped peripheral bore.
    # Every physical angular state has positive mass; inactive regions have
    # no source and are excluded from the finite-volume algebraic system.
    active=a>1e-16
    if np.any(active.sum(axis=1)<4):
        raise ValueError("too few physical angular ZrC sectors")
    mass=dz[:,None,None]*a[None,:,:]*density_kg_m3
    nodenum=np.arange(nz*19*angular_sectors).reshape(
        nz,19,angular_sectors)
    left=[]
    right=[]
    geom_left=[]
    geom_right=[]
    kind=[]
    for iz in range(nz):
        for ic in range(19):
            for ia in range(angular_sectors):
                ib=(ia+1)%angular_sectors
                if not (active[ic,ia] and active[ic,ib]):
                    continue
                available=radius_face[ic,ia]
                if available<=inner:
                    continue
                integral=math.log(available/inner)
                if integral<=0:
                    continue
                # G = dz*int_bore^rface(dr/r)
                #       / (dtheta/2/k_left + dtheta/2/k_right).
                left.append(int(nodenum[iz,ic,ia]))
                right.append(int(nodenum[iz,ic,ib]))
                half=step/(2.*dz[iz]*integral)
                geom_left.append(half)
                geom_right.append(half)
                kind.append(1)  # circumferential sleeve face
    for iz in range(nz-1):
        for ic in range(19):
            for ia in range(angular_sectors):
                if not active[ic,ia]:
                    continue
                area=a[ic,ia]
                left.append(int(nodenum[iz,ic,ia]))
                right.append(int(nodenum[iz+1,ic,ia]))
                geom_left.append(dz[iz]/(2.*area))
                geom_right.append(dz[iz+1]/(2.*area))
                kind.append(0)  # axial sleeve face
    return AngularCoatingMesh(
        angular_edges_rad=edges,
        sector_area_m2=a,
        sector_mass_kg=mass,
        sector_active=active,
        edge_left=np.asarray(left,dtype=np.int32),
        edge_right=np.asarray(right,dtype=np.int32),
        face_resistance_left_geom=np.asarray(geom_left,dtype=float),
        face_resistance_right_geom=np.asarray(geom_right,dtype=float),
        face_kind=np.asarray(kind,dtype=np.int8),
    )


def advance_angular_coating_enthalpy(
    coating: ResolvedCoatingModel,
    geometry: AngularCoatingMesh,
    old_temperature_k: object,
    *,
    conductivity_w_m_k: TemperatureTable,
    dt_s: float,
    deposition_w: object,
    max_iterations: int=40,
    relative_tolerance: float=1e-11,
) -> AngularCoatingStep:
    """Implicit conservative axial+angular diffusion on clipped ZrC annuli.

    m_i * (h(T_i^{new})-h(T_i^{old}))/dt
        + SUM_{faces i->j} G_face(T_new)*(T_i^{new}-T_j^{new})
        = Q_i

    No guessed fuel contact, hydrogen film or inter-channel conductance.
    Frozen-conductance Newton/Picard Jacobian is SPD on active nodes.
    The exact nonlinear residual is used for line-search acceptance.
    """
    if (not math.isfinite(dt_s) or dt_s<=0
            or not isinstance(max_iterations,int) or max_iterations<1
            or not math.isfinite(relative_tolerance) or relative_tolerance<=0):
        raise ValueError("invalid angular coating time step or tolerances")
    if not isinstance(conductivity_w_m_k,TemperatureTable):
        raise ValueError("bounded angular-coating k(T) table required")
    mass=np.asarray(geometry.sector_mass_kg,dtype=float)
    t0=np.asarray(old_temperature_k,dtype=float)
    p=np.asarray(deposition_w,dtype=float)
    if (t0.shape!=mass.shape or p.shape!=mass.shape or
            t0.ndim!=3 or t0.shape[1]!=19 or
            not np.all(np.isfinite(t0)) or not np.all(np.isfinite(p))):
        raise ValueError("angular coating state/watts need finite [z,19,theta]")
    active=np.broadcast_to(geometry.sector_active[None,:,:],t0.shape)
    if np.any(np.abs(p[~active])>1e-14):
        raise ValueError("nuclear heating assigned to nonexistent ZrC sector")
    if not np.any(active):
        raise ValueError("coating mesh has no active material")
    coating.heat_capacity.at(t0)
    conductivity_w_m_k.at(t0)
    if not np.array_equal(np.where(active,mass,0),mass):
        raise ValueError("mass in inactive coating sector")
    if np.any(mass[active]<=0):
        raise ValueError("nonpositive physical coating mass")
    if not np.allclose(
            mass.sum(axis=2),coating.mass_kg,atol=1e-10,rtol=2e-7):
        raise ArithmeticError("angular mesh coating mass not conserved")
    nz,nc,na=t0.shape
    size=t0.size
    flat_active=np.flatnonzero(active.ravel())
    lookup=np.full(size,-1,dtype=np.int32)
    lookup[flat_active]=np.arange(len(flat_active),dtype=np.int32)
    ll=lookup[geometry.edge_left]
    rr=lookup[geometry.edge_right]
    if np.any(ll<0) or np.any(rr<0):
        raise ArithmeticError("coating interface touches nonexistent material")
    e0=coating.heat_capacity.integral(t0).ravel()[flat_active]
    m=mass.ravel()[flat_active]
    q=p.ravel()[flat_active]
    left_geo=geometry.face_resistance_left_geom
    right_geo=geometry.face_resistance_right_geom
    if np.any(left_geo<=0) or np.any(right_geo<=0):
        raise ValueError("coating geometry implies nonpositive thermal resistance")
    def evaluate(x):
        full=t0.ravel().copy()
        full[flat_active]=x
        vals=full[flat_active]
        k=conductivity_w_m_k.at(vals)
        cp=coating.heat_capacity.at(vals)
        h=coating.heat_capacity.integral(vals)
        g=1./(left_geo/k[ll]+right_geo/k[rr])
        flux=g*(x[ll]-x[rr])
        resid=m*(h-e0)/dt_s-q
        np.add.at(resid,ll,flux)
        np.add.at(resid,rr,-flux)
        diag=m*cp/dt_s
        np.add.at(diag,ll,g)
        np.add.at(diag,rr,g)
        rows=np.concatenate((np.arange(len(x)),ll,rr))
        cols=np.concatenate((np.arange(len(x)),rr,ll))
        data=np.concatenate((diag,-g,-g))
        jac=coo_matrix(
            (data,(rows,cols)),shape=(len(x),len(x))).tocsr()
        return resid,jac,flux,full
    x=t0.ravel()[flat_active].copy()
    tolerance=2e-9+relative_tolerance*max(1.,np.max(np.abs(q)))
    iterations=0
    for iteration in range(max_iterations):
        r,jac,flux,full=evaluate(x)
        err=float(np.max(np.abs(r)))
        if err<=tolerance:
            iterations=iteration
            break
        correction=np.asarray(spsolve(jac,-r),dtype=float)
        if not np.all(np.isfinite(correction)):
            raise ArithmeticError("invalid angular coating sparse solve")
        alpha=1.
        for _ in range(48):
            trial=x+alpha*correction
            try:
                trial_r,_,_,_=evaluate(trial)
            except ValueError:
                alpha*=.5
                continue
            new_error=float(np.max(np.abs(trial_r)))
            if new_error<=tolerance or new_error<err*(1.-1e-5*alpha):
                x=trial
                break
            alpha*=.5
        else:
            raise ArithmeticError("angular coating nonlinear line search failed")
    else:
        raise ArithmeticError("angular coating nonlinear enthalpy not converged")
    residual,_,flux,full=evaluate(x)
    final=full.reshape(t0.shape)
    store=m*(coating.heat_capacity.integral(x)-e0)
    deposited=dt_s*q
    faces=dt_s*flux
    global_residual=float(np.sum(store)-np.sum(deposited))
    # Sum of EACH assembled equal-and-opposite face contribution, not an
    # assumed all-zero constant, must cancel within floating point noise.
    net_faces=np.zeros_like(x)
    np.add.at(net_faces,ll,faces)
    np.add.at(net_faces,rr,-faces)
    cancel=float(np.sum(net_faces))
    scale=max(1.,float(np.sum(np.abs(store))),
              float(np.sum(np.abs(deposited))),
              float(np.sum(np.abs(faces))))
    max_node=float(np.max(np.abs(residual*dt_s)))
    if (max_node>2.2*tolerance*dt_s or
            abs(global_residual)>2e-9*scale or
            abs(cancel)>1e-11*scale):
        raise ArithmeticError("coating axial/azimuthal energy not conserved")
    return AngularCoatingStep(
        temperature_k=final,
        deposited_energy_j=float(np.sum(deposited)),
        enthalpy_storage_change_j=float(np.sum(store)),
        face_transfer_j=faces,
        face_kind=geometry.face_kind.copy(),
        maximum_node_residual_j=max_node,
        total_first_law_residual_j=global_residual,
        internal_flux_cancellation_j=cancel,
        iterations=iterations,
    )
