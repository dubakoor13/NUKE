"""Stage74: conservative radial+angular+axial conduction in hex-clipped ZrC sleeves.

Unlike Stage73's one-radial-cell coatings, each of the 19 sleeves has
multiple finite volumes across the true physical coating thickness.
Geometry is split by N radial shells, angular sectors and axial zones.
Shell volume equals the integral of a *clipped* circular annular sector;
no phantom full-cylinder material or inter-channel thermal bridge exists.

Internal face conductances are positive harmonic half-cell approximations:
  radial:  k * dz * dtheta / log(r_right/r_left)
  angular: k * dz * integral_(rlo)^rhi dr/r / dtheta
  axial:   k * sector_area / dz
with angular face clipping and exact cylindrical interface arc spans.

This is a standalone internal thermal conduction operator. Physical
fuel-to-outermost-shell and innermost-shell-to-H2 heat transfer from the
Stage73 contact map must be coupled in a later stage, using the SAME
radial enthalpy states. No nuclear source provenance is implied.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .coating_angular_fvm import (
    AngularCoatingMesh, _coating_sector_area_cm2, _radial_limit_cm,
)
from .coating_angular_interfaces import _angle_events
from .fuel_cutcell_conduction_2d import _inside_hex
from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .config import NervaConfig
from .layout import coolant_channel_positions
from .resolved_coating_thermal import ResolvedCoatingModel


@dataclass(frozen=True)
class RadialCoatingMesh:
    """Finite-volume unknowns [axial,channel,angle,radial_shell]."""
    radius_edges_m: np.ndarray
    sector_volume_m3: np.ndarray
    sector_mass_kg: np.ndarray
    sector_active: np.ndarray
    face_left: np.ndarray
    face_right: np.ndarray
    face_left_geom: np.ndarray
    face_right_geom: np.ndarray
    face_kind: np.ndarray  # 0:axial, 1:angular, 2:radial
    radial_area_full_m2: np.ndarray  # [channel,angle,interior radial face]
    angular_edges_rad: np.ndarray
    physical_coating_mass_kg: float


@dataclass(frozen=True)
class RadialCoatingStep:
    """Signed inter-cell heat transfer is positive left -> right."""
    temperature_k: np.ndarray
    face_heat_j: np.ndarray
    face_kind: np.ndarray
    total_storage_change_j: float
    total_deposited_energy_j: float
    global_first_law_residual_j: float
    maximum_node_residual_j: float
    assembled_face_cancellation_j: float
    iterations: int


def _arc_span_inside_hex(
    cx:float, cy:float, radius_cm:float, ap:float,
    angular_edges:np.ndarray,
) -> np.ndarray:
    """Exact angular extent [rad] of a circle within hex, per sector."""
    n=len(angular_edges)-1
    result=np.zeros(n)
    cuts=_angle_events(
        cx,cy,radius_cm,None,None,ap,angular_edges)
    for a,b in zip(cuts,cuts[1:]):
        if b-a<1e-13:
            continue
        middle=(a+b)/2.
        if _inside_hex(cx+radius_cm*math.cos(middle),
                       cy+radius_cm*math.sin(middle),ap):
            j=int(np.searchsorted(angular_edges,middle,side="right")-1)
            result[j]+=b-a
    return result


def build_radial_coating_mesh(
    config:NervaConfig,
    coating:ResolvedCoatingModel,
    angular:AngularCoatingMesh,
    *,
    radial_layers:int=3,
    density_kg_m3:float,
    radial_edges_m:object|None=None,
    max_volume_error_fraction:float=2e-7,
) -> RadialCoatingMesh:
    """Hex-clipped ZrC shell subdivision with exact total mass closure."""
    if not isinstance(radial_layers,int) or not 2<=radial_layers<=12:
        raise ValueError("radial coating layers must lie in [2,12]")
    if not math.isfinite(density_kg_m3) or density_kg_m3<=0:
        raise ValueError("radial coating density must be positive")
    if radial_edges_m is None:
        radial=np.linspace(coating.inner_radius_m,
                           coating.outer_radius_m,radial_layers+1)
    else:
        radial=np.asarray(radial_edges_m,dtype=float)
    if (radial.shape!=(radial_layers+1,)
            or not np.all(np.isfinite(radial))
            or np.any(np.diff(radial)<=0)
            or abs(radial[0]-coating.inner_radius_m)>1e-13
            or abs(radial[-1]-coating.outer_radius_m)>1e-13):
        raise ValueError("radial coating edges must cover exactly bore-to-outer radius")
    if coating.axial_edges_m is None:
        raise ValueError("radial coating requires physical axial mesh")
    dz=np.diff(np.asarray(coating.axial_edges_m,dtype=float))
    nzone=len(dz)
    ang=np.asarray(angular.angular_edges_rad,dtype=float)
    nangle=len(ang)-1
    if (nzone<1 or angular.sector_area_m2.shape!=(19,nangle)
            or not np.all(dz>0)):
        raise ValueError("radial coating angular and axial geometry mismatch")
    nrad=radial_layers
    ap=config.inner_fuel_flat_to_flat_cm/2.
    centers=coolant_channel_positions(config)
    area=np.zeros((19,nangle,nrad))
    radial_face_angle=np.zeros((19,nangle,nrad-1))
    # True clipping within the hex is integrated independently for each
    # radial layer; summing all shells recovers the parent Stage72 sector.
    for channel,(cx,cy) in enumerate(centers):
        for k in range(nrad):
            for a in range(nangle):
                area[channel,a,k]=1e-4*_coating_sector_area_cm2(
                    (cx,cy),float(ang[a]),float(ang[a+1]),
                    apothem=ap,
                    inner=radial[k]*100.,outer=radial[k+1]*100.)
        for k in range(nrad-1):
            radial_face_angle[channel,:,k]=_arc_span_inside_hex(
                cx,cy,float(radial[k+1]*100),ap,ang)
    reconstructed=area.sum(axis=2)
    if not np.allclose(reconstructed,angular.sector_area_m2,
                       rtol=max_volume_error_fraction,atol=1e-13):
        relative=float(np.max(np.abs(reconstructed-angular.sector_area_m2)/
                              np.maximum(angular.sector_area_m2,1e-15)))
        raise ArithmeticError(
            f"radial annular shells do not reproduce angular coating geometry: {relative}")
    volume=dz[:,None,None,None]*area[None,:,:,:]
    mass=volume*density_kg_m3
    if not np.allclose(mass.sum(axis=(2,3)),
                       coating.mass_kg,rtol=max_volume_error_fraction,atol=1e-10):
        raise ArithmeticError("radial coating mass is not conserved")
    active=area>1e-17
    nz,nc,na,nr=mass.shape
    node=np.arange(nz*nc*na*nr,dtype=np.int64).reshape(
        (nz,nc,na,nr))
    ll=[]
    rr=[]
    rleft=[]
    rright=[]
    kind=[]
    # Cell centers chosen as geometric radial midpoints. In peripheral
    # clipped cells, centroid offset remains a mesh-convergence error.
    centroid=np.sqrt(radial[:-1]*radial[1:])
    def push(a,b,rg1,rg2,k):
        if rg1<=0 or rg2<=0 or not math.isfinite(rg1+rg2):
            raise ArithmeticError("nonpositive physical radial coating face geometry")
        ll.append(int(a))
        rr.append(int(b))
        rleft.append(float(rg1))
        rright.append(float(rg2))
        kind.append(k)
    # Face between radial shells, limited by the real clipped angular arc.
    for iz in range(nz):
        for ch in range(nc):
            for ia in range(na):
                for k in range(nr-1):
                    dtheta=radial_face_angle[ch,ia,k]
                    if (dtheta<=1e-13 or not active[ch,ia,k]
                            or not active[ch,ia,k+1]):
                        continue
                    face=radial[k+1]
                    factor=1./(dz[iz]*dtheta)
                    push(node[iz,ch,ia,k],node[iz,ch,ia,k+1],
                         factor*math.log(face/centroid[k]),
                         factor*math.log(centroid[k+1]/face),2)
    # Angular face within each radial layer. The clipped radial integral
    # is log(rmax(theta_face)/rmin) truncated by the hex and layer edges.
    for iz in range(nz):
        for ch,(cx,cy) in enumerate(centers):
            for ia in range(na):
                ib=(ia+1)%na
                theta=float(ang[ia+1])
                allowed=_radial_limit_cm(
                    cx,cy,theta,
                    apothem=ap,outer_radius=float(radial[-1]*100))*0.01
                for k in range(nr):
                    if not(active[ch,ia,k] and active[ch,ib,k]):
                        continue
                    permitted=min(allowed,radial[k+1])
                    if permitted<=radial[k]+1e-14:
                        continue
                    geom=math.log(permitted/radial[k])
                    half=(ang[ia+1]-ang[ia])/(2*dz[iz]*geom)
                    push(node[iz,ch,ia,k],node[iz,ch,ib,k],
                         half,half,1)
    # Axial face uses each radial+angular sector's true cross-section.
    for iz in range(nz-1):
        for ch in range(nc):
            for ia in range(na):
                for k in range(nr):
                    a=area[ch,ia,k]
                    if a<=1e-17:
                        continue
                    push(node[iz,ch,ia,k],node[iz+1,ch,ia,k],
                         dz[iz]/(2*a),dz[iz+1]/(2*a),0)
    return RadialCoatingMesh(
        radius_edges_m=radial,sector_volume_m3=volume,
        sector_mass_kg=mass,sector_active=active,
        face_left=np.asarray(ll,dtype=np.int64),
        face_right=np.asarray(rr,dtype=np.int64),
        face_left_geom=np.asarray(rleft,dtype=float),
        face_right_geom=np.asarray(rright,dtype=float),
        face_kind=np.asarray(kind,dtype=np.int8),
        radial_area_full_m2=radial_face_angle*radial[1:-1][None,None,:]*
            np.sum(dz),angular_edges_rad=ang.copy(),
        physical_coating_mass_kg=float(np.sum(mass)),
    )


def distribute_angular_power_to_radial_layers(
    coating_power_w:object,mesh:RadialCoatingMesh,
) -> np.ndarray:
    """Conservative uniform volumetric assumption, NOT measured OpenMC data."""
    p=np.asarray(coating_power_w,dtype=float)
    mass=mesh.sector_mass_kg
    if (p.shape!=mass.shape[:-1] or not np.all(np.isfinite(p))
            or np.any(p<0)):
        raise ValueError("angular coating input must be nonnegative finite [z,19,theta]")
    frac=mass/mass.sum(axis=-1,keepdims=True)
    out=p[...,None]*frac
    if not np.allclose(out.sum(axis=-1),p,atol=1e-11,rtol=1e-13):
        raise ArithmeticError("radial coating nuclear source partition not conserved")
    return out


def advance_radial_coating_enthalpy(
    coating:ResolvedCoatingModel,
    mesh:RadialCoatingMesh,
    initial_temperature_k:object,
    *,
    conductivity_w_m_k:TemperatureTable,
    dt_s:float,
    deposition_w:object,
    max_iterations:int=45,
    relative_tolerance:float=1e-11,
) -> RadialCoatingStep:
    """Solve 3D radial+angular+axial diffusion with exact cp(T) enthalpy."""
    if (not math.isfinite(dt_s) or dt_s<=0 or
            not isinstance(max_iterations,int) or max_iterations<1 or
            not math.isfinite(relative_tolerance) or relative_tolerance<=0):
        raise ValueError("radial coating time step/solver tolerances invalid")
    if not isinstance(conductivity_w_m_k,TemperatureTable):
        raise ValueError("radial coating conductivity requires bounded k(T)")
    t=np.asarray(initial_temperature_k,dtype=float)
    q=np.asarray(deposition_w,dtype=float)
    m=mesh.sector_mass_kg
    if (t.shape!=m.shape or q.shape!=m.shape
            or not np.all(np.isfinite(t)) or not np.all(np.isfinite(q))):
        raise ValueError("radial coating state and power require finite [z,19,theta,r]")
    active=np.broadcast_to(mesh.sector_active[None,:,:,:],t.shape)
    if np.any(np.abs(q[~active])>1e-13):
        raise ValueError("radial coating deposition in nonphysical material")
    if np.any(m[active]<=0) or np.any(m[~active]!=0):
        raise ValueError("radial coating cell mass must be physically positive or zero")
    coating.heat_capacity.at(t)
    conductivity_w_m_k.at(t)
    flat=np.flatnonzero(active.ravel())
    lookup=np.full(t.size,-1,dtype=np.int64)
    lookup[flat]=np.arange(len(flat))
    left=lookup[mesh.face_left]
    right=lookup[mesh.face_right]
    if np.any(left<0) or np.any(right<0):
        raise ArithmeticError("radial coating face references an empty cell")
    mactive=m.ravel()[flat]
    power=q.ravel()[flat]
    old_h=coating.heat_capacity.integral(t).ravel()[flat]
    gl=mesh.face_left_geom
    gr=mesh.face_right_geom
    def evaluate(x):
        k=conductivity_w_m_k.at(x)
        cp=coating.heat_capacity.at(x)
        h=coating.heat_capacity.integral(x)
        g=1./(gl/k[left]+gr/k[right])
        flux=g*(x[left]-x[right])
        residual=mactive*(h-old_h)/dt_s-power
        np.add.at(residual,left,flux)
        np.add.at(residual,right,-flux)
        diagonal=mactive*cp/dt_s
        np.add.at(diagonal,left,g)
        np.add.at(diagonal,right,g)
        idx=np.arange(x.size)
        jac=coo_matrix(
            (np.concatenate((diagonal,-g,-g)),
             (np.concatenate((idx,left,right)),
              np.concatenate((idx,right,left)))),
            shape=(x.size,x.size)).tocsr()
        return residual,jac,flux
    x=t.ravel()[flat].copy()
    tolerance=2e-9+relative_tolerance*max(1.,float(np.max(np.abs(power))))
    iterations=0
    for iteration in range(max_iterations):
        res,jac,flux=evaluate(x)
        norm=float(np.max(np.abs(res)))
        if norm<=tolerance:
            iterations=iteration
            break
        direction=np.asarray(spsolve(jac,-res),dtype=float)
        if not np.all(np.isfinite(direction)):
            raise ArithmeticError("radial coating nonlinear solve failed")
        alpha=1.
        for _ in range(48):
            trial=x+alpha*direction
            try:
                candidate,_,_=evaluate(trial)
            except ValueError:
                alpha*=.5
                continue
            new=float(np.max(np.abs(candidate)))
            if new<=tolerance or new<norm*(1.-1e-5*alpha):
                x=trial
                break
            alpha*=.5
        else:
            raise ArithmeticError("radial coating Newton line search failed")
    else:
        raise ArithmeticError("radial coating nonlinear solve did not converge")
    residual,_,flux=evaluate(x)
    final=t.ravel().copy()
    final[flat]=x
    storage=mactive*(coating.heat_capacity.integral(x)-old_h)
    deposited=power*dt_s
    face_j=flux*dt_s
    nodal=np.zeros_like(x)
    np.add.at(nodal,left,face_j)
    np.add.at(nodal,right,-face_j)
    cancellation=float(nodal.sum())
    balance=float(storage.sum()-deposited.sum())
    scale=max(1.,float(np.abs(storage).sum()),float(np.abs(deposited).sum()),
              float(np.abs(face_j).sum()))
    maximum=float(np.max(np.abs(residual*dt_s)))
    if (maximum>2.2*tolerance*dt_s or
            abs(balance)>2e-9*scale or abs(cancellation)>1e-11*scale):
        raise ArithmeticError("radial coating first-law / face cancellation failed")
    return RadialCoatingStep(
        temperature_k=final.reshape(t.shape),
        face_heat_j=face_j,face_kind=mesh.face_kind.copy(),
        total_storage_change_j=float(storage.sum()),
        total_deposited_energy_j=float(deposited.sum()),
        global_first_law_residual_j=balance,
        maximum_node_residual_j=maximum,
        assembled_face_cancellation_j=cancellation,
        iterations=iterations,
    )
