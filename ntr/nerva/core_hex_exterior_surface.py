"""External fuel-element perimeter conduction/contact and thermal radiation.

Extract *true boundary edges* from the conforming fuel/coating triangle mesh,
excluding the 19 internal hydrogen coolant bore walls. The exposed outer
surface is explicitly distinguished from internal fuel/ZrC interfaces.

The optional external heat flux consists of two physical mechanisms:

  q_out = h_contact * (T_wall - T_reservoir)
        + epsilon * sigma * (T_wall**4 - T_radiative_environment**4).

The second term is linearized with the exact secant coefficient

  h_rad(T,T_env) = epsilon*sigma*(T+T_env)*(T*T+T_env*T_env).

This gives a nonnegative outward heat-transfer operator and a physically
meaningful energy ledger for the outer hex surfaces in steady or transient
FEM calculations. Zero contact conductance and emissivity = adiabatic.

'Contact' here is a boundary to an *externally prescribed temperature field*,
not a solved two-element interface. It supports externally supplied neighbor
temperatures but does NOT independently assert a self-consistent reactor-array
contact calculation or historical material properties. In particular, the
optional external perimeter ZrC skin remains a separate geometry issue.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .core_hex_conforming_mesh import HexTriangleSection

STEFAN_BOLTZMANN_W_M2_K4=5.670374419e-8


def exterior_fuel_edges(section:HexTriangleSection)->np.ndarray:
    """Identify external hex edges, never coolant holes or material interfaces.

    Returns globally indexed [n_exposed,2] triangle edges; orientations
    are irrelevant for scalar Robin and radiation surface integrals.
    """
    section.validate()
    tri=np.asarray(section.triangles,dtype=np.int64)
    edges=np.sort(
        np.concatenate((tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]])),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True)
    if np.any(counts>2):
        raise ValueError("non-manifold fuel/coating mesh")
    exposed=unique[counts==1]
    wall=np.sort(np.asarray(section.wall_edges,dtype=np.int64),axis=1)
    wall_set={tuple(int(x) for x in row) for row in wall}
    out=np.asarray(
        [row for row in exposed if tuple(int(x) for x in row) not in wall_set],
        dtype=np.int64).reshape(-1,2)
    if len(out)==0:
        raise ValueError("no external fuel-element boundary edges")
    # Every exterior edge must be a non-hydrogen one-owner triangle edge;
    # excluding a coolant wall must not accidentally remove its neighbor.
    if len(out)+len(wall_set)!=len(exposed):
        raise ValueError("duplicate or unmatched external/coolant boundaries")
    return out


@dataclass(frozen=True)
class CoreExteriorHeatBoundary:
    """Bath-contact + greybody radiation along exposed outer hex edges.

    Fields are either physical scalars or array [n_axial_bins,n_outer_edges].
    All temperatures absolute Kelvin; conductance W/(m2 K);
    greybody emissivity dimensionless [0..1]. A single external radiation
    environment at one temperature is an explicit approximation.
    """
    contact_conductance_w_m2_k: float|np.ndarray=0.
    contact_reservoir_temperature_k: float|np.ndarray=500.
    emissivity: float|np.ndarray=0.
    radiation_environment_temperature_k: float|np.ndarray=500.

    def expanded(self,nz:int,nedge:int
                 )->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        if nz<1 or nedge<1:
            raise ValueError("external boundary must have nonempty geometry")
        arrays=[]
        names=("contact conductance","contact reservoir temperature",
               "emissivity","radiative environment temperature")
        for name,value in zip(names,(
            self.contact_conductance_w_m2_k,
            self.contact_reservoir_temperature_k,
            self.emissivity,
            self.radiation_environment_temperature_k)):
            raw=np.asarray(value,dtype=float)
            if raw.ndim==0:
                a=np.full((nz,nedge),float(raw))
            elif raw.shape==(nz,nedge):
                a=raw.copy()
            else:
                raise ValueError(
                    f"{name} must be scalar or [nz,n_exterior_edges]")
            if not np.isfinite(a).all():
                raise ValueError("nonfinite external surface "+name)
            arrays.append(a)
        hc,Tc,eps,Tr=arrays
        if (np.any(hc<0) or np.any(Tc<=0)
                or np.any(eps<0) or np.any(eps>1)
                or np.any(Tr<=0)):
            raise ValueError("nonphysical external contact or radiation field")
        return hc,Tc,eps,Tr


def exterior_heat_coefficients(
    solid_temperature:np.ndarray,
    z_edges_m:np.ndarray,
    section:HexTriangleSection,
    exterior:CoreExteriorHeatBoundary,
    *, edges:np.ndarray|None=None,
)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Return real surface wall conductance and effective bath temperature.

    The radiative heat-flux factorization is exact for an isothermal
    surface. Within one P1 prism face, the code evaluates the secant at
    the area-average wall temperature (spatial quadrature approximation).
    'bath temperature' is the conductance-weighted combination of contact
    and radiation environments, and is irrelevant when total h=0.
    """
    edges=(exterior_fuel_edges(section) if edges is None
           else np.asarray(edges,dtype=np.int64))
    z=np.asarray(z_edges_m,dtype=float)
    T=np.asarray(solid_temperature,dtype=float)
    if (z.ndim!=1 or z.size<2 or np.any(np.diff(z)<=0)
            or T.shape!=(len(z),len(section.xy_m))
            or not np.isfinite(T).all() or np.any(T<=0)):
        raise ValueError("invalid exterior surface temperature/axial state")
    nz=len(z)-1
    hc,Tc,eps,Tr=exterior.expanded(nz,len(edges))
    # Mean of all 4 bilinear face nodes is its exact FE surface integral.
    Te=0.25*(T[:-1,edges[:,0]]+T[:-1,edges[:,1]]
             +T[1:,edges[:,0]]+T[1:,edges[:,1]])
    hrad=eps*STEFAN_BOLTZMANN_W_M2_K4 * (
        (Te+Tr)*(Te*Te+Tr*Tr))
    htot=hc+hrad
    scaled=hc*Tc+hrad*Tr
    Tref=np.divide(scaled,htot,out=np.full_like(htot,500.),
                   where=htot>0)
    if not np.isfinite(htot).all() or not np.isfinite(Tref).all():
        raise ValueError("overflow in nonlinear boundary conductance")
    return edges,htot,Tref,Te,hc*(Te-Tc)+eps*(
        STEFAN_BOLTZMANN_W_M2_K4)*(Te**4-Tr**4)


def exterior_boundary_power(
    solid_temperature:np.ndarray,
    z_edges_m:np.ndarray,
    section:HexTriangleSection,
    exterior:CoreExteriorHeatBoundary,
)->tuple[np.ndarray,float]:
    """Signed [z,outer_edge] outward watts and total external loss watts."""
    edges,coeff,Tref,mean_t,flux=exterior_heat_coefficients(
        solid_temperature,z_edges_m,section,exterior)
    side=np.linalg.norm(section.xy_m[edges[:,1]]-
                        section.xy_m[edges[:,0]],axis=1)
    area=np.diff(z_edges_m)[:,None]*side[None,:]
    power=flux*area
    return power,float(np.sum(power,dtype=np.float64))
