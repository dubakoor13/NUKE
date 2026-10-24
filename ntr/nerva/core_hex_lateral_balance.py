"""Read-only per-volume first-law check for 19-sector coupled core physics.

Each Voronoi face is reconstructed as a single signed Fourier flux. This
audit does not reuse the solver matrix, test OpenMC transport, or replace the
need for spatial thermal and nuclear-data physics qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .core_conjugate_fv import CoreSolidLaw, _gas_enthalpy
from .core_hex_lateral_fv import (
    HexSectorNetwork,HexCoreState,HexCoreStep,HexCoolantBoundary,
)


@dataclass(frozen=True)
class HexCoreEnergyAudit:
    fuel_cell_residual_w: np.ndarray
    coating_cell_residual_w: np.ndarray
    hydrogen_cell_residual_w: np.ndarray
    lateral_face_heat_w: np.ndarray
    maximum_cell_residual_w: float
    maximum_scaled_cell_residual: float
    global_residual_w: float
    global_scaled_residual: float
    hotspot_lateral_exchange_w: np.ndarray
    "Not nuclear physics validation"


def audit_hex_core_cell_energy(
    geometry: HexSectorNetwork,
    z_edges_m: np.ndarray,
    previous: HexCoreState,
    result: HexCoreStep,
    gas: HexCoolantBoundary,
    fuel: CoreSolidLaw,
    coat: CoreSolidLaw,
    prescribed_fuel_power_w: np.ndarray,
    *,
    dt_s: float | None,
    contact_resistance_m2_k_w: float=0.0,
)->HexCoreEnergyAudit:
    geometry.validate();fuel.validate();coat.validate()
    z=np.asarray(z_edges_m,dtype=float)
    if (z.ndim!=1 or z.size<2 or np.any(np.diff(z)<=0)
            or not np.isfinite(z).all()):
        raise ValueError("invalid axial mesh in physical ledger")
    nz=len(z)-1
    gas.validate(nz)
    if (dt_s is not None and (not math.isfinite(dt_s) or dt_s<=0)
            or not math.isfinite(contact_resistance_m2_k_w)
            or contact_resistance_m2_k_w<0):
        raise ValueError("invalid energy audit configuration")
    states=[previous.fuel_temperature_k,previous.coating_temperature_k,
            previous.hydrogen_temperature_k,
            result.state.fuel_temperature_k,result.state.coating_temperature_k,
            result.state.hydrogen_temperature_k]
    vals=[np.asarray(x,dtype=float) for x in states]
    if any(x.shape!=(nz,19) or not np.isfinite(x).all() or np.any(x<=0)
           for x in vals):
        raise ValueError("nonphysical saved temperature fields")
    fo,co,go,fn,cn,gn=vals
    q=np.asarray(prescribed_fuel_power_w,dtype=float)
    if q.shape!=(nz,19) or not np.isfinite(q).all():
        raise ValueError("invalid nuclear power field")
    dz=np.diff(z)
    r0=geometry.coolant_bore_radius_m
    r1=geometry.coating_outer_radius_m
    area_f=geometry.fuel_area_m2
    area_c=math.pi*(r1*r1-r0*r0)
    rc=math.sqrt((r0*r0+r1*r1)/2)
    radius_f=np.sqrt(r1*r1+area_f/math.pi)
    rfm=np.sqrt((r1*r1+radius_f*radius_f)/2)
    kf=np.asarray([[fuel.conductivity(float(t)) for t in row] for row in fn])
    kc=np.asarray([[coat.conductivity(float(t)) for t in row] for row in cn])
    ff=np.zeros((nz,len(geometry.shared_pair_indices)),dtype=float)
    lateral=np.zeros((nz,19),dtype=float)
    for edge,(j,m) in enumerate(geometry.shared_pair_indices):
        j=int(j);m=int(m)
        G=(geometry.shared_face_lengths_m[edge]*dz /
           (0.5*geometry.channel_pitch_pair_m[edge]/kf[:,j]+
            0.5*geometry.channel_pitch_pair_m[edge]/kf[:,m]))
        ff[:,edge]=G*(fn[:,j]-fn[:,m])
        lateral[:,j]+=ff[:,edge]
        lateral[:,m]-=ff[:,edge]
    qrad=np.zeros((nz,19))
    qgas=np.zeros((nz,19))
    hfilm=np.asarray(gas.film_w_m2_k)
    for i in range(nz):
        for j in range(19):
            resistance=(math.log(rfm[j]/r1)/(2*math.pi*kf[i,j]*dz[i])+
                        math.log(r1/rc)/(2*math.pi*kc[i,j]*dz[i])+
                        contact_resistance_m2_k_w/(2*math.pi*r1*dz[i]))
            wall_resistance=(
                math.log(rc/r0)/(2*math.pi*kc[i,j]*dz[i])+
                1/(hfilm[i,j]*2*math.pi*r0*dz[i]))
            qrad[i,j]=(fn[i,j]-cn[i,j])/resistance
            qgas[i,j]=(cn[i,j]-gn[i,j])/wall_resistance
    axial_f=np.zeros((nz+1,19))
    axial_c=np.zeros((nz+1,19))
    for i in range(1,nz):
        rf=dz[i-1]/(2*kf[i-1,:])+dz[i]/(2*kf[i,:])
        rc2=dz[i-1]/(2*kc[i-1,:])+dz[i]/(2*kc[i,:])
        axial_f[i,:]=area_f*(fn[i-1,:]-fn[i,:])/rf
        axial_c[i,:]=area_c*(cn[i-1,:]-cn[i,:])/rc2
    store_f=np.zeros((nz,19))
    store_c=np.zeros((nz,19))
    store_g=np.zeros((nz,19))
    gas_book=gas.gas_boundary()
    if dt_s is not None:
        for i in range(nz):
            for j in range(19):
                store_f[i,j]=(fuel.density_kg_m3*area_f[j]*dz[i]*
                        fuel.enthalpy_change(float(fo[i,j]),float(fn[i,j]))/dt_s)
                store_c[i,j]=(coat.density_kg_m3*area_c*dz[i]*
                        coat.enthalpy_change(float(co[i,j]),float(cn[i,j]))/dt_s)
                store_g[i,j]=(gas.hydrogen_mass_kg[i,j]*
                    (_gas_enthalpy(float(gn[i,j]),gas_book)-
                     _gas_enthalpy(float(go[i,j]),gas_book))/dt_s)
    advection=np.zeros((nz,19))
    for i in range(nz):
        for j in range(19):
            upstream=(float(gas.inlet_temperature_k[j]) if i==0
                      else float(gn[i-1,j]))
            advection[i,j]=(gas.mdot_kg_s[j]*
                (_gas_enthalpy(float(gn[i,j]),gas_book)-
                 _gas_enthalpy(upstream,gas_book)))
    fuel_res=(store_f+lateral+
              axial_f[1:,:]-axial_f[:-1,:]+qrad-q)
    coat_res=(store_c+axial_c[1:,:]-axial_c[:-1,:]+qgas-qrad)
    gas_res=(store_g+advection-qgas-gas.direct_hydrogen_power_w)
    if (not np.allclose(ff,result.lateral_face_heat_w,rtol=1e-7,atol=1e-7)
            or not np.allclose(qrad,result.fuel_to_coating_w,rtol=1e-7,atol=1e-7)
            or not np.allclose(qgas,result.coating_to_hydrogen_w,rtol=1e-7,atol=1e-7)):
        raise RuntimeError("physical face fluxes differ from solve output")
    scale_f=np.maximum.reduce((np.ones_like(q),np.abs(q),
            np.abs(store_f),np.abs(lateral),np.abs(qrad),
            np.abs(axial_f[1:,:]),np.abs(axial_f[:-1,:])))
    scale_c=np.maximum.reduce((np.ones_like(q),np.abs(store_c),
            np.abs(qrad),np.abs(qgas),
            np.abs(axial_c[1:,:]),np.abs(axial_c[:-1,:])))
    scale_g=np.maximum.reduce((np.ones_like(q),np.abs(store_g),
            np.abs(advection),np.abs(qgas),
            np.abs(gas.direct_hydrogen_power_w)))
    scaled=max(float(np.max(np.abs(fuel_res)/scale_f)),
               float(np.max(np.abs(coat_res)/scale_c)),
               float(np.max(np.abs(gas_res)/scale_g)))
    global_balance=float(
        np.sum(fuel_res,dtype=np.float64)+
        np.sum(coat_res,dtype=np.float64)+
        np.sum(gas_res,dtype=np.float64))
    norm=max(1.,float(np.sum(np.abs(q)))+
                  float(np.sum(np.abs(gas.direct_hydrogen_power_w)))+
                  float(np.sum(np.abs(store_f)))+
                  float(np.sum(np.abs(store_c)))+
                  float(np.sum(np.abs(store_g))))
    return HexCoreEnergyAudit(
        fuel_res,coat_res,gas_res,ff,
        max(float(np.max(np.abs(x))) for x in (fuel_res,coat_res,gas_res)),
        scaled,global_balance,abs(global_balance)/norm,lateral
    )
