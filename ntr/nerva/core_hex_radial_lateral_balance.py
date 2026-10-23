"""Independent per-cell first-law audit for the unified radial/axial/lateral core.

Reconstruct Fourier face flux from final temperature data and exact geometric
face resistances, *without* reading the solver's assembled matrix or relying
on its heat-flux outputs. One shared lateral face is inserted with opposite
signs into the two neighboring outer fuel cells. Returns residuals in W.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .core_hex_radial_lateral_fv import (
    RadialHexMesh, RadialHexState, RadialHexStep, _solid_sources,
)
from .core_hex_lateral_fv import HexCoolantBoundary
from .core_conjugate_fv import CoreSolidLaw, _gas_enthalpy


@dataclass(frozen=True)
class RadialHexEnergyAudit:
    solid_cell_residual_w: np.ndarray
    hydrogen_cell_residual_w: np.ndarray
    independent_lateral_face_w: np.ndarray
    independent_wall_heat_w: np.ndarray
    global_residual_w: float
    max_cell_residual_w: float
    max_relative_cell_residual: float
    global_relative_residual: float


def audit_radial_hex_energy(
    mesh:RadialHexMesh,
    before:RadialHexState,
    after:RadialHexStep,
    gas:HexCoolantBoundary,
    fuel:CoreSolidLaw,
    coating:CoreSolidLaw,
    nuclear_w:np.ndarray,
    *,
    dt_s:float|None,
    contact_resistance_m2_k_w:float=0.0,
)->RadialHexEnergyAudit:
    mesh.validate();fuel.validate();coating.validate()
    nz,ns,nr=mesh.shape
    gas.validate(nz)
    if (dt_s is not None and (not math.isfinite(dt_s) or dt_s<=0)
            or not math.isfinite(contact_resistance_m2_k_w)
            or contact_resistance_m2_k_w<0):
        raise ValueError("invalid radial-hex audit timestep/contact")
    initial_s=np.asarray(before.solid_temperature_k,dtype=float)
    current_s=np.asarray(after.state.solid_temperature_k,dtype=float)
    initial_h=np.asarray(before.hydrogen_temperature_k,dtype=float)
    current_h=np.asarray(after.state.hydrogen_temperature_k,dtype=float)
    if (initial_s.shape!=(nz,ns,nr) or current_s.shape!=(nz,ns,nr)
            or initial_h.shape!=(nz,ns) or current_h.shape!=(nz,ns)
            or any(not np.isfinite(a).all() or np.min(a)<=0 for a in
                   (initial_s,current_s,initial_h,current_h))):
        raise ValueError("invalid temperature states in independent audit")
    Q=_solid_sources(mesh,nuclear_w)
    dz=np.diff(mesh.z_edges_m)
    edges=mesh.radial_edges_m
    center=np.sqrt((edges[:,:-1]**2+edges[:,1:]**2)*0.5)
    area=np.pi*np.diff(edges**2,axis=1)
    volumes=mesh.volumes_m3
    laws=[coating if j<mesh.coating_cells else fuel for j in range(nr)]
    k=np.empty((nz,ns,nr))
    for j,law in enumerate(laws):
        k[:,:,j]=np.array(
            [law.conductivity(float(t)) for t in current_s[:,:,j].flat]
        ).reshape(nz,ns)
    radial=np.zeros((nz,ns,nr+1))
    wall=np.zeros((nz,ns))
    for i in range(nz):
        for c in range(ns):
            bore=edges[c,0]
            R=(math.log(center[c,0]/bore)/(2*math.pi*k[i,c,0]*dz[i])
               +1/(gas.film_w_m2_k[i,c]*2*math.pi*bore*dz[i]))
            wall[i,c]=(current_s[i,c,0]-current_h[i,c])/R
            radial[i,c,0]=-wall[i,c]
            for j in range(1,nr):
                face=edges[c,j]
                R=(math.log(face/center[c,j-1])/k[i,c,j-1]
                   +math.log(center[c,j]/face)/k[i,c,j])
                if j==mesh.coating_cells:
                    R+=contact_resistance_m2_k_w/face
                radial[i,c,j]=(current_s[i,c,j-1]-current_s[i,c,j]
                                )*2*math.pi*dz[i]/R
    axial=np.zeros((nz+1,ns,nr))
    for i in range(1,nz):
        R=(dz[i-1]/(2*k[i-1,:,:])+dz[i]/(2*k[i,:,:]))
        axial[i,:,:]=(current_s[i-1,:,:]-current_s[i,:,:])*area/R
    pair=mesh.network.shared_pair_indices
    lateral=np.zeros((nz,len(pair)))
    lateral_out=np.zeros((nz,ns))
    for e,(a,b) in enumerate(pair):
        a=int(a);b=int(b)
        k_a=k[:,a,-1]; k_b=k[:,b,-1]
        resistance=(0.5*mesh.network.channel_pitch_pair_m[e]/k_a+
                    0.5*mesh.network.channel_pitch_pair_m[e]/k_b)
        conductance=mesh.network.shared_face_lengths_m[e]*dz/resistance
        lateral[:,e]=conductance*(current_s[:,a,-1]-current_s[:,b,-1])
        lateral_out[:,a]+=lateral[:,e]
        lateral_out[:,b]-=lateral[:,e]
    storing=np.zeros_like(current_s)
    gas_storing=np.zeros_like(current_h)
    enthalpy_b=gas.gas_boundary()
    if dt_s is not None:
        for i in range(nz):
            for c in range(ns):
                for j,law in enumerate(laws):
                    storing[i,c,j]=(
                        volumes[i,c,j]*law.density_kg_m3*
                        law.enthalpy_change(
                            float(initial_s[i,c,j]),float(current_s[i,c,j]))
                        /dt_s)
                gas_storing[i,c]=(
                    gas.hydrogen_mass_kg[i,c]*
                    (_gas_enthalpy(float(current_h[i,c]),enthalpy_b)-
                     _gas_enthalpy(float(initial_h[i,c]),enthalpy_b))/dt_s)
    lateral_per_cell=np.zeros_like(current_s)
    lateral_per_cell[:,:,-1]=lateral_out
    solid_res=(
        storing+radial[:,:,1:]-radial[:,:,:-1]+
        axial[1:,:,:]-axial[:-1,:,:]+lateral_per_cell-Q
    )
    adv=np.empty((nz,ns))
    for i in range(nz):
        for c in range(ns):
            inlet=(float(gas.inlet_temperature_k[c]) if i==0
                   else float(current_h[i-1,c]))
            adv[i,c]=gas.mdot_kg_s[c]*(
                _gas_enthalpy(float(current_h[i,c]),enthalpy_b)-
                _gas_enthalpy(inlet,enthalpy_b))
    hres=gas_storing+adv-wall-gas.direct_hydrogen_power_w
    if (not np.allclose(radial,after.radial_outward_heat_w,rtol=1e-7,atol=1e-7)
            or not np.allclose(lateral,after.lateral_face_heat_w,
                               rtol=1e-7,atol=1e-7)
            or not np.allclose(axial,after.axial_positive_heat_w,
                               rtol=1e-7,atol=1e-7)
            or not np.allclose(wall,after.wall_to_hydrogen_w,
                               rtol=1e-7,atol=1e-7)):
        raise RuntimeError("independently reconstructed core face flux mismatch")
    scale_s=np.maximum.reduce((
        np.ones_like(Q),np.abs(Q),np.abs(storing),
        np.abs(radial[:,:,1:]),np.abs(radial[:,:,:-1]),
        np.abs(axial[1:,:,:]),np.abs(axial[:-1,:,:]),
        np.abs(lateral_per_cell)))
    scale_g=np.maximum.reduce((
        np.ones_like(hres),np.abs(gas_storing),np.abs(adv),
        np.abs(wall),np.abs(gas.direct_hydrogen_power_w)))
    scaled=max(float(np.max(np.abs(solid_res)/scale_s)),
               float(np.max(np.abs(hres)/scale_g)))
    globally=float(np.sum(solid_res,dtype=np.float64)+
                   np.sum(hres,dtype=np.float64))
    gross=max(1.0,float(np.sum(np.abs(Q)))+
        float(np.sum(np.abs(gas.direct_hydrogen_power_w)))+
        float(np.sum(np.abs(storing)))+
        float(np.sum(np.abs(gas_storing))))
    return RadialHexEnergyAudit(
        solid_res,hres,lateral,wall,globally,
        max(float(np.max(np.abs(solid_res))),float(np.max(np.abs(hres)))),
        scaled,abs(globally)/gross)
