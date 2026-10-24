"""Unified radial + axial + lateral conjugate energy solver for 19 coolant sectors.

Every channel carries an independently meshed cylindrical ZrC coating and
volumetrically heated fuel annulus. Each fuel/coating radial cell conducts
axially. Every neighboring pair of channels shares an equal-and-opposite
transverse Fourier flux coupling their *outermost* fuel radial cells.
Each coated inner bore exchanges heat with its own axially advected hydrogen.

The lateral link is a conservative, two-point *subgrid closure* based on the
true clipped Voronoi shared-face length and channel-site separation. Joining
the annular outer fuel cells does NOT make these cells true hexagonal XY
volumes. Unlike the previous pair of alternative solvers, this system has
radial, axial and transverse exchange in a single nonlinear sparse solve.
A true XY-conforming, circumferentially resolved 3-D finite-volume fuel
mesh, qualified fuel properties, local OpenMC KERMA, and reactive hydrogen
thermochemistry remain separate unimplemented capabilities.

Unknowns: solid T[z,19,r] and H2 T[z,19], in K. Prescribed nuclear source
power W is NOT calculated from keff or other reactor design information.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .core_conjugate_fv import (
    CoreSectorGrid, CoreSolidLaw, _gas_enthalpy, _gas_secant,
)
from .core_hex_lateral_fv import HexCoolantBoundary, HexSectorNetwork


@dataclass(frozen=True)
class RadialHexMesh:
    z_edges_m: np.ndarray
    radial_edges_m: np.ndarray      # [19, nrad+1]
    coating_cells: int
    network: HexSectorNetwork

    @classmethod
    def from_network(cls, network: HexSectorNetwork, z_edges_m: np.ndarray,
                     *, coating_cells: int = 2,
                     fuel_cells: int = 8) -> "RadialHexMesh":
        network.validate()
        meshes = [
            CoreSectorGrid.equivalent_annulus(
                z_edges_m, bore_radius_m=network.coolant_bore_radius_m,
                coating_thickness_m=(
                    network.coating_outer_radius_m-network.coolant_bore_radius_m),
                fuel_sector_area_m2=float(area),
                coating_cells=coating_cells, fuel_cells=fuel_cells)
            for area in network.fuel_area_m2
        ]
        ans=cls(np.asarray(z_edges_m, dtype=float),
                np.stack([x.r_edges_m for x in meshes]),
                coating_cells, network)
        ans.validate()
        return ans

    def validate(self) -> None:
        self.network.validate()
        z=np.asarray(self.z_edges_m, dtype=float)
        r=np.asarray(self.radial_edges_m, dtype=float)
        if (z.ndim!=1 or z.size<2 or not np.isfinite(z).all()
                or np.any(np.diff(z)<=0) or r.ndim!=2 or r.shape[0]!=19
                or r.shape[1]<3 or not np.isfinite(r).all()
                or np.any(np.diff(r,axis=1)<=0)
                or not 1<=self.coating_cells<r.shape[1]-1
                or not np.allclose(r[:,0],self.network.coolant_bore_radius_m,
                                   rtol=0,atol=1e-12)
                or not np.allclose(
                    r[:,self.coating_cells],
                    self.network.coating_outer_radius_m,rtol=0,atol=1e-12)):
            raise ValueError("invalid shared axial / channel radial FV mesh")

    @property
    def shape(self)->tuple[int,int,int]:
        return (len(self.z_edges_m)-1,19,self.radial_edges_m.shape[1]-1)

    @property
    def volumes_m3(self)->np.ndarray:
        dz=np.diff(self.z_edges_m)
        cross=np.pi*np.diff(self.radial_edges_m**2,axis=1)
        return dz[:,None,None]*cross[None,:,:]


@dataclass(frozen=True)
class RadialHexState:
    solid_temperature_k: np.ndarray       # [nz,19,nrad]
    hydrogen_temperature_k: np.ndarray    # [nz,19]


@dataclass(frozen=True)
class RadialHexStep:
    state: RadialHexState
    radial_outward_heat_w: np.ndarray      # [nz,19,nrad+1]; inner boundary negative
    lateral_face_heat_w: np.ndarray        # [nz,npairs]; pair i->j
    axial_positive_heat_w: np.ndarray      # [nz+1,19,nrad]
    wall_to_hydrogen_w: np.ndarray         # [nz,19]
    stored_solid_energy_j: float
    stored_hydrogen_energy_j: float
    inlet_enthalpy_w: float
    outlet_enthalpy_w: float
    first_law_residual: float              # J transient, W steady
    iterations: int
    transient: bool


def _solid_sources(mesh:RadialHexMesh,
                   nuclear_w:np.ndarray)->np.ndarray:
    nz,ns,nr=mesh.shape
    p=np.asarray(nuclear_w,dtype=float)
    if p.shape==(nz,ns,nr):
        ans=p.copy()
    elif p.shape==(nz,ns):
        vv=mesh.volumes_m3.copy()
        vv[:,:,:mesh.coating_cells]=0
        ans=p[:,:,None]*vv/vv.sum(axis=2)[:,:,None]
    else:
        raise ValueError("nuclear solid power must be [z,19] or [z,19,nrad] W")
    if not np.isfinite(ans).all() or np.any(ans<0):
        raise ValueError("solid nuclear source power must be finite and >=0")
    return ans


def _face_conductance(mesh:RadialHexMesh,
                      k:np.ndarray,
                      film:np.ndarray,
                      contact_resistance:float
                      )->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Conservative positive cell-face conductances, SI W/K."""
    nz,ns,nr=mesh.shape
    r=mesh.radial_edges_m
    dz=np.diff(mesh.z_edges_m)
    rc=np.sqrt(0.5*(r[:,1:]**2+r[:,:-1]**2))
    radial=np.zeros((nz,ns,nr-1))
    for j in range(nr-1):
        face=r[:,j+1][None,:]
        resistance=(np.log(face/rc[:,j][None,:])/k[:,:,j]+
                    np.log(rc[:,j+1][None,:]/face)/k[:,:,j+1])
        if j==mesh.coating_cells-1:
            resistance=resistance+contact_resistance/face
        radial[:,:,j]=2*np.pi*dz[:,None]/resistance
    wall_resistance=(
        np.log(rc[:,0][None,:]/r[:,0][None,:])/
        (2*np.pi*k[:,:,0]*dz[:,None])
        +1/(film*2*np.pi*r[:,0][None,:]*dz[:,None]))
    wall=1/wall_resistance
    area=np.pi*np.diff(r**2,axis=1)
    axial=np.zeros((nz-1,ns,nr))
    for i in range(nz-1):
        axial[i,:,:]=area/(
            0.5*dz[i]/k[i,:,:]+0.5*dz[i+1]/k[i+1,:,:])
    pairs=mesh.network.shared_pair_indices
    a=pairs[:,0];b=pairs[:,1]
    pitch=mesh.network.channel_pitch_pair_m
    lengths=mesh.network.shared_face_lengths_m
    lateral=dz[:,None]*lengths[None,:]/(
        0.5*pitch[None,:]/k[:,a,-1]+
        0.5*pitch[None,:]/k[:,b,-1])
    return radial,wall,axial,lateral


def solve_radial_hex_core(
    mesh: RadialHexMesh,
    previous: RadialHexState,
    gas: HexCoolantBoundary,
    fuel: CoreSolidLaw,
    coating: CoreSolidLaw,
    nuclear_solid_power_w: np.ndarray,
    *,
    dt_s: float | None = None,
    contact_resistance_m2_k_w: float = 0.0,
    max_iterations: int = 90,
    relaxation: float = 0.75,
    tolerance_k: float = 1e-9,
    balance_tolerance: float = 1e-7,
)->RadialHexStep:
    """One Picard-converged monolithic implicit conjugate fuel-element step.

    Axial solid ends and all outer surfaces outside shared Voronoi fuel faces
    are adiabatic. Mass flow is fixed and positive; each H2 path runs +z.
    The prescribed nuclear heating is held fixed for the entire timestep.
    """
    mesh.validate()
    fuel.validate();coating.validate()
    nz,ns,nr=mesh.shape
    gas.validate(nz)
    q=_solid_sources(mesh,nuclear_solid_power_w)
    hq=np.asarray(gas.direct_hydrogen_power_w,dtype=float)
    b=gas.gas_boundary()
    s0=np.asarray(previous.solid_temperature_k,dtype=float)
    h0=np.asarray(previous.hydrogen_temperature_k,dtype=float)
    if (s0.shape!=(nz,ns,nr) or h0.shape!=(nz,ns)
            or not np.isfinite(s0).all() or not np.isfinite(h0).all()
            or np.min(s0)<=0 or np.min(h0)<=0):
        raise ValueError("nonphysical starting radial hex thermal field")
    for t in h0.flat:_gas_enthalpy(float(t),b)
    if (dt_s is not None and (not math.isfinite(dt_s) or dt_s<=0)
            or max_iterations<2 or not math.isfinite(tolerance_k)
            or tolerance_k<=0 or not math.isfinite(balance_tolerance)
            or balance_tolerance<=0 or not math.isfinite(relaxation)
            or not 0<relaxation<=1 or not math.isfinite(contact_resistance_m2_k_w)
            or contact_resistance_m2_k_w<0):
        raise ValueError("invalid radial-hex nonlinear solver controls")
    laws=[coating if j<mesh.coating_cells else fuel for j in range(nr)]
    vol=mesh.volumes_m3
    old=s0.copy();old_h=h0.copy()
    s=s0.copy();h=h0.copy()
    nsolid=nz*ns*nr
    ntotal=nsolid+nz*ns
    dz=np.diff(mesh.z_edges_m)
    pair=mesh.network.shared_pair_indices

    def si(i:int,c:int,j:int)->int:
        return (i*ns+c)*nr+j
    def hi(i:int,c:int)->int:
        return nsolid+i*ns+c

    for iteration in range(1,max_iterations+1):
        k=np.empty((nz,ns,nr),dtype=float)
        for j,law in enumerate(laws):
            k[:,:,j]=np.array(
                [law.conductivity(float(t)) for t in s[:,:,j].flat]
            ).reshape((nz,ns))
        radial,wall,axial,lateral=_face_conductance(
            mesh,k,np.asarray(gas.film_w_m2_k),contact_resistance_m2_k_w)
        diagonal=np.zeros(ntotal,dtype=float)
        rhs=np.zeros(ntotal,dtype=float)
        rows:list[int]=[];cols:list[int]=[];values:list[float]=[]

        def link(a:int,d:int,v:float)->None:
            if not math.isfinite(v) or v<0:
                raise ValueError("invalid shared solid/gas conductance")
            diagonal[a]+=v;diagonal[d]+=v
            rows.extend((a,d));cols.extend((d,a));values.extend((-v,-v))

        for i in range(nz):
            for c in range(ns):
                for j,law in enumerate(laws):
                    idx=si(i,c,j)
                    rhs[idx]+=q[i,c,j]
                    if dt_s is not None:
                        cap=(vol[i,c,j]*law.density_kg_m3*
                             law.secant_cp(float(old[i,c,j]),float(s[i,c,j]))
                             /dt_s)
                        diagonal[idx]+=cap
                        rhs[idx]+=cap*old[i,c,j]
                    if j+1<nr:
                        link(idx,si(i,c,j+1),float(radial[i,c,j]))
                    if i+1<nz:
                        link(idx,si(i+1,c,j),float(axial[i,c,j]))
                gidx=hi(i,c)
                link(si(i,c,0),gidx,float(wall[i,c]))
                rhs[gidx]+=hq[i,c]
                if dt_s is not None:
                    mass=gas.hydrogen_mass_kg[i,c]
                    cap=mass*_gas_secant(
                        float(old_h[i,c]),float(h[i,c]),b)/dt_s
                    diagonal[gidx]+=cap
                    rhs[gidx]+=cap*old_h[i,c]
                upstream=(float(gas.inlet_temperature_k[c]) if i==0
                          else float(h[i-1,c]))
                coeff=gas.mdot_kg_s[c]*_gas_secant(
                    upstream,float(h[i,c]),b)
                if not math.isfinite(coeff) or coeff<=0:
                    raise ValueError("invalid H2 advective enthalpy derivative")
                diagonal[gidx]+=coeff
                if i==0:
                    rhs[gidx]+=coeff*gas.inlet_temperature_k[c]
                else:
                    rows.append(gidx)
                    cols.append(hi(i-1,c))
                    values.append(-coeff)
            for p,(c,d) in enumerate(pair):
                link(si(i,int(c),nr-1),si(i,int(d),nr-1),
                     float(lateral[i,p]))
        ii=np.arange(ntotal)
        rows.extend(ii.tolist());cols.extend(ii.tolist())
        values.extend(diagonal.tolist())
        A=coo_matrix((values,(rows,cols)),shape=(ntotal,ntotal)).tocsr()
        x=np.asarray(spsolve(A,rhs),dtype=float)
        if not np.isfinite(x).all() or np.min(x)<=0:
            raise RuntimeError("nonphysical coupled radial-hex temperatures")
        trial_s=x[:nsolid].reshape((nz,ns,nr))
        trial_h=x[nsolid:].reshape((nz,ns))
        for t in trial_h.flat:_gas_enthalpy(float(t),b)
        error=max(float(np.max(np.abs(trial_s-s))),
                  float(np.max(np.abs(trial_h-h))))
        s=relaxation*trial_s+(1-relaxation)*s
        h=relaxation*trial_h+(1-relaxation)*h
        if error<=tolerance_k:break
    else:
        raise RuntimeError("radial/lateral core Picard iteration did not converge")

    # Recompute final physical face fluxes from converged states; signed
    # shared fluxes cancel exactly in combined energy conservation.
    k=np.empty_like(s)
    for j,law in enumerate(laws):
        k[:,:,j]=np.array([
            law.conductivity(float(t)) for t in s[:,:,j].flat]).reshape((nz,ns))
    gr,gw,gz,gl=_face_conductance(
        mesh,k,np.asarray(gas.film_w_m2_k),contact_resistance_m2_k_w)
    radial_flux=np.zeros((nz,ns,nr+1))
    wall_flux=gw*(s[:,:,0]-h)
    radial_flux[:,:,0]=-wall_flux
    for j in range(nr-1):
        radial_flux[:,:,j+1]=gr[:,:,j]*(s[:,:,j]-s[:,:,j+1])
    axial_flux=np.zeros((nz+1,ns,nr))
    if nz>1:
        axial_flux[1:nz,:,:]=gz*(s[:-1,:,:]-s[1:,:,:])
    la=pair[:,0];lb=pair[:,1]
    lateral_flux=gl*(s[:,la,-1]-s[:,lb,-1])

    energy_s=0.0;energy_g=0.0
    if dt_s is not None:
        energy_s=math.fsum(
            vol[i,c,j]*laws[j].density_kg_m3*
            laws[j].enthalpy_change(float(old[i,c,j]),float(s[i,c,j]))
            for i in range(nz) for c in range(ns) for j in range(nr))
        energy_g=math.fsum(
            gas.hydrogen_mass_kg[i,c]*(
                _gas_enthalpy(float(h[i,c]),b)-
                _gas_enthalpy(float(old_h[i,c]),b))
            for i in range(nz) for c in range(ns))
    hin=math.fsum(
        gas.mdot_kg_s[c]*_gas_enthalpy(float(gas.inlet_temperature_k[c]),b)
        for c in range(ns))
    hout=math.fsum(
        gas.mdot_kg_s[c]*_gas_enthalpy(float(h[-1,c]),b)
        for c in range(ns))
    Q=float(np.sum(q)+np.sum(hq)+hin-hout)
    residual=energy_s+energy_g-Q*(1. if dt_s is None else dt_s)
    scale=max(1.,float(np.sum(np.abs(q))),float(np.sum(np.abs(hq))),
              abs(hin-hout))
    scale*=1. if dt_s is None else dt_s
    if abs(residual)>balance_tolerance*scale+1e-6:
        raise RuntimeError("radial-hex combined first law violation: "
                           f"{residual:g} {'W' if dt_s is None else 'J'}")
    return RadialHexStep(
        RadialHexState(s.copy(),h.copy()),radial_flux,
        lateral_flux,axial_flux,wall_flux,energy_s,energy_g,
        hin,hout,residual,iteration,dt_s is not None)
