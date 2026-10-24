"""Conservative annular fuel conduction and axial H2 advection core benchmark.

Each *neutronic* (radial-zone, axial-cell) has an externally described
equivalent fuel-channel annulus, resolved into radial fuel rings. Parallel
channel count is explicit, so solid cell volume and face conductances
scale consistently. This effective annulus is NOT historical NERVA
hexagonal/Voronoi fuel geometry, its reconstructed channel layout, or an
OpenMC material mesh; geometric compatibility must be established upstream.

Solid heat is conducted both radially (exact cylindrical logarithmic
half-cell resistances) and axially (two half-cell slab resistances);
the exterior radius and both axial ends are insulated. The inner ring
exchanges with coolant via externally supplied wall UA [W/K], including
any coating/film effects upstream. One-dimensional H2 +z upwind enthalpy
transport is constant-cp, fixed-flow and fixed coolant inventory:

 Cs (Ts1-Ts0)/dt = Qsolid + SUM_faces(UAface*(Tadj1-Ts1))
                              - UAinner*(Ts1-Th1)
 Ch (Th1-Th0)/dt = Qdirect + UAinner*(Ts1-Th1)
                              + mdot*cp*(T_up1-Th1)

The inlet is T_up=Tin and upstream cell values are implicit new-time
values. A solid or fluid may heat or cool; internal fluxes are signed.

A single sparse backward-Euler solve advances every fuel ring and
coolant segment, preserving storage+outlet enthalpy+other deposition
exactly for user-supplied power [W]. There is no neutron-flux-to-watts
conversion, pressure/density-dependent hydrogen EOS, flow acceleration,
dissociation, heat-capacity dependence on T, fuel structural stress or
physical NASA NERVA geometry.

This solves a FIXED-coefficient time interval. For neutronic
temperature feedback use volume-weighted radial solid temperature only
as an explicit effective homogenization assumption; it does not replace
physically qualified group condensation or transport.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve


def _check(name, value, shape, *, strict=False):
    a=np.asarray(value,dtype=float)
    if a.shape!=shape or np.any(~np.isfinite(a)) or np.any(a<=0 if strict else a<0):
        raise ValueError(f"{name} must be finite {'positive' if strict else 'nonnegative'} {shape}")
    return a


@dataclass(frozen=True)
class ConjugateCoreGeometry:
    axial_edges_m: np.ndarray                    # [nz+1]
    fuel_radial_edges_m: np.ndarray              # [nr_zones,nring+1]
    channels_per_radial_zone: np.ndarray         # [nr_zones] int
    solid_density_kg_m3: np.ndarray              # [zone,z,ring]
    solid_heat_capacity_j_kg_k: np.ndarray      # [zone,z,ring]
    solid_conductivity_w_m_k: np.ndarray        # [zone,z,ring]
    hydrogen_mass_kg: np.ndarray                # [zone,z]
    hydrogen_cp_j_kg_k: np.ndarray              # [zone] constant along branch
    hydrogen_mass_flow_kg_s: np.ndarray         # [zone]
    inlet_temperature_k: np.ndarray            # [zone]
    inner_wall_conductance_w_k: np.ndarray      # [zone,z] total channels combined

    def __post_init__(self):
        z=np.asarray(self.axial_edges_m,dtype=float)
        r=np.asarray(self.fuel_radial_edges_m,dtype=float)
        if (z.ndim!=1 or len(z)<2 or not np.all(np.isfinite(z))
                or np.any(np.diff(z)<=0)):
            raise ValueError("strictly increasing axial edges in metres required")
        if (r.ndim!=2 or r.shape[0]<1 or r.shape[1]<2
                or not np.all(np.isfinite(r)) or np.any(r<=0)
                or np.any(np.diff(r,axis=1)<=0)):
            raise ValueError("strictly increasing positive fuel annulus radius edges required")
        npth,nring=r.shape[0],r.shape[1]-1
        nz=len(z)-1
        counts=np.asarray(self.channels_per_radial_zone)
        if (counts.shape!=(npth,) or
                np.any(~np.isfinite(counts)) or
                np.any(counts<1) or np.any(counts!=np.floor(counts))):
            raise ValueError("positive integer number of coolant channels per zone required")
        for name in ("solid_density_kg_m3","solid_heat_capacity_j_kg_k",
                     "solid_conductivity_w_m_k"):
            _check(name,getattr(self,name),(npth,nz,nring),strict=True)
        for name in ("hydrogen_mass_kg","inner_wall_conductance_w_k"):
            _check(name,getattr(self,name),(npth,nz),
                   strict=(name=="hydrogen_mass_kg"))
        for name in ("hydrogen_cp_j_kg_k","inlet_temperature_k"):
            _check(name,getattr(self,name),(npth,),strict=True)
        _check("hydrogen_mass_flow",self.hydrogen_mass_flow_kg_s,(npth,))
        # In particular zero flow is a valid closed stagnant test case.

    @property
    def shape(self)->tuple[int,int,int]:
        return (len(self.fuel_radial_edges_m),
                len(self.axial_edges_m)-1,
                self.fuel_radial_edges_m.shape[1]-1)

    @property
    def solid_volumes_m3(self):
        r=np.asarray(self.fuel_radial_edges_m,dtype=float)
        dz=np.diff(self.axial_edges_m)
        cross=np.pi*(r[:,1:]**2-r[:,:-1]**2)
        return (np.asarray(self.channels_per_radial_zone)[:,None,None]*
                dz[None,:,None]*cross[:,None,:])


@dataclass(frozen=True)
class ConjugateCoreState:
    time_s: float
    fuel_temperature_k: np.ndarray      # [zone,z,ring]
    hydrogen_temperature_k: np.ndarray  # [zone,z]


@dataclass(frozen=True)
class ConjugateCoreDeposition:
    source_id: str
    solid_w: np.ndarray        # [zone,z,ring]
    direct_hydrogen_w: np.ndarray     # [zone,z]
    other_w: np.ndarray       # [zone,z]; never allocated twice
    declared_total_w: float
    synthetic: bool

    def check(self,shape):
        npth,nz,nring=shape
        if not isinstance(self.source_id,str) or not self.source_id.strip():
            raise ValueError("deposition source ID required")
        if not isinstance(self.synthetic,(bool,np.bool_)):
            raise ValueError("explicit synthetic boolean required")
        solid=_check("fuel nuclear power",self.solid_w,shape)
        h2=_check("H2 direct power",self.direct_hydrogen_w,(npth,nz))
        other=_check("other unmodeled power",self.other_w,(npth,nz))
        if not math.isfinite(self.declared_total_w) or self.declared_total_w<0:
            raise ValueError("declared deposition must be finite nonnegative W")
        total=float(np.sum(solid)+np.sum(h2)+np.sum(other))
        if abs(total-self.declared_total_w)>1e-10*max(1.,total,self.declared_total_w):
            raise ValueError("deposition partitions do not match total reactor power")
        return solid,h2,other


@dataclass(frozen=True)
class ConjugateCoreLinearSystem:
    matrix: csc_matrix
    rhs: np.ndarray
    solid_storage_j_k: np.ndarray
    hydrogen_storage_j_k: np.ndarray
    solid_volume_m3: np.ndarray


@dataclass(frozen=True)
class ConjugateCoreStep:
    state: ConjugateCoreState
    fuel_storage_j: float
    hydrogen_storage_j: float
    outlet_net_enthalpy_j: float
    other_deposition_j: float
    deposited_total_j: float
    wall_transfer_j: float
    internal_radial_w: np.ndarray
    internal_axial_w: np.ndarray
    thermal_balance_residual_j: float
    thermal_relative_residual: float
    deposition_source_id: str
    synthetic: bool


def _radial_centres(r):
    return np.sqrt(.5*(r[:,:-1]**2+r[:,1:]**2))


def assemble_conjugate_core(
    geo:ConjugateCoreGeometry,
    old:ConjugateCoreState,
    power:ConjugateCoreDeposition,
    dt_s:float,
)->ConjugateCoreLinearSystem:
    """Return complete strictly conservative sparse backward Euler operator."""
    if not math.isfinite(dt_s) or dt_s<=0:
        raise ValueError("positive finite thermal timestep required")
    if not math.isfinite(old.time_s) or old.time_s<0:
        raise ValueError("finite nonnegative initial time required")
    npath,nz,nring=geo.shape
    solid=_check("old fuel temperature",old.fuel_temperature_k,geo.shape,strict=True)
    coolant=_check("old H2 temperature",old.hydrogen_temperature_k,(npath,nz),strict=True)
    qs,qh,_=power.check(geo.shape)
    rho=np.asarray(geo.solid_density_kg_m3)
    cps=np.asarray(geo.solid_heat_capacity_j_kg_k)
    k=np.asarray(geo.solid_conductivity_w_m_k)
    v=geo.solid_volumes_m3
    Cs=v*rho*cps
    mass=np.asarray(geo.hydrogen_mass_kg)
    cph=np.asarray(geo.hydrogen_cp_j_kg_k)
    Ch=mass*cph[:,None]
    G=np.asarray(geo.hydrogen_mass_flow_kg_s)*cph
    wall=np.asarray(geo.inner_wall_conductance_w_k)
    r=np.asarray(geo.fuel_radial_edges_m)
    center=_radial_centres(r)
    dz=np.diff(geo.axial_edges_m)
    counts=np.asarray(geo.channels_per_radial_zone,dtype=float)
    ns=npath*nz*nring
    total=ns+npath*nz
    diag=np.zeros(total)
    rhs=np.zeros(total)
    rows=[];cols=[];vals=[]
    def solididx(p,z,j):return (p*nz+z)*nring+j
    def gasidx(p,z):return ns+p*nz+z
    def pair(a,b,conductance):
        if conductance<=0 or not math.isfinite(conductance):
            raise ArithmeticError("negative/nonfinite face conductance")
        diag[a]+=conductance
        diag[b]+=conductance
        rows.extend((a,b));cols.extend((b,a))
        vals.extend((-conductance,-conductance))
    for p in range(npath):
        for z in range(nz):
            gas=gasidx(p,z)
            diag[gas]+=Ch[p,z]/dt_s+G[p]+wall[p,z]
            rhs[gas]=Ch[p,z]/dt_s*coolant[p,z]+qh[p,z]
            if z>0:
                rows.append(gas)
                cols.append(gasidx(p,z-1))
                vals.append(-G[p])
            else:
                rhs[gas]+=G[p]*geo.inlet_temperature_k[p]
            for j in range(nring):
                a=solididx(p,z,j)
                diag[a]+=Cs[p,z,j]/dt_s
                rhs[a]=Cs[p,z,j]/dt_s*solid[p,z,j]+qs[p,z,j]
                if j==0 and wall[p,z]:
                    diag[a]+=wall[p,z]
                    rows.append(a);cols.append(gas);vals.append(-wall[p,z])
                    # Gas diagonal already contains the same wall UA.
                    # Add the opposite gas-to-solid coupling, not the diagonal.
                    rows.append(gas);cols.append(a);vals.append(-wall[p,z])
                if j+1<nring:
                    edge=r[p,j+1]
                    rleft=np.log(edge/center[p,j])/(2*np.pi*dz[z]*counts[p]*k[p,z,j])
                    rright=np.log(center[p,j+1]/edge)/(2*np.pi*dz[z]*counts[p]*k[p,z,j+1])
                    pair(a,solididx(p,z,j+1),1./(rleft+rright))
                if z+1<nz:
                    area=counts[p]*np.pi*(r[p,j+1]**2-r[p,j]**2)
                    resistance=(dz[z]/(2*k[p,z,j]*area)+
                               dz[z+1]/(2*k[p,z+1,j]*area))
                    pair(a,solididx(p,z+1,j),1./resistance)
    rows.extend(range(total));cols.extend(range(total));vals.extend(diag)
    matrix=coo_matrix((vals,(rows,cols)),shape=(total,total)).tocsc()
    matrix.sum_duplicates()
    if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs)):
        raise ArithmeticError("invalid conjugate thermal matrix or RHS")
    return ConjugateCoreLinearSystem(matrix,rhs,Cs,Ch,v)


def advance_conjugate_core(
    geo:ConjugateCoreGeometry,
    old:ConjugateCoreState,
    power:ConjugateCoreDeposition,
    dt_s:float,
)->ConjugateCoreStep:
    """Solve every solid radial ring and +z H2 node simultaneously."""
    system=assemble_conjugate_core(geo,old,power,dt_s)
    new=spsolve(system.matrix,system.rhs)
    if np.any(~np.isfinite(new)) or np.any(new<=0):
        raise ArithmeticError("nonpositive/nonfinite conjugate fuel or hydrogen temperature")
    npth,nz,nring=geo.shape
    ns=npth*nz*nring
    solid=new[:ns].reshape(geo.shape)
    h2=new[ns:].reshape((npth,nz))
    c=ConjugateCoreState(old.time_s+dt_s,solid,h2)
    Cs=system.solid_storage_j_k
    Ch=system.hydrogen_storage_j_k
    dUs=float(np.sum(Cs*(solid-old.fuel_temperature_k)))
    dUh=float(np.sum(Ch*(h2-old.hydrogen_temperature_k)))
    G=np.asarray(geo.hydrogen_mass_flow_kg_s)*np.asarray(geo.hydrogen_cp_j_kg_k)
    export=float(dt_s*np.sum(G*(h2[:,-1]-geo.inlet_temperature_k)))
    qs,qh,other=power.check(geo.shape)
    unmodeled=float(dt_s*np.sum(other))
    deposited=dt_s*power.declared_total_w
    wall_power=float(np.sum(np.asarray(geo.inner_wall_conductance_w_k)*
                            (solid[:,:,0]-h2)))
    residual=dUs+dUh+export+unmodeled-deposited
    scale=max(1.,abs(dUs),abs(dUh),abs(export),abs(unmodeled),abs(deposited))
    relative=abs(residual)/scale
    if relative>1e-9:
        raise ArithmeticError("conjugate radial/axial fuel and H2 first-law conservation failed")
    r=np.asarray(geo.fuel_radial_edges_m)
    center=_radial_centres(r)
    k=np.asarray(geo.solid_conductivity_w_m_k)
    dz=np.diff(geo.axial_edges_m)
    counts=np.asarray(geo.channels_per_radial_zone,dtype=float)
    rad=np.zeros((npth,nz,max(0,nring-1)))
    ax=np.zeros((npth,max(0,nz-1),nring))
    for p in range(npth):
        for z in range(nz):
            for j in range(nring-1):
                face=r[p,j+1]
                R=(np.log(face/center[p,j])/k[p,z,j]+
                   np.log(center[p,j+1]/face)/k[p,z,j+1])
                conductance=2*np.pi*counts[p]*dz[z]/R
                rad[p,z,j]=conductance*(solid[p,z,j+1]-solid[p,z,j])
            if z+1<nz:
                for j in range(nring):
                    area=counts[p]*np.pi*(r[p,j+1]**2-r[p,j]**2)
                    conductance=area/(dz[z]/(2*k[p,z,j])+
                                       dz[z+1]/(2*k[p,z+1,j]))
                    ax[p,z,j]=conductance*(solid[p,z+1,j]-solid[p,z,j])
    return ConjugateCoreStep(
        state=c,fuel_storage_j=dUs,hydrogen_storage_j=dUh,
        outlet_net_enthalpy_j=export,
        other_deposition_j=unmodeled,deposited_total_j=deposited,
        wall_transfer_j=dt_s*wall_power,
        internal_radial_w=rad,internal_axial_w=ax,
        thermal_balance_residual_j=residual,
        thermal_relative_residual=relative,
        deposition_source_id=power.source_id,synthetic=bool(power.synthetic)
    )


def effective_solid_temperature_for_xs(geo:ConjugateCoreGeometry,
                                       state:ConjugateCoreState)->np.ndarray:
    """Volume-weighted effective temperature [radial-zone,z] for XS lookups.

    NOTE: this is a numerical approximation, not validated homogenization:
    material-specific temperatures, hydrogen density and cross-section
    energy condensation are not determined by simple volume averages.
    """
    npath,nz,nring=geo.shape
    T=_check("fuel ring temperatures",state.fuel_temperature_k,geo.shape,strict=True)
    vol=geo.solid_volumes_m3
    return np.sum(vol*T,axis=2)/np.sum(vol,axis=2)
