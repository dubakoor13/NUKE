"""Conservative 19-channel *laterally coupled* NERVA-derived thermal core.

Physics-only opt-in solver. Each axial plane uses the true clipped Voronoi
fuel-sector *adjacency and shared interface lengths* of the 19-bore hexagonal
fuel element. Each fuel sector has a lumped fuel thermal state, a coaxial ZrC
coating state, and an axially advected hydrogen state. Lateral heat conduction
between neighboring fuel sectors and axial fuel/coating conduction are
explicit. This replaces the independent/adiabatic sector assumption, but is
NOT a resolved 2-D finite element fuel-matrix temperature field; sub-sector
temperature gradients use an equivalent-annulus resistance closure.

Steady and backward-Euler transient heat equations are assembled into ONE
sparse coupled linearized system, solved by Picard iteration. Atomic source
terms are caller-prescribed energy input; no neutron source, KERMA, criticality
optimization, fuel enrichment or reactor control is implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .config import NervaConfig
from .layout import coolant_channel_positions
from .fuel_sector_geometry import (
    fuel_sector_fuel_areas_cm2, fuel_sector_polygons_cm,
)
from .core_conjugate_fv import CoreSolidLaw, _gas_enthalpy, _gas_secant


@dataclass(frozen=True)
class HexSectorNetwork:
    """Geometric 19-sector Voronoi interfaces. SI throughout."""
    fuel_area_m2: np.ndarray
    shared_pair_indices: np.ndarray       # [face,2] i<j, zero-based
    shared_face_lengths_m: np.ndarray
    channel_pitch_pair_m: np.ndarray
    coolant_bore_radius_m: float
    coating_outer_radius_m: float

    @classmethod
    def from_nerva_config(cls, cfg: NervaConfig|None=None) -> "HexSectorNetwork":
        cfg=cfg or NervaConfig()
        cfg.validate()
        sites=np.asarray(coolant_channel_positions(cfg),dtype=float)
        polys=fuel_sector_polygons_cm(cfg)
        pair=[]; lengths=[]; pitches=[]
        for i in range(len(polys)):
            for j in range(i+1,len(polys)):
                diff=sites[j]-sites[i]
                offset=0.5*(sites[j]@sites[j]-sites[i]@sites[i])
                # A shared Voronoi face is the part of the clipped polygon
                # on the i/j bisector. A single meeting vertex has zero
                # face length and must not form a heat-transfer connection.
                onface=[np.asarray(v,dtype=float) for v in polys[i]
                        if abs(diff@v-offset)<=2.0e-10]
                if len(onface)<2:
                    continue
                L=max(float(np.linalg.norm(x-y)) for x in onface for y in onface)
                if L<=1.0e-10:
                    continue
                pair.append((i,j))
                lengths.append(L*0.01)
                pitches.append(float(np.linalg.norm(diff))*0.01)
        result=cls(
            fuel_area_m2=fuel_sector_fuel_areas_cm2(cfg)*1e-4,
            shared_pair_indices=np.asarray(pair,dtype=int).reshape(-1,2),
            shared_face_lengths_m=np.asarray(lengths,dtype=float),
            channel_pitch_pair_m=np.asarray(pitches,dtype=float),
            coolant_bore_radius_m=cfg.channel_radius_cm*0.01,
            coating_outer_radius_m=cfg.coated_channel_radius_cm*0.01,
        )
        result.validate()
        return result

    def validate(self)->None:
        areas=np.asarray(self.fuel_area_m2,dtype=float)
        pairs=np.asarray(self.shared_pair_indices)
        length=np.asarray(self.shared_face_lengths_m,dtype=float)
        pitch=np.asarray(self.channel_pitch_pair_m,dtype=float)
        if (areas.shape!=(19,) or not np.isfinite(areas).all()
                or np.any(areas<=0) or pairs.ndim!=2 or pairs.shape[1]!=2
                or pairs.shape[0]<1 or pairs.dtype.kind not in "iu"
                or length.shape!=(pairs.shape[0],)
                or pitch.shape!=length.shape
                or not np.isfinite(length).all() or np.any(length<=0)
                or not np.isfinite(pitch).all() or np.any(pitch<=0)
                or np.any(pairs[:,0]<0) or np.any(pairs[:,1]>=19)
                or np.any(pairs[:,0]>=pairs[:,1])
                or len({tuple(p) for p in pairs})!=len(pairs)
                or not (0<self.coolant_bore_radius_m<
                        self.coating_outer_radius_m)):
            raise ValueError("invalid hex-sector Voronoi face geometry")


@dataclass(frozen=True)
class HexCoolantBoundary:
    """Fixed positive mass flow and H2 inventory for each channel."""
    mdot_kg_s: np.ndarray                # [19]
    inlet_temperature_k: np.ndarray      # [19]
    hydrogen_mass_kg: np.ndarray         # [axial,19]
    film_w_m2_k: np.ndarray              # [axial,19]
    direct_hydrogen_power_w: np.ndarray  # [axial,19]
    hydrogen_model: str = "constant"
    cp_reference_j_kg_k: float = 14300.0

    def gas_boundary(self):
        # Bridge only the enthalpy API, not generic pressure/flow models.
        from .core_conjugate_fv import CoreSectorBoundary
        return CoreSectorBoundary(
            mass_flow_kg_s=1.0,
            inlet_temperature_k=500.0,
            convection_w_m2_k=np.ones(1),
            direct_hydrogen_heating_w=np.zeros(1),
            gas_mass_kg=np.ones(1),
            gas_enthalpy_model=self.hydrogen_model,
            constant_gas_cp_j_kg_k=self.cp_reference_j_kg_k,
        )

    def validate(self,nz:int)->None:
        if self.hydrogen_model not in ("constant","nasa7-h2"):
            raise ValueError("unknown hydrogen enthalpy option")
        if not math.isfinite(self.cp_reference_j_kg_k) or self.cp_reference_j_kg_k<=0:
            raise ValueError("nonphysical hydrogen heat capacity")
        for name,value,shape,positive in (
            ("mdot",self.mdot_kg_s,(19,),True),
            ("inlet",self.inlet_temperature_k,(19,),True),
            ("H2 mass",self.hydrogen_mass_kg,(nz,19),True),
            ("film",self.film_w_m2_k,(nz,19),True),
            ("direct heat",self.direct_hydrogen_power_w,(nz,19),False),
        ):
            a=np.asarray(value,dtype=float)
            if a.shape!=shape or not np.isfinite(a).all():
                raise ValueError("invalid "+name+" dimensions/nonfinite input")
            if np.any(a<=0) if positive else np.any(a<0):
                raise ValueError("invalid "+name+" domain")
        b=self.gas_boundary()
        for t in np.asarray(self.inlet_temperature_k):
            _gas_enthalpy(float(t),b)


@dataclass(frozen=True)
class HexCoreState:
    fuel_temperature_k: np.ndarray       # [nz,19]
    coating_temperature_k: np.ndarray    # [nz,19]
    hydrogen_temperature_k: np.ndarray   # [nz,19]


@dataclass(frozen=True)
class HexCoreStep:
    state: HexCoreState
    lateral_face_heat_w: np.ndarray      # [nz,nfaces], positive i->j
    fuel_to_coating_w: np.ndarray         # [nz,19]
    coating_to_hydrogen_w: np.ndarray     # [nz,19]
    solid_storage_j: float
    hydrogen_storage_j: float
    inlet_enthalpy_w: float
    outlet_enthalpy_w: float
    energy_residual: float               # J if transient else W
    iterations: int
    transient: bool


def _resistances(network:HexSectorNetwork,z:np.ndarray,
                 fuel_k:np.ndarray,coat_k:np.ndarray,film:np.ndarray,
                 contact:float)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """Two coaxial resistors plus Voronoi two-point flux approximation.

    The center distance approximation is geometrically orthogonal for
    channel-centered Voronoi sites. The lumped fuel temperature is NOT
    the exact value at the channel center (there is a bore); this is a
    reduced-order lateral closure requiring independent mesh studies.
    """
    dz=np.diff(z)
    r0=network.coolant_bore_radius_m
    r1=network.coating_outer_radius_m
    r2=np.sqrt(r1*r1+network.fuel_area_m2/math.pi)
    rc=np.sqrt(0.5*(r0*r0+r1*r1))
    rf=np.sqrt(0.5*(r1*r1+r2*r2))
    Rfc=(np.log(rf/r1)[None,:]/(2*math.pi*fuel_k*dz[:,None])
         +np.log(r1/rc)/(2*math.pi*coat_k*dz[:,None])
         +contact/(2*math.pi*r1*dz[:,None]))
    Rcg=(np.log(rc/r0)/(2*math.pi*coat_k*dz[:,None])
         +1/(film*2*math.pi*r0*dz[:,None]))
    face=network.shared_pair_indices
    a=face[:,0];b=face[:,1]
    Gxy=(network.shared_face_lengths_m[None,:]*dz[:,None] /
         (0.5*network.channel_pitch_pair_m[None,:]/fuel_k[:,a]+
          0.5*network.channel_pitch_pair_m[None,:]/fuel_k[:,b]))
    return 1/Rfc,1/Rcg,Gxy


def solve_hex_core_conjugate(
    network: HexSectorNetwork,
    z_edges_m: np.ndarray,
    initial: HexCoreState,
    boundary: HexCoolantBoundary,
    fuel: CoreSolidLaw,
    coating: CoreSolidLaw,
    solid_nuclear_heat_w: np.ndarray,
    *,
    dt_s: float|None=None,
    fuel_coating_contact_resistance_m2_k_w: float=0.0,
    max_iterations: int=80,
    relaxation: float=0.75,
    tolerance_k: float=1e-8,
    balance_tolerance: float=1e-7,
)->HexCoreStep:
    """Monolithic nonlinear fuel/coat/H2 energy equation in 19 sectors.

    The model does not solve H2 momentum, neutron transport or criticality.
    Gas and solids use shared temperatures for interface flux cancellation.
    """
    network.validate()
    fuel.validate();coating.validate()
    z=np.asarray(z_edges_m,dtype=float)
    if (z.ndim!=1 or z.size<2 or not np.isfinite(z).all()
            or np.any(np.diff(z)<=0)):
        raise ValueError("invalid axial grid")
    nz=z.size-1
    boundary.validate(nz)
    if (dt_s is not None and (not math.isfinite(dt_s) or dt_s<=0)
            or max_iterations<2 or not (0<relaxation<=1)
            or not math.isfinite(tolerance_k) or tolerance_k<=0
            or not math.isfinite(balance_tolerance) or balance_tolerance<=0
            or not math.isfinite(fuel_coating_contact_resistance_m2_k_w)
            or fuel_coating_contact_resistance_m2_k_w<0):
        raise ValueError("invalid coupled-solve control")
    f0=np.asarray(initial.fuel_temperature_k,dtype=float)
    c0=np.asarray(initial.coating_temperature_k,dtype=float)
    g0=np.asarray(initial.hydrogen_temperature_k,dtype=float)
    Q=np.asarray(solid_nuclear_heat_w,dtype=float)
    if (any(a.shape!=(nz,19) or not np.isfinite(a).all() for a in
            (f0,c0,g0,Q))
            or np.min(f0)<=0 or np.min(c0)<=0 or np.min(g0)<=0):
        raise ValueError("nonphysical fuel/coating/hydrogen initial state")
    b=boundary.gas_boundary()
    for t in g0.flat:
        _gas_enthalpy(float(t),b)
    dz=np.diff(z)
    vf=dz[:,None]*network.fuel_area_m2[None,:]
    ac=math.pi*(network.coating_outer_radius_m**2-
                network.coolant_bore_radius_m**2)
    vc=dz[:,None]*ac*np.ones((1,19))
    fg=f0.copy();cg=c0.copy();gg=g0.copy()
    npair=network.shared_pair_indices.shape[0]
    nc=nz*19
    total=nc*3

    def idx(kind:int,i:int,ch:int)->int:
        return kind*nc+i*19+ch

    for iteration in range(1,max_iterations+1):
        kf=np.array([[fuel.conductivity(float(t)) for t in row] for row in fg])
        kc=np.array([[coating.conductivity(float(t)) for t in row] for row in cg])
        gf,gc,glat=_resistances(
            network,z,kf,kc,np.asarray(boundary.film_w_m2_k),
            fuel_coating_contact_resistance_m2_k_w)
        diag=np.zeros(total)
        rhs=np.zeros(total)
        rr:list[int]=[];cc:list[int]=[];vv:list[float]=[]

        def link(a:int,d:int,conductance:float)->None:
            if not math.isfinite(conductance) or conductance<0:
                raise ValueError("negative/nonfinite thermal conductance")
            diag[a]+=conductance;diag[d]+=conductance
            rr.extend((a,d));cc.extend((d,a));vv.extend((-conductance,-conductance))

        for i in range(nz):
            for ch in range(19):
                a=idx(0,i,ch);d=idx(1,i,ch);g=idx(2,i,ch)
                if dt_s is not None:
                    cf=fuel.density_kg_m3*vf[i,ch]*fuel.secant_cp(
                        float(f0[i,ch]),float(fg[i,ch]))/dt_s
                    cm=coating.density_kg_m3*vc[i,ch]*coating.secant_cp(
                        float(c0[i,ch]),float(cg[i,ch]))/dt_s
                    gh=(boundary.hydrogen_mass_kg[i,ch]*
                        _gas_secant(float(g0[i,ch]),float(gg[i,ch]),b)/dt_s)
                    diag[a]+=cf;rhs[a]+=cf*f0[i,ch]
                    diag[d]+=cm;rhs[d]+=cm*c0[i,ch]
                    diag[g]+=gh;rhs[g]+=gh*g0[i,ch]
                rhs[a]+=Q[i,ch]
                rhs[g]+=boundary.direct_hydrogen_power_w[i,ch]
                link(a,d,float(gf[i,ch]))
                link(d,g,float(gc[i,ch]))
                # Gas advection is directional, not a symmetric Fourier link.
                up=(float(boundary.inlet_temperature_k[ch]) if i==0
                    else float(gg[i-1,ch]))
                cp=_gas_secant(up,float(gg[i,ch]),b)
                if cp<=0 or not math.isfinite(cp):
                    raise ValueError("invalid H2 enthalpy slope")
                adv=boundary.mdot_kg_s[ch]*cp
                diag[g]+=adv
                if i==0:rhs[g]+=adv*boundary.inlet_temperature_k[ch]
                else:
                    rr.append(g);cc.append(idx(2,i-1,ch));vv.append(-adv)

                if i+1<nz:
                    a2=idx(0,i+1,ch)
                    c2=idx(1,i+1,ch)
                    df=(dz[i]/(2*kf[i,ch])+
                        dz[i+1]/(2*kf[i+1,ch]))
                    dc=(dz[i]/(2*kc[i,ch])+
                        dz[i+1]/(2*kc[i+1,ch]))
                    link(a,a2,float(network.fuel_area_m2[ch]/df))
                    link(d,c2,float(ac/dc))
            for m,(ch1,ch2) in enumerate(network.shared_pair_indices):
                link(idx(0,i,int(ch1)),idx(0,i,int(ch2)),
                     float(glat[i,m]))
        ii=np.arange(total)
        rr.extend(ii.tolist());cc.extend(ii.tolist());vv.extend(diag.tolist())
        mat=coo_matrix((vv,(rr,cc)),shape=(total,total)).tocsr()
        solution=np.asarray(spsolve(mat,rhs),dtype=float)
        if not np.isfinite(solution).all() or np.min(solution)<=0:
            raise RuntimeError("nonphysical coupled solution")
        nf=solution[:nc].reshape((nz,19))
        nm=solution[nc:2*nc].reshape((nz,19))
        ng=solution[2*nc:].reshape((nz,19))
        for t in ng.flat:_gas_enthalpy(float(t),b)
        err=max(float(np.max(np.abs(nf-fg))),
                float(np.max(np.abs(nm-cg))),
                float(np.max(np.abs(ng-gg))))
        fg=fg*(1-relaxation)+nf*relaxation
        cg=cg*(1-relaxation)+nm*relaxation
        gg=gg*(1-relaxation)+ng*relaxation
        if err<tolerance_k:break
    else:
        raise RuntimeError("19-sector heat transport did not converge")

    kf=np.array([[fuel.conductivity(float(t)) for t in row] for row in fg])
    kc=np.array([[coating.conductivity(float(t)) for t in row] for row in cg])
    gf,gc,glat=_resistances(
        network,z,kf,kc,np.asarray(boundary.film_w_m2_k),
        fuel_coating_contact_resistance_m2_k_w)
    a=network.shared_pair_indices[:,0]
    d=network.shared_pair_indices[:,1]
    lateral=glat*(fg[:,a]-fg[:,d])
    fuel_to_coat=gf*(fg-cg)
    coat_to_gas=gc*(cg-gg)
    solid_j=0.; gas_j=0.
    if dt_s is not None:
        solid_j=math.fsum(
            fuel.density_kg_m3*vf[i,ch]*
                fuel.enthalpy_change(float(f0[i,ch]),float(fg[i,ch]))+
            coating.density_kg_m3*vc[i,ch]*
                coating.enthalpy_change(float(c0[i,ch]),float(cg[i,ch]))
            for i in range(nz) for ch in range(19))
        gas_j=math.fsum(
            boundary.hydrogen_mass_kg[i,ch]*
            (_gas_enthalpy(float(gg[i,ch]),b)-
             _gas_enthalpy(float(g0[i,ch]),b))
            for i in range(nz) for ch in range(19))
    hin=math.fsum(
        float(boundary.mdot_kg_s[ch])*
        _gas_enthalpy(float(boundary.inlet_temperature_k[ch]),b)
        for ch in range(19))
    hout=math.fsum(
        float(boundary.mdot_kg_s[ch])*_gas_enthalpy(float(gg[-1,ch]),b)
        for ch in range(19))
    balance=(solid_j+gas_j if dt_s is not None else 0.0)-(
        float(np.sum(Q)+np.sum(boundary.direct_hydrogen_power_w))+hin-hout
        )*(dt_s if dt_s is not None else 1.0)
    scale=max(1.0,abs(hin-hout),float(np.sum(np.abs(Q))),
              float(np.sum(boundary.direct_hydrogen_power_w)))
    scale*=dt_s if dt_s is not None else 1.0
    if abs(balance)>balance_tolerance*scale+1e-6:
        raise RuntimeError(f"19-sector coupled first-law failed: {balance:g}")

    return HexCoreStep(
        HexCoreState(fg,cg,gg),lateral,fuel_to_coat,coat_to_gas,
        solid_j,gas_j,hin,hout,balance,iteration,dt_s is not None,
    )
