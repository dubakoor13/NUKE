"""Stage82: independently exact harmonic heat flux near a true coolant hole.

Use the analytical 2-D harmonic function
    T(x,y) = T_wall + A log(|(x,y)-c|/r_hole)
through the *actual* 19-hole/inner-hex fuel material region. The
singularity lies INSIDE the excluded hole, so Laplace(T)=0 throughout
the fuel. At the central fully contained circular hole, the exact
fuel-outward surface heat is 2*pi*k*L*A (W), and exact q''=k*A/r_hole.

For each Stage53 physically open INTERIOR face, independently integrate
the continuous Fourier flux -k*grad(T) dot n over the exact unblocked
Cartesian line intervals. The reference is obtained by direct Gaussian
quadrature of the analytic gradient; it does NOT reuse Stage81's
reconstructed-gradient or conductance operators. Compare both the old
Stage53 TPFA and Stage81 full nonorthogonal interior-face flux.

Separately compare the Stage79/80 fuel-only wall normal resistance and
tangential correction against the exact central circular-wall flux.

This is a *manufactured flux/truncation benchmark*, NOT a numerical
solution of the curved-hole boundary-value problem. A subsequent stage
must impose all exact outer-hex/other-hole BCs, SOLVE the discrete PDE,
and study temperature/flux errors versus mesh for full verification.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .coating_mesh_convergence import CoupledMeshResolution
from .fuel_interior_nonorthogonal import nonorthogonal_face_heat_w
from .fuel_wall_nonorthogonal import reconstruct_fuel_gradient
from .fuel_cutcell_conduction_2d import _NORMALS
from .layout import coolant_channel_positions


@dataclass(frozen=True)
class HarmonicHoleField:
    """Exact, source-free conduction field on fuel excluding selected hole."""
    hole_center_cm: tuple[float,float]
    hole_radius_cm: float
    amplitude_k: float
    conductivity_w_m_k: float
    axial_length_m: float
    wall_temperature_k: float = 450.

    def __post_init__(self):
        values=(*self.hole_center_cm,self.hole_radius_cm,self.amplitude_k,
                self.conductivity_w_m_k,self.axial_length_m,
                self.wall_temperature_k)
        if (not all(math.isfinite(float(v)) for v in values)
                or self.hole_radius_cm<=0 or self.amplitude_k==0
                or self.conductivity_w_m_k<=0 or self.axial_length_m<=0):
            raise ValueError("harmonic curved-hole field requires finite physical inputs")

    def temperature(self, xy_cm:np.ndarray)->np.ndarray:
        points=np.asarray(xy_cm,dtype=float)
        if points.shape[-1]!=2 or not np.all(np.isfinite(points)):
            raise ValueError("harmonic temperature expects finite [...,2] coordinates")
        dx=points[...,0]-self.hole_center_cm[0]
        dy=points[...,1]-self.hole_center_cm[1]
        r=np.hypot(dx,dy)
        if np.any(r<self.hole_radius_cm-1e-9):
            raise ValueError("harmonic fuel field evaluated inside excluded central hole")
        return self.wall_temperature_k+self.amplitude_k*np.log(r/self.hole_radius_cm)

    def gradient_per_cm(self, xy_cm:np.ndarray)->np.ndarray:
        points=np.asarray(xy_cm,dtype=float)
        delta=points-np.asarray(self.hole_center_cm)
        norm2=np.sum(delta*delta,axis=-1,keepdims=True)
        if (not np.all(np.isfinite(delta))
                or np.any(norm2<(self.hole_radius_cm-1e-9)**2)):
            raise ValueError("harmonic gradient evaluated inside excluded cylinder")
        return self.amplitude_k*delta/norm2

    def exact_central_wall_power_w(self)->float:
        return 2.*math.pi*self.conductivity_w_m_k*self.axial_length_m*self.amplitude_k

    def exact_central_wall_flux_w_m2(self)->float:
        return self.conductivity_w_m_k*self.amplitude_k/(self.hole_radius_cm*.01)


def _line_material_intervals(
    p:tuple[float,float],q:tuple[float,float],
    ap:float,holes:list[tuple[float,float]],radius:float,
)->list[tuple[float,float]]:
    """Return exact physical [t0,t1] line intervals in hex minus circles."""
    dx,dy=q[0]-p[0],q[1]-p[1]
    aa=dx*dx+dy*dy
    if aa<=0:
        return []
    lower=0.
    upper=1.
    for nx,ny in _NORMALS:
        value=nx*p[0]+ny*p[1]-ap
        slope=nx*dx+ny*dy
        if abs(slope)<1e-15:
            if value>1e-12:
                return []
            continue
        cut=-value/slope
        if slope>0:
            upper=min(upper,cut)
        else:
            lower=max(lower,cut)
    lower=max(0.,lower)
    upper=min(1.,upper)
    if upper-lower<=1e-15:
        return []
    blocked=[]
    for cx,cy in holes:
        ox,oy=p[0]-cx,p[1]-cy
        bb=2.*(ox*dx+oy*dy)
        cc=ox*ox+oy*oy-radius*radius
        disc=bb*bb-4.*aa*cc
        if disc<=0:
            continue
        root=math.sqrt(disc)
        left=max(lower,(-bb-root)/(2.*aa))
        right=min(upper,(-bb+root)/(2.*aa))
        if right>left:
            blocked.append((left,right))
    blocked.sort()
    output=[]
    start=lower
    for left,right in blocked:
        if left>start+1e-15:
            output.append((start,left))
        start=max(start,right)
    if upper>start+1e-15:
        output.append((start,upper))
    return output


@dataclass(frozen=True)
class HarmonicFluxObservation:
    mesh_xy:tuple[int,int]
    face_count:int
    open_area_reference_max_discrepancy_m2:float
    exact_interior_flux_w:np.ndarray
    stage53_flux_w:np.ndarray
    stage81_flux_w:np.ndarray
    stage53_normalized_flux_l1_error:float
    stage81_normalized_flux_l1_error:float
    stage53_normalized_flux_l2_error:float
    stage81_normalized_flux_l2_error:float
    exact_central_hole_wall_power_w:float
    stage79_central_wall_power_w:float
    stage80_central_wall_power_w:float
    stage79_central_wall_relative_error:float
    stage80_central_wall_relative_error:float
    central_arc_area_residual_m2:float
    central_wall_reference_temperature_max_error_k:float
    stage81_internal_flux_global_cancellation_w:float


def evaluate_harmonic_curved_hole(
    model, field:HarmonicHoleField,
    *,
    gauss_points:int=24,
) -> HarmonicFluxObservation:
    """Independent exact central-hole and ALL interior fuel-face flux errors.

    Input model requires the Stage81 INTERIOR operator and Stage80
    exact-centroid gradient. The manufactured field uses *fuel-only*
    boundary Dirichlet temperatures so that coating/H2 series resistance
    cannot conceal fuel-side spatial error. Full coupled system is not
    modified by this reference diagnostic.
    """
    if not isinstance(gauss_points,int) or gauss_points<12 or gauss_points>96:
        raise ValueError("reference wall/face quadrature order outside range")
    faces=model.interior_fuel_faces
    contacts=model.angular_contacts
    solid=model.solid
    if faces is None or contacts is None:
        raise ValueError("manufactured study needs real fuel centroids and physical arcs")
    base=solid.transverse
    ids=base.cell_ids
    ny,nx=ids.shape
    f=faces
    xy=f.gradient.centroids_cm
    centers=coolant_channel_positions(model.config)
    if not np.allclose(field.hole_center_cm,centers[0],atol=1e-12,rtol=0):
        raise ValueError("manufactured source must be centered on fully contained central channel")
    if abs(field.hole_radius_cm-model.config.coated_channel_radius_cm)>1e-12:
        raise ValueError("manufactured log singularity must be inside the actual excluded channel")
    if abs(field.axial_length_m-base.axial_length_m)>1e-12:
        raise ValueError("manufactured axial length does not equal modeled fuel length")
    if abs(field.amplitude_k)>200.:
        raise ValueError("manufactured temperature slope outside benchmark guard")
    ap=model.config.inner_fuel_flat_to_flat_cm*.5
    rad=model.config.coated_channel_radius_cm
    nface=f.face_count
    ixiy=np.full((base.n_fuel,2),-1,dtype=int)
    for iy,ix in zip(*np.where(ids>=0)):
        ixiy[int(ids[iy,ix])]=[ix,iy]
    glx,gweights=np.polynomial.legendre.leggauss(gauss_points)
    exact=np.zeros(nface)
    area_error=0.
    for j,(i0,i1) in enumerate(zip(f.face_cell_i,f.face_cell_j)):
        ix0,iy0=ixiy[i0]
        ix1,iy1=ixiy[i1]
        if f.normal_x[j]>0.5:
            x=float(base.x_edges_cm[ix1])
            p=(x,float(base.y_edges_cm[iy0]))
            q=(x,float(base.y_edges_cm[iy0+1]))
        else:
            y=float(base.y_edges_cm[iy1])
            p=(float(base.x_edges_cm[ix0]),y)
            q=(float(base.x_edges_cm[ix0+1]),y)
        segment=np.asarray(q)-np.asarray(p)
        length=float(np.linalg.norm(segment))
        intervals=_line_material_intervals(p,q,ap,centers,rad)
        area=field.axial_length_m*.01*length*sum(t1-t0 for t0,t1 in intervals)
        area_error=max(area_error,abs(area-f.physical_face_full_area_m2[j]))
        total=0.
        for low,high in intervals:
            weights=.5*(high-low)*gweights
            locations=.5*(high+low)+.5*(high-low)*glx
            points=np.asarray(p)[None,:]+locations[:,None]*segment[None,:]
            grad=field.gradient_per_cm(points)
            normal=f.normal_x[j]*grad[:,0]+f.normal_y[j]*grad[:,1]
            total+=float(np.dot(weights,normal))
        # grad [K/cm]*ds[cm] cancels 0.01 m/cm against 100 cm/m.
        exact[j]=-field.conductivity_w_m_k*field.axial_length_m*length*total
    if area_error>2e-12:
        raise ArithmeticError("independent line/hex/circle face areas disagree with Stage53")
    temperature=field.temperature(xy)
    k=np.full((1,base.n_fuel),field.conductivity_w_m_k)
    predicted,skew,g=nonorthogonal_face_heat_w(
        f,temperature[None,:],k,np.ones(1))
    current=predicted[0]
    old=field.conductivity_w_m_k*f.reference_full_length_geometry_m*(
        temperature[f.face_cell_i]-temperature[f.face_cell_j])
    denom_l1=max(1e-14,float(np.sum(np.abs(exact))))
    denom_l2=max(1e-14,float(np.linalg.norm(exact)))
    l1old=float(np.sum(np.abs(old-exact))/denom_l1)
    l1new=float(np.sum(np.abs(current-exact))/denom_l1)
    l2old=float(np.linalg.norm(old-exact)/denom_l2)
    l2new=float(np.linalg.norm(current-exact)/denom_l2)
    arc=contacts.fuel_outer_full_m2.tocoo()
    central=arc.col//contacts.sector_count==0
    row=arc.row[central]
    col=arc.col[central]
    wall_area=arc.data[central]
    reference_area=2.*math.pi*rad*.01*field.axial_length_m
    area_res=float(wall_area.sum()-reference_area)
    if abs(area_res)>2e-12:
        raise ArithmeticError("manufactured central circular wall not fully exposed")
    # Analytical T is strictly constant on the entire central circle.
    wall_pts=np.column_stack((
        field.hole_center_cm[0]+rad*np.cos(np.linspace(0,2*math.pi,128,endpoint=False)),
        field.hole_center_cm[1]+rad*np.sin(np.linspace(0,2*math.pi,128,endpoint=False))))
    wall_error=float(np.max(np.abs(field.temperature(wall_pts)-field.wall_temperature_k)))
    d=np.asarray(contacts.fuel_wall_moment_depth_m[row,col]).ravel()
    if np.any(d<=0):
        raise ArithmeticError("nonphysical moment contact distance in benchmark")
    fuel_jump=temperature[row]-field.wall_temperature_k
    gtx,gty=reconstruct_fuel_gradient(f.gradient,temperature[None,:])
    # Independent Stage80 correction maps: signed actual material
    # centroid-to-arc tangent, and exact area-weighted arc normal.
    tang=np.asarray(contacts.fuel_wall_tangential_offset_m[row,col]).ravel()
    nxm=np.asarray(contacts.fuel_wall_mean_normal_x[row,col]).ravel()
    nym=np.asarray(contacts.fuel_wall_mean_normal_y[row,col]).ravel()
    mag=np.hypot(nxm,nym)
    if np.any(mag<=1e-10):
        raise ArithmeticError("central wall arc has invalid averaged normal")
    g_t=(-gtx[0,row]*nym+gty[0,row]*nxm)/mag
    legacy=float(np.sum(field.conductivity_w_m_k*wall_area/d*fuel_jump))
    corrected=float(np.sum(
        field.conductivity_w_m_k*wall_area/d*(fuel_jump-g_t*tang)))
    exact_wall=field.exact_central_wall_power_w()
    if exact_wall==0:
        raise ArithmeticError("zero manufactured analytic reference flux")
    # A global face-divergence check isolates internal conservation only.
    from .fuel_interior_nonorthogonal import conservative_face_divergence_w
    internal=conservative_face_divergence_w(
        f,current[None,:],base.n_fuel)
    cancellation=float(abs(np.sum(internal)))
    return HarmonicFluxObservation(
        mesh_xy=(nx,ny),face_count=nface,
        open_area_reference_max_discrepancy_m2=area_error,
        exact_interior_flux_w=exact,
        stage53_flux_w=old,stage81_flux_w=current,
        stage53_normalized_flux_l1_error=l1old,
        stage81_normalized_flux_l1_error=l1new,
        stage53_normalized_flux_l2_error=l2old,
        stage81_normalized_flux_l2_error=l2new,
        exact_central_hole_wall_power_w=exact_wall,
        stage79_central_wall_power_w=legacy,
        stage80_central_wall_power_w=corrected,
        stage79_central_wall_relative_error=abs(legacy-exact_wall)/abs(exact_wall),
        stage80_central_wall_relative_error=abs(corrected-exact_wall)/abs(exact_wall),
        central_arc_area_residual_m2=area_res,
        central_wall_reference_temperature_max_error_k=wall_error,
        stage81_internal_flux_global_cancellation_w=cancellation,
    )


@dataclass(frozen=True)
class HarmonicFluxRefinement:
    levels:tuple[HarmonicFluxObservation,...]
    interior_flux_l1_improvement:tuple[float,float]
    wall_total_flux_errors:tuple[float,float,float]
    unresolved_nonmonotonic_wall_flux:bool
    disclaimer:str=(
        "Operator truncation/face-flux comparison only: reference T is "
        "sampled directly at exact centroids; the discrete PDE is NOT "
        "solved and convergence of computed T is not established."
    )


def run_harmonic_flux_refinement(
    factory,
    *,
    grids:tuple[tuple[int,int],...]=((20,24),(28,32),(36,40)),
    amplitude_k:float=60.,
    conductivity_w_m_k:float=19.,
)->HarmonicFluxRefinement:
    """Run independently referenced curved-hole face/wall errors on meshes."""
    if (len(grids)<3 or any(
            nx<20 or ny<20 for nx,ny in grids)
            or any(a>=b or c>=d
                   for (a,c),(b,d) in zip(grids,grids[1:]))):
        raise ValueError("manufactured refinement needs >=3 increasing XY meshes")
    observations=[]
    for nx,ny in grids:
        m,_=factory(CoupledMeshResolution(
            nx,ny,2,8,2))
        cfg=m.config
        field=HarmonicHoleField(
            hole_center_cm=coolant_channel_positions(cfg)[0],
            hole_radius_cm=cfg.coated_channel_radius_cm,
            amplitude_k=amplitude_k,
            conductivity_w_m_k=conductivity_w_m_k,
            axial_length_m=float(cfg.active_length_cm*.01))
        observations.append(evaluate_harmonic_curved_hole(m,field))
    errors=[obs.stage81_normalized_flux_l1_error for obs in observations]
    wall=[obs.stage80_central_wall_relative_error for obs in observations]
    return HarmonicFluxRefinement(
        levels=tuple(observations),
        interior_flux_l1_improvement=(errors[1]-errors[0],errors[2]-errors[1]),
        wall_total_flux_errors=tuple(wall[:3]),
        unresolved_nonmonotonic_wall_flux=not(
            wall[0]>wall[1]>wall[2]),
    )
