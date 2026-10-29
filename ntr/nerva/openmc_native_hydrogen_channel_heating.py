"""Stage67: native OpenMC 19-channel H2 energy-deposition handoff.

Read a native statepoint tally with [regular mesh, 19 distinct hydrogen Cell
IDs, optional neutron/photon ParticleFilter]. Native mesh X-index is fastest,
then Y, then Z. The coolant CellFilter identifies the 19 channel cells of a
REUSED fuel-element universe; it does not identify a unique lattice instance.

Fail closed unless the caller confirms single-element transport provenance,
the native mesh XY extent matches the Stage53 physical fuel element, and
all 19 expected OpenMC cell IDs agree exactly with the ordered tally bins.
Every source XY voxel's intersection with each circular coolant bore is
computed analytically, to reject heating placed in a nonexistent channel.

heating-local is accepted ONLY in neutron-only transport (secondary photon
energy presumed locally deposited); heating plus separate neutron/photon
bins is required for coupled transport. Never add heating-local to heating.
The scores represent eV per source particle. Watt normalization is an
EXTERNAL declaration and nuclear-data provenance is NOT verified here.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import h5py
import numpy as np

from .config import NervaConfig
from .layout import coolant_channel_positions
from .fuel_cutcell_conduction_2d import _disk_polygon_intersection_area
from .fuel_hydrogen_coupled_transient import CoupledCoreModel
from .openmc_native_fuel_statepoint import (
    _sha256_file, _scalar_string, _metadata_list,
)
from .openmc_fuel_axial_heat_handoff import (
    EV_TO_J, build_axial_overlap_m,
)


@dataclass(frozen=True)
class NativeHydrogenDeposition:
    """Per-[axial slab, physical coolant channel] deposited watts.

    Bounds are sums of marginal standard errors: covariance among mesh,
    particle, and channel bins is unavailable. No claim of historical
    NERVA power, a verified fuel-element instance or qualified nuclear data.
    """
    deposition_w: np.ndarray
    neutron_deposition_w: np.ndarray
    photon_deposition_w: np.ndarray
    uncertainty_upper_bound_w: np.ndarray
    source_axial_channel_w: np.ndarray
    source_channel_ids: tuple[int, ...]
    physical_bore_area_cm2: np.ndarray
    total_deposition_w: float
    first_law_source_w: float
    total_uncertainty_upper_bound_w: float
    source_statepoint_sha256: str
    source_rate_basis: str
    score_name: str
    geometry_scope: str = (
        "single-element mesh and ordered cell identities asserted; "
        "OpenMC lattice instance NOT independently proven"
    )


def _bore_intersection_area(
    xs: np.ndarray, ys: np.ndarray, config: NervaConfig,
) -> np.ndarray:
    """Exact circle/rectangle areas indexed [source x,source y,19]."""
    locations=coolant_channel_positions(config)
    radius=float(config.channel_radius_cm)
    nx,ny=len(xs)-1,len(ys)-1
    result=np.zeros((nx,ny,19))
    for channel,(cx,cy) in enumerate(locations):
        for i in range(nx):
            if xs[i+1] <= cx-radius or xs[i] >= cx+radius:
                continue
            for j in range(ny):
                if ys[j+1] <= cy-radius or ys[j] >= cy+radius:
                    continue
                rectangle=[
                    (float(xs[i]),float(ys[j])),
                    (float(xs[i+1]),float(ys[j])),
                    (float(xs[i+1]),float(ys[j+1])),
                    (float(xs[i]),float(ys[j+1])),
                ]
                result[i,j,channel]=max(0.,float(
                    _disk_polygon_intersection_area(
                        rectangle,cx,cy,radius)))
    expected=math.pi*radius*radius
    if not np.allclose(
            result.sum(axis=(0,1)),expected,rtol=2e-9,atol=2e-10):
        raise ValueError("OpenMC mesh does not cover all 19 H2 bore areas")
    return result


def read_native_19_channel_hydrogen_heating(
    statepoint_path: str | Path,
    model: CoupledCoreModel,
    *,
    expected_channel_cell_ids: tuple[int, ...],
    source_rate_particles_per_s: float,
    source_rate_basis: str,
    nuclear_data_sha256: str,
    isolated_element_geometry_confirmed: bool,
    expected_statepoint_sha256: str | None = None,
) -> NativeHydrogenDeposition:
    """Conservatively map genuine native HDF5 tally MOMENTS to H2 watts.

    Source means are per STARTING PARTICLE, not per fuel element. The caller
    must provide independently justified source-rate and unique element
    scope. A material/cell ID alone cannot distinguish reused instances.
    """
    if isolated_element_geometry_confirmed is not True:
        raise ValueError("isolated physical fuel-element instance must be confirmed")
    if (len(expected_channel_cell_ids)!=19
            or any(not isinstance(x,int) or x<=0
                   for x in expected_channel_cell_ids)
            or len(set(expected_channel_cell_ids))!=19):
        raise ValueError("19 distinct ordered physical coolant cell IDs required")
    if (not math.isfinite(source_rate_particles_per_s)
            or source_rate_particles_per_s<=0
            or not isinstance(source_rate_basis,str)
            or not source_rate_basis.strip()):
        raise ValueError("independent positive physical source-rate basis required")
    if (not isinstance(nuclear_data_sha256,str)
            or re.fullmatch("[0-9a-fA-F]{64}",nuclear_data_sha256) is None):
        raise ValueError("external evaluated nuclear-data SHA256 required")
    path=Path(statepoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest=_sha256_file(path)
    if (expected_statepoint_sha256 is not None
            and digest.lower()!=expected_statepoint_sha256.lower()):
        raise ValueError("statepoint physical source SHA256 mismatch")

    with h5py.File(path,"r") as h:
        version=np.asarray(h.attrs.get("version",[]))
        if ("filetype" not in h.attrs
                or _scalar_string(h.attrs["filetype"])!="statepoint"
                or version.shape!=(2,) or int(version[0])!=18
                or int(version[1])<0):
            raise ValueError("unsupported native OpenMC statepoint format")
        if ("photon_transport" not in h.attrs
                or "openmc_version" not in h.attrs):
            raise ValueError("native statepoint transport-mode metadata missing")
        photon_flag=np.asarray(h.attrs["photon_transport"])
        if (photon_flag.ndim or
                not np.issubdtype(photon_flag.dtype,np.integer)
                or int(photon_flag) not in (0,1)):
            raise ValueError("invalid photon transport mode")
        coupled=bool(int(photon_flag))
        if ("run_mode" not in h
                or _scalar_string(h["run_mode"][()])!="eigenvalue"
                or "energy_mode" not in h
                or _scalar_string(h["energy_mode"][()])!="continuous-energy"):
            raise ValueError("continuous-energy eigenvalue OpenMC statepoint required")
        required=("n_batches","n_inactive","current_batch","n_particles")
        if any(key not in h for key in required):
            raise ValueError("native eigenvalue batch metadata missing")
        n_batches,n_inactive,current,n_particles=(
            int(h[key][()]) for key in required)
        if (n_batches<2 or not 0<=n_inactive<current<=n_batches
                or n_particles<=0):
            raise ValueError("native source-batch metadata invalid")
        if ("tallies_present" not in h.attrs
                or int(h.attrs["tallies_present"])!=1
                or "tallies" not in h):
            raise ValueError("statepoint has no heating tally")
        tg=h["tallies"]
        if "ids" not in tg.attrs or "n_tallies" not in tg.attrs:
            raise ValueError("native tally manifest absent")
        manifest=np.asarray(tg.attrs["ids"],dtype=int).ravel()
        if (len(manifest)!=int(tg.attrs["n_tallies"])
                or len(set(manifest.tolist()))!=len(manifest)):
            raise ValueError("invalid native tally manifest")
        matches=[v for k,v in tg.items()
                 if k.startswith("tally ") and isinstance(v,h5py.Group)
                 and "name" in v
                 and _scalar_string(v["name"][()])
                 =="nerva_3d_fuel_hydrogen_heating"]
        if len(matches)!=1:
            raise ValueError("exactly one named hydrogen-heating tally required")
        t=matches[0]
        if int(t.name.rsplit(" ",1)[-1]) not in manifest:
            raise ValueError("hydrogen tally omitted from native manifest")
        fields=("filters","score_bins","nuclides","n_realizations","results")
        if any(key not in t for key in fields):
            raise ValueError("hydrogen tally lacks native score/filter/moment data")
        filters=np.asarray(t["filters"][()],dtype=int).reshape(-1)
        if (len(filters)!=(3 if coupled else 2)
                or len(set(filters.tolist()))!=len(filters)):
            raise ValueError("invalid hydrogen mesh/cell/particle filter stride")
        groups=[]
        for fil in filters:
            key=f"filters/filter {int(fil)}"
            if key not in tg:
                raise ValueError("native hydrogen filter group missing")
            groups.append(tg[key])
        kind=[_scalar_string(g["type"][()]).lower() for g in groups]
        if kind!=(
                ["mesh","cell","particle"] if coupled
                else ["mesh","cell"]):
            raise ValueError("unexpected native hydrogen filter order")
        cell_ids=np.asarray(groups[1]["bins"][()],dtype=int).ravel()
        if not np.array_equal(cell_ids,
                              np.asarray(expected_channel_cell_ids,dtype=int)):
            raise ValueError("hydrogen cell IDs do not match ordered physical 19-channel identity")
        parties=["neutron"]
        if coupled:
            parties=_metadata_list(groups[2]["bins"][()])
            if len(parties)!=2 or set(parties)!={"neutron","photon"}:
                raise ValueError("coupled hydrogen tally requires neutron and photon bins")
        score="heating" if coupled else "heating-local"
        scores=_metadata_list(t["score_bins"][()])
        if (scores.count(score)!=1 or
                ("heating-local" if coupled else "heating") in scores
                or _metadata_list(t["nuclides"][()])!=["total"]):
            raise ValueError("inconsistent neutron/photon hydrogen deposited-energy score")
        mesh_ids=np.asarray(groups[0]["bins"][()],dtype=int).ravel()
        if mesh_ids.shape!=(1,):
            raise ValueError("source hydrogen mesh ID invalid")
        mp=f"meshes/mesh {int(mesh_ids[0])}"
        if mp not in tg:
            raise ValueError("native hydrogen mesh not found")
        mesh=tg[mp]
        if _scalar_string(mesh["type"][()]).lower()!="regular":
            raise ValueError("hydrogen heating needs a native regular mesh")
        dims=np.asarray(mesh["dimension"][()],dtype=int)
        lower=np.asarray(mesh["lower_left"][()],dtype=float)
        upper=np.asarray(mesh["upper_right"][()],dtype=float)
        if (dims.shape!=(3,) or np.any(dims<1) or np.prod(dims)>250000
                or lower.shape!=(3,) or upper.shape!=(3,)
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(upper<=lower)):
            raise ValueError("native hydrogen mesh dimensions invalid")
        sx,sy,sz=(int(v) for v in dims)
        expected=np.array([
            model.solid.transverse.x_edges_cm[0],
            model.solid.transverse.y_edges_cm[0],
            model.solid.transverse.x_edges_cm[-1],
            model.solid.transverse.y_edges_cm[-1]])
        bounds=np.array([lower[0],lower[1],upper[0],upper[1]])
        if not np.allclose(expected,bounds,atol=2e-10,rtol=0):
            raise ValueError("whole-core OpenMC hydrogen mesh cannot heat one fuel element")
        if not np.isclose((upper[2]-lower[2])/100.,
                          model.solid.axial_edges_m[-1],atol=2e-11,rtol=0):
            raise ValueError("hydrogen core active length mismatch")
        area=_bore_intersection_area(
            np.linspace(lower[0],upper[0],sx+1),
            np.linspace(lower[1],upper[1],sy+1),model.config)
        n=int(t["n_realizations"][()])
        if n<2 or n>current-n_inactive:
            raise ValueError("hydrogen native tally realizations invalid")
        count=(2 if coupled else 1)
        data=np.asarray(t["results"][()],dtype=float)
        if (data.shape!=(sx*sy*sz*19*count,len(scores),2)
                or not np.all(np.isfinite(data))):
            raise ValueError("native hydrogen channel tally moments malformed")
        # Native filter stride: mesh (slowest), cell (19), particle
        # (1 or 2, fastest). Each mesh row is i+nx*(j+ny*k).
        m=data.reshape((sx*sy*sz,19,count,len(scores),2))
        score_moments=m[:,:,:,scores.index(score),:]
        mean=score_moments[...,0]/n
        var=(score_moments[...,1]/n-mean**2)/(n-1)
        if np.any(var < -1e-10*np.maximum(1.,mean**2)):
            raise ValueError("negative hydrogen heating tally variance")
        if np.any(mean<0):
            raise ValueError("negative physical energy deposition in H2 cell")
        err=np.sqrt(np.maximum(var,0.))
        # Reshape each channel/particle independent of filter strides.
        native=np.empty((sx,sy,sz,19,count))
        margins=np.empty_like(native)
        for channel in range(19):
            for particle in range(count):
                native[:,:,:,channel,particle]=mean[:,channel,particle].reshape(
                    (sx,sy,sz),order="F")
                margins[:,:,:,channel,particle]=err[:,channel,particle].reshape(
                    (sx,sy,sz),order="F")
        if (np.any(native[area<=1e-13,:]>1e-12)
                or np.any(margins[area<=1e-13,:]>1e-12)):
            raise ValueError("hydrogen heating scored outside physical channel bore")
        zsrc=np.linspace(0.,(upper[2]-lower[2])/100.,sz+1)
        overlap=build_axial_overlap_m(zsrc,model.solid.axial_edges_m)
        weight=overlap/np.diff(zsrc)[None,:]
        # Preserve both particle-type ledgers; propagate unknown covariance
        # conservatively by summing the marginal standard errors.
        power=np.sum(native,axis=(0,1))*source_rate_particles_per_s*EV_TO_J
        margin=np.sum(margins,axis=(0,1,4))*source_rate_particles_per_s*EV_TO_J
        neutron=weight@power[:,:,parties.index("neutron")]
        photon=(weight@power[:,:,parties.index("photon")]
                if coupled else np.zeros_like(neutron))
        mapped=neutron+photon
        error=weight@margin
        total_source=float(np.sum(power))
        if abs(float(np.sum(mapped))-total_source)>max(
                2e-12,2e-11*abs(total_source)):
            raise ArithmeticError("channel nuclear deposition power not conserved")
        return NativeHydrogenDeposition(
            deposition_w=mapped,
            neutron_deposition_w=neutron,
            photon_deposition_w=photon,
            uncertainty_upper_bound_w=error,
            source_axial_channel_w=np.sum(power,axis=-1),
            source_channel_ids=tuple(expected_channel_cell_ids),
            physical_bore_area_cm2=np.sum(area,axis=(0,1)),
            total_deposition_w=float(np.sum(mapped)),
            first_law_source_w=float(np.sum(mapped)),
            total_uncertainty_upper_bound_w=float(np.sum(error)),
            source_statepoint_sha256=digest,
            source_rate_basis=source_rate_basis,
            score_name=score,
        )
