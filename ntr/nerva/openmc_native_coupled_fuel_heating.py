"""Stage65: native coupled neutron/photon fuel-heating HDF5 tally reader.

Requires a real OpenMC photon-enabled eigenvalue *statepoint* with the
specific fuel-material/regular-mesh/particle-filter tally. OpenMC's
`heating` score (NOT `heating-local`) deposits neutron/photons in their
actual scored material/voxel. Particle bins are recorded separately and
summed once, never mixed with local-photon approximations.

Returns the existing native 3D fuel heating object, with distinct neutron
and photon source-voxel means. The sum of marginal standard errors bounds
combined uncertainty without assuming neutron/gamma score independence.

This parser validates *structure*, not actual evaluated nuclear library
identity, transport source convergence, fission criticality, reactor
power normalization, or which physical fuel-element instance is selected.
"""
from __future__ import annotations

import math
from pathlib import Path
import h5py
import numpy as np

from .openmc_native_fuel_statepoint import (
    NativeFuelHeatingTally, _sha256_file, _scalar_string,
    _metadata_list,
)
from .openmc_fuel_axial_heat_handoff import OpenMCAxialFuelHeating


def _positive_int(node, label: str) -> int:
    a=np.asarray(node)
    if a.ndim or not np.issubdtype(a.dtype,np.integer):
        raise ValueError(f"{label} must be an integer scalar")
    n=int(a)
    if n<=0:
        raise ValueError(f"{label} must be positive")
    return n


def read_openmc_coupled_fuel_statepoint(
    statepoint_path: str | Path,
    *,
    expected_fuel_material_id: int,
    nuclear_data_sha256: str,
    source_rate_particles_per_s: float,
    source_rate_basis: str,
    expected_active_length_m: float | None=None,
    expected_statepoint_sha256: str | None=None,
) -> NativeFuelHeatingTally:
    """Read 3-filter HEATING tally with native mesh X-fastest, particle last.

    Tally filters MUST be in this sequence: mesh, material, particle,
    and the latter must name both 'neutron' and 'photon'. The score index
    can be mixed with ordinary fission diagnostics, but `heating-local`
    is forbidden in this coupled fuel tally to prevent ambiguous physics.
    """
    path=Path(statepoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(expected_fuel_material_id,int) or expected_fuel_material_id<=0:
        raise ValueError("expected fuel material id must be positive integer")
    if not isinstance(nuclear_data_sha256,str) or len(nuclear_data_sha256)!=64:
        raise ValueError("external evaluated nuclear-data hash is required")
    try:
        int(nuclear_data_sha256,16)
    except ValueError as exc:
        raise ValueError("nuclear-data SHA256 must be hexadecimal") from exc
    if not math.isfinite(source_rate_particles_per_s) or source_rate_particles_per_s<=0:
        raise ValueError("externally justified physical source rate required")
    if not isinstance(source_rate_basis,str) or not source_rate_basis.strip():
        raise ValueError("physical source-rate provenance is required")
    digest=_sha256_file(path)
    if expected_statepoint_sha256 is not None and digest!=expected_statepoint_sha256.lower():
        raise ValueError("native statepoint SHA256 mismatch")

    with h5py.File(path,"r") as h:
        if ("filetype" not in h.attrs
                or _scalar_string(h.attrs["filetype"])!="statepoint"):
            raise ValueError("not a native OpenMC HDF5 statepoint")
        ver=np.asarray(h.attrs.get("version",[]))
        if ver.shape!=(2,) or ver[0]!=18 or ver[1]<0:
            raise ValueError("unsupported OpenMC statepoint format version")
        omc=np.asarray(h.attrs.get("openmc_version",[]))
        if omc.ndim!=1 or len(omc)<3 or not np.issubdtype(omc.dtype,np.integer):
            raise ValueError("native OpenMC version metadata missing")
        photon=np.asarray(h.attrs.get("photon_transport",-1))
        if photon.ndim or int(photon)!=1:
            raise ValueError("coupled fuel heating requires native photon_transport=1")
        if ("run_mode" not in h or _scalar_string(h["run_mode"][()])!="eigenvalue"
                or "energy_mode" not in h
                or _scalar_string(h["energy_mode"][()])!="continuous-energy"):
            raise ValueError("coupled energy heating requires continuous-energy eigenvalue transport")
        needed=("n_batches","n_inactive","current_batch","n_particles")
        if any(k not in h for k in needed):
            raise ValueError("native source-generation metadata incomplete")
        nb,ni,cur,npart=(int(h[k][()]) for k in needed)
        if nb<2 or not 0<=ni<cur<=nb or npart<=0:
            raise ValueError("invalid source-generation counts")
        if "tallies_present" not in h.attrs or int(h.attrs["tallies_present"])!=1:
            raise ValueError("native statepoint does not contain tallies")
        if "tallies" not in h:
            raise ValueError("no native OpenMC tally group")
        tg=h["tallies"]
        if "ids" not in tg.attrs or "n_tallies" not in tg.attrs:
            raise ValueError("missing native OpenMC tally manifest")
        ids=np.asarray(tg.attrs["ids"],dtype=int).ravel()
        if (len(ids)!=int(tg.attrs["n_tallies"])
                or len(set(ids.tolist()))!=len(ids)):
            raise ValueError("invalid native tally-ID manifest")
        named=[]
        for key,group in tg.items():
            if (key.startswith("tally ") and isinstance(group,h5py.Group)
                    and "name" in group
                    and _scalar_string(group["name"][()])=="nerva_3d_fuel_heating"):
                named.append(group)
        if len(named)!=1:
            raise ValueError("exactly one named coupled fuel heating tally required")
        tally=named[0]
        tally_id=int(tally.name.rsplit(" ",1)[-1])
        if tally_id not in ids:
            raise ValueError("coupled fuel tally not in native manifest")
        required=("score_bins","filters","nuclides","n_realizations","results")
        if any(field not in tally for field in required):
            raise ValueError("incomplete native fuel tally metadata")
        scores=_metadata_list(tally["score_bins"][()])
        if (scores.count("heating")!=1
                or "heating-local" in scores
                or _metadata_list(tally["nuclides"][()])!=["total"]):
            raise ValueError("coupled fuel tally needs heating (not heating-local) and nuclide total")
        fid=np.asarray(tally["filters"][()],dtype=int).ravel()
        if len(fid)!=3 or len(set(fid.tolist()))!=3:
            raise ValueError("coupled tally requires mesh/material/particle filters")
        kinds=[]
        fg=[]
        for j in fid:
            key=f"filters/filter {int(j)}"
            if key not in tg:
                raise ValueError("native coupled tally filter missing")
            group=tg[key]
            if "type" not in group or "bins" not in group:
                raise ValueError("coupled filter metadata missing")
            kinds.append(_scalar_string(group["type"][()]).lower())
            fg.append(group)
        if kinds!=["mesh","material","particle"]:
            raise ValueError("unexpected coupled tally filter stride: need mesh/material/particle")
        mat=np.asarray(fg[1]["bins"][()],dtype=int).ravel()
        if mat.shape!=(1,) or mat[0]!=expected_fuel_material_id:
            raise ValueError("coupled fuel tally material identity mismatch")
        particles=_metadata_list(fg[2]["bins"][()])
        if len(particles)!=2 or set(particles)!={"neutron","photon"}:
            raise ValueError("coupled tally must resolve exactly neutron and photon bins")
        neutron_idx=particles.index("neutron")
        photon_idx=particles.index("photon")
        mid=np.asarray(fg[0]["bins"][()],dtype=int).ravel()
        if mid.shape!=(1,):
            raise ValueError("coupled tally has incorrect mesh IDs")
        meshpath=f"meshes/mesh {int(mid[0])}"
        if meshpath not in tg:
            raise ValueError("native coupled regular mesh missing")
        mesh=tg[meshpath]
        if "type" not in mesh or _scalar_string(mesh["type"][()]).lower()!="regular":
            raise ValueError("coupled heating requires a regular Cartesian mesh")
        if any(k not in mesh for k in ("dimension","lower_left","upper_right")):
            raise ValueError("coupled mesh bounding data incomplete")
        dims=np.asarray(mesh["dimension"][()],dtype=int)
        low=np.asarray(mesh["lower_left"][()],dtype=float)
        high=np.asarray(mesh["upper_right"][()],dtype=float)
        if (dims.shape!=(3,) or np.any(dims<=0) or np.any(dims>500)
                or low.shape!=(3,) or high.shape!=(3,)
                or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high))
                or np.any(high<=low)):
            raise ValueError("invalid physical coupled source mesh geometry")
        if expected_active_length_m is not None and not np.isclose(
            (high[2]-low[2])*.01,expected_active_length_m,
            rtol=0,atol=2e-11):
            raise ValueError("coupled fuel source axial length mismatch")
        count=_positive_int(tally["n_realizations"][()],"tally realizations")
        if count<2 or count>cur-ni:
            raise ValueError("coupled tally realizations outside active batch count")
        nmesh=int(np.prod(dims))
        data=np.asarray(tally["results"][()],dtype=float)
        if (data.shape!=(2*nmesh,len(scores),2)
                or not np.all(np.isfinite(data))):
            raise ValueError("coupled native tally moment dimensions invalid")
        # Mesh (slowest filter), material (one bin), then particle
        # (fastest filter). Mesh native indices X fastest.
        bins=data.reshape(nmesh,2,len(scores),2)
        si=scores.index("heating")
        moment=bins[:,:,si,:]
        means=moment[:,:,0]/count
        var=(moment[:,:,1]/count-means**2)/(count-1)
        if np.any(var < -1e-10*np.maximum(1.,means**2)):
            raise ValueError("negative physical tally variance")
        if np.any(means<0):
            raise ValueError("negative coupled deposited-energy mean")
        sd=np.sqrt(np.maximum(0.,var))
        n3=means[:,neutron_idx].reshape(tuple(dims),order="F")
        p3=means[:,photon_idx].reshape(tuple(dims),order="F")
        # Correlation of n/gamma scores in each source realization is
        # unavailable: total sigma <= sigma(neutron)+sigma(photon).
        stderr3=(sd[:,neutron_idx]+sd[:,photon_idx]).reshape(
            tuple(dims),order="F")
        combined=n3+p3
        axial_n=n3.sum(axis=(0,1))
        axial_p=p3.sum(axis=(0,1))
        zface=np.linspace(0.,(high[2]-low[2])*.01,int(dims[2])+1)
        score=OpenMCAxialFuelHeating(
            axial_edges_m=zface,
            fuel_neutron_ev_per_source=axial_n,
            fuel_photon_ev_per_source=axial_p,
            source_rate_particles_per_s=source_rate_particles_per_s,
            statepoint_sha256=digest,
            nuclear_data_sha256=nuclear_data_sha256,
            source_rate_basis=source_rate_basis,
        )
        score.validate()
        return NativeFuelHeatingTally(
            score=score,
            tally_id=tally_id,
            fuel_material_id=expected_fuel_material_id,
            mesh_dimensions=tuple(int(x) for x in dims),
            active_realizations=count,
            source_mesh_origin_cm=tuple(float(x) for x in low),
            source_mesh_upper_cm=tuple(float(x) for x in high),
            axial_uncertainty_upper_bound_ev_per_source=stderr3.sum(axis=(0,1)),
            source_mesh_mean_ev_per_source=combined,
            source_mesh_stderr_ev_per_source=stderr3,
            statepoint_sha256=digest,
            openmc_version=tuple(int(x) for x in omc[:3]),
            statepoint_format_version=tuple(int(x) for x in ver),
            current_batch=cur,
            energy_mode="continuous-energy",
            photon_transport_verified_disabled=False,
            n_batches=nb,n_inactive=ni,n_particles=npart,
            photon_status="coupled heating score: neutron and photon physical deposition are separately binned",
            source_mesh_neutron_mean_ev_per_source=n3,
            source_mesh_photon_mean_ev_per_source=p3,
        )
