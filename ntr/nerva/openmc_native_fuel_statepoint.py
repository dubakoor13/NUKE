"""Stage60: read *native* OpenMC HDF5 fuel heating without manufactured tallies.

Restricted to NERVA's neutron-only 'nerva_3d_fuel_heating' tally with exactly
one regular mesh filter, one fuel-material filter, nuclide 'total', score
'heating-local', and >1 active tally realizations. Native HDF5 stores SUM
and SUM OF SQUARES, not direct watts. Recover the mean eV/source and
standard error per mesh bin from independent-realization moments.

OpenMC mesh tally bins have z as the fastest-varying index (x,y,z array).
Axial totals sum x/y means. Correlations between x/y bins within the same
source batch are *unknown from sum/sum_sq alone*: the reported axial
uncertainty is the triangle-inequality upper bound, not an unjustified
quadrature standard deviation. Do not interpret it as an exact covariance.

Photon-transport mode is explicitly DISALLOWED because heating-local alone
is not the complete coupled neutron/photon deposited energy. An external
caller must provide evidence that the actual run had photon transport OFF;
the HDF5 file alone is not accepted as proof of that settings choice.

File SHA256 is *computed* from bytes; library hash and source-particle
rate are externally supplied evidence that cannot be verified by this
reader. This module never infers physical source normalization from keff.
It does not certify historical NERVA geometry, criticality, power, or
evaluated hot nuclear cross-section coverage.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re

import h5py
import numpy as np

from .openmc_fuel_axial_heat_handoff import (
    OpenMCAxialFuelHeating, ResolvedFuelHeatingInput,
    project_openmc_axial_fuel_heating,
)

_SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")
_EXPECTED_TALLY = "nerva_3d_fuel_heating"
_EXPECTED_SCORE = "heating-local"


@dataclass(frozen=True)
class NativeFuelHeatingTally:
    """Extracted genuine HDF5 moments, NOT physical NERVA certification."""
    score: OpenMCAxialFuelHeating
    tally_id: int
    fuel_material_id: int
    mesh_dimensions: tuple[int,int,int]
    active_realizations: int
    source_mesh_origin_cm: tuple[float,float,float]
    source_mesh_upper_cm: tuple[float,float,float]
    axial_uncertainty_upper_bound_ev_per_source: np.ndarray
    source_mesh_mean_ev_per_source: np.ndarray
    source_mesh_stderr_ev_per_source: np.ndarray
    statepoint_sha256: str
    uncertainty_status: str = (
        "per-bin sample mean stderr; axial uncertainty upper-bound "
        "by sum of marginal stderr; inter-bin covariance unavailable"
    )
    photon_status: str = (\n        "neutron-only transport heating-local: secondary photon energy "
        "is ASSUMED DEPOSITED LOCALLY; this is NOT a pure-neutron "
        "energy partition and must not be added to independent photon "
        "heating without a separately validated transport accounting"
    )


@dataclass(frozen=True)
class NativeMappedFuelHeating:
    """Mean fuel watts and conservative marginal-error bounds per target bin.

    Bounds are NOT an independently measured uncertainty for each mapped
    bin and are NOT propagated reactor-system covariance. With only
    tally sum/sum_sq, covariance cannot be reconstructed.
    """
    resolved: ResolvedFuelHeatingInput
    axial_uncertainty_upper_bound_w: np.ndarray
    fuel_cell_uncertainty_upper_bound_w: np.ndarray
    total_fuel_uncertainty_upper_bound_w: float
    relative_total_uncertainty_upper_bound: float
    uncertainty_status: str = (
        "conservative triangle-inequality bound from marginal batch "
        "standard errors; no inter-bin covariance data"
    )


def project_native_fuel_heating(
    native: NativeFuelHeatingTally,
    model: object,
) -> NativeMappedFuelHeating:
    """Project the exact same positive linear mapping for means and bounds.

    Mean heating is the existing Stage59 source->axial->fuel-only-volume
    map. Absolute uncertainty upper bounds propagate through those
    nonnegative weights, without asserting missing covariance is zero.
    """
    resolved=project_openmc_axial_fuel_heating(model,native.score)
    rate=native.score.source_rate_particles_per_s
    uncertainty_source=np.asarray(
        native.axial_uncertainty_upper_bound_ev_per_source,dtype=float)
    shape=(len(native.score.axial_edges_m)-1,)
    if (uncertainty_source.shape!=shape or
            not np.all(np.isfinite(uncertainty_source)) or
            np.any(uncertainty_source<0)):
        raise ValueError("native axial uncertainty bounds are invalid")
    unit_rate=rate*1.602176634e-19
    fractions=resolved.axial_overlap_m/np.diff(
        np.asarray(native.score.axial_edges_m,dtype=float))[None,:]
    uncertainty_axial=fractions@(uncertainty_source*unit_rate)
    uncertainty_fuel=(uncertainty_axial[:,None]
                      *resolved.transverse_fuel_weights[None,:])
    conservative_total=float(np.sum(uncertainty_fuel))
    mean_total=float(resolved.mapped_fuel_power_w)
    if mean_total<=0:
        raise ValueError("cannot report relative uncertainty for zero deposited fuel heat")
    if not np.isclose(float(np.sum(uncertainty_axial)),
                      float(np.sum(uncertainty_source*unit_rate)),
                      rtol=1e-13,atol=1e-13):
        raise ArithmeticError("HDF5 uncertainty-bound weights not conservative")
    return NativeMappedFuelHeating(
        resolved=resolved,
        axial_uncertainty_upper_bound_w=uncertainty_axial,
        fuel_cell_uncertainty_upper_bound_w=uncertainty_fuel,
        total_fuel_uncertainty_upper_bound_w=conservative_total,
        relative_total_uncertainty_upper_bound=conservative_total/mean_total,
    )


def _scalar_string(value: object) -> str:
    a = np.asarray(value)
    if a.ndim == 0:
        item=a.item()
    elif a.size==1:
        item=a.reshape(-1)[0]
    else:
        raise ValueError("OpenMC metadata must be one scalar string")
    if isinstance(item,bytes):
        return item.decode("utf-8").strip("\x00 ")
    return str(item).strip("\x00 ")


def _sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata_list(dataset: object) -> list[str]:
    v=np.asarray(dataset)
    return [_scalar_string(x) for x in np.atleast_1d(v)]


def _validate_file_and_tally(hdf: h5py.File, fuel_id: int) -> tuple[h5py.Group,int,tuple[int,int,int],np.ndarray,np.ndarray]:
    if "filetype" not in hdf.attrs or _scalar_string(hdf.attrs["filetype"])!="statepoint":
        raise ValueError("not a native OpenMC statepoint HDF5 file")
    if "run_mode" not in hdf or _scalar_string(hdf["run_mode"][()])!="eigenvalue":
        raise ValueError("only original OpenMC eigenvalue source-normalized tallies supported")
    if "tallies" not in hdf or "tallies_present" not in hdf.attrs or int(hdf.attrs["tallies_present"])!=1:
        raise ValueError("native statepoint does not contain user tally data")
    tg=hdf["tallies"]
    matches=[group for label,group in tg.items()
             if isinstance(group,h5py.Group) and label.startswith("tally ")
             and "name" in group and _scalar_string(group["name"][()])==_EXPECTED_TALLY]
    if len(matches)!=1:
        raise ValueError("exactly one named NERVA fuel heating tally required")
    tally=matches[0]
    if "score_bins" not in tally or "nuclides" not in tally or "filters" not in tally:
        raise ValueError("OpenMC fuel tally lacks score/nuclide/filter metadata")
    scores=_metadata_list(tally["score_bins"][()])
    if scores.count(_EXPECTED_SCORE)!=1:
        raise ValueError("native fuel tally must score heating-local exactly once")
    if _metadata_list(tally["nuclides"][()])!=["total"]:
        raise ValueError("nuclide-resolved energy tally unsupported without explicit audit")
    ids=np.asarray(tally["filters"][()],dtype=int).reshape(-1)
    if len(ids)!=2 or len(set(ids.tolist()))!=2:
        raise ValueError("require exactly mesh and fuel material tally filters")
    filters={}
    for id_ in ids:
        key=f"filters/filter {int(id_)}"
        if key not in tg:
            raise ValueError("missing native OpenMC filter group")
        fg=tg[key]
        if "type" not in fg or "bins" not in fg:
            raise ValueError("native filter lacks type or bin identities")
        kind=_scalar_string(fg["type"][()]).lower()
        if kind in filters or kind not in ("mesh","material"):
            raise ValueError("only a unique mesh and fuel-material filter supported")
        filters[kind]=fg
    if set(filters)!=set(("mesh","material")):
        raise ValueError("native filter identities do not match NERVA fuel tally")
    mat=np.asarray(filters["material"]["bins"][()],dtype=int).reshape(-1)
    if mat.shape!=(1,) or int(mat[0])!=fuel_id:
        raise ValueError("OpenMC tally material ID does not equal expected fuel material")
    mesh_filter=filters["mesh"]
    meshnames=np.asarray(mesh_filter["bins"][()],dtype=int).reshape(-1)
    if meshnames.shape!=(1,):
        raise ValueError("mesh filter must identify exactly one RegularMesh")
    path=f"meshes/mesh {int(meshnames[0])}"
    if path not in tg:
        raise ValueError("native mesh ID does not exist")
    mesh=tg[path]
    if "type" not in mesh or _scalar_string(mesh["type"][()]).lower()!="regular":
        raise ValueError("only native regular Cartesian meshes supported")
    needed=("dimension","lower_left","upper_right")
    if any(name not in mesh for name in needed):
        raise ValueError("native regular mesh lacks dimensions or geometry")
    dimension=np.asarray(mesh["dimension"][()],dtype=int)
    low=np.asarray(mesh["lower_left"][()],dtype=float)
    high=np.asarray(mesh["upper_right"][()],dtype=float)
    if (dimension.shape!=(3,) or np.any(dimension<1)
            or np.any(dimension>1000) or low.shape!=(3,) or high.shape!=(3,)
            or not np.all(np.isfinite(low)) or not np.all(np.isfinite(high))
            or np.any(high<=low)):
        raise ValueError("invalid native regular mesh geometry")
    return tally,int(tally.name.rsplit(" ",1)[-1]),tuple(int(i) for i in dimension),low,high


def read_openmc_neutron_fuel_statepoint(
    statepoint_path: str | Path,
    *,
    expected_fuel_material_id: int,
    nuclear_data_sha256: str,
    source_rate_particles_per_s: float,
    source_rate_basis: str,
    photon_transport_confirmed_disabled: bool,
    expected_statepoint_sha256: str | None = None,
    expected_active_length_m: float | None = None,
) -> NativeFuelHeatingTally:
    """Read real OpenMC HDF5 tally moments, fail closed on metadata mismatch.

    photon_transport_confirmed_disabled must be positively confirmed from
    independently recorded OpenMC settings. It is not inferred here.
    """
    path=Path(statepoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(photon_transport_confirmed_disabled,bool) or not photon_transport_confirmed_disabled:
        raise ValueError("neutron-only evidence required; coupled photons cannot use heating-local alone")
    if not isinstance(expected_fuel_material_id,int) or expected_fuel_material_id<=0:
        raise ValueError("exact positive fuel material ID is required")
    if not _SHA256.fullmatch(str(nuclear_data_sha256)):
        raise ValueError("evaluated nuclear-data SHA256 required")
    if (not math.isfinite(source_rate_particles_per_s) or source_rate_particles_per_s<=0
            or not isinstance(source_rate_basis,str) or not source_rate_basis.strip()):
        raise ValueError("externally justified source rate and basis required")
    if expected_active_length_m is not None and (
        not math.isfinite(expected_active_length_m) or expected_active_length_m<=0):
        raise ValueError("expected active length must be finite positive")
    file_sha=_sha256_file(path)
    if expected_statepoint_sha256 is not None and (
            file_sha.lower()!=str(expected_statepoint_sha256).lower()):
        raise ValueError("statepoint bytes SHA256 mismatch")
    with h5py.File(path,"r") as hdf:
        tally,tid,dims,low,high=_validate_file_and_tally(
            hdf,expected_fuel_material_id)
        if expected_active_length_m is not None and not np.isclose(
                (high[2]-low[2])/100.,expected_active_length_m,
                rtol=0,atol=2e-11):
            raise ValueError("statepoint mesh length differs from selected fuel core")
        if "n_realizations" not in tally or "results" not in tally:
            raise ValueError("actual OpenMC tally moments are absent")
        n=int(tally["n_realizations"][()])
        if n<2:
            raise ValueError("cannot derive tally uncertainty from fewer than two realizations")
        scores=_metadata_list(tally["score_bins"][()])
        index=scores.index(_EXPECTED_SCORE)
        data=np.asarray(tally["results"][()],dtype=float)
        if (data.ndim!=3 or data.shape!=(int(np.prod(dims)),len(scores),2)
                or not np.all(np.isfinite(data))):
            raise ValueError("native OpenMC tally moment shape or values invalid")
        # sum, sum_sq are per active realization; score_units=eV/source
        total=data[:,index,0]
        sum_sq=data[:,index,1]
        mean=total/n
        variance=(sum_sq/n-mean**2)/(n-1)
        if np.any(variance < -1e-10*np.maximum(1.,mean**2)):
            raise ValueError("OpenMC sum/sum_sq imply nonphysical negative variance")
        variance=np.maximum(variance,0.)
        stderr=np.sqrt(variance)
        if np.any(mean < 0):
            raise ValueError("negative mean deposited-energy score cannot be used as heat")
        # Native OpenMC mesh bins: x,y,z with z fastest (C-order).
        m=mean.reshape(dims)
        err=stderr.reshape(dims)
        axial=np.sum(m,axis=(0,1))
        # Covariance between mesh bins is absent from the native two-moment
        # export. Sum of individual standard errors bounds correlated total.
        bound=np.sum(err,axis=(0,1))
        # Stage59's axial-score coordinate originates at z=0, whereas
        # native OpenMC geometry is generally z=[-L/2,+L/2] centimetres.
        zfaces=(np.linspace(low[2],high[2],dims[2]+1)-low[2])/100.
        score=OpenMCAxialFuelHeating(
            axial_edges_m=zfaces,
            fuel_neutron_ev_per_source=axial,
            fuel_photon_ev_per_source=np.zeros_like(axial),
            source_rate_particles_per_s=source_rate_particles_per_s,
            statepoint_sha256=file_sha,
            nuclear_data_sha256=nuclear_data_sha256,
            source_rate_basis=source_rate_basis,
            unresolved_nonfuel_ev_per_source=0.,
        )
        score.validate()
        return NativeFuelHeatingTally(
            score=score,tally_id=tid,
            fuel_material_id=expected_fuel_material_id,
            mesh_dimensions=dims,
            active_realizations=n,
            source_mesh_origin_cm=tuple(low.tolist()),
            source_mesh_upper_cm=tuple(high.tolist()),
            axial_uncertainty_upper_bound_ev_per_source=bound,
            source_mesh_mean_ev_per_source=m,
            source_mesh_stderr_ev_per_source=err,
            statepoint_sha256=file_sha,
        )
