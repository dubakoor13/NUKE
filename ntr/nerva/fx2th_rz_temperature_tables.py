"""Fixed-mesh, temperature-indexed MULTIGROUP input tables for core neutronics.

This module is a strict *interpolation and differentiation interface* for
externally supplied spatial nuclear-data tables, NOT a Doppler model, nuclear
data processor, cross-section generator, or physically qualified NERVA data.

A table has two or more common absolute-temperature knots [K]. Each knot
contains physical SI spatial coefficients on the same (r,z,energy) mesh:
D [m], Sigma_a [1/m], Sigma_s[in_group,out_group] [1/m],
nuSigma_f [1/m], normalized outgoing prompt chi, beta[in_group,family],
normalized outgoing chi_delayed[family,out_group].

For each spatial cell, use continuous piecewise LINEAR interpolation in
absolute K independently of other cells; no extrapolation. Convex
interpolation preserves positive transport input coefficients, normalized
birth spectra and 0 <= sum(beta_family) < 1. Different nuclear coefficients
have different physical temperature dependence: LINEAR interpolation here
is only an explicit numerical approximation to supplied evaluated tables,
NOT proof of physically correct resonance Doppler or thermal scattering.

The derivative in an interior knot is discontinuous; eigen sensitivity
therefore fails closed for nonzero temperature perturbation at any knot.
At non-knot states, dXS/dT and dSpectrum/dT are analytical table slopes,
and Stage-16 exact r-z harmonic-face and full-family adjoint derivative
calculates d ln(k)/dtheta for user-supplied dT_cell/dtheta.

The output is not real NERVA k(T) without consistent nuclear-data library,
root-reachable geometry/material chemistry, group collapsing, temperature
coverage, source-convergence, and independent OpenMC transport checks.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
import re

import numpy as np

from .fx2th_rz_family_diffusion import RZGrid, rz_to_spatial_families
from .fx2th_family_spacetime import SpatialFamilyData
from .fx2th_rz_adjoint_sensitivity import (
    CoreEigenSensitivity, evaluate_rz_material_sensitivity,
)


_HASH = re.compile(r"^[0-9a-f]{64}$")


def _array(value, name: str, shape: tuple[int,...],
           *, positive: bool = False) -> np.ndarray:
    a=np.asarray(value,dtype=float)
    if (a.shape!=shape or not np.all(np.isfinite(a))
            or np.any(a<=0) if positive else
            a.shape!=shape or not np.all(np.isfinite(a)) or np.any(a<0)):
        raise ValueError(f"{name} must be finite {'positive' if positive else 'nonnegative'} {shape}")
    return a


@dataclass(frozen=True)
class RZTemperatureTables:
    """Explicit multigroup material arrays; T first, then r,z,group.

    'externally-tabulated' means source data are supplied, NOT independent
    physical qualification. Manifest SHA is a DECLARED hash token; it is
    not independently compared with an actual data file by this module.
    """
    source_id: str
    source_kind: str
    nuclear_manifest_sha256: str
    reference_grid: RZGrid
    temperature_knots_k: np.ndarray
    energy_group_edges_ev: np.ndarray
    diffusion_m: np.ndarray
    absorption_inv_m: np.ndarray
    scattering_inv_m: np.ndarray
    nu_fission_inv_m: np.ndarray
    prompt_spectrum: np.ndarray
    beta_by_incident_group_and_family: np.ndarray
    delayed_spectra_by_family: np.ndarray
    group_speeds_m_s: np.ndarray
    decay_constants_s_inv: np.ndarray

    def __post_init__(self):
        if not isinstance(self.source_id,str) or not self.source_id.strip():
            raise ValueError("source ID must be a nonblank provenance label")
        if self.source_kind not in ("synthetic","externally-tabulated"):
            raise ValueError("source kind must be synthetic or externally-tabulated")
        if not isinstance(self.nuclear_manifest_sha256,str) or not _HASH.fullmatch(
                self.nuclear_manifest_sha256):
            raise ValueError("nuclear manifest must have declared lowercase SHA256")
        if not isinstance(self.reference_grid,RZGrid):
            raise TypeError("reference grid must be RZGrid")
        k=np.asarray(self.temperature_knots_k,dtype=float)
        if (k.ndim!=1 or len(k)<2 or not np.all(np.isfinite(k))
                or np.any(k<=0) or np.any(np.diff(k)<=0)):
            raise ValueError("two or more strictly increasing positive absolute K knots required")
        nr,nz,ng=np.asarray(self.reference_grid.diffusion_m).shape
        nt=len(k)
        e=np.asarray(self.energy_group_edges_ev,dtype=float)
        if (e.shape!=(ng+1,) or not np.all(np.isfinite(e))
                or e[0]<0 or np.any(np.diff(e)<=0)):
            raise ValueError("energy group edges must be finite increasing eV array")
        lam=np.asarray(self.decay_constants_s_inv,dtype=float)
        if lam.ndim!=1 or lam.size<1:
            raise ValueError("one or more delayed families required")
        nf=len(lam)
        _array(lam,"decay constants",(nf,),positive=True)
        _array(self.group_speeds_m_s,"group speeds",(ng,),positive=True)
        shape=(nt,nr,nz,ng)
        _array(self.diffusion_m,"D",shape,positive=True)
        _array(self.absorption_inv_m,"absorption",shape,positive=True)
        _array(self.nu_fission_inv_m,"nuSigmaF",shape)
        _array(self.scattering_inv_m,"scattering",shape+(ng,))
        beta=_array(self.beta_by_incident_group_and_family,"beta",shape+(nf,))
        chip=_array(self.prompt_spectrum,"prompt spectra",shape)
        chid=_array(self.delayed_spectra_by_family,"family spectra",
                    (nt,nr,nz,nf,ng))
        s=np.asarray(self.scattering_inv_m)
        if np.any(np.diagonal(s,axis1=-2,axis2=-1)!=0):
            raise ValueError("self-scattering diagonal must be zero")
        if np.any(beta.sum(axis=-1)>=1):
            raise ValueError("total beta must remain below one")
        if not np.allclose(chip.sum(axis=-1),1.,rtol=0,atol=1e-12):
            raise ValueError("prompt birth energy spectra must be normalized")
        if not np.allclose(chid.sum(axis=-1),1.,rtol=0,atol=1e-12):
            raise ValueError("delayed family birth energy spectra must be normalized")
        if np.any(np.asarray(self.nu_fission_inv_m).sum(axis=(1,2,3))<=0):
            raise ValueError("each temperature knot needs nonzero fission source")

    @property
    def spatial_shape(self) -> tuple[int,int]:
        a=np.asarray(self.reference_grid.diffusion_m)
        return int(a.shape[0]),int(a.shape[1])


@dataclass(frozen=True)
class RZTemperatureSlopes:
    diffusion_m_per_k: np.ndarray
    absorption_inv_m_per_k: np.ndarray
    scattering_inv_m_per_k: np.ndarray
    nu_fission_inv_m_per_k: np.ndarray
    prompt_spectrum_per_k: np.ndarray
    beta_by_incident_group_and_family_per_k: np.ndarray
    delayed_spectra_by_family_per_k: np.ndarray


@dataclass(frozen=True)
class RZTemperatureSample:
    temperature_k: np.ndarray
    grid: RZGrid
    material: SpatialFamilyData
    slopes: RZTemperatureSlopes
    lower_knot_indices: np.ndarray
    at_any_knot: np.ndarray
    source_kind: str
    nuclear_manifest_sha256: str


@dataclass(frozen=True)
class RZTemperatureSensitivity:
    sampled: RZTemperatureSample
    temperature_direction_k: np.ndarray
    adjoint: CoreEigenSensitivity


def sample_rz_temperature_tables(
    table:RZTemperatureTables,
    spatial_temperatures_k: np.ndarray,
) -> RZTemperatureSample:
    """Use individual cell-temperature brackets, without extrapolation.

    Exactly at a knot, the *value* is valid, but the derivative is
    generally not unique. The one-sided tabular slope is returned along
    with at_any_knot markers; sensitivity evaluation refuses such cells
    if their temperature perturbation is nonzero.
    """
    nr,nz=table.spatial_shape
    T=np.asarray(spatial_temperatures_k,dtype=float)
    if (T.shape!=(nr,nz) or not np.all(np.isfinite(T)) or
            np.any(T<=0)):
        raise ValueError("spatial temperatures must be finite positive [r,z] K")
    knots=np.asarray(table.temperature_knots_k,dtype=float)
    if np.any(T<knots[0]) or np.any(T>knots[-1]):
        raise ValueError("temperature extrapolation outside evaluated knot domain is forbidden")
    j=np.searchsorted(knots,T,side="right")-1
    j=np.clip(j,0,len(knots)-2)
    ir,iz=np.indices((nr,nz))
    Tmin=knots[j]
    Tmax=knots[j+1]
    w=(T-Tmin)/(Tmax-Tmin)
    at_knot=np.zeros((nr,nz),dtype=bool)
    for knot in knots:
        # Do not accept a derivative at a nondifferentiable temperature knot.
        at_knot|=(T==knot)

    def interpolate(field):
        a=np.asarray(field,dtype=float)
        low=a[j,ir,iz]
        high=a[j+1,ir,iz]
        tail=low.ndim-2
        weight=w.reshape((nr,nz)+(1,)*tail)
        width=(Tmax-Tmin).reshape((nr,nz)+(1,)*tail)
        slope=(high-low)/width
        value=(1.-weight)*low+weight*high
        return value,slope

    D,dD=interpolate(table.diffusion_m)
    sa,dsa=interpolate(table.absorption_inv_m)
    ss,dss=interpolate(table.scattering_inv_m)
    nf,dnf=interpolate(table.nu_fission_inv_m)
    cp,dcp=interpolate(table.prompt_spectrum)
    beta,dbeta=interpolate(table.beta_by_incident_group_and_family)
    cd,dcd=interpolate(table.delayed_spectra_by_family)
    grid=replace(table.reference_grid,
        diffusion_m=D,absorption_inv_m=sa,
        scattering_inv_m=ss,nu_fission_inv_m=nf,fission_spectrum=cp)
    identity=hashlib.sha256()
    identity.update(b"NUKE-RZ-MULTIGROUP-T-STATE-v1\x00")
    identity.update(table.source_id.encode("utf-8"))
    identity.update(table.nuclear_manifest_sha256.encode("ascii"))
    identity.update(np.ascontiguousarray(T,dtype="<f8").tobytes())
    source=f"{table.source_id}:Tsha256={identity.hexdigest()}"
    nc=nr*nz;ng=D.shape[-1];nfamily=len(table.decay_constants_s_inv)
    material=rz_to_spatial_families(
        grid,source_id=source,
        group_speeds_m_s=np.asarray(table.group_speeds_m_s,dtype=float),
        beta_by_incident_group_and_family=beta.reshape(nc,ng,nfamily),
        delayed_spectra_by_family=cd.reshape(nc,nfamily,ng),
        decay_constants_s_inv=np.asarray(table.decay_constants_s_inv,dtype=float),
        prompt_spectrum=cp.reshape(nc,ng)
    )
    return RZTemperatureSample(
        temperature_k=T.copy(),grid=grid,material=material,
        slopes=RZTemperatureSlopes(dD,dsa,dss,dnf,dcp,dbeta,dcd),
        lower_knot_indices=j.copy(),at_any_knot=at_knot.copy(),
        source_kind=table.source_kind,
        nuclear_manifest_sha256=table.nuclear_manifest_sha256
    )


def evaluate_rz_temperature_direction(
    table:RZTemperatureTables,
    spatial_temperatures_k:np.ndarray,
    temperature_direction_k:np.ndarray,
) -> RZTemperatureSensitivity:
    """Compute adjoint d(k)/dtheta for user supplied dT_cell/dtheta.

    For dT/dtheta==1 K in all cells, this yields the sensitivity to
    uniform temperature shift ALONG THE PROVIDED TABULAR MATERIAL CURVES,
    not a physically qualified NERVA Doppler coefficient. When dT=0 in
    a nondifferentiable knot cell, the local field derivative contributes
    zero and may safely be omitted.
    """
    sampled=sample_rz_temperature_tables(table,spatial_temperatures_k)
    direction=np.asarray(temperature_direction_k,dtype=float)
    nr,nz=table.spatial_shape
    if direction.shape!=(nr,nz) or not np.all(np.isfinite(direction)):
        raise ValueError("temperature perturbation must be finite [r,z] K per parameter")
    if np.any(sampled.at_any_knot & (direction!=0)):
        raise ValueError("temperature derivative at a tabulated knot is ambiguous")
    s=sampled.slopes
    def scale(x):
        a=np.asarray(x)
        return (a*direction.reshape((nr,nz)+(1,)*(a.ndim-2)))
    nc=nr*nz
    ng=sampled.material.groups
    nf=sampled.material.families
    result=evaluate_rz_material_sensitivity(
        sampled.grid,sampled.material,
        d_diffusion_m=scale(s.diffusion_m_per_k),
        d_absorption_inv_m=scale(s.absorption_inv_m_per_k),
        d_scattering_inv_m=scale(s.scattering_inv_m_per_k),
        d_nu_sigma_f_inv_m=scale(s.nu_fission_inv_m_per_k).reshape(nc,ng),
        d_beta_by_incident_group_and_family=scale(
            s.beta_by_incident_group_and_family_per_k).reshape(nc,ng,nf),
        d_prompt_spectrum=scale(s.prompt_spectrum_per_k).reshape(nc,ng),
        d_delayed_spectra_by_family=scale(
            s.delayed_spectra_by_family_per_k).reshape(nc,nf,ng),
    )
    return RZTemperatureSensitivity(
        sampled=sampled,temperature_direction_k=direction.copy(),
        adjoint=result
    )
