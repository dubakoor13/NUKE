"""Stage59 conservative OpenMC-score-to-resolved-fuel heating handoff.

This is a *numeric handoff* for exported fuel heating scores, NOT an OpenMC
statepoint reader, fission-power normalization algorithm or an assertion
that a genuine hot NERVA run was physically validated. Supplied heating
data are per starting/source particle in eV (one neutron and one photon
component per axial bin). The caller must supply an independently
justified *physical source-particle rate* before any power in W can be
reported.

Axial bin mapping uses exact length intersections; transverse deposition
is EXPLICITLY uniform per unit *actual fuel-only cut-cell volume*, as
there is no local per-cell OpenMC tally in this interface. This uniform
shape assumption can mask local peaks and must not be called genuine 3D
OpenMC spatial heating. All unmapped deposited material power remains
outside the fuel solver as a separate unresolved ledger entry.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np
from scipy.sparse import csr_matrix

from .fuel_cutcell_axial_transport import AxialCutCellModel

EV_TO_J = 1.602176634e-19
_SHA = re.compile(r"^[0-9a-fA-F]{64}$")


def _edges(label: str, data: object) -> np.ndarray:
    v=np.asarray(data,dtype=float)
    if (v.ndim!=1 or len(v)<2 or not np.all(np.isfinite(v))
            or abs(float(v[0]))>1e-12 or np.any(np.diff(v)<=0)):
        raise ValueError(f"{label}: require finite increasing z faces starting at zero")
    return v


def _nonnegative(label: str, data: object, shape: tuple[int,...]
                 ) -> np.ndarray:
    v=np.asarray(data,dtype=float)
    if v.shape != shape or not np.all(np.isfinite(v)) or np.any(v<0):
        raise ValueError(f"{label}: finite nonnegative values with shape {shape} required")
    return v


@dataclass(frozen=True)
class OpenMCAxialFuelHeating:
    """User-supplied OpenMC-exported heating scores per starting particle.

    Hashes identify claimed input files, but the solver cannot verify
    their actual provenance without opening the corresponding files.
    `source_rate_particles_per_s` is a physical input, NOT inferred from
    k_eff or a tally value. All scores are deposited energy eV/source.
    """
    axial_edges_m: object
    fuel_neutron_ev_per_source: object
    fuel_photon_ev_per_source: object
    source_rate_particles_per_s: float
    statepoint_sha256: str
    nuclear_data_sha256: str
    source_rate_basis: str
    unresolved_nonfuel_ev_per_source: float = 0.

    def validate(self) -> None:
        z=_edges("OpenMC source axial edges",self.axial_edges_m)
        n=len(z)-1
        _nonnegative("neutron heating",self.fuel_neutron_ev_per_source,(n,))
        _nonnegative("photon heating",self.fuel_photon_ev_per_source,(n,))
        if (not math.isfinite(self.source_rate_particles_per_s)
                or self.source_rate_particles_per_s<=0):
            raise ValueError("externally justified positive source rate required")
        if (not math.isfinite(self.unresolved_nonfuel_ev_per_source)
                or self.unresolved_nonfuel_ev_per_source<0):
            raise ValueError("unresolved nonfuel deposition must be nonnegative")
        if not _SHA.fullmatch(self.statepoint_sha256):
            raise ValueError("statepoint SHA256 must be recorded")
        if not _SHA.fullmatch(self.nuclear_data_sha256):
            raise ValueError("nuclear-data SHA256 must be recorded")
        if not isinstance(self.source_rate_basis,str) or not self.source_rate_basis.strip():
            raise ValueError("source-particle normalization basis must be documented")


@dataclass(frozen=True)
class ResolvedFuelHeatingInput:
    fuel_deposition_w: np.ndarray  # [axial slab, actual Stage53 fuel cell]
    source_axial_bin_w: np.ndarray
    mapped_axial_bin_w: np.ndarray
    axial_overlap_m: np.ndarray
    transverse_fuel_weights: np.ndarray
    fuel_neutron_power_w: float
    fuel_photon_power_w: float
    unresolved_nonfuel_power_w: float
    fuel_power_balance_residual_w: float
    statepoint_sha256: str
    nuclear_data_sha256: str
    source_rate_basis: str
    transverse_approximation: str = "uniform_per_actual_fuel_only_volume"

    @property
    def mapped_fuel_power_w(self) -> float:
        return float(np.sum(self.fuel_deposition_w))


def build_axial_overlap_m(
    source_edges_m: object, destination_edges_m: object,
) -> np.ndarray:
    """Exact piecewise-constant source-to-destination length intersections.

    Output [destination zone, source zone]. Supports differing bin counts
    or nonuniform z widths, but NO partial coverage or overhanging volume.
    """
    source=_edges("source axial faces",source_edges_m)
    dest=_edges("destination axial faces",destination_edges_m)
    if not np.isclose(source[-1],dest[-1],rtol=0,atol=2e-11):
        raise ValueError("source and target axial geometry have different lengths")
    overlap=np.maximum(
        0.,np.minimum(dest[1:,None],source[None,1:])
        -np.maximum(dest[:-1,None],source[None,:-1]))
    if not np.allclose(
        np.sum(overlap,axis=0),np.diff(source),rtol=0,atol=3e-11):
        raise ArithmeticError("axial source energy coverage not conserved")
    if not np.allclose(
        np.sum(overlap,axis=1),np.diff(dest),rtol=0,atol=3e-11):
        raise ArithmeticError("axial destination volume coverage not conserved")
    return overlap


def project_openmc_axial_fuel_heating(
    model: AxialCutCellModel,
    score: OpenMCAxialFuelHeating,
) -> ResolvedFuelHeatingInput:
    """Conservatively project *already exported* per-source heating scores.

    Uses exact z-overlap and exact Stage53 solid fuel cut-cell areas.
    Spatial detail within each source axial bin is not independently
    determined by the native OpenMC score; record this approximation.
    """
    score.validate()
    original=_edges("source",score.axial_edges_m)
    overlap=build_axial_overlap_m(original,model.axial_edges_m)
    qn=np.asarray(score.fuel_neutron_ev_per_source,dtype=float)
    qp=np.asarray(score.fuel_photon_ev_per_source,dtype=float)
    rate=score.source_rate_particles_per_s
    power_source=(qn+qp)*rate*EV_TO_J
    frac=overlap/np.diff(original)[None,:]
    power_axial=frac@power_source
    area=np.asarray(model.transverse.fuel_areas_cm2,dtype=float)
    if (area.ndim!=1 or area.shape!=(model.transverse.n_fuel,)
            or np.any(area<=0) or not np.all(np.isfinite(area))):
        raise ValueError("invalid physical cut-cell fuel-only areas")
    weights=area/float(np.sum(area))
    watts=power_axial[:,None]*weights[None,:]
    residual=float(np.sum(watts)-np.sum(power_source))
    tolerance=max(2e-12,2e-12*float(np.sum(power_source)))
    if abs(residual)>tolerance:
        raise ArithmeticError("per-source OpenMC fuel energy lost in mapping")
    if not np.all(np.isfinite(watts)):
        raise ArithmeticError("nonfinite mapped fuel heating")
    return ResolvedFuelHeatingInput(
        fuel_deposition_w=watts,
        source_axial_bin_w=power_source,
        mapped_axial_bin_w=power_axial,
        axial_overlap_m=overlap,
        transverse_fuel_weights=weights,
        fuel_neutron_power_w=float(np.sum(qn)*rate*EV_TO_J),
        fuel_photon_power_w=float(np.sum(qp)*rate*EV_TO_J),
        unresolved_nonfuel_power_w=float(
            score.unresolved_nonfuel_ev_per_source*rate*EV_TO_J),
        fuel_power_balance_residual_w=residual,
        statepoint_sha256=score.statepoint_sha256,
        nuclear_data_sha256=score.nuclear_data_sha256,
        source_rate_basis=score.source_rate_basis,
    )


def check_heating_for_coupled_core(
    fuel_heating: ResolvedFuelHeatingInput,
    model: object,
) -> np.ndarray:
    """Fail-closed shape and finite-power check for Stage58 fuel input.

    The caller must independently verify OpenMC source/geometry identity.
    This function does not make a provenance claim from a string hash.
    """
    shape=(model.nzone,model.solid.transverse.n_fuel)
    a=np.asarray(fuel_heating.fuel_deposition_w,dtype=float)
    if (a.shape!=shape or not np.all(np.isfinite(a))
            or np.any(a<0)):
        raise ValueError("heating grid does not match coupled fuel control volumes")
    if abs(float(a.sum())-float(
        np.sum(fuel_heating.mapped_axial_bin_w)))>max(
            2e-12,2e-12*float(a.sum())):
        raise ValueError("heating ledger does not match the model's fuel source")
    return a
