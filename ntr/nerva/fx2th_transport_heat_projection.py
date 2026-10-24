"""Conservative transport-heating tally -> spatial thermal-cell power projection.

This is a numerical bridge, NOT a statepoint reader or an OpenMC transport run.
Inputs are externally extracted, mutually compatible energy-deposition arrays:
    tally mesh shape [x,y,z], values eV/source history/mesh bin.
No division by mesh bin volume: the scores are INTEGRATED BIN ENERGIES, not
energy densities. Neutron/photon estimators and particle normalization must
be identical for total, solid and hydrogen partitions.

Power requires ONE explicit external normalization:
  fixed_source_rate: histories/s * score [eV/history] * eV_to_J
  prescribed_power: W * (bin score)/(sum of all bin scores)
Never infer source rate, source strength, wattage or power from k_eff or an
arbitrarily normalized diffusion eigenflux.

The spatial overlap operator integrates x,y completely and redistributes z
by exact shared interval length; all mesh volume and power conservation
checks are explicit. Solid and hydrogen are disjoint source compartments;
everything else is retained as unmodeled deposition, never silently lost.

Statepoint/data hashes are DECLARED external provenance only; this module
does NOT establish their authenticity, run OpenMC, evaluate cross sections
or qualify actual NERVA fuel/reflector geometry or temperatures.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Callable

import numpy as np

from .fx2th_thermal_energy import ThermalDeposition
from .fx2th_family_spacetime import SpatialFamilyData, SpatialFamilyState

EV_TO_J = 1.602176634e-19   # SI exact definition
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _valid_hash(name: str, value: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def material_fingerprint(material: SpatialFamilyData) -> str:
    """Stable hash of the actual discrete nuclear material/operator inputs.

    A caller must still establish that its transport geometry, temperature,
    nuclear data and normalization match these *effective diffusion inputs*.
    This hash cannot certify physical equivalence of two solvers.
    """
    if not isinstance(material, SpatialFamilyData):
        raise TypeError("material must be SpatialFamilyData")
    h = hashlib.sha256()
    h.update(b"NUKE-FX2TH-SPATIAL-FAMILY-V1\x00")
    h.update(material.source_id.encode("utf-8"))
    for a in (
        material.volumes_m3, material.group_speeds_m_s,
        material.nu_sigma_f_inv_m,
        material.beta_by_incident_group_and_family,
        material.prompt_spectrum, material.delayed_spectra_by_family,
        material.decay_constants_s_inv,
    ):
        x = np.ascontiguousarray(np.asarray(a, dtype="<f8"))
        h.update(np.asarray(x.shape,dtype="<i8").tobytes())
        h.update(x.tobytes())
    # Fix sparse storage representation (CSR canonical indices, sorted).
    csr = material.integrated_loss.tocsr(copy=True)
    csr.sum_duplicates()
    csr.sort_indices()
    for a in (csr.indptr.astype("<i8"), csr.indices.astype("<i8"),
              csr.data.astype("<f8")):
        h.update(np.asarray(a.shape,dtype="<i8").tobytes())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class HeatingMeshScore:
    source_id: str
    source_kind: str          # synthetic / extracted-openmc-arrays
    material_fingerprint_sha256: str
    statepoint_sha256: str
    nuclear_data_sha256: str
    tally_score: str          # "heating" only
    score_units: str          # "eV/source-history/mesh-bin" only
    lower_left_cm: np.ndarray   # xyz; arrays [nx,ny,nz]
    upper_right_cm: np.ndarray
    total_ev_per_history_bin: np.ndarray
    solid_ev_per_history_bin: np.ndarray
    hydrogen_ev_per_history_bin: np.ndarray

    def validate(self) -> tuple[int, int, int]:
        if not isinstance(self.source_id,str) or not self.source_id.strip():
            raise ValueError("nonblank source_id is required")
        if self.source_kind not in ("synthetic", "extracted-openmc-arrays"):
            raise ValueError("source_kind must state synthetic or extracted-openmc-arrays")
        for name in ("material_fingerprint_sha256", "statepoint_sha256",
                     "nuclear_data_sha256"):
            _valid_hash(name,getattr(self,name))
        if self.tally_score != "heating":
            raise ValueError("only a declared heating tally is supported")
        if self.score_units != "eV/source-history/mesh-bin":
            raise ValueError("score must be integrated eV per source history per mesh bin")
        lo=np.asarray(self.lower_left_cm,dtype=float)
        hi=np.asarray(self.upper_right_cm,dtype=float)
        if (lo.shape != (3,) or hi.shape != (3,) or
                not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or
                np.any(hi<=lo)):
            raise ValueError("finite increasing xyz mesh bounds required")
        total=np.asarray(self.total_ev_per_history_bin,dtype=float)
        if total.ndim != 3 or min(total.shape)<1:
            raise ValueError("total heating must be nonempty [nx,ny,nz]")
        for name in ("total_ev_per_history_bin","solid_ev_per_history_bin",
                     "hydrogen_ev_per_history_bin"):
            a=np.asarray(getattr(self,name),dtype=float)
            if a.shape!=total.shape or not np.all(np.isfinite(a)) or np.any(a<0):
                raise ValueError(f"{name} must match mesh and be nonnegative finite")
        if not float(total.sum())>0:
            raise ValueError("total heating score must be positive")
        # Never allow one region to compensate another region's overcount.
        if np.any(np.asarray(self.solid_ev_per_history_bin)+
                  np.asarray(self.hydrogen_ev_per_history_bin)>total):
            raise ValueError("disjoint material heating overcounts a source mesh bin")
        return tuple(int(i) for i in total.shape)


@dataclass(frozen=True)
class HeatingNormalization:
    mode: str         # fixed-source-rate / externally-prescribed-power
    value: float      # source histories/s or externally prescribed W

    def factor_w_per_ev_history(self, integrated_total_ev: float) -> float:
        if (not math.isfinite(self.value) or self.value<=0 or
                not math.isfinite(integrated_total_ev) or integrated_total_ev<=0):
            raise ValueError("positive finite source normalization and heating required")
        if self.mode=="fixed-source-rate":
            return self.value*EV_TO_J
        if self.mode=="externally-prescribed-power":
            return self.value/integrated_total_ev
        raise ValueError("unsupported normalization: never infer reactor watts from flux")


@dataclass(frozen=True)
class ProjectionResult:
    deposition: ThermalDeposition
    mesh_total_power_w: float
    solid_total_power_w: float
    hydrogen_total_power_w: float
    unmodeled_total_power_w: float
    source_mode: str
    source_kind: str
    global_power_residual_w: float
    axial_overlap_weights: np.ndarray  # [mesh_z,thermal_z] weights sum=1 per mesh bin


def _overlap_weights(mesh: HeatingMeshScore,
                     thermal_z_edges_cm: np.ndarray) -> np.ndarray:
    bounds=np.asarray(mesh.lower_left_cm,dtype=float)
    upper=np.asarray(mesh.upper_right_cm,dtype=float)
    edges=np.asarray(thermal_z_edges_cm,dtype=float)
    nz=int(mesh.total_ev_per_history_bin.shape[2])
    if (edges.ndim!=1 or len(edges)<2 or not np.all(np.isfinite(edges))
            or np.any(np.diff(edges)<=0)):
        raise ValueError("thermal axial edges must be increasing finite values")
    tol=2e-12*max(1.,abs(bounds[2]),abs(upper[2]))
    if abs(edges[0]-bounds[2])>tol or abs(edges[-1]-upper[2])>tol:
        raise ValueError("thermal axial cells must fully cover source-mesh z extent")
    z=np.linspace(bounds[2],upper[2],nz+1)
    overlap=np.maximum(0.0, np.minimum(z[1:,None],edges[None,1:])
                            -np.maximum(z[:-1,None],edges[None,:-1]))
    weights=overlap/np.diff(z)[:,None]
    if not np.allclose(weights.sum(axis=1),1.,rtol=0,atol=5e-13):
        raise ArithmeticError("axial overlap does not conservatively partition source bins")
    return weights


def project_mesh_heating(
    score: HeatingMeshScore, normalization: HeatingNormalization,
    thermal_z_edges_cm: np.ndarray,
) -> ProjectionResult:
    """Project integrated per-history mesh scores to 1D thermal cell powers.

    No clipping, arbitrary power normalization, default material temperature,
    or declaration of physical qualification occurs. Fields MUST be comparable
    tallies on exactly the same three-dimensional mesh.
    """
    score.validate()
    zweights=_overlap_weights(score,thermal_z_edges_cm)
    total=np.asarray(score.total_ev_per_history_bin,dtype=float)
    solid=np.asarray(score.solid_ev_per_history_bin,dtype=float)
    hydrogen=np.asarray(score.hydrogen_ev_per_history_bin,dtype=float)
    other=total-solid-hydrogen
    if np.any(other<0):
        raise ValueError("unmodeled mesh heating cannot be negative")
    factor=normalization.factor_w_per_ev_history(float(total.sum()))

    def project(a: np.ndarray) -> np.ndarray:
        return (a.sum(axis=(0,1)) @ zweights)*factor

    solid_w=project(solid)
    hydrogen_w=project(hydrogen)
    other_w=project(other)
    mapped_total=float(np.sum(solid_w+hydrogen_w+other_w))
    expected=float(total.sum())*factor
    residual=mapped_total-expected
    if abs(residual)>1e-12*max(expected,1.):
        raise ArithmeticError("projected thermal heating violates global conservation")
    # The Stage 8 interface consumes exactly these disjoint channels;
    # source_kind controls synthetic status; SHA strings alone do not
    # establish historical accuracy.
    dep=ThermalDeposition(
        source_id=score.source_id,
        solid_power_w=solid_w,hydrogen_power_w=hydrogen_w,
        other_power_w=other_w,declared_total_w=expected,
        synthetic=(score.source_kind=="synthetic"),
    )
    dep.check(len(solid_w))
    return ProjectionResult(
        deposition=dep,mesh_total_power_w=expected,
        solid_total_power_w=float(solid_w.sum()),
        hydrogen_total_power_w=float(hydrogen_w.sum()),
        unmodeled_total_power_w=float(other_w.sum()),
        source_mode=normalization.mode,
        source_kind=score.source_kind,
        global_power_residual_w=residual,axial_overlap_weights=zweights
    )


def make_checked_deposition_provider(
    score_provider: Callable[[SpatialFamilyState,SpatialFamilyData],
                             HeatingMeshScore],
    normalization: HeatingNormalization,
    thermal_z_edges_cm: np.ndarray,
) -> Callable[[SpatialFamilyState,SpatialFamilyData],ThermalDeposition]:
    """Opt-in Stage 8 callback, requiring a material-matched transport score.

    Each call obtains a score from the external provider. Reject a stale
    transport snapshot whose declared material fingerprint does not match
    the current Stage 7 nuclear material/operator coefficients.
    This is a consistency check, NOT proof that the provider ran OpenMC.
    """
    if not callable(score_provider):
        raise TypeError("explicit score provider is required")
    edges=np.asarray(thermal_z_edges_cm,dtype=float)
    def callback(state: SpatialFamilyState,data: SpatialFamilyData) -> ThermalDeposition:
        score=score_provider(state,data)
        if not isinstance(score,HeatingMeshScore):
            raise TypeError("provider must return HeatingMeshScore")
        score.validate()
        if score.material_fingerprint_sha256!=material_fingerprint(data):
            raise ValueError("stale transport heating: material fingerprint does not match")
        result=project_mesh_heating(score,normalization,edges)
        if len(result.deposition.solid_power_w)!=data.cells:
            raise ValueError("transport thermal bins do not match Stage 7 cell count")
        return result.deposition
    return callback
