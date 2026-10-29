"""Stage68: independent full-core deposited-energy conservation audit.

OpenMC's existing global 'nerva_particle_heating' tally explicitly resolves
NEUTRON, PHOTON, ELECTRON and POSITRON 'heating' score contributions. Energy
stored by electrons/positrons must not disappear when comparing this score
with the independent 'nerva_material_transport' material-heating tally.

Both tallies cover the WHOLE reactor. This is NOT a single-element nuclear
power normalization and MUST NOT be injected into one Stage58 fuel element.

This audit checks independent complete-material vs complete-particle means,
uses native sum/sum_sq realization statistics without inventing covariance,
and separately reports energy in fuel, H2 and nonthermal structures. It
refuses missing charged species, missing materials, malformed source modes,
and statistically inconsistent totals. A passing HDF5 fixture alone does
NOT certify real OpenMC transport, the cross section library, the physical
power source rate, or nuclear/thermal solution convergence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Mapping

import h5py
import numpy as np

from .openmc_native_fuel_statepoint import (
    _sha256_file, _scalar_string, _metadata_list,
)
from .openmc_fuel_axial_heat_handoff import EV_TO_J

MATERIAL_ORDER = (
    "fuel", "hydrogen", "zrc", "reflector", "graphite",
    "zrh", "inconel", "b4c", "aluminum",
)
PARTICLE_ORDER = ("neutron", "photon", "electron", "positron")
TALLY_MATERIAL = "nerva_material_transport"
TALLY_PARTICLE = "nerva_particle_heating"


@dataclass(frozen=True)
class CoreDepositedEnergyLedger:
    """Per-source means, conservative stderr and full-core watts.

    A whole-core material ledger is never an input to ONE resolved
    fuel element; instance-level normalization is still outstanding.
    """
    material_ev_per_source: dict[str, float]
    particle_ev_per_source: dict[str, float]
    material_std_error_ev_per_source: dict[str, float]
    particle_std_error_ev_per_source: dict[str, float]
    material_power_w: dict[str, float]
    particle_power_w: dict[str, float]
    fuel_power_w: float
    hydrogen_power_w: float
    nonthermal_material_power_w: float
    charged_secondary_power_w: float
    material_total_w: float
    particle_total_w: float
    material_minus_particle_w: float
    conservative_difference_stderr_upper_bound_w: float
    statistical_acceptance_limit_w: float
    statepoint_sha256: str
    source_rate_basis: str
    nuclear_data_sha256: str
    n_realizations: int
    active_batches_at_statepoint: int
    material_coverage_status: str = (
        "whole-core MATERIAL_ORDER explicitly required; fuel/H2 and "
        "nonthermal structures are not a per-element source normalization"
    )


def _native_tally(tallies: h5py.Group, ids: set[int], name: str) -> h5py.Group:
    matched=[v for k,v in tallies.items()
             if k.startswith("tally ") and isinstance(v,h5py.Group)
             and "name" in v and _scalar_string(v["name"][()])==name]
    if len(matched)!=1:
        raise ValueError(f"exactly one {name} tally is required")
    found=matched[0]
    if int(found.name.rsplit(" ",1)[-1]) not in ids:
        raise ValueError(f"{name}: tally not registered in OpenMC manifest")
    return found


def _moments(
    tally: h5py.Group,
    *,
    scores: list[str],
    expected_bins: int,
    count_limit: int,
) -> tuple[np.ndarray,np.ndarray,int]:
    if "n_realizations" not in tally or "results" not in tally:
        raise ValueError("missing native tally realizations and moments")
    n=int(tally["n_realizations"][()])
    if n<2 or n>count_limit:
        raise ValueError("incompatible active tally realization count")
    x=np.asarray(tally["results"][()],dtype=float)
    if (x.shape!=(expected_bins,len(scores),2)
            or not np.all(np.isfinite(x))):
        raise ValueError("native OpenMC energy tally moments shape invalid")
    i=scores.index("heating")
    mean=x[:,i,0]/n
    var=(x[:,i,1]/n-mean**2)/(n-1)
    if np.any(var < -1e-10*np.maximum(1.,mean**2)):
        raise ValueError("negative variance in native deposited energy tally")
    if np.any(mean < -1e-14):
        raise ValueError("negative deposited-energy mean in native tally")
    return np.maximum(0.,mean),np.sqrt(np.maximum(0.,var)),n


def _tally_score_filter(
    tallies: h5py.Group,
    tally: h5py.Group,
    *,
    kind: str,
) -> tuple[list[str],np.ndarray]:
    if (not all(k in tally for k in ("filters","score_bins","nuclides"))
            or _metadata_list(tally["nuclides"][()])!=["total"]):
        raise ValueError("energy tally missing filter/score or total nuclide")
    scores=_metadata_list(tally["score_bins"][()])
    if scores.count("heating")!=1:
        raise ValueError("whole-core photon-mode audit needs exactly one heating score")
    filter_ids=np.asarray(tally["filters"][()],dtype=int).ravel()
    if filter_ids.shape!=(1,):
        raise ValueError("whole-core audit requires exactly one tally filter")
    k=f"filters/filter {int(filter_ids[0])}"
    if k not in tallies:
        raise ValueError("whole-core tally filter identity missing")
    f=tallies[k]
    if "type" not in f or "bins" not in f or _scalar_string(f["type"][()]).lower()!=kind:
        raise ValueError(f"expected independent {kind} filter")
    return scores,np.asarray(f["bins"][()])


def read_openmc_full_core_energy_ledger(
    statepoint_path: str | Path,
    *,
    expected_material_ids: Mapping[str,int],
    source_rate_particles_per_s: float,
    source_rate_basis: str,
    nuclear_data_sha256: str,
    sigma_multiplier: float=5.,
    absolute_tolerance_ev_per_source: float=1e-8,
    expected_statepoint_sha256: str | None=None,
) -> CoreDepositedEnergyLedger:
    """Independently reconcile 9-material HEATING and 4-particle HEATING.

    A statistical acceptance check is a configurable guard, NOT physical
    reactor validation or a proof of a chosen confidence level. Correlations
    of the deposited-energy tallies are unknown, so the sum of marginal
    standard errors bounds the standard deviation of their difference.
    """
    ids=dict(expected_material_ids)
    if (set(ids)!=set(MATERIAL_ORDER)
            or any(not isinstance(v,int) or v<=0 for v in ids.values())
            or len(set(ids.values()))!=len(MATERIAL_ORDER)):
        raise ValueError("full physical material-ID manifest required")
    if (not isinstance(nuclear_data_sha256,str)
            or re.fullmatch("[0-9a-fA-F]{64}",nuclear_data_sha256) is None):
        raise ValueError("externally recorded nuclear data SHA256 required")
    if (not math.isfinite(source_rate_particles_per_s)
            or source_rate_particles_per_s<=0
            or not isinstance(source_rate_basis,str) or not source_rate_basis.strip()):
        raise ValueError("positive, externally justified source particle rate required")
    if (not math.isfinite(sigma_multiplier) or sigma_multiplier<=0
            or not math.isfinite(absolute_tolerance_ev_per_source)
            or absolute_tolerance_ev_per_source<0):
        raise ValueError("invalid statistical closure tolerance")
    path=Path(statepoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest=_sha256_file(path)
    if (expected_statepoint_sha256 is not None
            and digest.lower()!=expected_statepoint_sha256.lower()):
        raise ValueError("full-core energy statepoint SHA256 mismatch")

    with h5py.File(path,"r") as h:
        v=np.asarray(h.attrs.get("version",[]))
        photon=np.asarray(h.attrs.get("photon_transport",-1))
        if (h.attrs.get("filetype",None) is None
                or _scalar_string(h.attrs["filetype"])!="statepoint"
                or v.shape!=(2,) or int(v[0])!=18 or int(v[1])<0):
            raise ValueError("invalid native OpenMC statepoint schema")
        if (photon.ndim or not np.issubdtype(photon.dtype,np.integer)
                or int(photon)!=1):
            raise ValueError("full four-particle HEATING ledger requires photon transport")
        if ("energy_mode" not in h
                or _scalar_string(h["energy_mode"][()])!="continuous-energy"
                or "run_mode" not in h
                or _scalar_string(h["run_mode"][()])!="eigenvalue"):
            raise ValueError("continuous-energy eigenvalue transport required")
        needed=("n_batches","n_inactive","current_batch","n_particles")
        if any(k not in h for k in needed):
            raise ValueError("native active-batch metadata required")
        nb,ni,current,npop=(int(h[k][()]) for k in needed)
        if nb<2 or not 0<=ni<current<=nb or npop<=0:
            raise ValueError("invalid native source generation count")
        if ("tallies_present" not in h.attrs
                or int(h.attrs["tallies_present"])!=1
                or "tallies" not in h):
            raise ValueError("native deposited-energy tallies not available")
        tg=h["tallies"]
        if "ids" not in tg.attrs or "n_tallies" not in tg.attrs:
            raise ValueError("native tally IDs absent")
        registered=np.asarray(tg.attrs["ids"],dtype=int).ravel()
        if (len(registered)!=int(tg.attrs["n_tallies"])
                or len(set(registered.tolist()))!=len(registered)):
            raise ValueError("native tally manifest inconsistent")
        material_tally=_native_tally(tg,set(registered.tolist()),TALLY_MATERIAL)
        particle_tally=_native_tally(tg,set(registered.tolist()),TALLY_PARTICLE)
        ms,mb=_tally_score_filter(tg,material_tally,kind="material")
        ps,pb=_tally_score_filter(tg,particle_tally,kind="particle")
        material_bins=np.asarray(mb,dtype=int).reshape(-1)
        if (len(material_bins)!=len(MATERIAL_ORDER)
                or set(material_bins.tolist())!=set(ids.values())):
            raise ValueError("whole-core heating material IDs do not cover expected materials")
        particle_bins=_metadata_list(pb)
        if (len(particle_bins)!=4
                or set(particle_bins)!=set(PARTICLE_ORDER)):
            raise ValueError("charged electron/positron energy bins must be present")
        active=current-ni
        mat,mat_stderr,n_mat=_moments(
            material_tally,scores=ms,
            expected_bins=len(material_bins),count_limit=active)
        part,part_stderr,n_part=_moments(
            particle_tally,scores=ps,
            expected_bins=4,count_limit=active)
        if n_mat!=n_part:
            raise ValueError("material and particle energy tallies have different realizations")
    m_by_id={int(material_bins[i]):(float(mat[i]),float(mat_stderr[i]))
             for i in range(len(material_bins))}
    mats={name:m_by_id[ids[name]][0] for name in MATERIAL_ORDER}
    materr={name:m_by_id[ids[name]][1] for name in MATERIAL_ORDER}
    parts={name:float(part[particle_bins.index(name)]) for name in PARTICLE_ORDER}
    parterr={name:float(part_stderr[particle_bins.index(name)])
             for name in PARTICLE_ORDER}
    e_mat=float(sum(mats.values()))
    e_part=float(sum(parts.values()))
    stderr_bound=float(sum(materr.values())+sum(parterr.values()))
    acceptance=max(absolute_tolerance_ev_per_source,sigma_multiplier*stderr_bound)
    if abs(e_mat-e_part)>acceptance:
        raise ArithmeticError(
            f"OpenMC complete material/particle heating does not close: "
            f"delta={e_mat-e_part:.9g} eV/source, "
            f"allowance={acceptance:.9g} eV/source")
    rate=source_rate_particles_per_s*EV_TO_J
    watts={k:v*rate for k,v in mats.items()}
    p_watts={k:v*rate for k,v in parts.items()}
    return CoreDepositedEnergyLedger(
        material_ev_per_source=mats,
        particle_ev_per_source=parts,
        material_std_error_ev_per_source=materr,
        particle_std_error_ev_per_source=parterr,
        material_power_w=watts,
        particle_power_w=p_watts,
        fuel_power_w=watts["fuel"],
        hydrogen_power_w=watts["hydrogen"],
        nonthermal_material_power_w=sum(
            watts[name] for name in MATERIAL_ORDER if name not in ("fuel","hydrogen")),
        charged_secondary_power_w=p_watts["electron"]+p_watts["positron"],
        material_total_w=e_mat*rate,
        particle_total_w=e_part*rate,
        material_minus_particle_w=(e_mat-e_part)*rate,
        conservative_difference_stderr_upper_bound_w=stderr_bound*rate,
        statistical_acceptance_limit_w=acceptance*rate,
        statepoint_sha256=digest,source_rate_basis=source_rate_basis,
        nuclear_data_sha256=nuclear_data_sha256,
        n_realizations=n_mat,active_batches_at_statepoint=active,
    )
