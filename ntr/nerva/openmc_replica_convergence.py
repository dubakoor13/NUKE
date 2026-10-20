"""Independent OpenMC heating-replica convergence and thermal consequence.

Two separately seeded *real* OpenMC statepoints are processed by NUKE's
existing heating-local normalizer, compared on an IDENTICAL voxel mesh
at the same externally imposed reactor power, and optionally propagated
through the same channel-resolved transient thermal problem.

OpenMC per-bin std_dev after normalization is treated as a DIAGNOSTIC:
source-rate normalization from the same noisy tally introduces
correlations not captured by diagonal errors. A z-score is therefore
NOT a rigorous confidence interval. Comparisons of two eigenvalue
runs may also contain source-convergence biases or common systematics.
A synthetic-only mode is explicitly labeled and cannot assert that
transport was executed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .resolved_transient import (
    ResolvedParameters, advance_resolved,
    initial_resolved_state, read_resolved_heating,
)
from .temporal_convergence import field_error


@dataclass(frozen=True)
class MeshReplica:
    directory: Path
    dimension: tuple[int, int, int]
    lower_left_cm: np.ndarray
    upper_right_cm: np.ndarray
    power_density_w_cm3: np.ndarray
    power_std_w_cm3: np.ndarray
    reactor_power_w: float
    source_kind: str
    statepoint_sha256: str | None
    nuclear_data_sha256: str | None
    openmc_seed: int | None

    @property
    def voxel_volume_cm3(self) -> float:
        return float(np.prod(
            (self.upper_right_cm-self.lower_left_cm)
            / np.array(self.dimension)
        ))


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_path(actual: str, required: Path, what: str) -> None:
    if not actual or Path(actual).resolve() != required.resolve():
        raise ValueError(f"{what} does not refer to the correct statepoint")


def _check_transport_identity(
    provenance_a: dict, provenance_b: dict
) -> None:
    """Prove that two samples have the same model, not merely different seeds."""
    if provenance_a.get("assembly") != provenance_b.get("assembly"):
        raise ValueError("independent replicas have different assembly")
    cfg_a, cfg_b = provenance_a.get("config"), provenance_b.get("config")
    if not isinstance(cfg_a, dict) or not isinstance(cfg_b, dict):
        raise ValueError("transport configuration is missing")
    for cfg in (cfg_a, cfg_b):
        if not isinstance(cfg.get("seed"), int) or cfg["seed"] <= 0:
            raise ValueError("real replica must have an explicit OpenMC seed")
    if cfg_a["seed"] == cfg_b["seed"]:
        raise ValueError("OpenMC replica seeds are identical")
    if {k:v for k,v in cfg_a.items() if k!="seed"} != {
        k:v for k,v in cfg_b.items() if k!="seed"
    }:
        raise ValueError("replica transport model parameters differ")
    libs = [p.get("nuclear_data",{}) for p in (provenance_a, provenance_b)]
    if not all(p.get("cross_sections_sha256") for p in libs):
        raise ValueError("nuclear cross sections checksum missing")
    if libs[0]["cross_sections_sha256"] != libs[1]["cross_sections_sha256"]:
        raise ValueError("different nuclear-data libraries between replicas")
    if not all(p.get("explicit_temperature_coverage_ok") is True for p in libs):
        raise ValueError("replica XS temperature coverage not validated")
    for key in ("geometry.xml","materials.xml","tallies.xml"):
        a=provenance_a.get("xml_sha256",{}).get(key)
        b=provenance_b.get("xml_sha256",{}).get(key)
        if not a or a != b:
            raise ValueError(f"OpenMC replica input XML differs: {key}")


def load_mesh_replica(
    power_directory: Path, *, provenance_path: Path | None = None,
    allow_synthetic: bool = False,
) -> tuple[MeshReplica, dict | None]:
    """Load normalized mesh fields, enforce real identity when requested."""
    directory = Path(power_directory).resolve()
    file = directory/"nerva_mesh_fields.npz"
    meta_path = directory/"metadata.json"
    if not file.is_file():
        raise FileNotFoundError(file)
    with np.load(file, allow_pickle=False) as d:
        required = {
            "dimension","lower_left_cm","upper_right_cm",
            "power_density_w_cm3","power_density_std_w_cm3"
        }
        if not required.issubset(d.files):
            raise ValueError("replica missing mesh geometry/power/std_dev arrays")
        dims = np.asarray(d["dimension"], dtype=int)
        lo = np.asarray(d["lower_left_cm"], dtype=float)
        hi = np.asarray(d["upper_right_cm"], dtype=float)
        if (dims.shape != (3,) or np.any(dims < 1)
                or lo.shape != (3,) or hi.shape != (3,)
                or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
                or np.any(hi <= lo)):
            raise ValueError("invalid normalized OpenMC mesh geometry")
        shape = tuple(int(x) for x in dims)
        mean = np.asarray(d["power_density_w_cm3"], dtype=float)
        std = np.asarray(d["power_density_std_w_cm3"], dtype=float)
        if mean.shape != shape or std.shape != shape:
            raise ValueError("normalized mesh heating mean/std shape mismatch")
        if (not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std))
                or np.any(mean < 0) or np.any(std < 0)):
            raise ValueError("nonfinite/negative power density or uncertainty")
        if float(np.sum(mean)) <= 0:
            raise ValueError("zero OpenMC heating")
        voxel = float(np.prod((hi-lo)/dims))
        total = float(np.sum(mean)*voxel)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = float(meta["normalized_power_W"])
    if not math.isfinite(expected) or expected <= 0:
        raise ValueError("invalid normalized reactor power")
    if not math.isclose(total, expected, rel_tol=2e-7):
        raise ValueError("mesh energy integral fails metadata power closure")
    if not math.isclose(expected, float(meta["requested_power_MW"])*1e6, rel_tol=2e-7):
        raise ValueError("inconsistent external power normalization")

    provenance = None
    digest: str | None = None
    xs: str | None = None
    seed: int | None = None
    kind="synthetic_unverified"
    if not allow_synthetic:
        if provenance_path is None:
            raise ValueError("real replica requires transport provenance")
        provenance = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
        sp=provenance.get("statepoint",{})
        if not isinstance(sp, dict) or not isinstance(sp.get("path"),str):
            raise ValueError("completed OpenMC statepoint missing")
        original=Path(sp["path"]).resolve()
        if not original.is_file() or original.suffix.lower()!=".h5":
            raise FileNotFoundError("real statepoint unavailable")
        _same_path(str(meta.get("statepoint","")), original,
                   "mesh-normalization metadata")
        digest = _hash(original)
        if digest != sp.get("sha256") or original.stat().st_size != sp.get("size_bytes"):
            raise ValueError("statepoint provenance hash/size inconsistent")
        config=provenance.get("config",{})
        seed = config.get("seed")
        if not isinstance(seed,int) or seed <= 0:
            raise ValueError("real comparison requires explicitly seeded transport")
        nuclear=provenance.get("nuclear_data",{})
        xs=nuclear.get("cross_sections_sha256")
        if not xs or nuclear.get("explicit_temperature_coverage_ok") is not True:
            raise ValueError("missing nuclear XS audit")
        kind="verified_openmc_file_provenance"
    elif provenance_path is not None:
        raise ValueError("synthetic mode cannot claim real provenance")

    return MeshReplica(directory,shape,lo,hi,mean,std,total,kind,digest,xs,seed),provenance


def compare_mesh_replicas(
    first: MeshReplica, second: MeshReplica, *,
    max_total_variation: float = 0.05,
    max_axial_fraction_difference: float = 0.05,
    max_significant_fraction: float = 0.10,
    significance_sigma: float = 3.,
) -> dict:
    """Compare independent normalized heating distributions at same power.

    Total variation = 0.5 * sum_voxels |P1/Ptot - P2/Ptot|.
    This is dimensionless and zero for identical spatial power shapes.
    The z-score is a diagnostic using a diagonal approximation of std.
    """
    for name,value in (
        ("total variation limit",max_total_variation),
        ("axial fraction limit",max_axial_fraction_difference),
        ("significant fraction limit",max_significant_fraction),
    ):
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid {name}")
    if not math.isfinite(significance_sigma) or significance_sigma <= 0:
        raise ValueError("invalid significance_sigma")
    if first.dimension != second.dimension:
        raise ValueError("different OpenMC mesh dimensions")
    if (not np.allclose(first.lower_left_cm,second.lower_left_cm,rtol=0,atol=1e-12)
            or not np.allclose(first.upper_right_cm,second.upper_right_cm,
                               rtol=0,atol=1e-12)):
        raise ValueError("different OpenMC mesh spatial bounds")
    if not math.isclose(first.reactor_power_w,second.reactor_power_w,
                        rel_tol=1e-8):
        raise ValueError("different imposed reactor power")
    if first.source_kind!=second.source_kind:
        raise ValueError("cannot mix verified OpenMC and synthetic fixtures")
    if first.statepoint_sha256 is not None:
        if first.statepoint_sha256 == second.statepoint_sha256:
            raise ValueError("replica statepoint files are identical")
        if first.nuclear_data_sha256!=second.nuclear_data_sha256:
            raise ValueError("replica nuclear XS library changed")
        if first.openmc_seed==second.openmc_seed:
            raise ValueError("replica OpenMC seeds are identical")

    voxel=first.voxel_volume_cm3
    p1=first.power_density_w_cm3*voxel/first.reactor_power_w
    p2=second.power_density_w_cm3*voxel/second.reactor_power_w
    tv=0.5*float(np.sum(np.abs(p1-p2)))
    z1=p1.sum(axis=(0,1))
    z2=p2.sum(axis=(0,1))
    axial=float(np.max(np.abs(z1-z2)))
    mean1,mean2=first.power_density_w_cm3,second.power_density_w_cm3
    peak=max(float(mean1.max()),float(mean2.max()))
    active=np.maximum(mean1,mean2)>peak*1e-8
    denominator=np.hypot(first.power_std_w_cm3,second.power_std_w_cm3)
    delta=np.abs(mean1-mean2)
    # If a tally reports zero std but a nonzero difference, discrepancy
    # is significant instead of being hidden by a zero denominator.
    sig=np.where(
        denominator>0,delta/np.maximum(denominator,np.finfo(float).tiny),
        np.where(delta>0,np.inf,0.)
    )
    fraction=float(np.mean(sig[active]>significance_sigma))
    maximum_z=float(np.max(sig[active]))
    a_rel=first.power_std_w_cm3[active]/np.maximum(mean1[active],peak*1e-8)
    b_rel=second.power_std_w_cm3[active]/np.maximum(mean2[active],peak*1e-8)
    result={
        "source_kind":first.source_kind,
        "independent_real_statepoints": first.statepoint_sha256 is not None,
        "mesh_shape":list(first.dimension),
        "reactor_power_w":first.reactor_power_w,
        "active_mesh_voxels":int(np.count_nonzero(active)),
        "total_variation_power_fraction":tv,
        "max_axial_power_fraction_difference":axial,
        "fraction_active_voxels_above_sigma":fraction,
        "significance_sigma":significance_sigma,
        "max_approximate_zscore":maximum_z,
        "replica_1_p95_local_relative_std":float(np.percentile(a_rel,95)),
        "replica_2_p95_local_relative_std":float(np.percentile(b_rel,95)),
        "criteria":{
            "total_variation_at_most":max_total_variation,
            "axial_fraction_at_most":max_axial_fraction_difference,
            "active_fraction_above_sigma_at_most":max_significant_fraction,
        },
        "pass": (
            tv<=max_total_variation
            and axial<=max_axial_fraction_difference
            and fraction<=max_significant_fraction
        ),
        "uncertainty_interpretation":(
            "Per-bin z-scores are approximate diagnostics: independent "
            "normalized replicas share a noisy total-power normalization "
            "within each sample, so diagonal std ignores covariance. "
            "No hypothesis-test p-value or experimental validation."
        ),
    }
    return result


def thermal_replica_consequence(
    first_mesh: Path, first_fuel: Path, first_tie: Path,
    second_mesh: Path, second_fuel: Path, second_tie: Path,
    case: dict,
) -> dict:
    """Run identical resolved thermal problems on TWO heating realizations.

    Source reconstructions should be generated independently from their
    own transport runs. Structural parameters and flow are held constant
    so the difference isolates the changed spatial heating distribution.
    """
    def required(name: str):
        if name not in case:
            raise ValueError(f"thermal case missing {name}")
        return case[name]
    args={}
    for f in fields(ResolvedParameters):
        if f.name in case:
            args[f.name]=case[f.name]
    params=ResolvedParameters(**args)
    params.validate()
    steps=required("steps")
    dt=float(required("dt_s"))
    t_s=float(required("initial_solid_k"))
    t_g=float(required("initial_gas_k"))
    inlet=float(required("inlet_temperature_k"))
    threshold=float(required("max_temperature_difference_k"))
    if (not isinstance(steps,int) or steps<=0
            or not math.isfinite(dt) or dt<=0
            or not math.isfinite(threshold) or threshold<=0):
        raise ValueError("invalid thermal replica time or tolerance")
    powers=[
        read_resolved_heating(first_fuel,first_tie,first_mesh),
        read_resolved_heating(second_fuel,second_tie,second_mesh),
    ]
    a,b=powers
    if (a.fuel_solid_w.shape!=b.fuel_solid_w.shape
            or a.tie_solid_w.shape!=b.tie_solid_w.shape
            or not np.allclose(a.fuel_flow_kg_s,b.fuel_flow_kg_s,rtol=1e-12)
            or not np.allclose(a.tie_flow_kg_s,b.tie_flow_kg_s,rtol=1e-12)):
        raise ValueError("replicas have incompatible channel/tie geometry or flows")
    state0=initial_resolved_state(
        a,solid_temperature_k=t_s,gas_temperature_k=t_g
    )
    outcomes=[]
    max_energy_residual=[]
    for h in powers:
        s=state0
        max_res=0.
        for _ in range(steps):
            out=advance_resolved(h,s,params,inlet,dt)
            s=out.state
            max_res=max(max_res,abs(out.energy_residual_j))
        outcomes.append(s)
        max_energy_residual.append(max_res)
    errors=field_error(outcomes[0],outcomes[1])
    maximum=max(errors.values())
    fuel_exit=max(abs(outcomes[0].fuel_h2_t_k[:,:,-1]-
                      outcomes[1].fuel_h2_t_k[:,:,-1]).ravel())
    tie_exit=max(abs(outcomes[0].tie_return_t_k[:,0]-
                     outcomes[1].tie_return_t_k[:,0]).ravel())
    return {
        "status": "PASS" if maximum<=threshold else "FAIL",
        "steps":steps,
        "dt_s":dt,
        "physical_end_time_s":steps*dt,
        "temperature_difference_by_field_k":errors,
        "max_resolved_temperature_difference_k":maximum,
        "max_fuel_outlet_difference_k":float(fuel_exit),
        "max_tie_return_outlet_difference_k":float(tie_exit),
        "maximum_energy_residual_by_run_j":max_energy_residual,
        "max_allowed_temperature_difference_k":threshold,
        "historical_nerva_validation":False,
        "interpretation":"Thermal consequences of two fixed power maps, not time-dependent neutron transport.",
    }
