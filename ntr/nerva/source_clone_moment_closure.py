"""Physical first/second moment closure conditional on exact cloned fission sites.

For ACTUAL OpenMC eigenvalue source banks, separate variation of emitted
neutron DIRECTIONS and ENERGIES into within-exact-position and
between-exact-position components. This is an exact weighted empirical
law-of-total-covariance identity, NOT an independent-fission-history
sample-size estimate, an anisotropy significance test, or a reactor tune.

No new source particles, material values, geometry, nuclide evaluations,
power, reactor settings or criticality optimization are introduced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .source_direction_physics_audit import read_real_bank_angular_physics


def _sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with Path(file).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def partition_clone_moments(xyz: np.ndarray, directions: np.ndarray,
                            energy: np.ndarray, weight: np.ndarray) -> dict:
    """Empirical weighted total = within-source-location + between-location.

    Every source record is retained with its original *physical* weight.
    Multiple outgoing energies or directions at identical locations never
    increase the number of separate source *positions*. Weighted clone
    populations are not independent neutron genealogies.
    """
    xyz=np.asarray(xyz, dtype=np.float64)
    u=np.asarray(directions, dtype=np.float64)
    e=np.asarray(energy, dtype=np.float64)
    w=np.asarray(weight, dtype=np.float64)
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or u.shape != xyz.shape
            or e.shape != (len(xyz),) or w.shape != (len(xyz),)
            or len(xyz) < 2 or not np.isfinite(xyz).all()
            or not np.isfinite(u).all() or not np.isfinite(e).all()
            or not np.isfinite(w).all() or np.any(e <= 0)
            or np.any(w <= 0)):
        raise ValueError("invalid actual weighted neutron source position, direction or energy")
    if np.max(np.abs(np.linalg.norm(u,axis=1)-1.)) > 1e-8:
        raise ValueError("non-unit neutron emission direction")
    total=float(np.sum(w,dtype=np.float64))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("invalid actual total neutron source weight")

    _, inverse, counts = np.unique(
        xyz,axis=0,return_inverse=True,return_counts=True
    )
    site_weights = np.bincount(inverse,weights=w,minlength=len(counts))
    probabilities = site_weights/total
    if (not np.isfinite(probabilities).all()
            or np.any(probabilities <= 0)
            or abs(float(np.sum(probabilities))-1.)>1e-12):
        raise ValueError("invalid exact fission-site weight probabilities")
    group_mean_u=np.column_stack([
        np.bincount(inverse,weights=w*u[:,j],minlength=len(counts))/site_weights
        for j in range(3)
    ])
    group_mean_e=np.bincount(
        inverse,weights=w*e,minlength=len(counts)
    )/site_weights
    mean_u=np.sum(u*w[:,None],axis=0,dtype=np.float64)/total
    mean_e=float(np.dot(w,e)/total)
    uc=u-mean_u
    ec=e-mean_e
    all_u_cov=np.einsum("ni,nj,n->ij",uc,uc,w,optimize=True)/total
    all_e_var=float(np.dot(w,ec*ec)/total)
    all_eu_cov=(w*ec)@uc/total

    u_within=u-group_mean_u[inverse]
    e_within=e-group_mean_e[inverse]
    within_u_cov=np.einsum(
        "ni,nj,n->ij",u_within,u_within,w,optimize=True
    )/total
    within_e_var=float(np.dot(w,e_within*e_within)/total)
    within_eu_cov=(w*e_within)@u_within/total
    centered_u_site=group_mean_u-mean_u
    centered_e_site=group_mean_e-mean_e
    between_u_cov=np.einsum(
        "ni,nj,n->ij",centered_u_site,centered_u_site,probabilities,optimize=True
    )
    between_e_var=float(np.dot(probabilities,centered_e_site**2))
    between_eu_cov=(probabilities*centered_e_site)@centered_u_site

    u_resid=float(np.linalg.norm(all_u_cov-within_u_cov-between_u_cov))
    u_scale=max(1.,float(np.linalg.norm(all_u_cov)))
    e_resid=abs(all_e_var-within_e_var-between_e_var)
    e_scale=max(1.,all_e_var)
    eu_resid=float(np.linalg.norm(all_eu_cov-within_eu_cov-between_eu_cov))
    eu_scale=max(1.,float(np.linalg.norm(all_eu_cov)))
    if (u_resid/u_scale > 5e-11 or e_resid/e_scale > 5e-11
            or eu_resid/eu_scale > 5e-11):
        raise ValueError("weighted fission-source clone total-covariance identity failed")
    for matrix in (all_u_cov,within_u_cov,between_u_cov):
        if (not np.isfinite(matrix).all()
                or min(np.linalg.eigvalsh(matrix)) < -5e-11):
            raise ValueError("nonphysical fission-source direction covariance matrix")
    if all_e_var < 0 or within_e_var < 0 or between_e_var < 0:
        raise ValueError("negative emitted-neutron energy variance")
    d_eff=float(1/np.dot(probabilities,probabilities))
    if not (0 < d_eff <= len(counts)+1e-8):
        raise ValueError("invalid effective distinct source positions")

    return {
        "nominal_source_particles": int(len(xyz)),
        "exact_distinct_fission_source_positions": int(len(counts)),
        "inverse_simpson_effective_source_positions": d_eff,
        "maximum_exact_position_particle_multiplicity": int(counts.max()),
        "mean_source_direction": mean_u.tolist(),
        "mean_emission_energy_eV": mean_e,
        "weighted_total_direction_covariance": all_u_cov.tolist(),
        "weighted_within_exact_position_direction_covariance": within_u_cov.tolist(),
        "weighted_between_exact_position_direction_covariance": between_u_cov.tolist(),
        "weighted_total_emission_energy_variance_eV2": all_e_var,
        "weighted_within_exact_position_energy_variance_eV2": within_e_var,
        "weighted_between_exact_position_energy_variance_eV2": between_e_var,
        "weighted_total_energy_direction_covariance_eV": all_eu_cov.tolist(),
        "weighted_within_exact_position_energy_direction_covariance_eV": within_eu_cov.tolist(),
        "weighted_between_exact_position_energy_direction_covariance_eV": between_eu_cov.tolist(),
        "fraction_total_direction_variance_within_exact_position": float(
            np.trace(within_u_cov)/np.trace(all_u_cov)
        ) if np.trace(all_u_cov)>0 else None,
        "fraction_total_energy_variance_within_exact_position": (
            within_e_var/all_e_var if all_e_var>0 else None
        ),
        "direction_covariance_partition_relative_residual":u_resid/u_scale,
        "energy_variance_partition_relative_residual":e_resid/e_scale,
        "energy_direction_partition_relative_residual":eu_resid/eu_scale,
        "direction_energy_cloning_is_independent_fission_genealogy": False,
    }


def read_actual_clone_partition(file: Path) -> dict:
    path=Path(file)
    physically_valid, _=read_real_bank_angular_physics(path)
    with h5py.File(path,"r") as h:
        source=h["source_bank"][()]
    xyz=np.column_stack([source["r"][c] for c in ("x","y","z")])
    u=np.column_stack([source["u"][c] for c in ("x","y","z")])
    out=partition_clone_moments(xyz,u,source["E"],source["wgt"])
    if (out["nominal_source_particles"] != physically_valid["nominal_source_particles"]
            or out["exact_distinct_fission_source_positions"] != physically_valid[
                "distinct_exact_source_positions"]
            or not math.isclose(out["inverse_simpson_effective_source_positions"],
                                physically_valid[
                                    "effective_source_position_count_inverse_simpson"],
                                rel_tol=1e-12, abs_tol=1e-9)
            or not np.allclose(out["mean_source_direction"],
                               physically_valid[
                                   "mean_source_emission_direction_vector"],
                               rtol=1e-12,atol=1e-10)
            or not math.isclose(out["mean_emission_energy_eV"],
                                physically_valid["emission_energy_weighted_mean_eV"],
                                rel_tol=1e-12,abs_tol=1e-8)):
        raise ValueError("independent real HDF5 source-clone direction/energy identity mismatch")
    out.update({
        "seed":physically_valid["seed"],
        "checkpoint_batch":physically_valid["checkpoint_batch"],
        "statepoint_sha256": physically_valid["actual_statepoint_sha256"],
    })
    return out


def audit_real_clone_moments(summary_path: Path, direction_audit_path: Path) -> dict:
    summary_path=Path(summary_path)
    direction_audit_path=Path(direction_audit_path)
    s=json.loads(summary_path.read_text(encoding="utf-8"))
    direction=json.loads(direction_audit_path.read_text(encoding="utf-8"))
    if (s.get("execution")!="ACTUAL_OPENMC_TRANSPORT"
            or s.get("assembly")!="core"
            or direction.get("audit")!="REAL_OPENMC_FISSION_BANK_DIRECTION_ENERGY_AND_SPATIAL_CLONING"
            or direction.get("actual_transport_summary_sha256")!=_sha256(summary_path)):
        raise ValueError("missing genuine source and angular physics provenance")
    runs=s.get("runs")
    batches=s.get("saved_source_bank_batches")
    terminal=s.get("batches")
    direction_runs=direction.get("independent_seed_real_source_records")
    if (not isinstance(runs,list) or len(runs)<4
            or not isinstance(direction_runs,list) or len(direction_runs)!=len(runs)
            or not isinstance(batches,list) or len(batches)<3
            or any(type(n)!=int or n<=0 for n in batches)
            or any(a>=b for a,b in zip(batches,batches[1:]))
            or type(terminal)!=int or terminal!=batches[-1]):
        raise ValueError("invalid source trajectory checkpoints or independent seeds")
    records=[]
    seen=set()
    hashes=set()
    for run,reference in zip(runs,direction_runs):
        seed=run.get("seed")
        if type(seed)!=int or seed<=0 or seed in seen or reference.get("seed")!=seed:
            raise ValueError("inconsistent real independent neutron fission-source seed")
        seen.add(seed)
        path=Path(run.get("statepoint_path",""))
        if not path.is_file() or _sha256(path)!=run.get("statepoint_sha256"):
            raise ValueError("real terminal OpenMC statepoint SHA-256 provenance mismatch")
        width=len(str(terminal))
        ref_snapshots=reference.get("checkpoints")
        if not isinstance(ref_snapshots,list) or len(ref_snapshots)!=len(batches):
            raise ValueError("mismatched independent source direction checkpoints")
        observed=[]
        for batch,ref in zip(batches,ref_snapshots):
            cp=path.parent/f"statepoint.{batch:0{width}d}.h5"
            if not cp.is_file():
                raise ValueError("missing actual OpenMC neutron source checkpoint")
            row=read_actual_clone_partition(cp)
            digest=row["statepoint_sha256"]
            if (row["seed"]!=seed or row["checkpoint_batch"]!=batch
                    or digest!=ref.get("actual_statepoint_sha256")
                    or digest in hashes):
                raise ValueError("real source checkpoint hash, batch or seed identity mismatch")
            if (row["nominal_source_particles"]!=ref.get("nominal_source_particles")
                    or row["exact_distinct_fission_source_positions"]!=ref.get(
                        "distinct_exact_source_positions")):
                raise ValueError("executed angular and cloned-site population disagree")
            hashes.add(digest)
            observed.append(row)
        if observed[-1]["statepoint_sha256"]!=run["statepoint_sha256"]:
            raise ValueError("final real source checkpoint is not executed statepoint")
        records.append({"seed":seed,"checkpoints":observed})

    rows=[r for seed in records for r in seed["checkpoints"]]
    fractions=[
        r["fraction_total_direction_variance_within_exact_position"]
        for r in rows
    ]
    return {
        "audit":"ACTUAL_OPENMC_SOURCE_POSITION_CONDITIONAL_MOMENT_CLOSURE",
        "real_transport_summary_sha256":_sha256(summary_path),
        "real_source_direction_report_sha256":_sha256(direction_audit_path),
        "independent_seed_count":len(seen),
        "matched_checkpoint_batches":batches,
        "all_20_real_checkpoint_moment_partitions_closed":True,
        "maximum_direction_covariance_partition_relative_residual":max(
            r["direction_covariance_partition_relative_residual"] for r in rows
        ),
        "maximum_energy_variance_partition_relative_residual":max(
            r["energy_variance_partition_relative_residual"] for r in rows
        ),
        "maximum_energy_direction_partition_relative_residual":max(
            r["energy_direction_partition_relative_residual"] for r in rows
        ),
        "minimum_direction_variance_within_cloned_site_fraction":min(
            x for x in fractions if x is not None
        ),
        "maximum_direction_variance_within_cloned_site_fraction":max(
            x for x in fractions if x is not None
        ),
        "minimum_effective_source_site_fraction":min(
            r["inverse_simpson_effective_source_positions"]/
            r["nominal_source_particles"] for r in rows
        ),
        "seeds":records,
        "cloned_source_sites_are_independent_neutron_histories":False,
        "empirical_covariance_partition_is_fission_source_stationarity":False,
        "criticality_or_nasa_nerva_reactor_physics_benchmarked":False,
        "interpretation":(
            "Exact, provenance-linked OpenMC neutron source-site identities "
            "partition weighted angular and energy variance within and between "
            "resampled fission positions. Diverse emission directions and "
            "energies at an identical point do NOT create new fission sites. "
            "This is a mathematical closure check, NOT a physically calibrated "
            "source genealogical ESS, isotropy test or proof of stationary "
            "fundamental-mode source."
        ),
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",type=Path,required=True)
    parser.add_argument("--source-direction",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=audit_real_clone_moments(args.summary,args.source_direction)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Actual cloned fission-site direction/energy moment identities: PASS")
    print("Maximum weighted covariance closure error:",
          report["maximum_direction_covariance_partition_relative_residual"])
    print("Worst inverse-Simpson spatial fission-site fraction:",
          report["minimum_effective_source_site_fraction"])
    print("Neutron source independence/stationarity: NOT ESTABLISHED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
