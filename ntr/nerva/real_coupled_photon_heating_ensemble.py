"""Stage 31: genuine four-seed coupled n/gamma heating covariance audit.

All evidence MUST be four independently executed OpenMC 0.15.3
neutron+photon cold-surrogate StatePoints with genuine native HDF5.
The observable is per starting Monte Carlo neutron, NOT reactor watts.

Cross-bin covariance is estimated from independent *seed-level vectors*,
not falsely reconstructed by summing within-statepoint marginal SEMs.
With only four independent RNG seeds, covariance matrices can have rank at
most 3; reported standard errors are DESCRIPTIVE if fission sources are
nonstationary and are not calibrated intervals or NASA validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .real_coupled_photon_heating_audit import inspect_actual_coupled_heating

_SEEDS=(13007,23011,33013,43019)
_PARTICLE_NAMES=("neutron","photon","electron","positron")


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(1<<20),b""):
            h.update(data)
    return h.hexdigest()


def seed_vector_covariance(
    matrix: np.ndarray, *, label: str,
) -> dict:
    """Paired per-seed vector covariance, preserving bin cross-correlations.

    It is mathematically incorrect to sum per-voxel native SEMs in quadrature
    to get the uncertainty of a summed heating rate. Here entire same-seed
    component vectors are available, so cross-bin covariance is observable
    *across seeds* with the small sample caveat.
    """
    data=np.asarray(matrix,dtype=float)
    if (data.ndim!=2 or data.shape[0]<4 or data.shape[1]<1
            or not np.isfinite(data).all() or np.any(data<0)):
        raise ValueError("invalid original coupled seed vectors: "+label)
    n,k=data.shape
    means=np.mean(data,axis=0)
    residual=data-means
    cov=residual.T@residual/(n-1)
    cov=(cov+cov.T)/2.
    sd=np.sqrt(np.maximum(0.,np.diag(cov)))
    total_by_seed=np.sum(data,axis=1)
    sum_cov=float(np.sum(cov))
    observed_total_var=float(np.var(total_by_seed,ddof=1))
    if (not math.isclose(sum_cov,observed_total_var,rel_tol=1e-10,
                         abs_tol=1e-8*max(1.,observed_total_var))
            or not np.isfinite(cov).all()):
        raise ValueError("paired seed covariance violates total heating variance: "+label)
    total_mean=float(np.mean(total_by_seed))
    return {
        "bin_count":k,
        "unweighted_mean_per_starting_neutron":means.tolist(),
        "between_seed_sample_standard_deviation":sd.tolist(),
        "descriptive_between_seed_mean_SEM_IF_STATIONARY":
            (sd/math.sqrt(n)).tolist(),
        "paired_between_seed_sample_covariance":cov.tolist(),
        "observed_covariance_rank_upper_bound":min(n-1,k),
        "total_by_seed":total_by_seed.tolist(),
        "total_mean_per_starting_neutron":total_mean,
        "total_between_seed_sample_sd":float(math.sqrt(max(0.,sum_cov))),
        "total_descriptive_seed_mean_SEM_IF_STATIONARY":
            float(math.sqrt(max(0.,sum_cov)/n)),
        "max_absolute_covariance_vs_direct_total_variance_residual":
            abs(sum_cov-observed_total_var),
        "native_within_seed_spatial_bin_SEMs_not_summed":True,
        "formal_confidence_coverage_established":False,
    }


def audit_actual_four_seed_coupled_heating(root: str | Path) -> dict:
    base=Path(root)
    observations=[]
    matching_config=None
    matching_xml=None
    matching_data=None
    matching_grid=None
    matching_materials=None
    hashes=set()
    for seed in _SEEDS:
        folder=base/f"seed_{seed}"
        statepoint=folder/"statepoint.65.h5"
        nuclear_report=folder/"nuclear_data_report.json"
        provenance=folder/"transport_provenance.json"
        for src in (statepoint,nuclear_report,provenance):
            if not src.is_file() or src.stat().st_size==0:
                raise FileNotFoundError("actual original four-seed binary evidence required: "+str(src))
        meta=json.loads(provenance.read_text(encoding="utf-8"))
        cfg=dict(meta.get("config",{}))
        if cfg.pop("seed",None)!=seed or not cfg.get("photon_transport"):
            raise ValueError("native coupled photon RNG seed/provenance mismatch")
        for filename,expected in meta.get("xml_sha256",{}).items():
            local=folder/filename
            if not local.is_file() or _sha256(local)!=expected:
                raise ValueError("original OpenMC input XML SHA mismatch: "+filename)
        xml_fingerprints=tuple((filename,meta["xml_sha256"].get(filename))
                               for filename in ("geometry.xml","materials.xml","tallies.xml"))
        if any(sha is None for _,sha in xml_fingerprints):
            raise ValueError("missing original neutron/photon geometry/material/tally XML identity")
        r=inspect_actual_coupled_heating(statepoint,nuclear_report,provenance)
        if (r["random_seed"]!=seed or r["batches"]!=65 or
                r["inactive_batches"]!=20 or
                r["actual_active_tally_realizations"]!=45 or
                not r["photon_transport_enabled_in_native_HDF5"] or
                r["particle_filter_order"]!=list(_PARTICLE_NAMES) or
                not r["all_particle_heating_types_scored"]):
            raise ValueError("native source configuration inconsistent with fixed four-seed design")
        if (not r["per_source_partition"]["same_domain_partition_means_close"]
                or not r["per_source_partition"]["material_particle_mesh_estimator_consistent"]
                or not r["per_source_partition"]["photon_energy_deposition_observed"]):
            raise ValueError("native original coupled neutron/photon energy partition not closed")
        if (r["fission_source_stationarity_qualified"] or
                r["historical_hot_NERVA_model_validated"] or
                r["absolute_reactor_watts_qualified"]):
            raise ValueError("fake scientific qualification in native score ledger")
        core_config=cfg
        grid=(r["mesh_dimensions_xyz"],r["mesh_lower_left_cm"],r["mesh_upper_right_cm"])
        mats=r["material_filter_bin_ids"]
        nuclear=(r["actual_cross_sections_index_sha256"],
                 r["actual_evaluated_photon_element_sha256"])
        if matching_config is None:
            matching_config=core_config
            matching_xml=xml_fingerprints
            matching_data=nuclear
            matching_grid=grid
            matching_materials=mats
        elif (core_config!=matching_config or xml_fingerprints!=matching_xml
              or nuclear!=matching_data or grid!=matching_grid or
              mats!=matching_materials):
            raise ValueError("independent seeded runs do not share identical physical model and evaluated data")
        checksum=r["actual_native_statepoint_sha256"]
        if checksum in hashes:
            raise ValueError("reused/duplicated OpenMC statepoint binary across seeds")
        hashes.add(checksum)
        observations.append(r)

    particle_vectors=np.asarray([
        [r["per_source_partition"][name+"_heating_ev_per_source"]
         for name in _PARTICLE_NAMES] for r in observations
    ],dtype=float)
    material_vectors=np.asarray([
        r["material_heating_ev_per_source"] for r in observations],dtype=float)
    axial_vectors=np.asarray([
        r["axial_heating_ev_per_source"] for r in observations],dtype=float)
    particles=seed_vector_covariance(particle_vectors,label="actual four particle types")
    materials=seed_vector_covariance(material_vectors,label="actual model material filter")
    axial=seed_vector_covariance(axial_vectors,label="actual native axial heating")
    mesh_total=np.asarray([
        r["per_source_partition"]["mesh_integrated_heating_ev_per_source"]
        for r in observations],dtype=float)
    for domain,result in (("particles",particles),("materials",materials),("axial",axial)):
        for total,reference in zip(result["total_by_seed"],mesh_total):
            if not math.isclose(total,reference,rel_tol=1e-10,abs_tol=1e-7):
                raise ValueError("actual independent-seed "+domain+" partition not conserved")
    material_sd=materials["between_seed_sample_standard_deviation"]
    axial_sd=axial["between_seed_sample_standard_deviation"]
    ratios=[]
    for mean,sd in zip(axial["unweighted_mean_per_starting_neutron"],axial_sd):
        ratios.append(sd/mean if mean>0 else None)
    per_run=[]
    for r in observations:
        per_run.append({
            "seed":r["random_seed"],
            "native_statepoint_sha256":r["actual_native_statepoint_sha256"],
            "native_active_batches":r["actual_active_tally_realizations"],
            "native_total_heating_ev_per_source":r["per_source_partition"]["mesh_integrated_heating_ev_per_source"],
            "native_material_mesh_closure_relative":r["per_source_partition"]["mesh_vs_material_relative_residual"],
            "native_all_particle_mesh_closure_relative":r["per_source_partition"]["mesh_vs_particle_relative_residual"],
            "native_particle_ev_per_source":{
                name:r["per_source_partition"][name+"_heating_ev_per_source"]
                for name in _PARTICLE_NAMES
            },
        })
    return {
        "schema":"nuke-executed-coupled-neutron-photon-four-seed-stage31-v1",
        "evidence":"FOUR_INDEPENDENT_ACTUAL_OPENMC_0153_COUPLED_NEUTRON_PHOTON_STATEPOINTS",
        "original_independent_openmc_rng_seeds":list(_SEEDS),
        "real_native_coupled_hdf5_count":len(observations),
        "all_actual_geometry_material_and_tally_XML_SHA_unchanged":True,
        "all_actual_evaluated_neutron_photon_data_index_identity_unchanged":True,
        "all_electron_positron_neutron_photon_energy_deposition_partitions_conserved":True,
        "mesh_dimensions_xyz":matching_grid[0],
        "mesh_lower_left_cm":matching_grid[1],
        "mesh_upper_right_cm":matching_grid[2],
        "material_filter_bin_ids":matching_materials,
        "per_actual_seed":per_run,
        "correlated_particle_heating_seed_statistics":particles,
        "correlated_material_heating_seed_statistics":materials,
        "correlated_axial_heating_seed_statistics":axial,
        "axial_relative_between_seed_SD_by_bin":ratios,
        "max_relative_deviation_axial_by_bin_if_nonzero":
            max((r for r in ratios if r is not None),default=None),
        "material_between_seed_sample_sd_by_bin":material_sd,
        "conservative_cross_bin_covariance_from_complete_independent_seed_vectors":True,
        "statistically_calibrated_fission_source_convergence":False,
        "real_neutron_photon_heating_hot_NERVA_physically_qualified":False,
        "independent_reactor_watt_normalization_exists":False,
        "nasa_historical_core_benchmark_validated":False,
        "limitations":[
            "Four genuine RNG seeds are distinct but fission-source stationarity is NOT demonstrated.",
            "With 4 seeds each sample covariance matrix has rank at most 3.",
            "Between-seed spread is descriptive; confidence coverage and bias remain unqualified.",
            "Native per-bin within-seed standard errors are NOT combined as statistically independent.",
            "Cold far-subcritical three-ring demonstrator is not operating NASA NERVA.",
            "Energy is eV per Monte Carlo source; no physical source strength or hot-core watts inferred.",
            "Particle types indicate energy-tally collision event, not fission gamma ancestry.",
        ],
    }


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=audit_actual_four_seed_coupled_heating(a.root)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    pstat=r["correlated_particle_heating_seed_statistics"]
    print("Four real coupled neutron/photon OpenMC seeds audited:",r["original_independent_openmc_rng_seeds"])
    print("Source-normalized total heating 4-seed mean:",pstat["total_mean_per_starting_neutron"])
    print("Observed between-seed total sample SD:",pstat["total_between_seed_sample_sd"])
    print("All actual energy partitions conserved with cross-bin covariance: YES")
    print("Fission source stationary and hot NASA NERVA validated: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
