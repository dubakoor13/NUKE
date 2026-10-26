"""Actual coupled neutron+photon OpenMC HDF5 energy-deposition ledger.

Read *executed* OpenMC StatePoint score arrays. Enforce photon_transport
and identical native HDF5/score identities; separate neutron/photon heating
and material/mesh partitions. The four prior Stage21-29 cold neutron-only
runs are not substitutes for this executed coupled transport.

All scores are eV per Monte Carlo source particle. This does NOT normalize
to physical reactor watts, infer spatial energy-deposition error covariance,
establish stationary fission banks, or qualify a hot NERVA reactor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EV_TO_J = 1.602176634e-19


def reconcile_energy_partitions(
    mesh_integral: float, material_heating: list[float],
    neutron_heating: float, photon_heating: float,
    *, all_scores_share_estimator: bool,
    relative_tolerance: float = 1.e-8,
) -> dict:
    """Cross-check mean scores; material/particle bins are CORRELATED.

    Mean identities are only exact (to arithmetic tolerance) when the
    estimators and scored domains genuinely coincide. Different estimators
    have irreducible sampling differences; do NOT accept/deny them using
    independent-bin SE quadrature, as covariance is not available.
    """
    vals=np.asarray([mesh_integral, *material_heating,
                     neutron_heating, photon_heating],dtype=float)
    if (not np.isfinite(vals).all() or np.any(vals<0)
            or len(material_heating)==0 or
            not isinstance(all_scores_share_estimator,bool)
            or not math.isfinite(relative_tolerance)
            or not 0 < relative_tolerance <= 0.01):
        raise ValueError("invalid/negative complete energy-deposition score partition")
    mesh=float(mesh_integral)
    mat=float(math.fsum(material_heating))
    particle=float(neutron_heating+photon_heating)
    if mesh<=0 or mat<=0 or particle<=0:
        raise ValueError("zero/negative spatial deposited heating in executed transport")
    mat_res=mat-mesh
    part_res=particle-mesh
    matched = (
        abs(mat_res) <= relative_tolerance*mesh and
        abs(part_res) <= relative_tolerance*mesh)
    return {
        "mesh_integrated_heating_ev_per_source":mesh,
        "material_integrated_heating_ev_per_source":mat,
        "particle_integrated_heating_ev_per_source":particle,
        "mesh_vs_material_relative_residual":mat_res/mesh,
        "mesh_vs_particle_relative_residual":part_res/mesh,
        "neutron_heating_ev_per_source":float(neutron_heating),
        "photon_heating_ev_per_source":float(photon_heating),
        "photon_fraction_of_recorded_heating":float(photon_heating/particle),
        "photon_energy_deposition_observed":bool(photon_heating>0),
        "material_particle_mesh_estimator_consistent":all_scores_share_estimator,
        "same_domain_partition_means_close":bool(matched),
        "energy_source_normalization_is_ev_per_MC_start":True,
        "physical_reactor_watts_known":False,
    }


def _unique(sp, name: str):
    groups=[t for t in sp.tallies.values() if t.name==name]
    if len(groups)!=1:
        raise ValueError("missing/duplicate native OpenMC tally "+name)
    return groups[0]


def _read_score(tally, score: str, n: int, expected_bins: int):
    if (score not in tally.scores or tally.num_realizations!=n or
            len(tally.nuclides)!=1 or str(tally.nuclides[0])!="total"):
        raise ValueError("incompatible source batch or nuclide/score on "+tally.name)
    mean=np.asarray(tally.get_values(scores=[score],value="mean"),dtype=float)
    sem=np.asarray(tally.get_values(scores=[score],value="std_dev"),dtype=float)
    if mean.shape!=(expected_bins,1,1) or sem.shape!=mean.shape:
        raise ValueError("unexpected native filtered score array on "+tally.name)
    vals=mean.reshape(-1)
    errors=sem.reshape(-1)
    if (not np.isfinite(vals).all() or not np.isfinite(errors).all()
            or np.any(vals<0) or np.any(errors<0)):
        raise ValueError("nonphysical negative/nonfinite deposited score/SEM")
    return vals, errors


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_actual_coupled_heating(
    statepoint_path: str | Path, nuclear_report: str | Path,
    transport_provenance: str | Path,
) -> dict:
    """Inspect original native coupled-transport binary, no synthetic fallback."""
    import openmc
    path=Path(statepoint_path)
    nuclear_path=Path(nuclear_report)
    provenance_path=Path(transport_provenance)
    if not all(p.is_file() for p in (path,nuclear_path,provenance_path)):
        raise FileNotFoundError("actual statepoint, data report, provenance mandatory")
    nd=json.loads(nuclear_path.read_text(encoding="utf-8"))
    prov=json.loads(provenance_path.read_text(encoding="utf-8"))
    if (not nd.get("ready") or nd.get("photon_transport_requested") is not True
            or set(nd.get("verified_photon_elements",())) !=
                    set(nd.get("required_photon_elements",()))
            or not nd.get("required_photon_elements")
            or nd.get("missing_photon_elements") or nd.get("photon_binary_audit_errors")
            or prov.get("config",{}).get("photon_transport") is not True
            or prov.get("nuclear_data",{}).get("cross_sections_sha256") != nd.get("sha256")):
        raise ValueError("actual evaluated photon nuclear data transport provenance missing")
    digest=_sha256(path)
    if prov.get("statepoint",{}).get("sha256")!=digest:
        raise ValueError("native coupled OpenMC statepoint SHA256/provenance mismatch")

    with openmc.StatePoint(str(path),autolink=False) as sp:
        if (not sp.photon_transport or sp.run_mode!="eigenvalue"
                or int(sp.seed)!=int(prov["config"]["seed"])
                or not sp.tallies):
            raise ValueError("not recorded original coupled OpenMC eigenvalue transport")
        core=_unique(sp,"nerva_3d_neutronics")
        mat=_unique(sp,"nerva_material_transport")
        particle=_unique(sp,"nerva_particle_heating")
        if (len(core.filters)!=1 or not isinstance(core.filters[0],openmc.MeshFilter)
                or len(mat.filters)!=1 or not isinstance(mat.filters[0],openmc.MaterialFilter)
                or len(particle.filters)!=1 or not isinstance(particle.filters[0],openmc.ParticleFilter)):
            raise ValueError("incompatible native photon/mesh/material scoring filters")
        if list(str(k) for k in particle.filters[0].bins)!=["neutron","photon"]:
            raise ValueError("particle heating does not distinguish neutron and photon")
        material_bins=mat.filters[0].bins
        nm=len(material_bins)
        n=int(core.num_realizations)
        if n<5:
            raise ValueError("too few physical active OpenMC tally realizations")
        if any(t.num_realizations!=n for t in (mat,particle)):
            raise ValueError("incompatible active-batch samples across energy tallies")
        mesh_bins=core.filters[0].num_bins
        mesh, mesh_sem=_read_score(core,"heating",n,mesh_bins)
        mats, mats_sem=_read_score(mat,"heating",n,nm)
        species, species_sem=_read_score(particle,"heating",n,2)
        estimator_names={str(t.estimator) for t in (core,mat,particle)}
        partition=reconcile_energy_partitions(
            float(math.fsum(mesh)),[float(x) for x in mats],
            float(species[0]),float(species[1]),
            all_scores_share_estimator=len(estimator_names)==1
        )
        # A positive photon deposition score is a data OBSERVATION, not
        # automatic permission to normalize to a reactor power.
        report={
            "schema":"nuke-real-coupled-neutron-photon-heating-stage30-v1",
            "evidence":"ORIGINAL_OPENMC_0153_EXECUTED_COUPLED_NEUTRON_PHOTON_HDF5",
            "actual_native_statepoint_sha256":digest,
            "actual_cross_sections_index_sha256":nd["sha256"],
            "actual_evaluated_photon_element_sha256":nd["photon_binary_sha256"],
            "run_mode":str(sp.run_mode),
            "random_seed":int(sp.seed),
            "batches":int(sp.n_batches),
            "inactive_batches":int(sp.n_inactive),
            "actual_active_tally_realizations":n,
            "photon_transport_enabled_in_native_HDF5":bool(sp.photon_transport),
            "mesh_bin_count":int(mesh_bins),
            "material_filter_bin_count":int(nm),
            "material_filter_bin_ids":[str(getattr(v,"id",v)) for v in material_bins],
            "native_heating_estimators":{
                "mesh":str(core.estimator),"materials":str(mat.estimator),
                "particles":str(particle.estimator)},
            "mesh_bin_conditional_SEMs_not_aggregable_without_covariance":mesh_sem.tolist(),
            "material_bin_conditional_SEMs_not_aggregable_without_covariance":mats_sem.tolist(),
            "particle_global_conditional_SEMs":species_sem.tolist(),
            "material_heating_ev_per_source":mats.tolist(),
            "per_source_partition":partition,
            "fission_source_stationarity_qualified":False,
            "historical_hot_NERVA_model_validated":False,
            "neutron_photon_energy_data_release_qualified":False,
            "absolute_reactor_watts_qualified":False,
            "actual_photon_source_spatial_heating_map_qualified_for_NASA":False,
            "scientific_notes":[
                "Executed eigenvalue snapshot of the current cold NERVA-derived SURROGATE only.",
                "eV per Monte Carlo source is not physical reactor watts; an independently justified power/source-rate is required.",
                "No summing spatial-bin SEMs without paired per-batch covariance.",
                "Shared neutron-photon histories correlate energy scores; marginal SEMs do not establish partition uncertainty.",
                "Fission-bank nonstationarity persists from prior cold-core experiments unless independently proven otherwise.",
                "Positive evaluated photon transport data inventory does not qualify neutron evaluated temperatures or NASA historical geometry.",
            ],
        }
        return report


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--statepoint",type=Path,required=True)
    p.add_argument("--nuclear-data",type=Path,required=True)
    p.add_argument("--provenance",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    report=inspect_actual_coupled_heating(
        a.statepoint,a.nuclear_data,a.provenance)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("Executed coupled photon StatePoint SHA256:",report["actual_native_statepoint_sha256"])
    print("Neutron-photon material energy closure:",report["per_source_partition"])
    print("Hot NERVA model physical validation: NOT ESTABLISHED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
