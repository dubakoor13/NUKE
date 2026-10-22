"""Synthetic exact-identity tests for real OpenMC population physics audit.

The HDF5 integration workflow separately executes official OpenMC C++ and
reads the real evaluated-data statepoints. None of these constructed
dictionaries are claimed to be reactor neutron transport.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .population_source_physics import compare_population_physics


def reject(call, fragment):
    try:
        call()
    except (ValueError, KeyError) as e:
        assert fragment in str(e), (fragment,str(e))
    else:
        raise AssertionError("invalid OpenMC neutron population physics accepted: "+fragment)


def fixture(n, seed_base, real_sha):
    seeds=(seed_base+1,seed_base+2,seed_base+3,seed_base+4)
    batches=(40,60,80,100,120)
    trajectories=[]
    energy_runs=[]
    for seed in seeds:
        snapshots=[{"seed":seed,"completed_batch":batch,
                    "statepoint_sha256":f"{seed*1000+batch:064x}",
                    "inverse_simpson_distinct_position_count":int(n*.01)}
                   for batch in batches]
        trajectories.append({"seed":seed,"snapshots":snapshots})
        energy_runs.append({"seed":seed,"snapshots":[
            {"statepoint_sha256":s["statepoint_sha256"]}
            for s in snapshots]})
    summary={
        "particles_per_generation":n,"batches":120,"inactive_batches":30,
        "cross_sections_xml_sha256":"verified-evaluated-nndc-xs",
        "saved_source_bank_batches":list(batches),
        "runs":[{"seed":r["seed"],"statepoint_sha256":r["snapshots"][-1]["statepoint_sha256"]}
                for r in trajectories],
        "independent_seed_source_bank_shape_diagnostics":{
            "spatial_lower_cm":[-30.,-30.,-66.],
            "spatial_upper_cm":[30.,30.,66.],
        },
        "multi_checkpoint_source_bank_trajectory":{
            "runs":trajectories,
            "between_seed_mean_total_variation_by_batch":[.50,.54,.55,.56,.57],
        },
    }
    matrix=[[s["statepoint_sha256"] for s in r["snapshots"]]
            for r in trajectories]
    resolutions=[]
    for side in (2,4,8):
        factor={2:.5,4:1,8:1.4}[side]
        resolutions.append({
            "bins_per_axis":side,
            "mean_cross_seed_tv_per_checkpoint":[
                factor*x for x in [.50,.54,.55,.56,.57]
            ],
            "minimum_effective_distinct_position_fraction":.01,
            "all_checkpoint_statepoint_sha256":matrix,
        })
    space={
        "real_transport_summary_sha256":real_sha,
        "grid_resolution_audit_passed":True,
        "source_stationarity_validated":False,
        "resolutions":resolutions,
        "source_diversity_screen_status":"FAIL_SOURCE_CLONING_SCREEN",
    }
    energies={
        "real_transport_summary_sha256":real_sha,
        "saved_source_batches":list(batches),
        "neutron_energy_bin_edges_eV":[0.,1e5,1e6,3e7],
        "mean_cross_seed_energy_spectrum_tv_by_checkpoint":[.1,.12,.13,.14,.11],
        "fission_source_spectral_stationarity_validated":False,
        "runs":energy_runs,
    }
    reactions={
        "real_transport_summary_sha256":real_sha,
        "statepoint_sha256":[row["statepoint_sha256"] for row in summary["runs"]],
        "nuclear_data_index_sha256":"verified-evaluated-nndc-xs",
        "material_ids":[1,2,3],
        "incident_energy_bins_eV":[[0,1e5],[1e5,1e6],[1e6,3e7]],
        "independent_fission_source_stationarity_demonstrated":False,
        "reactions":{
            score:{
                "per_seed_global_rate_per_source":[v]*4,
                "mean_global_reaction_per_source":v,
                "between_seed_coefficient_of_variation":.01,
                "mean_pairwise_incident_energy_spectrum_tv":.025,
            }
            for score,v in (("absorption",.04),("nu-fission",.0117))
        }
    }
    return summary,space,energies,reactions


def main():
    a=fixture(3000,13000,"a"*64)
    b=fixture(6000,53000,"b"*64)
    def run(left=a,right=b):
        return compare_population_physics(
            (left[0],right[0]),(left[1],right[1]),
            (left[2],right[2]),(left[3],right[3]),
            summary_shas=("a"*64,"b"*64),
        )
    report=run()
    assert report["model_unchanged"]
    assert report["neutron_tally_and_source_histogram_identity_passed"]
    assert report["source_stationarity_demonstrated"] is False
    assert report["reactor_physical_nasa_benchmark_validated"] is False
    assert len(report["population_levels"]) == 2
    assert math.isclose(report["population_levels"][0][
        "final_cross_seed_tv_8cube"],.57*1.4,abs_tol=1e-12)
    assert math.isclose(report["population_levels"][1][
        "mean_effective_distinct_position_fraction"],.01,abs_tol=1e-12)
    altered=deepcopy(b)
    altered[0]["particles_per_generation"]=2500
    reject(lambda:run(right=altered),"strictly increase")
    altered=deepcopy(b)
    altered[0]["cross_sections_xml_sha256"]="changed-xs"
    reject(lambda:run(right=altered),"evaluated data")
    altered=deepcopy(b)
    altered[0]["saved_source_bank_batches"][-1]=130
    reject(lambda:run(right=altered),"generations")
    altered=deepcopy(b)
    altered[0]["independent_seed_source_bank_shape_diagnostics"][
        "spatial_lower_cm"][0]=-100
    reject(lambda:run(right=altered),"envelope")
    altered=deepcopy(b)
    altered[1]["real_transport_summary_sha256"]="f"*64
    reject(lambda:run(right=altered),"same transport summary")
    altered=deepcopy(b)
    altered[1]["resolutions"][1]["mean_cross_seed_tv_per_checkpoint"][-1]=.01
    reject(lambda:run(right=altered),"source TV inconsistent")
    altered=deepcopy(b)
    altered[1]["resolutions"][2]["all_checkpoint_statepoint_sha256"][0][0]="f"*64
    reject(lambda:run(right=altered),"saved HDF5 source banks")
    altered=deepcopy(b)
    altered[2]["runs"][0]["snapshots"][-1]["statepoint_sha256"]="f"*64
    reject(lambda:run(right=altered),"different HDF5 source particles")
    altered=deepcopy(b)
    altered[3]["statepoint_sha256"][0]="f"*64
    reject(lambda:run(right=altered),"different evaluated neutron data or HDF5")
    altered=deepcopy(b)
    altered[3]["material_ids"][0]=999
    reject(lambda:run(right=altered),"material neutron tally")
    altered=deepcopy(b)
    altered[3]["reactions"]["nu-fission"]["mean_global_reaction_per_source"]=-.1
    reject(lambda:run(right=altered),"unphysical neutron reaction score")
    altered=deepcopy(b)
    altered[1]["source_stationarity_validated"]=True
    reject(lambda:run(right=altered),"must not certify reactor stationarity")
    print("NERVA real source-population neutron HDF5 multi-resolution/marginal reconciliation: PASS")
    print("  synthetic dictionaries; independent real OpenMC C++ workflow required")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
