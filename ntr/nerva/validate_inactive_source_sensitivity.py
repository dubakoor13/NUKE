"""Deterministic tests for OpenMC inactive-batch source-relaxation math.

These dictionaries are SYNTHETIC; real C++ OpenMC neutron transport and
HDF5 checks run separately in the dedicated GitHub workflow.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from .inactive_source_sensitivity import (
    assess_actual_source_burnin, assess_burnin_evidence, _same_physical_core,
)


def reject(function, needle):
    try:
        function()
    except ValueError as exc:
        assert needle in str(exc), str(exc)
    else:
        raise AssertionError("unphysical source burn-in evidence accepted: "+needle)


def fixture(inactive, seeds, k, tv):
    batches = [inactive+10,inactive+30,inactive+50,inactive+70,inactive+90]
    summary={
        "execution":"ACTUAL_OPENMC_TRANSPORT", # structurally mocked only
        "assembly":"core", "rings":3,
        "particles_per_generation":3000,
        "inactive_batches":inactive, "batches":inactive+90,
        "cross_sections_xml_sha256":"immutable-xs-id",
        "saved_source_bank_batches":batches,
        "independent_seed_source_bank_shape_diagnostics":{
            "spatial_lower_cm":[-30,-30,-66],
            "spatial_upper_cm":[30,30,66],
        },
        "run_statistics":{
            "inverse_variance_mean_keff":k,
            "dispersion_inflated_mean_keff_error":0.00005,
        },
        "runs":[{"seed":s} for s in seeds],
        "multi_checkpoint_source_bank_trajectory":{
            "runs":[{
                "seed":s,
                "snapshots":[{
                    "seed":s,"completed_batch":batch,
                    "inverse_simpson_distinct_position_count":24,
                    "statepoint_sha256":f"{s * 1000 + batch:064x}",
                } for batch in batches],
            } for s in seeds],
        },
    }
    resolution=[]
    for side in (2,4,8):
        resolution.append({
            "bins_per_axis":side,
            "final_mean_cross_seed_tv":tv*(side/8.),
            "minimum_effective_distinct_position_fraction":.006,
        })
    audit={
        "analysis":"EXACT_SAME_OPENMC_SOURCE_BANKS_ON_NESTED_SPATIAL_GRIDS",
        "grid_resolution_audit_passed":True,
        "source_stationarity_validated":False,
        "source_diversity_screen_status":"FAIL_SOURCE_CLONING_SCREEN",
        "resolutions":resolution,
    }
    return summary,audit


def main() -> int:
    short,a=fixture(30,(13007,23011,33013,43019),.01170,.93)
    long,b=fixture(110,(53003,63007,73009,83003),.01165,.91)
    good=assess_burnin_evidence(short,long,a,b)
    assert good["same_number_of_active_tally_generations"]
    assert good["identical_relative_source_checkpoint_active_batches"]==[10,30,50,70,90]
    assert good["source_a"]["inactive_generations"]==30
    assert good["source_b"]["inactive_generations"]==110
    assert good["difference_in_fine_grid_cross_seed_source_TV"]<0
    assert good["fission_source_stationarity_independently_proven"] is False
    assert good["historical_nasa_nerva_reactor_validated"] is False

    invalid=deepcopy(long)
    invalid["batches"]=210
    reject(lambda:assess_burnin_evidence(short,invalid,a,b),"same nontrivial")

    invalid=deepcopy(long)
    invalid["saved_source_bank_batches"][-1]=199
    reject(lambda:assess_burnin_evidence(short,invalid,a,b),"observation windows")

    invalid=deepcopy(long)
    invalid["runs"][0]["seed"]=13007
    reject(lambda:assess_burnin_evidence(short,invalid,a,b),"nonoverlapping")

    invalid=deepcopy(long)
    invalid["particles_per_generation"]=6000
    reject(lambda:assess_burnin_evidence(short,invalid,a,b),"source particle")

    invalid=deepcopy(long)
    invalid["cross_sections_xml_sha256"]="different-evaluated-nuclear-data"
    reject(lambda:assess_burnin_evidence(short,invalid,a,b),"nuclear data")

    bad=deepcopy(b)
    bad["resolutions"][2]["final_mean_cross_seed_tv"]=1.5
    reject(lambda:assess_burnin_evidence(short,long,a,bad),"spatial total-variation")

    bad=deepcopy(b)
    bad["source_stationarity_validated"]=True
    reject(lambda:assess_burnin_evidence(short,long,a,bad),"audit required")

    cfg={
        "seed":13007,"batches":120,"inactive":30,
        "statepoint_batches":[40,60,80,100,120],
        "particles":3000,"core_rings":3,
        "hydrogen_temperature_k":293.6059129982851,
        "uranium_enrichment_wt_percent":5.0,
    }
    p={"assembly":"core","openmc_version":"0.15.3",
       "config":cfg,
       "nuclear_data":{"cross_sections_sha256":"same-physical-data"},
       "xml_sha256":{
           "geometry.xml":"sha-g","materials.xml":"sha-m",
           "tallies.xml":"sha-t"}}
    q=deepcopy(p)
    q["config"].update({
        "seed":53003,"batches":200,"inactive":110,
        "statepoint_batches":[120,140,160,180,200]})
    _same_physical_core(p,q)
    q["config"]["hydrogen_temperature_k"]=600.
    reject(lambda:_same_physical_core(p,q),"physical core")
    q=deepcopy(p)
    q["xml_sha256"]["materials.xml"]="falsified"
    reject(lambda:_same_physical_core(p,q),"materials.xml")

    with TemporaryDirectory() as folder:
        bogus=Path(folder)/"short"
        fake=Path(folder)/"long"
        bogus.mkdir()
        fake.mkdir()
        (bogus/"real_core_transport_summary.json").write_text(
            json.dumps({"execution":"SYNTHETIC","assembly":"core","rings":3}))
        (fake/"real_core_transport_summary.json").write_text("{}")
        reject(lambda:assess_actual_source_burnin(bogus,fake),"actual OpenMC")

    print("NERVA source inactive-generation sensitivity and physical-input identity: PASS")
    print("  synthetic structural data; genuine OpenMC is run separately")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
