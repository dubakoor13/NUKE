"""Native real-OpenMC neutron economy and heating score closure audit.

Strictly observational on the original 2026-10-23 four-seed cold NERVA-
derived SURROGATE transport HDF5. Reconciles the same physical scoring
domain through mesh, material, and material+incident-energy filters, and
native analog-versus-tracklength neutron reaction estimators.

Critically distinguishes a nonzero fission KAPPA/HEATING score from actual
USABLE LOCAL HEAT deposition. The neutron-only heating-local first and
second batch moments are identically ZERO in the genuine run. Do not
normalize them into watts, guess a neutron/photon energy partition, or
treat a pass of the neutron accounting identity as reactor qualification.

No change to core geometry, control drums, fuel, cross sections, neutron
source, evaluated data, or OpenMC transport results. No synthetic or mock
statepoint fallback.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .real_transport_evidence_gate import (
    _Bundle, audit_real_transport_evidence, sha256_bytes,
)

_REQUIRED={
    "nerva_3d_neutronics": {
        "flux","absorption","fission","nu-fission",
        "kappa-fission","heating","heating-local",
    },
    "nerva_material_transport": {
        "flux","absorption","fission","nu-fission",
        "heating","heating-local",
    },
    "nerva_material_energy_transport": {
        "flux","absorption","fission","nu-fission","heating-local",
    },
    "nerva_global_reaction_tracklength": {"absorption","nu-fission"},
    "nerva_global_neutron_balance": {
        "absorption","scatter","nu-scatter","nu-fission",
    },
}
_SCORES=("flux","absorption","fission","nu-fission","heating","heating-local")
_TOL=1e-10


def _byte_string(raw:object)->str:
    if not isinstance(raw,(bytes,np.bytes_)):
        raise ValueError("OpenMC native tally must have bytes name and estimator")
    return bytes(raw).decode("utf-8").rstrip("\x00")


def _native_tallies(statepoint:h5py.File,realizations:int)->dict:
    result={}
    for group in statepoint["tallies"].values():
        if not isinstance(group,h5py.Group) or "name" not in group:
            continue
        name=_byte_string(group["name"][()])
        if name not in _REQUIRED:
            continue
        if name in result:
            raise ValueError("duplicated native reactor tally name: "+name)
        scores=tuple(_byte_string(b) for b in group["score_bins"][()])
        if (len(set(scores))!=len(scores)
                or not _REQUIRED[name].issubset(scores)):
            raise ValueError("missing or repeated native core reaction score: "+name)
        estimator=_byte_string(group["estimator"][()])
        required="analog" if name=="nerva_global_neutron_balance" else "tracklength"
        if estimator!=required:
            raise ValueError("incompatible source estimator: "+name)
        n=int(group["n_realizations"][()])
        array=np.asarray(group["results"][()],dtype=float)
        if (n!=realizations or array.ndim!=3
                or array.shape[1:]!=(len(scores),2)
                or not np.isfinite(array).all()):
            raise ValueError("native realization count/result moments mismatch: "+name)
        mean=array[:,:,0].sum(axis=0)/n
        stats={score:{"mean_per_source":float(mean[i]),
                      "raw_sum":float(array[:,i,0].sum()),
                      "raw_sum_sq":float(array[:,i,1].sum()),
                      "all_moments_zero":bool(np.all(array[:,i,:]==0.))}
               for i,score in enumerate(scores)}
        result[name]=stats
    if set(result)!=set(_REQUIRED):
        raise ValueError("missing native OpenMC core tallies")
    return result


def _matched(a:float,b:float,context:str):
    if not math.isclose(a,b,rel_tol=_TOL,abs_tol=1e-13):
        raise ValueError("incomplete/inconsistent native spatial tally partition: "+context)


def _global_leakage(statepoint:h5py.File,realizations:int)->float:
    global_tallies=np.asarray(statepoint["global_tallies"][()],dtype=float)
    # OpenMC global tallies order: k-collision, k-absorption,
    # k-tracklength, neutron leakage; second column is cumulative sum.
    if (global_tallies.shape!=(4,3) or not np.isfinite(global_tallies).all()
            or global_tallies[3,1]<=0):
        raise ValueError("missing neutron global leakage transport estimator")
    return float(global_tallies[3,1]/realizations)


def audit_native_neutron_heating(path:str|Path)->dict:
    """Independently inspect actual native HDF5 on all four transport seeds."""
    evidence=audit_real_transport_evidence(path)
    archive=_Bundle(Path(path))
    try:
        heating=archive.report("heating_data_qualification.json")
        economics=archive.report("core_neutron_economy_physical_validation.json")
        summary=archive.report("real_core_transport_summary.json")
        if (heating.get("thermal_power_normalization_allowed") is not False
                or heating.get("all_real_statepoints_have_usable_neutron_local_heating") is not False
                or economics.get("independent_neutron_genealogy_or_spatial_source_stationarity_validated") is not False):
            raise ValueError("actual heating/source qualification report unexpectedly changed")
        # Do not assign arbitrary historical physical reactor power to
        # source-normalized score vectors.
        records=[]
        for sp in evidence["statepoints"]:
            seed=sp["seed"]
            path_in_archive=f"seed_{seed}/statepoint.{sp['batch']:03d}.h5"
            native_bytes=archive.read(path_in_archive)
            if sha256_bytes(native_bytes)!=sp["statepoint_sha256"]:
                raise ValueError("native OpenMC binary statepoint digest mismatch")
            with h5py.File(io.BytesIO(native_bytes),"r") as state:
                n=int(state["n_realizations"][()])
                if n!=sp["active_realizations"]:
                    raise ValueError("native source batches inconsistent")
                t=_native_tallies(state,n)
                leakage=_global_leakage(state,n)
                mesh=t["nerva_3d_neutronics"]
                material=t["nerva_material_transport"]
                energy=t["nerva_material_energy_transport"]
                track=t["nerva_global_reaction_tracklength"]
                analog=t["nerva_global_neutron_balance"]
                # Distinct filters of the SAME domain, not independent
                # measurements: equality checks do not create new information.
                for score in _SCORES:
                    if score in material:
                        _matched(mesh[score]["mean_per_source"],
                                 material[score]["mean_per_source"],
                                 f"{seed}: mesh/material/{score}")
                    if score in energy:
                        _matched(mesh[score]["mean_per_source"],
                                 energy[score]["mean_per_source"],
                                 f"{seed}: mesh/material+energy/{score}")
                for score in ("absorption","nu-fission"):
                    _matched(track[score]["mean_per_source"],
                             mesh[score]["mean_per_source"],
                             f"{seed}: tracklength/mesh/{score}")
                # The thermal energy score is intentionally NOT
                # interchangeable with actual local deposited power.
                for name in (
                    "nerva_3d_neutronics",
                    "nerva_material_transport",
                    "nerva_material_energy_transport",
                ):
                    if not t[name]["heating-local"]["all_moments_zero"]:
                        raise ValueError("real neutron-only local heating unexpectedly nonzero")
                heating_total=mesh["heating"]["mean_per_source"]
                fission_energy=mesh["kappa-fission"]["mean_per_source"]
                if not (math.isfinite(heating_total) and
                        math.isfinite(fission_energy) and
                        heating_total>0 and fission_energy>0):
                    raise ValueError("missing documented nonlocal heating/fission energy signals")
                # Within one eigenvalue generation fission progeny are
                # BANKED for the next cycle; scattering multiplicity can
                # increase *within-cycle* neutrons.
                absorption=analog["absorption"]["mean_per_source"]
                scatter=analog["scatter"]["mean_per_source"]
                nu_scatter=analog["nu-scatter"]["mean_per_source"]
                population_accounting=absorption+leakage-(nu_scatter-scatter)
                _matched(population_accounting,1.,
                         f"{seed}: neutron population conservation")
                records.append({
                    "seed":seed,
                    "native_statepoint_sha256":sp["statepoint_sha256"],
                    "same_domain_mesh_material_energy_cross_scores_verified":True,
                    "global_tracklength_absorption_per_start":track["absorption"]["mean_per_source"],
                    "global_analog_absorption_per_start":absorption,
                    "global_tracklength_nu_fission_per_start":track["nu-fission"]["mean_per_source"],
                    "global_analog_nu_fission_per_start":analog["nu-fission"]["mean_per_source"],
                    "native_neutron_vacuum_leakage_per_start":leakage,
                    "net_multiplicative_scattering_per_start":nu_scatter-scatter,
                    "neutron_transport_population_accounting":population_accounting,
                    "mesh_heating_score_ev_per_source":heating_total,
                    "mesh_kappa_fission_ev_per_source":fission_energy,
                    "local_heating_all_first_and_second_moments_zero":True,
                    "material_separated_local_heating_usable_for_watts":False,
                    "tally_estimator_difference_has_qualified_covariance":False,
                })
        results=np.array([p["neutron_transport_population_accounting"] for p in records])
        if not np.allclose(results,1.,rtol=0,atol=1e-10):
            raise ArithmeticError("transport neutron population accounting failed")
        means={
            key:float(np.mean([row[key] for row in records]))
            for key in (
                "global_tracklength_absorption_per_start",
                "global_analog_absorption_per_start",
                "global_tracklength_nu_fission_per_start",
                "global_analog_nu_fission_per_start",
                "native_neutron_vacuum_leakage_per_start",
                "net_multiplicative_scattering_per_start",
                "mesh_heating_score_ev_per_source",
                "mesh_kappa_fission_ev_per_source",
            )
        }
        return {
            "schema":"nuke-native-openmc-physics-ledger-stage24-v1",
            "evidence_kind":"FOUR_ACTUAL_NEUTRON_ONLY_OPENMC_HDF5_STATEPOINTS",
            "source_transport_summary_sha256":evidence["transport_summary_sha256"],
            "source_artifact_sha256":evidence["archive_sha256"],
            "independent_rng_seed_count":len(records),
            "seed_reaction_scores":records,
            "native_mean_scores_per_source":means,
            "spatially_partitioned_neutron_scores_close":True,
            "native_neutron_population_identity_closes":True,
            "physical_local_heating_qualified":False,
            "hot_core_reactor_power_normalization_allowed":False,
            "source_stationarity_demonstrated":False,
            "historical_nerva_operating_state_validated":False,
            "cautions":[
                "Mesh/material/energy filters of same histories are correlated representations, not independent samples.",
                "Global neutron population closure is a per-generation transport accounting identity, not criticality validation.",
                "Nonzero HEATING or KAPPA-FISSION scores are not accepted as local fuel/hydrogen deposited thermal power.",
                "Four RNG seeds alone do not establish independent fission-source stationarity.",
                "No energy source-rate-to-watts conversion or reactor design change is performed.",
            ],
        }
    finally:
        archive.close()


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=audit_native_neutron_heating(args.evidence)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("REAL transport seeds checked:",report["independent_rng_seed_count"])
    for name,v in report["native_mean_scores_per_source"].items():
        print(name,":",v)
    print("Local neutron/photon heating valid for core watts: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
