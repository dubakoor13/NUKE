"""Synthetic mathematics tests of independent-seed OpenMC neutron reactions.

These arrays are deliberately not physical Monte Carlo outputs. Genuine
statepoint and evaluated cross-section checks run in the real OpenMC job.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from .core_reaction_ensemble import (
    analyze_reaction_replica_statistics, audit_real_neutron_reactions
)


def reject(action, needle):
    try:
        action()
    except (ValueError, KeyError) as exc:
        assert needle in str(exc), (needle, str(exc))
    else:
        raise AssertionError("invalid reactor neutron reaction ensemble accepted")


def fixture():
    materials=[1,2,3]
    bins=[[1.e-5,0.625],[0.625,1e5],[1e5,2e7]]
    out=[]
    for seed,shift in zip((13007,23011,33013,43019),
                           (0.,.03,-.02,.01)):
        absorption=0.040 + 0.002*shift
        fission=0.012 + 0.001*shift
        a_spectrum=[.2+.25*shift,.3,.5-.25*shift]
        f_spectrum=[.3-.1*shift,.3+.2*shift,.4-.1*shift]
        out.append({
            "seed":seed,
            "material_ids":materials,
            "incident_energy_bins_eV":bins,
            "scores":{
                "absorption":{
                    "global_per_source":absorption,
                    "material_per_source":[absorption*.75,absorption*.15,absorption*.10],
                    "incident_energy_per_source":[absorption*v for v in a_spectrum],
                    "normalized_incident_energy_spectrum":a_spectrum,
                },
                "nu-fission":{
                    "global_per_source":fission,
                    "material_per_source":[fission,0.,0.],
                    "incident_energy_per_source":[fission*v for v in f_spectrum],
                    "normalized_incident_energy_spectrum":f_spectrum,
                }
            }
        })
    return out


def main():
    rows=fixture()
    report=analyze_reaction_replica_statistics(rows)
    assert report["independent_seed_count"] == 4
    assert report["seeds"] == [13007,23011,33013,43019]
    assert report["independent_fission_source_stationarity_demonstrated"] is False
    assert report["nasa_historical_nerva_neutron_physics_validated"] is False
    assert math.isclose(report["reactions"]["absorption"]["mean_global_reaction_per_source"],
                        sum(x["scores"]["absorption"]["global_per_source"]
                            for x in rows)/4,rel_tol=1e-12)
    assert report["reactions"]["absorption"]["mean_pairwise_incident_energy_spectrum_tv"] > 0
    assert report["reactions"]["nu-fission"]["max_pairwise_incident_energy_spectrum_tv"] > 0
    assert report["reactions"]["nu-fission"]["material_reaction_partition"][1][
        "mean_fraction_of_global_reaction"] == 0

    reject(lambda:analyze_reaction_replica_statistics(rows[:3]),"four")
    duplicated=deepcopy(rows)
    duplicated[-1]["seed"] = rows[0]["seed"]
    reject(lambda:analyze_reaction_replica_statistics(duplicated),"duplicate")
    different_model=deepcopy(rows)
    different_model[0]["material_ids"]=[1,3,2]
    reject(lambda:analyze_reaction_replica_statistics(different_model),"filters changed")
    different_energy=deepcopy(rows)
    different_energy[-1]["incident_energy_bins_eV"][1][0]=.66
    reject(lambda:analyze_reaction_replica_statistics(different_energy),"filters changed")
    negative=deepcopy(rows)
    negative[1]["scores"]["nu-fission"]["material_per_source"][0]=-0.01
    reject(lambda:analyze_reaction_replica_statistics(negative),"nonconservative")
    nonfinite=deepcopy(rows)
    nonfinite[0]["scores"]["absorption"]["global_per_source"]=float("nan")
    reject(lambda:analyze_reaction_replica_statistics(nonfinite),"nonconservative")
    mismatch=deepcopy(rows)
    mismatch[2]["scores"]["absorption"]["incident_energy_per_source"][0]=0.
    reject(lambda:analyze_reaction_replica_statistics(mismatch),"nonconservative")
    wrong_spectrum=deepcopy(rows)
    wrong_spectrum[3]["scores"]["nu-fission"][
        "normalized_incident_energy_spectrum"]=[.1,.2,.3]
    reject(lambda:analyze_reaction_replica_statistics(wrong_spectrum),"nonconservative")

    with TemporaryDirectory() as d:
        bad=Path(d)/"fake.json"
        bad.write_text(json.dumps({"execution":"SYNTHETIC","runs":rows}))
        reject(lambda:audit_real_neutron_reactions(bad),"genuine")

    print("Four-seed material/incident-energy neutron absorption/fission numerics: PASS")
    print("  pure synthetic reaction arrays; real OpenMC integration runs separately")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
