"""Validate physically sampled OpenMC materials against executed geometry.

A material declared in materials.xml may be absent from the root-reachable
geometry. This independent read of the *native neutron flux tracklength
tally* identifies which materials neutrons actually traversed. It reconciles
with the separate geometry graph/evaluated MT901 report across all four real
OpenMC seeds and five saved HDF5 checkpoints each.

A positive flux in an UNREACHABLE material is a physical/provenance
contradiction and must raise. Zero flux in a reachable material is a
sampling/coverage gap, not proof that its density/volume is zero. Source
stationarity, physical criticality and operating-temperature energy
deposition remain unqualified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def _sha(file: Path) -> str:
    d = hashlib.sha256()
    with Path(file).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def _text(v) -> str:
    if not isinstance(v, (bytes, np.bytes_)):
        raise ValueError("invalid native OpenMC neutron tally label")
    return bytes(v).decode("utf-8").rstrip("\0")


def real_material_neutron_flux(statepoint: Path, expected_ids: set[int]) -> dict:
    """Read actual native material-filtered neutron flux, not XML metadata."""
    statepoint = Path(statepoint)
    with h5py.File(statepoint, "r") as h:
        if (_text(h.attrs["filetype"]) != "statepoint"
                or _text(h["run_mode"][()]) != "eigenvalue"
                or tuple(int(v) for v in h.attrs["openmc_version"]) != (0,15,3)
                or int(h.attrs["photon_transport"]) != 0
                or int(h.attrs.get("source_present",0)) != 1):
            raise ValueError("requires genuine neutron-only OpenMC 0.15.3 statepoint")
        seed = int(h["seed"][()])
        batch = int(h["current_batch"][()])
        inactive = int(h["n_inactive"][()])
        number = batch - inactive
        if seed <= 0 or number < 2:
            raise ValueError("invalid real OpenMC neutron tally active generations")
        tally = [
            v for k,v in h["tallies"].items()
            if k.startswith("tally ") and isinstance(v,h5py.Group)
            and "name" in v and _text(v["name"][()]) == "nerva_material_transport"
        ]
        if len(tally) != 1:
            raise ValueError("missing/duplicate native material neutron flux tally")
        t = tally[0]
        if (_text(t["estimator"][()]) != "tracklength"
                or int(t["n_filters"][()]) != 1
                or int(t["n_realizations"][()]) != number
                or int(t["n_score_bins"][()]) < 1):
            raise ValueError("wrong OpenMC neutron material flux estimator/realizations")
        ids = np.atleast_1d(t["filters"][()])
        if len(ids) != 1:
            raise ValueError("material flux tally is not singly material-filtered")
        fil = h["tallies"]["filters"].get("filter " + str(int(ids[0])))
        if fil is None or _text(fil["type"][()]) != "material":
            raise ValueError("native flux tally lacks actual OpenMC material filter")
        raw_ids = np.asarray(fil["bins"][()],dtype=np.int64)
        all_ids = [int(x) for x in raw_ids]
        if (not all_ids or any(x<=0 for x in all_ids)
                or len(set(all_ids)) != len(all_ids)
                or set(all_ids) != expected_ids):
            raise ValueError("actual neutron tracklength material IDs disagree with executed materials.xml")
        labels = [_text(v) for v in np.atleast_1d(t["score_bins"][()])]
        if labels.count("flux") != 1 or len(labels) != len(set(labels)):
            raise ValueError("genuine material neutron flux reaction score missing/duplicated")
        if [_text(v) for v in np.atleast_1d(t["nuclides"][()])] != ["total"]:
            raise ValueError("native neutron material flux must cover all nuclides")
        result = np.asarray(t["results"][()],dtype=np.float64)
        if (result.shape != (len(all_ids),len(labels),2)
                or not np.isfinite(result).all()
                or np.any(result[:,:,1] < 0)):
            raise ValueError("invalid genuine OpenMC material flux first/second tally moments")
        j = labels.index("flux")
        first = result[:,j,0]
        second = result[:,j,1]
        # A material absent from transport must have EXACT zero native
        # cumulative flux and squared moment (not epsilon-based screening).
        if np.any(first < 0):
            raise ValueError("negative physical neutron material tracklength")
        if np.any(second*number + 1e-8*np.maximum(1.,second*number)
                  < first*first):
            raise ValueError("nonphysical neutron tally flux second moments")
        flux = {str(mid):float(x/number) for mid,x in zip(all_ids,first)}
        raw = {str(mid):{"raw_flux_sum":float(a),"raw_flux_squared_sum":float(b)}
               for mid,a,b in zip(all_ids,first,second)}
    return {
        "seed":seed,
        "batch":batch,
        "active_tally_realizations":number,
        "hdf5_sha256":_sha(statepoint),
        "material_ids":all_ids,
        "mean_neutron_tracklength_by_material":flux,
        "raw_tracklength_moments_by_material":raw,
    }


def audit_real_material_transport(summary_path: Path, kerma_report_path: Path) -> dict:
    summary_path = Path(summary_path)
    kerma_report_path = Path(kerma_report_path)
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    kerma=json.loads(kerma_report_path.read_text(encoding="utf-8"))
    if (summary.get("execution") != "ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly") != "core"
            or kerma.get("transport_summary_sha256") != _sha(summary_path)
            or kerma.get("real_independent_seed_count") != 4):
        raise ValueError("material neutron flux audit lacks actual four-seed physical provenance")
    active = set(kerma.get("executed_reachable_material_ids", []))
    actual = kerma.get("physical_materials", [])
    unused = kerma.get("uninstantiated_declared_materials", [])
    defined = active | {int(m["material_id"]) for m in unused}
    if (len(active) != kerma.get("physical_material_count")
            or len(defined) != kerma.get("declared_material_count")
            or len(active.intersection({m["material_id"] for m in unused}))
            or not active or len(unused) < 1
            or {int(m["material_id"]) for m in actual} != active):
        raise ValueError("geometry material identities do not match real evaluated-data report")
    batches=summary.get("saved_source_bank_batches")
    total=summary.get("batches")
    runs=summary.get("runs")
    if (not isinstance(batches,list) or len(batches)<3
            or any(type(x) is not int for x in batches)
            or any(a>=b for a,b in zip(batches,batches[1:]))
            or type(total) is not int or batches[-1] != total
            or not isinstance(runs,list) or len(runs)!=4):
        raise ValueError("invalid real OpenMC source checkpoint sequence")
    distinct_seeds=set()
    checkpoints=[]
    all_hashes=set()
    witnessed=set()
    for run in runs:
        seed=run.get("seed")
        final=Path(run.get("statepoint_path",""))
        if (type(seed) is not int or seed <= 0 or seed in distinct_seeds
                or not final.is_file() or _sha(final)!=run.get("statepoint_sha256")):
            raise ValueError("real OpenMC neutron source seed/statepoint provenance mismatch")
        distinct_seeds.add(seed)
        fmt=len(str(total))
        for batch in batches:
            path=final.parent / f"statepoint.{batch:0{fmt}d}.h5"
            if not path.is_file():
                raise ValueError("missing genuine OpenMC neutron flux checkpoint")
            record=real_material_neutron_flux(path,defined)
            if (record["seed"]!=seed or record["batch"]!=batch
                    or record["hdf5_sha256"] in all_hashes):
                raise ValueError("actual HDF5 neutron flux batch/seed identity mismatch")
            all_hashes.add(record["hdf5_sha256"])
            positively_sampled = {
                int(i) for i,v in record["mean_neutron_tracklength_by_material"].items()
                if v > 0
            }
            if positively_sampled - active:
                raise ValueError("neutron flux scored in unreachable/unused OpenMC material")
            for i in defined-active:
                x=record["raw_tracklength_moments_by_material"][str(i)]
                if x["raw_flux_sum"] != 0.0 or x["raw_flux_squared_sum"] != 0.0:
                    raise ValueError("nonzero neutron transport moments in declared but unreachable material")
            witnessed.update(positively_sampled)
            record.update({
                "reachable_material_ids":sorted(active),
                "positively_sampled_reachable_material_ids":sorted(positively_sampled),
                "unplaced_material_ids_with_exact_zero_flux":sorted(defined-active),
                "reachable_material_ids_with_no_observed_flux":sorted(active-positively_sampled),
            })
            checkpoints.append(record)
    status=("ALL_REACHABLE_SAMPLED_AND_UNPLACED_ZERO"
            if witnessed == active and all(
                not row["reachable_material_ids_with_no_observed_flux"]
                for row in checkpoints
            ) else "REACHABLE_MATERIAL_NEUTRON_SAMPLING_INCOMPLETE")
    return {
        "audit":"ACTUAL_OPENMC_NEUTRON_FLUX_VERSUS_ROOT_MATERIAL_PLACEMENT",
        "real_transport_summary_sha256":_sha(summary_path),
        "executed_material_kerma_audit_sha256":_sha(kerma_report_path),
        "four_independent_seed_count":len(distinct_seeds),
        "verified_statepoint_checkpoint_count":len(checkpoints),
        "declared_material_ids":sorted(defined),
        "reachable_material_ids":sorted(active),
        "uninstantiated_material_ids":sorted(defined-active),
        "observed_reachable_material_ids_with_nonzero_flux":sorted(witnessed),
        "all_unplaced_materials_have_exact_zero_neutron_tracklength":True,
        "material_transport_support_status":status,
        "genuine_checkpoint_material_flux":checkpoints,
        "local_nuclear_heat_deposition_qualified":False,
        "criticality_or_fission_source_convergence_validated":False,
        "nasa_nerva_core_physically_validated":False,
        "interpretation":(
            "Native OpenMC tracklength scores from 20 statepoint checkpoints "
            "confirm which placed physical materials were actually traversed "
            "by neutrons. Defined-but-unreachable materials must be exactly zero "
            "in genuine tally moments. A reachable material with no observed "
            "tracklength is a source-sampling gap, not an assertion of zero "
            "physical geometry volume or isotope density. This does not prove "
            "source convergence or valid hot local nuclear-energy deposition."
        ),
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--material-kerma",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    report=audit_real_material_transport(args.summary,args.material_kerma)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Actual OpenMC root/material neutron tracklength support:",
          report["material_transport_support_status"])
    print("Actual visited materials:",report["observed_reachable_material_ids_with_nonzero_flux"])
    print("Unplaced nuclear material definitions with exactly zero flux:",
          report["uninstantiated_material_ids"])
    print("NASA NERVA physical qualification: NOT VALIDATED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
