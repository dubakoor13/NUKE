"""Genuine OpenMC active-generation NEUTRON conservation including leakage.

The native `global_tallies` HDF5 array stores cumulative analog-vacuum
leakage and three k estimators. Its ordering is OpenMC 0.15.3 native
(k-collision, k-absorption, k-tracklength, leakage); require the native
k-tracklength cumulative first AND second moments to exactly match the
independently named unfiltered TRACKLENGTH nu-fission tally. This guards
against accidentally using wrong unlabeled native global-tally rows.

For each saved statepoint, read the ANALOG absorption, scatter and
nu-scatter cumulative sums and sum-of-squares. Subtract consecutive
checkpoints to reconstruct disjoint active-generation windows and test

    absorption + native neutron leakage - (nu-scatter - scatter) = 1

on the original eigenvalue source-neutron-normalized generation basis.
Fission daughters feed NEXT-generation source; do not count them as live
scatter production in the same generation.

This is native simulated-neutron numerical conservation, NOT nuclear data
qualification, independent fission ancestry, historical NASA NERVA
benchmark agreement or actual NERVA criticality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .core_reaction_history import audit_real_reaction_windows

ANALOG_SCORES = ("absorption", "scatter", "nu-scatter", "nu-fission")
NATIVE_K_LABELS = ("k-collision", "k-absorption", "k-tracklength")
MAX_PARTICLE_BALANCE_RESIDUAL = 1e-7


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _hdf_text(value) -> str:
    if not isinstance(value, bytes):
        raise ValueError("native OpenMC HDF5 label is not a byte string")
    return value.decode("utf-8").rstrip("\x00")


def _named_unfiltered(tallies, name: str, estimator: str) -> tuple[np.ndarray,int]:
    matching=[]
    for key,obj in tallies.items():
        if (key.startswith("tally ") and isinstance(obj,h5py.Group)
                and "name" in obj and _hdf_text(obj["name"][()]) == name):
            matching.append(obj)
    if len(matching)!=1:
        raise ValueError("required whole-geometry OpenMC tally missing or duplicated: "+name)
    obj=matching[0]
    if (_hdf_text(obj["estimator"][()])!=estimator or
            int(obj["n_filters"][()]) != 0):
        raise ValueError("wrong physical OpenMC estimator or clipped global neutron tally")
    n=int(obj["n_realizations"][()])
    scores=tuple(_hdf_text(x) for x in obj["score_bins"][()])
    if len(set(scores))!=len(scores):
        raise ValueError("duplicate OpenMC score names in genuine tally")
    data=np.asarray(obj["results"][()],dtype=float)
    if (n < 2 or data.shape != (1,len(scores),2)
            or not np.isfinite(data).all() or np.any(data < 0)):
        raise ValueError("invalid cumulative neutron reaction moments")
    return data[0],n


def read_native_neutron_checkpoint(path: Path) -> dict:
    path=Path(path)
    try:
        with h5py.File(path,"r") as h:
            if (_hdf_text(h.attrs["filetype"])!="statepoint"
                    or _hdf_text(h["run_mode"][()])!="eigenvalue"
                    or int(h.attrs["photon_transport"])!=0):
                raise ValueError("requires real neutron-only eigenvalue HDF5 statepoint")
            seed=int(h["seed"][()])
            batch=int(h["current_batch"][()])
            inactive=int(h["n_inactive"][()])
            population=int(h["n_particles"][()])
            active=int(h["n_realizations"][()])
            if (seed < 1 or population < 2 or batch<=inactive
                    or active!=batch-inactive or active<2):
                raise ValueError("invalid source population or completed neutron generations")
            native=np.asarray(h["global_tallies"][()],dtype=float)
            if (native.shape!=(4,3) or not np.isfinite(native).all()
                    or np.any(native[:,1:] < 0)
                    or np.any(np.abs(native[:,0])>1e-12)):
                raise ValueError("unexpected genuine OpenMC native global tallies")
            # Names in HDF5 belong to the tallies; validate positions by
            # matching the independently exported scored k-tracklength.
            tally_items={}
            for name,estim in (
                ("nerva_global_neutron_balance","analog"),
                ("nerva_global_reaction_tracklength","tracklength"),
            ):
                obj=[x for key,x in h["tallies"].items()
                     if key.startswith("tally ") and isinstance(x,h5py.Group)
                     and "name" in x and _hdf_text(x["name"][()])==name]
                if len(obj)!=1:
                    raise ValueError("missing genuine global neutron score tally")
                t=obj[0]
                if (_hdf_text(t["estimator"][()])!=estim or
                        int(t["n_filters"][()])!=0 or
                        int(t["n_realizations"][()])!=active):
                    raise ValueError("mismatched whole-core estimator and active generations")
                names=tuple(_hdf_text(x) for x in t["score_bins"][()])
                required=ANALOG_SCORES if estim=="analog" else ("absorption","nu-fission")
                if len(set(names))!=len(names) or any(s not in names for s in required):
                    raise ValueError("missing physical neutron absorption/multiplicity/fission scores")
                values=np.asarray(t["results"][()],dtype=float)
                if (values.shape!=(1,len(names),2) or
                        not np.isfinite(values).all() or np.any(values<0)):
                    raise ValueError("nonphysical cumulative neutron reaction sums")
                tally_items[estim]={
                    s:values[0,names.index(s),:] for s in required
                }
            expected=tally_items["tracklength"]["nu-fission"]
            if not np.allclose(native[2,1:], expected,rtol=2e-10,atol=2e-12):
                raise ValueError("native OpenMC k-tracklength row does not equal scored nu-fission")
            rows=np.stack([
                tally_items["analog"][score] for score in ANALOG_SCORES
            ] + [native[3,1:]])
    except (KeyError,OSError,TypeError) as err:
        raise ValueError("incomplete native neutron-only OpenMC HDF5 statepoint") from err
    return {
        "seed":seed,"batch":batch,"inactive":inactive,
        "active_realizations":active,"particles":population,
        "sha256":_sha(path),
        "cumulative_analog_abs_scatter_nuscatter_nufission_leakage":rows,
        "cumulative_native_k":native[:3,1:],
    }


def check_native_neutron_windows(
    rows: list[dict], *, closure_tolerance: float = MAX_PARTICLE_BALANCE_RESIDUAL
) -> dict:
    if (len(rows)<3 or not math.isfinite(closure_tolerance)
            or closure_tolerance<=0 or closure_tolerance>1e-4):
        raise ValueError("three physical checkpoint observations and strict closure required")
    if (len({r["seed"] for r in rows})!=1
            or len({r["particles"] for r in rows})!=1
            or len({r["inactive"] for r in rows})!=1):
        raise ValueError("mixed nuclear transport seeds or active-generation settings")
    before=np.zeros((5,2))
    prev_n=0
    last_batch=rows[0]["inactive"]
    windows=[]
    for row in rows:
        n=row["active_realizations"]
        width=n-prev_n
        raw=np.asarray(row["cumulative_analog_abs_scatter_nuscatter_nufission_leakage"])
        if (width<2 or row["batch"]<=last_batch
                or row["batch"]-row["inactive"]!=n
                or width != row["batch"]-last_batch):
            raise ValueError("invalid active-generation neutron window schedule")
        if raw.shape!=(5,2) or not np.isfinite(raw).all():
            raise ValueError("malformed HDF5 analog/native leakage cumulative moments")
        delta=raw-before
        if np.any(delta< -1e-9):
            raise ValueError("negative disjoint neutron balance accumulator")
        rate=delta[:,0]/width
        m2=delta[:,1]/width
        var=(m2-rate**2)/(width-1)
        if (np.any(rate<0) or not np.isfinite(var).all()
                or np.any(var< -1e-11)):
            raise ValueError("invalid neutron leakage/reaction per-generation moment statistics")
        sem=np.sqrt(np.maximum(var,0))
        absorption,scatter,nu_scatter,nu_fission,leakage=map(float,rate)
        if absorption<=0 or leakage<=0 or nu_fission<=0:
            raise ValueError("nonpositive physical neutron absorption, fission or leakage")
        multiplication=nu_scatter-scatter
        closure=absorption+leakage-multiplication
        residual=closure-1.
        windows.append({
            "first_active_batch":last_batch+1,
            "last_active_batch":row["batch"],
            "active_realizations":width,
            "analog_absorption_per_source":absorption,
            "native_vacuum_leakage_per_source":leakage,
            "net_multiplicative_scattering_per_source":multiplication,
            "analog_nu_fission_per_source":nu_fission,
            "analog_scatter_per_source":scatter,
            "analog_nu_scatter_per_source":nu_scatter,
            "conservation_balance_per_source":closure,
            "signed_neutron_conservation_residual":residual,
            "neutron_balance_strict_physical_screen_passed":
                bool(abs(residual)<=closure_tolerance),
            "per_batch_marginal_SEM":{
                k:float(sem[j]) for j,k in enumerate(
                    ("absorption","scatter","nu-scatter","nu-fission","leakage")
                )
            },
            "statepoint_sha256":row["sha256"],
        })
        prev_n=n
        last_batch=row["batch"]
        before=raw
    reconstructed=sum(
        v["active_realizations"]*v["conservation_balance_per_source"]
        for v in windows)/prev_n
    cumulative=rows[-1]["cumulative_analog_abs_scatter_nuscatter_nufission_leakage"]
    final=cumulative[:,0]/prev_n
    if (not math.isclose(reconstructed,float(final[0]+final[4]-final[2]+final[1]),
                         rel_tol=0,abs_tol=5e-12)):
        raise ValueError("disjoint neutron balance fails to reconstruct the final OpenMC statepoint")
    return {
        "seed":rows[0]["seed"],
        "source_particles_per_generation":rows[0]["particles"],
        "inactive_generations":rows[0]["inactive"],
        "final_statepoint_sha256":rows[-1]["sha256"],
        "final_cumulative_balance_per_source":float(final[0]+final[4]-final[2]+final[1]),
        "disjoint_windows":windows,
        "maximum_absolute_disjoint_neutron_conservation_residual":
            max(abs(w["signed_neutron_conservation_residual"]) for w in windows),
        "all_disjoint_neutron_conservation_windows_passed":
            all(w["neutron_balance_strict_physical_screen_passed"] for w in windows),
    }


def audit_real_neutron_window_balance(summary_path: Path) -> dict:
    path=Path(summary_path)
    # This independently authenticates the model XML, the evaluated data
    # index, four seeds, all physical statepoint hashes and tracklength
    # reaction/incident-neutron-energy partitions for every checkpoint.
    reaction_history=audit_real_reaction_windows(path)
    if not reaction_history["all_reaction_window_partition_and_reconstruction_checks_passed"]:
        raise ValueError("core reaction reference fails physical conservation")
    summary=json.loads(path.read_text(encoding="utf-8"))
    stages=summary["saved_source_bank_batches"]
    digits=len(str(summary["batches"]))
    per_seed=[]
    for run in summary["runs"]:
        final=Path(run["statepoint_path"])
        if not final.is_file() or _sha(final)!=run["statepoint_sha256"]:
            raise ValueError("final native neutron leakage HDF5 provenance changed")
        files=[final.parent/f"statepoint.{batch:0{digits}d}.h5" for batch in stages]
        if files[-1].resolve()!=final.resolve():
            raise ValueError("native leakage and neutron scores use different statepoint")
        rows=[read_native_neutron_checkpoint(f) for f in files]
        if (any(x["seed"]!=run["seed"] for x in rows)
                or any(x["batch"]!=b for x,b in zip(rows,stages))):
            raise ValueError("source seed or real checkpoint batch mismatch")
        out=check_native_neutron_windows(rows)
        reference=next((entry for entry in reaction_history["observed_seed_history"]
                        if entry["seed"]==run["seed"]),None)
        if reference is None or len(reference["windows"]) != len(out["disjoint_windows"]):
            raise ValueError("independent source and neutron production windows differ")
        for w,ref in zip(out["disjoint_windows"],reference["windows"]):
            if w["last_active_batch"]!=ref["last_active_batch"]:
                raise ValueError("independent neutron reaction temporal index mismatch")
        previous=run["global_neutron_balance"]
        if (not math.isclose(
                out["final_cumulative_balance_per_source"],
                previous["balance_disappearance_per_source"],
                rel_tol=0,abs_tol=3e-12)):
            raise ValueError("native leakage disjoint balance differs from full real OpenMC report")
        per_seed.append(out)
    if len(per_seed)<4 or len({x["seed"] for x in per_seed})!=len(per_seed):
        raise ValueError("missing independent real OpenMC neutron sources")
    every=all(z["all_disjoint_neutron_conservation_windows_passed"] for z in per_seed)
    return {
        "analysis":"ACTUAL_OPENMC_GLOBAL_NEUTRON_LEAKAGE_ACTIVE_WINDOW_BALANCE",
        "real_transport_summary_sha256":_sha(path),
        "independent_seed_count":len(per_seed),
        "saved_source_batches":stages,
        "source_normalization":"one starting source neutron per active eigenvalue generation",
        "native_global_tally_row_semantics_proven_against_tracklength_nu_fission":True,
        "per_seed_neutron_conservation":per_seed,
        "max_absolute_neutron_conservation_residual":
            max(x["maximum_absolute_disjoint_neutron_conservation_residual"]
                for x in per_seed),
        "all_windows_physical_neutron_balance_passed":bool(every),
        "physical_nasa_nerva_core_validated":False,
        "independent_fission_source_stationarity_demonstrated":False,
        "interpretation":(
            "Observed neutron analog absorption and multiplicative scattering "
            "plus OpenMC native vacuum escape exactly conserve the original "
            "normalized transported source neutron within each disjoint "
            "active-generation period. Leakage may exceed one due to "
            "same-generation neutron multiplication. Native HDF5 row indices "
            "were crosschecked against independently scored nu-fission "
            "cumulative first and second moments. Unknown score covariances "
            "and correlated eigenvalue source ancestry preclude independent-"
            "history variance inference; this conservation law is necessary "
            "software physics accounting, NOT nuclear-material evaluation "
            "or NASA NERVA benchmark validation."
        ),
    }


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    a=parser.parse_args()
    result=audit_real_neutron_window_balance(a.summary)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("Genuine OpenMC neutron leakage/absorption/scattering disjoint-window balance:",
          "PASS" if result["all_windows_physical_neutron_balance_passed"] else "FAIL")
    print("Max neutron-balance residual:",
          result["max_absolute_neutron_conservation_residual"])
    print("Independent source stationarity and NASA historic core: NOT VALIDATED")
    return 0 if result["all_windows_physical_neutron_balance_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
