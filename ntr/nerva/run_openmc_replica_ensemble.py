"""Assess >=4 independent OpenMC normalized heating maps.

Real mode requires a paired normalized power directory and transport
provenance file from each separately seeded OpenMC statepoint. Synthetic
mode is explicitly labeled; it does not execute neutron transport.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .openmc_replica_ensemble import evaluate_ensemble_directories


def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--power-dirs",nargs="+",type=Path,required=True)
    p.add_argument("--provenance-files",nargs="+",type=Path)
    p.add_argument("--allow-synthetic",action="store_true")
    p.add_argument("--max-p95-relative-sem",type=float,default=0.10)
    p.add_argument("--max-axial-fraction-sem",type=float,default=0.01)
    p.add_argument("--max-half-split-total-variation",type=float,default=0.05,
                   help="Maximum allowed worst balanced split TV (exact for <=12 replicas).")
    p.add_argument("--max-leave-one-out-total-variation",type=float,default=0.05,
                   help="Maximum allowed deviation of any replica from all others.")
    p.add_argument("--active-threshold-relative",type=float,default=1e-5)
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_openmc_replica_ensemble"))
    return p


def main() -> int:
    a=build_parser().parse_args()
    summary,data,identities=evaluate_ensemble_directories(
        a.power_dirs,provenance_paths=a.provenance_files,
        allow_synthetic=a.allow_synthetic,
        max_p95_relative_sem=a.max_p95_relative_sem,
        max_axial_fraction_sem=a.max_axial_fraction_sem,
        max_half_split_total_variation=a.max_half_split_total_variation,
        max_leave_one_out_total_variation=a.max_leave_one_out_total_variation,
        active_threshold_relative=a.active_threshold_relative,
    )
    out=a.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"ensemble_heating_statistics.npz",**data)
    report={
        "source_mode":"synthetic_unverified" if a.allow_synthetic
                       else "verified_openmc_file_provenance",
        "historical_nerva_validation":False,
        "real_openmc_executed_by_this_command":False,
        "summary":asdict(summary),
        "replicas":identities,
        "fields": {
            key:"ensemble_heating_statistics.npz::"+key
            for key in data
        },
    }
    (out/"ensemble_convergence_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    print("NERVA independent spatial-heating ensemble: "+summary.status)
    print(f"  replicas: {summary.replica_count}")
    print(f"  p95 relative sample SEM: {summary.p95_active_relative_sem:.6g}")
    print(f"  max axial fraction SEM: {summary.max_axial_fraction_sem:.6g}")
    print(f"  original-order half-split TV (diagnostic only): {summary.half_split_total_variation:.6g}")
    print(f"  worst balanced TV (exhaustive <=12): {summary.maximum_balanced_split_total_variation}")
    print(f"  tested balanced TV ({summary.balanced_split_method}): "
          f"{summary.tested_balanced_split_total_variation:.6g}")
    print(f"  balanced split partitions tested: {summary.balanced_splits_tested}")
    print(f"  maximum leave-one-out TV: {summary.maximum_leave_one_out_total_variation:.6g}")
    print(f"  report: {out/'ensemble_convergence_report.json'}")
    return 0 if summary.status=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
