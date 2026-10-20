"""Compare two independently sampled OpenMC heating replicas, optionally
propagating spatial heating differences into the same resolved thermal case.

For real data provide the two power subdirectories from run_analysis
(each with nerva_mesh_fields.npz and metadata.json), together with
their respective transport_provenance.json files and original statepoints.

Synthetic demos require explicit --allow-synthetic and produce no
claims about actual OpenMC transport.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .openmc_replica_convergence import (
    _check_transport_identity, compare_mesh_replicas,
    load_mesh_replica, thermal_replica_consequence,
)


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--power-a",type=Path,required=True)
    p.add_argument("--power-b",type=Path,required=True)
    p.add_argument("--provenance-a",type=Path)
    p.add_argument("--provenance-b",type=Path)
    p.add_argument("--allow-synthetic",action="store_true")
    p.add_argument("--maximum-total-variation",type=float,default=0.05)
    p.add_argument("--maximum-axial-difference",type=float,default=0.05)
    p.add_argument("--maximum-significant-voxel-fraction",type=float,default=0.10)
    p.add_argument("--significance-sigma",type=float,default=3.)
    p.add_argument("--fuel-a",type=Path)
    p.add_argument("--tie-a",type=Path)
    p.add_argument("--fuel-b",type=Path)
    p.add_argument("--tie-b",type=Path)
    p.add_argument("--thermal-case",type=Path,
                   help="JSON of prescribed thermal parameters and fixed timestep.")
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_openmc_replica_convergence.json"))
    return p


def main() -> int:
    args=parser().parse_args()
    if args.allow_synthetic and (args.provenance_a or args.provenance_b):
        raise ValueError("synthetic comparisons cannot claim statepoint provenance")
    if not args.allow_synthetic and not (args.provenance_a and args.provenance_b):
        raise ValueError("real comparison requires BOTH transport provenance files")
    first,pa=load_mesh_replica(
        args.power_a,provenance_path=args.provenance_a,
        allow_synthetic=args.allow_synthetic,
    )
    second,pb=load_mesh_replica(
        args.power_b,provenance_path=args.provenance_b,
        allow_synthetic=args.allow_synthetic,
    )
    if not args.allow_synthetic:
        _check_transport_identity(pa,pb)
    comparison=compare_mesh_replicas(
        first,second,
        max_total_variation=args.maximum_total_variation,
        max_axial_fraction_difference=args.maximum_axial_difference,
        max_significant_fraction=args.maximum_significant_voxel_fraction,
        significance_sigma=args.significance_sigma,
    )
    data={
        "source_mode": "synthetic_unverified" if args.allow_synthetic
                       else "independent_openmc_provenance_checked",
        "replica_a":{
            "power_directory":str(first.directory),
            "statepoint_sha256":first.statepoint_sha256,
            "seed":first.openmc_seed,
            "nuclear_data_sha256":first.nuclear_data_sha256,
        },
        "replica_b":{
            "power_directory":str(second.directory),
            "statepoint_sha256":second.statepoint_sha256,
            "seed":second.openmc_seed,
            "nuclear_data_sha256":second.nuclear_data_sha256,
        },
        "mesh_comparison":comparison,
        "thermal_comparison":None,
        "real_openmc_executed_in_comparison":False,
        "historical_nerva_validation":False,
    }
    thermal_inputs=(args.fuel_a,args.tie_a,args.fuel_b,args.tie_b,args.thermal_case)
    if any(v is not None for v in thermal_inputs):
        if not all(v is not None for v in thermal_inputs):
            raise ValueError("thermal comparison requires all 4 reconstruction files and case JSON")
        case=json.loads(args.thermal_case.read_text(encoding="utf-8"))
        data["thermal_comparison"]=thermal_replica_consequence(
            args.power_a/"nerva_mesh_fields.npz",
            args.fuel_a,args.tie_a,
            args.power_b/"nerva_mesh_fields.npz",
            args.fuel_b,args.tie_b,case
        )
    passed=comparison["pass"] and (
        data["thermal_comparison"] is None
        or data["thermal_comparison"]["status"] == "PASS"
    )
    data["status"]="PASS" if passed else "FAIL"
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    print("Independent normalized OpenMC heating replica assessment: "+data["status"])
    print(f"  total-variation spatial-power fraction: {comparison['total_variation_power_fraction']:.5g}")
    print(f"  maximum axial power-fraction shift: {comparison['max_axial_power_fraction_difference']:.5g}")
    print(f"  fraction above {args.significance_sigma:g} nominal sigma: "
          f"{comparison['fraction_active_voxels_above_sigma']:.5g}")
    if data["thermal_comparison"]:
        print("  maximum thermal-state difference: "
              f"{data['thermal_comparison']['max_resolved_temperature_difference_k']:.5g} K")
    print(f"  result: {args.output}")
    return 0 if passed else 2


if __name__=="__main__":
    raise SystemExit(main())
