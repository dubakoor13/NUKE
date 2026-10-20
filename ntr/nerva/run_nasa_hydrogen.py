"""Command-line research runner for NASA-7 NERVA hot-H2 nozzle comparisons.

This is NOT NASA CEA or a validated XE Prime/1972 engine reconstruction.
"""
import argparse
import json
from dataclasses import asdict
from .nasa_hydrogen import hydrogen_nozzle, power_coupled_hydrogen_nozzle


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pc-mpa", type=float, default=3.10264)
    p.add_argument("--tc-k", type=float, default=2425.0)
    p.add_argument("--at-m2", type=float, required=True,
                   help="Throat area is an independent geometric input (m²)")
    p.add_argument("--area-ratio", type=float, default=100.0)
    p.add_argument("--mode", choices=["frozen", "equilibrium"], default="equilibrium")
    p.add_argument("--nozzle-efficiency", type=float, default=0.95)
    p.add_argument("--power-mw", type=float,
                   help="If set, solve T from power balance rather than use --tc-k")
    p.add_argument("--inlet-k", type=float, default=300.0)
    args = p.parse_args()
    if args.power_mw is None:
        result = hydrogen_nozzle(args.pc_mpa*1e6, args.tc_k, args.at_m2,
            args.area_ratio, mode=args.mode, nozzle_efficiency=args.nozzle_efficiency)
    else:
        result = power_coupled_hydrogen_nozzle(
            args.power_mw*1e6, args.pc_mpa*1e6, args.inlet_k,
            args.at_m2, args.area_ratio, mode=args.mode,
            nozzle_efficiency=args.nozzle_efficiency)
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
