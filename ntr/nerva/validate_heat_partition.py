"""Pure numerical regression for spatial OpenMC heating conservation.

Synthetic arrays only; NOT real neutron transport or NASA validation.
"""
from __future__ import annotations

import math
import numpy as np

from .heat_partition import audit_heat_partition


def rejects(fn, expected: str):
    try:
        fn()
    except ValueError as exc:
        if expected not in str(exc):
            raise AssertionError(f"wrong rejection: {exc!r}") from exc
    else:
        raise AssertionError(f"expected heat-budget rejection: {expected}")


def main() -> int:
    mesh = np.array([[[10., 20.], [15., 30.]]])
    fuel = mesh * .55
    tie = mesh * .12
    fuel_h2 = mesh * .09
    tie_h2 = mesh * .07
    supply = tie_h2 * .4
    ret = tie_h2 * .6
    args = (mesh, fuel, tie, fuel_h2, tie_h2, supply, ret, 0.5)
    good = audit_heat_partition(*args)
    assert good["passed"] and good["mesh_total_w"] == 37.5
    assert math.isclose(good["remaining_unassigned_fraction"], 0.17, abs_tol=1e-12)
    assert good["maximum_tie_supply_return_residual_w_cm3"] < 1e-14

    # An overlap or double-counted coolant heat is a LOCAL error, even
    # when another voxel has plenty of unassigned power to cancel it.
    bad_fuel = fuel.copy()
    bad_fuel[0, 0, 0] = 11.
    rejects(lambda: audit_heat_partition(
        mesh, bad_fuel, tie, fuel_h2, tie_h2, supply, ret, .5
    ), "overcounts")
    bad_supply = supply.copy()
    bad_supply[0, 1, 0] += 0.1
    rejects(lambda: audit_heat_partition(
        mesh, fuel, tie, fuel_h2, tie_h2, bad_supply, ret, .5
    ), "supply + return")

    bad_mesh = mesh.copy()
    bad_mesh[0, 0, 0] = float("nan")
    rejects(lambda: audit_heat_partition(
        bad_mesh, fuel, tie, fuel_h2, tie_h2, supply, ret, .5
    ), "nonfinite")
    bad_tie = tie.copy()
    bad_tie[0, 0, 1] = -0.1
    rejects(lambda: audit_heat_partition(
        mesh, fuel, bad_tie, fuel_h2, tie_h2, supply, ret, .5
    ), "negative")
    rejects(lambda: audit_heat_partition(
        mesh, fuel[..., :-1], tie, fuel_h2, tie_h2, supply, ret, .5
    ), "dimensions")
    rejects(lambda: audit_heat_partition(*args[:-1], -1.), "volume")
    rejects(lambda: audit_heat_partition(*args, relative_tolerance=float("nan")),
            "tolerance")
    print("NERVA conservative OpenMC voxelwise heat-budget regression: PASS")
    print("  actual OpenMC neutron/photon transport executed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
