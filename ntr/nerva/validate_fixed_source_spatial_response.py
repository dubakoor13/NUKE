"""Adversarial numerical fixtures; these are NOT actual OpenMC executions."""
from __future__ import annotations

import numpy as np

from .fixed_source_spatial_response import reconcile_axial_three_d, SCORES


def _reject(callback, contains: str) -> None:
    try:
        callback()
    except ValueError as exc:
        assert contains in str(exc), (str(exc), contains)
    else:
        raise AssertionError("physically invalid neutron mesh data accepted")


def main() -> int:
    axial_z = np.asarray([1.,2.,3.,4.,4.,3.,2.,1.])
    axial = np.column_stack((axial_z, axial_z*.001, axial_z*.0024))
    volume = np.tile(axial.sum(axis=0)/64., (64,1))
    report = reconcile_axial_three_d(axial,volume)
    for score in SCORES:
        assert report[score]["relative_mesh_integral_error"] < 1e-14
        assert report[score]["axial_mirror_total_variation"] < 1e-14
        assert abs(sum(report[score]["normalized_axial_distribution"])-1.) < 1e-14
    off_center = axial.copy()
    off_center[0,0] += .5
    result = reconcile_axial_three_d(
        off_center,np.tile(off_center.sum(axis=0)/64.,(64,1)))
    assert result["flux"]["axial_mirror_total_variation"] > 0
    wrong = volume.copy()
    wrong[0,2] += 0.0001
    _reject(lambda:reconcile_axial_three_d(axial,wrong),
            "mesh integrals do not close")
    nonfinite = axial.copy()
    nonfinite[1,1] = np.nan
    _reject(lambda:reconcile_axial_three_d(nonfinite,volume),
            "invalid neutron spatial")
    nonphysical = axial.copy()
    nonphysical[1,1] = -1.
    _reject(lambda:reconcile_axial_three_d(nonphysical,volume),
            "invalid neutron spatial")
    print("Cold-core fixed-source 3D/axial neutron mesh physics: PASS (synthetic fixtures)")
    print("Actual neutron transport requires the separate compiled OpenMC workflow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
