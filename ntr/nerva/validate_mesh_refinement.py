"""Synthetic exact-mesh refinement regression. No OpenMC transport run."""
from __future__ import annotations

from dataclasses import replace
import math
import numpy as np

from .mesh_refinement import HeatingMesh, compare_heating_meshes


def reject(fn, phrase):
    try:
        fn()
    except ValueError as exc:
        assert phrase in str(exc), str(exc)
    else:
        raise AssertionError("expected rejection: " + phrase)


def main() -> int:
    lo = np.array([-2., -2., -2.])
    hi = np.array([2., 2., 2.])
    coarse = np.arange(8, dtype=float).reshape(2, 2, 2) + 5.
    voxel = 8.
    power = float(coarse.sum() * voxel)
    a = HeatingMesh(coarse, lo, hi, power, "heating-local", False)
    fine = np.repeat(
        np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1),
        2, axis=2
    )
    b = HeatingMesh(fine, lo, hi, power, "heating-local", False)
    same = compare_heating_meshes(a, b)
    assert same["comparison_type"] == "conservative_nested_mesh_projection"
    assert same["refinement_factors"] == [2, 2, 2]
    assert same["common_coarse_shape"] == [2, 2, 2]
    assert same["relative_l2_power_density_difference"] < 1e-14
    assert same["total_variation_power_fraction"] < 1e-14
    assert same["fine_to_coarse_power_integral_relative_error"] < 1e-14
    assert same["fine_subvoxel_unresolved_relative_l2"] < 1e-14
    backward = compare_heating_meshes(b, a)
    assert backward["relative_l2_power_density_difference"] < 1e-14

    # Opposing fine-voxel power shifts keep each parent's volume-average
    # identical, but represent unresolved local heating within that cell.
    shifted = fine.copy()
    shifted[::2, :, :] *= 1.1
    shifted[1::2, :, :] *= .9
    subvoxel = compare_heating_meshes(a, replace(b, density_w_cm3=shifted))
    assert subvoxel["relative_l2_power_density_difference"] < 1e-13
    assert subvoxel["fine_subvoxel_unresolved_relative_l2"] > 0.09
    assert subvoxel["total_variation_power_fraction"] < 1e-13

    # Genuine coarse-scale spatial bias must be exposed.
    biased = coarse.copy()
    biased[0, 0, 0] += .5
    biased[1, 1, 1] -= .5
    changed = compare_heating_meshes(
        a, replace(a, density_w_cm3=biased)
    )
    assert changed["comparison_type"] == "same_grid"
    assert changed["relative_l2_power_density_difference"] > 0
    assert changed["total_variation_power_fraction"] > 0

    reject(lambda: compare_heating_meshes(
        a, replace(b, lower_left_cm=np.array([-1., -2., -2.]))
    ), "bounds")
    reject(lambda: compare_heating_meshes(
        a, replace(b, score="heating", photon_transport=True)
    ), "estimators")
    reject(lambda: compare_heating_meshes(
        a, replace(b, normalized_power_w=power*1.1)
    ), "does not close")
    reject(lambda: compare_heating_meshes(
        a, replace(b, density_w_cm3=np.ones((3, 4, 4)))
    ), "does not close")
    # Keep power normalized to test the integer nesting rather than a
    # misleading failure on overall power.
    third = np.full((3, 4, 4), power / 64.)
    reject(lambda: compare_heating_meshes(
        a, replace(b, density_w_cm3=third)
    ), "integer-multiple")
    mixed = np.full((4, 1, 4), power / (4 * 4 * 4))
    reject(lambda: compare_heating_meshes(
        a, replace(b, density_w_cm3=mixed)
    ), "mixed finer/coarser")
    reject(lambda: compare_heating_meshes(
        a, replace(a, density_w_cm3=np.full((2, 2, 2), np.nan))
    ), "nonfinite")
    reject(lambda: compare_heating_meshes(
        a, replace(a, density_w_cm3=-coarse)
    ), "negative")
    print("NERVA OpenMC conservative spatial-mesh refinement: PASS")
    print("  real OpenMC transport executed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
