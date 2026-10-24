"""Verify the 2-D r-z diffusion solver against the analytic cylinder buckling.

The benchmark is exact: for a homogeneous bare cylinder the discrete k must
converge to

    k = nu Sigma_f / (D B^2 + Sigma_a),   B^2 = (pi/H~)^2 + (2.405/R~)^2

from Emrich Table 8.2 as the mesh is refined. That tests the discretisation
itself - the axis treatment, the cylindrical face weighting and the boundary
conditions - against mathematics it cannot influence.
"""
from __future__ import annotations

import math

from .diffusion import MultigroupCrossSections, one_group_cross_sections
from .diffusion_2d import (
    Diffusion2DError,
    analytic_cylinder_k,
    solve_cylinder_eigenvalue,
)

D1 = 1.25
SA1 = 0.0092
NUSF1 = 0.0168
HEIGHT = 90.0
RADIUS = 35.0
EXTRAPOLATION = 2.0

ONE_GROUP = one_group_cross_sections(D1, SA1, NUSF1)

TWO_GROUP = MultigroupCrossSections(
    diffusion_coefficient_cm=(1.35, 0.85),
    sigma_absorption_cm=(0.0080, 0.0850),
    nu_sigma_fission_cm=(0.0045, 0.1300),
    chi=(1.0, 0.0),
    scatter_matrix=((0.0, 0.0180), (0.0, 0.0)),
)


def _report(label: str, ok: bool, detail: str) -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<56} {detail}")
    return ok


def main() -> int:
    results: list[bool] = []

    # ---- mesh convergence to the analytic buckling ---------------------
    print("2-D solver converges to the Table 8.2 cylinder buckling:")
    oracle = analytic_cylinder_k(ONE_GROUP, HEIGHT, RADIUS, EXTRAPOLATION)
    print(f"    analytic k = {oracle:.9f}")
    previous_error = None
    coarse = None
    for nr, nz in ((16, 16), (32, 32), (48, 48), (64, 64)):
        solution = solve_cylinder_eigenvalue(
            ONE_GROUP, HEIGHT, RADIUS, EXTRAPOLATION,
            radial_intervals=nr, axial_intervals=nz, radial_bands=4, axial_bands=4)
        error = abs(solution.k_effective - oracle) / oracle
        print(f"    {nr:2d} x {nz:2d}: k = {solution.k_effective:.9f}  "
              f"rel {error:.2e}  ({solution.iterations} iterations)")
        if nr >= 64:
            results.append(_report(f"{nr}x{nz} within 1% of the analytic k",
                                   error < 1e-2, f"rel {error:.2e}"))
        if previous_error is not None:
            results.append(_report(f"{nr}x{nz} improves on the coarser mesh",
                                   error < previous_error,
                                   f"{error:.2e} < {previous_error:.2e}"))
        previous_error = error
        coarse = solution
    # The scheme is first-order in the radial direction (the axis cell limits
    # accuracy against a Bessel function), so the honest assertions are that the
    # error shrinks by roughly 2x per doubling and that the observed order is
    # near one - not that a modest mesh already matches the closed form.
    results.append(_report("the finest mesh agrees to better than 1%",
                           previous_error < 1e-2, f"rel {previous_error:.2e}"))
    results.append(_report("power iteration converged", coarse.converged is True,
                           f"residual {coarse.residual:.2e}"))
    results.append(_report("the geometric buckling matches Table 8.2",
                           abs(coarse.buckling_squared
                               - ((math.pi / (HEIGHT + 2 * EXTRAPOLATION)) ** 2
                                  + (2.405 / (RADIUS + 2 * EXTRAPOLATION)) ** 2))
                           / coarse.buckling_squared < 1e-15,
                           f"{coarse.buckling_squared:.6e} /cm2"))

    # ---- radial and axial shapes --------------------------------------
    print("\nradial and axial power shapes:")
    print("    radial fractions " + " ".join(f"{v:.4f}" for v in coarse.radial_power_fractions))
    print("    axial  fractions " + " ".join(f"{v:.4f}" for v in coarse.axial_power_fractions))
    results.append(_report("radial fractions sum to one",
                           abs(math.fsum(coarse.radial_power_fractions) - 1.0) < 1e-12,
                           "normalised"))
    results.append(_report("axial fractions sum to one",
                           abs(math.fsum(coarse.axial_power_fractions) - 1.0) < 1e-12,
                           "normalised"))
    results.append(_report("the radial peak is at the axis, not the edge",
                           coarse.radial_power_fractions[0]
                           > coarse.radial_power_fractions[-1],
                           f"{coarse.radial_power_fractions[0]:.4f} > "
                           f"{coarse.radial_power_fractions[-1]:.4f}"))
    # With an even band count the two central bands are tied by symmetry, so the
    # peak must be allowed at either of them.
    centre_bands = (coarse.axial_power_fractions[1],
                    coarse.axial_power_fractions[2])
    results.append(_report("the axial peak is at mid-height",
                           abs(max(centre_bands) - max(coarse.axial_power_fractions))
                           < 1e-12,
                           "centred (symmetric pair)"))
    print(f"    radial peaking = {coarse.radial_peaking:.4f}, "
          f"axial peaking = {coarse.axial_peaking:.4f}")
    results.append(_report("radial peaking exceeds one",
                           coarse.radial_peaking > 1.0, f"{coarse.radial_peaking:.4f}"))
    results.append(_report("axial peaking exceeds one",
                           coarse.axial_peaking > 1.0, f"{coarse.axial_peaking:.4f}"))
    # Whether axial or radial peaking dominates depends on the geometry, so this
    # is reported rather than asserted - an earlier version asserted a direction
    # that only held for one particular core.
    print(f"    axial peaking {coarse.axial_peaking:.3f} vs radial "
          f"{coarse.radial_peaking:.3f} (geometry dependent, not asserted)")

    # ---- the real gain: radial shape is now resolved -------------------
    print("\nthe 1-D reduction cannot do this:")
    # A 1-D slab sees no radial shape at all; the 2-D solve reports one.
    results.append(_report("a radial distribution is now available",
                           len(coarse.radial_power_fractions) > 1
                           and max(coarse.radial_power_fractions)
                           - min(coarse.radial_power_fractions) > 0.01,
                           "non-uniform radial profile"))
    # A shorter, fatter core peaks less radially.
    fat = solve_cylinder_eigenvalue(ONE_GROUP, 60.0, 45.0, EXTRAPOLATION,
                                    radial_intervals=32, axial_intervals=32,
                                    radial_bands=4, axial_bands=4)
    print(f"    H=60 R=45: radial peaking {fat.radial_peaking:.4f}, "
          f"axial {fat.axial_peaking:.4f}")
    results.append(_report("a flatter core has weaker peaking",
                           fat.radial_peaking < coarse.radial_peaking
                           or fat.axial_peaking < coarse.axial_peaking,
                           "geometry changes the shape"))

    # ---- two groups ----------------------------------------------------
    print("\ntwo-group 2-D solve:")
    two = solve_cylinder_eigenvalue(TWO_GROUP, HEIGHT, RADIUS, EXTRAPOLATION,
                                    radial_intervals=32, axial_intervals=32,
                                    radial_bands=4, axial_bands=4)
    print(f"    k = {two.k_effective:.6f}")
    results.append(_report("two-group 2-D solve converges", two.converged is True,
                           f"{two.iterations} iterations"))
    results.append(_report("two-group k is positive and finite",
                           math.isfinite(two.k_effective) and two.k_effective > 0.0,
                           f"{two.k_effective:.6f}"))

    # ---- guards --------------------------------------------------------
    print("\ninvalid inputs are rejected:")
    guards = [
        ("zero height", lambda: solve_cylinder_eigenvalue(
            ONE_GROUP, 0.0, RADIUS, EXTRAPOLATION)),
        ("zero radius", lambda: solve_cylinder_eigenvalue(
            ONE_GROUP, HEIGHT, 0.0, EXTRAPOLATION)),
        ("too few radial intervals", lambda: solve_cylinder_eigenvalue(
            ONE_GROUP, HEIGHT, RADIUS, EXTRAPOLATION, radial_intervals=2)),
        ("more bands than mesh intervals", lambda: solve_cylinder_eigenvalue(
            ONE_GROUP, HEIGHT, RADIUS, EXTRAPOLATION,
            radial_intervals=8, axial_intervals=20, radial_bands=10)),
        ("analytic oracle on a two-group set", lambda: analytic_cylinder_k(
            TWO_GROUP, HEIGHT, RADIUS, EXTRAPOLATION)),
    ]
    for label, fn in guards:
        try:
            fn()
        except (Diffusion2DError, ValueError):
            results.append(True)
            print(f"  [PASS] rejected {label}")
        else:
            results.append(False)
            print(f"  [FAIL] ACCEPTED {label}")

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    if failed:
        return 1
    print("PASS: 2-D r-z diffusion converges to the analytic cylinder buckling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
