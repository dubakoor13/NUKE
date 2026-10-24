"""Independent cylindrical geometry checks for bundle-derived diffusion code.

Only synthetic, uniform one-group diffusion; no NERVA materials, reactor
sizing, enrichment or physical nuclear-data qualification. This deliberately
uses analytic Bessel and sine solutions rather than the source's own test
helpers to detect mistakes in boundary-face and annular-power accounting.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import j1, jn_zeros

from .diffusion import one_group_cross_sections
from .diffusion_2d import solve_cylinder_eigenvalue


def main() -> int:
    d, sigma_a, nu_sigma_f = 1.25, 0.0092, 0.0168  # synthetic
    height, radius, extrap = 90.0, 35.0, 2.0
    xs = one_group_cross_sections(d, sigma_a, nu_sigma_f)
    alpha = float(jn_zeros(0, 1)[0])
    total_r, total_h = radius + 2.0 * extrap, height + 2.0 * extrap
    expected_k = nu_sigma_f / (
        sigma_a + d * ((alpha / total_r)**2 + (math.pi / total_h)**2)
    )

    answers = []
    for n in (16, 32, 48):
        result = solve_cylinder_eigenvalue(
            xs, height, radius, extrap,
            radial_intervals=n, axial_intervals=n,
            radial_bands=4, axial_bands=4,
        )
        assert result.converged and result.residual < 1e-8
        err = abs(result.k_effective / expected_k - 1.)
        answers.append((n, err, result))
        print(f"{n} x {n}  k={result.k_effective:.9f}  relative error={err:.4e}")

    if not (answers[-1][1] < answers[0][1] / 2.):
        raise AssertionError("mesh refinement failed to improve cylindrical eigenvalue")
    fine = answers[-1][2]

    # This is a volume integral over actual annuli, not the unweighted
    # arithmetic sum of flux at equally spaced radial positions.
    fraction_edges = np.linspace(0.0, 1.0, 5)
    primitive = fraction_edges * j1(alpha * fraction_edges)
    expected_radial = np.diff(primitive) / primitive[-1]
    actual_radial = np.asarray(fine.radial_power_fractions)
    print("volume-integrated radial power bands:", actual_radial)
    print("analytic Bessel radial power bands:", expected_radial)
    if np.max(np.abs(actual_radial - expected_radial)) > 0.025:
        raise AssertionError("annular power fractions fail independent volume integration")

    # Longitudinal integral of sin(pi*z/H) over four equal bands.
    u = fraction_edges
    expected_axial = (np.cos(math.pi*u[:-1]) - np.cos(math.pi*u[1:])) / 2.
    actual_axial = np.asarray(fine.axial_power_fractions)
    print("axial power bands:", actual_axial)
    if np.max(np.abs(actual_axial - expected_axial)) > 0.020:
        raise AssertionError("axial power fractions fail independent integration")

    # Volumetric radial peaking against area-averaged fission density:
    # average(J0(alpha*r/R))=2*J1(alpha)/alpha.
    exact_radial_peaking = alpha / (2.0 * j1(alpha))
    print("radial peaking:", fine.radial_peaking,
          "analytic:", exact_radial_peaking)
    if abs(fine.radial_peaking / exact_radial_peaking - 1.) > 0.06:
        raise AssertionError("radial peaking must use volume-mean power density")
    print("INDEPENDENT_RZ_VOLUME_AND_BOUNDARY_CHECK_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
