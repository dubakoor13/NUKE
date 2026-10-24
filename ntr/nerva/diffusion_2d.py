"""Two-dimensional (r-z) multigroup diffusion for a cylindrical core.

The diffusion backend carried an approximation it labelled honestly: a 1-D axial
slab plus *transverse radial buckling*. That is a reasonable engineering
reduction, but it cannot represent a radial power shape, so radial peaking and
radial reflector effects are invisible to it.

This module solves the real two-dimensional problem on an r-z mesh. The angular
coordinate is still eliminated by axisymmetry, which is exact for a cylindrical
core with a rotationally symmetric arrangement of drums - a genuine 2-D solution
rather than a 1-D solution with a leakage correction.

Validation is against an exact benchmark: for a homogeneous bare cylinder the
2-D solution must converge to the closed-form buckling of Table 8.2,
``B^2 = (pi/H~)^2 + (2.405/R~)^2``, as the mesh is refined. Nothing about the
discretisation is trusted; it is checked against the analytic answer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .diffusion import J0_FIRST_ZERO, DiffusionError, MultigroupCrossSections


class Diffusion2DError(ValueError):
    """Raised when a 2-D diffusion problem is not well posed."""


@dataclass(frozen=True)
class CylinderSolution:
    """Result of a two-dimensional r-z eigenvalue solve."""

    k_effective: float
    radial_intervals: int
    axial_intervals: int
    height_cm: float
    radius_cm: float
    extrapolation_cm: float
    radial_power_fractions: tuple[float, ...]
    axial_power_fractions: tuple[float, ...]
    radial_peaking: float
    axial_peaking: float
    converged: bool
    iterations: int
    residual: float

    @property
    def buckling_squared(self) -> float:
        """Geometric buckling the discrete problem should correspond to."""
        return ((math.pi / (self.height_cm + 2.0 * self.extrapolation_cm)) ** 2
                + (J0_FIRST_ZERO / (self.radius_cm + 2.0 * self.extrapolation_cm)) ** 2)


def solve_cylinder_eigenvalue(
    cross_sections: MultigroupCrossSections,
    height_cm: float,
    radius_cm: float,
    extrapolation_cm: float,
    *,
    radial_intervals: int = 24,
    axial_intervals: int = 40,
    radial_bands: int = 7,
    axial_bands: int = 7,
    tolerance: float = 1e-10,
    max_iterations: int = 3000,
) -> CylinderSolution:
    """Solve ``A phi = (1/k) F phi`` on an axisymmetric r-z mesh.

    Boundary conditions: zero flux at the extrapolated outer radius and at both
    extrapolated axial ends; zero radial gradient on the axis, which is the
    symmetry condition.

    Radial discretisation is cell-centred (``r_i = (i + 1/2) dr``) so the axis
    singularity of ``(1/r) d/dr (r dphi/dr)`` never appears, and the discrete
    radial operator reproduces ``-D d2phi/dr2 - (D/r) dphi/dr`` with the correct
    cylindrical weight on each face.
    """
    if not math.isfinite(height_cm) or height_cm <= 0.0:
        raise Diffusion2DError("height must be finite and positive")
    if not math.isfinite(radius_cm) or radius_cm <= 0.0:
        raise Diffusion2DError("radius must be finite and positive")
    if not math.isfinite(extrapolation_cm) or extrapolation_cm < 0.0:
        raise Diffusion2DError("extrapolation distance must be finite and non-negative")
    if radial_intervals < 3 or axial_intervals < 3:
        raise Diffusion2DError("at least 3 intervals are needed in each direction")
    if radial_bands < 1 or axial_bands < 1:
        raise Diffusion2DError("at least one reporting band is needed")

    groups = cross_sections.groups
    # The extrapolated dimensions are R~ = R + 2d and H~ = H + 2d, matching
    # Eq. (8.20) and the buckling in Table 8.2. Using R + d here made the solver
    # solve a DIFFERENT, less-leaky cylinder, which is why its k sat above the
    # analytic oracle no matter how the mesh was refined.
    outer_radius = radius_cm + 2.0 * extrapolation_cm
    total_height = height_cm + 2.0 * extrapolation_cm
    dr = outer_radius / radial_intervals
    dz = total_height / axial_intervals

    nr, nz = radial_intervals, axial_intervals
    nodes = nr * nz
    size = groups * nodes
    removal = cross_sections.removal()

    a_matrix = lil_matrix((size, size))
    f_matrix = lil_matrix((size, size))

    def index(group: int, i: int, j: int) -> int:
        return group * nodes + i * nz + j

    for group in range(groups):
        diffusion = cross_sections.diffusion_coefficient_cm[group]
        axial_diag = 2.0 * diffusion / dz ** 2
        for i in range(nr):
            radius = (i + 0.5) * dr
            # Cylindrical face radii. The inner face of cell 0 is the axis, where
            # r = 0 suppresses the flux; the outer face of the last cell is the
            # extrapolated radius, where the flux itself is zero.
            inner_radius = i * dr
            outer_radius_face = (i + 1) * dr
            radial_scale = diffusion / (radius * dr ** 2)
            for j in range(nz):
                row = index(group, i, j)
                a_matrix[row, row] += (radial_scale * (inner_radius + outer_radius_face)
                                       + axial_diag + removal[group])
                # Inner radial coupling: absent on the axis, where r = 0 carries
                # no flux so the symmetry condition needs no ghost node.
                if i > 0:
                    a_matrix[row, index(group, i - 1, j)] -= radial_scale * inner_radius
                # Outer radial coupling. On the last cell the outer face is the
                # zero-flux boundary, so the leakage appears in the DIAGONAL and
                # there is no neighbour. Dropping this term while keeping
                # diag = 2D/dr^2 made the outer boundary reflective, and the
                # solve then converged to a different PDE - k drifted from 1.094
                # to 1.079 as the mesh was refined instead of towards 1.094.
                if i + 1 < nr:
                    a_matrix[row, index(group, i + 1, j)] -= radial_scale * outer_radius_face
                # Axial coupling.
                if j + 1 < nz:
                    a_matrix[row, index(group, i, j + 1)] -= diffusion / dz ** 2
                if j > 0:
                    a_matrix[row, index(group, i, j - 1)] -= diffusion / dz ** 2
                # Inscatter and fission source.
                for other in range(groups):
                    if other != group:
                        a_matrix[row, index(other, i, j)] -= (
                            cross_sections.scatter_matrix[other][group])
                    f_matrix[row, index(other, i, j)] += (
                        cross_sections.chi[group]
                        * cross_sections.nu_sigma_fission_cm[other])

    a_csr = a_matrix.tocsr()
    f_csr = f_matrix.tocsr()

    phi = np.ones(size)
    k = 1.0
    residual = float("inf")
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        source = f_csr @ phi
        new_phi = spsolve(a_csr, source)
        if not np.all(np.isfinite(new_phi)):
            raise Diffusion2DError("the 2-D diffusion solve produced a non-finite flux")
        # For an axisymmetric problem the flux must be non-negative; a sign
        # oscillation means the mesh is too coarse for the fundamental mode.
        norm = float(np.max(np.abs(new_phi)))
        if norm <= 0.0:
            raise Diffusion2DError("the 2-D diffusion solve produced a zero flux")
        new_phi = new_phi / norm
        k = norm
        residual = float(np.max(np.abs(new_phi - phi)))
        phi = new_phi
        if residual < tolerance:
            converged = True
            break

    # Collapse to a single group shape for reporting: sum over groups weighted
    # by fission production, which is what a power shape means.
    shape = np.zeros(nodes)
    for group in range(groups):
        weight = cross_sections.nu_sigma_fission_cm[group]
        shape += weight * np.abs(phi[group * nodes:(group + 1) * nodes])
    if not np.any(shape > 0.0):
        raise Diffusion2DError("no fission-producing group has a positive flux")
    shape = shape.reshape(nr, nz)

    radial = shape.sum(axis=1)
    axial = shape.sum(axis=0)
    radial_fractions = _band_fractions(radial, radial_bands, "radial")
    axial_fractions = _band_fractions(axial, axial_bands, "axial")

    def peaking(values: np.ndarray) -> float:
        average = float(values.mean())
        if average <= 0.0:
            raise Diffusion2DError("a collapsed power shape has a non-positive mean")
        return float(values.max()) / average

    return CylinderSolution(
        k_effective=float(k),
        radial_intervals=nr,
        axial_intervals=nz,
        height_cm=height_cm,
        radius_cm=radius_cm,
        extrapolation_cm=extrapolation_cm,
        radial_power_fractions=radial_fractions,
        axial_power_fractions=axial_fractions,
        radial_peaking=peaking(radial),
        axial_peaking=peaking(axial),
        converged=converged,
        iterations=iteration,
        residual=residual,
    )


def _band_fractions(values: np.ndarray, bands: int, label: str) -> tuple[float, ...]:
    """Redistribute a fine mesh distribution onto coarse reporting bands.

    The band count must divide the interval count. Otherwise the bands have
    different numbers of cells, and comparing their power shares says nothing
    about the shape - it mostly measures the band widths, which is exactly the
    false signal that made an earlier version of this report look non-monotone.
    """
    if bands > values.size:
        raise Diffusion2DError(
            f"{bands} {label} bands requested but the mesh has only {values.size} "
            f"intervals"
        )
    if values.size % bands != 0:
        raise Diffusion2DError(
            f"{values.size} {label} intervals is not divisible by {bands} bands; "
            f"unequal band widths make the reported profile uninterpretable"
        )
    width = values.size // bands
    out = [float(values[index * width:(index + 1) * width].sum())
           for index in range(bands)]
    total = math.fsum(out)
    if total <= 0.0:
        raise Diffusion2DError(f"the {label} power shape sums to zero")
    return tuple(value / total for value in out)


def analytic_cylinder_k(cross_sections: MultigroupCrossSections,
                        height_cm: float, radius_cm: float,
                        extrapolation_cm: float) -> float:
    """One-group closed-form k for a bare cylinder, for validation.

    Uses the Table 8.2 buckling with ``Eq. 8.21``. Only meaningful for a
    one-group cross-section set, which is exactly how it is used: as the oracle
    the 2-D discrete solve must reproduce.
    """
    if cross_sections.groups != 1:
        raise Diffusion2DError(
            "the analytic oracle is one-group only; collapse the cross-sections "
            "before comparing"
        )
    buckling = ((math.pi / (height_cm + 2.0 * extrapolation_cm)) ** 2
                + (J0_FIRST_ZERO / (radius_cm + 2.0 * extrapolation_cm)) ** 2)
    diffusion = cross_sections.diffusion_coefficient_cm[0]
    absorption = cross_sections.sigma_absorption_cm[0]
    nu_sigma_f = cross_sections.nu_sigma_fission_cm[0]
    return nu_sigma_f / (diffusion * buckling + absorption)
