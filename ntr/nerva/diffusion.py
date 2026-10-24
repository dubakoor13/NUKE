"""One-group and multigroup neutron diffusion (Emrich Ch. 8).

The last book topic not covered. Two capabilities:

1. **Closed-form criticality** — Eq. (8.21) with the geometric bucklings of
   Table 8.2, critical radius and mass (Eqs 8.27/8.28), cylinder critical height
   (8.29), the flux shape (8.23) and its power normalisation (8.24/8.25).
2. **A multigroup finite-difference eigenvalue solver** for a bare slab, which
   is what the chapter's title actually names. This is the deterministic
   counterpart to Monte Carlo: cheap, differentiable, and useful as a fast
   surrogate inside the coupled core loop.

The solver is validated against the closed form: for one group the discrete
eigenvalue must converge to Eq. (8.21) with ``B^2 = (pi/L~)^2`` as the mesh is
refined. That is an exact oracle, not a self-consistency check.

Unit convention: centimetres and barn-derived macroscopic cross-sections [1/cm].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# First zero of J0, from Emrich Table 8.2 (cylinder radial buckling).
J0_FIRST_ZERO = 2.405


class DiffusionError(ValueError):
    """Raised when a diffusion problem is not well posed."""


# ------------------------------------------------------- closed-form theory


def geometric_buckling(geometry: str, dimensions_cm: dict[str, float],
                       extrapolation_cm: float = 0.0) -> float:
    """Squared geometric buckling from Emrich Table 8.2 (p.123).

    ``extrapolation_cm`` is the extrapolation distance d; each dimension is
    extended by twice it, so ``L~ = L + 2d``, matching Eq. (8.20).
    """
    if not math.isfinite(extrapolation_cm) or extrapolation_cm < 0.0:
        raise DiffusionError("extrapolation distance must be finite and non-negative")
    two_d = 2.0 * extrapolation_cm

    def pi_over(name: str) -> float:
        key = name if name in dimensions_cm else name.lower()
        if key not in dimensions_cm:
            raise DiffusionError(f"dimension {name!r} is required for {geometry!r}")
        value = dimensions_cm[key]
        if not math.isfinite(value) or value <= 0.0:
            raise DiffusionError(f"dimension {name!r} must be positive, got {value!r}")
        return math.pi / (value + two_d)

    if geometry == "slab":
        return pi_over("L") ** 2
    if geometry == "box":
        return (pi_over("Lx") ** 2 + pi_over("Ly") ** 2 + pi_over("Lz") ** 2)
    if geometry == "sphere":
        return pi_over("R") ** 2
    if geometry == "cylinder":
        radius = dimensions_cm.get("R") or dimensions_cm.get("r")
        if radius is None:
            raise DiffusionError("dimension 'R' is required for a cylinder")
        height = dimensions_cm.get("H")
        if height is None:
            raise DiffusionError("dimension 'H' is required for a cylinder")
        axial = math.pi / (height + two_d)
        radial = J0_FIRST_ZERO / (radius + two_d)
        return axial ** 2 + radial ** 2
    raise DiffusionError(
        f"unknown geometry {geometry!r}; expected slab, box, sphere or cylinder"
    )


def k_effective(nu_sigma_f: float, sigma_a: float, diffusion_coefficient: float,
                buckling_squared: float) -> float:
    """Eq. (8.21): k_eff = nu*Sigma_f / (D B^2 + Sigma_a)."""
    for name, value in (("nu_sigma_f", nu_sigma_f), ("diffusion_coefficient",
                                                      diffusion_coefficient),
                        ("buckling_squared", buckling_squared)):
        if not math.isfinite(value) or value < 0.0:
            raise DiffusionError(f"{name} must be finite and non-negative, got {value!r}")
    if not math.isfinite(sigma_a) or sigma_a <= 0.0:
        raise DiffusionError("absorption cross-section must be finite and positive")
    denominator = diffusion_coefficient * buckling_squared + sigma_a
    if denominator <= 0.0:
        raise DiffusionError("the loss term D B^2 + Sigma_a must be positive")
    return nu_sigma_f / denominator


def k_infinity(nu_sigma_f: float, sigma_a: float) -> float:
    """Eq. (8.22): k_inf = nu*Sigma_f / Sigma_a."""
    if not math.isfinite(sigma_a) or sigma_a <= 0.0:
        raise DiffusionError("absorption cross-section must be finite and positive")
    return nu_sigma_f / sigma_a


def material_buckling(nu_sigma_f: float, sigma_a: float,
                      diffusion_coefficient: float) -> float:
    """``B_m^2 = (nu Sigma_f - Sigma_a) / D``, the criticality condition of (8.21)."""
    if not math.isfinite(diffusion_coefficient) or diffusion_coefficient <= 0.0:
        raise DiffusionError("diffusion coefficient must be finite and positive")
    return (nu_sigma_f - sigma_a) / diffusion_coefficient


def critical_sphere_radius(nu_sigma_f: float, sigma_a: float,
                           diffusion_coefficient: float,
                           extrapolation_cm: float | None = None) -> float:
    """Eq. (8.27): critical radius of a bare sphere.

    ``R_c = pi / sqrt(D / (nu Sigma_f - Sigma_a)) - 2D``, with the extrapolation
    distance taken as the diffusion coefficient by default, which is the
    convention Eqs. (8.27)-(8.29) use.
    """
    buckling = material_buckling(nu_sigma_f, sigma_a, diffusion_coefficient)
    if buckling <= 0.0:
        raise DiffusionError(
            f"material buckling is {buckling:.4e}; the material cannot sustain a "
            f"chain reaction at any size (k_inf = {nu_sigma_f/sigma_a:.4f} <= 1)"
        )
    extrapolation = diffusion_coefficient if extrapolation_cm is None else extrapolation_cm
    radius = math.pi / math.sqrt(buckling) - 2.0 * extrapolation
    if radius <= 0.0:
        raise DiffusionError(
            "the critical radius is non-positive: the extrapolation correction "
            "exceeds the extrapolated critical radius"
        )
    return radius


def critical_sphere_mass(nu_sigma_f: float, sigma_a: float,
                         diffusion_coefficient: float, density_g_cm3: float,
                         extrapolation_cm: float | None = None) -> float:
    """Eq. (8.28): critical mass M_c = rho_f * (4/3) pi R_c^3."""
    if not math.isfinite(density_g_cm3) or density_g_cm3 <= 0.0:
        raise DiffusionError("density must be finite and positive")
    radius = critical_sphere_radius(nu_sigma_f, sigma_a, diffusion_coefficient,
                                    extrapolation_cm)
    return density_g_cm3 * 4.0 / 3.0 * math.pi * radius ** 3


def critical_cylinder_height(nu_sigma_f: float, sigma_a: float,
                             diffusion_coefficient: float, radius_cm: float,
                             extrapolation_cm: float | None = None) -> float:
    """Eq. (8.29): critical height of a bare cylinder of known radius."""
    if not math.isfinite(radius_cm) or radius_cm <= 0.0:
        raise DiffusionError("radius must be finite and positive")
    buckling = material_buckling(nu_sigma_f, sigma_a, diffusion_coefficient)
    extrapolation = diffusion_coefficient if extrapolation_cm is None else extrapolation_cm
    two_d = 2.0 * extrapolation
    radial = (J0_FIRST_ZERO / (radius_cm + two_d)) ** 2
    remainder = buckling - radial
    if remainder <= 0.0:
        raise DiffusionError(
            "the radius alone already exceeds the material buckling; this "
            "cylinder is critical at any height"
        )
    extrapolated_height = math.pi / math.sqrt(remainder)
    return extrapolated_height - two_d


def slab_flux(z_cm: float, thickness_cm: float, extrapolation_cm: float,
              amplitude: float = 1.0) -> float:
    """Eq. (8.23): the chopped-sine flux shape in a bare slab."""
    if not math.isfinite(z_cm):
        raise DiffusionError("position must be finite")
    if not math.isfinite(thickness_cm) or thickness_cm <= 0.0:
        raise DiffusionError("thickness must be finite and positive")
    extrapolated = thickness_cm + 2.0 * extrapolation_cm
    return amplitude * math.sin(math.pi / extrapolated * (z_cm + extrapolation_cm))


def slab_peak_power_density(nu_sigma_f: float, thickness_cm: float,
                            extrapolation_cm: float, power_per_area_w_cm2: float,
                            energy_per_fission_j: float = 3.204e-11) -> float:
    """Power normalisation for a bare slab (Eqs 8.24/8.25 rearranged).

    Returns the *peak* fission power density [W/cm3] for a slab of the stated
    thickness carrying the stated power per unit face area. Integrates the
    chopped sine analytically, so no numerical quadrature is involved.
    """
    if not math.isfinite(power_per_area_w_cm2) or power_per_area_w_cm2 <= 0.0:
        raise DiffusionError("power per unit area must be finite and positive")
    extrapolated = thickness_cm + 2.0 * extrapolation_cm
    # Integral of sin(pi(z+d)/L~) from 0 to L, with d the extrapolation distance.
    upper = math.pi / extrapolated * (thickness_cm + extrapolation_cm)
    lower = math.pi / extrapolated * extrapolation_cm
    integral_cm = extrapolated / math.pi * (math.cos(lower) - math.cos(upper))
    if integral_cm <= 0.0:
        raise DiffusionError("the slab flux integral is non-positive")
    average_fission_density = power_per_area_w_cm2 / (integral_cm * energy_per_fission_j)
    # Peak-to-average for the chopped sine.
    peak_to_average = 1.0 / (integral_cm / thickness_cm)
    return average_fission_density * peak_to_average


# ------------------------------------------------- multigroup slab solver


@dataclass(frozen=True)
class MultigroupCrossSections:
    """Multigroup data for a homogeneous bare slab.

    ``scatter_matrix[g_from][g_to]`` is the group-to-group scattering
    cross-section. Removal is assembled from absorption plus outscatter, so the
    caller cannot supply an inconsistent removal term.
    """

    diffusion_coefficient_cm: tuple[float, ...]
    sigma_absorption_cm: tuple[float, ...]
    nu_sigma_fission_cm: tuple[float, ...]
    chi: tuple[float, ...]
    scatter_matrix: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        groups = len(self.diffusion_coefficient_cm)
        if groups < 1:
            raise DiffusionError("at least one energy group is required")
        for name in ("sigma_absorption_cm", "nu_sigma_fission_cm", "chi"):
            if len(getattr(self, name)) != groups:
                raise DiffusionError(f"{name} length must match the group count")
        if len(self.scatter_matrix) != groups:
            raise DiffusionError("scatter matrix must be square with one row per group")
        for row in self.scatter_matrix:
            if len(row) != groups:
                raise DiffusionError("scatter matrix rows must have one entry per group")
        for name in ("diffusion_coefficient_cm", "sigma_absorption_cm",
                     "nu_sigma_fission_cm", "chi"):
            for value in getattr(self, name):
                if not math.isfinite(value) or value < 0.0:
                    raise DiffusionError(f"{name} entries must be finite and non-negative")
        for group, value in enumerate(self.diffusion_coefficient_cm):
            if value <= 0.0:
                raise DiffusionError(f"group {group} diffusion coefficient must be positive")
        chi_total = math.fsum(self.chi)
        if abs(chi_total - 1.0) > 1e-9:
            raise DiffusionError(
                f"fission spectrum must sum to 1, got {chi_total!r}"
            )

    @property
    def groups(self) -> int:
        return len(self.diffusion_coefficient_cm)

    def removal(self) -> tuple[float, ...]:
        """Sigma_r[g] = Sigma_a[g] + sum over g' != g of Sigma_s[g -> g']."""
        out = []
        for group in range(self.groups):
            outscatter = math.fsum(
                self.scatter_matrix[group][other]
                for other in range(self.groups) if other != group)
            out.append(self.sigma_absorption_cm[group] + outscatter)
        return tuple(out)


@dataclass(frozen=True)
class SlabSolution:
    """Eigenvalue and flux shape from the finite-difference solver."""

    k_effective: float
    mesh_intervals: int
    thickness_cm: float
    buckling_squared: float
    group_flux_peaks: tuple[float, ...]
    converged: bool
    iterations: int
    power_iteration_residual: float

    @property
    def flux_is_positive(self) -> bool:
        return all(peak > 0.0 for peak in self.group_flux_peaks)


def slab_system_matrices(cross_sections: MultigroupCrossSections,
                         thickness_cm: float, extrapolation_cm: float,
                         mesh_intervals: int) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the ``A phi = (1/k) F phi`` system for a bare slab.

    Exposed separately from the solver because the **adjoint** problem needs the
    transposes of the same matrices, and building them twice invites the two to
    drift apart.
    """
    if not math.isfinite(thickness_cm) or thickness_cm <= 0.0:
        raise DiffusionError("thickness must be finite and positive")
    if not math.isfinite(extrapolation_cm) or extrapolation_cm < 0.0:
        raise DiffusionError("extrapolation distance must be finite and non-negative")
    if mesh_intervals < 4:
        raise DiffusionError("at least 4 mesh intervals are required")

    groups = cross_sections.groups
    extrapolated = thickness_cm + 2.0 * extrapolation_cm
    h = extrapolated / mesh_intervals
    nodes = mesh_intervals - 1
    if nodes < 1:
        raise DiffusionError("mesh is too coarse to have interior nodes")

    removal = cross_sections.removal()
    size = groups * nodes
    a_matrix = np.zeros((size, size))
    f_matrix = np.zeros((size, size))

    def index(group: int, node: int) -> int:
        return group * nodes + node

    for group in range(groups):
        d_over_h2 = cross_sections.diffusion_coefficient_cm[group] / h ** 2
        for node in range(nodes):
            row = index(group, node)
            a_matrix[row, row] += 2.0 * d_over_h2 + removal[group]
            if node > 0:
                a_matrix[row, index(group, node - 1)] -= d_over_h2
            if node < nodes - 1:
                a_matrix[row, index(group, node + 1)] -= d_over_h2
            for other in range(groups):
                if other == group:
                    continue
                a_matrix[row, index(other, node)] -= cross_sections.scatter_matrix[other][group]
            for other in range(groups):
                f_matrix[row, index(other, node)] += (
                    cross_sections.chi[group]
                    * cross_sections.nu_sigma_fission_cm[other])
    return a_matrix, f_matrix


def solve_slab_eigenvalue(cross_sections: MultigroupCrossSections,
                          thickness_cm: float, extrapolation_cm: float,
                          *, mesh_intervals: int = 200,
                          tolerance: float = 1e-10,
                          max_iterations: int = 5000) -> SlabSolution:
    """Multigroup finite-difference k-eigenvalue for a bare slab.

    Solves ``A phi = (1/k) F phi`` on interior nodes with the flux forced to zero
    at the extrapolated boundaries, then power-iterates for the fundamental mode.
    """
    if not math.isfinite(thickness_cm) or thickness_cm <= 0.0:
        raise DiffusionError("thickness must be finite and positive")
    if not math.isfinite(extrapolation_cm) or extrapolation_cm < 0.0:
        raise DiffusionError("extrapolation distance must be finite and non-negative")
    if mesh_intervals < 4:
        raise DiffusionError("at least 4 mesh intervals are required")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise DiffusionError("tolerance must be finite and positive")

    groups = cross_sections.groups
    extrapolated = thickness_cm + 2.0 * extrapolation_cm
    h = extrapolated / mesh_intervals
    nodes = mesh_intervals - 1          # interior nodes only
    if nodes < 1:
        raise DiffusionError("mesh is too coarse to have interior nodes")

    removal = cross_sections.removal()
    size = groups * nodes
    a_matrix = np.zeros((size, size))
    f_matrix = np.zeros((size, size))

    def index(group: int, node: int) -> int:
        return group * nodes + node

    for group in range(groups):
        d_over_h2 = cross_sections.diffusion_coefficient_cm[group] / h ** 2
        for node in range(nodes):
            row = index(group, node)
            # -D d2phi/dz2 discretised: diagonal 2D/h^2, off-diagonals -D/h^2.
            a_matrix[row, row] += 2.0 * d_over_h2 + removal[group]
            if node > 0:
                a_matrix[row, index(group, node - 1)] -= d_over_h2
            if node < nodes - 1:
                a_matrix[row, index(group, node + 1)] -= d_over_h2
            # Inscatter from every other group.
            for other in range(groups):
                if other == group:
                    continue
                a_matrix[row, index(other, node)] -= cross_sections.scatter_matrix[other][group]
            # Fission source.
            for other in range(groups):
                f_matrix[row, index(other, node)] += (
                    cross_sections.chi[group]
                    * cross_sections.nu_sigma_fission_cm[other])

    try:
        solve = np.linalg.solve
    except AttributeError:  # pragma: no cover - numpy always provides solve
        raise DiffusionError("numpy linear solver unavailable") from None

    phi = np.ones(size)
    k = 1.0
    converged = False
    residual = float("inf")
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        source = f_matrix @ phi
        new_phi = solve(a_matrix, source)
        norm = float(np.max(np.abs(new_phi)))
        if not math.isfinite(norm) or norm <= 0.0:
            raise DiffusionError("the diffusion solve produced a non-physical flux")
        new_phi = new_phi / norm
        # The flux is renormalised to unit maximum each step, so the norm of the
        # *unnormalised* solution IS the eigenvalue. Compounding it (multiplying
        # by the previous k) raised k to the iteration count instead of
        # converging - the first version of this loop reported k = 5.1 instead of
        # 1.085 after 17 iterations of lambda ~ 1.09.
        new_k = norm
        residual = float(np.max(np.abs(new_phi - phi)))
        phi = new_phi
        k = new_k
        if residual < tolerance:
            converged = True
            break

    peaks = tuple(float(max(abs(phi[index(group, node)])
                            for node in range(nodes)))
                  for group in range(groups))
    return SlabSolution(
        k_effective=float(k),
        mesh_intervals=mesh_intervals,
        thickness_cm=thickness_cm,
        buckling_squared=(math.pi / extrapolated) ** 2,
        group_flux_peaks=peaks,
        converged=converged,
        iterations=iteration,
        power_iteration_residual=residual,
    )


def one_group_cross_sections(diffusion_coefficient_cm: float,
                             sigma_absorption_cm: float,
                             nu_sigma_fission_cm: float) -> MultigroupCrossSections:
    """Convenience constructor for a single-group problem."""
    return MultigroupCrossSections(
        diffusion_coefficient_cm=(diffusion_coefficient_cm,),
        sigma_absorption_cm=(sigma_absorption_cm,),
        nu_sigma_fission_cm=(nu_sigma_fission_cm,),
        chi=(1.0,),
        scatter_matrix=((0.0,),),
    )
