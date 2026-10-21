"""Power-conservative comparison of regular OpenMC heating tally meshes.

A fine spatial heating map can be compared with a coarse one only when the
physical bounds agree and the fine dimensions are integer multiples of
the coarse dimensions. Coarsen the *deposited power*, never interpolate
temperatures or invent subvoxel source distributions. Source convergence,
Monte Carlo bias and historical NERVA fidelity require independent studies.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HeatingMesh:
    density_w_cm3: np.ndarray
    lower_left_cm: np.ndarray
    upper_right_cm: np.ndarray
    normalized_power_w: float
    score: str
    photon_transport: bool

    def validate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        density = np.asarray(self.density_w_cm3, dtype=float)
        lower = np.asarray(self.lower_left_cm, dtype=float)
        upper = np.asarray(self.upper_right_cm, dtype=float)
        if density.ndim != 3 or any(length < 1 for length in density.shape):
            raise ValueError("heating mesh must be a nonempty 3-D field")
        if (lower.shape != (3,) or upper.shape != (3,)
                or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                or np.any(upper <= lower)):
            raise ValueError("invalid physical OpenMC heating mesh bounds")
        if not np.isfinite(density).all() or np.any(density < 0):
            raise ValueError("nonfinite or negative deposited heating")
        if not math.isfinite(self.normalized_power_w) or self.normalized_power_w <= 0:
            raise ValueError("normalized thermal power must be positive finite")
        if self.score not in ("heating", "heating-local"):
            raise ValueError("unknown OpenMC power-normalization score")
        if not isinstance(self.photon_transport, (bool, np.bool_)):
            raise ValueError("OpenMC photon-transport flag must be boolean")
        if self.score != ("heating" if self.photon_transport else "heating-local"):
            raise ValueError("OpenMC heating score conflicts with neutron/photon mode")
        volume = float(np.prod((upper - lower) / np.asarray(density.shape)))
        integrated = float(np.sum(density, dtype=np.float64) * volume)
        if not math.isclose(
            integrated, self.normalized_power_w, rel_tol=2e-7, abs_tol=1e-8
        ):
            raise ValueError("OpenMC mesh power does not close with metadata")
        return density, lower, upper, volume


def _coarsen_volume_average(fine: np.ndarray, coarse_shape: tuple[int, int, int]
                            ) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Conserve total power for aligned Cartesian bins of uniform size."""
    ratios = tuple(f // c for f, c in zip(fine.shape, coarse_shape))
    if any(f < c or f % c for f, c in zip(fine.shape, coarse_shape)):
        raise ValueError(
            "OpenMC mesh refinement needs nested, integer-multiple "
            "dimensions in every direction"
        )
    nx, ny, nz = coarse_shape
    rx, ry, rz = ratios
    coarsened = fine.reshape(nx, rx, ny, ry, nz, rz).mean(axis=(1, 3, 5))
    return coarsened, ratios


def compare_heating_meshes(first: HeatingMesh, second: HeatingMesh) -> dict:
    """Compare two power-normalized OpenMC heating maps on a common mesh.

    Same-resolution input is allowed for Monte Carlo sampling differences.
    If one grid is finer on every axis, it is projected conservatively onto
    the coarser grid. A non-nested/mixed resolution comparison is REFUSED;
    a null convergence statistic must not be mistaken for convergence.
    """
    a, a_lo, a_hi, volume_a = first.validate()
    b, b_lo, b_hi, volume_b = second.validate()
    if (not np.allclose(a_lo, b_lo, rtol=0, atol=1e-10)
            or not np.allclose(a_hi, b_hi, rtol=0, atol=1e-10)):
        raise ValueError("OpenMC mesh physical bounds differ")
    if (first.score != second.score
            or first.photon_transport != second.photon_transport):
        raise ValueError("OpenMC heating estimators/transport modes differ")
    if not math.isclose(
        first.normalized_power_w, second.normalized_power_w, rel_tol=2e-7
    ):
        raise ValueError("reactor thermal-power normalization differs")

    a_shape = a.shape
    b_shape = b.shape
    if all(ai <= bi for ai, bi in zip(a_shape, b_shape)):
        coarse, fine, coarse_first = a, b, True
    elif all(bi <= ai for ai, bi in zip(a_shape, b_shape)):
        coarse, fine, coarse_first = b, a, False
    else:
        raise ValueError(
            "OpenMC mesh dimensions are mixed finer/coarser across axes"
        )
    projected, factors = _coarsen_volume_average(fine, coarse.shape)
    volume_coarse = volume_a if coarse_first else volume_b
    volume_fine = volume_b if coarse_first else volume_a
    energy_fine = float(np.sum(fine) * volume_fine)
    energy_coarsened = float(np.sum(projected) * volume_coarse)
    if not math.isclose(
        energy_fine, energy_coarsened, rel_tol=1e-12, abs_tol=1e-9
    ):
        raise ValueError("power conservation failed during mesh coarsening")

    denom = float(np.linalg.norm(coarse.ravel()))
    if denom <= 0:
        raise ValueError("coarse mesh has zero deposited power")
    l2 = float(np.linalg.norm((projected - coarse).ravel()) / denom)
    total_variation = 0.5 * float(
        np.sum(np.abs(projected - coarse))
        * volume_coarse / first.normalized_power_w
    )
    # Sub-coarse structure (aliasing): same coarse average does not imply
    # the fine grid is spatially uniform inside its parent voxels.
    expanded = projected
    for axis, ratio in enumerate(factors):
        expanded = np.repeat(expanded, ratio, axis=axis)
    fine_norm = float(np.linalg.norm(fine.ravel()))
    unresolved = (
        float(np.linalg.norm((fine - expanded).ravel()) / fine_norm)
        if fine_norm > 0 else 0.0
    )
    return {
        "comparison_type": (
            "same_grid" if a.shape == b.shape
            else "conservative_nested_mesh_projection"
        ),
        "first_shape": list(a_shape),
        "second_shape": list(b_shape),
        "common_coarse_shape": list(coarse.shape),
        "refinement_factors": list(factors),
        "relative_l2_power_density_difference": l2,
        "total_variation_power_fraction": total_variation,
        "fine_subvoxel_unresolved_relative_l2": unresolved,
        "fine_to_coarse_power_integral_relative_error": abs(
            energy_fine - energy_coarsened
        ) / energy_fine,
        "normalization_score": first.score,
        "power_w": first.normalized_power_w,
        "interpretation": (
            "Same physical bounds and deposition estimator. Fine voxel "
            "powers were conservatively summed into parent voxels. "
            "Agreement at coarse scale does not prove absence of fine "
            "hotspots, Monte Carlo source convergence, or reactor fidelity."
        ),
    }
