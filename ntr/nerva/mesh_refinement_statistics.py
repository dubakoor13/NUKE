"""Independent-seed Monte Carlo uncertainty for NERVA mesh refinement.

Compare mean power-normalized OpenMC heating fractions at two nested mesh
resolutions. Independently seeded statepoints are required for legitimate
sampling statistics. The coarse-vs-fine difference and noise are distinct:
Monte Carlo variance cannot be mistaken for a mesh-discretization error.

No real OpenMC transport is performed by this module. Even a favorable
screen is NOT proof of fission-source convergence or NASA model fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np

from .mesh_refinement import HeatingMesh, compare_heating_meshes, _coarsen_volume_average


@dataclass(frozen=True)
class MeshRefinementStatistics:
    coarse_replicas: int
    fine_replicas: int
    coarse_shape: tuple[int, int, int]
    fine_shape: tuple[int, int, int]
    refinement_factors: tuple[int, int, int]
    observed_common_grid_relative_l2: float
    common_grid_sampling_sem_relative_l2: float
    upper_screen_relative_l2: float
    observed_common_grid_total_variation: float
    fine_subvoxel_structure_relative_l2: float
    fine_subvoxel_sampling_sem_relative_l2: float
    maximum_sampling_sem_relative_l2: float
    maximum_screen_relative_l2: float
    screening_status: str
    interpretation: str


def assess_independent_mesh_refinement(
    coarse: list[HeatingMesh],
    fine: list[HeatingMesh],
    *,
    maximum_sampling_sem_relative_l2: float = 0.01,
    maximum_screen_relative_l2: float = 0.05,
) -> tuple[MeshRefinementStatistics, dict[str, np.ndarray]]:
    """Estimate common-grid noise and conservative refinement screening.

    Independent-seed estimator:
       SEM_p² = sample_variance(p) / n
       ||SEM_delta||² = sum_bins(SEM_coarse² + SEM_fine_to_coarse²)

    The two mesh ensembles must not share seeds or histories; this function
    verifies spatial/normalization identity only. The data-loading wrapper
    is responsible for provenance and independent-seed enforcement.

    The screen uses ||delta_mean|| + 2 ||SEM_delta|| on normalized power
    fractions relative to ||coarse_mean||. The 2x factor is a conservative
    descriptive margin, NOT a simultaneous confidence interval or p-value.
    """
    if len(coarse) < 4 or len(fine) < 4:
        raise ValueError("at least four independent samples per mesh level required")
    for name, value in (
        ("maximum_sampling_sem_relative_l2", maximum_sampling_sem_relative_l2),
        ("maximum_screen_relative_l2", maximum_screen_relative_l2),
    ):
        if not math.isfinite(value) or value <= 0 or value >= 1:
            raise ValueError(f"{name} must be finite within (0,1)")

    first_coarse, first_fine = coarse[0], fine[0]
    reference = compare_heating_meshes(first_coarse, first_fine)
    shape_c = tuple(reference["common_coarse_shape"])
    shape_f = tuple(first_fine.validate()[0].shape)
    shape_a = tuple(first_coarse.validate()[0].shape)
    if shape_c != shape_a or any(x > y for x,y in zip(shape_a,shape_f)):
        raise ValueError("first group must be coarse and second group fine")
    if all(x == y for x,y in zip(shape_a,shape_f)):
        raise ValueError("refinement must increase at least one mesh dimension")
    factors = tuple(reference["refinement_factors"])
    if int(np.prod(factors)) <= 1:
        raise ValueError("mesh refinement factor must exceed unity")

    common_coarse = []
    fine_as_common = []
    full_fine = []
    all_items = (coarse, fine)
    for level, replicas in enumerate(all_items):
        for entry in replicas:
            base = first_coarse if level == 0 else first_fine
            if compare_heating_meshes(base, entry)["comparison_type"] != "same_grid":
                raise ValueError("mesh dimensions change within one seed ensemble")
            array, _, _, voxel_cm3 = entry.validate()
            fractions = array * (voxel_cm3 / entry.normalized_power_w)
            if not np.isclose(float(fractions.sum()),1.,rtol=0,atol=1e-7):
                raise ValueError("replica heating fractions do not conserve power")
            if level == 0:
                common_coarse.append(fractions)
            else:
                full_fine.append(fractions)
                average, _ = _coarsen_volume_average(fractions, shape_c)
                fine_as_common.append(average * int(np.prod(factors)))

    group_c = np.stack(common_coarse,axis=0)
    group_f = np.stack(fine_as_common,axis=0)
    stack_f_full = np.stack(full_fine,axis=0)
    mean_c = np.mean(group_c,axis=0)
    mean_f = np.mean(group_f,axis=0)
    sem_c = np.std(group_c,axis=0,ddof=1) / math.sqrt(len(coarse))
    sem_f = np.std(group_f,axis=0,ddof=1) / math.sqrt(len(fine))
    sem_delta = np.hypot(sem_c,sem_f)
    norm = float(np.linalg.norm(mean_c.ravel()))
    if norm <= 0 or not math.isfinite(norm):
        raise ValueError("coarse mean normalized heating is invalid")
    delta = mean_f - mean_c
    observed = float(np.linalg.norm(delta.ravel()) / norm)
    noise = float(np.linalg.norm(sem_delta.ravel()) / norm)
    upper = observed + 2.*noise
    variation = 0.5*float(np.sum(np.abs(delta)))

    fine_mean = np.mean(stack_f_full, axis=0)
    fine_parent, _ = _coarsen_volume_average(fine_mean, shape_c)
    parent_expanded = fine_parent.copy()
    for axis, r in enumerate(factors):
        parent_expanded = np.repeat(parent_expanded, r, axis=axis)
    structure = fine_mean - parent_expanded

    # Estimate sampling noise in the subvoxel STRUCTURE, not full field.
    subvoxel_replicas = []
    for realization in stack_f_full:
        parent, _ = _coarsen_volume_average(realization, shape_c)
        expanded = parent
        for axis, r in enumerate(factors):
            expanded = np.repeat(expanded, r, axis=axis)
        subvoxel_replicas.append(realization - expanded)
    subvoxel_sem = np.std(np.stack(subvoxel_replicas,axis=0),axis=0,ddof=1) / math.sqrt(len(fine))
    denom_full = float(np.linalg.norm(fine_mean.ravel()))
    if denom_full <= 0:
        raise ValueError("fine mean heating is zero")
    fine_structure = float(np.linalg.norm(structure.ravel()) / denom_full)
    structure_noise = float(np.linalg.norm(subvoxel_sem.ravel()) / denom_full)

    # A low observed difference at an unresolved noise level is NOT PASS.
    if noise > maximum_sampling_sem_relative_l2:
        status = "INCONCLUSIVE_SAMPLING_NOISE"
    elif upper <= maximum_screen_relative_l2:
        status = "SCREEN_PASS_NOT_PHYSICS_VALIDATED"
    elif observed - 2.*noise > maximum_screen_relative_l2:
        status = "SCREEN_FAIL_REFINEMENT_DIFFERENCE"
    else:
        status = "INCONCLUSIVE_REFINEMENT_DIFFERENCE"

    result = MeshRefinementStatistics(
        coarse_replicas=len(coarse),
        fine_replicas=len(fine),
        coarse_shape=shape_c,
        fine_shape=shape_f,
        refinement_factors=factors,
        observed_common_grid_relative_l2=observed,
        common_grid_sampling_sem_relative_l2=noise,
        upper_screen_relative_l2=upper,
        observed_common_grid_total_variation=variation,
        fine_subvoxel_structure_relative_l2=fine_structure,
        fine_subvoxel_sampling_sem_relative_l2=structure_noise,
        maximum_sampling_sem_relative_l2=maximum_sampling_sem_relative_l2,
        maximum_screen_relative_l2=maximum_screen_relative_l2,
        screening_status=status,
        interpretation=(
            "Independent-replica Monte Carlo SEM on power-normalized spatial "
            "fractions at a common coarse grid. A two-SEM vector-norm margin "
            "is a numerical SCREEN, not a confidence interval. Refined "
            "subvoxel structure and its sampling SEM are reported separately. "
            "Reactor power was prescribed and is not an uncertainty result. "
            "Shared nuclear-data bias, neutron-source nonstationarity, "
            "temperature feedback, and historical NASA validation are excluded."
        ),
    )
    outputs = {
        "coarse_mean_power_fraction": mean_c,
        "fine_projected_mean_power_fraction": mean_f,
        "common_grid_difference_power_fraction": delta,
        "coarse_sem_power_fraction": sem_c,
        "fine_projected_sem_power_fraction": sem_f,
        "difference_sem_power_fraction": sem_delta,
        "fine_mean_power_fraction": fine_mean,
        "fine_subvoxel_structure": structure,
        "fine_subvoxel_structure_sem": subvoxel_sem,
    }
    return result, outputs
