"""Numerical tests for independent-replica OpenMC mesh-refinement screen.

Synthetic normalized power fields only; no neutron transport or NASA
benchmark data. All randomness is seeded for deterministic regression.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

import numpy as np

from .mesh_refinement import HeatingMesh
from .mesh_refinement_statistics import assess_independent_mesh_refinement
from .run_openmc_mesh_refinement_ensemble import (
    evaluate_mesh_refinement_directories,
    main as cli,
)


def _fields(
    *,
    count: int = 6,
    strength: float = 0.003,
    bias: float = 0.,
    subvoxel: float = 0.,
    shape: tuple[int, int, int] = (4, 4, 4),
    seed: int = 20261021,
) -> list[HeatingMesh]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-1., 1., 4)[:,None,None]
    y = np.linspace(-1., 1., 4)[None,:,None]
    z = np.linspace(-1., 1., 4)[None,None,:]
    coarse = np.broadcast_to(2.0 + .2*x + .1*y + .1*z, (4,4,4)).copy()
    if shape == (8,8,8):
        base = np.repeat(np.repeat(np.repeat(coarse,2,axis=0),2,axis=1),2,axis=2)
        left_right = np.repeat(np.sign(x),2,axis=0)
        base = base * (1. + bias * left_right)
        alternating = np.where(np.arange(8)%2 == 0, 1., -1.)[:,None,None]
        base = base * (1. + subvoxel * alternating)
    elif shape == (4,4,4):
        base = coarse
    else:
        raise AssertionError("unsupported synthetic fixture")
    p = 1000.
    volume = 64./np.prod(shape)
    result = []
    for _ in range(count):
        density = base * (1. + strength * rng.normal(size=shape))
        density = np.maximum(density,1e-10)
        density *= p/(float(np.sum(density))*volume)
        result.append(HeatingMesh(
            density,np.array([-2.,-2.,-2.]),np.array([2.,2.,2.]),
            p,"heating-local",False
        ))
    return result


def _write(directory: Path, mesh: HeatingMesh) -> None:
    directory.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        directory/"nerva_mesh_fields.npz",
        power_density_w_cm3=mesh.density_w_cm3,
        power_density_std_w_cm3=np.zeros_like(mesh.density_w_cm3),
        dimension=np.asarray(mesh.density_w_cm3.shape,dtype=int),
        lower_left_cm=mesh.lower_left_cm,
        upper_right_cm=mesh.upper_right_cm,
    )
    (directory/"metadata.json").write_text(json.dumps({
        "requested_power_MW":mesh.normalized_power_w/1e6,
        "normalized_power_W":mesh.normalized_power_w,
        "power_normalization_score":mesh.score,
        "photon_transport":mesh.photon_transport,
        "statepoint":"DELIBERATE-SYNTHETIC-FIXTURE",
    }),encoding="utf-8")


def reject(action, expected: str) -> None:
    try:
        action()
    except ValueError as e:
        assert expected in str(e), str(e)
    else:
        raise AssertionError(f"should reject {expected}")


def main() -> int:
    coarse = _fields()
    fine = _fields(shape=(8,8,8),seed=38911)
    nominal, fields = assess_independent_mesh_refinement(
        coarse,fine,maximum_screen_relative_l2=.02
    )
    assert nominal.screening_status == "SCREEN_PASS_NOT_PHYSICS_VALIDATED"
    assert nominal.coarse_replicas == nominal.fine_replicas == 6
    assert nominal.refinement_factors == (2,2,2)
    assert nominal.observed_common_grid_relative_l2 < .01
    assert nominal.common_grid_sampling_sem_relative_l2 > 0
    assert nominal.fine_subvoxel_sampling_sem_relative_l2 > 0
    assert fields["coarse_mean_power_fraction"].shape == (4,4,4)
    assert fields["fine_mean_power_fraction"].shape == (8,8,8)
    assert np.isclose(fields["coarse_mean_power_fraction"].sum(),1.)
    assert np.isclose(fields["fine_projected_mean_power_fraction"].sum(),1.)
    assert np.isclose(fields["common_grid_difference_power_fraction"].sum(),0.,atol=1e-12)
    assert np.isclose(fields["fine_subvoxel_structure"].sum(),0.,atol=1e-12)

    # Fine-scale hotspots do not become spurious coarse-map bias.
    tiny_hotspots, _ = assess_independent_mesh_refinement(
        coarse,_fields(shape=(8,8,8),subvoxel=.07,seed=38911),
        maximum_screen_relative_l2=.02
    )
    assert tiny_hotspots.observed_common_grid_relative_l2 < .02
    assert tiny_hotspots.fine_subvoxel_structure_relative_l2 > .06

    biased, _ = assess_independent_mesh_refinement(
        coarse,_fields(shape=(8,8,8),bias=.10,seed=38911),
        maximum_screen_relative_l2=.02
    )
    assert biased.screening_status == "SCREEN_FAIL_REFINEMENT_DIFFERENCE"
    assert biased.observed_common_grid_relative_l2 > .05

    noisy, _ = assess_independent_mesh_refinement(
        _fields(strength=.5,seed=44),
        _fields(shape=(8,8,8),strength=.5,seed=45),
        maximum_sampling_sem_relative_l2=.02
    )
    assert noisy.screening_status == "INCONCLUSIVE_SAMPLING_NOISE"

    # A single coarse+fine pair is not an independent-sample SEM.
    reject(lambda: assess_independent_mesh_refinement(coarse[:1],fine[:1]),
           "four independent samples")
    reject(lambda: assess_independent_mesh_refinement(coarse,fine,
                  maximum_screen_relative_l2=float("nan")), "finite")
    reject(lambda: assess_independent_mesh_refinement(coarse,
                  [replace(f,score="heating",photon_transport=True) for f in fine]),
           "estimators")
    reject(lambda: assess_independent_mesh_refinement(coarse,
                  [replace(f,normalized_power_w=500) for f in fine]),
           "does not close")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        coarse_dirs = [root/f"coarse_{i}" for i in range(6)]
        fine_dirs = [root/f"fine_{i}" for i in range(6)]
        for path, obj in zip(coarse_dirs,coarse):
            _write(path,obj)
        for path, obj in zip(fine_dirs,fine):
            _write(path,obj)

        summary, _, identities = evaluate_mesh_refinement_directories(
            coarse_dirs,fine_dirs,allow_synthetic=True,
            maximum_screen_relative_l2=.02
        )
        assert summary.screening_status == nominal.screening_status
        assert len(identities) == 12
        assert all(i["statepoint_sha256"] is None for i in identities)
        reject(lambda: evaluate_mesh_refinement_directories(
            coarse_dirs[:3],fine_dirs,allow_synthetic=True
        ),">=4")
        reject(lambda: evaluate_mesh_refinement_directories(
            coarse_dirs,fine_dirs,allow_synthetic=False
        ),"provenance")

        old_argv = sys.argv
        output = root/"report"
        sys.argv=[
            "mesh_refinement_ensemble",
            "--coarse-power-dirs",*[str(p) for p in coarse_dirs],
            "--fine-power-dirs",*[str(p) for p in fine_dirs],
            "--allow-synthetic","--max-screen-relative-l2","0.02",
            "--output",str(output),
        ]
        try:
            assert cli()==0
        finally:
            sys.argv=old_argv
        report=json.loads((output/"refinement_report.json").read_text())
        assert report["source_mode"]=="synthetic_unverified"
        assert report["historical_nerva_validation"] is False
        assert report["real_openmc_executed_by_this_command"] is False
        assert report["summary"]["screening_status"] == nominal.screening_status
        with np.load(output/"refinement_statistics.npz",allow_pickle=False) as data:
            assert data["fine_projected_mean_power_fraction"].shape==(4,4,4)
            assert data["fine_subvoxel_structure_sem"].shape==(8,8,8)

    print("NERVA independent-replica mesh refinement and sampling-noise QA: PASS")
    print("  actual OpenMC neutron transport executed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
