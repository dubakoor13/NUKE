"""Synthetic validation of multi-seed OpenMC heating ensemble assessment.

No real cross sections or OpenMC neutron transport are used in this test.
It verifies sample standard error, energy closure, provenance rejection,
decision thresholds and the JSON/NPZ command-line workflow.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

from .openmc_replica_ensemble import (
    assess_replica_ensemble, evaluate_ensemble_directories
)
from .openmc_replica_convergence import load_mesh_replica
from .validate_openmc_replica_convergence import _write_replica


def expect_error(callback, err=(ValueError, RuntimeError)):
    try:
        callback()
    except err:
        return
    raise AssertionError("expected refusal of invalid replica inputs")


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        paths=[]
        for i,shift in enumerate((-0.06,-0.02,0.02,0.06)):
            p=root/f"replica_{i}"
            p.mkdir()
            _write_replica(p,axial_shift=shift)
            paths.append(p)
        members=[load_mesh_replica(p,allow_synthetic=True)[0] for p in paths]
        result, fields = assess_replica_ensemble(
            members, max_p95_relative_sem=0.04,
            max_axial_fraction_sem=0.01,
            max_half_split_total_variation=0.04,
        )
        assert result.status=="PASS"
        assert result.replica_count==4
        assert result.active_voxel_count==12
        assert not result.independent_real_statepoints
        assert math.isclose(result.reactor_power_w,1200.)
        # Analytic spread: deviations [-.06,-.02,.02,.06].
        # Sample std = sqrt(0.008/3); SEM divides by sqrt(4).
        expected_sem=math.sqrt(0.008/3)/2
        assert math.isclose(result.p95_active_relative_sem,expected_sem,
                            rel_tol=1e-12)
        assert math.isclose(result.max_axial_fraction_sem,expected_sem/3,
                            rel_tol=1e-12)
        # First two replicates have -0.04 axial bias, latter +0.04.
        assert math.isclose(result.half_split_total_variation,0.08/3,
                            rel_tol=1e-12)
        assert result.balanced_splits_exhaustive
        assert result.balanced_splits_tested == 3
        assert math.isclose(
            result.maximum_balanced_split_total_variation,
            0.08/3, rel_tol=1e-12
        )
        assert result.maximum_leave_one_out_total_variation <= (
            result.tolerance_max_leave_one_out_total_variation
        )
        assert result.mean_pairwise_total_variation>0
        assert result.maximum_leave_one_out_total_variation>0
        assert fields["mean_power_fraction"].shape==(2,2,3)
        assert math.isclose(fields["mean_power_fraction"].sum(),1.,
                            rel_tol=1e-14)
        assert np.allclose(fields["mean_power_fraction"],1/12,atol=1e-14)
        assert np.allclose(fields["sem_power_fraction"][:,:,1],0,atol=1e-16)
        assert np.all(fields["active_mask"])
        assert math.isclose(fields["axial_power_fraction_mean"].sum(),1.,
                            rel_tol=1e-14)

        bad, _ = assess_replica_ensemble(
            members,max_p95_relative_sem=0.02,
            max_axial_fraction_sem=0.01,
            max_half_split_total_variation=0.04,
        )
        assert bad.status=="FAIL"
        assert not bad.p95_active_relative_sem <= bad.tolerance_p95_relative_sem

        bad_axial, _ = assess_replica_ensemble(
            members,max_p95_relative_sem=0.04,
            max_axial_fraction_sem=0.001,
            max_half_split_total_variation=0.04,
        )
        assert bad_axial.status=="FAIL"
        bad_split, _ = assess_replica_ensemble(
            members,max_p95_relative_sem=0.04,
            max_axial_fraction_sem=0.01,
            max_half_split_total_variation=0.01,
        )
        assert bad_split.status=="FAIL"

        # Regression for the original input-order loophole: alternating
        # strongly divergent maps can make the naive half-split look near
        # zero, although another balanced split has large spatial TV.
        shuffled = [members[i] for i in (0, 3, 1, 2)]
        shuffled_qa, _ = assess_replica_ensemble(
            shuffled,
            max_p95_relative_sem=0.04,
            max_axial_fraction_sem=0.01,
            max_half_split_total_variation=0.01,
            max_leave_one_out_total_variation=1.0,
        )
        assert shuffled_qa.half_split_total_variation < 1e-12
        assert shuffled_qa.status=="FAIL"
        assert math.isclose(
            shuffled_qa.maximum_balanced_split_total_variation,
            result.maximum_balanced_split_total_variation,
            rel_tol=1e-12,
        )

        # Leave-one-out is also an active gate, unlike the previous code.
        # An innocuous half split cannot hide an extreme single run.
        loo_bad, _ = assess_replica_ensemble(
            shuffled,
            max_p95_relative_sem=1.,
            max_axial_fraction_sem=1.,
            max_half_split_total_variation=1.,
            max_leave_one_out_total_variation=0.02,
        )
        assert loo_bad.status == "FAIL"
        assert loo_bad.maximum_leave_one_out_total_variation > 0.02

        # Exhaustive odd-sized split requires C(5,2)=10 partitions.
        odd, _ = assess_replica_ensemble(
            members + [members[1]],
            max_p95_relative_sem=0.1,
        )
        assert odd.balanced_splits_exhaustive
        assert odd.balanced_splits_tested == 10

        # For larger ensembles the explicitly approximate fallback must
        # still be input-order INVARIANT for its acceptance decision.
        many = members * 4
        large_a, _ = assess_replica_ensemble(
            many, max_p95_relative_sem=0.1
        )
        large_b, _ = assess_replica_ensemble(
            many[::-1], max_p95_relative_sem=0.1
        )
        assert not large_a.balanced_splits_exhaustive
        assert large_a.maximum_balanced_split_total_variation is None
        assert large_a.status == large_b.status
        assert math.isclose(
            large_a.maximum_leave_one_out_total_variation,
            large_b.maximum_leave_one_out_total_variation,
            rel_tol=1e-12,
        )

        # A changed absolute nuclear power or mesh is a hard error.
        expect_error(lambda:assess_replica_ensemble(members[:3]))
        expect_error(lambda:assess_replica_ensemble(
            members[:3]+[replace(members[3],reactor_power_w=2000.)]
        ))
        expect_error(lambda:assess_replica_ensemble(
            members[:3]+[replace(members[3],dimension=(2,2,4))]
        ))
        expect_error(lambda:assess_replica_ensemble(
            members,max_p95_relative_sem=float("nan")
        ))
        expect_error(lambda:assess_replica_ensemble(
            members,active_threshold_relative=1.0
        ))
        expect_error(lambda:evaluate_ensemble_directories(
            paths,allow_synthetic=False
        ))
        expect_error(lambda:evaluate_ensemble_directories(
            paths,allow_synthetic=True,provenance_paths=[root/"fake"]*4
        ))

        # Simulate provenance objects WITHOUT ever claiming these came
        # from real transport. This only unit-tests identity/seed checks.
        real_objects=[]
        provenance=[]
        for i,rep in enumerate(members):
            real_objects.append(replace(
                rep,source_kind="unit-test-pseudo-real",
                statepoint_sha256=f"unit-test-sp-{i}",
                nuclear_data_sha256="unit-test-xs",
                openmc_seed=1234+i,
            ))
            provenance.append({
                "assembly":"reactor",
                "config":{"seed":1234+i,"particles":4000,"batches":80},
                "nuclear_data":{
                    "cross_sections_sha256":"unit-test-xs",
                    "explicit_temperature_coverage_ok":True
                },
                "xml_sha256":{
                    "geometry.xml":"unit-g",
                    "materials.xml":"unit-m",
                    "tallies.xml":"unit-t"
                }
            })
        good_real, _ = assess_replica_ensemble(
            real_objects,provenance=provenance,
            max_p95_relative_sem=0.04,
        )
        assert good_real.independent_real_statepoints
        assert good_real.status=="PASS"
        expect_error(lambda:assess_replica_ensemble(
            real_objects[:3]+[replace(real_objects[3],openmc_seed=1234)],
            provenance=provenance
        ))
        expect_error(lambda:assess_replica_ensemble(
            real_objects,provenance=provenance[:3]
        ))
        invalid_provenance=[dict(p) for p in provenance]
        invalid_provenance[-1]={**provenance[-1],
                                "xml_sha256":{**provenance[-1]["xml_sha256"],
                                              "materials.xml":"different"}}
        expect_error(lambda:assess_replica_ensemble(
            real_objects,provenance=invalid_provenance
        ))

        # Real CLI entrypoint invoked against synthetic data:
        # verify exported machine-readable report and mean+SEM arrays.
        from .run_openmc_replica_ensemble import main as cli
        argv=sys.argv
        sys.argv=[
            "run_openmc_replica_ensemble",
            "--power-dirs",*[str(p) for p in paths],
            "--allow-synthetic",
            "--max-p95-relative-sem","0.04",
            "--output",str(root/"output"),
        ]
        try:
            assert cli()==0
        finally:
            sys.argv=argv
        report=json.loads((root/"output"/"ensemble_convergence_report.json")
                          .read_text(encoding="utf-8"))
        assert report["source_mode"]=="synthetic_unverified"
        assert report["summary"]["status"]=="PASS"
        assert len(report["replicas"])==4
        with np.load(root/"output"/"ensemble_heating_statistics.npz",
                     allow_pickle=False) as archived:
            assert np.allclose(archived["mean_power_fraction"],1/12)

    print(
        "Multi-seed heating ensemble numerical QA: PASS; "
        f"four-replica p95 SEM={expected_sem:.8g}, "
        "half-split TV=0.02666667 (synthetic only)"
    )
    return 0


if __name__=="__main__":
    raise SystemExit(main())
