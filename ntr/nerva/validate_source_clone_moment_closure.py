"""Adversarial numerical validation of actual neutron fission clone moments.

Synthetic data verify weighted statistical identities and provenance guards;
only the separate real four-seed OpenMC workflow supplies physical evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import h5py
import numpy as np

from .source_clone_moment_closure import (
    partition_clone_moments, read_actual_clone_partition, audit_real_clone_moments,
)
from .source_direction_physics_audit import _sha256, audit_real_source_angular_physics
from .validate_source_direction_physics_audit import fake


def reject(fn, token: str):
    try:
        fn()
    except ValueError as error:
        assert token in str(error), str(error)
    else:
        raise AssertionError("invalid source clone physical moments accepted: " + token)


def main() -> int:
    xyz=np.array([[0.,0.,0.],[0.,0.,0.],[1.,0.,0.],[1.,0.,0.]])
    dirs=np.array([[1.,0.,0.],[-1.,0.,0.],
                   [0.,1.,0.],[0.,-1.,0.]])
    energy=np.array([2.,2.,8.,8.])
    weight=np.ones(4)
    exact=partition_clone_moments(xyz,dirs,energy,weight)
    assert exact["exact_distinct_fission_source_positions"]==2
    assert exact["inverse_simpson_effective_source_positions"]==2
    assert exact["maximum_exact_position_particle_multiplicity"]==2
    assert np.allclose(exact["mean_source_direction"],[0,0,0],atol=1e-12)
    assert np.allclose(exact["weighted_total_direction_covariance"],
                       np.diag([.5,.5,0]))
    assert np.allclose(exact["weighted_within_exact_position_direction_covariance"],
                       np.diag([.5,.5,0]))
    assert np.allclose(exact["weighted_between_exact_position_direction_covariance"],0)
    assert exact["weighted_total_emission_energy_variance_eV2"]==9.
    assert exact["weighted_within_exact_position_energy_variance_eV2"]==0
    assert exact["weighted_between_exact_position_energy_variance_eV2"]==9.
    assert exact["fraction_total_direction_variance_within_exact_position"]==1.
    assert exact["fraction_total_energy_variance_within_exact_position"]==0
    assert not exact["direction_energy_cloning_is_independent_fission_genealogy"]

    within=partition_clone_moments(xyz,dirs,[2.,8.,2.,8.],weight)
    assert abs(within["weighted_within_exact_position_energy_variance_eV2"]-9.)<1e-12
    assert abs(within["weighted_between_exact_position_energy_variance_eV2"])<1e-12

    unequal=partition_clone_moments(xyz,dirs,energy,[1.,3.,2.,4.])
    assert unequal["direction_covariance_partition_relative_residual"]<5e-12
    assert unequal["energy_variance_partition_relative_residual"]<5e-12
    assert unequal["energy_direction_partition_relative_residual"]<5e-12
    assert unequal["inverse_simpson_effective_source_positions"]<2
    reject(lambda:partition_clone_moments(xyz,dirs,energy,[1.,0.,1.,1.]),
           "invalid actual weighted")
    reject(lambda:partition_clone_moments(xyz,dirs,energy,[1.,-1.,1.,1.]),
           "invalid actual weighted")
    reject(lambda:partition_clone_moments(xyz,dirs,energy,[1.,np.nan,1.,1.]),
           "invalid actual weighted")
    reject(lambda:partition_clone_moments(xyz,dirs,energy,[1.,1.,1.,np.inf]),
           "invalid actual weighted")
    reject(lambda:partition_clone_moments(xyz,dirs,[2.,2.,0.,8.],weight),
           "invalid actual weighted")
    reject(lambda:partition_clone_moments(xyz,dirs,[2.,2.,np.nan,8.],weight),
           "invalid actual weighted")
    reject(lambda:partition_clone_moments(xyz,dirs*2,energy,weight),
           "non-unit")
    reject(lambda:partition_clone_moments(xyz,dirs[:2],energy,weight),
           "invalid actual weighted")

    seeds=(13007,23011,33013,43019)
    batches=(40,60,80)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        runs=[]
        for seed in seeds:
            folder=root/f"seed_{seed}"
            folder.mkdir()
            for batch in batches:
                fake(folder/f"statepoint.{batch:02}.h5",seed,batch)
            final=folder/"statepoint.80.h5"
            runs.append({"seed":seed,"statepoint_path":str(final),
                         "statepoint_sha256":_sha256(final)})
        summary=root/"transport_summary.json"
        summary.write_text(json.dumps({
            "execution":"ACTUAL_OPENMC_TRANSPORT",
            "assembly":"core",
            "runs":runs,"batches":80,
            "saved_source_bank_batches":list(batches),
        }),encoding="utf-8")
        direc=root/"direction.json"
        direc.write_text(json.dumps(audit_real_source_angular_physics(summary)),
                         encoding="utf-8")
        audit=audit_real_clone_moments(summary,direc)
        assert audit["independent_seed_count"]==4
        assert audit["matched_checkpoint_batches"]==[40,60,80]
        assert audit["all_20_real_checkpoint_moment_partitions_closed"]
        assert audit["minimum_effective_source_site_fraction"]==2/8
        assert audit["maximum_direction_covariance_partition_relative_residual"]<5e-12
        assert not audit["empirical_covariance_partition_is_fission_source_stationarity"]
        assert not audit["cloned_source_sites_are_independent_neutron_histories"]
        assert len(audit["seeds"])==4
        assert all(len(a["checkpoints"])==3 for a in audit["seeds"])
        for run in runs:
            row=read_actual_clone_partition(Path(run["statepoint_path"]))
            assert row["statepoint_sha256"]==run["statepoint_sha256"]
            assert row["exact_distinct_fission_source_positions"]==2

        # Tamper with readout of the *same* source bank: the persisted
        # orientation report's SHA must agree with the actual HDF5.
        mismatch=json.loads(direc.read_text(encoding="utf-8"))
        mismatch["independent_seed_real_source_records"][0]["checkpoints"][0][
            "actual_statepoint_sha256"]="f"*64
        direc.write_text(json.dumps(mismatch),encoding="utf-8")
        reject(lambda:audit_real_clone_moments(summary,direc),
               "real source checkpoint hash")
        direc.write_text(json.dumps(audit_real_source_angular_physics(summary)),
                         encoding="utf-8")

        fake(Path(runs[0]["statepoint_path"]),seeds[0],80,angle_bad=True)
        reject(lambda:read_actual_clone_partition(Path(runs[0]["statepoint_path"])),
               "unit vector")
        reject(lambda:audit_real_clone_moments(summary,direc),
               "SHA-256 provenance mismatch")

        fake(Path(runs[0]["statepoint_path"]),seeds[0],80)
        with h5py.File(runs[0]["statepoint_path"],"r+") as h:
            b=h["source_bank"][()]
            b["wgt"][0]=-1.
            h["source_bank"][...]=b
        reject(lambda:read_actual_clone_partition(Path(runs[0]["statepoint_path"])),
               "nonphysical neutron source")

    print("NERVA source clone-conditioned angular and energy moments: PASS")
    print("  synthetic identities only; real OpenMC independent transport separate")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
