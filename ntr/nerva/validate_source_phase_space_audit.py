"""Synthetic regression of joint x/y/z/E source-transport HDF5 numerics.

The test files are ARTIFICIAL. Physical measurements are generated separately
from genuine C++ OpenMC source-bank statepoints in the real-core CI workflow.
"""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .source_phase_space_audit import (
    _sha, audit_joint_phase_space, audit_real_joint_summary,
    joint_position_energy_bank,
)
from .source_resolution_audit import audit_source_grids
from .source_energy_audit import assess_source_energy_trajectories


def create_hdf5(path, *, seed, batch, coords, energies,
                weights=None, photon_transport=0):
    dtype = np.dtype([
        ("r", np.dtype([("x","f8"),("y","f8"),("z","f8")])),
        ("E", "f8"), ("wgt","f8"), ("particle","i4"),
    ])
    data = np.zeros(len(energies), dtype=dtype)
    for j, axis in enumerate(("x", "y", "z")):
        data["r"][axis] = coords[:,j]
    data["E"] = energies
    data["wgt"] = 1 if weights is None else weights
    with h5py.File(path, "w") as h:
        h.attrs["filetype"] = np.bytes_("statepoint")
        h.attrs["source_present"] = 1
        h.attrs["photon_transport"] = photon_transport
        h.create_dataset("run_mode", data=np.bytes_("eigenvalue"))
        h.create_dataset("seed", data=seed)
        h.create_dataset("current_batch", data=batch)
        h.create_dataset("source_bank", data=data)


def reject(action, substring):
    try:
        action()
    except ValueError as error:
        assert substring in str(error), (substring, str(error))
    else:
        raise AssertionError("invalid OpenMC source phase-space QA accepted: " + substring)


def main():
    seeds = (13007, 23011, 33013, 43019)
    batches = (40,80,120)
    lo, hi = (-5.,-5.,-66.), (5.,5.,66.)
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        trajectories = []
        for i, seed in enumerate(seeds):
            folder = root / f"seed_{seed}"
            folder.mkdir()
            run = []
            rng = np.random.default_rng(seed)
            for batch in batches:
                xyz = rng.uniform(-3.,3.,(350,3))
                xyz[:,2] *= 10
                energy = rng.uniform(1e5,7e6,350)
                # Final spatial distributions disagree between seeds,
                # while source-energy distributions still have overlap.
                if batch == 120:
                    xyz[:,0] += float(i)*0.35
                filename = folder / f"statepoint.{batch:03d}.h5"
                create_hdf5(filename,seed=seed,batch=batch,
                            coords=xyz,energies=energy)
                run.append(filename)
            trajectories.append(run)
        report = audit_joint_phase_space(
            trajectories,lower_cm=lo,upper_cm=hi
        )
        assert report["independent_seed_count"] == 4
        assert report["saved_source_batches"] == list(batches)
        assert [r["spatial_bins_per_axis"] for r in report["resolutions"]] == [2,4,8]
        assert report["energy_spatial_marginal_and_nested_conservation_passed"]
        assert report["fission_source_stationarity_validated"] is False
        assert report["core_neutron_transport_physics_historically_validated"] is False
        for resolution in report["resolutions"]:
            for metric in ("mean_cross_seed_tv_per_checkpoint",
                           "mean_within_seed_adjacent_tv_per_interval"):
                joint = resolution["joint"][metric]
                spatial = resolution["spatial_marginal"][metric]
                energy = resolution["energy_marginal"][metric]
                assert all(a + 1e-12 >= b and a + 1e-12 >= c
                           for a,b,c in zip(joint,spatial,energy))
        joint2=report["resolutions"][0]["joint"][
            "mean_cross_seed_tv_per_checkpoint"][-1]
        joint8=report["resolutions"][-1]["joint"][
            "mean_cross_seed_tv_per_checkpoint"][-1]
        assert joint8 + 1e-12 >= joint2

        spatial_ref=audit_source_grids(
            trajectories, lower_cm=lo, upper_cm=hi,
        )
        energy_ref=assess_source_energy_trajectories(trajectories)
        summary=root/"summary.json"
        summary.write_text(json.dumps({
            "execution":"ACTUAL_OPENMC_TRANSPORT",
            "assembly":"core",
            "runs":[{
                "seed":seed, "statepoint_path":str(t[-1]),
                "statepoint_sha256":_sha(t[-1])
            } for seed,t in zip(seeds,trajectories)],
            "saved_source_bank_batches":list(batches),
            "batches":120,
            "independent_seed_source_bank_shape_diagnostics":{
                "spatial_lower_cm":lo, "spatial_upper_cm":hi,
            },
        }))
        spatial_ref["real_transport_summary_sha256"]=_sha(summary)
        energy_ref["real_transport_summary_sha256"]=_sha(summary)
        spatial_file=root/"spatial.json"
        energy_file=root/"energy.json"
        spatial_file.write_text(json.dumps(spatial_ref))
        energy_file.write_text(json.dumps(energy_ref))
        verified=audit_real_joint_summary(
            summary,spatial_reference=spatial_file,energy_reference=energy_file
        )
        assert verified["energy_spatial_marginal_and_nested_conservation_passed"]
        assert verified["real_transport_summary_sha256"]==_sha(summary)

        forged=dict(energy_ref)
        forged["mean_cross_seed_energy_spectrum_tv_by_checkpoint"] = [0,0,0]
        energy_file.write_text(json.dumps(forged))
        reject(lambda:audit_real_joint_summary(
            summary,spatial_reference=spatial_file,energy_reference=energy_file
        ), "energy marginal differs")
        summary.write_text(json.dumps({"execution":"SYNTHETIC"}))
        reject(lambda:audit_real_joint_summary(
            summary,spatial_reference=spatial_file,energy_reference=energy_file
        ), "genuine")

        reject(lambda:audit_joint_phase_space(
            trajectories[:3], lower_cm=lo,upper_cm=hi
        ),"four independent")
        reject(lambda:audit_joint_phase_space(
            [trajectories[0]]*4, lower_cm=lo,upper_cm=hi
        ),"duplicate")
        reject(lambda:audit_joint_phase_space(
            [trajectories[0][::-1]]+trajectories[1:],
            lower_cm=lo,upper_cm=hi
        ),"not increasing")
        reject(lambda:audit_joint_phase_space(
            trajectories,lower_cm=lo,upper_cm=hi,sides=(4,2)
        ),"strictly nested")

        corrupt=root/"corrupt.h5"
        rng=np.random.default_rng(81)
        xyz=rng.uniform(-1,1,(30,3))
        energies=np.full(30,2e6)
        create_hdf5(corrupt,seed=1901,batch=120,coords=xyz,energies=energies,
                    photon_transport=1)
        reject(lambda:joint_position_energy_bank(
            corrupt,lower_cm=lo,upper_cm=hi
        ),"neutron-only")
        create_hdf5(corrupt,seed=1901,batch=120,coords=xyz,energies=energies,
                    weights=np.zeros(30))
        reject(lambda:joint_position_energy_bank(
            corrupt,lower_cm=lo,upper_cm=hi
        ),"nonphysical")
        energies[3]=float("nan")
        create_hdf5(corrupt,seed=1901,batch=120,coords=xyz,energies=energies)
        reject(lambda:joint_position_energy_bank(
            corrupt,lower_cm=lo,upper_cm=hi
        ),"nonphysical")
        energies[3]=31e6
        create_hdf5(corrupt,seed=1901,batch=120,coords=xyz,energies=energies)
        reject(lambda:joint_position_energy_bank(
            corrupt,lower_cm=lo,upper_cm=hi
        ),"energy window")
        energies[3]=2e6
        xyz[3,0]=15.
        create_hdf5(corrupt,seed=1901,batch=120,coords=xyz,energies=energies)
        reject(lambda:joint_position_energy_bank(
            corrupt,lower_cm=lo,upper_cm=hi
        ),"outside common")

    print("NERVA 4D fission-source spatial-energy conservation identities: PASS")
    print("  synthetic HDF5 fixtures only; real C++ OpenMC CI runs separately")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
