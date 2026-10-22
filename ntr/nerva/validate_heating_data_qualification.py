"""Unit regression of OpenMC deposited-energy score qualification.

Synthetic math tests only; real neutron heating comes from OpenMC's
eigenvalue statepoint in the separate four-seed transport workflow.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import h5py
import numpy as np

from .heating_data_qualification import (
    evaluate_heating_scores, inspect_statepoint,
)


def reject(action, word):
    try:
        action()
    except (ValueError, FileNotFoundError) as error:
        assert word in str(error), str(error)
    else:
        raise AssertionError("physically invalid heating accepted: " + word)


def mock_statepoint(path: Path, *, local: float) -> None:
    with h5py.File(path, "w") as h:
        h.attrs["filetype"] = np.bytes_("statepoint")
        h.attrs["photon_transport"] = 0
        h.attrs["tallies_present"] = 1
        h.create_dataset("run_mode", data=np.bytes_("eigenvalue"))
        h.create_dataset("seed", data=13007)
        h.create_dataset("current_batch", data=60)
        h.create_dataset("n_inactive", data=20)
        ts = h.create_group("tallies")
        for i, (name, scores, values) in enumerate((
            ("nerva_3d_neutronics", ["heating","heating-local","kappa-fission"],
             [8e5,local,9e5]),
            ("nerva_material_transport", ["heating","heating-local"],
             [8e5,local]),
        )):
            t = ts.create_group(f"tally {i+1}")
            t.create_dataset("name", data=np.bytes_(name))
            t.create_dataset("n_realizations", data=40)
            t.create_dataset("score_bins", data=np.array(
                scores, dtype="S30"))
            values = np.asarray(values, dtype=float)
            results = np.zeros((1, len(scores), 2))
            results[0, :, 0] = values * 40
            results[0, :, 1] = values * values * 40
            t.create_dataset("results", data=results)


def main() -> int:
    mesh = {"heating":8e5,"heating-local":0.,"kappa-fission":9e5}
    material = {"heating":8e5,"heating-local":0.}
    bad = evaluate_heating_scores(mesh,material)
    assert bad["mesh_material_energy_partition_consistent"]
    assert bad["physical_heating_qualification_status"] == (
        "BLOCKED_ZERO_NEUTRON_LOCAL_HEATING"
    )
    assert bad["neutron_only_local_heating_data_usable"] is False
    assert bad["thermal_power_normalization_allowed"] is False
    assert bad["nasa_nerva_core_thermal_physics_validated"] is False

    # Positive means without genuine statistical moments cannot qualify.
    missing_stats = evaluate_heating_scores(
        {**mesh,"heating-local":7.5e5},
        {**material,"heating-local":7.5e5},
    )
    assert not missing_stats["neutron_only_local_heating_data_usable"]
    assert missing_stats["physical_heating_qualification_status"] == (
        "BLOCKED_MISSING_LOCAL_HEATING_STATISTICS"
    )
    healthy = evaluate_heating_scores(
        {**mesh,"heating-local":7.5e5},
        {**material,"heating-local":7.5e5},
        mesh_local_sem_upper_bound=1.0e4,
        material_local_sem_upper_bound=1.5e4,
    )
    assert healthy["neutron_only_local_heating_data_usable"]
    assert healthy["local_heating_statistically_resolved"]
    noisy = evaluate_heating_scores(
        {**mesh,"heating-local":7.5e5},
        {**material,"heating-local":7.5e5},
        mesh_local_sem_upper_bound=2e5,
        material_local_sem_upper_bound=2.5e5,
    )
    assert not noisy["thermal_power_normalization_allowed"]
    assert noisy["physical_heating_qualification_status"] == (
        "BLOCKED_UNRESOLVED_LOCAL_HEATING_STATISTICS"
    )
    assert healthy["nasa_nerva_core_thermal_physics_validated"] is False

    reject(lambda: evaluate_heating_scores(
        mesh,{**material,"heating":7.8e5}
    ), "consistently")
    reject(lambda: evaluate_heating_scores(
        {**mesh,"heating":0},material
    ), "nonpositive")
    reject(lambda: evaluate_heating_scores(
        {**mesh,"heating-local":float("nan")},material
    ), "nonfinite")
    reject(lambda: evaluate_heating_scores(
        mesh,material,coherence_relative_tolerance=0
    ), "tolerance")
    reject(lambda: evaluate_heating_scores(
        mesh,material,mesh_local_sem_upper_bound=float("nan")
    ), "standard-error")
    reject(lambda: evaluate_heating_scores(
        mesh,material,minimum_signal_to_sem=0
    ), "statistical-resolution")

    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp)/"fixture.h5"
        mock_statepoint(file,local=0)
        report = inspect_statepoint(file)
        assert report["source"] == "OPENMC_FORMAT_EIGENVALUE_HDF5_CONTENT"
        assert report["seed"] == 13007
        assert not report["thermal_power_normalization_allowed"]
        mock_statepoint(file,local=700000)
        report = inspect_statepoint(file)
        assert report["thermal_power_normalization_allowed"]
        assert report["active_tally_realizations"] == 40
        assert report["local_heating_statistically_resolved"]
        assert report["mesh_local_heating_conservative_sem_upper_bound"] == 0.0

        # Corrupt only the SECOND moments: means remain nonzero and
        # perfectly coherent between mesh and material, but are noisy.
        with h5py.File(file,"r+") as h:
            for name, index in (("tally 1",1),("tally 2",1)):
                data=h["tallies"][name]["results"][()]
                data[0,index,1]=40*(700000.0**2 + 39*300000.0**2)
                h["tallies"][name]["results"][...]=data
        unresolved=inspect_statepoint(file)
        assert unresolved["local_heating_eV_per_source"] == 700000.0
        assert not unresolved["thermal_power_normalization_allowed"]
        assert unresolved["physical_heating_qualification_status"] == (
            "BLOCKED_UNRESOLVED_LOCAL_HEATING_STATISTICS"
        )

        mock_statepoint(file,local=700000)
        with h5py.File(file,"r+") as h:
            h["tallies"]["tally 2"]["n_realizations"][...]=39
        reject(lambda:inspect_statepoint(file),"realizations mismatch")

        mock_statepoint(file,local=700000)
        with h5py.File(file,"r+") as h:
            data=h["tallies"]["tally 1"]["results"][()]
            data[0,1,1]=1.0
            h["tallies"]["tally 1"]["results"][...]=data
        reject(lambda:inspect_statepoint(file),"nonnegative variance")

        mock_statepoint(file,local=700000)
        with h5py.File(file,"r+") as h:
            h.attrs["photon_transport"] = 1
        reject(lambda:inspect_statepoint(file),"neutron-only")

    print("NERVA neutron-local deposition numerical QA: PASS")
    print("  synthetic fixtures only; real HDF5 integration is separate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
