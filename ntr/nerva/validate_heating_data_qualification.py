"""Unit regression of OpenMC deposited-energy score qualification.

Synthetic math tests only; real neutron heating comes from OpenMC's
eigenvalue statepoint in the separate four-seed transport workflow.
"""
from __future__ import annotations

from pathlib import Path
import json
import tempfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .heating_data_qualification import (
    evaluate_heating_scores, inspect_statepoint, audit_real_ensemble, digest,
)


def reject(action, word):
    try:
        action()
    except (ValueError, FileNotFoundError) as error:
        assert word in str(error), str(error)
    else:
        raise AssertionError("physically invalid heating accepted: " + word)


def mock_statepoint(path: Path, *, local: float, seed: int = 13007,
                    particles: int = 3000) -> None:
    with h5py.File(path, "w") as h:
        h.attrs["filetype"] = np.bytes_("statepoint")
        h.attrs["photon_transport"] = 0
        h.attrs["tallies_present"] = 1
        h.create_dataset("run_mode", data=np.bytes_("eigenvalue"))
        h.create_dataset("seed", data=seed)
        h.create_dataset("current_batch", data=60)
        h.create_dataset("n_inactive", data=20)
        h.create_dataset("n_particles", data=particles)
        ts = h.create_group("tallies")
        flt = ts.create_group("filters")
        for filter_id, ftype, bins in ((1,"mesh",2),(2,"material",[19,20])):
            item = flt.create_group(f"filter {filter_id}")
            item.create_dataset("type",data=np.bytes_(ftype))
            item.create_dataset("bins",data=bins)
            item.create_dataset("n_bins",data=1 if filter_id==1 else 2)
        meshes = ts.create_group("meshes")
        mesh = meshes.create_group("mesh 2")
        mesh.create_dataset("name",data=np.bytes_("nerva_core_mesh"))
        mesh.create_dataset("type",data=np.bytes_("regular"))
        mesh.create_dataset("dimension",data=[1,1,1])
        mesh.create_dataset("lower_left",data=[-10.,-10.,-66.])
        mesh.create_dataset("upper_right",data=[10.,10.,66.])
        for i, (name, scores, values) in enumerate((
            ("nerva_3d_neutronics", ["heating","heating-local","kappa-fission"],
             [8e5,local,9e5]),
            ("nerva_material_transport", ["heating","heating-local"],
             [8e5,local]),
        )):
            t = ts.create_group(f"tally {i+1}")
            t.create_dataset("name", data=np.bytes_(name))
            t.create_dataset("n_realizations", data=40)
            t.create_dataset("estimator",data=np.bytes_("tracklength"))
            t.create_dataset("n_filters",data=1)
            t.create_dataset("filters",data=[i+1])
            t.create_dataset("nuclides",data=np.array(["total"],dtype="S10"))
            t.create_dataset("score_bins", data=np.array(
                scores, dtype="S30"))
            values = np.asarray(values, dtype=float)
            n_bins = 1 if i==0 else 2
            results = np.zeros((n_bins, len(scores), 2))
            results[:, :, 0] = values * 40 / n_bins
            results[:, :, 1] = (values / n_bins)**2 * 40
            t.create_dataset("results", data=results)
    root = ET.Element("materials")
    for mid in (19,20):
        node = ET.SubElement(root,"material",{"id":str(mid),"name":f"fixture-{mid}"})
        ET.SubElement(node,"nuclide",{"name":"H1","ao":"1.0"})
    ET.ElementTree(root).write(path.parent/"materials.xml",encoding="utf-8")
    geom = ET.Element("geometry")
    ET.SubElement(geom,"surface",{"id":"1","type":"z-cylinder",
        "boundary":"vacuum","coeffs":"0.0 0.0 9.0"})
    for sid,z in ((2,-66.),(3,66.)):
        ET.SubElement(geom,"surface",{"id":str(sid),"type":"z-plane",
            "boundary":"vacuum","coeffs":str(z)})
    ET.ElementTree(geom).write(path.parent/"geometry.xml",encoding="utf-8")


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
        assert report["whole_core_heating_tally_spatial_and_material_coverage_verified"]
        assert report["executed_material_ids_in_heating_tally"]==[19,20]
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

        # Identical incomplete tallies can agree numerically, but their
        # filter sets must be proven to span the ACTUAL reactor.
        mock_statepoint(file,local=700000)
        with h5py.File(file,"r+") as h:
            h["tallies"]["filters"]["filter 2"]["bins"][...] = [19,19]
        reject(lambda:inspect_statepoint(file),"every executed material")
        mock_statepoint(file,local=700000)
        with h5py.File(file,"r+") as h:
            h["tallies"]["meshes"]["mesh 2"]["upper_right"][0] = 8.0
        reject(lambda:inspect_statepoint(file),"clips the executed physical core")
        mock_statepoint(file,local=700000)
        with h5py.File(file,"r+") as h:
            h["tallies"]["tally 1"]["estimator"][...] = np.bytes_("collision")
        reject(lambda:inspect_statepoint(file),"estimators")
        mock_statepoint(file,local=700000)
        with h5py.File(file,"r+") as h:
            h["tallies"]["filters"]["filter 1"]["bins"][...] = 999
        reject(lambda:inspect_statepoint(file),"whole-core regular mesh")
        mock_statepoint(file,local=700000)
        geometry = file.parent/"geometry.xml"
        root = ET.parse(geometry).getroot()
        root.findall("surface")[-1].set("coeffs","70.0")
        ET.ElementTree(root).write(geometry,encoding="utf-8")
        reject(lambda:inspect_statepoint(file),"clips the executed physical core")

        # Fully executed OpenMC HDF5 is checked in real transport CI. The
        # following forged HDF5s ONLY test the numerical audit's rejection
        # logic; no synthetic result can certify genuine reactor physics.
        paths = [Path(tmp)/f"seed_{seed}.h5" for seed in
                 (13007,23011,33013,43019)]
        seeds = (13007,23011,33013,43019)
        summary = Path(tmp)/"fake_summary.json"
        def write_summary():
            summary.write_text(json.dumps({
                "execution":"ACTUAL_OPENMC_TRANSPORT",
                "assembly":"core",
                "batches":60,"inactive_batches":20,
                "particles_per_generation":3000,
                "runs":[{
                    "seed":seed,"statepoint_path":str(path),
                    "statepoint_sha256":digest(path),
                } for seed,path in zip(seeds,paths)],
            }),encoding="utf-8")

        for seed,path in zip(seeds,paths):
            mock_statepoint(path,local=0.0,seed=seed)
        write_summary()
        real_format_zero = audit_real_ensemble(summary)
        assert real_format_zero["exactly_zero_local_deposition_moments_in_every_seed"]
        assert real_format_zero["ensemble_local_heat_deposition_evidence"] == (
            "ZERO_LOCAL_HEAT_DEPOSITION_IN_ALL_REAL_SEEDS"
        )
        assert real_format_zero["independent_seed_local_heating_mean_eV_per_source"] == 0.
        assert real_format_zero["independent_seed_local_heating_sem_eV_per_source"] == 0.
        assert real_format_zero["all_runs_have_identical_executed_source_settings"]
        assert not real_format_zero["thermal_power_normalization_allowed"]

        for seed,path in zip(seeds,paths):
            mock_statepoint(path,local=700000.,seed=seed)
        write_summary()
        statistically_resolved = audit_real_ensemble(summary)
        assert statistically_resolved["all_real_statepoints_have_usable_neutron_local_heating"]
        assert statistically_resolved["ensemble_local_heat_deposition_evidence"] == (
            "POSITIVE_LOCAL_HEATING_BUT_CORE_PHYSICS_UNQUALIFIED"
        )
        assert not statistically_resolved["thermal_power_normalization_allowed"]
        assert statistically_resolved["executed_active_batches"]==40

        mock_statepoint(paths[0],local=0.,seed=seeds[0])
        write_summary()
        mixed = audit_real_ensemble(summary)
        assert mixed["inconsistent_zero_and_nonzero_local_deposition_across_seeds"]
        assert mixed["ensemble_local_heat_deposition_evidence"] == (
            "INCONSISTENT_ZERO_AND_NONZERO_LOCAL_HEATING_ACROSS_SEEDS"
        )
        assert not mixed["thermal_power_normalization_allowed"]

        mock_statepoint(paths[0],local=0.,seed=seeds[0],particles=2000)
        write_summary()
        reject(lambda:audit_real_ensemble(summary),"source population/batches")
        mock_statepoint(paths[0],local=0.,seed=seeds[0])
        write_summary()
        # Tampering with persisted statepoint must invalidate provenance.
        with h5py.File(paths[0],"r+") as h:
            h["n_particles"][...]=2000
        reject(lambda:audit_real_ensemble(summary),"source identity/hash")

    print("NERVA neutron-local deposition numerical QA: PASS")
    print("  synthetic fixtures only; real HDF5 integration is separate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
