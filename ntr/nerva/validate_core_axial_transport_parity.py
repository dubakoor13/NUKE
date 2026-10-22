"""Adversarial numerical and native-layout synthetic HDF5 axial neutron QA.

Fixtures are ARTIFICIAL. Actual validation is performed against four
independently executed OpenMC C++ seeds and saved HDF5 checkpoints.
"""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

from .core_axial_transport_parity import (
    SCORES,
    measure_disjoint_axial_window,
    read_axial_transport_checkpoint,
)


def reject(call, substring: str) -> None:
    try:
        call()
    except (ValueError, KeyError) as exc:
        assert substring in str(exc), (substring, str(exc))
    else:
        raise AssertionError("invalid physical neutron transport evidence accepted: " + substring)


def fake_neutron_statepoint(path: Path, *, clip: bool = False,
                            tamper: bool = False) -> None:
    labels = np.array([b"flux", b"fission", b"nu-fission", b"heating-local"])
    with h5py.File(path, "w") as h:
        h.attrs["filetype"] = np.bytes_("statepoint")
        h.attrs["photon_transport"] = 0
        h.attrs["tallies_present"] = 1
        for name, value in (("run_mode", np.bytes_("eigenvalue")),
                            ("seed", 13007), ("current_batch", 40),
                            ("n_inactive", 30), ("n_particles", 3000),
                            ("n_realizations", 10)):
            h.create_dataset(name, data=value)
        tallies = h.create_group("tallies")
        filters = tallies.create_group("filters")
        meshes = tallies.create_group("meshes")
        for tid, tally_name, mesh_name, dims in (
            (1, "nerva_axial_transport", "nerva_axial_mesh", (1, 1, 8)),
            (2, "nerva_3d_neutronics", "nerva_core_mesh", (2, 2, 8)),
        ):
            filt = filters.create_group(f"filter {tid}")
            filt.create_dataset("type", data=np.bytes_("mesh"))
            filt.create_dataset("bins", data=tid)
            mesh = meshes.create_group(f"mesh {tid}")
            mesh.create_dataset("name", data=np.bytes_(mesh_name))
            mesh.create_dataset("type", data=np.bytes_("regular"))
            mesh.create_dataset("dimension", data=dims)
            mesh.create_dataset("lower_left", data=(-2., -2., -66.))
            mesh.create_dataset("upper_right", data=(2., 2., 65. if clip and tid == 1 else 66.))
            tally = tallies.create_group(f"tally {tid}")
            tally.create_dataset("name", data=np.bytes_(tally_name))
            tally.create_dataset("estimator", data=np.bytes_("tracklength"))
            tally.create_dataset("n_realizations", data=10)
            tally.create_dataset("n_filters", data=1)
            tally.create_dataset("filters", data=[tid])
            tally.create_dataset("score_bins", data=labels)
            data = np.zeros((int(np.prod(dims)), len(labels), 2))
            data[:, :3, 0] = np.asarray((40., 4., 10.)) / (1 if tid == 1 else 4)
            data[:, :3, 1] = data[:, :3, 0] ** 2
            if tamper and tid == 2:
                data[0, 1, 0] += 1.0
            # Nonlocal evaluated KERMA negativity must NOT invalidate
            # physically positive neutron flux/fission scores.
            data[:, 3, 0] = -2.
            tally.create_dataset("results", data=data)
    (path.parent / "geometry.xml").write_text(
        '<geometry><surface type="z-plane" boundary="vacuum" coeffs="-66"/>'
        '<surface type="z-plane" boundary="vacuum" coeffs="66"/>'
        '<surface type="z-cylinder" boundary="vacuum" coeffs="0 0 2"/></geometry>',
        encoding="utf-8",
    )


def main() -> int:
    z = np.array([1., 2., 4., 5., 5., 4., 2., 1.])
    axial = np.column_stack((z, z / 10., z / 4.))
    totals = axial.sum(axis=0)
    volume = np.tile(totals / 32., (32, 1))
    result = measure_disjoint_axial_window(axial, volume, 10)
    for score in SCORES:
        assert result["scores"][score]["independent_mesh_relative_integral_difference"] < 1e-14
        assert result["scores"][score]["axial_mirror_total_variation"] < 1e-14
    displaced = axial.copy()
    displaced[0, :] += .5
    skew = measure_disjoint_axial_window(displaced, np.tile(displaced.sum(axis=0)/32.,(32,1)),10)
    assert skew["scores"]["flux"]["axial_mirror_total_variation"] > 0
    altered = volume.copy()
    altered[0, 1] *= 1.1
    reject(lambda: measure_disjoint_axial_window(axial, altered, 10), "closure failed")
    altered = axial.copy()
    altered[1, 2] = np.nan
    reject(lambda: measure_disjoint_axial_window(altered, volume, 10), "invalid disjoint")
    reject(lambda: measure_disjoint_axial_window(axial, volume, 1), "invalid disjoint")
    with TemporaryDirectory() as root:
        case = Path(root) / "statepoint.040.h5"
        fake_neutron_statepoint(case)
        observed = read_axial_transport_checkpoint(case)
        assert observed["seed"] == 13007
        assert observed["axial_cumulative"].shape == (8,3)
        assert observed["three_d_cumulative"].shape == (32,3)
        assert measure_disjoint_axial_window(
            observed["axial_cumulative"],
            observed["three_d_cumulative"], observed["active"],
        )["scores"]["nu-fission"]["independent_mesh_relative_integral_difference"] < 1e-14
        fake_neutron_statepoint(case, clip=True)
        reject(lambda: read_axial_transport_checkpoint(case), "mesh volumes")
        fake_neutron_statepoint(case, tamper=True)
        observed = read_axial_transport_checkpoint(case)
        reject(lambda: measure_disjoint_axial_window(
            observed["axial_cumulative"], observed["three_d_cumulative"],
            observed["active"]), "closure failed")
    print("Native neutron 3D/axial physical conservation and parity: PASS (synthetic fixtures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
