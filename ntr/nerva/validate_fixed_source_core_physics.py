"""Adversarial native-format HDF5 verification; synthetic, not reactor evidence."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .fixed_source_core_physics import read_fixed_source_balance


def fixture(where: Path, *, wrong_leakage: bool = False,
            fission_daughters: bool = False) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    path = where / "statepoint.30.h5"
    root = ET.Element("settings")
    ET.SubElement(root, "run_mode").text = "fixed source"
    ET.SubElement(root, "create_fission_neutrons").text = (
        "true" if fission_daughters else "false"
    )
    ET.SubElement(root, "inactive").text = "0"
    ET.ElementTree(root).write(where / "settings.xml")
    (where / "geometry.xml").write_text(
        '<geometry><surface id="1" type="z-plane" boundary="vacuum" coeffs="-66"/>'
        '<surface id="2" type="z-plane" boundary="vacuum" coeffs="66"/>'
        '<surface id="3" type="z-cylinder" boundary="vacuum" coeffs="0 0 10"/>'
        '</geometry>', encoding="utf-8",
    )
    (where / "materials.xml").write_text("<materials/>", encoding="utf-8")
    with h5py.File(path, "w") as h:
        h.attrs["filetype"] = np.bytes_("statepoint")
        h.attrs["photon_transport"] = 0
        h.attrs["tallies_present"] = 1
        for k, value in (("run_mode", np.bytes_("fixed source")),
                         ("seed", 13007), ("current_batch", 30),
                         ("n_realizations", 30),
                         ("n_particles", 3000)):
            h.create_dataset(k, data=value)
        group = h.create_group("tallies")
        group.create_group("filters")
        b = group.create_group("tally 1")
        b.create_dataset("name", data=np.bytes_("nerva_global_neutron_balance"))
        b.create_dataset("estimator", data=np.bytes_("analog"))
        b.create_dataset("n_realizations", data=30)
        b.create_dataset("n_filters", data=0)
        labels = [b"absorption", b"scatter", b"nu-scatter", b"nu-fission"]
        b.create_dataset("score_bins", data=labels)
        values = np.zeros((1, 4, 2))
        values[0, :, 0] = 30 * np.array([.10, 3., 3.05, .01])
        values[0, :, 1] = values[0, :, 0]**2
        b.create_dataset("results", data=values)
        f = group["filters"].create_group("filter 1")
        f.create_dataset("type", data=np.bytes_("surface"))
        f.create_dataset("bins", data=[1, 2, 3])
        s = group.create_group("tally 2")
        s.create_dataset("name", data=np.bytes_("nerva_vacuum_boundary_current"))
        s.create_dataset("estimator", data=np.bytes_("analog"))
        s.create_dataset("n_realizations", data=30)
        s.create_dataset("n_filters", data=1)
        s.create_dataset("filters", data=[1])
        s.create_dataset("score_bins", data=[b"current"])
        currents = np.zeros((3, 1, 2))
        currents[:, 0, 0] = 30 * np.array([
            -.1, .15, .66 if wrong_leakage else .70
        ])
        currents[:, 0, 1] = currents[:, 0, 0]**2
        s.create_dataset("results", data=currents)
    return path


def reject(fn, reason: str) -> None:
    try:
        fn()
    except (ValueError, KeyError) as exc:
        assert reason in str(exc), (reason, str(exc))
    else:
        raise AssertionError("invalid non-multiplying source neutron balance passed")


def main() -> int:
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        path = fixture(folder)
        r = read_fixed_source_balance(path)
        assert abs(r["nonfission_source_neutron_balance"]-1.) < 1.e-14
        assert abs(r["outward_vacuum_leakage_per_injected_neutron"] - .95) < 1e-14
        assert r["source_driven_no_fission_secondaries"]
        assert not r["reactor_eigenvalue_or_criticality_validated"]
        fixture(folder, wrong_leakage=True)
        reject(lambda: read_fixed_source_balance(path), "violates conservation")
        fixture(folder, fission_daughters=True)
        reject(lambda: read_fixed_source_balance(path), "suppressed-fission")
    print("Native suppressed-fission fixed-source neutron accounting: PASS (synthetic fixtures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
