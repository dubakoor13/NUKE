"""Synthetic adversarial neutron accounting checks; NOT actual OpenMC evidence."""
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import h5py

from .validate_fixed_source_core_physics import fixture
from .fixed_source_fission_branching import read_branching_fixed_source_balance


def make_on(path: Path, *, broken: bool = False) -> None:
    root = ET.parse(path.parent / "settings.xml")
    root.getroot().find("create_fission_neutrons").text = "true"
    root.write(path.parent / "settings.xml")
    with h5py.File(path, "r+") as state:
        collision = state["tallies"]["tally 1"]["results"]
        data = collision[()]
        # Physically permit 0.02 added daughters per injected neutron.
        data[0, 3, 0] = 30 * .02
        data[0, 3, 1] = data[0, 3, 0]**2
        collision[...] = data
        surface = state["tallies"]["tally 2"]["results"]
        a = surface[()]
        # Outflow rises exactly with the externally driven fission daughters.
        # The control below therefore has a strict physical balance of one.
        a[2, 0, 0] += 0 if broken else 30 * .02
        a[2, 0, 1] = a[2, 0, 0]**2
        surface[...] = a


def rejects(f, term: str) -> None:
    try:
        f()
    except (ValueError, KeyError) as ex:
        assert term in str(ex), (term, str(ex))
    else:
        raise AssertionError("physically invalid externally driven neutron data passed")


def main() -> int:
    with TemporaryDirectory() as temporary:
        path = fixture(Path(temporary))
        rejects(lambda: read_branching_fixed_source_balance(path),
                "branching fixed-source transport")
        make_on(path)
        result = read_branching_fixed_source_balance(path)
        assert abs(result["external_source_neutron_balance"] - 1.) < 1e-12
        assert result["real_fission_daughter_neutrons_per_external_source"] > 0
        assert result["fixed_external_source_fission_branching_executed"]
        assert not result["self_sustaining_reactor_criticality_validated"]
        path = fixture(Path(temporary))
        make_on(path, broken=True)
        # Missing fission daughters must violate particle conservation;
        # this is an exact event-count identity, not an uncertainty band.
        with h5py.File(path, "r+") as h:
            moments = h["tallies"]["tally 1"]["results"]
            data = moments[()]
            data[0, 3, 0] = 30 * .05
            moments[...] = data
        rejects(lambda: read_branching_fixed_source_balance(path),
                "do not conserve source")
    print("Fission-on fixed external-source neutron-balance HDF5 fixtures: PASS (synthetic only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
