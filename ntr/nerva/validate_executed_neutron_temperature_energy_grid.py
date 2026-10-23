"""Adversarial numerical tests for neutron evaluation energy grids.

These are synthetic arrays, not fabricated OpenMC transport or certified
evaluated nuclear data. The real nuclear-data HDF5 check runs separately.
"""
from __future__ import annotations

import numpy as np

from .executed_neutron_temperature_energy_grid import (
    assess_evaluated_neutron_energy_grids,
)


def reject(fn, expected: str) -> None:
    try:
        fn()
    except (TypeError, ValueError) as exc:
        assert expected in str(exc), (expected, str(exc))
    else:
        raise AssertionError("invalid evaluated neutron energy-grid accepted: " + expected)


def main() -> int:
    good = assess_evaluated_neutron_energy_grids(
        "SyntheticNuclide", [293.6, 600.0],
        {"294K":np.array([1e-5,1.,1e6]),
         "600K":np.array([2e-5,2.,2e6])},
        ["294K","600K"]
    )
    assert good["all_native_temperature_energy_grids_valid"]
    assert good["evaluated_multitemperature_library_for_this_nuclide"]
    assert good["common_incident_energy_min_eV"] == 2e-5
    assert good["common_incident_energy_max_eV"] == 1e6

    one=assess_evaluated_neutron_energy_grids(
        "SingleTable", [293.6], {"294K":[0.,1e-2,20e6]},["294K"]
    )
    assert one["all_native_temperature_energy_grids_valid"]
    assert not one["evaluated_multitemperature_library_for_this_nuclide"]
    assert one["evaluated_temperature_node_count"]==1

    reject(lambda:assess_evaluated_neutron_energy_grids(
        "X",[293.6,600.],{"294K":[1.,2.]},["294K","600K"]),
        "missing evaluated neutron incident-energy grid")
    reject(lambda:assess_evaluated_neutron_energy_grids(
        "X",[293.6,600.],{"294K":[1.,2.],"600K":[2.,1.]},["294K","600K"]),
        "malformed/unsorted")
    reject(lambda:assess_evaluated_neutron_energy_grids(
        "X",[293.6],{"294K":[1.,1.,2.]},["294K"]),
        "malformed/unsorted")
    reject(lambda:assess_evaluated_neutron_energy_grids(
        "X",[293.6],{"294K":[1.,np.inf]},["294K"]),
        "malformed/unsorted")
    reject(lambda:assess_evaluated_neutron_energy_grids(
        "X",[293.6],{"294K":[-1.,1.]},["294K"]),
        "malformed/unsorted")
    reject(lambda:assess_evaluated_neutron_energy_grids(
        "X",[293.6],{"294K":[0.]},["294K"]),
        "malformed/unsorted")
    reject(lambda:assess_evaluated_neutron_energy_grids(
        "X",[293.6,293.6],{"294K":[1.,2.],"295K":[1.,2.]},
        ["294K","295K"]), "duplicate native evaluated temperatures")
    reject(lambda:assess_evaluated_neutron_energy_grids(
        "X",[293.6,float("nan")],{"294K":[1.,2.],"NaNK":[1.,2.]},
        ["294K","NaNK"]), "nonphysical")
    reject(lambda:assess_evaluated_neutron_energy_grids(
        "X",[293.6,600.],{"294K":[0.,1.],"600K":[2.,3.]},
        ["294K","600K"]), "no common domain")
    reject(lambda:assess_evaluated_neutron_energy_grids(
        "X",[293.6,600.],{"294K":[0.,1.],"600K":[0.,1.]},
        ["294K","294K"]), "invalid evaluated nuclide")
    print("Evaluated neutron-temperature incident-energy numerical regressions: PASS")
    print("  All fixtures synthetic; real indexed isotope HDF5 required in OpenMC CI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
