"""Regression of whole-core reaction-partition identities (synthetic arrays).

Actual filtered OpenMC transport statepoints are tested separately by the
real-core CI job. These fixtures establish the mathematics, not NASA physics.
"""
from __future__ import annotations

import math

import numpy as np

from .core_reaction_partition import assess_material_partition


def reject(action, fragment: str) -> None:
    try:
        action()
    except ValueError as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError("invalid core reactions accepted: " + fragment)


def example():
    ids = [1, 2, 3]
    names = {
        1: "conceptual (U,Zr)C-graphite surrogate",
        2: "hydrogen coolant",
        3: "beryllium radial reflector",
    }
    bins = np.array([
        [1e-5, 0.625],
        [0.625, 1e5],
        [1e5, 2e7],
    ])
    absorption = np.array([0.03, 0.005, 0.005])
    fission = np.array([0.012, 0., 0.])
    material = {"absorption": absorption, "nu-fission": fission}
    fractional_spectrum = np.array([0.5, 0.3, 0.2])
    energy = {score: values[:, None] * fractional_spectrum[None, :]
              for score, values in material.items()}
    global_rates = {"absorption": 0.04, "nu-fission": 0.012}
    return ids, names, global_rates, material, energy, bins


def main() -> int:
    ids, names, global_rates, material, energy, bins = example()
    valid = assess_material_partition(
        ids, names, global_rates, material, energy, bins
    )
    assert valid["material_vs_global_screen_passed"] is True
    assert valid["material_vs_energy_screen_passed"] is True
    assert valid["fuel_only_fission_origin_screen_passed"] is True
    assert valid["all_partition_screens_passed"] is True
    assert valid["source_stationarity_proven"] is False
    assert valid["nasa_nerva_physical_core_validated"] is False
    assert math.isclose(
        valid["scores"]["absorption"]["summed_material_reactions_per_source"],
        0.04, rel_tol=1e-12
    )
    assert valid["nonfuel_nu_fission_fraction"] == 0

    # Omitted material reaction is NOT a valid whole-core absorption tally.
    missing_absorption = {**material,
                          "absorption": material["absorption"].copy()}
    missing_absorption["absorption"][2] = 0.
    incomplete = assess_material_partition(
        ids, names, global_rates, missing_absorption, energy, bins
    )
    assert incomplete["material_vs_global_screen_passed"] is False
    assert incomplete["all_partition_screens_passed"] is False

    # EnergyFilter bins omit 20% of otherwise counted collisions.
    bad_energy = {**energy,
                  "absorption": energy["absorption"].copy()}
    bad_energy["absorption"][:, -1] = 0.
    unresolved = assess_material_partition(
        ids, names, global_rates, material, bad_energy, bins
    )
    assert unresolved["material_vs_global_screen_passed"] is True
    assert unresolved["material_vs_energy_screen_passed"] is False
    assert unresolved["scores"]["absorption"][
        "unaccounted_energy_fraction_of_material"] > 0.19

    # The declared surrogate does not contain fissile isotopes outside fuel.
    odd_fission = {**material,
                   "nu-fission": np.array([0.010, 0.002, 0.])}
    odd_energy = {**energy,
                  "nu-fission":
                  odd_fission["nu-fission"][:, None] * np.array([.5,.3,.2])}
    misplaced = assess_material_partition(
        ids, names, global_rates, odd_fission, odd_energy, bins
    )
    assert misplaced["material_vs_global_screen_passed"]
    assert misplaced["material_vs_energy_screen_passed"]
    assert misplaced["nonfuel_nu_fission_fraction"] > 0.16
    assert not misplaced["fuel_only_fission_origin_screen_passed"]
    assert not misplaced["all_partition_screens_passed"]

    reject(lambda: assess_material_partition(
        [1, 1, 3], names, global_rates, material, energy, bins
    ), "duplicate")
    reject(lambda: assess_material_partition(
        [1, 2], names, global_rates, material, energy, bins
    ), "every exported")
    reject(lambda: assess_material_partition(
        ids, {1:"fuel", 2:"H", 3:"Be"}, global_rates, material, energy, bins
    ), "fuel material")
    reject(lambda: assess_material_partition(
        ids, names, global_rates,
        {**material, "absorption": np.array([.03,-.005,.005])},
        energy, bins
    ), "invalid or negative")
    reject(lambda: assess_material_partition(
        ids, names, global_rates, material, energy,
        np.array([[1e-5,.625],[1.,1e5],[1e5,2e7]])
    ), "contiguous")
    reject(lambda: assess_material_partition(
        ids, names, global_rates, material, energy, bins,
        maximum_partition_absolute_mismatch=0
    ), "finite within")

    print("NERVA OpenMC material/global/energy neutron reaction partition numerics: PASS")
    print("  artificial arrays only; actual neutron transport: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
