"""Validate construction of the generic OpenMC air-heater model."""

from __future__ import annotations

from .openmc_airheater import (
    GenericAirHeaterConfig,
    build_openmc_airheater,
)


def main() -> int:
    config = GenericAirHeaterConfig(
        batches=4,
        particles=100,
        photon_transport=False,
        mesh_dimension=(8, 8, 8),
        spectrum_groups=16,
    )
    build = build_openmc_airheater(config)

    assert build.model.settings.run_mode == "fixed source"
    assert build.model.settings.batches == 4
    assert build.model.settings.particles == 100

    assert set(build.materials) == {"fuel", "air", "vessel"}
    assert set(build.cells) == {"fuel", "air", "vessel"}

    tally_names = {tally.name for tally in build.model.tallies}
    assert "airheater_3d_transport" in tally_names
    assert "airheater_region_transport" in tally_names
    assert "airheater_fuel_spectrum" in tally_names
    assert "airheater_air_spectrum" in tally_names
    assert "airheater_boundary_current" in tally_names

    mesh_tally = next(
        tally
        for tally in build.model.tallies
        if tally.name == "airheater_3d_transport"
    )
    assert "heating" in mesh_tally.scores
    assert "heating-local" in mesh_tally.scores
    assert "fission" in mesh_tally.scores
    assert "nu-fission" in mesh_tally.scores

    print("Generic OpenMC air-heater construction validation: PASS")
    print("  fixed-source transport: enabled")
    print("  3-D mesh transport/heating: enabled")
    print("  region heating: enabled")
    print("  energy spectra: enabled")
    print("  boundary current/leakage diagnostic: enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
