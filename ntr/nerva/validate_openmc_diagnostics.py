"""Validate the expanded NERVA OpenMC diagnostics configuration."""

from __future__ import annotations

import openmc

from .config import NervaConfig
from .model import build_model
from .postprocess_openmc_diagnostics import (
    _peaking_metrics,
    _spectral_fractions,
)


def main() -> int:
    config = NervaConfig(
        core_rings=3,
        particles=100,
        batches=12,
        inactive=4,
        photon_transport=False,
        diagnostic_energy_groups=16,
        axial_mesh_bins=24,
        entropy_mesh_xy=6,
        entropy_mesh_z=4,
    )
    model = build_model(config, assembly="reactor")

    assert model.settings.run_mode == "eigenvalue"
    assert model.settings.photon_transport is False
    assert model.settings.entropy_mesh is not None

    tallies = {t.name: t for t in model.tallies}
    required = {
        "nerva_3d_neutronics",
        "nerva_3d_fuel_heating",
        "nerva_3d_tie_heating",
        "nerva_3d_fuel_hydrogen_heating",
        "nerva_3d_tie_hydrogen_heating",
        "nerva_3d_tie_supply_hydrogen_heating",
        "nerva_3d_tie_return_hydrogen_heating",
        "nerva_fuel_sector_01_instances",
        "nerva_fuel_sector_01_axial_instances",
        "nerva_fuel_sector_19_instances",
        "nerva_fuel_sector_19_axial_instances",
        "nerva_hydrogen_channel_01_instances",
        "nerva_hydrogen_channel_01_axial_instances",
        "nerva_hydrogen_channel_19_instances",
        "nerva_hydrogen_channel_19_axial_instances",
        "nerva_material_transport",
        "nerva_material_energy_transport",
        "nerva_fuel_spectrum",
        "nerva_hydrogen_spectrum",
        "nerva_axial_transport",
        "nerva_vacuum_boundary_current",
    }
    missing = required - set(tallies)
    assert not missing, f"missing diagnostics tallies: {sorted(missing)}"
    assert any(
        name.startswith("nerva_tie_axial_instance_")
        for name in tallies
    )

    mesh_scores = set(tallies["nerva_3d_neutronics"].scores)
    assert {
        "flux",
        "absorption",
        "fission",
        "nu-fission",
        "heating",
        "heating-local",
    } <= mesh_scores

    material_filter = tallies[
        "nerva_material_transport"
    ].find_filter(openmc.MaterialFilter)
    assert len(material_filter.bins) == 9

    energy_filter = tallies[
        "nerva_fuel_spectrum"
    ].find_filter(openmc.EnergyFilter)
    assert len(energy_filter.bins) == 16

    axial_filter = tallies[
        "nerva_axial_transport"
    ].find_filter(openmc.MeshFilter)
    assert tuple(axial_filter.mesh.dimension) == (1, 1, 24)

    # Pure postprocessing helper checks.
    peak = _peaking_metrics(
        __import__("numpy").array([1.0, 2.0, 4.0])
    )
    assert peak["max_to_mean_active"] > 1.0

    bins = __import__("numpy").array(
        [
            [1.0e-5, 0.625],
            [0.625, 1.0e5],
            [1.0e5, 2.0e7],
        ]
    )
    fractions = _spectral_fractions(
        bins,
        __import__("numpy").array([1.0, 2.0, 3.0]),
    )
    assert abs(sum(fractions.values()) - 1.0) < 1.0e-12

    trigger_config = NervaConfig(
        core_rings=3,
        particles=100,
        batches=12,
        inactive=4,
        tally_rel_err_trigger=0.25,
        trigger_max_batches=30,
        trigger_batch_interval=2,
        diagnostic_energy_groups=16,
        axial_mesh_bins=24,
    )
    trigger_model = build_model(
        trigger_config,
        assembly="reactor",
    )
    assert trigger_model.settings.trigger_active is True
    assert trigger_model.settings.trigger_max_batches == 30
    assert trigger_model.settings.trigger_batch_interval == 2
    trigger_tally = next(
        tally
        for tally in trigger_model.tallies
        if tally.name == "nerva_3d_neutronics"
    )
    assert len(trigger_tally.triggers) == 1
    assert trigger_tally.triggers[0].trigger_type == "rel_err"

    photon_config = NervaConfig(
        core_rings=3,
        particles=100,
        batches=12,
        inactive=4,
        photon_transport=True,
        diagnostic_energy_groups=16,
        axial_mesh_bins=24,
    )
    photon_model = build_model(
        photon_config,
        assembly="reactor",
    )
    photon_tallies = {t.name for t in photon_model.tallies}
    assert "nerva_particle_heating" in photon_tallies

    print("Expanded NERVA OpenMC diagnostics validation: PASS")
    print(f"  tallies: {len(tallies)}")
    print("  material-resolved transport: enabled")
    print("  fuel/H2 spectra: enabled")
    print("  axial transport: enabled")
    print("  vacuum leakage current: enabled")
    print("  entropy mesh: enabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
