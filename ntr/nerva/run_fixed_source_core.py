"""Run independently seeded, genuine OpenMC subcritical fixed-source transport.

This is a NON-BRANCHING prescribed-source neutron attenuation experiment on
the *unchanged* three-ring cold core geometry/materials. Suppressing fission
daughter transport removes the severely cloned k-eigenvalue source but also
means the result CANNOT represent NERVA reactor multiplication/criticality.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from .config import NervaConfig
from .model import build_model
from .nuclear_data import inspect_nuclear_data, write_report
from .fixed_source_core_physics import audit_four_seed_fixed_source, sha


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cross-sections", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs=4,
                   default=(13007, 23011, 33013, 43019))
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--particles", type=int, default=3000)
    a = p.parse_args()
    if (a.batches < 20 or a.particles < 100
            or len(set(a.seeds)) != 4 or any(seed <= 0 for seed in a.seeds)):
        raise ValueError("invalid independent fixed-source transport statistics")
    binary = shutil.which("openmc")
    if binary is None:
        raise RuntimeError("real OpenMC transport executable unavailable")
    xs = a.cross_sections
    if xs is None:
        val = os.environ.get("OPENMC_CROSS_SECTIONS")
        if not val:
            raise RuntimeError("genuine evaluated neutron cross-sections.xml required")
        xs = Path(val)
    xs = xs.resolve(strict=True)
    cfg = NervaConfig(
        core_rings=3,
        hydrogen_temperature_k=293.6059129982851,
        particles=a.particles, batches=a.batches, inactive=0,
        heating_mesh_dimension=(8, 8, 8), axial_mesh_bins=8,
        diagnostic_energy_groups=8,
    )
    cfg.validate()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    nuclear = inspect_nuclear_data(
        cross_sections=xs, config=cfg,
        photon_transport=False, require_referenced_files=True,
    )
    write_report(nuclear, root / "evaluated_neutron_data_preflight.json")
    if not nuclear.ready:
        raise RuntimeError("actual OpenMC neutron-data physical preflight failed")
    model = build_model(cfg, assembly="core")
    model.settings.run_mode = "fixed source"
    model.settings.inactive = 0
    model.settings.entropy_mesh = None
    model.settings.create_fission_neutrons = False
    model.settings.statepoint = {"batches": [a.batches]}
    # Same explicitly declared fissionable-region initial source as the
    # eigenvalue demonstrator, but PRESCRIBED on every fixed-source batch.
    # OpenMC's default Watt energy distribution is a source assumption.
    source = model.settings.source
    if source is None:
        raise RuntimeError("physical prescribed source was not defined")
    os.environ["OPENMC_CROSS_SECTIONS"] = str(xs)
    baseline_hashes = None
    for seed in a.seeds:
        model.settings.seed = seed
        folder = root / f"seed_{seed}"
        folder.mkdir(exist_ok=True)
        model.export_to_xml(directory=folder)
        identity = {x: sha(folder / x)
                    for x in ("geometry.xml", "materials.xml", "tallies.xml")}
        if baseline_hashes is None:
            baseline_hashes = identity
        elif identity != baseline_hashes:
            raise RuntimeError("core material geometry or physical tallies changed between RNG seeds")
        with (folder / "openmc_fixed_source.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [binary], cwd=folder,
                env={**os.environ, "OPENMC_CROSS_SECTIONS": str(xs),
                     "OMP_NUM_THREADS": "2"},
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                f"genuine fixed-source neutron transport seed {seed} failed; "
                f"see {folder / 'openmc_fixed_source.log'}")
    result = audit_four_seed_fixed_source(root, list(a.seeds), a.batches)
    result["evaluated_cross_sections_index_sha256"] = sha(xs)
    result["run_mode"] = "fixed source"
    result["source_distribution"] = (
        "prescribed NERVA demonstration fissionable-region source, "
        "default OpenMC fission-like energy spectrum; NOT NASA measured"
    )
    result["source_population_per_seed"] = a.particles * a.batches
    target = root / "genuine_fixed_source_core_neutron_balance.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print("Genuine four-seed suppressed-fission fixed-source neutron balance: PASS")
    print("Measured neutron fate:", result["mean_rates"])
    print("Reactor criticality, fission genealogy and local heating: NOT VALIDATED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
