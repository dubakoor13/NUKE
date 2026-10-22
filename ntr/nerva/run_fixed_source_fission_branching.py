"""Run two matched cold-core fixed-source physical transport experiments.

One executes actual OpenMC with fission daughters enabled; one is an
otherwise identical prescribed-source control with daughters disabled.
The same geometry, materials, cross-sections, source, seeds, particle
population and tally XML are used in both modes. No criticality or
geometry search is performed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .config import NervaConfig
from .model import build_model
from .fixed_source_core_physics import sha
from .fixed_source_fission_branching import compare_four_seed_external_source_modes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cross-sections", type=Path)
    p.add_argument("--seeds", type=int, nargs=4,
                   default=[13007, 23011, 33013, 43019])
    p.add_argument("--batches", type=int, default=30)
    p.add_argument("--particles", type=int, default=3000)
    args = p.parse_args()
    if (len(set(args.seeds)) != 4 or any(i <= 0 for i in args.seeds)
            or args.batches < 20 or args.particles < 100):
        raise ValueError("require four independent actual fixed-source neutron runs")
    binary = shutil.which("openmc")
    if binary is None:
        raise RuntimeError("no compiled real OpenMC neutron executable available")
    xs = args.cross_sections or os.environ.get("OPENMC_CROSS_SECTIONS")
    if xs is None:
        raise RuntimeError("verified evaluated neutron nuclear cross-section data required")
    xs = Path(xs).resolve(strict=True)
    base = args.output.resolve()
    no_fission = base / "suppressed_fission_control"
    with_fission = base / "fission_daughters_transported"
    base.mkdir(parents=True, exist_ok=True)

    # Reuse the already independently qualified and audited fixed-source
    # suppressed-daughter experiment on the same nuclear model.
    cmd = [
        sys.executable, "-m", "ntr.nerva.run_fixed_source_core",
        "--cross-sections", str(xs),
        "--output", str(no_fission),
        "--seeds", *[str(s) for s in args.seeds],
        "--batches", str(args.batches),
        "--particles", str(args.particles),
    ]
    print("Executing real suppressed-fission control", flush=True)
    subprocess.run(cmd, check=True)

    cfg = NervaConfig(
        core_rings=3, hydrogen_temperature_k=293.6059129982851,
        batches=args.batches, inactive=0, particles=args.particles,
        heating_mesh_dimension=(8, 8, 8), axial_mesh_bins=8,
        diagnostic_energy_groups=8,
    )
    cfg.validate()
    model = build_model(cfg, assembly="core", eigenvalue_entropy_mesh=False)
    model.settings.run_mode = "fixed source"
    model.settings.inactive = 0
    model.settings.create_fission_neutrons = True
    model.settings.statepoint = {"batches": [args.batches]}
    if model.settings.source is None:
        raise ValueError("actual physical external-source distribution is missing")
    baseline = None
    for seed in args.seeds:
        directory = with_fission / f"seed_{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        model.settings.seed = seed
        model.export_to_xml(directory=directory)
        control = no_fission / f"seed_{seed}"
        identity = tuple(sha(directory / name) for name in
                         ("geometry.xml", "materials.xml", "tallies.xml"))
        identical = tuple(sha(control / name) for name in
                          ("geometry.xml", "materials.xml", "tallies.xml"))
        if identity != identical or (baseline is not None and identity != baseline):
            raise ValueError("different physical materials/geometry/neutron scores across modes")
        baseline = identity
        # XML difference must be specifically the fission-secondary setting.
        print("Executing genuine externally-driven fission transport, seed", seed, flush=True)
        with (directory / "openmc_branching_transport.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [binary], cwd=directory,
                env={**os.environ, "OPENMC_CROSS_SECTIONS": str(xs),
                     "OMP_NUM_THREADS": "2"},
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode:
            raise RuntimeError(
                f"real fission-daughter transport seed {seed} failed with "
                f"exit {result.returncode}; inspect saved neutron execution log")
    report = compare_four_seed_external_source_modes(
        with_fission, no_fission, list(args.seeds), args.batches
    )
    report["evaluated_cross_sections_xml_sha256"] = sha(xs)
    report["particles_per_seed_per_mode"] = args.particles * args.batches
    report["reactor_conditions"] = (
        "cold unchanged 3-ring NERVA-derived demonstration core; "
        "not calibrated NASA NERVA historical reactor"
    )
    result = base / "real_source_driven_fission_multiplication_validation.json"
    result.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print("REAL OpenMC fixed-source fission secondary neutron closure: PASS", flush=True)
    print("Mean fission daughters produced per initial neutron:",
          report["mean_fission_daughter_neutrons_per_external_source"], flush=True)
    print("No criticality, eigenvalue, fuel design or local power validation", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
