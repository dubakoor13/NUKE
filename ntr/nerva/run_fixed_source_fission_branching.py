"""Run two matched cold-core fixed-source physical neutron experiments.

One *unchanged* OpenMC model object and external source are exported twice:
with fission daughter neutron transport off and on, respectively. The actual
geometry.xml, materials.xml and tallies.xml must be byte-for-byte identical.
Only create_fission_neutrons differs in settings.xml; the RNG seed is
identical per paired mode. No criticality or reactor design tuning.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

from .config import NervaConfig
from .model import build_model
from .nuclear_data import inspect_nuclear_data, write_report
from .fixed_source_core_physics import audit_four_seed_fixed_source, sha
from .fixed_source_fission_branching import compare_four_seed_external_source_modes


def _physics_setting_identity(path: Path) -> bytes:
    """Require all OpenMC source, batch, transport settings except daughter switch."""
    root = ET.parse(path).getroot()
    if root.tag != "settings":
        raise ValueError("not actual OpenMC settings XML")
    fission = root.find("create_fission_neutrons")
    if fission is None or fission.text not in ("true", "false"):
        raise ValueError("fission daughter transport setting absent")
    root.remove(fission)
    return ET.tostring(root, encoding="utf-8")


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
    os.environ["OPENMC_CROSS_SECTIONS"] = str(xs)
    base = args.output.resolve()
    no_fission = base / "suppressed_fission_control"
    with_fission = base / "fission_daughters_transported"
    base.mkdir(parents=True, exist_ok=True)

    cfg = NervaConfig(
        core_rings=3, hydrogen_temperature_k=293.6059129982851,
        batches=args.batches, inactive=0, particles=args.particles,
        heating_mesh_dimension=(8, 8, 8), axial_mesh_bins=8,
        diagnostic_energy_groups=8,
    )
    cfg.validate()
    nuclear = inspect_nuclear_data(
        cross_sections=xs, config=cfg,
        photon_transport=False, require_referenced_files=True,
    )
    write_report(nuclear, base / "evaluated_neutron_data_preflight.json")
    if not nuclear.ready:
        raise RuntimeError("evaluated neutron cross-section physical preflight failed")

    model = build_model(cfg, assembly="core", eigenvalue_entropy_mesh=False)
    model.settings.run_mode = "fixed source"
    model.settings.inactive = 0
    model.settings.statepoint = {"batches": [args.batches]}
    if model.settings.source is None:
        raise ValueError("actual physical prescribed external source is missing")

    fixed_xml = None
    def perform(seed: int, directory: Path, *, daughters: bool) -> None:
        nonlocal fixed_xml
        directory.mkdir(parents=True, exist_ok=True)
        model.settings.seed = seed
        model.settings.create_fission_neutrons = daughters
        model.export_to_xml(directory=directory)
        identity = tuple(sha(directory / name) for name in
                         ("geometry.xml", "materials.xml", "tallies.xml"))
        if fixed_xml is None:
            fixed_xml = identity
        elif identity != fixed_xml:
            raise ValueError("physical geometry, composition, or neutron scores changed between modes")
        if daughters:
            control = no_fission / f"seed_{seed}"
            if _physics_setting_identity(directory / "settings.xml") != (
                    _physics_setting_identity(control / "settings.xml")):
                raise ValueError("external source distribution or neutron solver settings differ")
        print("GENUINE OpenMC fixed-source mode", "fission-on" if daughters else "fission-off",
              "seed", seed, flush=True)
        with (directory / "openmc_neutron_transport.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [binary], cwd=directory,
                env={**os.environ, "OPENMC_CROSS_SECTIONS": str(xs),
                     "OMP_NUM_THREADS": "2"},
                stdout=log, stderr=subprocess.STDOUT, check=False,
            )
        if result.returncode:
            raise RuntimeError(
                f"genuine neutron transport failed, seed {seed}, "
                f"fission daughters {daughters}; see neutron execution log")

    for seed in args.seeds:
        perform(seed, no_fission / f"seed_{seed}", daughters=False)
    control = audit_four_seed_fixed_source(no_fission, list(args.seeds), args.batches)
    (no_fission / "genuine_fixed_source_core_neutron_balance.json").write_text(
        json.dumps(control, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if any(row["absolute_balance_error"] >= 1e-6 for row in control["per_seed"]):
        raise ValueError("actual baseline neutron fate is not conservative")

    for seed in args.seeds:
        perform(seed, with_fission / f"seed_{seed}", daughters=True)
    report = compare_four_seed_external_source_modes(
        with_fission, no_fission, list(args.seeds), args.batches
    )
    report["evaluated_cross_sections_xml_sha256"] = sha(xs)
    report["particles_per_seed_per_mode"] = args.particles * args.batches
    report["same_physical_neutron_model_byte_identical"] = True
    report["reactor_conditions"] = (
        "cold unchanged 3-ring NERVA-derived demonstration core; "
        "not calibrated NASA NERVA historical reactor"
    )
    target = base / "real_source_driven_fission_multiplication_validation.json"
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print("REAL OpenMC fixed-source fission secondary neutron closure: PASS", flush=True)
    print("Mean fission daughters produced per injected neutron:",
          report["mean_fission_daughter_neutrons_per_external_source"], flush=True)
    print("No criticality, eigenvalue, fuel design or local power validation", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
