"""Run power-normalized OpenMC/thermal time stepping with explicit QA.

Real feedback mode calls existing OpenMC neutron-transport runner for every
Picard iterate with updated uniform fuel and coolant material conditions.
A real installed OpenMC executable and verified nuclear XS library are
required. Reactor power is prescribed, never inferred from keff.

Examples:
  python -m ntr.nerva.run_openmc_transient --help
  python -m ntr.nerva.run_openmc_transient --fields SYNTHETIC.npz \
      --allow-synthetic ...capacity/UA/flow arguments...
  python -m ntr.nerva.run_openmc_transient --cross-sections DATA.xml \
      ...capacity/UA/flow arguments...
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

from .openmc_transient import (
    Branch, initial_state, advance_openmc_transient,
    advance_with_transport_feedback, power_map_from_npz,
)

GAS_CONSTANT_H2 = 4124.0  # J/kg/K, bulk ideal H2 approximation for OpenMC density


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--cross-sections", type=Path,
                        help="Run fresh OpenMC statepoints for each thermal Picard iteration.")
    source.add_argument("--statepoint", type=Path,
                        help="Existing real OpenMC statepoint; frozen neutronic shape.")
    source.add_argument("--fields", type=Path,
                        help="Existing NPZ power map, synthetic ONLY with opt-in.")
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--power-mw", type=float, required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--dt-s", type=float, required=True)
    p.add_argument("--inlet-temperature-k", type=float, required=True)
    p.add_argument("--initial-solid-k", type=float, required=True)
    p.add_argument("--initial-gas-k", type=float, required=True)
    p.add_argument("--inlet-pressure-mpa", type=float, default=5.0,
                   help="Fixed pressure used ONLY for volume-mean H2 density feedback.")
    p.add_argument("--fuel-mass-flow-kg-s", type=float, required=True)
    p.add_argument("--tie-mass-flow-kg-s", type=float, required=True)
    p.add_argument("--fuel-solid-capacity-j-k", type=float, required=True,
                   help="Effective axial-cell heat capacity J/K (not entire core).")
    p.add_argument("--fuel-gas-capacity-j-k", type=float, required=True)
    p.add_argument("--tie-solid-capacity-j-k", type=float, required=True)
    p.add_argument("--tie-gas-capacity-j-k", type=float, required=True)
    p.add_argument("--fuel-ua-w-k", type=float, required=True)
    p.add_argument("--tie-ua-w-k", type=float, required=True)
    p.add_argument("--hydrogen-cp-j-kg-k", type=float, default=14300.0,
                   help="Reference Cp: gas inventory Cgas/Cp_ref, constant-model Cp.")
    p.add_argument("--integrator", choices=("implicit", "explicit"),
                   default="implicit",
                   help="Implicit finite-volume enthalpy integration is recommended.")
    p.add_argument("--gas-enthalpy-model", choices=("nasa7-h2", "constant"),
                   default="nasa7-h2",
                   help="Pure H2 NASA-7 variable enthalpy or legacy constant Cp.")
    p.add_argument("--feedback-max-iterations", type=int, default=4)
    p.add_argument("--feedback-temperature-tolerance-k", type=float, default=0.5)
    p.add_argument("--max-unresolved-fraction", type=float, default=0.50)
    p.add_argument("--assembly", choices=["core", "cluster", "mixed-core", "reactor"],
                   default="reactor")
    p.add_argument("--rings", type=int, default=5)
    p.add_argument("--particles", type=int, default=4000)
    p.add_argument("--batches", type=int, default=80)
    p.add_argument("--inactive", type=int, default=20)
    p.add_argument("--drum-angle", type=float, default=0.)
    p.add_argument("--output", type=Path, default=Path("build/nerva_openmc_transient"))
    return p.parse_args()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _subprocess(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def postprocess_statepoint(path: Path, power_mw: float, directory: Path) -> Path:
    """Use existing OpenMC statepoint postprocessing, not an independent
    tally normalization that might double-count score filters.
    """
    if not path.is_file() or path.suffix.lower() != ".h5":
        raise ValueError("real transport source requires existing .h5 statepoint")
    _subprocess([
        sys.executable, "-m", "ntr.nerva.postprocess_statepoint",
        str(path.resolve()), "--power-mw", str(power_mw),
        "--output", str(directory.resolve()),
    ])
    f = directory / "nerva_mesh_fields.npz"
    if not f.is_file():
        raise RuntimeError("OpenMC postprocessing produced no mesh power map")
    return f


def main() -> int:
    args = parse_args()
    if args.fields and not args.allow_synthetic:
        raise ValueError("NPZ alone is not verifiable OpenMC provenance. "
                         "For CI/demo fixture provide --allow-synthetic.")
    if args.allow_synthetic and not args.fields:
        raise ValueError("--allow-synthetic only valid with --fields")
    if args.integrator == "explicit" and args.gas_enthalpy_model != "constant":
        raise ValueError("explicit mode requires --gas-enthalpy-model constant")
    if args.steps <= 0 or not 0 < args.max_unresolved_fraction <= 1:
        raise ValueError("steps must be positive and unresolved limit in (0,1]")
    if args.dt_s <= 0 or not math.isfinite(args.dt_s):
        raise ValueError("dt-s must be positive and finite")
    if not math.isfinite(args.power_mw) or args.power_mw <= 0:
        raise ValueError("reactor power must be positive and finite")
    if args.cross_sections and (
        not args.cross_sections.is_file() or args.inlet_pressure_mpa <= 0
    ):
        raise ValueError("OpenMC feedback requires valid cross_sections.xml and positive gas pressure")
    fuel = Branch(args.fuel_solid_capacity_j_k, args.fuel_gas_capacity_j_k,
                  args.fuel_ua_w_k, args.fuel_mass_flow_kg_s,
                  args.hydrogen_cp_j_kg_k)
    tie = Branch(args.tie_solid_capacity_j_k, args.tie_gas_capacity_j_k,
                 args.tie_ua_w_k, args.tie_mass_flow_kg_s,
                 args.hydrogen_cp_j_kg_k)
    fuel.validate()
    tie.validate()
    power_w = args.power_mw*1e6
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    snapshot_count = 0
    statepoint_digests: list[dict] = []

    def real_provider(
        fuel_temperature_k: float, gas_temperature_k: float,
        istep: int
    ):
        nonlocal snapshot_count
        if not 0 < fuel_temperature_k <= 3500 or not 0 < gas_temperature_k <= 3500:
            raise RuntimeError("out-of-range bulk material feedback temperature")
        density_g_cm3 = (
            args.inlet_pressure_mpa*1e6
            /(GAS_CONSTANT_H2*gas_temperature_k)/1000.0
        )
        if not math.isfinite(density_g_cm3) or density_g_cm3 <= 0:
            raise RuntimeError("invalid hydrogen density estimate")
        name = f"s{istep:04d}_picard{snapshot_count:04d}"
        snapshot_count += 1
        transport = root / "openmc_runs" / name
        command = [
            sys.executable, "-m", "ntr.nerva.run_openmc_transport",
            "--cross-sections", str(args.cross_sections.resolve()),
            "--assembly", args.assembly,
            "--rings", str(args.rings),
            "--particles", str(args.particles),
            "--batches", str(args.batches),
            "--inactive", str(args.inactive),
            "--drum-angle", str(args.drum_angle),
            "--fuel-temperature-k", str(fuel_temperature_k),
            "--hydrogen-temperature-k", str(gas_temperature_k),
            "--hydrogen-density-g-cm3", str(density_g_cm3),
            "--output", str(transport),
        ]
        _subprocess(command)
        prov = json.loads(
            (transport/"transport_provenance.json").read_text(encoding="utf-8")
        )
        statepoint_info = prov.get("statepoint")
        if not isinstance(statepoint_info, dict):
            raise RuntimeError("OpenMC transport did not finish with statepoint")
        sp = Path(statepoint_info["path"])
        if _sha(sp) != statepoint_info["sha256"]:
            raise RuntimeError("OpenMC statepoint changed from transport provenance")
        if not prov["nuclear_data"]["explicit_temperature_coverage_ok"]:
            raise RuntimeError("temperature-dependent XS coverage was not verified")
        outfields = postprocess_statepoint(sp, args.power_mw, transport/"normalized")
        statepoint_digests.append({
            "step": istep, "snapshot": name, "statepoint_sha256": _sha(sp),
            "statepoint": str(sp), "fuel_temperature_k": fuel_temperature_k,
            "hydrogen_temperature_k": gas_temperature_k,
            "hydrogen_density_g_cm3": density_g_cm3,
            "neutron_data_sha256": prov["nuclear_data"]["cross_sections_sha256"],
        })
        return power_map_from_npz(outfields, power_w,
            source_kind="openmc-statepoint", require_full_channels=True)

    if args.fields:
        fixture = power_map_from_npz(args.fields, power_w,
            source_kind="synthetic", require_full_channels=False)
    elif args.statepoint:
        fp = postprocess_statepoint(
            args.statepoint, args.power_mw, root/"normalized_statepoint"
        )
        fixture = power_map_from_npz(
            fp, power_w, source_kind="openmc-statepoint", require_full_channels=True
        )
        statepoint_digests.append({
            "statepoint": str(args.statepoint.resolve()),
            "statepoint_sha256": _sha(args.statepoint)
        })
    else:
        fixture = None

    # Feedback adapter has to get the power map first to determine axial bins.
    # This first real snapshot is consumed by the first Picard iterate, so
    # its source temperature is the initial condition and no work is wasted.
    cache: dict[tuple[float, float, int], object] = {}
    def provider(solid: float, gas: float, istep: int):
        key = (solid, gas, istep)
        if key not in cache:
            cache[key] = real_provider(solid, gas, istep)
        return cache[key]

    if fixture is None:
        first = provider(args.initial_solid_k, args.initial_gas_k, 0)
        n = len(first.fuel_solid_w)
    else:
        n = len(fixture.fuel_solid_w)
    state = initial_state(n, args.initial_solid_k, args.initial_gas_k)
    records: list[dict] = []
    for k in range(args.steps):
        if fixture is None:
            result = advance_with_transport_feedback(
                lambda fs, gs: provider(fs, gs, k),
                state, fuel, tie, args.inlet_temperature_k, args.dt_s,
                max_iterations=args.feedback_max_iterations,
                temperature_tolerance_k=args.feedback_temperature_tolerance_k,
                integrator=args.integrator,
                enthalpy_model=args.gas_enthalpy_model,
            )
            trial = result.step
            power = result.last_map
            converged, iterations, feedback_delta = (
                result.converged, result.iterations,
                result.max_temperature_change_k
            )
        else:
            power = fixture
            trial = advance_openmc_transient(
                power, state, fuel, tie,
                args.inlet_temperature_k, args.dt_s,
                integrator=args.integrator,
                enthalpy_model=args.gas_enthalpy_model,
            )
            converged, iterations, feedback_delta = None, 0, None
        frac = power.unresolved_w / power.prescribed_power_w
        if frac > args.max_unresolved_fraction:
            raise RuntimeError(
                "unresolved reactor deposition fraction exceeds configured ceiling: "
                f"{frac:.4%} > {args.max_unresolved_fraction:.4%}"
            )
        state = trial.state
        records.append({
            "time_s": (k+1)*args.dt_s,
            "fuel_temperature_k": list(state.fuel.solid_t_k),
            "fuel_coolant_temperature_k": list(state.fuel.gas_t_k),
            "tie_temperature_k": list(state.tie.solid_t_k),
            "tie_coolant_temperature_k": list(state.tie.gas_t_k),
            "fuel_outlet_temperature_k": trial.outlet_fuel_k,
            "tie_outlet_temperature_k": trial.outlet_tie_k,
            "input_reactor_power_w": trial.reactor_power_w,
            "modeled_power_w": trial.modeled_power_w,
            "unresolved_power_w": trial.unresolved_power_w,
            "direct_gas_heating_power_w": trial.gas_direct_power_w,
            "unresolved_stored_energy_j": state.unresolved_energy_j,
            "stored_energy_change_j": trial.stored_energy_change_j,
            "inlet_enthalpy_power_w": trial.inlet_enthalpy_power_w,
            "outlet_enthalpy_power_w": trial.outlet_enthalpy_power_w,
            "energy_residual_j": trial.energy_residual_j,
            "source_kind": power.source_kind,
            "source_npz_sha256": power.source_sha256,
            "source_reported_power_w_before_normalization": power.source_reported_power_w,
            "temperature_feedback_converged": converged,
            "temperature_feedback_iterations": iterations,
            "feedback_temperature_change_k": feedback_delta,
        })
        print(f"step {k+1}/{args.steps}: t={records[-1]['time_s']:.3g}s, "
              f"fuel outlet={trial.outlet_fuel_k:.2f} K, "
              f"tie outlet={trial.outlet_tie_k:.2f} K, "
              f"unresolved={frac:.2%}, balance={trial.energy_residual_j:.3g} J")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "source_type": "real_feedback_openmc" if args.cross_sections else (
            "openmc_statepoint_frozen" if args.statepoint else "synthetic_unverified"
        ),
        "power_control": "EXTERNALLY PRESCRIBED constant thermal power; no reactor kinetics",
        "thermal_model": (
            "aggregate fuel and tie axial solid/gas; fixed gas inventory, "
            "constant solid heat capacity and UA, co-current; "
            + args.integrator + " / " + args.gas_enthalpy_model
        ),
        "integrator": args.integrator,
        "gas_enthalpy_model": args.gas_enthalpy_model,
        "gas_inventory_definition": "gas mass per axial bin = gas_capacity_J_K / reference_cp_J_kg_K",
        "feedback_model": "uniform fuel and H2 material-state OpenMC Picard iteration"
           if args.cross_sections else "NONE; frozen power distribution",
        "historical_model_validated": False,
        "system_parameters": {
            "power_w": power_w, "step_s": args.dt_s, "steps": args.steps,
            "fuel": asdict(fuel), "tie": asdict(tie),
            "maximum_unresolved_fraction": args.max_unresolved_fraction,
        },
        "statepoints": statepoint_digests,
        "records": records,
    }
    dest = root/"openmc_transient_report.json"
    dest.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with (root/"openmc_transient_history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "time_s", "fuel_outlet_temperature_k", "tie_outlet_temperature_k",
            "input_reactor_power_w", "modeled_power_w",
            "unresolved_power_w", "direct_gas_heating_power_w",
            "unresolved_stored_energy_j", "energy_residual_j",
            "temperature_feedback_converged", "temperature_feedback_iterations",
        ])
        w.writeheader()
        for row in records:
            w.writerow({key: row[key] for key in w.fieldnames})
    print(f"OpenMC transient output: {dest}")
    if fixture is not None:
        print("NOTE: power distribution frozen; NO neutron/thermal feedback solved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
