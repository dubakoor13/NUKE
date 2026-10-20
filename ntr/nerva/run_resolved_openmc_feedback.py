"""Run REAL quasi-static OpenMC feedback for resolved NERVA fuel/tie channels.

This is a launchable engineering integration pathway, NOT a completed
historical NERVA solution. Requires installed OpenMC transport executable,
compatible nuclear cross sections, and enough memory/time for multiple
independent Monte Carlo eigenvalue statepoints.

Each Picard iterate:
  run_openmc_transport with updated UNIFORM fuel and H2 material T/density
  -> run_analysis --no-plots (normalization + instance reconstructions)
  -> validate_analysis_outputs -> reconstruct per-channel/tie deposition
  -> advance_resolved from the SAME old thermal state
  -> compare all resolved thermal profiles and material-state projections.

The prescribed global thermal reactor power is held constant and NEVER
computed from k-effective; Monte Carlo statistical convergence alone does
not establish physical transient nuclear-kinetics validity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

from .resolved_transient import (
    ResolvedParameters, initial_resolved_state, read_resolved_heating
)
from .resolved_openmc_feedback import (
    TransportSnapshot, advance_resolved_with_openmc
)


R_H2 = 4124.0  # nominal pure-H2 gas constant J/(kg K), uniform proxy only


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for buf in iter(lambda: fp.read(2**20), b""):
            h.update(buf)
    return h.hexdigest()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cross-sections", type=Path, required=True)
    p.add_argument("--power-mw", type=float, required=True)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--dt-s", type=float, required=True)
    p.add_argument("--initial-fuel-solid-k", type=float, required=True)
    p.add_argument("--initial-hydrogen-k", type=float, required=True)
    p.add_argument("--inlet-temperature-k", type=float, required=True)
    p.add_argument("--inlet-pressure-mpa", type=float, required=True)
    p.add_argument("--fuel-mass-flow-kg-s", type=float, required=True)
    p.add_argument("--tie-mass-flow-kg-s", type=float, required=True)
    p.add_argument("--fuel-solid-capacity-j-k", type=float, required=True)
    p.add_argument("--fuel-h2-capacity-j-k", type=float, required=True)
    p.add_argument("--fuel-ua-w-k", type=float, required=True)
    p.add_argument("--tie-solid-capacity-j-k", type=float, required=True)
    p.add_argument("--tie-supply-h2-capacity-j-k", type=float, required=True)
    p.add_argument("--tie-return-h2-capacity-j-k", type=float, required=True)
    p.add_argument("--tie-supply-ua-w-k", type=float, required=True)
    p.add_argument("--tie-return-ua-w-k", type=float, required=True)
    p.add_argument("--h2-reference-cp-j-kg-k", type=float, default=14300.)
    p.add_argument("--gas-enthalpy-model",
                   choices=("nasa7-h2", "constant"), default="nasa7-h2")
    p.add_argument("--feedback-max-iterations", type=int, default=5)
    p.add_argument("--feedback-tol-k", type=float, default=0.5)
    p.add_argument("--feedback-relaxation", type=float, default=1.0)
    p.add_argument("--max-unresolved-fraction", type=float, default=0.6)
    p.add_argument("--assembly",
                   choices=("core", "cluster", "mixed-core", "reactor"),
                   default="reactor")
    p.add_argument("--rings", type=int, default=5)
    p.add_argument("--particles", type=int, default=4000)
    p.add_argument("--batches", type=int, default=80)
    p.add_argument("--inactive", type=int, default=20)
    p.add_argument("--drum-angle", type=float, default=0.0)
    p.add_argument("--output", type=Path,
                   default=Path("build/nerva_resolved_openmc_feedback"))
    return p


def main() -> int:
    a = parser().parse_args()
    if not a.cross_sections.is_file():
        raise FileNotFoundError(a.cross_sections)
    if (a.steps < 1 or not math.isfinite(a.dt_s) or a.dt_s <= 0
            or a.power_mw <= 0 or not math.isfinite(a.power_mw)
            or a.inlet_pressure_mpa <= 0 or not math.isfinite(a.inlet_pressure_mpa)):
        raise ValueError("invalid positive power, pressure or step interval")
    if (a.initial_fuel_solid_k <= 0 or a.initial_hydrogen_k <= 0
            or a.inlet_temperature_k <= 0
            or a.fuel_mass_flow_kg_s <= 0 or a.tie_mass_flow_kg_s <= 0):
        raise ValueError("invalid temperature or flow input")
    params = ResolvedParameters(
        fuel_solid_capacity_j_k=a.fuel_solid_capacity_j_k,
        fuel_h2_capacity_j_k=a.fuel_h2_capacity_j_k,
        fuel_ua_w_k=a.fuel_ua_w_k,
        tie_solid_capacity_j_k=a.tie_solid_capacity_j_k,
        tie_supply_h2_capacity_j_k=a.tie_supply_h2_capacity_j_k,
        tie_return_h2_capacity_j_k=a.tie_return_h2_capacity_j_k,
        tie_supply_ua_w_k=a.tie_supply_ua_w_k,
        tie_return_ua_w_k=a.tie_return_ua_w_k,
        hydrogen_cp_reference_j_kg_k=a.h2_reference_cp_j_kg_k,
        enthalpy_model=a.gas_enthalpy_model,
    )
    params.validate()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    count = 0
    snapshot_records: list[dict] = []

    def provider(fuel_t: float, hydrogen_t: float, step: int) -> TransportSnapshot:
        nonlocal count
        if any(not math.isfinite(t) or t <= 0 or t > 3500
               for t in (fuel_t, hydrogen_t)):
            raise ValueError("transport material temperature outside study range")
        # Hold bulk pressure constant; no momentum or compressible pipe solve.
        density = (a.inlet_pressure_mpa*1e6)/(R_H2*hydrogen_t)/1000.0
        if not math.isfinite(density) or density <= 0:
            raise ValueError("invalid bulk H2 density")
        name = f"step_{step:04d}_transport_{count:05d}"
        count += 1
        transport_dir = root/"snapshots"/name/"transport"
        analysis_dir = root/"snapshots"/name/"analysis"
        run([
            sys.executable, "-m", "ntr.nerva.run_openmc_transport",
            "--cross-sections", str(a.cross_sections.resolve()),
            "--assembly", a.assembly,
            "--rings", str(a.rings),
            "--particles", str(a.particles),
            "--batches", str(a.batches),
            "--inactive", str(a.inactive),
            "--drum-angle", str(a.drum_angle),
            "--fuel-temperature-k", str(fuel_t),
            "--hydrogen-temperature-k", str(hydrogen_t),
            "--hydrogen-density-g-cm3", str(density),
            "--output", str(transport_dir),
        ])
        provenance = json.loads(
            (transport_dir/"transport_provenance.json").read_text(encoding="utf-8")
        )
        sp_info = provenance.get("statepoint")
        if not isinstance(sp_info, dict) or "sha256" not in sp_info:
            raise RuntimeError("transport did not complete and hash a statepoint")
        sp_path = Path(sp_info["path"])
        sp_hash = sha256(sp_path)
        if sp_hash != sp_info["sha256"]:
            raise RuntimeError("transport statepoint provenance checksum mismatch")
        config = provenance.get("config", {})
        if (not math.isclose(float(config.get("fuel_temperature_k", -1)), fuel_t,
                            rel_tol=1e-12)
                or not math.isclose(float(config.get("hydrogen_temperature_k", -1)),
                                    hydrogen_t, rel_tol=1e-12)):
            raise RuntimeError("transport material temperatures do not match Picard input")
        nuclear = provenance.get("nuclear_data", {})
        if nuclear.get("explicit_temperature_coverage_ok") is not True:
            raise RuntimeError("transport nuclear temperature coverage failed")
        cross_sections_sha = nuclear.get("cross_sections_sha256")
        if not cross_sections_sha:
            raise RuntimeError("missing nuclear-data hash in transport result")

        run([
            sys.executable, "-m", "ntr.nerva.run_analysis", str(sp_path),
            "--power-mw", str(a.power_mw),
            "--rings", str(a.rings),
            "--fuel-mass-flow-kg-s", str(a.fuel_mass_flow_kg_s),
            "--tie-mass-flow-kg-s", str(a.tie_mass_flow_kg_s),
            "--inlet-temperature-k", str(a.inlet_temperature_k),
            "--inlet-pressure-mpa", str(a.inlet_pressure_mpa),
            "--no-plots", "--output-root", str(analysis_dir),
        ])
        qa_path = analysis_dir/"analysis_validation.json"
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        if qa.get("status") != "PASS":
            raise RuntimeError("OpenMC power/reconstruction QA did not pass")
        source_paths = {
            "power": analysis_dir/"power"/"nerva_mesh_fields.npz",
            "fuel": (analysis_dir/"element_channels"/
                     "element_channel_power_reconstruction.npz"),
            "tie": (analysis_dir/"tie_instances"/
                    "tie_instance_reconstruction.npz"),
        }
        heating = read_resolved_heating(
            source_paths["fuel"], source_paths["tie"], source_paths["power"]
        )
        if not math.isclose(heating.total_openmc_power_w,
                            a.power_mw*1e6, rel_tol=1e-7):
            raise RuntimeError("OpenMC total power differs from imposed thermal power")
        snapshot_records.append({
            "step": step, "name": name,
            "requested_fuel_temperature_k": fuel_t,
            "requested_hydrogen_temperature_k": hydrogen_t,
            "hydrogen_bulk_density_g_cm3": density,
            "statepoint": str(sp_path),
            "statepoint_sha256": sp_hash,
            "nuclear_data_sha256": cross_sections_sha,
            "analysis_qa_passed": True,
            "power_w": heating.total_openmc_power_w,
            "unresolved_power_w": heating.unresolved_power_w,
            "reconstruction_sha256": {
                k: sha256(p) for k,p in source_paths.items()
            },
        })
        return TransportSnapshot(
            heating=heating, source_kind="openmc-statepoint",
            statepoint_sha256=sp_hash,
            nuclear_data_sha256=str(cross_sections_sha),
            fuel_material_temperature_k=fuel_t,
            hydrogen_material_temperature_k=hydrogen_t,
            analysis_qa_passed=True,
            source_path=str(sp_path),
        )

    # Seed first snapshot from declared initial state so its channel
    # layout can initialize the transient state; reuse it exactly once.
    seed = provider(a.initial_fuel_solid_k, a.initial_hydrogen_k, 0)
    state = initial_resolved_state(
        seed.heating,
        solid_temperature_k=a.initial_fuel_solid_k,
        gas_temperature_k=a.initial_hydrogen_k,
    )
    initial_used = False
    step_records: list[dict] = []
    for step_id in range(a.steps):
        def transport_at(fts: float, h2t: float) -> TransportSnapshot:
            nonlocal initial_used
            if step_id == 0 and not initial_used:
                initial_used = True
                if (math.isclose(fts, a.initial_fuel_solid_k, rel_tol=1e-12)
                        and math.isclose(h2t, a.initial_hydrogen_k, rel_tol=1e-12)):
                    return seed
            return provider(fts, h2t, step_id)

        coupled = advance_resolved_with_openmc(
            state, params, a.inlet_temperature_k, a.dt_s,
            transport_at,
            max_iterations=a.feedback_max_iterations,
            convergence_temperature_k=a.feedback_tol_k,
            relaxation=a.feedback_relaxation,
            max_unresolved_fraction=a.max_unresolved_fraction,
        )
        state = coupled.result.state
        rec = {
            "step": step_id+1,
            "time_s": (step_id+1)*a.dt_s,
            "iterations": coupled.iterations,
            "feedback_error_k": coupled.temperature_error_k,
            "profile_error_k": coupled.profile_error_k,
            "max_fuel_outlet_k": float(np.max(coupled.result.fuel_outlet_t_k)),
            "max_tie_outlet_k": float(np.max(coupled.result.tie_outlet_t_k)),
            "first_law_residual_j": coupled.result.energy_residual_j,
            "unresolved_energy_j": state.unresolved_energy_j,
            "last_statepoint_sha256": coupled.last_snapshot.statepoint_sha256,
            "iteration_trace": [asdict(x) for x in coupled.diagnostics],
        }
        step_records.append(rec)
        print(
            f"RESOLVED OpenMC step {step_id+1}/{a.steps}, "
            f"Picard iters={coupled.iterations}, "
            f"thermal residual={coupled.temperature_error_k:.5g} K, "
            f"energy residual={rec['first_law_residual_j']:.5g} J"
        )
        (root/"resume_checkpoint.json").write_text(
            json.dumps({"complete_steps": step_id+1,
                        "current_time_s": rec["time_s"],
                        "last_statepoint_sha256":
                        coupled.last_snapshot.statepoint_sha256,
                        "historical_nerva_validation": False}, indent=2)+"\n",
            encoding="utf-8"
        )
        np.savez_compressed(
            root/"last_converged_resolved_state.npz",
            fuel_solid_t_k=state.fuel_solid_t_k,
            fuel_h2_t_k=state.fuel_h2_t_k,
            tie_solid_t_k=state.tie_solid_t_k,
            tie_supply_t_k=state.tie_supply_t_k,
            tie_return_t_k=state.tie_return_t_k,
            unresolved_energy_j=np.array([state.unresolved_energy_j])
        )

    report = {
        "status": "PASS",
        "source": "REAL new OpenMC statepoint per Picard iterate",
        "reactor_thermal_power_model": "EXTERNALLY PRESCRIBED; no point kinetics",
        "neutron_transport_feedback": "uniform fuel and H2 material temperatures / "
            "fixed-pressure ideal-gas density proxy",
        "channel_thermal_scope": "individual 19-channel fuel elements and "
            "shared-solid per-tie opposing hydrogen flow",
        "historical_nerva_validation": False,
        "nuclear_data": str(a.cross_sections.resolve()),
        "nuclear_data_sha256": seed.nuclear_data_sha256,
        "temperature_feedback_tolerance_k": a.feedback_tol_k,
        "max_feedback_iterations": a.feedback_max_iterations,
        "reconstruction_qa": "run_analysis and independent channel/tie "
                              "partition-integral checks per iteration",
        "thermal_parameters_are_user_assumptions": True,
        "snapshots": snapshot_records,
        "steps": step_records,
    }
    report_path = root/"resolved_openmc_feedback_report.json"
    report_path.write_text(json.dumps(report, indent=2)+"\n",
                           encoding="utf-8")
    print(f"Resolved OpenMC feedback complete: {report_path}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
