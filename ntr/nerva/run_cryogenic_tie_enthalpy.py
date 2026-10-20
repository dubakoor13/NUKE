"""Run real-fluid cryogenic tie solid/H2 counterflow on existing NPZ heating.

CoolProp HEOS::Hydrogen provides p,h,T,rho,quality across saturation.
OpenMC deposition is provided and frozen: this program NEVER runs transport.
The user supplies pressure, constant H2 mass per cell, and wall UA.
Momentum, CHF, void slip, compressible inventory and pump coupling are
outside this model. Outputs never assert NASA-qualified NERVA behavior.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .cryogenic_tie_enthalpy import (
    CryogenicTieParameters, CryogenicTieState, advance_cryogenic_tie,
)
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh_fields",type=Path)
    p.add_argument("fuel_reconstruction",type=Path)
    p.add_argument("tie_reconstruction",type=Path)
    p.add_argument("--pressure-pa",type=float,required=True)
    inlet=p.add_mutually_exclusive_group(required=True)
    inlet.add_argument("--inlet-temperature-k",type=float)
    inlet.add_argument("--inlet-quality",type=float)
    old=p.add_mutually_exclusive_group()
    old.add_argument("--initial-fluid-temperature-k",type=float)
    old.add_argument("--initial-fluid-quality",type=float)
    p.add_argument("--initial-solid-temperature-k",type=float,required=True)
    p.add_argument("--solid-capacity-j-k",type=float,required=True)
    p.add_argument("--supply-inventory-kg",type=float,required=True)
    p.add_argument("--return-inventory-kg",type=float,required=True)
    p.add_argument("--supply-ua-w-k",type=float,required=True)
    p.add_argument("--return-ua-w-k",type=float,required=True)
    p.add_argument("--dt-s",type=float,required=True)
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_cryogenic_tie"))
    return p


def _enthalpy(eos: HydrogenEOS, pressure: float,
              temp: float | None, quality: float | None) -> float:
    if quality is not None:
        if not math.isfinite(quality) or not 0<=quality<=1:
            raise ValueError("hydrogen vapor quality must be within [0,1]")
        return eos.get("Hmass","P",pressure,"Q",quality)
    if temp is None or not math.isfinite(temp) or temp<=0:
        raise ValueError("invalid real-fluid hydrogen inlet temperature")
    return eos.tp("Hmass",temp,pressure)


def main() -> int:
    a=parser().parse_args()
    if not 1<=a.steps<=10000:
        raise ValueError("invalid cryogenic tie number of timesteps")
    if not math.isfinite(a.dt_s) or a.dt_s<=0:
        raise ValueError("positive cryogenic tie timestep required")
    eos=HydrogenEOS()
    heating=read_resolved_heating(
        a.fuel_reconstruction,a.tie_reconstruction,a.mesh_fields,
    )
    params=CryogenicTieParameters(
        pressure_pa=a.pressure_pa,
        solid_capacity_j_k=a.solid_capacity_j_k,
        supply_inventory_kg=a.supply_inventory_kg,
        return_inventory_kg=a.return_inventory_kg,
        supply_ua_w_k=a.supply_ua_w_k,
        return_ua_w_k=a.return_ua_w_k,
    )
    params.validate()
    inlet_h=_enthalpy(eos,a.pressure_pa,
                       a.inlet_temperature_k,a.inlet_quality)
    old_h=(
        _enthalpy(eos,a.pressure_pa,a.initial_fluid_temperature_k,
                  a.initial_fluid_quality)
        if a.initial_fluid_temperature_k is not None or
           a.initial_fluid_quality is not None else inlet_h
    )
    shape=heating.tie_solid_w.shape
    if (not math.isfinite(a.initial_solid_temperature_k)
            or a.initial_solid_temperature_k<=0):
        raise ValueError("invalid initial tie solid temperature")
    state=CryogenicTieState(
        np.full(shape,a.initial_solid_temperature_k),
        np.full(shape,old_h),np.full(shape,old_h),
    )
    rows=[]
    latest=None
    for i in range(a.steps):
        latest=advance_cryogenic_tie(
            heating,state,params,inlet_h,a.dt_s,eos=eos,
        )
        state=latest.state
        rows.append({
            "step":i+1,"elapsed_s":(i+1)*a.dt_s,
            "nuclear_deposition_w":latest.nuclear_deposition_w,
            "solid_storage_rate_w":latest.solid_storage_rate_w,
            "hydrogen_storage_rate_w":latest.hydrogen_storage_rate_w,
            "supply_wall_power_w":float(np.sum(latest.supply_wall_power_w)),
            "return_wall_power_w":float(np.sum(latest.return_wall_power_w)),
            "energy_residual_j":latest.energy_residual_j,
            "maximum_return_quality":float(np.max(latest.return_quality)),
            "maximum_return_temperature_k":float(np.max(latest.return_temperature_k)),
            "picard_iterations":latest.max_picard_iterations,
        })
    root=a.output
    root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        root/"cryogenic_tie_final_state.npz",
        solid_temperature_k=state.solid_temperature_k,
        supply_enthalpy_j_kg=state.supply_enthalpy_j_kg,
        return_enthalpy_j_kg=state.return_enthalpy_j_kg,
        supply_temperature_k=latest.supply_temperature_k,
        return_temperature_k=latest.return_temperature_k,
        supply_quality=latest.supply_quality,
        return_quality=latest.return_quality,
        supply_density_kg_m3=latest.supply_density_kg_m3,
        return_density_kg_m3=latest.return_density_kg_m3,
        supply_wall_power_w=latest.supply_wall_power_w,
        return_wall_power_w=latest.return_wall_power_w,
        pressure_pa=np.array(a.pressure_pa),
    )
    summary={
        "status":"PASS",
        "source_kind":"external normalized NPZ nuclear heating; no transport provenance verified",
        "real_openmc_transport_executed":False,
        "historical_nerva_validation":False,
        "physical_scope":"one pressure, prescribed mass flow, fixed H2 cell inventory; "
                         "CoolProp phase equilibrium; implicit shared-solid/two-duct energy",
        "momentum_pressure_drop_solved":False,
        "boiling_heat_transfer_or_chf_validated":False,
        "volume_mass_pressure_continuity_solved":False,
        "pressure_pa":a.pressure_pa,
        "inlet_enthalpy_j_kg":inlet_h,
        "parameters":asdict(params),
        "reactor_power_w":heating.total_openmc_power_w,
        "resolved_tie_deposition_w":latest.nuclear_deposition_w,
        "unresolved_reactor_power_w":heating.unresolved_power_w,
        "initialization_kind":"quality" if a.initial_fluid_quality is not None
                              else "temperature" if a.initial_fluid_temperature_k is not None
                              else "at-inlet",
        "history":rows,
        "final":{"inlet_enthalpy_power_w":latest.inlet_enthalpy_power_w,
                 "outlet_enthalpy_power_w":latest.outlet_enthalpy_power_w,
                 "max_local_solid_residual_w":latest.max_local_solid_residual_w,
                 "max_local_hydrogen_residual_w":latest.max_local_hydrogen_residual_w,
                 "supply_two_phase_cells":int(np.count_nonzero(
                     (latest.supply_quality>=0)&(latest.supply_quality<=1)
                 )),
                 "return_two_phase_cells":int(np.count_nonzero(
                     (latest.return_quality>=0)&(latest.return_quality<=1)
                 )),
        },
    }
    (root/"cryogenic_tie_report.json").write_text(
        json.dumps(summary,indent=2,allow_nan=False)+"\n",
        encoding="utf-8",
    )
    print("NERVA cryogenic real-fluid tie counterflow: PASS")
    print(f"  p={a.pressure_pa/1e6:.6g} MPa; tie deposition={latest.nuclear_deposition_w:.8g} W")
    print(f"  T-return-max={latest.return_temperature_k.max():.8g} K; "
          f"first-law residual={latest.energy_residual_j:.8g} J")
    print(f"  report: {root/'cryogenic_tie_report.json'}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
