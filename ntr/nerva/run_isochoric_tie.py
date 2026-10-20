"""Run NERVA rigid-volume real-fluid tie counterflow on conserved power NPZ.

The solver enforces fixed-volume hydrogen mass continuity for imposed
equal in/out tie flow, updates (u,p,T,h,Q) via CoolProp Hydrogen, and
reports true internal-energy and advective enthalpy balances. No
spatial pressure-loss or independently solved fluid momentum model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .isochoric_tie import (
    IsochoricTieParameters, initial_isochoric_tie_state,
    advance_isochoric_tie,
)
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh_fields",type=Path)
    p.add_argument("fuel_reconstruction",type=Path)
    p.add_argument("tie_reconstruction",type=Path)
    p.add_argument("--initial-pressure-pa",type=float,required=True)
    init=p.add_mutually_exclusive_group(required=True)
    init.add_argument("--initial-quality",type=float)
    init.add_argument("--initial-fluid-temperature-k",type=float)
    p.add_argument("--initial-solid-temperature-k",type=float,required=True)
    p.add_argument("--supply-cell-volume-m3",type=float,required=True)
    p.add_argument("--return-cell-volume-m3",type=float,required=True)
    p.add_argument("--solid-capacity-j-k",type=float,required=True)
    p.add_argument("--supply-ua-w-k",type=float,required=True)
    p.add_argument("--return-ua-w-k",type=float,required=True)
    p.add_argument("--dt-s",type=float,required=True)
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--output",type=Path,default=Path("build/nerva_isochoric_tie"))
    a=p.parse_args()
    if not 1<=a.steps<=10000:
        raise ValueError("invalid number of transient steps")
    heating=read_resolved_heating(
        a.fuel_reconstruction,a.tie_reconstruction,a.mesh_fields,
    )
    eos=HydrogenEOS()
    params=IsochoricTieParameters(
        supply_cell_volume_m3=a.supply_cell_volume_m3,
        return_cell_volume_m3=a.return_cell_volume_m3,
        solid_capacity_j_k=a.solid_capacity_j_k,
        supply_ua_w_k=a.supply_ua_w_k,
        return_ua_w_k=a.return_ua_w_k,
    )
    state=initial_isochoric_tie_state(
        heating,params,a.initial_pressure_pa,
        quality=a.initial_quality,
        fluid_temperature_k=a.initial_fluid_temperature_k,
        solid_temperature_k=a.initial_solid_temperature_k,
        eos=eos,
    )
    rho=float(state.supply_mass_kg[0,0])/params.supply_cell_volume_m3
    u=float(state.supply_internal_energy_j_kg[0,0])
    initial_h=u+a.initial_pressure_pa/rho
    rows=[]
    step=None
    for i in range(a.steps):
        step=advance_isochoric_tie(
            heating,state,params,initial_h,a.dt_s,eos=eos,
        )
        state=step.state
        rows.append({
            "step":i+1,
            "elapsed_s":(i+1)*a.dt_s,
            "nuclear_tie_deposition_w":step.nuclear_deposition_w,
            "solid_storage_rate_w":step.solid_storage_rate_w,
            "hydrogen_internal_energy_storage_rate_w":
                step.fluid_internal_energy_storage_rate_w,
            "inlet_enthalpy_power_w":step.inlet_enthalpy_power_w,
            "outlet_enthalpy_power_w":step.outlet_enthalpy_power_w,
            "first_law_residual_j":step.energy_residual_j,
            "max_local_mass_residual_kg_s":
                step.max_local_mass_balance_residual_kg_s,
            "max_local_energy_residual_w":
                step.max_local_energy_balance_residual_w,
            "min_pressure_pa":min(float(np.min(step.supply_pressure_pa)),
                                  float(np.min(step.return_pressure_pa))),
            "max_pressure_pa":max(float(np.max(step.supply_pressure_pa)),
                                  float(np.max(step.return_pressure_pa))),
        })
    root=a.output
    root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        root/"isochoric_tie_final_state.npz",
        solid_temperature_k=state.solid_temperature_k,
        supply_mass_kg=state.supply_mass_kg,
        return_mass_kg=state.return_mass_kg,
        supply_internal_energy_j_kg=state.supply_internal_energy_j_kg,
        return_internal_energy_j_kg=state.return_internal_energy_j_kg,
        supply_pressure_pa=step.supply_pressure_pa,
        return_pressure_pa=step.return_pressure_pa,
        supply_temperature_k=step.supply_temperature_k,
        return_temperature_k=step.return_temperature_k,
        supply_enthalpy_j_kg=step.supply_enthalpy_j_kg,
        return_enthalpy_j_kg=step.return_enthalpy_j_kg,
        supply_density_kg_m3=step.supply_density_kg_m3,
        return_density_kg_m3=step.return_density_kg_m3,
        supply_quality=step.supply_quality,
        return_quality=step.return_quality,
    )
    report={
        "status":"PASS",
        "source_kind":"normalized external NPZ power reconstruction; provenance unverified",
        "real_openmc_transport_executed":False,
        "historical_nerva_validation":False,
        "compressible_eos_internal_energy_solved":True,
        "equal_cell_in_out_mass_flow_prescribed":True,
        "independent_momentum_pressure_loss_solved":False,
        "hydraulic_pressure_coupling_converged":False,
        "two_phase_slip_or_chf_validated":False,
        "initial_pressure_pa":a.initial_pressure_pa,
        "initial_hydrogen_specific_internal_energy_j_kg":u,
        "initial_hydrogen_inlet_enthalpy_j_kg":initial_h,
        "tie_total_mass_flow_kg_s":heating.total_tie_flow_kg_s,
        "tie_nuclear_deposition_w":step.nuclear_deposition_w,
        "parameters":asdict(params),
        "history":rows,
        "source_files":{
            "mesh":str(a.mesh_fields),
            "fuel":str(a.fuel_reconstruction),
            "tie":str(a.tie_reconstruction),
        },
    }
    (root/"isochoric_tie_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8",
    )
    print("NERVA fixed-volume cryogenic H2 tie energy closure: PASS")
    print(f"  final pressure {rows[-1]['min_pressure_pa']:.8g} "
          f".. {rows[-1]['max_pressure_pa']:.8g} Pa")
    print(f"  first-law residual {step.energy_residual_j:.8g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
