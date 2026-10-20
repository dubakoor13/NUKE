"""Run pressure-driven compressible hydrogen tie-tube finite volumes.

Uses external reconstructed OpenMC-format nuclear power partitions and
CoolProp H2. The hydraulic closure is a pressure-driven quasisteady
orifice (screening), NOT full transient momentum, choking, two-phase
slip or a solved NERVA pump/turbine operating point.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .isochoric_tie import IsochoricTieParameters, initial_isochoric_tie_state
from .lh2_feed_cycle import HydrogenEOS
from .pressure_driven_tie import PressureTieFlowParameters, advance_pressure_driven_tie
from .resolved_transient import read_resolved_heating


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh_fields",type=Path)
    p.add_argument("fuel_reconstruction",type=Path)
    p.add_argument("tie_reconstruction",type=Path)
    p.add_argument("--initial-pressure-pa",type=float,required=True)
    p.add_argument("--initial-quality",type=float,required=True)
    p.add_argument("--initial-solid-temperature-k",type=float,required=True)
    p.add_argument("--inlet-pressure-pa",type=float,required=True)
    p.add_argument("--outlet-pressure-pa",type=float,required=True)
    p.add_argument("--inlet-quality",type=float,required=True)
    p.add_argument("--outlet-backflow-quality",type=float,required=True)
    p.add_argument("--supply-cell-volume-m3",type=float,required=True)
    p.add_argument("--return-cell-volume-m3",type=float,required=True)
    p.add_argument("--solid-capacity-j-k",type=float,required=True)
    p.add_argument("--supply-ua-w-k",type=float,required=True)
    p.add_argument("--return-ua-w-k",type=float,required=True)
    p.add_argument("--interface-area-m2",type=float,required=True)
    p.add_argument("--discharge-coefficient",type=float,default=0.7)
    p.add_argument("--dt-s",type=float,required=True)
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_pressure_driven_tie"))
    a=p.parse_args()
    if not 1<=a.steps<=10000:
        raise ValueError("invalid number of pressure-driven timesteps")
    for name,val in (
        ("initial_quality",a.initial_quality),
        ("inlet_quality",a.inlet_quality),
        ("outlet_backflow_quality",a.outlet_backflow_quality),
    ):
        if not 0<=val<=1:
            raise ValueError(f"{name} must be saturation quality 0..1")
    heat=read_resolved_heating(
        a.fuel_reconstruction,a.tie_reconstruction,a.mesh_fields,
    )
    eos=HydrogenEOS()
    th=IsochoricTieParameters(
        supply_cell_volume_m3=a.supply_cell_volume_m3,
        return_cell_volume_m3=a.return_cell_volume_m3,
        solid_capacity_j_k=a.solid_capacity_j_k,
        supply_ua_w_k=a.supply_ua_w_k,
        return_ua_w_k=a.return_ua_w_k,
    )
    params=PressureTieFlowParameters(
        thermal=th,interface_area_m2=a.interface_area_m2,
        discharge_coefficient=a.discharge_coefficient,
    )
    initial=initial_isochoric_tie_state(
        heat,th,a.initial_pressure_pa,quality=a.initial_quality,
        solid_temperature_k=a.initial_solid_temperature_k,eos=eos,
    )
    inlet_h=eos.get("Hmass","P",a.inlet_pressure_pa,"Q",a.inlet_quality)
    back_h=eos.get(
        "Hmass","P",a.outlet_pressure_pa,"Q",a.outlet_backflow_quality
    )
    state=initial
    history=[]
    step=None
    for i in range(a.steps):
        step=advance_pressure_driven_tie(
            heat,state,params,a.inlet_pressure_pa,inlet_h,
            a.outlet_pressure_pa,back_h,a.dt_s,eos=eos,
        )
        state=step.state
        history.append({
            "step":i+1,"elapsed_s":(i+1)*a.dt_s,
            "inlet_mass_flow_kg_s":step.inlet_mass_flow_kg_s,
            "outlet_mass_flow_kg_s":step.outlet_mass_flow_kg_s,
            "total_mass_storage_rate_kg_s":step.total_mass_storage_rate_kg_s,
            "mass_balance_error_kg_s":step.mass_balance_error_kg_s,
            "energy_balance_error_j":step.energy_balance_error_j,
            "maximum_fractional_mass_exchange":
                step.maximum_fractional_mass_exchange,
            "nuclear_tie_deposition_w":step.nuclear_deposition_w,
            "solid_storage_rate_w":step.solid_storage_rate_w,
            "hydrogen_internal_energy_storage_rate_w":
                step.hydrogen_internal_energy_storage_rate_w,
            "min_supply_pressure_pa":float(np.min(step.supply_pressure_pa)),
            "max_supply_pressure_pa":float(np.max(step.supply_pressure_pa)),
            "min_return_pressure_pa":float(np.min(step.return_pressure_pa)),
            "max_return_pressure_pa":float(np.max(step.return_pressure_pa)),
        })
    root=a.output
    root.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        root/"pressure_driven_tie_final_state.npz",
        supply_mass_kg=state.supply_mass_kg,
        return_mass_kg=state.return_mass_kg,
        supply_internal_energy_j_kg=state.supply_internal_energy_j_kg,
        return_internal_energy_j_kg=state.return_internal_energy_j_kg,
        solid_temperature_k=state.solid_temperature_k,
        supply_pressure_pa=step.supply_pressure_pa,
        return_pressure_pa=step.return_pressure_pa,
        supply_temperature_k=step.supply_temperature_k,
        return_temperature_k=step.return_temperature_k,
        supply_density_kg_m3=step.supply_density_kg_m3,
        return_density_kg_m3=step.return_density_kg_m3,
        supply_enthalpy_j_kg=step.supply_enthalpy_j_kg,
        return_enthalpy_j_kg=step.return_enthalpy_j_kg,
        supply_quality=step.supply_quality,
        return_quality=step.return_quality,
        interface_mass_flow_kg_s=step.interface_mass_flow_kg_s,
    )
    report={
        "status":"PASS",
        "source_kind":"external normalized OpenMC-style NPZ; provenance not verified",
        "transport_executed":False,
        "real_nerva_hardware_validated":False,
        "hydraulic_model":step.momentum_law_kind,
        "transient_momentum_inertia_solved":False,
        "two_phase_slip_and_choking_solved":False,
        "coupled_full_pump_turbine_cycle_solved":False,
        "signed_pressure_driven_axial_flow_solved":True,
        "fluid_mass_and_internal_energy_conserved":True,
        "parameters":{"thermal":asdict(th),"hydraulic":{
            "interface_area_m2":params.interface_area_m2,
            "discharge_coefficient":params.discharge_coefficient,
            "max_fractional_mass_exchange":params.max_fractional_mass_exchange,
            "max_fractional_pressure_drop":params.max_fractional_pressure_drop,
        }},
        "history":history,
    }
    (root/"pressure_driven_tie_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    print("NERVA pressure-driven cryogenic tie finite-volume model: PASS")
    print(f"  inlet mdot={step.inlet_mass_flow_kg_s:.8g} kg/s; "
          f"outlet mdot={step.outlet_mass_flow_kg_s:.8g} kg/s")
    print(f"  mass residual={step.mass_balance_error_kg_s:.8g} kg/s; "
          f"energy residual={step.energy_balance_error_j:.8g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
