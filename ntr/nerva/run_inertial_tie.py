"""Run finite-inertance cryogenic tie flow with conservative CoolProp CVs.

This model integrates a lumped face momentum equation with quadratic
pressure loss and returns time-dependent H2 mass, internal energy,
temperature, density and pressure. Face kinetic-energy changes, full
distributed momentum, choking and two-phase slip are NOT represented.
External normalized power NPZ inputs do NOT imply real OpenMC transport.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .inertial_tie import (
    InertialTieParameters,InertialTieState,advance_inertial_tie,
)
from .isochoric_tie import (
    IsochoricTieParameters, initial_isochoric_tie_state,
)
from .pressure_driven_tie import PressureTieFlowParameters
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating


def main()->int:
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
    p.add_argument("--face-inertial-length-m",type=float,required=True)
    p.add_argument("--initial-face-mass-flow-kg-s",type=float,default=0.)
    p.add_argument("--dt-s",type=float,required=True)
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_inertial_tie"))
    a=p.parse_args()
    if not 1<=a.steps<=10000:
        raise ValueError("invalid number of inertial tie timesteps")
    for name,value in (
        ("initial_quality",a.initial_quality),
        ("inlet_quality",a.inlet_quality),
        ("outlet_backflow_quality",a.outlet_backflow_quality),
    ):
        if not 0<=value<=1:
            raise ValueError(f"{name} must lie within [0,1]")
    heat=read_resolved_heating(
        a.fuel_reconstruction,a.tie_reconstruction,a.mesh_fields,
    )
    th=IsochoricTieParameters(
        supply_cell_volume_m3=a.supply_cell_volume_m3,
        return_cell_volume_m3=a.return_cell_volume_m3,
        solid_capacity_j_k=a.solid_capacity_j_k,
        supply_ua_w_k=a.supply_ua_w_k,
        return_ua_w_k=a.return_ua_w_k,
    )
    op=PressureTieFlowParameters(
        thermal=th,interface_area_m2=a.interface_area_m2,
        discharge_coefficient=a.discharge_coefficient,
    )
    param=InertialTieParameters(
        thermal_hydraulic=op,
        interface_inertial_length_m=a.face_inertial_length_m,
    )
    param.validate()
    eos=HydrogenEOS()
    initial=initial_isochoric_tie_state(
        heat,th,a.initial_pressure_pa,quality=a.initial_quality,
        solid_temperature_k=a.initial_solid_temperature_k,eos=eos,
    )
    if not np.isfinite(a.initial_face_mass_flow_kg_s):
        raise ValueError("initial face mass flow must be finite")
    edges=np.full(
        (heat.tie_solid_w.shape[0],2*heat.tie_solid_w.shape[1]+1),
        a.initial_face_mass_flow_kg_s,
    )
    state=InertialTieState(initial,edges)
    hin=eos.get("Hmass","P",a.inlet_pressure_pa,"Q",a.inlet_quality)
    hout=eos.get("Hmass","P",a.outlet_pressure_pa,"Q",a.outlet_backflow_quality)
    rows=[]
    latest=None
    for i in range(a.steps):
        latest=advance_inertial_tie(
            heat,state,param,a.inlet_pressure_pa,hin,
            a.outlet_pressure_pa,hout,a.dt_s,eos=eos,
        )
        state=latest.state
        f=latest.fluid
        rows.append({
            "step":i+1,"elapsed_s":(i+1)*a.dt_s,
            "inlet_mass_flow_kg_s":f.inlet_mass_flow_kg_s,
            "outlet_mass_flow_kg_s":f.outlet_mass_flow_kg_s,
            "stored_mass_rate_kg_s":f.total_mass_storage_rate_kg_s,
            "stored_hydrogen_internal_energy_rate_w":
                f.hydrogen_internal_energy_storage_rate_w,
            "stored_solid_energy_rate_w":f.solid_storage_rate_w,
            "nuclear_tie_deposition_w":f.nuclear_deposition_w,
            "mass_residual_kg_s":f.mass_balance_error_kg_s,
            "energy_residual_j":f.energy_balance_error_j,
            "momentum_pressure_residual_pa":latest.maximum_momentum_residual_pa,
            "max_fractional_mass_exchange":f.maximum_fractional_mass_exchange,
            "max_pressure_jump_fraction":latest.maximum_fractional_pressure_jump,
        })
    root=a.output
    root.mkdir(parents=True,exist_ok=True)
    f=latest.fluid
    np.savez_compressed(
        root/"inertial_tie_final_state.npz",
        supply_mass_kg=state.thermal.supply_mass_kg,
        return_mass_kg=state.thermal.return_mass_kg,
        supply_internal_energy_j_kg=state.thermal.supply_internal_energy_j_kg,
        return_internal_energy_j_kg=state.thermal.return_internal_energy_j_kg,
        solid_temperature_k=state.thermal.solid_temperature_k,
        interface_mass_flow_kg_s=state.interface_mass_flow_kg_s,
        face_driving_pressure_pa=latest.driving_pressure_pa,
        face_drag_pressure_pa=latest.drag_pressure_pa,
        face_acceleration_pressure_pa=latest.acceleration_pressure_pa,
        supply_pressure_pa=f.supply_pressure_pa,
        return_pressure_pa=f.return_pressure_pa,
        supply_temperature_k=f.supply_temperature_k,
        return_temperature_k=f.return_temperature_k,
        supply_quality=f.supply_quality,
        return_quality=f.return_quality,
        supply_density_kg_m3=f.supply_density_kg_m3,
        return_density_kg_m3=f.return_density_kg_m3,
    )
    report={
        "status":"PASS",
        "transport_source_kind":"external OpenMC-style NPZ; unverified provenance",
        "real_openmc_transport_executed":False,
        "historical_nerva_calibration":False,
        "momentum_model":latest.momentum_law,
        "full_unsteady_momentum_euler_solved":False,
        "kinetic_energy_storage_included":False,
        "phase_slip_critical_flow_choking_included":False,
        "full_pump_turbine_cycle_solved":False,
        "mass_internal_energy_and_lumped_face_inertia_solved":True,
        "parameter_set":{
            "thermal":asdict(th),
            "orifice":asdict(op),
            "face_inertial_length_m":param.interface_inertial_length_m,
        },
        "history":rows,
    }
    (root/"inertial_tie_report.json").write_text(
        json.dumps(report,allow_nan=False,indent=2)+"\n",
        encoding="utf-8",
    )
    print("NERVA finite-inertance cryogenic tie CV: PASS")
    print(f"  inlet flow={f.inlet_mass_flow_kg_s:.9g} kg/s")
    print(f"  mass residual={f.mass_balance_error_kg_s:.9g} kg/s; "
          f"energy residual={f.energy_balance_error_j:.9g} J")
    print(f"  momentum residual={latest.maximum_momentum_residual_pa:.9g} Pa")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
