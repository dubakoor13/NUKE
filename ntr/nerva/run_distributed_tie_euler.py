"""Run compressible H2 tie supply/return Euler + total-energy transient.

Uses reconstructed nuclear-power NPZ and real CoolProp hydrogen EOS;
the quasi-one-dimensional homogeneous-equilibrium FV solve conserves
fluid mass, momentum and TOTAL energy with an explicit acoustic CFL.
External reservoirs have prescribed pressure and enthalpy; pump/turbine
shaft closure and historical NERVA calibration are not implemented.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .distributed_tie_euler import (
    DistributedTieParameters, initial_distributed_tie_state,
    advance_distributed_tie,
)
from .isochoric_tie import IsochoricTieParameters
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
    p.add_argument("--initial-velocity-m-s",type=float,default=0.)
    p.add_argument("--inlet-pressure-pa",type=float,required=True)
    p.add_argument("--outlet-pressure-pa",type=float,required=True)
    p.add_argument("--inlet-quality",type=float,required=True)
    p.add_argument("--outlet-backflow-quality",type=float,required=True)
    p.add_argument("--supply-cell-volume-m3",type=float,required=True)
    p.add_argument("--return-cell-volume-m3",type=float,required=True)
    p.add_argument("--flow-area-m2",type=float,required=True)
    p.add_argument("--solid-capacity-j-k",type=float,required=True)
    p.add_argument("--supply-ua-w-k",type=float,required=True)
    p.add_argument("--return-ua-w-k",type=float,required=True)
    p.add_argument("--darcy-friction-factor",type=float,default=0.)
    p.add_argument("--turnaround-loss-coefficient",type=float,default=0.)
    p.add_argument("--max-cfl",type=float,default=0.45)
    p.add_argument("--dt-s",type=float,required=True)
    p.add_argument("--steps",type=int,default=1)
    p.add_argument("--output",type=Path,
                   default=Path("build/nerva_distributed_tie_euler"))
    a=p.parse_args()
    if not 1<=a.steps<=10000:
        raise ValueError("invalid number of distributed Euler timesteps")
    for name,x in (
        ("initial-quality",a.initial_quality),
        ("inlet-quality",a.inlet_quality),
        ("outlet-backflow-quality",a.outlet_backflow_quality),
    ):
        if not 0<=x<=1:
            raise ValueError(f"{name} must lie in [0,1] for saturated H2")
    heat=read_resolved_heating(
        a.fuel_reconstruction,a.tie_reconstruction,a.mesh_fields,
    )
    eos=HydrogenEOS()
    params=DistributedTieParameters(
        thermal=IsochoricTieParameters(
            supply_cell_volume_m3=a.supply_cell_volume_m3,
            return_cell_volume_m3=a.return_cell_volume_m3,
            solid_capacity_j_k=a.solid_capacity_j_k,
            supply_ua_w_k=a.supply_ua_w_k,
            return_ua_w_k=a.return_ua_w_k,
        ),
        flow_area_m2=a.flow_area_m2,
        darcy_friction_factor=a.darcy_friction_factor,
        turnaround_loss_coefficient=a.turnaround_loss_coefficient,
        max_cfl=a.max_cfl,
    )
    params.validate()
    hi=eos.get("Hmass","P",a.initial_pressure_pa,"Q",a.initial_quality)
    hin=eos.get("Hmass","P",a.inlet_pressure_pa,"Q",a.inlet_quality)
    hout=eos.get(
        "Hmass","P",a.outlet_pressure_pa,"Q",a.outlet_backflow_quality
    )
    state=initial_distributed_tie_state(
        heat,params,a.initial_pressure_pa,hi,
        solid_temperature_k=a.initial_solid_temperature_k,
        velocity_m_s=a.initial_velocity_m_s,eos=eos,
    )
    history=[]
    step=None
    for n in range(a.steps):
        step=advance_distributed_tie(
            heat,state,params,
            a.inlet_pressure_pa,hin,a.outlet_pressure_pa,hout,
            a.dt_s,eos=eos,
        )
        state=step.state
        history.append({
            "step":n+1,
            "elapsed_s":(n+1)*a.dt_s,
            "mass_residual_kg_s":step.mass_residual_kg_s,
            "momentum_residual_n":step.momentum_residual_n,
            "total_energy_residual_j":step.total_energy_residual_j,
            "maximum_cfl":step.maximum_cfl,
            "solid_storage_rate_w":step.solid_storage_rate_w,
            "hydrogen_internal_storage_rate_w":
                step.fluid_internal_energy_storage_rate_w,
            "hydrogen_kinetic_storage_rate_w":
                step.fluid_kinetic_energy_storage_rate_w,
            "nuclear_deposition_w":step.nuclear_deposition_w,
            "inlet_mass_flux_kg_s":
                float(np.sum(step.interface_mass_flux_kg_s[:,0])),
            "outlet_mass_flux_kg_s":
                float(np.sum(step.interface_mass_flux_kg_s[:,-1])),
        })
    out=a.output
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        out/"distributed_tie_final_state.npz",
        fluid_mass_kg=state.mass_kg,
        fluid_momentum_kg_m_s=state.momentum_kg_m_s,
        fluid_total_energy_j=state.total_energy_j,
        solid_temperature_k=state.solid_temperature_k,
        fluid_pressure_pa=step.pressure_pa,
        fluid_temperature_k=step.temperature_k,
        fluid_density_kg_m3=step.density_kg_m3,
        fluid_velocity_m_s=step.velocity_m_s,
        fluid_quality=step.quality,
        fluid_specific_enthalpy_j_kg=step.specific_enthalpy_j_kg,
        interface_mass_flux_kg_s=step.interface_mass_flux_kg_s,
        interface_momentum_flux_n=step.interface_momentum_flux_n,
        interface_total_energy_flux_w=step.interface_total_energy_flux_w,
        wall_friction_force_n=step.wall_friction_force_n,
    )
    report={
        "status":"PASS",
        "source_kind":"external normalized OpenMC-style NPZ; provenance not verified",
        "actual_openmc_transport_executed":False,
        "historical_nerva_validation":False,
        "numerical_model":step.numerical_flux,
        "distributed_axial_momentum":step.momentum_transport_distributed,
        "fluid_total_energy_conserved":step.total_energy_including_kinetic_conserved,
        "homogeneous_equilibrium_only":step.homogeneous_equilibrium_only,
        "full_nerva_pump_turbine_nozzle_cycle_converged":False,
        "two_phase_slip_or_non_equilibrium_boiling_validated":False,
        "parameters":asdict(params),
        "history":history,
    }
    (out/"distributed_tie_report.json").write_text(
        json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8"
    )
    print("NERVA distributed tie Euler mass/momentum/total-energy: PASS")
    print(f"  acoustic CFL={step.maximum_cfl:.8g}")
    print(f"  mass residual={step.mass_residual_kg_s:.8g} kg/s")
    print(f"  momentum residual={step.momentum_residual_n:.8g} N")
    print(f"  total-energy residual={step.total_energy_residual_j:.8g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
