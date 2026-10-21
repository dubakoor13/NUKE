"""Finite-volume common-header <-> distributed tie Euler feedback tests.

The independently solved pump/turbine/regen gives a *reference* nozzle
branch and tank flow demand; the header updates its actual mass,
internal energy, pressure and enthalpy with the distributed transient
tie return using the EXACT shared Euler Rusanov flux. Synthetic power
and uncalibrated hardware parameters, not historical NASA validation.
"""
from __future__ import annotations

from dataclasses import asdict,replace
import json
import math
import sys
from pathlib import Path
import tempfile

import numpy as np

from .distributed_tie_euler import (
    DistributedTieParameters,initial_distributed_tie_state,
    advance_distributed_tie,
)
from .dynamic_header_tie import (
    DynamicHeaderParameters,DynamicHeaderState,DynamicHeaderTieState,
    initial_dynamic_header,header_properties,advance_dynamic_header_tie,
)
from .distributed_tie_engine_boundary import engine_tie_reservoirs
from .isochoric_tie import IsochoricTieParameters
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating
from .turbine_bleed_cycle import solve_turbine_bleed
from .validate_hot_lh2_cycle import hot_case
from .validate_regenerative_cooling import network
from .validate_turbine_bleed_cycle import sample_turbine
from .validate_resolved_transient import write_synthetic_reconstructions
from .run_dynamic_header_tie import main as run_dynamic_header_cli


def rejects(call):
    try:
        call()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("invalid dynamic header/tie input accepted")


def audits(result,old,dt):
    step=result.tie_step
    assert result.tie_header_pressure_feedback_active
    assert not result.dynamic_core_flow_computed
    assert not result.full_shaft_cycle_feedback_converged
    assert not result.actual_openmc_transport_executed
    assert abs(result.header_mass_residual_kg_s)<1e-8
    assert abs(result.header_internal_energy_residual_j)<1e-5
    assert abs(result.combined_tie_header_mass_residual_kg_s)<1e-8
    assert abs(result.combined_tie_solid_header_energy_residual_j)<1e-5
    assert abs(result.combined_tie_header_momentum_residual_n)<1e-5
    assert abs(step.mass_residual_kg_s)<1e-8
    assert abs(step.total_energy_residual_j)<1e-5
    assert result.header_new_pressure_pa>0
    assert result.header_new_temperature_k>0
    assert step.maximum_cfl<0.45
    assert math.isclose(
        result.header_inventory_rate_kg_s,
        result.nozzle_branch_inlet_flow_kg_s+
        result.tie_header_interface_flow_kg_s-
        result.fixed_engine_discharge_flow_kg_s,
        abs_tol=1e-8,
    )
    assert math.isclose(
        result.header_internal_energy_rate_w,
        result.nozzle_branch_inlet_flow_kg_s*
        result.nozzle_branch_inlet_enthalpy_j_kg+
        result.tie_header_interface_total_energy_power_w-
        result.fixed_engine_discharge_flow_kg_s*
        result.header_old_enthalpy_j_kg,
        rel_tol=1e-8,abs_tol=0.02,
    )
    assert math.isclose(
        result.tie_header_interface_flow_kg_s,
        float(np.sum(step.interface_mass_flux_kg_s[:,-1])),
        rel_tol=1e-12,abs_tol=1e-10,
    )
    assert math.isclose(
        result.tie_header_interface_total_energy_power_w,
        float(np.sum(step.interface_total_energy_flux_w[:,-1])),
        rel_tol=1e-12,abs_tol=1e-8,
    )
    # Header and tie share one boundary: sum their mass and energy
    # inventories and eliminate the internal tie->header flux.
    old_mass=float(np.sum(old.tie.mass_kg))+old.header.mass_kg
    new_mass=float(np.sum(result.state.tie.mass_kg))+result.state.header.mass_kg
    expected_mass=dt*(
        result.tie_inlet_flow_kg_s+
        result.nozzle_branch_inlet_flow_kg_s-
        result.fixed_engine_discharge_flow_kg_s
    )
    assert math.isclose(new_mass-old_mass,expected_mass,abs_tol=1e-10)
    old_energy=(
        float(np.sum(old.tie.total_energy_j))+
        old.header.internal_energy_j
    )
    new_energy=(
        float(np.sum(result.state.tie.total_energy_j))+
        result.state.header.internal_energy_j
    )
    solid=3000.*float(np.sum(
        result.state.tie.solid_temperature_k-
        old.tie.solid_temperature_k
    ))
    external=dt*(
        float(np.sum(step.interface_total_energy_flux_w[:,0]))+
        step.nuclear_deposition_w+
        result.nozzle_branch_inlet_flow_kg_s*
        result.nozzle_branch_inlet_enthalpy_j_kg-
        result.fixed_engine_discharge_flow_kg_s*
        result.header_old_enthalpy_j_kg
    )
    assert abs((new_energy-old_energy)+solid-external)<1e-5


def main()->int:
    eos=HydrogenEOS()
    shaft=solve_turbine_bleed(
        hot_case(),sample_turbine(),cooling_network=network(),eos=eos,
    )
    bc=engine_tie_reservoirs(shaft)
    hdr=DynamicHeaderParameters(volume_m3=0.0001)
    hdr_old=initial_dynamic_header(shaft,hdr,eos=eos)
    p,h,t,rho=header_properties(hdr_old,hdr,eos=eos)
    assert math.isclose(p,bc.steady_common_header_pressure_pa,rel_tol=1e-7)
    assert math.isclose(
        h,bc.steady_common_header_stagnation_enthalpy_j_kg,rel_tol=1e-7
    )
    assert math.isclose(rho*hdr.volume_m3,hdr_old.mass_kg,rel_tol=1e-12)

    with tempfile.TemporaryDirectory() as td:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(td))
        heating=read_resolved_heating(fuel,tie,mesh)
        thermal=IsochoricTieParameters(
            supply_cell_volume_m3=0.0001,
            return_cell_volume_m3=0.0001,
            solid_capacity_j_k=3000.,
            supply_ua_w_k=0.4,
            return_ua_w_k=0.4,
        )
        params=DistributedTieParameters(
            thermal=thermal,flow_area_m2=0.0002,
            darcy_friction_factor=0.04,
            turnaround_loss_coefficient=0.6,
        )
        tie_old=initial_distributed_tie_state(
            heating,params,bc.upstream_pressure_pa,
            bc.upstream_stagnation_enthalpy_j_kg,
            solid_temperature_k=35.,eos=eos,
        )
        state=DynamicHeaderTieState(tie=tie_old,header=hdr_old)
        dt=1e-7
        result=advance_dynamic_header_tie(
            heating,state,params,hdr,shaft,dt,eos=eos
        )
        audits(result,state,dt)
        assert result.header_new_pressure_pa!=result.header_old_pressure_pa
        assert result.header_new_enthalpy_j_kg!=result.header_old_enthalpy_j_kg
        assert abs(result.tie_header_interface_flow_kg_s)<2.
        assert math.isclose(
            result.nozzle_branch_inlet_flow_kg_s+
            bc.steady_tie_branch_mass_flow_kg_s,
            result.fixed_engine_discharge_flow_kg_s,rel_tol=1e-11
        )

        second=advance_dynamic_header_tie(
            heating,result.state,params,hdr,shaft,dt,eos=eos
        )
        audits(second,result.state,dt)
        assert math.isclose(
            second.header_old_pressure_pa,result.header_new_pressure_pa,
            rel_tol=1e-12,
        )
        assert math.isclose(
            second.header_old_enthalpy_j_kg,result.header_new_enthalpy_j_kg,
            rel_tol=1e-12,
        )
        # Feedback is real: supplying the ORIGINAL fixed engine header
        # to the same second-step Euler state must change the face flux.
        fixed=advance_distributed_tie(
            heating,result.state.tie,params,
            bc.upstream_pressure_pa,
            bc.upstream_stagnation_enthalpy_j_kg,
            result.header_old_pressure_pa,
            result.header_old_enthalpy_j_kg,
            dt,eos=eos,
        )
        assert not np.allclose(
            second.tie_step.interface_mass_flux_kg_s[:,-1],
            fixed.interface_mass_flux_kg_s[:,-1],
            atol=1e-12,rtol=1e-8,
        )
        assert abs(second.combined_tie_solid_header_energy_residual_j)<1e-5

        # Header volume affects storage / EOS pressure response:
        # no arbitrary assumption of fixed header temperature.
        large=replace(hdr,volume_m3=hdr.volume_m3*20)
        large_init=initial_dynamic_header(shaft,large,eos=eos)
        larger=advance_dynamic_header_tie(
            heating,DynamicHeaderTieState(tie_old,large_init),
            params,large,shaft,dt,eos=eos,
        )
        audits(larger,DynamicHeaderTieState(tie_old,large_init),dt)
        assert (
            abs(larger.header_new_pressure_pa-larger.header_old_pressure_pa)<
            abs(result.header_new_pressure_pa-result.header_old_pressure_pa)
        )

        # Public CLI replays shaft, EOS and dynamic header feedback,
        # saving all conserved variables and source-provenance flags.
        config=Path(td)/"engine.json"
        config.write_text(json.dumps({
            "lh2_cycle":asdict(hot_case()),
            "turbine":asdict(sample_turbine()),
            "regenerative_cooling":asdict(network()),
        }),encoding="utf-8")
        out=Path(td)/"report"
        prev_args=sys.argv
        try:
            sys.argv=[
                "run_dynamic_header_tie",str(mesh),str(fuel),str(tie),
                "--engine-config",str(config),
                "--header-volume-m3","0.0001",
                "--initial-solid-temperature-k","35",
                "--supply-cell-volume-m3","0.0001",
                "--return-cell-volume-m3","0.0001",
                "--flow-area-m2","0.0002",
                "--solid-capacity-j-k","3000",
                "--supply-ua-w-k","0.4",
                "--return-ua-w-k","0.4",
                "--darcy-friction-factor","0.04",
                "--turnaround-loss-coefficient","0.6",
                "--dt-s",str(dt),"--steps","2",
                "--output",str(out),
            ]
            assert run_dynamic_header_cli()==0
        finally:
            sys.argv=prev_args
        report=json.loads(
            (out/"dynamic_tie_header_report.json").read_text(
                encoding="utf-8"
            )
        )
        assert report["status"]=="PASS"
        assert report["tie_header_pressure_enthalpy_feedback_active"]
        assert not report["full_shaft_cycle_feedback_converged"]
        assert not report["dynamic_pump_turbine_nozzle_flows_solved"]
        assert not report["actual_openmc_transport_executed"]
        assert len(report["history"])==2
        assert all(
            abs(r["combined_mass_residual_kg_s"])<1e-8
            and abs(r["combined_total_energy_residual_j"])<1e-5
            and abs(r["tie_momentum_residual_n"])<1e-5
            for r in report["history"]
        )
        assert (
            report["history"][0]["header_pressure_pa"]!=
            report["history"][1]["header_pressure_pa"]
        )
        with np.load(
            out/"dynamic_tie_header_final_state.npz",allow_pickle=False,
        ) as data:
            assert data["tie_mass_kg"].shape==state.tie.mass_kg.shape
            assert data["tie_interface_mass_flux_kg_s"].shape==(
                heating.tie_solid_w.shape[0],
                2*heating.tie_solid_w.shape[1]+1,
            )
            assert float(data["header_mass_kg"])>0
            assert float(data["header_pressure_pa"])>0

        rejects(lambda:advance_dynamic_header_tie(
            heating,state,params,hdr,shaft,0.,eos=eos
        ))
        rejects(lambda:advance_dynamic_header_tie(
            heating,state,params,replace(hdr,volume_m3=0.),
            shaft,dt,eos=eos
        ))
        rejects(lambda:header_properties(
            DynamicHeaderState(0.,0.),hdr,eos=eos
        ))
        rejects(lambda:advance_dynamic_header_tie(
            heating,state,params,hdr,replace(
                shaft,regenerative_cooling=None
            ),dt,eos=eos
        ))
        rejects(lambda:advance_dynamic_header_tie(
            heating,state,params,hdr,shaft,0.1,eos=eos
        ))

    print("NERVA tie Euler <-> dynamic common-header H2 feedback: PASS")
    print(f"  steady reference tie flow: {bc.steady_tie_branch_mass_flow_kg_s:.9g} kg/s")
    print(f"  transient tie-to-header flow: {result.tie_header_interface_flow_kg_s:.9g} kg/s")
    print(f"  header initial pressure: {result.header_old_pressure_pa:.9g} Pa")
    print(f"  header new pressure: {result.header_new_pressure_pa:.9g} Pa")
    print(f"  combined mass residual: {result.combined_tie_header_mass_residual_kg_s:.9g} kg/s")
    print(f"  combined energy residual: {result.combined_tie_solid_header_energy_residual_j:.9g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
