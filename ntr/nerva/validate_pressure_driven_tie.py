"""Pressure-driven CoolProp tie finite-volume mass/energy test.

Uses synthetic OpenMC-shape heat arrays and a subcritical two-phase
hydrogen EOS. Verifies conservative interface fluxes, separate
supply/return cell mass and internal energy, EOS pressures, flow
reversal, zero-flow equilibrium, and strict explicit CFL rejection.
No independently validated NERVA momentum coefficients or OpenMC run.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
import sys
from pathlib import Path
import tempfile

import numpy as np

from .isochoric_tie import (
    IsochoricTieParameters, initial_isochoric_tie_state,
)
from .lh2_feed_cycle import HydrogenEOS
from .pressure_driven_tie import (
    PressureTieFlowParameters, advance_pressure_driven_tie,
)
from .run_pressure_driven_tie import main as run_pressure_tie_cli
from .resolved_transient import read_resolved_heating
from .validate_resolved_transient import write_synthetic_reconstructions


def rejects(f):
    try:
        f()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("physically unsupported flow state accepted")


def main() -> int:
    eos=HydrogenEOS()
    p0=550000.
    th=IsochoricTieParameters(
        supply_cell_volume_m3=0.001,
        return_cell_volume_m3=0.001,
        solid_capacity_j_k=3000.,
        supply_ua_w_k=1.,
        return_ua_w_k=1.,
    )
    params=PressureTieFlowParameters(
        thermal=th,interface_area_m2=1e-5,
        discharge_coefficient=0.7,
    )
    h_in=eos.get("Hmass","P",600000.,"Q",0.30)
    h_back=eos.get("Hmass","P",500000.,"Q",0.30)
    with tempfile.TemporaryDirectory() as temp:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(temp))
        heat=read_resolved_heating(fuel,tie,mesh)
        state=initial_isochoric_tie_state(
            heat,th,p0,quality=0.30,solid_temperature_k=50.,eos=eos,
        )
        initial_mass=float(np.sum(state.supply_mass_kg)+
                           np.sum(state.return_mass_kg))
        dt=0.001
        step=advance_pressure_driven_tie(
            heat,state,params,600000.,h_in,500000.,h_back,dt,eos=eos,
        )
        assert step.interface_mass_flow_kg_s.shape==(heat.tie_solid_w.shape[0],
                                                       2*heat.tie_solid_w.shape[1]+1)
        assert np.all(step.interface_mass_flow_kg_s[:,0]>0)
        assert np.all(step.interface_mass_flow_kg_s[:,-1]>0)
        assert np.allclose(step.interface_mass_flow_kg_s[:,1:-1],0.)
        assert step.maximum_fractional_mass_exchange<0.20
        assert step.max_local_mass_balance_error_kg_s<1e-9
        assert step.max_local_energy_balance_error_w<1e-3
        assert abs(step.mass_balance_error_kg_s)<1e-9
        assert abs(step.energy_balance_error_j)<1e-5
        assert not np.array_equal(step.state.supply_mass_kg,state.supply_mass_kg)
        assert not np.array_equal(step.state.return_mass_kg,state.return_mass_kg)
        assert np.all(step.supply_density_kg_m3>0)
        assert np.all(step.return_density_kg_m3>0)
        assert math.isclose(
            float(np.sum(step.state.supply_mass_kg)+np.sum(step.state.return_mass_kg)),
            initial_mass+dt*(step.inlet_mass_flow_kg_s-
                             step.outlet_mass_flow_kg_s),
            rel_tol=1e-12,abs_tol=1e-10,
        )
        assert math.isclose(
            step.solid_storage_rate_w+
            step.hydrogen_internal_energy_storage_rate_w,
            step.nuclear_deposition_w+
            step.inlet_enthalpy_power_w-
            step.outlet_enthalpy_power_w,
            rel_tol=1e-9,abs_tol=1e-4,
        )
        assert np.allclose(
            step.state.supply_mass_kg,
            step.supply_density_kg_m3*th.supply_cell_volume_m3,
            rtol=1e-12,atol=1e-12,
        )
        assert np.allclose(
            step.supply_enthalpy_j_kg,
            step.state.supply_internal_energy_j_kg+
            step.supply_pressure_pa/step.supply_density_kg_m3,
            rtol=1e-12,atol=1e-6,
        )
        assert np.all(np.isfinite(step.supply_quality))
        assert np.all(np.isfinite(step.return_quality))

        # The second step propagates the pressure disturbance into
        # the interior and through the single physical turnaround
        # interface, without duplicating either hydrogen mass or heat.
        next_step=advance_pressure_driven_tie(
            heat,step.state,params,600000.,h_in,500000.,h_back,dt,eos=eos,
        )
        assert np.any(abs(next_step.interface_mass_flow_kg_s[:,1:-1])>0)
        assert abs(next_step.mass_balance_error_kg_s)<1e-9
        assert abs(next_step.energy_balance_error_j)<1e-5

        # Zero heating and equal reservoir pressure must be static:
        # no fake transfer or EOS pressure drift.
        tied_power=float(np.sum(heat.tie_solid_w)+
                         np.sum(heat.tie_supply_h2_w)+
                         np.sum(heat.tie_return_h2_w))
        zero=replace(
            heat,
            tie_solid_w=np.zeros_like(heat.tie_solid_w),
            tie_supply_h2_w=np.zeros_like(heat.tie_supply_h2_w),
            tie_return_h2_w=np.zeros_like(heat.tie_return_h2_w),
            total_openmc_power_w=heat.total_openmc_power_w-tied_power,
        )
        zero.validate()
        insulated=replace(
            params,thermal=replace(th,supply_ua_w_k=0.,
                                   return_ua_w_k=0.)
        )
        h0=eos.get("Hmass","P",p0,"Q",0.30)
        static=advance_pressure_driven_tie(
            zero,state,insulated,p0,h0,p0,h0,dt,eos=eos,
        )
        assert np.array_equal(static.state.supply_mass_kg,state.supply_mass_kg)
        assert np.array_equal(static.state.return_mass_kg,state.return_mass_kg)
        assert np.allclose(static.supply_pressure_pa,p0,rtol=1e-10)
        assert np.allclose(static.return_pressure_pa,p0,rtol=1e-10)
        assert np.all(static.interface_mass_flow_kg_s==0.)
        assert abs(static.energy_balance_error_j)<1e-5

        # Reversed pressure gradient must reverse boundary flows and
        # use upwind enthalpy from cells/backflow reservoir.
        h_reverse_in=eos.get("Hmass","P",500000.,"Q",0.30)
        h_reverse_out=eos.get("Hmass","P",600000.,"Q",0.30)
        reverse=advance_pressure_driven_tie(
            heat,state,params,500000.,h_reverse_in,
            600000.,h_reverse_out,dt,eos=eos,
        )
        assert np.all(reverse.interface_mass_flow_kg_s[:,0]<0)
        assert np.all(reverse.interface_mass_flow_kg_s[:,-1]<0)
        assert abs(reverse.mass_balance_error_kg_s)<1e-9
        assert abs(reverse.energy_balance_error_j)<1e-5

        # Reproduce the pressure-driven calculation through the
        # public CLI and verify persisted fluxes, state and diagnostics.
        output=Path(temp)/"pressure_driven_analysis"
        original_argv=sys.argv
        try:
            sys.argv=[
                "run_pressure_driven_tie",
                str(mesh),str(fuel),str(tie),
                "--initial-pressure-pa",str(p0),
                "--initial-quality","0.30",
                "--initial-solid-temperature-k","50",
                "--inlet-pressure-pa","600000",
                "--outlet-pressure-pa","500000",
                "--inlet-quality","0.30",
                "--outlet-backflow-quality","0.30",
                "--supply-cell-volume-m3","0.001",
                "--return-cell-volume-m3","0.001",
                "--solid-capacity-j-k","3000",
                "--supply-ua-w-k","1",
                "--return-ua-w-k","1",
                "--interface-area-m2","1e-5",
                "--discharge-coefficient","0.7",
                "--dt-s","0.001","--steps","2",
                "--output",str(output),
            ]
            assert run_pressure_tie_cli()==0
        finally:
            sys.argv=original_argv
        report=json.loads(
            (output/"pressure_driven_tie_report.json").read_text(
                encoding="utf-8"
            )
        )
        assert report["status"]=="PASS"
        assert report["signed_pressure_driven_axial_flow_solved"]
        assert not report["transient_momentum_inertia_solved"]
        assert not report["transport_executed"]
        assert len(report["history"])==2
        assert all(abs(row["mass_balance_error_kg_s"])<1e-9
                   and abs(row["energy_balance_error_j"])<1e-5
                   for row in report["history"])
        with np.load(
            output/"pressure_driven_tie_final_state.npz",
            allow_pickle=False,
        ) as data:
            assert data["interface_mass_flow_kg_s"].shape==(
                heat.tie_solid_w.shape[0],2*heat.tie_solid_w.shape[1]+1,
            )
            assert np.all(data["supply_mass_kg"]>0)
            assert np.all(data["return_density_kg_m3"]>0)

        # Reject unresolvable steps and high-drop/choked flows. Never
        # silently clamp masses or use two-phase orifice closure at a
        # clearly inadmissible pressure ratio.
        rejects(lambda:advance_pressure_driven_tie(
            heat,state,params,600000.,h_in,500000.,h_back,0.6,eos=eos,
        ))
        rejects(lambda:advance_pressure_driven_tie(
            heat,state,params,2e6,h_in,500000.,h_back,dt,eos=eos,
        ))
        rejects(lambda:advance_pressure_driven_tie(
            heat,state,replace(params,interface_area_m2=-1.),
            600000.,h_in,500000.,h_back,dt,eos=eos,
        ))
        rejects(lambda:advance_pressure_driven_tie(
            heat,state,params,600000.,h_in,500000.,h_back,-dt,eos=eos,
        ))

    print("NERVA pressure-driven cryogenic tie mass/internal-energy CV: PASS")
    print(f"  inlet flow: {step.inlet_mass_flow_kg_s:.9g} kg/s")
    print(f"  outlet flow: {step.outlet_mass_flow_kg_s:.9g} kg/s")
    print(f"  largest explicit mass exchange: {step.maximum_fractional_mass_exchange:.9g}")
    print(f"  global mass residual: {step.mass_balance_error_kg_s:.9g} kg/s")
    print(f"  global energy residual: {step.energy_balance_error_j:.9g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
