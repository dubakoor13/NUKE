"""Verify shaft-driven warm-H2 turbine bleed, mass and first-law closure.

Synthetic fixed pressure-source curve and deposited reactor power, not
historical NERVA turbine maps or real transport. Compare with previous
externally powered pump operating point to ensure actual coupling.
"""
from __future__ import annotations

from dataclasses import asdict,replace
import json
import math
from pathlib import Path
import sys

from .lh2_feed_cycle import CryogenicCycleError, HydrogenEOS
from .hot_lh2_cycle import solve_hot_core_cycle
from .nasa_hydrogen import G0, hydrogen_state
from .turbine_bleed_cycle import (
    TurbineBleed, evaluate_turbine_bleed, solve_turbine_bleed
)
from .validate_hot_lh2_cycle import hot_case


def sample_turbine() -> TurbineBleed:
    return TurbineBleed(
        exhaust_pressure_pa=0.75e6,
        isentropic_efficiency=0.8,
        shaft_mechanical_efficiency=0.9,
        maximum_bleed_fraction=0.5,
    )


def rejects(action):
    try:
        action()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("unphysical operating point accepted")


def check(point,core,turbine,eos):
    mt=point.tank_mass_flow_kg_s
    mc=point.core_mass_flow_kg_s
    mb=point.turbine_bleed_mass_flow_kg_s
    assert mt>mc>0 and mb>0
    assert math.isclose(mc+mb,mt,rel_tol=1e-12,abs_tol=1e-12)
    assert math.isclose(mb/mt,point.turbine_bleed_fraction,rel_tol=1e-12)
    assert point.turbine_bleed_fraction<=turbine.maximum_bleed_fraction
    assert point.pump_discharge_pressure_pa>point.core_inlet_pressure_pa
    assert point.core_inlet_pressure_pa>point.chamber_pressure_pa
    assert point.chamber_temperature_k>point.warm_header_temperature_k
    assert point.maximum_core_mach<core.allowable_mach
    assert 120<point.turbine_exhaust_temperature_k<core.maximum_eos_temperature_k
    assert point.turbine_shaft_delivered_power_w>0
    assert point.pump_required_power_w>0
    assert point.mechanical_loss_w>=0
    assert math.isclose(
        point.turbine_gas_power_w,
        mb*point.turbine_actual_specific_work_j_kg,
        rel_tol=1e-12
    )
    assert math.isclose(
        point.turbine_shaft_delivered_power_w,
        point.turbine_gas_power_w*turbine.shaft_mechanical_efficiency,
        rel_tol=1e-12
    )
    assert math.isclose(
        point.turbine_actual_specific_work_j_kg/
        point.turbine_isentropic_specific_work_j_kg,
        turbine.isentropic_efficiency,rel_tol=1e-12
    )
    assert abs(point.shaft_power_residual_w)<1e-8*point.pump_required_power_w
    assert abs(point.nozzle_mass_flow_residual_kg_s)<2e-6*mc
    assert math.isclose(point.nozzle.mass_flow_kg_s,mc,rel_tol=2e-6)
    assert abs(point.channel_pressure_residual_pa)<0.1
    assert abs(point.total_cycle_energy_residual_w)<1e-7*(
        point.preheater_power_w+point.core_heating_power_w
    )
    assert point.main_nozzle_thrust_n>0
    assert point.main_nozzle_isp_s>point.effective_isp_excluding_bleed_thrust_s
    assert math.isclose(point.effective_isp_excluding_bleed_thrust_s,
                        point.main_nozzle_thrust_n/(G0*mt),rel_tol=1e-12)
    assert math.isclose(point.main_nozzle_isp_s,
                        point.main_nozzle_thrust_n/(G0*mc),rel_tol=1e-10)

    # Independent system-boundary energy verification using CoolProp
    # absolute enthalpies and the shifted NASA7 hot-core enthalpy.
    htank=eos.tp("Hmass",core.tank_temperature_k,core.tank_pressure_pa)
    h3=eos.tp("Hmass",point.warm_header_temperature_k,
              point.core_inlet_pressure_pa)
    hot_header=hydrogen_state(point.warm_header_temperature_k,
                              point.core_inlet_pressure_pa,
                              mode=point.chemistry_mode)
    hot_chamber=hydrogen_state(point.chamber_temperature_k,
                               point.chamber_pressure_pa,
                               mode=point.chemistry_mode)
    hc=h3+hot_chamber.enthalpy_j_kg-hot_header.enthalpy_j_kg
    budget=(
        mc*hc+mb*point.turbine_exhaust_enthalpy_j_kg+point.mechanical_loss_w
        -(mt*htank+point.preheater_power_w+point.core_heating_power_w)
    )
    assert abs(budget)<1e-7*(point.preheater_power_w+point.core_heating_power_w)
    assert abs(budget-point.total_cycle_energy_residual_w)<0.05


def main() -> int:
    eos=HydrogenEOS()
    core=hot_case()
    turbine=sample_turbine()
    old=solve_hot_core_cycle(core,eos=eos,chemistry_mode="frozen")
    point=solve_turbine_bleed(core,turbine,eos=eos,chemistry_mode="frozen")
    check(point,core,turbine,eos)

    assert point.core_mass_flow_kg_s < point.tank_mass_flow_kg_s
    assert point.core_mass_flow_kg_s != old.mass_flow_kg_s
    assert point.effective_isp_excluding_bleed_thrust_s < point.main_nozzle_isp_s

    none,mass_error=evaluate_turbine_bleed(
        core,turbine,point.tank_mass_flow_kg_s,eos=eos
    )
    assert none is None
    assert abs(mass_error)<2e-6*point.core_mass_flow_kg_s

    # Improve shaft efficiency: same pump requirement at fixed total
    # mass consumes a smaller fraction in turbine bleed.
    good=replace(turbine,shaft_mechanical_efficiency=0.99)
    candidate, _ = evaluate_turbine_bleed(
        core,good,point.tank_mass_flow_kg_s,eos=eos,full_nozzle=True
    )
    check(candidate,core,good,eos)
    assert candidate.turbine_bleed_fraction < point.turbine_bleed_fraction

    # More turbine pressure expansion extracts more specific work and
    # therefore reduces bleed at the SAME proposed tank mass flow.
    expanded=replace(turbine,exhaust_pressure_pa=0.45e6)
    deeper,_=evaluate_turbine_bleed(
        core,expanded,point.tank_mass_flow_kg_s,eos=eos,full_nozzle=True
    )
    check(deeper,core,expanded,eos)
    assert deeper.turbine_bleed_fraction < point.turbine_bleed_fraction

    # The chemistry option stays consistent through hot channels/nozzle.
    eq=solve_turbine_bleed(core,turbine,eos=eos,chemistry_mode="equilibrium")
    check(eq,core,turbine,eos)
    assert eq.chemistry_mode=="equilibrium"
    assert eq.chamber_dissociation_fraction>0
    assert abs(eq.chamber_temperature_k-point.chamber_temperature_k)>1

    rejects(lambda:solve_turbine_bleed(
        core,replace(turbine,exhaust_pressure_pa=20e6),eos=eos
    ))
    rejects(lambda:solve_turbine_bleed(
        core,replace(turbine,isentropic_efficiency=0.0),eos=eos
    ))
    rejects(lambda:solve_turbine_bleed(
        core,replace(turbine,maximum_bleed_fraction=0.0001),eos=eos
    ))
    rejects(lambda:solve_turbine_bleed(
        replace(core,preheater_thermal_power_w=1.),
        turbine,eos=eos
    ))
    rejects(lambda:solve_turbine_bleed(
        core,turbine,eos=eos,chemistry_mode="invalid"
    ))

    if "--write-example" in sys.argv:
        i=sys.argv.index("--write-example")
        if len(sys.argv)<=i+1:
            raise ValueError("--write-example requires filename")
        path=Path(sys.argv[i+1])
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps({
            "lh2_cycle":asdict(core),
            "turbine":asdict(turbine),
        },indent=2)+"\n",encoding="utf-8")
        from .run_turbine_bleed_cycle import main as runner
        old_argv=sys.argv
        sys.argv=["run_turbine_bleed_cycle","--config",str(path),
                  "--output",str(path.with_name("turbine_bleed_result.json"))]
        try:
            assert runner()==0
        finally:
            sys.argv=old_argv

    print("NERVA H2 turbine bleed drives LH2 pump: PASS")
    print(f"  total tank flow {point.tank_mass_flow_kg_s:.8g} kg/s")
    print(f"  core flow {point.core_mass_flow_kg_s:.8g} kg/s")
    print(f"  turbine bleed fraction {point.turbine_bleed_fraction:.7g}")
    print(f"  pump power {point.pump_required_power_w:.8g} W")
    print(f"  turbine delivered {point.turbine_shaft_delivered_power_w:.8g} W")
    print(f"  chamber {point.chamber_pressure_pa/1e6:.8g} MPa "
          f"at {point.chamber_temperature_k:.8g} K")
    print(f"  main nozzle Isp {point.main_nozzle_isp_s:.7g} s, "
          f"tank-flow Isp {point.effective_isp_excluding_bleed_thrust_s:.7g} s")
    print(f"  total-cycle energy residual {point.total_cycle_energy_residual_w:.7g} W")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
