"""Validate cryogenic LH2 pump and real-fluid warm-core physics.

All scenarios use hypothetical input data, NOT a historical NERVA pump
characteristic or actual OpenMC heating distribution. Unlike earlier
gas-phase checks, this test requires CoolProp and starts at 20 K
compressed liquid H2, with full enthalpy change through vaporization.
"""
from __future__ import annotations

from dataclasses import asdict,replace
import json
import math
from pathlib import Path
import sys

from .lh2_feed_cycle import (
    CryogenicCycleError, HydrogenEOS, LH2Cycle,
    evaluate_lh2_cycle, solve_lh2_cycle,
)


def example_case() -> LH2Cycle:
    return LH2Cycle(
        tank_temperature_k=20.,
        tank_pressure_pa=0.27e6,
        pump_shutoff_pressure_rise_pa=6.0e6,
        pump_head_quadratic_pa_s2_kg2=8e6,
        pump_isentropic_efficiency=0.68,
        feed_loss_quadratic_pa_s2_kg2=6.e6,
        chamber_loss_quadratic_pa_s2_kg2=5.e6,
        preheater_thermal_power_w=350000.,
        channel_wall_power_w=(180000.,170000.,130000.),
        channel_direct_hydrogen_power_w=(5000.,10000.,5000.),
        z_edges_m=(0.,0.1,0.2,0.3),
        number_of_channels=60,
        channel_diameter_m=0.008,
        throat_area_m2=4.0e-5,
        nozzle_expansion_ratio=8.,
        maximum_mass_flow_kg_s=0.25,
        allowable_mach=0.8,
    )


def expect_rejected(fn,error=(ValueError,RuntimeError)):
    try:
        fn()
    except error:
        return
    raise AssertionError("expected failed CoolProp/physics case")


def check_conservation(config: LH2Cycle, result, eos: HydrogenEOS) -> None:
    m=result.mass_flow_kg_s
    assert m>0.
    assert abs(result.mass_residual_kg_s)<2e-6*m
    assert result.nozzle.mass_flow_kg_s>0
    assert abs(result.nozzle.mass_flow_kg_s-m)<2e-6*m
    assert result.pump_discharge_pressure_pa>config.tank_pressure_pa
    assert result.core_inlet_pressure_pa<result.pump_discharge_pressure_pa
    assert result.core_outlet_pressure_pa<result.core_inlet_pressure_pa
    assert 0<result.chamber_pressure_pa<result.core_outlet_pressure_pa
    assert result.core_inlet_temperature_k>200.
    assert result.core_inlet_temperature_k<result.core_outlet_temperature_k
    assert result.chamber_temperature_k<config.maximum_eos_temperature_k
    assert result.pump_discharge_temperature_k<34.
    assert result.pump_inlet_density_kg_m3>20.
    assert result.inlet_npsh_available_m>0.
    assert result.maximum_core_mach<config.allowable_mach
    assert result.nozzle.thrust_n>0
    assert result.nozzle.isp_s>0
    assert result.core_friction_loss_pa>0
    assert abs(result.energy_residual_w)<1e-6*max(
        1.,result.pump_shaft_power_w+
        result.preheater_thermal_power_w+result.core_thermal_power_w
    )
    assert abs(result.pressure_residual_pa)<1e-7*result.pump_discharge_pressure_pa

    p1=config.tank_pressure_pa
    p2=result.pump_discharge_pressure_pa
    h1=eos.tp("Hmass",config.tank_temperature_k,p1)
    s1=eos.tp("Smass",config.tank_temperature_k,p1)
    h2s=eos.sp("Hmass",s1,p2)
    h2=eos.tp("Hmass",result.pump_discharge_temperature_k,p2)
    assert math.isclose((h2s-h1)/(h2-h1),
                        config.pump_isentropic_efficiency,
                        rel_tol=2e-7)
    assert math.isclose(m*(h2-h1),result.pump_shaft_power_w,
                        rel_tol=2e-7,abs_tol=1e-3)
    p3=result.core_inlet_pressure_pa
    h3=eos.tp("Hmass",result.core_inlet_temperature_k,p3)
    assert math.isclose(m*(h3-h2),config.preheater_thermal_power_w,
                        rel_tol=3e-7,abs_tol=0.02)
    pc=result.chamber_pressure_pa
    hc=eos.tp("Hmass",result.chamber_temperature_k,pc)
    assert math.isclose(m*(hc-h3),result.core_thermal_power_w,
                        rel_tol=5e-7,abs_tol=0.05)


def main() -> int:
    eos=HydrogenEOS()
    case=example_case()
    result=solve_lh2_cycle(case,eos=eos)
    check_conservation(case,result,eos)

    direct,fx=evaluate_lh2_cycle(case,result.mass_flow_kg_s,eos=eos)
    assert abs(fx)<2e-6*result.mass_flow_kg_s
    assert math.isclose(result.chamber_pressure_pa,direct.chamber_pressure_pa,
                        rel_tol=1e-9)
    # Efficiency should alter LH2 pump shaft work and discharge enthalpy
    # without altering the pressure rise source curve.
    high_eta=replace(case,pump_isentropic_efficiency=0.85)
    better=solve_lh2_cycle(high_eta,eos=eos)
    check_conservation(high_eta,better,eos)
    assert better.pump_actual_enthalpy_rise_j_kg<result.pump_actual_enthalpy_rise_j_kg
    assert better.pump_shaft_power_w/better.mass_flow_kg_s<(
        result.pump_shaft_power_w/result.mass_flow_kg_s
    )

    # Same pump, more core heat yields a higher chamber temperature.
    more_heat=replace(case,channel_wall_power_w=(200000.,190000.,150000.))
    hotter=solve_lh2_cycle(more_heat,eos=eos)
    check_conservation(more_heat,hotter,eos)
    assert hotter.chamber_temperature_k>result.chamber_temperature_k

    # Not enough upstream preheat should never enter gas-only core.
    no_preheat=replace(case,preheater_thermal_power_w=1.)
    expect_rejected(lambda:solve_lh2_cycle(no_preheat,eos=eos),
                    CryogenicCycleError)

    # Tank vapor, nonliquid pump discharge, or EOS temperature outside
    # documented validity must fail, rather than extrapolate properties.
    expect_rejected(lambda:solve_lh2_cycle(
        replace(case,tank_temperature_k=300.),eos=eos))
    expect_rejected(lambda:solve_lh2_cycle(
        replace(case,tank_pressure_pa=0.02e6),eos=eos),CryogenicCycleError)
    expect_rejected(lambda:solve_lh2_cycle(
        replace(case,maximum_eos_temperature_k=1200.),eos=eos))
    expect_rejected(lambda:solve_lh2_cycle(
        replace(case,z_edges_m=(0.,0.2,0.1,0.3)),eos=eos))
    expect_rejected(lambda:solve_lh2_cycle(
        replace(case,pump_isentropic_efficiency=0.),eos=eos))

    if "--write-example" in sys.argv:
        idx=sys.argv.index("--write-example")
        if idx+1>=len(sys.argv):
            raise ValueError("must supply example destination path")
        target=Path(sys.argv[idx+1])
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(asdict(case),indent=2)+"\n",
                          encoding="utf-8")
        from .run_lh2_feed_cycle import main as runner
        previous=sys.argv
        sys.argv=["run_lh2_feed_cycle","--config",str(target),
                  "--output",str(target.parent/"lh2_cycle_result.json")]
        try:
            assert runner()==0
        finally:
            sys.argv=previous

    print("NERVA LH2 real-fluid cryopump/preheater/gas core/nozzle physics: PASS")
    print(f"  LH2 pump discharge {result.pump_discharge_pressure_pa/1e6:.6g} MPa")
    print(f"  mass flow {result.mass_flow_kg_s:.6g} kg/s")
    print(f"  pump work {result.pump_shaft_power_w:.6g} W")
    print(f"  core inlet {result.core_inlet_temperature_k:.6g} K")
    print(f"  chamber {result.chamber_pressure_pa/1e6:.6g} MPa / "
          f"{result.chamber_temperature_k:.6g} K")
    print(f"  nozzle thrust {result.nozzle.thrust_n:.6g} N / "
          f"Isp {result.nozzle.isp_s:.6g} s")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
