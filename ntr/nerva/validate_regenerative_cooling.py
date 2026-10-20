"""Conservation regressions for NASA-derived parallel regenerative cooling.

Two coolant branches from the LH2 pump:
* regenerative nozzle jacket -> reflector/control-drum path
* tie tube supply -> counterflow return (axis reversal is geometric)
The branches have pressure-balanced flows and an enthalpy-mixed header.
That header drives the existing pump/turbine-bleed/hot-core/nozzle
network. Every deposited W is counted exactly once.

Only surrogate heat sources and lumped loss coefficients are used.
The two-phase homogeneous equilibrium density is a screening model,
not validated NERVA startup or critical-heat-flux predictions.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import sys

from .lh2_feed_cycle import HydrogenEOS
from .regenerative_cooling import (
    CoolingLeg,CoolingNetwork,pump_and_regenerative_cooling,
    solve_cooling_network,
)
from .turbine_bleed_cycle import evaluate_turbine_bleed, solve_turbine_bleed
from .validate_hot_lh2_cycle import hot_case
from .validate_turbine_bleed_cycle import sample_turbine


def network() -> CoolingNetwork:
    return CoolingNetwork(
        nozzle_jacket=CoolingLeg(
            heat_power_w=110000.,flow_area_m2=0.0008,
            pressure_loss_coefficient=6.),
        reflector=CoolingLeg(
            heat_power_w=90000.,flow_area_m2=0.0006,
            pressure_loss_coefficient=9.),
        tie_supply=CoolingLeg(
            heat_power_w=65000.,flow_area_m2=0.00040,
            pressure_loss_coefficient=13.),
        tie_return=CoolingLeg(
            heat_power_w=85000.,flow_area_m2=0.00045,
            pressure_loss_coefficient=12.),
    )


def reject(fn):
    try:
        fn()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("unphysical regenerative cooling case was accepted")


def check_physics(core, net, solution):
    h=solution.regenerative_cooling
    assert h is not None
    assert h.total_heat_power_w==core.preheater_thermal_power_w
    assert 0<h.tie_tube_fraction<1.
    assert 200<h.common_header_temperature_k<900
    assert 0<h.common_header_pressure_pa<solution.pump_discharge_pressure_pa
    assert math.isclose(
        h.nozzle_reflector_mass_flow_kg_s+h.tie_tube_mass_flow_kg_s,
        solution.tank_mass_flow_kg_s,rel_tol=1e-12
    )
    a,b,c,d=h.legs
    assert [x.name for x in h.legs]==[
        "nozzle_jacket","reflector","tie_supply","tie_return"
    ]
    assert math.isclose(a.mass_flow_kg_s,b.mass_flow_kg_s,rel_tol=1e-12)
    assert math.isclose(c.mass_flow_kg_s,d.mass_flow_kg_s,rel_tol=1e-12)
    assert a.inlet_pressure_pa==c.inlet_pressure_pa
    assert math.isclose(
        a.outlet_pressure_pa,b.inlet_pressure_pa,rel_tol=1e-12
    )
    assert math.isclose(
        c.outlet_pressure_pa,d.inlet_pressure_pa,rel_tol=1e-12
    )
    assert abs(b.outlet_pressure_pa-d.outlet_pressure_pa)<1.
    assert h.maximum_leg_energy_residual_w<0.1
    assert h.maximum_leg_momentum_residual_pa<0.1
    assert abs(h.total_energy_residual_w)<1e-4
    assert abs(h.split_pressure_residual_pa)<1.
    for leg in h.legs:
        assert leg.mass_flow_kg_s>0
        assert leg.inlet_pressure_pa>leg.outlet_pressure_pa>0
        assert leg.heat_power_w>0
        assert math.isclose(
            leg.mass_flow_kg_s*(leg.outlet_stagnation_enthalpy_j_kg-
                               leg.inlet_stagnation_enthalpy_j_kg),
            leg.heat_power_w,rel_tol=2e-9,abs_tol=1e-4
        )
        assert math.isclose(
            leg.inlet_pressure_pa-leg.outlet_pressure_pa,
            leg.irreversible_pressure_drop_pa+
            leg.acceleration_pressure_drop_pa,
            rel_tol=3e-6,abs_tol=0.1
        )
    assert abs(solution.shaft_power_residual_w)<1e-5
    assert abs(solution.total_cycle_energy_residual_w)<1e-3
    assert abs(solution.channel_pressure_residual_pa)<0.1
    assert abs(solution.nozzle_mass_flow_residual_kg_s)<2e-6*solution.core_mass_flow_kg_s
    assert math.isclose(solution.turbine_shaft_delivered_power_w,
                        solution.pump_required_power_w,rel_tol=1e-9)
    assert 0<solution.turbine_bleed_fraction<0.5


def main() -> int:
    eos=HydrogenEOS()
    core=hot_case()
    bleed=sample_turbine()
    net=network()
    net.validate()
    assert net.total_heat_w==core.preheater_thermal_power_w

    # Fully solve hydraulic split, turbine bleed and core/nozzle flow,
    # not just an externally prescribed split in fixed preheater.
    result=solve_turbine_bleed(
        core,bleed,eos=eos,chemistry_mode="frozen",
        cooling_network=net
    )
    check_physics(core,net,result)
    _,root=evaluate_turbine_bleed(
        core,bleed,result.tank_mass_flow_kg_s,eos=eos,
        cooling_network=net
    )
    assert abs(root)<2e-6*result.core_mass_flow_kg_s

    # Original lumped preheater is still available and reproduces
    # prior results. Explicit cooling MUST change its pressure losses.
    original=solve_turbine_bleed(core,bleed,eos=eos)
    assert original.regenerative_cooling is None
    assert abs(result.chamber_pressure_pa-original.chamber_pressure_pa)>10.
    assert abs(result.tank_mass_flow_kg_s-original.tank_mass_flow_kg_s)>1e-8

    # Increasing tie resistance should shift flow AWAY from the ties;
    # the pressure split is solved, not hard-coded.
    higher_tie_resistance=replace(
        net,
        tie_supply=replace(
            net.tie_supply,pressure_loss_coefficient=26.
        ),
    )
    modified,_=evaluate_turbine_bleed(
        core,bleed,result.tank_mass_flow_kg_s,eos=eos,
        full_nozzle=True,cooling_network=higher_tie_resistance
    )
    # Fixed-flow sensitivity, NOT a second nozzle-closed operating point.
    assert modified.regenerative_cooling is not None
    assert abs(modified.regenerative_cooling.total_energy_residual_w)<1e-4
    assert (modified.regenerative_cooling.tie_tube_fraction<
            result.regenerative_cooling.tie_tube_fraction)

    # Repartition deposited heat WITHOUT changing the reactor's total
    # preheater heating. Common header total energy stays conservative.
    shifted=replace(
        net,
        reflector=replace(net.reflector,heat_power_w=100000.),
        tie_return=replace(net.tie_return,heat_power_w=75000.),
    )
    moved,_=evaluate_turbine_bleed(
        core,bleed,result.tank_mass_flow_kg_s,eos=eos,
        full_nozzle=True,cooling_network=shifted
    )
    assert moved.regenerative_cooling is not None
    assert abs(moved.regenerative_cooling.total_energy_residual_w)<1e-4
    assert moved.regenerative_cooling.total_heat_power_w==net.total_heat_w

    # Chemical-equilibrium core/nozzle must remain supported downstream
    # of real-fluid H2 mixing. CoolProp mixing is below 900 K.
    eq,_=evaluate_turbine_bleed(
        core,bleed,result.tank_mass_flow_kg_s,eos=eos,
        chemistry_mode="equilibrium",
        cooling_network=net,full_nozzle=True
    )
    assert abs(eq.total_cycle_energy_residual_w)<1e-3
    assert abs(eq.shaft_power_residual_w)<1e-5
    assert eq.chemistry_mode=="equilibrium"
    assert eq.chamber_dissociation_fraction>0

    reject(lambda:solve_turbine_bleed(
        core,bleed,eos=eos,cooling_network=replace(
            net,tie_return=replace(net.tie_return,heat_power_w=100000.)
        )
    ))  # reject nuclear power double-counting
    reject(lambda:solve_turbine_bleed(
        core,bleed,eos=eos,cooling_network=replace(
            net,tie_supply=replace(net.tie_supply,flow_area_m2=0.)
        )
    ))
    reject(lambda:solve_turbine_bleed(
        core,bleed,eos=eos,cooling_network=replace(
            net,nozzle_jacket=replace(
                net.nozzle_jacket,pressure_loss_coefficient=-1.
            )
        )
    ))

    if "--write-example" in sys.argv:
        idx=sys.argv.index("--write-example")
        if idx+1>=len(sys.argv):
            raise ValueError("missing output case path")
        target=Path(sys.argv[idx+1])
        target.parent.mkdir(parents=True,exist_ok=True)
        case={
            "lh2_cycle":asdict(core),
            "turbine":asdict(bleed),
            "regenerative_cooling":asdict(net),
        }
        target.write_text(json.dumps(case,indent=2)+"\n",encoding="utf-8")
        from .run_turbine_bleed_cycle import main as cli
        prior=sys.argv
        sys.argv=[
            "run_turbine_bleed_cycle","--config",str(target),
            "--output",str(target.parent/"regenerative_cooling_result.json")
        ]
        try:
            assert cli()==0
        finally:
            sys.argv=prior
        document=json.loads(
            (target.parent/"regenerative_cooling_result.json").read_text()
        )
        assert document["parallel_cooling_branches_solved"] is True
        assert document["point"]["regenerative_cooling"] is not None

    print("NERVA parallel regenerative nozzle/reflector/tie cooling physics: PASS")
    print(f"  tank flow {result.tank_mass_flow_kg_s:.8g} kg/s")
    print(f"  tie fraction {result.regenerative_cooling.tie_tube_fraction:.7g}")
    print(f"  mixed header {result.regenerative_cooling.common_header_temperature_k:.7g} K / "
          f"{result.regenerative_cooling.common_header_pressure_pa/1e6:.7g} MPa")
    print(f"  pump/turbine shaft {result.pump_required_power_w:.7g} W")
    print(f"  chamber {result.chamber_temperature_k:.7g} K / "
          f"{result.chamber_pressure_pa/1e6:.7g} MPa")
    print(f"  coolant first law {result.regenerative_cooling.total_energy_residual_w:.7g} W")
    print(f"  whole-cycle first law {result.total_cycle_energy_residual_w:.7g} W")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
