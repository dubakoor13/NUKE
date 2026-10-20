"""Numerical physical checks for cryogenic LH2-to-NASA7 hot hydrogen.

CoolProp handles the liquid pump / vaporizing header; NASA7 frozen
molecular H2 handles all >200 K core and nozzle thermodynamics.
Synthetic imposed nuclear heat / pump curve are NOT NASA test data.
"""
from __future__ import annotations

from dataclasses import asdict,replace
import json
import math
from pathlib import Path
import sys

from .hot_lh2_cycle import (
    evaluate_hot_core_cycle, solve_hot_core_cycle,
)
from .lh2_feed_cycle import LH2Cycle, HydrogenEOS, CryogenicCycleError
from .nasa_hydrogen import hydrogen_state
from .validate_lh2_feed_cycle import example_case


def hot_case() -> LH2Cycle:
    base=example_case()
    return replace(
        base,
        channel_wall_power_w=(600000.,570000.,480000.),
        channel_direct_hydrogen_power_w=(50000.,50000.,50000.),
        nozzle_expansion_ratio=8.,
    )


def rejected(f):
    try:
        f()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("expected physical model rejection")


def check_energy_and_nozzle(case,point,eos):
    mdot=point.mass_flow_kg_s
    assert 0. < mdot <= case.maximum_mass_flow_kg_s
    assert abs(point.nozzle_mass_flow_residual_kg_s)<2e-6*mdot
    assert math.isclose(mdot,point.nozzle.mass_flow_kg_s,rel_tol=2e-6)
    assert point.pump_shaft_power_w>0
    assert point.core_heating_power_w>0
    assert 200<point.bridge_temperature_k<900
    assert point.chamber_temperature_k>point.bridge_temperature_k
    assert point.chamber_temperature_k<=3500
    assert point.chamber_pressure_pa>0
    assert point.maximum_channel_mach<case.allowable_mach
    assert point.nozzle.thrust_n>0 and point.nozzle.isp_s>0
    assert point.friction_pressure_loss_pa>0
    assert math.isfinite(point.bridge_cp_relative_jump)
    assert math.isfinite(point.bridge_density_relative_jump)
    assert abs(point.max_channel_energy_residual_w)<0.01
    assert abs(point.max_channel_momentum_residual_pa)<0.01
    assert abs(point.pressure_residual_pa)<0.01

    q=point.core_heating_power_w+point.preheater_power_w
    scale=point.pump_shaft_power_w+q
    assert abs(point.overall_energy_residual_w)<=1e-7*scale
    h_tank=eos.tp("Hmass",case.tank_temperature_k,case.tank_pressure_pa)
    h3=eos.tp(
        "Hmass",point.bridge_temperature_k,point.core_inlet_pressure_pa
    )
    h_nasa3=hydrogen_state(
        point.bridge_temperature_k,point.core_inlet_pressure_pa,
        mode="frozen"
    ).enthalpy_j_kg
    h_nasa4=hydrogen_state(
        point.chamber_temperature_k,point.chamber_pressure_pa,
        mode="frozen"
    ).enthalpy_j_kg
    # Independently reconstruct the common cryogenic-to-hot enthalpy
    # through the same physical bridge using a second arithmetic path.
    h_chamber=h3+(h_nasa4-h_nasa3)
    assert math.isclose(
        mdot*(h_chamber-h_tank),
        point.pump_shaft_power_w+q,
        rel_tol=2e-7,abs_tol=0.05
    )
    assert math.isclose(
        point.bridge_enthalpy_offset_j_kg,h3-h_nasa3,
        rel_tol=1e-12,abs_tol=1e-7
    )
    assert math.isclose(
        point.pump_discharge_pressure_pa-
        point.chamber_pressure_pa,
        case.feed_loss_quadratic_pa_s2_kg2*mdot**2+
        point.friction_pressure_loss_pa+
        point.acceleration_pressure_loss_pa+
        case.chamber_loss_quadratic_pa_s2_kg2*mdot**2,
        rel_tol=1e-7,abs_tol=1e-3
    )
    assert abs(point.nozzle.enthalpy_to_kinetic_residual_j_kg)<1e-5
    assert point.nozzle.exit.temperature_k>=200


def main() -> int:
    eos=HydrogenEOS()
    case=hot_case()
    output=solve_hot_core_cycle(case,eos=eos)
    check_energy_and_nozzle(case,output,eos)
    _,residual=evaluate_hot_core_cycle(
        case,output.mass_flow_kg_s,eos=eos
    )
    assert abs(residual)<2e-6*output.mass_flow_kg_s

    # At a fixed pump curve, more core heat must raise gas thermal state
    # and alter nozzle mass flow; do not infer fixed pressure or fixed Isp.
    warmer=replace(case,
        channel_wall_power_w=tuple(
            1.1*x for x in case.channel_wall_power_w
        )
    )
    hotter=solve_hot_core_cycle(warmer,eos=eos)
    check_energy_and_nozzle(warmer,hotter,eos)
    assert hotter.chamber_temperature_k>output.chamber_temperature_k
    assert hotter.core_heating_power_w>output.core_heating_power_w

    # Higher preheater heat shifts the bridge temperature, and global
    # first-law closure must still preserve reference matching.
    more_preheat=replace(
        case,preheater_thermal_power_w=390000.
    )
    preheated=solve_hot_core_cycle(more_preheat,eos=eos)
    check_energy_and_nozzle(more_preheat,preheated,eos)
    assert preheated.bridge_temperature_k>output.bridge_temperature_k

    # An invalid liquid inlet, insufficient preheater, and no admissible
    # channel/nozzle root must fail rather than invent a hot core point.
    rejected(lambda:solve_hot_core_cycle(replace(
        case,tank_temperature_k=300.),eos=eos
    ))
    rejected(lambda:solve_hot_core_cycle(replace(
        case,preheater_thermal_power_w=1.),eos=eos
    ))
    rejected(lambda:solve_hot_core_cycle(replace(
        case,maximum_mass_flow_kg_s=0.002
    ),eos=eos))
    rejected(lambda:solve_hot_core_cycle(replace(
        case,channel_wall_power_w=(9e7,9e7,9e7)
    ),eos=eos))

    if "--write-example" in sys.argv:
        i=sys.argv.index("--write-example")
        if len(sys.argv)<=i+1:
            raise ValueError("--write-example requires output JSON filename")
        dest=Path(sys.argv[i+1])
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(asdict(case),indent=2)+"\n",
                        encoding="utf-8")
        from .run_hot_lh2_cycle import main as cli
        old=sys.argv
        sys.argv=["run_hot_lh2_cycle","--config",str(dest),
                  "--output",str(dest.with_name("hot_lh2_cycle.json"))]
        try:
            assert cli()==0
        finally:
            sys.argv=old

    print("NERVA LH2 realfluid -> hot NASA7 fuel channel/nozzle: PASS")
    print(f"  flow: {output.mass_flow_kg_s:.8g} kg/s")
    print(f"  warm-gas handoff: {output.bridge_temperature_k:.6g} K")
    print(f"  chamber: {output.chamber_pressure_pa/1e6:.6g} MPa, "
          f"{output.chamber_temperature_k:.6g} K")
    print(f"  thrust: {output.nozzle.thrust_n:.6g} N; "
          f"Isp: {output.nozzle.isp_s:.6g} s")
    print(f"  first-law residual: {output.overall_energy_residual_w:.8g} W")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
