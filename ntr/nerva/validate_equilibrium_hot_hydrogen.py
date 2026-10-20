"""Physical regressions of equilibrium H2 dissociation in hot NERVA flow.

Under H2 <-> 2H ideal-gas equilibrium, chemical bond-breaking absorbs
reactor heat. Verify changes in channel/chamber state against frozen-H2
baseline while preserving each cell's total energy and momentum and
the full LH2-to-nozzle mass and enthalpy balances.

Illustrative fixed nuclear power and source pressure curve only.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from .compressible_channel import (
    _mach, solve_fuel_channel_compressible, SubsonicFlowError
)
from .hot_lh2_cycle import solve_hot_core_cycle
from .lh2_feed_cycle import HydrogenEOS
from .nasa_hydrogen import hydrogen_state, T_MAX
from .validate_hot_lh2_cycle import hot_case


def rejects(callback):
    try:
        callback()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("invalid chemistry/state unexpectedly accepted")


def main() -> int:
    # Thermochemical physical limits: low T / high pressure yields
    # almost all molecular hydrogen; high T yields free atomic H.
    p=6e6
    low=hydrogen_state(500.,p,mode="equilibrium")
    hot=hydrogen_state(3000.,p,mode="equilibrium")
    frozen=hydrogen_state(3000.,p,mode="frozen")
    assert low.dissociation_fraction<1e-7
    assert 0<hot.dissociation_fraction<1
    assert hot.enthalpy_j_kg>frozen.enthalpy_j_kg
    assert hot.density_kg_m3<frozen.density_kg_m3
    assert 0<_mach(hot,0.7,"equilibrium")<1
    assert math.isfinite(hot.entropy_j_kg_k)

    # One-dimensional channel: direct H nuclear deposition and solid
    # wall heat both enter total-energy conservation EXACTLY once.
    q=np.array([60000.,80000.,90000.])
    direct=np.array([4000.,9000.,10000.])
    z=np.array([0.,0.1,0.2,0.3])
    state_kwargs=dict(
        axial_total_fuel_power_w=q,z_edges_m=z,
        fuel_channel_count=10,
        mass_flow_per_channel_kg_s=0.005,
        inlet_temperature_k=650.,inlet_pressure_pa=6e6,
        channel_diameter_m=0.009,
        axial_direct_coolant_power_w=direct,
        max_mach=0.85,
    )
    molecule=solve_fuel_channel_compressible(
        **state_kwargs,chemistry_mode="frozen"
    )
    reacting=solve_fuel_channel_compressible(
        **state_kwargs,chemistry_mode="equilibrium"
    )
    assert abs(reacting.absorbed_power_w-molecule.absorbed_power_w)<1e-8
    assert reacting.dissociation_fraction is not None
    assert reacting.dissociation_fraction.shape==(3,)
    assert np.all(np.isfinite(reacting.dissociation_fraction))
    assert np.all(np.diff(reacting.dissociation_fraction)>=-1e-11)
    assert max(abs(reacting.energy_residual_w))<0.1
    assert max(abs(reacting.momentum_residual_pa))<0.1
    assert abs(reacting.outlet_temperature_k-molecule.outlet_temperature_k)>1e-7
    assert reacting.outlet_pressure_pa<reacting.pressure_pa[0]
    assert np.max(reacting.mach_number)<0.85

    # Global cryogenic LH2 pump to reactor/nozzle balance.  All
    # absolute enthalpy zeros are matched at the warm header, once.
    config=hot_case()
    eos=HydrogenEOS()
    baseline=solve_hot_core_cycle(config,eos=eos,chemistry_mode="frozen")
    equilibrium=solve_hot_core_cycle(config,eos=eos,chemistry_mode="equilibrium")
    assert equilibrium.chemistry_mode=="equilibrium"
    assert baseline.chemistry_mode=="frozen"
    assert baseline.chamber_dissociation_fraction==0
    assert equilibrium.chamber_dissociation_fraction>0
    assert equilibrium.channel_outlet_dissociation_fraction>0
    assert equilibrium.chamber_temperature_k<baseline.chamber_temperature_k
    energy_scale=(equilibrium.pump_shaft_power_w+
                  equilibrium.preheater_power_w+
                  equilibrium.core_heating_power_w)
    print(f"Equilibrium cycle energy residual = "
          f"{equilibrium.overall_energy_residual_w:.8g} W "
          f"of {energy_scale:.8g} W input")
    assert abs(equilibrium.overall_energy_residual_w)<1e-9*energy_scale
    assert abs(equilibrium.pressure_residual_pa)<0.01
    assert abs(equilibrium.max_channel_energy_residual_w)<0.01
    assert abs(equilibrium.max_channel_momentum_residual_pa)<0.01
    assert abs(equilibrium.nozzle_mass_flow_residual_kg_s)<2e-6*equilibrium.mass_flow_kg_s
    assert equilibrium.maximum_channel_mach<config.allowable_mach
    assert equilibrium.nozzle.chemistry_mode=="equilibrium"
    assert math.isclose(
        equilibrium.nozzle.mass_flow_kg_s,equilibrium.mass_flow_kg_s,
        rel_tol=2e-6
    )
    assert 0<equilibrium.nozzle.exit.dissociation_fraction<1
    assert equilibrium.nozzle.enthalpy_to_kinetic_residual_j_kg<1e-4
    assert abs(equilibrium.chamber_temperature_k-baseline.chamber_temperature_k)>0.01
    assert abs(equilibrium.nozzle.isp_s-baseline.nozzle.isp_s)>0.01

    # All high-temperature thermodynamics have explicit limited domain.
    rejects(lambda:solve_hot_core_cycle(
        config,eos=eos,chemistry_mode="not-a-valid-chemistry"
    ))
    rejects(lambda:solve_fuel_channel_compressible(
        **state_kwargs,chemistry_mode="invalid"
    ))
    rejects(lambda:hydrogen_state(T_MAX+1.,p,mode="equilibrium"))

    # CLI with same frozen/equilibrium switch and machine-readable
    # dissociation output, without inventing reactor validation.
    if "--write-example" in sys.argv:
        import dataclasses
        i=sys.argv.index("--write-example")
        if len(sys.argv)<i+2:
            raise ValueError("missing example output filename")
        name=Path(sys.argv[i+1])
        name.parent.mkdir(parents=True,exist_ok=True)
        name.write_text(json.dumps(dataclasses.asdict(config),indent=2)+"\n")
        from .run_hot_lh2_cycle import main as cli
        old_argv=sys.argv
        for mode in ("frozen","equilibrium"):
            sys.argv=["run_hot_lh2_cycle","--config",str(name),
                      "--chemistry-mode",mode,
                      "--output",str(name.with_name("hot_"+mode+".json"))]
            try:
                assert cli()==0
            finally:
                sys.argv=old_argv
            output=json.loads(name.with_name("hot_"+mode+".json").read_text())
            assert output["point"]["chemistry_mode"]==mode
            assert output["status"]=="PASS"

    print("Hot NERVA hydrogen dissociation and equilibrium nozzle physics: PASS")
    print(f"  frozen chamber: {baseline.chamber_temperature_k:.6g} K, "
          f"{baseline.chamber_pressure_pa/1e6:.6g} MPa")
    print(f"  equilibrium chamber: {equilibrium.chamber_temperature_k:.6g} K, "
          f"{equilibrium.chamber_pressure_pa/1e6:.6g} MPa")
    print(f"  equilibrium alpha: {equilibrium.chamber_dissociation_fraction:.7g}")
    print(f"  frozen/equilibrium Isp: "
          f"{baseline.nozzle.isp_s:.6g}/{equilibrium.nozzle.isp_s:.6g} s")
    print(f"  global first law: {equilibrium.overall_energy_residual_w:.5g} W")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
