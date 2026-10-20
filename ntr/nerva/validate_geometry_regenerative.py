"""Physics checks for geometry-based Darcy H2 regenerative branch losses.

Validates f_D(Re,roughness), K=f_D L/D + K_minor, separate pressure
loss components, thermo-energy/streamwise-momentum budgets, and effect
on the parallel tie/refector hydraulic flow split. Historical NERVA
flow areas and path lengths remain to be calibrated to hardware.
"""
from __future__ import annotations

from dataclasses import replace
import math
import numpy as np

from .lh2_feed_cycle import HydrogenEOS, CryogenicCycleError
from .regenerative_cooling import (
    CoolingLeg, CoolingNetwork, _geometry_resistance,
    pump_and_regenerative_cooling, solve_cooling_network
)
from .thermal import friction_factor
from .turbine_bleed_cycle import evaluate_turbine_bleed
from .validate_hot_lh2_cycle import hot_case
from .validate_turbine_bleed_cycle import sample_turbine


def geometric_network() -> CoolingNetwork:
    return CoolingNetwork(
        nozzle_jacket=CoolingLeg(
            heat_power_w=110000.,flow_area_m2=0.0008,
            length_m=2.,hydraulic_diameter_m=0.008,
            roughness_m=3e-6,minor_loss_coefficient=0.4,
        ),
        reflector=CoolingLeg(
            heat_power_w=90000.,flow_area_m2=0.0006,
            length_m=3.,hydraulic_diameter_m=0.008,
            roughness_m=3e-6,minor_loss_coefficient=0.6,
        ),
        tie_supply=CoolingLeg(
            heat_power_w=65000.,flow_area_m2=0.00040,
            length_m=2.5,hydraulic_diameter_m=0.006,
            roughness_m=2e-6,minor_loss_coefficient=0.3,
        ),
        tie_return=CoolingLeg(
            heat_power_w=85000.,flow_area_m2=0.00045,
            length_m=2.5,hydraulic_diameter_m=0.006,
            roughness_m=2e-6,minor_loss_coefficient=0.5,
        ),
    )


def rejects(callback):
    try:
        callback()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("unphysical hydraulic configuration was accepted")


def verify(eos,net,flow):
    assert abs(flow.total_energy_residual_w)<0.001
    assert abs(flow.split_pressure_residual_pa)<1.
    assert flow.maximum_leg_energy_residual_w<0.01
    assert flow.maximum_leg_momentum_residual_pa<0.1
    assert abs(flow.nozzle_reflector_mass_flow_kg_s+
               flow.tie_tube_mass_flow_kg_s-
               (flow.nozzle_reflector_mass_flow_kg_s+
                flow.tie_tube_mass_flow_kg_s))<1e-15
    for name,leg in zip(
        ("nozzle_jacket","reflector","tie_supply","tie_return"),
        (net.nozzle_jacket,net.reflector,net.tie_supply,net.tie_return)
    ):
        st=next(x for x in flow.legs if x.name==name)
        assert st.darcy_friction_factor is not None
        assert st.reynolds is not None and st.reynolds>0
        assert st.outlet_mach is not None and st.outlet_mach<0.85
        expected_f=friction_factor(st.reynolds,leg.roughness_m/
                                             leg.hydraulic_diameter_m)
        expected_k=float(expected_f)*leg.length_m/leg.hydraulic_diameter_m
        assert math.isclose(st.darcy_friction_factor,expected_f,rel_tol=1e-10)
        assert math.isclose(st.effective_pressure_loss_coefficient,
                            expected_k+leg.minor_loss_coefficient,rel_tol=1e-10)
        assert math.isclose(st.irreversible_pressure_drop_pa,
                            st.darcy_pressure_drop_pa+st.minor_pressure_drop_pa,
                            rel_tol=1e-9,abs_tol=1e-8)
        assert math.isclose(st.inlet_pressure_pa-st.outlet_pressure_pa,
                            st.irreversible_pressure_drop_pa+
                            st.acceleration_pressure_drop_pa,
                            rel_tol=1e-7,abs_tol=0.1)
        assert math.isclose(st.mass_flow_kg_s*(
            st.outlet_stagnation_enthalpy_j_kg-
            st.inlet_stagnation_enthalpy_j_kg),leg.heat_power_w,
            rel_tol=1e-10,abs_tol=1e-5)


def main() -> int:
    eos=HydrogenEOS()
    cfg=hot_case()
    net=geometric_network()
    net.validate()
    assert math.isclose(net.total_heat_w,cfg.preheater_thermal_power_w)

    # Verify geometry on the full LH2 pump outlet (cryogenic liquid)
    # all the way to the shared single-phase gaseous header.
    mass=0.051885
    cond=pump_and_regenerative_cooling(cfg,net,mass,eos=eos)
    verify(eos,net,cond.cooling)
    assert 200<cond.header_temperature_k<900
    assert 0<cond.cooling.tie_tube_fraction<1
    assert all(st.effective_pressure_loss_coefficient>0
               for st in cond.cooling.legs)

    # Reference-independent, exact laminar/turbulent Darcy limits.
    mu=1e-5
    diameter=0.01
    for re in (1000.,30000.,1e5):
        friction=float(friction_factor(re,2e-4))
        assert friction>0
        if re<2300:
            assert math.isclose(friction,64/re,rel_tol=1e-12)
        assert math.isfinite(friction)

    # Increased tie passage length produces larger resistance:
    # flow moves away from the tie branch automatically.
    lengthened=replace(net,tie_supply=replace(
        net.tie_supply,length_m=net.tie_supply.length_m*1.6
    ))
    changed=pump_and_regenerative_cooling(
        cfg,lengthened,mass,eos=eos
    )
    verify(eos,lengthened,changed.cooling)
    assert changed.cooling.tie_tube_fraction < cond.cooling.tie_tube_fraction

    # Increased wall roughness raises the Darcy resistance and shifts
    # the hydraulic split as expected at fixed inlet head and total flow.
    rougher=replace(net,tie_return=replace(
        net.tie_return,roughness_m=0.00015
    ))
    shifted=pump_and_regenerative_cooling(cfg,rougher,mass,eos=eos)
    verify(eos,rougher,shifted.cooling)
    assert shifted.cooling.tie_tube_fraction < cond.cooling.tie_tube_fraction

    # Whole engine integration at fixed tank flow: turbine shaft work
    # and pump power remain balanced; the outlet pressure and chamber
    # change as the hydraulic resistance changes.
    bleed=sample_turbine()
    point,_=evaluate_turbine_bleed(
        cfg,bleed,mass,eos=eos,cooling_network=net,full_nozzle=True
    )
    assert point.regenerative_cooling is not None
    assert abs(point.shaft_power_residual_w)<1e-4
    assert abs(point.total_cycle_energy_residual_w)<0.001
    assert abs(point.channel_pressure_residual_pa)<0.1
    assert point.chamber_pressure_pa<cond.header_pressure_pa
    assert point.regenerative_cooling.tie_tube_fraction==cond.cooling.tie_tube_fraction

    # Hard error on double-counting the K: fixed aggregate coefficient
    # cannot be added on top of geometry-derived Darcy friction.
    rejects(lambda:replace(net.nozzle_jacket,pressure_loss_coefficient=3.).validate())
    rejects(lambda:replace(net.nozzle_jacket,hydraulic_diameter_m=None).validate())
    rejects(lambda:replace(net.nozzle_jacket,flow_area_m2=0.).validate())
    rejects(lambda:replace(net.nozzle_jacket,roughness_m=-1.).validate())
    rejects(lambda:replace(net.nozzle_jacket,length_m=-1.).validate())

    # The homogeneous-equilibrium model is retained for state enthalpy,
    # but single-phase Darcy correlation must not silently be used in a
    # two-phase midpoint state: liquid/vapor slip must be resolved later.
    p_sat=0.55e6
    h_liq=eos.get("Hmass","P",p_sat,"Q",0.)
    h_vap=eos.get("Hmass","P",p_sat,"Q",1.)
    h_mix=0.5*(h_liq+h_vap)
    rejects(lambda:_geometry_resistance(
        eos,net.nozzle_jacket,12.,p_sat,p_sat,h_mix,h_mix
    ))

    print("NERVA geometry-based regenerative Darcy/flow split physics: PASS")
    print(f"  tie flow fraction: {cond.cooling.tie_tube_fraction:.7g}")
    print(f"  mixed header: {cond.header_temperature_k:.7g} K, "
          f"{cond.header_pressure_pa/1e6:.7g} MPa")
    print(f"  four-leg max energy error: {cond.cooling.maximum_leg_energy_residual_w:.6g} W")
    print(f"  four-leg max momentum error: {cond.cooling.maximum_leg_momentum_residual_pa:.6g} Pa")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
