"""Physics checks for opt-in homogeneous equilibrium LH2 two-phase friction.

The underlying McAdams mixture-viscosity/Darcy approximation is for
screening only, NOT NERVA-calibrated two-phase slip, boiling, CHF or
chilldown data. Test a real subcritical two-phase hydrogen state
separately from the high-pressure, supercritical example cycle.
"""
from __future__ import annotations

from dataclasses import replace
import math

from .lh2_feed_cycle import HydrogenEOS, CryogenicCycleError
from .regenerative_cooling import (
    CoolingLeg, _leg, _geometry_resistance,
    _equilibrium_two_phase_sound_speed, pump_and_regenerative_cooling,
)
from .thermal import friction_factor
from .validate_geometry_regenerative import geometric_network
from .validate_hot_lh2_cycle import hot_case


def rejects(fn):
    try:
        fn()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("nonphysical two-phase case did not fail")


def main() -> int:
    eos=HydrogenEOS()
    p=0.55e6  # Below H2 critical pressure: true liquid+vapor coexistence
    hl=eos.get("Hmass","P",p,"Q",0.)
    hv=eos.get("Hmass","P",p,"Q",1.)
    assert hv>hl
    mu_l=eos.get("VISCOSITY","P",p,"Q",0.)
    mu_v=eos.get("VISCOSITY","P",p,"Q",1.)
    assert mu_l>0 and mu_v>0
    leg=CoolingLeg(
        heat_power_w=650.,
        flow_area_m2=0.003,
        length_m=0.8,
        hydraulic_diameter_m=0.007,
        roughness_m=3.e-6,
        minor_loss_coefficient=0.5,
        two_phase_friction_model="homogeneous-mcadams",
    )
    mass=0.02
    inlet_h=hl+0.30*(hv-hl)
    # Midpoint state is genuinely two phase, and the quality is
    # determined by the EOS rather than manually imposed.
    mflow=mass/leg.flow_area_m2
    k, kd, re, ff=_geometry_resistance(
        eos,leg,mflow,p,p,inlet_h,inlet_h
    )
    assert k>kd>0 and re>0 and ff>0
    mu_mix=1./(0.7/mu_l+0.3/mu_v)
    expected_re=mflow*leg.hydraulic_diameter_m/mu_mix
    assert math.isclose(re,expected_re,rel_tol=1e-7)
    assert math.isclose(ff,float(friction_factor(
        re,leg.roughness_m/leg.hydraulic_diameter_m
    )),rel_tol=1e-12)

    # McAdams reproduces pure liquid and vapor viscosities at
    # x=0 and x=1, without inventing two-phase viscosity from CoolProp.
    for quality,expected in ((0.,mu_l),(1.,mu_v)):
        h=hl+quality*(hv-hl)
        _,_,rx,_=_geometry_resistance(eos,leg,mflow,p,p,h,h)
        assert math.isclose(
            rx,mflow*leg.hydraulic_diameter_m/expected,rel_tol=1e-7
        )

    # Isentropic homogeneous-equilibrium sound speed is positive.
    a=_equilibrium_two_phase_sound_speed(eos,p,inlet_h)
    assert 0<a<2000

    # Heated two-phase coolant element obeys total enthalpy,
    # streamwise momentum and explicit friction/minor losses.
    result=_leg(eos,"two_phase_test",leg,mass,p,inlet_h)
    assert result.midpoint_quality is not None
    assert 0<result.midpoint_quality<1
    assert result.outlet_quality>result.inlet_quality
    assert result.reynolds>0
    assert result.outlet_mach<0.85
    assert result.inlet_pressure_pa>result.outlet_pressure_pa>0
    assert abs(result.energy_residual_w)<1e-4
    assert abs(result.momentum_residual_pa)<0.1
    assert math.isclose(
        result.darcy_pressure_drop_pa+result.minor_pressure_drop_pa,
        result.irreversible_pressure_drop_pa,rel_tol=1e-9
    )
    assert math.isclose(
        result.inlet_pressure_pa-result.outlet_pressure_pa,
        result.irreversible_pressure_drop_pa+
        result.acceleration_pressure_drop_pa,
        rel_tol=1e-7,abs_tol=0.1
    )

    # Longer passage raises the total and Darcy pressure drop while
    # keeping the same imposed energy deposition per total mass.
    longer=replace(leg,length_m=1.6)
    changed=_leg(eos,"two_phase_longer",longer,mass,p,inlet_h)
    assert changed.outlet_pressure_pa<result.outlet_pressure_pa
    assert math.isclose(
        changed.outlet_stagnation_enthalpy_j_kg-
        changed.inlet_stagnation_enthalpy_j_kg,
        650./mass,rel_tol=1e-10
    )

    # The default must still reject two-phase geometric friction.
    old=replace(leg,two_phase_friction_model="reject")
    rejects(lambda:_geometry_resistance(
        eos,old,mflow,p,p,inlet_h,inlet_h
    ))
    rejects(lambda:_leg(eos,"default_reject",old,mass,p,inlet_h))
    rejects(lambda:replace(
        leg,two_phase_friction_model="unavailable-correlation"
    ).validate())
    rejects(lambda:replace(
        leg,length_m=None,hydraulic_diameter_m=None
    ).validate())

    # The formerly validated supercritical full engine cycle is
    # numerically unchanged by choosing a two-phase option in a
    # geometry leg where there is no liquid/vapor coexistence.
    core=hot_case()
    network=geometric_network()
    flow=0.051885
    normal=pump_and_regenerative_cooling(core,network,flow,eos=eos)
    enabled=replace(network,
        nozzle_jacket=replace(
            network.nozzle_jacket,
            two_phase_friction_model="homogeneous-mcadams"
        ),
    )
    same=pump_and_regenerative_cooling(core,enabled,flow,eos=eos)
    assert math.isclose(
        same.header_pressure_pa,normal.header_pressure_pa,rel_tol=1e-9
    )
    assert math.isclose(
        same.header_temperature_k,normal.header_temperature_k,rel_tol=1e-10
    )
    assert math.isclose(
        same.cooling.tie_tube_fraction,normal.cooling.tie_tube_fraction,
        rel_tol=1e-9
    )
    assert all(st.midpoint_quality is None for st in same.cooling.legs)

    print("NERVA homogeneous two-phase H2 coolant friction physics: PASS")
    print(f"  subcritical pressure: {p/1e6:.6g} MPa")
    print(f"  inlet/outlet quality: {result.inlet_quality:.7g} / "
          f"{result.outlet_quality:.7g}")
    print(f"  mixture Re: {result.reynolds:.7g}")
    print(f"  two-phase equilibrium Mach: {result.outlet_mach:.7g}")
    print(f"  Darcy/acceleration losses: {result.irreversible_pressure_drop_pa:.7g} / "
          f"{result.acceleration_pressure_drop_pa:.7g} Pa")
    print(f"  heat conservation residual: {result.energy_residual_w:.7g} W")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
