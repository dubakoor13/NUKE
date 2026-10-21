"""Inertial face-momentum / conservative tie H2 regression with CoolProp.

Exercises real CoolProp H2 EOS and synthetic nuclear deposition; validates
finite delay from pressure step, flux against adverse pressure head,
quadratic-drag steady root, face momentum residual and BOTH conserved
rigid fluid mass and internal energy. This is not full two-phase
unsteady Euler momentum, OpenMC transport, or NASA hardware validation.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .inertial_tie import (
    InertialTieParameters,InertialTieState,advance_inertial_tie,
)
from .isochoric_tie import (
    IsochoricTieParameters,initial_isochoric_tie_state,
)
from .pressure_driven_tie import (
    PressureTieFlowParameters,advance_pressure_driven_tie,
)
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating
from .validate_resolved_transient import write_synthetic_reconstructions


def rejects(action):
    try:
        action()
    except (RuntimeError,ValueError):
        return
    raise AssertionError("inadmissible inertial/CV state accepted")


def main()->int:
    eos=HydrogenEOS()
    p0=550000.
    th=IsochoricTieParameters(
        supply_cell_volume_m3=0.001,return_cell_volume_m3=0.001,
        solid_capacity_j_k=3000.,
        supply_ua_w_k=1.,return_ua_w_k=1.,
    )
    orifice=PressureTieFlowParameters(
        thermal=th,interface_area_m2=1e-5,
        discharge_coefficient=0.7,
    )
    params=InertialTieParameters(
        thermal_hydraulic=orifice,interface_inertial_length_m=0.05,
    )
    dt=0.001
    p_in=600000.
    p_out=500000.
    h_in=eos.get("Hmass","P",p_in,"Q",0.30)
    h_out=eos.get("Hmass","P",p_out,"Q",0.30)
    with tempfile.TemporaryDirectory() as tmp:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(tmp))
        heat=read_resolved_heating(fuel,tie,mesh)
        initial=initial_isochoric_tie_state(
            heat,th,p0,quality=0.30,
            solid_temperature_k=50.,eos=eos,
        )
        face=np.zeros((heat.tie_solid_w.shape[0],
                       2*heat.tie_solid_w.shape[1]+1))
        old=InertialTieState(initial,face)
        old.validate(heat)
        step=advance_inertial_tie(
            heat,old,params,p_in,h_in,p_out,h_out,dt,eos=eos,
        )
        assert step.state.interface_mass_flow_kg_s.shape==face.shape
        assert np.all(step.state.interface_mass_flow_kg_s[:,0]>0)
        assert np.all(step.state.interface_mass_flow_kg_s[:,-1]>0)
        assert np.all(step.state.interface_mass_flow_kg_s[:,1:-1]==0.)
        assert step.maximum_momentum_residual_pa<1e-5
        assert abs(step.fluid.mass_balance_error_kg_s)<1e-9
        assert abs(step.fluid.energy_balance_error_j)<1e-5
        assert step.fluid.momentum_law_kind=="externally solved signed face momentum flux"
        assert not step.full_transient_momentum_equations
        assert not step.kinetic_energy_accounted_for
        assert math.isclose(
            step.fluid.total_mass_storage_rate_kg_s,
            step.fluid.inlet_mass_flow_kg_s-
            step.fluid.outlet_mass_flow_kg_s,
            rel_tol=1e-8,abs_tol=1e-9,
        )
        assert math.isclose(
            step.fluid.solid_storage_rate_w+
            step.fluid.hydrogen_internal_energy_storage_rate_w,
            step.fluid.nuclear_deposition_w+
            step.fluid.inlet_enthalpy_power_w-
            step.fluid.outlet_enthalpy_power_w,
            abs_tol=1e-4,rel_tol=1e-9,
        )
        assert np.allclose(
            step.driving_pressure_pa,
            step.drag_pressure_pa+step.acceleration_pressure_pa,
            atol=1e-5,rtol=1e-10,
        )

        # Unlike quasisteady orifice, the inertia face cannot jump
        # immediately to full steady mass flow at the same pressure.
        sudden=advance_pressure_driven_tie(
            heat,initial,orifice,p_in,h_in,p_out,h_out,dt,eos=eos,
        )
        assert np.all(
            step.state.interface_mass_flow_kg_s[:,0]<
            sudden.interface_mass_flow_kg_s[:,0]
        )
        assert np.all(
            step.state.interface_mass_flow_kg_s[:,-1]<
            sudden.interface_mass_flow_kg_s[:,-1]
        )

        # Iterate only FACE inertia with clamped initial EOS state to
        # verify long-time momentum relaxation to the actual orifice
        # solution. This is a controlled momentum test, not a physical
        # multi-timestep thermal loop.
        momentum_old=old
        for _ in range(35):
            trial=advance_inertial_tie(
                heat,momentum_old,params,p_in,h_in,p_out,h_out,dt,eos=eos,
            )
            momentum_old=InertialTieState(
                initial,trial.state.interface_mass_flow_kg_s,
            )
        assert np.allclose(
            trial.state.interface_mass_flow_kg_s,
            sudden.interface_mass_flow_kg_s,
            rtol=1e-4,atol=1e-8,
        )

        # Momentum resists reversal: a face already carrying positive
        # mass flow may still flow FORWARD against a new adverse dp.
        steady_positive=momentum_old
        adverse=advance_inertial_tie(
            heat,steady_positive,params,
            p_out,h_out,p_in,h_in,0.0001,eos=eos,
        )
        assert np.all(adverse.state.interface_mass_flow_kg_s[:,0]>0)
        assert np.all(adverse.state.interface_mass_flow_kg_s[:,-1]>0)
        assert np.all(adverse.driving_pressure_pa[:,0]<0)
        assert np.all(adverse.drag_pressure_pa[:,0]>0)
        assert adverse.maximum_momentum_residual_pa<1e-5
        assert abs(adverse.fluid.mass_balance_error_kg_s)<1e-9
        assert abs(adverse.fluid.energy_balance_error_j)<1e-5

        # Continue the adverse forcing with clamped fluid old state:
        # inertia dissipates, face flows finally reverse smoothly.
        trial_old=InertialTieState(
            initial,adverse.state.interface_mass_flow_kg_s
        )
        for _ in range(70):
            rev=advance_inertial_tie(
                heat,trial_old,params,
                p_out,h_out,p_in,h_in,0.0001,eos=eos,
            )
            trial_old=InertialTieState(
                initial,rev.state.interface_mass_flow_kg_s
            )
        assert np.all(rev.state.interface_mass_flow_kg_s[:,0]<0)
        assert np.all(rev.state.interface_mass_flow_kg_s[:,-1]<0)
        assert rev.maximum_momentum_residual_pa<1e-5

        # Equal reservoirs / no heating / zero face momentum ->
        # stationary mass, EOS pressure, energy, and no spurious
        # CoolProp pressure-inversion micro-flow.
        tie_power=float(
            np.sum(heat.tie_solid_w)+
            np.sum(heat.tie_supply_h2_w)+
            np.sum(heat.tie_return_h2_w)
        )
        zero=replace(
            heat,tie_solid_w=np.zeros_like(heat.tie_solid_w),
            tie_supply_h2_w=np.zeros_like(heat.tie_supply_h2_w),
            tie_return_h2_w=np.zeros_like(heat.tie_return_h2_w),
            total_openmc_power_w=heat.total_openmc_power_w-tie_power,
        )
        zero.validate()
        insulated=InertialTieParameters(
            thermal_hydraulic=replace(
                orifice,thermal=replace(
                    th,supply_ua_w_k=0.,return_ua_w_k=0.
                )
            ),interface_inertial_length_m=0.05,
        )
        h0=eos.get("Hmass","P",p0,"Q",0.30)
        static=advance_inertial_tie(
            zero,old,insulated,p0,h0,p0,h0,dt,eos=eos,
        )
        assert np.all(static.state.interface_mass_flow_kg_s==0.)
        assert np.array_equal(
            static.state.thermal.supply_mass_kg,initial.supply_mass_kg
        )
        assert np.array_equal(
            static.state.thermal.return_mass_kg,initial.return_mass_kg
        )
        assert abs(static.fluid.mass_balance_error_kg_s)<1e-9
        assert abs(static.fluid.energy_balance_error_j)<1e-5

        # Face integrator must reject invalid inputs or unresolvable
        # explicit CV mass stability without adjusting physics silently.
        rejects(lambda:advance_inertial_tie(
            heat,old,replace(params,interface_inertial_length_m=0.),
            p_in,h_in,p_out,h_out,dt,eos=eos,
        ))
        rejects(lambda:advance_inertial_tie(
            heat,old,params,2e6,h_in,p_out,h_out,dt,eos=eos,
        ))
        rejects(lambda:advance_inertial_tie(
            heat,old,params,p_in,h_in,p_out,h_out,-dt,eos=eos,
        ))
        rejects(lambda:advance_inertial_tie(
            heat,old,params,p_in,h_in,p_out,h_out,2.0,eos=eos,
        ))
        malformed=InertialTieState(initial,np.zeros((1,2)))
        rejects(lambda:advance_inertial_tie(
            heat,malformed,params,p_in,h_in,p_out,h_out,dt,eos=eos,
        ))

    print("NERVA finite-inertia tie face momentum and conserved H2 CV: PASS")
    print(f"  initial inertial inlet flux: {step.fluid.inlet_mass_flow_kg_s:.9g} kg/s")
    print(f"  quasisteady orifice inlet flux: {sudden.inlet_mass_flow_kg_s:.9g} kg/s")
    print(f"  first-step max momentum residual: {step.maximum_momentum_residual_pa:.9g} Pa")
    print(f"  first-step mass residual: {step.fluid.mass_balance_error_kg_s:.9g} kg/s")
    print(f"  first-step energy residual: {step.fluid.energy_balance_error_j:.9g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
