"""Real CoolProp hydrogen verification for distributed tie Euler equations.

Source maps are SYNTHETIC reconstructions with conservative partitions.
Checks spatial momentum, total energy (INCLUDING kinetic), one physical
turnaround, friction-to-internal conversion and fail-closed acoustic CFL.
Not NASA hardware calibration, real OpenMC transport or two-fluid CHF.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .distributed_tie_euler import (
    DistributedTieParameters, initial_distributed_tie_state,
    advance_distributed_tie,
)
from .isochoric_tie import IsochoricTieParameters
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating
from .validate_resolved_transient import write_synthetic_reconstructions


def rejects(fn):
    try:
        fn()
    except (RuntimeError,ValueError):
        return
    raise AssertionError("nonphysical distributed Euler state accepted")


def assert_balances(step):
    assert abs(step.mass_residual_kg_s)<1e-9
    assert abs(step.momentum_residual_n)<1e-5
    assert abs(step.total_energy_residual_j)<1e-5
    assert step.maximum_local_mass_residual_kg_s<1e-8
    assert step.maximum_local_momentum_residual_n<1e-5
    assert step.maximum_local_total_energy_residual_w<1e-3
    assert math.isclose(
        step.fluid_internal_energy_storage_rate_w+
        step.fluid_kinetic_energy_storage_rate_w+
        step.solid_storage_rate_w,
        step.nuclear_deposition_w+
        step.inlet_total_energy_power_w-
        step.outlet_total_energy_power_w,
        abs_tol=1e-4,rel_tol=1e-8,
    )


def main()->int:
    eos=HydrogenEOS()
    p0=550000.
    q=0.30
    h0=eos.get("Hmass","P",p0,"Q",q)
    th=IsochoricTieParameters(
        supply_cell_volume_m3=0.001,
        return_cell_volume_m3=0.001,
        solid_capacity_j_k=3000.,
        supply_ua_w_k=0.8,
        return_ua_w_k=0.8,
    )
    cfg=DistributedTieParameters(
        thermal=th,flow_area_m2=0.001,
        darcy_friction_factor=0.03,
        turnaround_loss_coefficient=0.6,
        max_cfl=0.45,
    )
    p_in,p_out=565000.,535000.
    h_in=eos.get("Hmass","P",p_in,"Q",q)
    h_out=eos.get("Hmass","P",p_out,"Q",q)
    dt=1e-5
    with tempfile.TemporaryDirectory() as td:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(td))
        heating=read_resolved_heating(fuel,tie,mesh)
        nt,nz=heating.tie_solid_w.shape
        state=initial_distributed_tie_state(
            heating,cfg,p0,h0,
            solid_temperature_k=45.,eos=eos,
        )
        assert state.mass_kg.shape==(nt,2*nz)
        assert state.momentum_kg_m_s.shape==(nt,2*nz)
        assert np.all(state.mass_kg>0)
        step=advance_distributed_tie(
            heating,state,cfg,
            p_in,h_in,p_out,h_out,dt,eos=eos,
        )
        assert_balances(step)
        assert step.interface_mass_flux_kg_s.shape==(nt,2*nz+1)
        assert step.interface_momentum_flux_n.shape==(nt,2*nz+1)
        assert step.interface_total_energy_flux_w.shape==(nt,2*nz+1)
        assert step.maximum_cfl<=cfg.max_cfl
        assert step.momentum_transport_distributed
        assert step.total_energy_including_kinetic_conserved
        assert step.homogeneous_equilibrium_only
        assert np.all(np.isfinite(step.quality))
        assert np.all(step.density_kg_m3>0)
        assert np.all(step.pressure_pa>0)
        assert np.max(np.abs(step.velocity_m_s))>0
        assert not np.array_equal(
            step.state.momentum_kg_m_s,state.momentum_kg_m_s
        )
        assert np.max(np.abs(step.pressure_pa-p0))>1e-5
        assert np.allclose(
            step.specific_enthalpy_j_kg,
            step.internal_energy_j_kg+
            step.pressure_pa/step.density_kg_m3,
            atol=1e-7,rtol=1e-12,
        )
        assert np.allclose(
            step.state.total_energy_j,
            step.state.mass_kg*(
                step.internal_energy_j_kg+
                0.5*step.velocity_m_s**2
            ),atol=1e-7,rtol=1e-12,
        )
        # Conservation across the shared routed supply/return turn:
        # exactly one interface is used on both neighboring cells.
        turn=nz
        assert np.all(np.isfinite(step.interface_mass_flux_kg_s[:,turn]))
        # Two consecutive physical timesteps propagate pressure
        # and velocity without resetting the conserved momentum field.
        second=advance_distributed_tie(
            heating,step.state,cfg,
            p_in,h_in,p_out,h_out,dt,eos=eos,
        )
        assert_balances(second)
        assert np.any(abs(second.velocity_m_s-step.velocity_m_s)>1e-8)
        assert second.maximum_cfl<=cfg.max_cfl

        # In a uniform P,h H2 state with NO source or wall heat,
        # Fmom=p*A on ALL faces and a quiescent domain stays quiescent.
        tie_source=float(
            np.sum(heating.tie_solid_w)+
            np.sum(heating.tie_supply_h2_w)+
            np.sum(heating.tie_return_h2_w)
        )
        zero=replace(
            heating,tie_solid_w=np.zeros_like(heating.tie_solid_w),
            tie_supply_h2_w=np.zeros_like(heating.tie_supply_h2_w),
            tie_return_h2_w=np.zeros_like(heating.tie_return_h2_w),
            total_openmc_power_w=heating.total_openmc_power_w-tie_source,
        )
        zero.validate()
        no_wall=replace(
            cfg,thermal=replace(
                th,supply_ua_w_k=0.,return_ua_w_k=0.
            )
        )
        uniform=advance_distributed_tie(
            zero,state,no_wall,p0,h0,p0,h0,dt,eos=eos,
        )
        assert_balances(uniform)
        assert np.max(abs(uniform.velocity_m_s))<1e-5
        assert np.allclose(
            uniform.state.mass_kg,state.mass_kg,rtol=1e-7,atol=1e-9
        )
        assert np.allclose(
            uniform.state.total_energy_j,state.total_energy_j,
            rtol=1e-7,atol=1e-7,
        )
        assert np.allclose(
            uniform.state.solid_temperature_k,state.solid_temperature_k,
            atol=1e-12,
        )

        # Wall drag and turnaround losses reduce momentum; their
        # dissipation converts fluid kinetic -> INTERNAL energy. Since
        # total energy and face flux are unchanged by local wall drag
        # for the SAME old state, the total energy remains identical.
        moving=initial_distributed_tie_state(
            heating,cfg,p0,h0,
            solid_temperature_k=45.,velocity_m_s=8.,eos=eos,
        )
        with_drag=advance_distributed_tie(
            zero,moving,replace(
                cfg,thermal=no_wall.thermal,
                darcy_friction_factor=0.20,
                turnaround_loss_coefficient=4.,
            ),p0,h0,p0,h0,dt,eos=eos,
        )
        without_drag=advance_distributed_tie(
            zero,moving,replace(
                cfg,thermal=no_wall.thermal,
                darcy_friction_factor=0.,
                turnaround_loss_coefficient=0.,
            ),p0,h0,p0,h0,dt,eos=eos,
        )
        assert_balances(with_drag)
        assert_balances(without_drag)
        assert np.allclose(
            with_drag.state.total_energy_j,
            without_drag.state.total_energy_j,
            rtol=1e-12,atol=1e-8,
        )
        assert np.any(
            abs(with_drag.state.momentum_kg_m_s)<
            abs(without_drag.state.momentum_kg_m_s)
        )
        assert np.sum(with_drag.state.mass_kg*
                      with_drag.internal_energy_j_kg)>np.sum(
            without_drag.state.mass_kg*
            without_drag.internal_energy_j_kg
        )
        assert np.allclose(
            with_drag.interface_total_energy_flux_w,
            without_drag.interface_total_energy_flux_w,
            atol=1e-8,rtol=1e-12,
        )

        # Reject unstable acoustic CFL, nonphysical geometric input
        # and invalid pre-existing momentum/total-energy state.
        rejects(lambda:advance_distributed_tie(
            heating,state,cfg,
            p_in,h_in,p_out,h_out,0.1,eos=eos,
        ))
        rejects(lambda:advance_distributed_tie(
            heating,state,replace(cfg,flow_area_m2=-1),
            p_in,h_in,p_out,h_out,dt,eos=eos,
        ))
        rejects(lambda:advance_distributed_tie(
            heating,state,replace(cfg,darcy_friction_factor=-1),
            p_in,h_in,p_out,h_out,dt,eos=eos,
        ))
        rejects(lambda:advance_distributed_tie(
            heating,state,cfg,p_in,h_in,p_out,h_out,-dt,eos=eos,
        ))
        rejects(lambda:advance_distributed_tie(
            heating,replace(state,mass_kg=np.zeros_like(state.mass_kg)),
            cfg,p_in,h_in,p_out,h_out,dt,eos=eos,
        ))

    print("NERVA distributed compressible tie mass/momentum/TOTAL-energy: PASS")
    print(f"  maximum acoustic CFL: {step.maximum_cfl:.8g}")
    print(f"  fluid internal-energy rate: {step.fluid_internal_energy_storage_rate_w:.8g} W")
    print(f"  fluid kinetic-energy rate: {step.fluid_kinetic_energy_storage_rate_w:.8g} W")
    print(f"  global mass residual: {step.mass_residual_kg_s:.8g} kg/s")
    print(f"  global momentum residual: {step.momentum_residual_n:.8g} N")
    print(f"  global total-energy residual: {step.total_energy_residual_j:.8g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
