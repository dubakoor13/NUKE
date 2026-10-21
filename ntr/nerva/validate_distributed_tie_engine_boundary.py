"""Shaft-closed pump/regenerator boundary → transient distributed tie regression.

Uses actual solved engine-side CoolProp pump and pressure-balanced
parallel regenerative branches, synthetic nuclear power deposition and
the real-fluid Rusanov tie transient. Tests that steady mass flow is
a *diagnostic*, not forced into a genuinely unsteady fluid state.
Does not run OpenMC or establish NASA hardware fidelity.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .distributed_tie_engine_boundary import (
    engine_tie_reservoirs,advance_engine_conditioned_distributed_tie,
)
from .distributed_tie_euler import (
    DistributedTieParameters, initial_distributed_tie_state,
    advance_distributed_tie,
)
from .isochoric_tie import IsochoricTieParameters
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating
from .turbine_bleed_cycle import solve_turbine_bleed
from .validate_hot_lh2_cycle import hot_case
from .validate_regenerative_cooling import network
from .validate_resolved_transient import write_synthetic_reconstructions
from .validate_turbine_bleed_cycle import sample_turbine


def rejects(action) -> None:
    try:
        action()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("physically inconsistent tie/shaft boundary accepted")


def main()->int:
    eos=HydrogenEOS()
    core=hot_case()
    net=network()
    turbine=sample_turbine()
    point=solve_turbine_bleed(
        core,turbine,eos=eos,cooling_network=net,
    )
    boundary=engine_tie_reservoirs(point)
    hydraulic=point.regenerative_cooling
    assert hydraulic is not None
    assert math.isclose(
        boundary.steady_tie_branch_mass_flow_kg_s,
        hydraulic.tie_tube_mass_flow_kg_s,rel_tol=1e-12,
    )
    assert math.isclose(
        boundary.upstream_pressure_pa,
        hydraulic.legs[2].inlet_pressure_pa,rel_tol=1e-12,
    )
    assert math.isclose(
        boundary.upstream_stagnation_enthalpy_j_kg,
        hydraulic.legs[2].inlet_stagnation_enthalpy_j_kg,rel_tol=1e-12,
    )
    assert math.isclose(
        boundary.downstream_pressure_pa,
        hydraulic.legs[3].outlet_pressure_pa,rel_tol=1e-12,
    )
    assert math.isclose(
        boundary.downstream_backflow_stagnation_enthalpy_j_kg,
        hydraulic.legs[3].outlet_stagnation_enthalpy_j_kg,rel_tol=1e-12,
    )
    assert math.isclose(
        boundary.steady_total_pump_mass_flow_kg_s,
        point.tank_mass_flow_kg_s,rel_tol=1e-12,
    )
    assert boundary.upstream_pressure_pa>boundary.downstream_pressure_pa
    assert abs(point.shaft_power_residual_w)<1e-3
    assert abs(hydraulic.split_pressure_residual_pa)<1.

    with tempfile.TemporaryDirectory() as temp:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(temp))
        heating=read_resolved_heating(fuel,tie,mesh)
        params=DistributedTieParameters(
            thermal=IsochoricTieParameters(
                supply_cell_volume_m3=0.0001,
                return_cell_volume_m3=0.0001,
                solid_capacity_j_k=3000.,
                supply_ua_w_k=0.4,
                return_ua_w_k=0.4,
            ),
            flow_area_m2=0.0002,
            darcy_friction_factor=0.04,
            turnaround_loss_coefficient=0.6,
        )
        old=initial_distributed_tie_state(
            heating,params,
            boundary.upstream_pressure_pa,
            boundary.upstream_stagnation_enthalpy_j_kg,
            solid_temperature_k=35.,eos=eos,
        )
        dt=1e-8
        coupling=advance_engine_conditioned_distributed_tie(
            heating,old,params,point,dt,eos=eos,
        )
        manual=advance_distributed_tie(
            heating,old,params,
            boundary.upstream_pressure_pa,
            boundary.upstream_stagnation_enthalpy_j_kg,
            boundary.downstream_pressure_pa,
            boundary.downstream_backflow_stagnation_enthalpy_j_kg,
            dt,eos=eos,
        )
        t=coupling.transient
        assert np.array_equal(
            t.state.mass_kg,manual.state.mass_kg,
        )
        assert np.array_equal(
            t.state.momentum_kg_m_s,manual.state.momentum_kg_m_s,
        )
        assert np.array_equal(
            t.state.total_energy_j,manual.state.total_energy_j,
        )
        assert t.maximum_cfl<params.max_cfl
        assert abs(t.mass_residual_kg_s)<1e-8
        assert abs(t.momentum_residual_n)<1e-4
        assert abs(t.total_energy_residual_j)<1e-5
        assert abs(coupling.transient_mass_accumulation_kg_s-
                   (coupling.inlet_transient_mass_flow_kg_s-
                    coupling.outlet_transient_mass_flow_kg_s))<1e-8
        assert coupling.steady_tie_mass_flow_reference_kg_s>0
        assert math.isclose(
            coupling.inlet_mass_flow_reference_residual_kg_s,
            coupling.inlet_transient_mass_flow_kg_s-
            coupling.steady_tie_mass_flow_reference_kg_s,
            abs_tol=1e-12,
        )
        assert math.isclose(
            coupling.outlet_mass_flow_reference_residual_kg_s,
            coupling.outlet_transient_mass_flow_kg_s-
            coupling.steady_tie_mass_flow_reference_kg_s,
            abs_tol=1e-12,
        )
        assert not coupling.transient_flow_matches_steady_engine
        assert not coupling.full_engine_feedback_converged
        assert not coupling.external_openmc_transport_executed
        # Different initial pressure perturbation affects transient
        # momentum, without changing the shaft reference operating point.
        disturbed=initial_distributed_tie_state(
            heating,params,
            boundary.downstream_pressure_pa,
            boundary.downstream_backflow_stagnation_enthalpy_j_kg,
            solid_temperature_k=35.,eos=eos,
        )
        another=advance_engine_conditioned_distributed_tie(
            heating,disturbed,params,point,dt,eos=eos,
        )
        assert not np.array_equal(
            another.transient.interface_mass_flux_kg_s,
            coupling.transient.interface_mass_flux_kg_s,
        )
        assert another.steady_tie_mass_flow_reference_kg_s==(
            coupling.steady_tie_mass_flow_reference_kg_s
        )
        assert abs(another.transient.total_energy_residual_j)<1e-5

        # Explicit transient condition rejects impossible CFL; no
        # hidden clipping or setting dynamic flux=steady branch flow.
        rejects(lambda:advance_engine_conditioned_distributed_tie(
            heating,old,params,point,0.1,eos=eos,
        ))
        rejects(lambda:advance_engine_conditioned_distributed_tie(
            heating,old,params,point,-dt,eos=eos,
        ))

    # Reject missing, corrupt, and nonpressure-balanced cooling.
    rejects(lambda:engine_tie_reservoirs(
        replace(point,regenerative_cooling=None),
    ))
    rejects(lambda:engine_tie_reservoirs(
        replace(point,regenerative_cooling=replace(
            hydraulic,tie_tube_mass_flow_kg_s=0.,
        )),
    ))
    rejects(lambda:engine_tie_reservoirs(
        replace(point,regenerative_cooling=replace(
            hydraulic,common_header_pressure_pa=1.,
        )),
    ))

    print("NERVA shaft-cycle-conditioned distributed cryogenic tie: PASS")
    print(f"  steady engine tie branch flow: {coupling.steady_tie_mass_flow_reference_kg_s:.9g} kg/s")
    print(f"  transient tie inlet flux: {coupling.inlet_transient_mass_flow_kg_s:.9g} kg/s")
    print(f"  transient tie outlet flux: {coupling.outlet_transient_mass_flow_kg_s:.9g} kg/s")
    print(f"  H2 mass accumulation: {coupling.transient_mass_accumulation_kg_s:.9g} kg/s")
    print(f"  mass residual: {t.mass_residual_kg_s:.9g} kg/s")
    print(f"  total energy residual: {t.total_energy_residual_j:.9g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
