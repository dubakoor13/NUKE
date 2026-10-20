"""Real CoolProp tie transient -> regenerative hydraulic fixed-point tests.

Use actual H2 EOS and solved parallel cooling branch hydraulics, with
synthetic/OpenMC-shaped reactor power sources. There is NO real transport
run, full pump/turbine/nozzle closure, or historical NERVA validation.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .cryogenic_tie_enthalpy import (
    CryogenicTieParameters, CryogenicTieState, advance_cryogenic_tie,
)
from .cryogenic_tie_hydraulic_coupling import couple_cryogenic_tie_hydraulics
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating
from .validate_resolved_transient import write_synthetic_reconstructions
from .validate_hot_lh2_cycle import hot_case
from .validate_regenerative_cooling import network


def rejects(f):
    try:
        f()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("invalid thermal-hydraulic case did not reject")


def physical_inlet(eos, total_mdot):
    core=hot_case()
    t=core.tank_temperature_k
    p=core.tank_pressure_pa
    rise=core.pump_shutoff_pressure_rise_pa-(
        core.pump_head_quadratic_pa_s2_kg2*total_mdot**2
    )
    assert rise>0
    pump_pressure=p+rise
    h1=eos.tp("Hmass",t,p)
    s1=eos.tp("Smass",t,p)
    h2=h1+(eos.sp("Hmass",s1,pump_pressure)-h1)/core.pump_isentropic_efficiency
    pressure=pump_pressure-core.feed_loss_quadratic_pa_s2_kg2*total_mdot**2
    assert pressure>p
    return pressure,h2


def main()->int:
    eos=HydrogenEOS()
    total_flow=0.051885
    pressure,hin=physical_inlet(eos,total_flow)
    with tempfile.TemporaryDirectory() as temp:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(temp))
        original=read_resolved_heating(fuel,tie,mesh)
        heating=replace(
            original,tie_flow_kg_s=np.full(original.tie_flow_kg_s.shape,0.015),
        )
        heating.validate()
        params=CryogenicTieParameters(
            pressure_pa=pressure*0.99,
            solid_capacity_j_k=3000.,
            supply_inventory_kg=0.0004,
            return_inventory_kg=0.0004,
            supply_ua_w_k=0.8,
            return_ua_w_k=0.8,
        )
        p0=params.pressure_pa
        h_initial=eos.tp("Hmass",24.,p0)
        state=CryogenicTieState(
            np.full_like(heating.tie_solid_w,150.),
            np.full_like(heating.tie_solid_w,h_initial),
            np.full_like(heating.tie_solid_w,h_initial),
        )
        result=couple_cryogenic_tie_hydraulics(
            heating,state,params,network(),total_flow,pressure,hin,0.1,
            nozzle_wall_heat_w=110000.,
            reflector_wall_heat_w=90000.,
            max_iterations=18,
            relative_flow_tolerance=0.004,
            relative_pressure_tolerance=0.004,
            relative_source_tolerance=0.004,
            relaxation=0.6,
            eos=eos,
        )
        assert result.iterations>=2
        assert len(result.diagnostics)==result.iterations
        assert result.relative_flow_residual<=0.004
        assert result.relative_pressure_residual<=0.004
        assert result.relative_heat_source_change<=0.004
        assert result.cooling_point.tie_tube_mass_flow_kg_s>0
        assert math.isclose(
            result.cooling_point.tie_tube_mass_flow_kg_s+
            result.cooling_point.nozzle_reflector_mass_flow_kg_s,
            total_flow,rel_tol=1e-12,
        )
        assert math.isclose(
            result.heating.total_tie_flow_kg_s,
            result.cooling_point.tie_tube_mass_flow_kg_s,
            rel_tol=0.004,
        )
        assert abs(result.cooling_point.total_energy_residual_w)<1e-3
        assert abs(result.cooling_point.split_pressure_residual_pa)<1.
        assert abs(result.nuclear_energy_power_residual_w)<1e-4
        assert abs(result.tie_step.energy_residual_j)<1e-4
        assert abs(result.outlet_enthalpy_difference_j_kg)<0.01
        assert not result.pressure_field_resolved
        assert not result.hydrogen_mass_inventory_updated
        assert not result.full_engine_cycle_converged
        assert not result.actual_openmc_transport_executed
        assert np.all(result.tie_step.supply_density_kg_m3>0)
        assert np.all(result.tie_step.return_density_kg_m3>0)
        assert np.allclose(
            result.heating.tie_flow_kg_s/
            result.heating.total_tie_flow_kg_s,
            heating.tie_flow_kg_s/heating.total_tie_flow_kg_s,
            rtol=1e-12,atol=1e-13,
        )

        # The actual delivered hydraulic heat must be corrected for the
        # signed hydrogen storage term to avoid double-counting
        # transient energy inside an otherwise steady leg balance.
        supplied=float(np.sum(result.tie_step.supply_wall_power_w)+
                       np.sum(heating.tie_supply_h2_w))
        returned=float(np.sum(result.tie_step.return_wall_power_w)+
                       np.sum(heating.tie_return_h2_w))
        hydrogen_storage=result.tie_step.hydrogen_storage_rate_w
        expected_leg=(supplied+returned-hydrogen_storage)
        delivered=(result.effective_cooling_network.tie_supply.heat_power_w+
                   result.effective_cooling_network.tie_return.heat_power_w)
        assert math.isclose(delivered,expected_leg,
                            rel_tol=1e-9,abs_tol=0.001)
        assert math.isclose(
            result.tie_step.nuclear_deposition_w,
            delivered+hydrogen_storage+
            result.tie_step.solid_storage_rate_w,
            rel_tol=1e-8,abs_tol=0.001,
        )
        assert math.isclose(
            result.effective_cooling_network.total_heat_w,
            delivered+200000.,rel_tol=1e-10,
        )
        # Recompute ONE transient from the same old state using the
        # converged feedback pressure/flow; never advance 18 times.
        final_p=result.diagnostics[-1].thermal_pressure_pa
        replay=advance_cryogenic_tie(
            result.heating,state,replace(params,pressure_pa=final_p),
            hin,0.1,eos=eos,
        )
        assert np.allclose(
            replay.state.supply_enthalpy_j_kg,
            result.tie_step.state.supply_enthalpy_j_kg,
            rtol=1e-12,atol=1e-5,
        )
        assert np.allclose(
            replay.state.return_enthalpy_j_kg,
            result.tie_step.state.return_enthalpy_j_kg,
            rtol=1e-12,atol=1e-5,
        )
        assert all(
            abs(row.nuclear_energy_power_residual_w)<1e-3 and
            abs(row.outlet_enthalpy_difference_j_kg)<0.05
            for row in result.diagnostics
        )

        # Fail closed rather than reporting only pressure OR mass-flow
        # convergence. Hard upper bound of 2 iterations is insufficient.
        rejects(lambda:couple_cryogenic_tie_hydraulics(
            heating,state,params,network(),total_flow,pressure,hin,0.1,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
            max_iterations=2,relative_flow_tolerance=1e-8,
            relative_pressure_tolerance=1e-8,
            relative_source_tolerance=1e-8,eos=eos,
        ))
        rejects(lambda:couple_cryogenic_tie_hydraulics(
            heating,state,params,network(),total_flow,pressure,hin,0.1,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
            relaxation=0.,eos=eos,
        ))
        rejects(lambda:couple_cryogenic_tie_hydraulics(
            heating,state,params,network(),total_flow,pressure,hin,-0.1,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
            eos=eos,
        ))
        rejects(lambda:couple_cryogenic_tie_hydraulics(
            heating,state,params,network(),total_flow,pressure,hin,0.1,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
            max_iterations=1,eos=eos,
        ))
        rejects(lambda:couple_cryogenic_tie_hydraulics(
            heating,state,replace(params,pressure_pa=pressure*1.02),
            network(),total_flow,pressure,hin,0.1,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
            eos=eos,
        ))

    print("NERVA real-fluid cryogenic tie / regenerative hydraulic coupling: PASS")
    print(f"  iterations={result.iterations}")
    print(f"  flow error={result.relative_flow_residual:.9g}")
    print(f"  pressure error={result.relative_pressure_residual:.9g}")
    print(f"  source update={result.relative_heat_source_change:.9g}")
    print(f"  tie nuclear heat={result.tie_step.nuclear_deposition_w:.9g} W")
    print(f"  tie advected heat={delivered:.9g} W")
    print(f"  tie fluid storage={hydrogen_storage:.9g} W")
    print(f"  solid energy storage={result.tie_step.solid_storage_rate_w:.9g} W")
    print(f"  integrated nuclear energy residual={result.nuclear_energy_power_residual_w:.9g} W")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
