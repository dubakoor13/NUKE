"""Power-partition and engine-cycle closure tests for the OpenMC heating bridge.

All inputs here are synthetic conservation fixtures, NOT independent
real OpenMC transport nor a validated historical NERVA model.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import sys
import math
from pathlib import Path
import tempfile

import numpy as np

from .openmc_cycle_coupling import (
    check_reconstruction_axial_edges, map_openmc_heating_to_cycle,
)
from .resolved_transient import (
    ResolvedHeating, ResolvedParameters, read_resolved_heating,
    initial_resolved_state,
)
from .turbine_bleed_cycle import solve_turbine_bleed
from .validate_hot_lh2_cycle import hot_case
from .validate_regenerative_cooling import network
from .validate_resolved_transient import write_synthetic_reconstructions
from .validate_turbine_bleed_cycle import sample_turbine
from .run_turbine_bleed_cycle import main as run_cycle_cli


def rejects(fn) -> None:
    try:
        fn()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("nonconservative OpenMC/engine mapping accepted")


def main() -> int:
    core = hot_case()
    net = network()
    # Exact axial power profiles of the established *synthetic* full-cycle
    # test, but now they enter through the resolved OpenMC source fields.
    fw = np.broadcast_to(
        np.array([600000., 570000., 480000.])/19.,
        (1,19,3),
    ).copy()
    fd = np.broadcast_to(
        np.array([50000., 50000., 50000.])/19.,
        (1,19,3),
    ).copy()
    ts = np.array([[20000.,30000.,30000.]])
    sd = np.full((1,3),65000./3.)
    rd = np.full((1,3),85000./3.)
    unresolved = 250000.
    power = sum(float(x.sum()) for x in (fw,fd,ts,sd,rd))+unresolved
    data = ResolvedHeating(
        fw,fd,np.full((1,19),0.02),ts,sd,rd,np.array([0.05]),
        power,unresolved,
    )
    data.validate()
    mapped_core, mapped_net, ledger = map_openmc_heating_to_cycle(
        data,core,net,
        fuel_solid_transfer_fraction=1.,
        tie_solid_transfer_fraction=0.,
        tie_solid_supply_fraction=0.5,
        nozzle_wall_heat_w=110000.,
        reflector_wall_heat_w=90000.,
    )
    assert np.allclose(mapped_core.channel_wall_power_w, core.channel_wall_power_w, rtol=1e-12)
    assert np.allclose(mapped_core.channel_direct_hydrogen_power_w, core.channel_direct_hydrogen_power_w, rtol=1e-12)
    assert math.isclose(mapped_core.preheater_thermal_power_w, core.preheater_thermal_power_w, rel_tol=1e-12)
    assert math.isclose(mapped_net.total_heat_w, net.total_heat_w, rel_tol=1e-12)
    assert ledger.axial_fuel_channels_per_element == 19
    assert abs(ledger.power_balance_error_w) < 1e-7
    assert math.isclose(ledger.reactor_power_w,
                        ledger.routed_openmc_power_w +
                        ledger.retained_or_unmodeled_openmc_power_w,
                        rel_tol=1e-12)
    assert math.isclose(ledger.cycle_heating_power_w,
                        ledger.routed_openmc_power_w +
                        ledger.prescribed_nonopenmc_wall_heat_w,
                        rel_tol=1e-12)
    assert math.isclose(ledger.tie_solid_untransferred_w,80000.)

    # The mapped source fields can drive the EXISTING nonlinear
    # pump/turbine/header/core/nozzle operating-point solver unchanged.
    turbine=sample_turbine()
    reference=solve_turbine_bleed(core,turbine,cooling_network=net)
    coupled=solve_turbine_bleed(mapped_core,turbine,cooling_network=mapped_net)
    assert math.isclose(coupled.tank_mass_flow_kg_s,
                        reference.tank_mass_flow_kg_s,rel_tol=2e-8)
    assert math.isclose(coupled.chamber_pressure_pa,
                        reference.chamber_pressure_pa,rel_tol=2e-8)
    assert abs(coupled.total_cycle_energy_residual_w)<1e-3
    assert abs(coupled.shaft_power_residual_w)<1e-4

    # Moving nuclear heat from solid inventory into tie coolant must
    # conserve *total reactor power* and change only tie coolant
    # power, not fuel direct-H2 deposition or prescribed jacket heat.
    changed_core, changed_net, changed_ledger = map_openmc_heating_to_cycle(
        data,core,net,
        fuel_solid_transfer_fraction=1.,
        tie_solid_transfer_fraction=0.5,
        tie_solid_supply_fraction=0.25,
        nozzle_wall_heat_w=110000.,
        reflector_wall_heat_w=90000.,
    )
    assert math.isclose(changed_net.tie_supply.heat_power_w, 75000.)
    assert math.isclose(changed_net.tie_return.heat_power_w,115000.)
    assert np.allclose(changed_core.channel_wall_power_w,mapped_core.channel_wall_power_w,rtol=1e-12)
    assert math.isclose(changed_ledger.retained_or_unmodeled_openmc_power_w,
                        ledger.retained_or_unmodeled_openmc_power_w-40000.)
    assert abs(changed_ledger.power_balance_error_w)<1e-7

    # Never silently move fuel heating into upstream cooling.
    partial_core, partial_net, partial_ledger = map_openmc_heating_to_cycle(
        data,core,net,
        fuel_solid_transfer_fraction=0.5,
        tie_solid_transfer_fraction=0.,
        tie_solid_supply_fraction=0.5,
        nozzle_wall_heat_w=110000.,
        reflector_wall_heat_w=90000.,
    )
    assert math.isclose(partial_ledger.fuel_solid_untransferred_w,825000.)
    assert math.isclose(partial_net.total_heat_w,mapped_net.total_heat_w,rel_tol=1e-12)
    assert math.isclose(sum(partial_core.channel_wall_power_w),825000.)

    rejects(lambda:map_openmc_heating_to_cycle(
        data,core,net,fuel_solid_transfer_fraction=1.1,
        tie_solid_transfer_fraction=0.,tie_solid_supply_fraction=0.,
        nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.))
    rejects(lambda:map_openmc_heating_to_cycle(
        data,core,net,fuel_solid_transfer_fraction=1.,
        tie_solid_transfer_fraction=0.,tie_solid_supply_fraction=-0.1,
        nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.))
    rejects(lambda:map_openmc_heating_to_cycle(
        data,core,net,fuel_solid_transfer_fraction=1.,
        tie_solid_transfer_fraction=0.,tie_solid_supply_fraction=0.,
        nozzle_wall_heat_w=-1.,reflector_wall_heat_w=90000.))
    rejects(lambda:map_openmc_heating_to_cycle(
        data,replace(core,z_edges_m=(0.,0.1,0.2)),
        net,fuel_solid_transfer_fraction=1.,
        tie_solid_transfer_fraction=0.,tie_solid_supply_fraction=0.,
        nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.))

    # Reuse actual existing NPZ file format rather than bypassing the
    # reconstruction reader; all normalized mesh source partitions close.
    with tempfile.TemporaryDirectory() as temp:
        mesh, fuel, tie = write_synthetic_reconstructions(Path(temp))
        reconstructed = read_resolved_heating(fuel,tie,mesh)
        aligned = replace(core,z_edges_m=tuple(np.linspace(0.,1.,4)))
        check_reconstruction_axial_edges(fuel,tie,aligned)
        small_core,small_net,small_ledger = map_openmc_heating_to_cycle(
            reconstructed,aligned,net,
            fuel_solid_transfer_fraction=0.8,
            tie_solid_transfer_fraction=0.5,
            tie_solid_supply_fraction=0.4,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.,
        )
        assert abs(small_ledger.power_balance_error_w)<1e-7
        assert math.isclose(
            small_ledger.routed_openmc_power_w +
            small_ledger.retained_or_unmodeled_openmc_power_w,
            reconstructed.total_openmc_power_w,rel_tol=1e-12,
        )
        assert np.allclose(small_core.channel_direct_hydrogen_power_w,(32.,32.,32.),rtol=1e-12)
        assert math.isclose(small_net.tie_supply.heat_power_w,60.)
        assert math.isclose(small_net.tie_return.heat_power_w,60.)
        rejects(lambda:check_reconstruction_axial_edges(fuel,tie,core))
        rejects(lambda:map_openmc_heating_to_cycle(
            reconstructed,replace(aligned,z_edges_m=(0.,0.1,0.2)),
            net,fuel_solid_transfer_fraction=1.,
            tie_solid_transfer_fraction=0.,tie_solid_supply_fraction=0.5,
            nozzle_wall_heat_w=110000.,reflector_wall_heat_w=90000.))

    # End-to-end runner: write normalized OpenMC-style power NPZ arrays
    # and separate reconstructed instance arrays, then solve the shaft-
    # closed operating point through its public CLI. This is synthetic.
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        z = np.asarray(core.z_edges_m)
        mesh = root/"mesh_fields.npz"
        fuel_path = root/"fuel_reconstruction.npz"
        tie_path = root/"tie_reconstruction.npz"
        np.savez_compressed(
            fuel_path,
            channel_wall_axial_power_w=fw,
            channel_direct_h2_axial_power_w=fd,
            branch_mass_flow_kg_s=np.full((1,19),0.02),
            z_edges_m=z,
        )
        np.savez_compressed(
            tie_path, tie_solid_axial_power_w=ts,
            tie_supply_direct_h2_axial_power_w=sd,
            tie_return_direct_h2_axial_power_w=rd,
            tie_mass_flow_kg_s=np.array([0.05]), z_edges_m=z,
        )
        fsum=fw.sum(axis=(0,1)).reshape((1,1,3))
        dsum=fd.sum(axis=(0,1)).reshape((1,1,3))
        tsum=ts.reshape((1,1,3))
        ssum=sd.reshape((1,1,3))
        rsum=rd.reshape((1,1,3))
        full=fsum+dsum+tsum+ssum+rsum+unresolved/3.
        np.savez_compressed(
            mesh, dimension=np.array((1,1,3)),
            lower_left_cm=np.zeros(3),
            upper_right_cm=np.array((1.,1.,3.)),
            power_density_w_cm3=full,
            fuel_power_density_w_cm3=fsum,
            fuel_hydrogen_power_density_w_cm3=dsum,
            tie_power_density_w_cm3=tsum,
            tie_supply_hydrogen_power_density_w_cm3=ssum,
            tie_return_hydrogen_power_density_w_cm3=rsum,
        )
        config=root/"cycle.json"
        output=root/"coupled_output.json"
        config.write_text(json.dumps({
            "lh2_cycle":asdict(core),
            "turbine":asdict(turbine),
            "regenerative_cooling":asdict(net),
        }),encoding="utf-8")
        args=[
            "run_turbine_bleed_cycle", "--config",str(config),
            "--openmc-mesh-fields",str(mesh),
            "--openmc-fuel-reconstruction",str(fuel_path),
            "--openmc-tie-reconstruction",str(tie_path),
            "--fuel-solid-transfer-fraction","1",
            "--tie-solid-transfer-fraction","0",
            "--tie-solid-supply-fraction","0.5",
            "--nozzle-wall-heat-w","110000",
            "--reflector-wall-heat-w","90000",
            "--output",str(output),
        ]
        prev=sys.argv
        try:
            sys.argv=args
            assert run_cycle_cli()==0
        finally:
            sys.argv=prev
        doc=json.loads(output.read_text(encoding="utf-8"))
        assert doc["status"]=="PASS"
        assert doc["reconstructed_openmc_heating_mapped"] is True
        assert doc["real_openmc_transport_performed"] is False
        assert doc["power_bridge"]["nuclear_transport_run_in_this_cli"] is False
        assert abs(doc["power_bridge"]["power_ledger"]["power_balance_error_w"])<1e-7
        assert math.isclose(
            doc["point"]["chamber_pressure_pa"],
            reference.chamber_pressure_pa,rel_tol=2e-8,
        )
        assert abs(doc["point"]["total_cycle_energy_residual_w"])<1e-3

        # New resolved-implicit wall model: use the same OpenMC source
        # partitions but compute coolant heat dynamically as UA*(Ts-Tg),
        # preserving solid energy instead of prescribing transfer fractions.
        # This fixture selects an initially warm fuel solid near the already
        # admissible reference cycle, while no tie-solid heat is transferred:
        # the 80 kW tie-solid deposition accumulates in solid inventory.
        params=ResolvedParameters(
            fuel_solid_capacity_j_k=5000.,
            fuel_h2_capacity_j_k=500.,
            fuel_ua_w_k=100.,
            tie_solid_capacity_j_k=3000.,
            tie_supply_h2_capacity_j_k=600.,
            tie_return_h2_capacity_j_k=700.,
            tie_supply_ua_w_k=0.,
            tie_return_ua_w_k=0.,
        )
        parameter_file=root/"resolved_parameters.json"
        parameter_file.write_text(json.dumps(asdict(params)),encoding="utf-8")
        old=initial_resolved_state(
            data,solid_temperature_k=600.,gas_temperature_k=300.,
        )
        initial_file=root/"resolved_initial.npz"
        np.savez_compressed(
            initial_file,
            fuel_solid_t_k=old.fuel_solid_t_k,
            fuel_h2_t_k=old.fuel_h2_t_k,
            tie_solid_t_k=old.tie_solid_t_k,
            tie_supply_t_k=old.tie_supply_t_k,
            tie_return_t_k=old.tie_return_t_k,
            unresolved_energy_j=np.array([old.unresolved_energy_j]),
        )
        derived_output=root/"implicit_output.json"
        implicit_args=[
            "run_turbine_bleed_cycle", "--config",str(config),
            "--openmc-mesh-fields",str(mesh),
            "--openmc-fuel-reconstruction",str(fuel_path),
            "--openmc-tie-reconstruction",str(tie_path),
            "--openmc-transfer-mode","resolved-implicit",
            "--resolved-thermal-parameters",str(parameter_file),
            "--resolved-thermal-state",str(initial_file),
            "--resolved-dt-s","0.1",
            "--resolved-inlet-temperature-k","300",
            "--nozzle-wall-heat-w","110000",
            "--reflector-wall-heat-w","90000",
            "--output",str(derived_output),
        ]
        prior=sys.argv
        try:
            sys.argv=implicit_args
            assert run_cycle_cli()==0
        finally:
            sys.argv=prior
        derived=json.loads(derived_output.read_text(encoding="utf-8"))
        assert derived["status"]=="PASS"
        assert derived["power_bridge"]["transfer_mode"]=="resolved-implicit"
        dynamic=derived["power_bridge"]["power_ledger"]
        assert abs(dynamic["nuclear_power_balance_error_w"])<1e-6
        assert dynamic["solid_energy_storage_rate_w"]!=0.
        assert math.isclose(
            dynamic["reactor_power_w"],
            dynamic["nuclear_heat_to_coolant_w"]+
            dynamic["solid_energy_storage_rate_w"]+
            dynamic["unresolved_power_w"],rel_tol=1e-11,
        )
        report=derived["power_bridge"]["resolved_transient"]
        assert report["full_conjugate_cycle_closed"] is False
        assert report["thermal_flow_not_iterated_with_engine"] is True
        assert report["fuel_mass_flow_relative_mismatch"]>=0.
        assert report["tie_mass_flow_relative_mismatch"]>=0.
        assert abs(derived["point"]["total_cycle_energy_residual_w"])<1e-3

        # Distinct tie-supply gas inlet must appear in reported thermal
        # boundary metadata and preserve the full engine first-law closure.
        distinct_output=root/"distinct_inlet.json"
        previous=sys.argv
        try:
            sys.argv=implicit_args[:-2]+[
                "--resolved-tie-inlet-temperature-k","301",
                "--output",str(distinct_output),
            ]
            assert run_cycle_cli()==0
        finally:
            sys.argv=previous
        distinct=json.loads(distinct_output.read_text(encoding="utf-8"))
        boundary=distinct["power_bridge"]["resolved_transient"]
        assert boundary["tie_supply_gas_inlet_temperature_k"]==301.
        assert boundary["thermal_inlet_temperature_k"]==300.
        assert boundary["fuel_and_tie_have_single_shared_inlet_temperature"] is False
        assert boundary["tie_supply_cryogenic_phase_model_present"] is False
        assert boundary["fuel_header_temperature_feedback_enabled"] is False
        assert abs(distinct["point"]["total_cycle_energy_residual_w"])<1e-3

        # Warm-header feedback requires the outer flow fixed point, which
        # must have its own independent tie inlet; no fake convergence.
        previous=sys.argv
        try:
            sys.argv=implicit_args[:-2]+[
                "--couple-fuel-header-temperature",
                "--resolved-tie-inlet-temperature-k","330",
                "--output",str(root/"invalid_no_outer.json"),
            ]
            rejects(lambda:run_cycle_cli())
        finally:
            sys.argv=previous
        assert not (root/"invalid_no_outer.json").exists()

        # The real shaft-cycle CLI must never falsely declare an outer
        # fixed point on one iteration: enforce strict two-iteration
        # nonconvergence, or reject an infeasible hot-core solution.
        previous=sys.argv
        try:
            sys.argv=implicit_args[:-2]+[
                "--resolved-flow-feedback",
                "--flow-feedback-iterations","2",
                "--flow-feedback-tolerance","1e-10",
                "--flow-feedback-profile-tol-k","1e-10",
                "--output",str(root/"unconverged.json"),
            ]
            rejects(lambda:run_cycle_cli())
        finally:
            sys.argv=previous
        assert not (root/"unconverged.json").exists()

        # Feedback is forbidden in the fraction-mode source mapper.
        previous=sys.argv
        try:
            sys.argv=args[:-2]+[
                "--resolved-flow-feedback",
                "--output",str(root/"illegal_mode.json"),
            ]
            rejects(lambda:run_cycle_cli())
        finally:
            sys.argv=previous

        # Supplying transient fields and old fractions simultaneously must
        # be rejected before solving the engine, preventing a double source.
        prior=sys.argv
        try:
            sys.argv=implicit_args[:-2]+[
                "--fuel-solid-transfer-fraction","0.8",
                "--output",str(root/"should_not_exist.json"),
            ]
            rejects(lambda:run_cycle_cli())
        finally:
            sys.argv=prior

    print("NERVA OpenMC reconstructed heating -> engine/cooling power closure: PASS")
    print(f"  reactor heating: {ledger.reactor_power_w:.8g} W")
    print(f"  routed nuclear heating: {ledger.routed_openmc_power_w:.8g} W")
    print(f"  retained/unmodeled nuclear heating: {ledger.retained_or_unmodeled_openmc_power_w:.8g} W")
    print(f"  externally prescribed wall transfer: {ledger.prescribed_nonopenmc_wall_heat_w:.8g} W")
    print(f"  cycle first law residual: {coupled.total_cycle_energy_residual_w:.8g} W")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
