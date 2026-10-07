"""Validation of the physics-based NERVA engine/nozzle model."""

from __future__ import annotations

import math

from .historical_presets import LBF_TO_N, PSI_TO_PA
from .nerva_engine import (
    NervaGasModel,
    evaluate_nerva_nozzle,
    solve_power_coupled_nerva,
    throat_area_for_target_thrust,
)


def main() -> int:
    gas = NervaGasModel(
        gamma=1.35,
        gas_constant_j_kg_k=4124.0,
        cp_j_kg_k=14300.0,
    )

    # 1972 NERVA public reference:
    # 75 klbf, Tc 2350-2500 K, Pc 450 psia, epsilon=100,
    # published vacuum Isp 825-850 s.
    target_thrust = 75_000.0 * LBF_TO_N
    pc = 450.0 * PSI_TO_PA
    tc = 0.5 * (2350.0 + 2500.0)
    epsilon = 100.0

    throat_area = throat_area_for_target_thrust(
        target_thrust_n=target_thrust,
        chamber_pressure_pa=pc,
        chamber_temperature_k=tc,
        expansion_ratio=epsilon,
        nozzle_efficiency=0.95,
        gas=gas,
    )
    reference = evaluate_nerva_nozzle(
        chamber_pressure_pa=pc,
        chamber_temperature_k=tc,
        throat_area_m2=throat_area,
        expansion_ratio=epsilon,
        nozzle_efficiency=0.95,
        gas=gas,
    )

    assert math.isclose(reference.thrust_n, target_thrust, rel_tol=1e-12)
    assert 825.0 <= reference.isp_s <= 850.0
    assert reference.mass_flow_kg_s > 0.0
    assert reference.thrust_coefficient > 0.0
    assert reference.characteristic_velocity_m_s > 0.0
    assert reference.exit_pressure_pa < reference.chamber_pressure_pa

    # Correct pressure relationship:
    # at fixed Tc, epsilon, gas model and vacuum ambient, doubling Pc through
    # the SAME throat doubles mdot and thrust, but leaves Isp unchanged.
    low = evaluate_nerva_nozzle(
        chamber_pressure_pa=pc,
        chamber_temperature_k=tc,
        throat_area_m2=throat_area,
        expansion_ratio=epsilon,
        nozzle_efficiency=0.95,
        gas=gas,
    )
    high = evaluate_nerva_nozzle(
        chamber_pressure_pa=2.0 * pc,
        chamber_temperature_k=tc,
        throat_area_m2=throat_area,
        expansion_ratio=epsilon,
        nozzle_efficiency=0.95,
        gas=gas,
    )
    assert math.isclose(
        high.mass_flow_kg_s / low.mass_flow_kg_s,
        2.0,
        rel_tol=1e-12,
    )
    assert math.isclose(
        high.thrust_n / low.thrust_n,
        2.0,
        rel_tol=1e-12,
    )
    assert math.isclose(high.isp_s, low.isp_s, rel_tol=1e-12)

    # Reactor-power coupling:
    # for fixed reactor power and throat, raising Pc raises mdot; the same
    # thermal power is spread over more H2, so Tc and Isp decrease.
    coupled_low = solve_power_coupled_nerva(
        reactor_thermal_power_w=1000.0e6,
        chamber_pressure_pa=pc,
        throat_area_m2=throat_area,
        expansion_ratio=epsilon,
        inlet_temperature_k=300.0,
        propellant_heating_efficiency=0.95,
        nozzle_efficiency=0.95,
        gas=gas,
    )
    coupled_high = solve_power_coupled_nerva(
        reactor_thermal_power_w=1000.0e6,
        chamber_pressure_pa=2.0 * pc,
        throat_area_m2=throat_area,
        expansion_ratio=epsilon,
        inlet_temperature_k=300.0,
        propellant_heating_efficiency=0.95,
        nozzle_efficiency=0.95,
        gas=gas,
    )

    assert (
        coupled_high.nozzle.mass_flow_kg_s
        > coupled_low.nozzle.mass_flow_kg_s
    )
    assert (
        coupled_high.chamber_temperature_k
        < coupled_low.chamber_temperature_k
    )
    assert coupled_high.nozzle.isp_s < coupled_low.nozzle.isp_s

    print("Physics-based NERVA engine validation: PASS")
    print(
        f"  1972 reference: {reference.thrust_n / 1000.0:.3f} kN, "
        f"{reference.isp_s:.3f} s, "
        f"{reference.mass_flow_kg_s:.3f} kg/s"
    )
    print(
        f"  fixed-T pressure test: Pc x2 -> mdot x"
        f"{high.mass_flow_kg_s / low.mass_flow_kg_s:.3f}, thrust x"
        f"{high.thrust_n / low.thrust_n:.3f}, Isp x"
        f"{high.isp_s / low.isp_s:.6f}"
    )
    print(
        f"  fixed-power test: Tc "
        f"{coupled_low.chamber_temperature_k:.1f} -> "
        f"{coupled_high.chamber_temperature_k:.1f} K; Isp "
        f"{coupled_low.nozzle.isp_s:.1f} -> "
        f"{coupled_high.nozzle.isp_s:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
