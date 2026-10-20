"""Independent numerical checks of variable-property NERVA H2 thermodynamics."""
from __future__ import annotations
import math
from .nasa_hydrogen import (
    RU, M_H2, hydrogen_state, hydrogen_nozzle,
    power_coupled_hydrogen_nozzle
)


def main() -> int:
    # NASA-7 cp at 300 K: 14.31 kJ/(kg K); molecular H2 should be intact.
    cold = hydrogen_state(300.0, 3.1e6)
    hot = hydrogen_state(2500.0, 3.1e6)
    assert 14000 < cold.frozen_cp_j_kg_k < 14700
    assert 17300 < hot.frozen_cp_j_kg_k < 18400
    assert hot.enthalpy_j_kg > cold.enthalpy_j_kg
    assert 0 < hot.dissociation_fraction < 0.02
    assert hydrogen_state(2900.0, 3.1e6).dissociation_fraction > hot.dissociation_fraction
    assert hydrogen_state(2500.0, 6.2e6).dissociation_fraction < hot.dissociation_fraction

    # Frozen pure H2 is a closed mass/entropy/energy balance; frozen
    # composition is not allowed to change in the nozzle.
    ref = hydrogen_nozzle(450.0*6894.757293168, 2425.0, 0.02, 100.0,
                          mode="frozen", nozzle_efficiency=0.95)
    assert ref.throat.pressure_pa < ref.chamber.pressure_pa
    assert ref.exit.pressure_pa < ref.throat.pressure_pa
    assert ref.exit.temperature_k < ref.throat.temperature_k
    assert ref.throat.dissociation_fraction == 0.0
    assert abs(ref.exit.entropy_j_kg_k - ref.chamber.entropy_j_kg_k) < 1e-5
    assert abs(ref.enthalpy_to_kinetic_residual_j_kg) < 1e-5
    g_exit = ref.exit.density_kg_m3 * (2*(ref.chamber.enthalpy_j_kg - ref.exit.enthalpy_j_kg))**0.5
    assert math.isclose(ref.mass_flux_kg_m2_s/g_exit, 100.0, rel_tol=1e-8)

    doubled = hydrogen_nozzle(900.0*6894.757293168, 2425.0, 0.02, 100.0,
                               mode="frozen", nozzle_efficiency=0.95)
    assert math.isclose(doubled.mass_flow_kg_s/ref.mass_flow_kg_s,
                        2.0, rel_tol=2e-8)
    assert math.isclose(doubled.thrust_n/ref.thrust_n, 2.0, rel_tol=2e-8)
    assert math.isclose(doubled.isp_s, ref.isp_s, rel_tol=2e-8)

    eq = hydrogen_nozzle(450.0*6894.757293168, 2700.0, 0.02, 100.0,
                          mode="equilibrium", nozzle_efficiency=0.95)
    assert eq.chamber.dissociation_fraction > eq.exit.dissociation_fraction
    assert abs(eq.exit.entropy_j_kg_k - eq.chamber.entropy_j_kg_k) < 1e-5
    assert abs(eq.enthalpy_to_kinetic_residual_j_kg) < 1e-5
    assert 500 < eq.isp_s < 1100  # broad plausibility, NOT historical calibration

    # Pressure -> mdot -> T -> Isp at fixed power and throat geometry.
    low = power_coupled_hydrogen_nozzle(
        1.0e9, 3.1e6, 300.0, 0.035, 100, mode="frozen")
    high = power_coupled_hydrogen_nozzle(
        1.0e9, 6.2e6, 300.0, 0.035, 100, mode="frozen")
    assert abs(low.power_residual_w) < 1e-5*low.hydrogen_heating_power_w
    assert abs(high.power_residual_w) < 1e-5*high.hydrogen_heating_power_w
    assert low.chamber_temperature_k > high.chamber_temperature_k
    assert low.nozzle.isp_s > high.nozzle.isp_s
    assert low.nozzle.mass_flow_kg_s < high.nozzle.mass_flow_kg_s

    # Invalid states should fail closed instead of silently extrapolating NASA7.
    for kwargs in ({"temperature_k": 190, "pressure_pa": 1e6},
                   {"temperature_k": 2800, "pressure_pa": -1},
                   {"temperature_k": 3600, "pressure_pa": 1e6}):
        try:
            hydrogen_state(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid thermodynamic state accepted")

    print("NASA-7 NERVA hydrogen validation: PASS")
    print(f"pure frozen: Isp {ref.isp_s:.2f} s; mdot {ref.mass_flow_kg_s:.3f} kg/s")
    print(f"equilibrium: alpha_c {eq.chamber.dissociation_fraction:.6g}; "
          f"alpha_e {eq.exit.dissociation_fraction:.6g}")
    print(f"power coupled: Tc {low.chamber_temperature_k:.1f} -> "
          f"{high.chamber_temperature_k:.1f} K; Isp "
          f"{low.nozzle.isp_s:.1f} -> {high.nozzle.isp_s:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
