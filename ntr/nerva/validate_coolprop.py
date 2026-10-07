"""Optional CoolProp smoke test for the hydrogen property backend."""

from __future__ import annotations

from .hydrogen_properties import CoolPropHydrogenModel


def main() -> int:
    model = CoolPropHydrogenModel()

    low = model.state(300.0, 5.0e6)
    high = model.state(1000.0, 5.0e6)

    for state in (low, high):
        assert state.density_kg_m3 > 0.0
        assert state.cp_j_kg_k > 0.0
        assert state.dynamic_viscosity_pa_s > 0.0
        assert state.thermal_conductivity_w_m_k > 0.0
        assert state.prandtl > 0.0

    assert low.density_kg_m3 > high.density_kg_m3

    print("NERVA CoolProp hydrogen backend smoke test: PASS")
    print(f"  rho(300 K, 5 MPa): {low.density_kg_m3:.6f} kg/m3")
    print(f"  rho(1000 K, 5 MPa): {high.density_kg_m3:.6f} kg/m3")
    print(f"  cp(1000 K, 5 MPa): {high.cp_j_kg_k:.6f} J/kg/K")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
