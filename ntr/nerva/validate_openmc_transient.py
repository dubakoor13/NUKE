"""Conservation, mapping, convergence, metadata and failure-mode validation."""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import numpy as np

from .heritage_transient import ThermalCell, ThermalState, advance_thermal
from .openmc_transient import (
    Branch, OpenMCPowerMap, initial_state, advance_openmc_transient,
    advance_with_transport_feedback, power_map_from_npz,
)


def _fields(path: Path, *, overlap: bool = False) -> None:
    dim = (2, 2, 4)
    power = np.full(dim, 62.5)    # total = 1000 W at 1cm3 per voxel
    fuel = power*0.60
    tie = power*0.15
    direct_f = power*0.10
    direct_t = power*(0.20 if overlap else 0.05)
    np.savez_compressed(
        path, dimension=np.array(dim), lower_left_cm=np.zeros(3),
        upper_right_cm=np.array(dim), power_density_w_cm3=power,
        fuel_power_density_w_cm3=fuel, tie_power_density_w_cm3=tie,
        fuel_hydrogen_power_density_w_cm3=direct_f,
        tie_hydrogen_power_density_w_cm3=direct_t,
    )


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)/"fixture.npz"
        _fields(p)
        pm = power_map_from_npz(p, 1000, source_kind="synthetic")
        assert math.isclose(sum(pm.fuel_solid_w), 600)
        assert math.isclose(sum(pm.fuel_h2_direct_w), 100)
        assert math.isclose(sum(pm.tie_solid_w), 150)
        assert math.isclose(sum(pm.tie_h2_direct_w), 50)
        assert math.isclose(pm.unresolved_w, 100)
        assert math.isclose(pm.modeled_power_w, 900)
        assert math.isclose(pm.prescribed_power_w, 1000)
        assert pm.source_sha256
        # Double counting rejected, not renormalized.
        _fields(p, overlap=True)
        try:
            power_map_from_npz(p, 1000)
        except ValueError:
            pass
        else:
            raise AssertionError("double-counted heating accepted")
        _fields(p)

        fuel = Branch(10000., 1000., 100., 0.02, 14000.)
        tie = Branch(3000., 600., 20., 0.005, 14000.)
        initial = initial_state(4, 600., 300.)
        # Explicit dt stays beneath stability limit.
        s = advance_openmc_transient(pm, initial, fuel, tie, 300., 0.1)
        assert math.isclose(s.reactor_power_w, 1000)
        assert math.isclose(s.modeled_power_w, 900)
        assert math.isclose(s.gas_direct_power_w, 150)
        assert math.isclose(s.state.unresolved_energy_j, 10)
        assert abs(s.energy_residual_j) < 1e-8
        assert s.outlet_fuel_k > 300.
        # Step 2 stores further unresolved power; energy balance still closes.
        s2 = advance_openmc_transient(pm, s.state, fuel, tie, 300., 0.1)
        assert math.isclose(s2.state.unresolved_energy_j, 20)
        assert abs(s2.energy_residual_j) < 1e-8

        # Direct coolant energy MUST enter gas rather than fuel solid.
        solo = (ThermalCell(1000, 1000, 0, 0, 500),)
        r = advance_thermal(solo, ThermalState((400.,), (300.,)), 300.,
                            0, 14000, 1.)
        assert math.isclose(r.state.solid_t_k[0], 400)
        assert math.isclose(r.state.gas_t_k[0], 300.5)
        assert abs(r.energy_balance_error_j) < 1e-8

        # Closed Picard fixed point from a mock tagged transport callback.
        # Tagged mock is NUMERICAL ONLY, never actual OpenMC verification.
        calls: list[tuple[float,float]] = []
        def mock(fuel_temp, h2_temp):
            calls.append((fuel_temp, h2_temp))
            # slight material-dependent shift between fuel and tie; global
            # heating stays fixed and unresolved compartment self-adjusts.
            frac = 1 + (fuel_temp-600.)*0.00001
            return replace(
                pm, source_kind="openmc-statepoint",
                fuel_solid_w=tuple(q*frac for q in pm.fuel_solid_w),
                tie_solid_w=tuple(q-(frac-1)*pm.fuel_solid_w[i]
                                   for i,q in enumerate(pm.tie_solid_w)),
            )
        result = advance_with_transport_feedback(
            mock, initial, fuel, tie, 300., 0.1,
            max_iterations=5, temperature_tolerance_k=0.5
        )
        assert result.converged and result.iterations >= 2
        assert len(calls) == result.iterations
        assert abs(result.step.energy_residual_j) < 1e-8

        # Explicit nonconvergence and inadmissible synthetic feedback.
        try:
            advance_with_transport_feedback(
                lambda a,b: pm, initial, fuel, tie, 300., 0.1,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("synthetic feedback masqueraded as OpenMC")
        try:
            advance_with_transport_feedback(mock, initial, fuel, tie,
                                            300., 0.1, max_iterations=2,
                                            temperature_tolerance_k=1e-15)
        except RuntimeError:
            pass
        else:
            raise AssertionError("unconverged transport feedback accepted")
    print("OpenMC-to-NERVA transient mapping, energy and feedback validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
