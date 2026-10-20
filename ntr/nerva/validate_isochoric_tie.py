"""CoolProp H2 fixed-volume tie counterflow regression; synthetic nuclear maps.

Tests use real 0.55 MPa two-phase hydrogen properties but synthetic
power distributions. They demonstrate INTERNAL-energy (m*u) storage
and EOS pressure changes at constant cell density/mass. No full
momentum/pump/nozzle or validated historical NERVA transport run.
"""
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import tempfile

import numpy as np

from .isochoric_tie import (
    IsochoricTieParameters, initial_isochoric_tie_state,
    advance_isochoric_tie,
)
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating
from .validate_resolved_transient import write_synthetic_reconstructions


def rejects(action):
    try:
        action()
    except (RuntimeError,ValueError):
        return
    raise AssertionError("invalid mass/EOS/energy case accepted")


def main() -> int:
    eos=HydrogenEOS()
    p0=550000.
    q0=0.30
    t_sat=eos.get("T","P",p0,"Q",q0)
    rho0=eos.get("Dmass","P",p0,"Q",q0)
    u0=eos.get("Umass","P",p0,"Q",q0)
    hin=eos.get("Hmass","P",p0,"Q",q0)
    assert math.isclose(hin,u0+p0/rho0,rel_tol=1e-8)
    with tempfile.TemporaryDirectory() as temporary:
        paths=write_synthetic_reconstructions(Path(temporary))
        heating=read_resolved_heating(*paths[1:],paths[0])
        shape=heating.tie_solid_w.shape
        params=IsochoricTieParameters(
            supply_cell_volume_m3=0.001,
            return_cell_volume_m3=0.0012,
            solid_capacity_j_k=3000.,
            supply_ua_w_k=1.0,
            return_ua_w_k=1.0,
            max_pressure_pa=3e6,
        )
        old=initial_isochoric_tie_state(
            heating,params,p0,quality=q0,
            solid_temperature_k=45.,eos=eos,
        )
        assert math.isclose(
            old.supply_mass_kg[0,0]/params.supply_cell_volume_m3,
            rho0,rel_tol=1e-10,
        )
        assert math.isclose(
            old.return_mass_kg[0,0]/params.return_cell_volume_m3,
            rho0,rel_tol=1e-10,
        )
        dt=0.05
        step=advance_isochoric_tie(
            heating,old,params,hin,dt,eos=eos,
        )
        assert step.max_picard_iterations>=2
        assert abs(step.energy_residual_j)<1e-4
        assert step.max_local_energy_balance_residual_w<1e-4
        assert step.max_local_mass_balance_residual_kg_s==0.
        assert np.array_equal(
            step.state.supply_mass_kg,old.supply_mass_kg
        )
        assert np.array_equal(
            step.state.return_mass_kg,old.return_mass_kg
        )
        assert np.allclose(step.supply_density_kg_m3,rho0,rtol=1e-12)
        assert np.allclose(step.return_density_kg_m3,rho0,rtol=1e-12)
        assert np.max(step.supply_pressure_pa)>p0
        assert np.max(step.return_pressure_pa)>p0
        assert np.max(step.supply_temperature_k)>=t_sat
        assert np.all(np.isfinite(step.supply_quality))
        assert np.all(np.isfinite(step.return_quality))
        # EOS h=u+p/rho, evaluated at each NEW isochoric state.
        assert np.allclose(
            step.supply_enthalpy_j_kg,
            step.state.supply_internal_energy_j_kg+
            step.supply_pressure_pa/step.supply_density_kg_m3,
            rtol=1e-12,atol=1e-7,
        )
        assert np.allclose(
            step.return_enthalpy_j_kg,
            step.state.return_internal_energy_j_kg+
            step.return_pressure_pa/step.return_density_kg_m3,
            rtol=1e-12,atol=1e-7,
        )
        assert math.isclose(
            step.solid_storage_rate_w+
            step.fluid_internal_energy_storage_rate_w,
            step.nuclear_deposition_w+
            step.inlet_enthalpy_power_w-
            step.outlet_enthalpy_power_w,
            rel_tol=1e-8,abs_tol=0.001,
        )
        # Changing the rigid volume changes density/pressure closure;
        # no hidden isobaric gas inventory is used.
        changed=replace(
            params,return_cell_volume_m3=0.0008
        )
        shifted=advance_isochoric_tie(
            heating,old,changed,hin,dt,eos=eos,
        )
        assert not np.allclose(
            shifted.return_pressure_pa,step.return_pressure_pa,
            rtol=1e-5,atol=1.,
        )
        assert abs(shifted.energy_residual_j)<0.001

        # At exact mass/temperature equilibrium with ZERO wall and
        # direct nuclear heating, EOS pressure, internal energy and
        # fluid mass remain unchanged in each control volume.
        tie_nuc=float(np.sum(heating.tie_solid_w)+
                      np.sum(heating.tie_supply_h2_w)+
                      np.sum(heating.tie_return_h2_w))
        zero=replace(
            heating,
            tie_solid_w=np.zeros_like(heating.tie_solid_w),
            tie_supply_h2_w=np.zeros_like(heating.tie_supply_h2_w),
            tie_return_h2_w=np.zeros_like(heating.tie_return_h2_w),
            total_openmc_power_w=heating.total_openmc_power_w-tie_nuc,
        )
        zero.validate()
        insulated=replace(params,supply_ua_w_k=0.,return_ua_w_k=0.)
        static=advance_isochoric_tie(
            zero,old,insulated,hin,dt,eos=eos,
        )
        assert np.allclose(static.supply_pressure_pa,p0,rtol=1e-8)
        assert np.allclose(static.return_pressure_pa,p0,rtol=1e-8)
        assert np.allclose(
            static.state.supply_internal_energy_j_kg,u0,
            rtol=1e-10,atol=1e-5,
        )
        assert np.allclose(
            static.state.return_internal_energy_j_kg,u0,
            rtol=1e-10,atol=1e-5,
        )
        assert np.allclose(static.state.solid_temperature_k,45.)
        assert abs(static.energy_residual_j)<1e-4

        # Reject nonphysical volume, initial quality, pressure limit
        # and excessive direct hydrogen energy input.
        rejects(lambda:initial_isochoric_tie_state(
            heating,params,p0,quality=1.5,
            solid_temperature_k=45.,eos=eos,
        ))
        rejects(lambda:advance_isochoric_tie(
            heating,old,replace(params,supply_cell_volume_m3=0.),
            hin,dt,eos=eos,
        ))
        rejects(lambda:advance_isochoric_tie(
            heating,old,replace(params,max_pressure_pa=200000.),
            hin,dt,eos=eos,
        ))
        rejects(lambda:advance_isochoric_tie(
            heating,old,params,hin,-dt,eos=eos,
        ))
        malformed=replace(
            old,return_mass_kg=np.zeros_like(old.return_mass_kg)
        )
        rejects(lambda:advance_isochoric_tie(
            heating,malformed,params,hin,dt,eos=eos,
        ))

    print("NERVA fixed-volume tie mass/internal-energy real-fluid closure: PASS")
    print(f"  initial H2 density: {rho0:.8g} kg/m3")
    print(f"  initial H2 pressure: {p0:.8g} Pa")
    print(f"  final supply pressure: {step.supply_pressure_pa.min():.8g} "
          f".. {step.supply_pressure_pa.max():.8g} Pa")
    print(f"  fluid internal-energy storage rate: "
          f"{step.fluid_internal_energy_storage_rate_w:.8g} W")
    print(f"  global residual: {step.energy_residual_j:.8g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
