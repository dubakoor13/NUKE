"""Real CoolProp phase-aware tie-tube implicit counterflow validation.

Synthetic OpenMC-shape fields, but actual CoolProp hydrogen liquid-vapor
coexistence at 0.55 MPa. These tests do NOT run OpenMC nor calibrate NERVA.
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
from .lh2_feed_cycle import HydrogenEOS
from .resolved_transient import read_resolved_heating
from .validate_resolved_transient import write_synthetic_reconstructions


def rejects(fn) -> None:
    try:
        fn()
    except (ValueError,RuntimeError):
        return
    raise AssertionError("invalid cryogenic fluid state accepted")


def main() -> int:
    eos=HydrogenEOS()
    p=0.55e6
    hliq=eos.get("Hmass","P",p,"Q",0.)
    hvap=eos.get("Hmass","P",p,"Q",1.)
    assert hvap>hliq
    latent=hvap-hliq
    hin=hliq+0.30*latent
    with tempfile.TemporaryDirectory() as temp:
        mesh,fuel,tie=write_synthetic_reconstructions(Path(temp))
        heating=read_resolved_heating(fuel,tie,mesh)
        params=CryogenicTieParameters(
            pressure_pa=p,
            solid_capacity_j_k=3000.,
            supply_inventory_kg=0.005,
            return_inventory_kg=0.005,
            supply_ua_w_k=0.8,
            return_ua_w_k=0.8,
        )
        shape=heating.tie_solid_w.shape
        old=CryogenicTieState(
            np.full(shape,400.),
            np.full(shape,hin),
            np.full(shape,hin),
        )
        dt=0.1
        step=advance_cryogenic_tie(
            heating,old,params,hin,dt,eos=eos,
        )
        assert step.max_picard_iterations>=2
        assert np.all(step.supply_density_kg_m3>0)
        assert np.all(step.return_density_kg_m3>0)
        assert np.all((step.supply_quality>0)&(step.supply_quality<1))
        assert np.all((step.return_quality>0)&(step.return_quality<1))
        assert np.all(step.supply_wall_power_w>0)
        assert np.all(step.return_wall_power_w>0)
        assert np.max(step.state.solid_temperature_k)<400.
        assert np.mean(step.state.supply_enthalpy_j_kg)>hin
        assert np.all(step.state.return_enthalpy_j_kg[:,0]>hin)
        # Isobaric boiling: temperature is constant at saturation even
        # as enthalpy and phase quality change (latent-heat physics).
        saturation=eos.get("T","P",p,"Q",0.)
        assert np.max(abs(step.supply_temperature_k-saturation))<1e-6
        assert np.max(abs(step.return_temperature_k-saturation))<1e-6
        assert float(np.ptp(step.supply_quality))>1e-5
        assert float(np.ptp(step.return_quality))>1e-5
        assert abs(step.energy_residual_j)<1e-5
        assert step.max_local_solid_residual_w<1e-4
        assert step.max_local_hydrogen_residual_w<1e-4
        assert math.isclose(
            step.solid_storage_rate_w+step.hydrogen_storage_rate_w,
            step.nuclear_deposition_w+step.inlet_enthalpy_power_w-
            step.outlet_enthalpy_power_w,
            rel_tol=1e-8,abs_tol=1e-4,
        )
        # Nuclear power deposited directly in H2 is NOT wall
        # transfer; the solid balance uses only wall UA and Cs*dTs/dt.
        assert math.isclose(
            step.solid_storage_rate_w,
            float(np.sum(heating.tie_solid_w)-
                  np.sum(step.supply_wall_power_w)-
                  np.sum(step.return_wall_power_w)),
            rel_tol=1e-8,abs_tol=1e-4,
        )

        # Direct deposition raises H2 quality while fuel/tie solid
        # source powers stay unchanged and global energy still closes.
        return_power=heating.tie_return_h2_w.copy()
        return_power[:,2]+=1000.
        extra=float(np.sum(return_power-heating.tie_return_h2_w))
        hotter=replace(
            heating,tie_return_h2_w=return_power,
            total_openmc_power_w=heating.total_openmc_power_w+extra,
        )
        hotter.validate()
        raised=advance_cryogenic_tie(
            hotter,old,params,hin,dt,eos=eos,
        )
        assert np.mean(raised.return_quality)>np.mean(step.return_quality)
        assert abs(raised.energy_residual_j)<1e-5
        assert math.isclose(raised.nuclear_deposition_w,
                            step.nuclear_deposition_w+extra,rel_tol=1e-12)

        # Zero UA, zero tie nuclear source: no invented heat or
        # spontaneous phase change, including 100% return of supply.
        zero=replace(
            heating,
            tie_solid_w=np.zeros_like(heating.tie_solid_w),
            tie_supply_h2_w=np.zeros_like(heating.tie_supply_h2_w),
            tie_return_h2_w=np.zeros_like(heating.tie_return_h2_w),
            total_openmc_power_w=(
                heating.total_openmc_power_w -
                np.sum(heating.tie_solid_w) -
                np.sum(heating.tie_supply_h2_w) -
                np.sum(heating.tie_return_h2_w)
            ),
        )
        zero.validate()
        no_wall=replace(params,supply_ua_w_k=0.,return_ua_w_k=0.)
        unchanged=advance_cryogenic_tie(
            zero,old,no_wall,hin,dt,eos=eos,
        )
        assert np.allclose(unchanged.state.solid_temperature_k,400.,atol=1e-9)
        assert np.allclose(unchanged.state.supply_enthalpy_j_kg,hin,atol=1e-6)
        assert np.allclose(unchanged.state.return_enthalpy_j_kg,hin,atol=1e-6)
        assert abs(unchanged.energy_residual_j)<1e-5

        # The old/new fluid H2 inventories and pressure must be
        # checked, not guessed from a nominal constant heat capacity.
        rejects(lambda:advance_cryogenic_tie(
            heating,replace(old,supply_enthalpy_j_kg=np.zeros((1,3))),
            params,hin,dt,eos=eos,
        ))
        rejects(lambda:advance_cryogenic_tie(
            heating,old,replace(params,return_inventory_kg=0.),
            hin,dt,eos=eos,
        ))
        rejects(lambda:advance_cryogenic_tie(
            heating,old,params,hin,-0.1,eos=eos,
        ))
        rejects(lambda:advance_cryogenic_tie(
            heating,old,params,1e12,dt,eos=eos,
        ))
        extreme=replace(
            heating,
            tie_return_h2_w=np.full_like(heating.tie_return_h2_w,1e10),
            total_openmc_power_w=(
                heating.total_openmc_power_w-
                float(np.sum(heating.tie_return_h2_w))+
                float(np.size(heating.tie_return_h2_w))*1e10
            ),
        )
        extreme.validate()
        rejects(lambda:advance_cryogenic_tie(
            extreme,old,params,hin,dt,eos=eos,
        ))

    print("NERVA real-fluid cryogenic tie-tube enthalpy/counterflow: PASS")
    print(f"  pressure: {p/1e6:.6g} MPa")
    print(f"  latent heat: {latent:.8g} J/kg")
    print(f"  quality at initial state: 0.3")
    print(f"  supply quality range: {step.supply_quality.min():.8g} .. {step.supply_quality.max():.8g}")
    print(f"  return quality range: {step.return_quality.min():.8g} .. {step.return_quality.max():.8g}")
    print(f"  nuclear tie heat: {step.nuclear_deposition_w:.8g} W")
    print(f"  first-law residual: {step.energy_residual_j:.8g} J")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
