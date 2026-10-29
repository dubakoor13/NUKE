"""Stage69 structural-material heat capacity and first-law verification.

All masses, cp tables and conductances below are synthetic verification
inputs. Exact whole-core Stage68 energy allocation does not prove contact
geometry between the seven structural categories and the Stage58 fuel/H2
thermal states; no such unqualified contact is manufactured here.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .fuel_cutcell_nonlinear_enthalpy import TemperatureTable
from .openmc_full_core_energy_ledger import read_openmc_full_core_energy_ledger
from .validate_stage68_full_core_energy_ledger import (
    fixture as stage68_fixture, IDS, RATE, SHA,
)
from .structural_enthalpy_network import (
    STRUCTURAL_NAMES, N_STRUCT,
    build_structural_thermal_network,
    allocate_full_core_structural_heating,
    advance_structural_enthalpy,
)


def make_model(*, varying=False, contact=0., bath=0., masses=None):
    cp={}
    for index,name in enumerate(STRUCTURAL_NAMES):
        value=400.+25*index
        cp[name]=TemperatureTable(
            [250.,500.,900.,1500.],
            ([value,1.4*value,2.1*value,3.*value] if varying else [value]*4),
        )
    g=np.zeros((N_STRUCT,N_STRUCT))
    g[0,1]=g[1,0]=contact
    return build_structural_thermal_network(
        mass_kg=(np.ones(N_STRUCT)*12 if masses is None else masses),
        cp_j_kg_k=cp,intermaterial_w_k=g,bath_w_k=bath,
    )


class StructuralEnthalpyEnergyTests(unittest.TestCase):
    def test_direct_heating_matches_constant_cp_adiabatic_analytic(self):
        model=make_model()
        t0=np.ones(N_STRUCT)*400
        p=np.array([100.,2.,3.,4.,5.,6.,7.])
        dt=20.
        r=advance_structural_enthalpy(
            model,t0,dt_s=dt,deposited_power_w=p)
        analytic=t0+dt*p/(model.mass_kg*
                                 np.array([c.values[0] for c in model.cp_j_kg_k]))
        np.testing.assert_allclose(r.temperature_k,analytic,rtol=0,atol=5e-13)
        np.testing.assert_allclose(r.stored_energy_change_j,dt*p,
                                   rtol=0,atol=5e-9)
        self.assertLess(abs(r.total_first_law_residual_j),5e-9)
        self.assertEqual(r.total_bath_export_j,0.)

    def test_internal_contact_heat_is_antisymmetric_and_keeps_total_stored(self):
        model=make_model(contact=360.)
        t0=np.ones(N_STRUCT)*400
        t0[0]=600.
        r=advance_structural_enthalpy(
            model,t0,dt_s=15.,
            deposited_power_w=np.zeros(N_STRUCT))
        self.assertLess(r.temperature_k[0],t0[0])
        self.assertGreater(r.temperature_k[1],t0[1])
        np.testing.assert_allclose(r.temperature_k[2:],t0[2:],
                                   atol=2e-12,rtol=0)
        self.assertAlmostEqual(float(r.internal_net_outflow_j[0]),
                               -float(r.internal_net_outflow_j[1]),
                               delta=3e-10)
        self.assertAlmostEqual(float(r.internal_net_outflow_j[2:].sum()),
                               0.,delta=1e-12)
        self.assertAlmostEqual(r.total_stored_energy_change_j,0.,delta=5e-10)
        self.assertLess(r.relative_first_law_residual,2e-11)

    def test_constant_cp_implicit_two_material_exchange_matches_closed_form(self):
        m=make_model(contact=500.)
        t=np.ones(N_STRUCT)*500.
        t[0]=600.
        t[1]=350.
        dt=10.
        c0=m.mass_kg[0]*m.cp_j_kg_k[0].values[0]
        c1=m.mass_kg[1]*m.cp_j_kg_k[1].values[0]
        a=np.array([[c0/dt+500.,-500.],[-500.,c1/dt+500.]])
        exact=np.linalg.solve(a,[c0/dt*t[0],c1/dt*t[1]])
        r=advance_structural_enthalpy(
            m,t,dt_s=dt,deposited_power_w=np.zeros(N_STRUCT))
        np.testing.assert_allclose(r.temperature_k[:2],exact,
                                   rtol=0,atol=5e-13)

    def test_reservoir_outflow_is_external_not_internal_energy(self):
        model=make_model(bath=np.array([170.,0.,0.,0.,0.,0.,0.]))
        t0=np.ones(N_STRUCT)*500.
        p=np.zeros(N_STRUCT)
        p[0]=110.
        dt=8.
        r=advance_structural_enthalpy(
            model,t0,dt_s=dt,deposited_power_w=p,
            bath_temperature_k=300.)
        cap=model.mass_kg[0]*model.cp_j_kg_k[0].values[0]
        expected=(cap*t0[0]/dt+110.+170.*300.)/(cap/dt+170.)
        self.assertAlmostEqual(r.temperature_k[0],expected,delta=1e-12)
        self.assertGreater(r.total_bath_export_j,0.)
        self.assertAlmostEqual(
            r.total_stored_energy_change_j+r.total_bath_export_j,
            r.total_deposited_energy_j,delta=1e-9)

    def test_varying_cp_uses_exact_table_integral_not_cp_delta(self):
        m=make_model(varying=True)
        q=np.zeros(N_STRUCT)
        q[0]=3000.
        t0=np.ones(N_STRUCT)*400.
        dt=6.
        r=advance_structural_enthalpy(
            m,t0,dt_s=dt,deposited_power_w=q)
        cp=m.cp_j_kg_k[0]
        true=12*(float(cp.integral(r.temperature_k[0]))-
                 float(cp.integral(t0[0])))
        self.assertAlmostEqual(true,q[0]*dt,delta=2e-8)
        linear=12*float(cp.at(t0[0]))*(r.temperature_k[0]-t0[0])
        self.assertGreater(abs(true-linear),5.)
        self.assertLess(r.relative_first_law_residual,2e-11)

    def test_nonlinear_contact_with_varying_cp_preserves_energy(self):
        m=make_model(varying=True,contact=1000.)
        t=np.array([650.,310.,450.,450.,450.,450.,450.])
        r=advance_structural_enthalpy(
            m,t,dt_s=12.,deposited_power_w=np.zeros(N_STRUCT))
        self.assertLess(r.temperature_k[0],650.)
        self.assertGreater(r.temperature_k[1],310.)
        self.assertLess(abs(r.total_first_law_residual_j),3e-8)
        self.assertLess(np.max(np.abs(r.per_material_first_law_residual_j)),3e-8)

    def test_exact_stage68_structural_source_allocation_matches_full_core(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/"synthetic.h5"
            stage68_fixture(file)
            ledger=read_openmc_full_core_energy_ledger(
                file,expected_material_ids=IDS,
                source_rate_particles_per_s=RATE,
                source_rate_basis="SYNTHETIC whole-core watts, never one fuel cell",
                nuclear_data_sha256=SHA)
        sources=allocate_full_core_structural_heating(ledger)
        self.assertEqual(len(sources.structural_power_w),7)
        self.assertAlmostEqual(float(sources.structural_power_w.sum()),
                               ledger.nonthermal_material_power_w,delta=1e-12)
        self.assertAlmostEqual(
            sources.fuel_power_w+sources.hydrogen_power_w+
            sources.structural_power_w.sum(),
            sources.whole_core_deposition_w,delta=1e-12)
        self.assertEqual(sources.source_statepoint_sha256,
                         ledger.statepoint_sha256)
        self.assertLess(abs(sources.allocation_residual_w),2e-12)

    def test_stage68_real_source_watts_stored_only_in_seven_structure_nodes(self):
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/"synthetic.h5"
            stage68_fixture(file)
            l=read_openmc_full_core_energy_ledger(
                file,expected_material_ids=IDS,
                source_rate_particles_per_s=RATE,
                source_rate_basis="SYNTHETIC whole-core input only",
                nuclear_data_sha256=SHA)
        src=allocate_full_core_structural_heating(l)
        m=make_model()
        r=advance_structural_enthalpy(
            m,np.ones(N_STRUCT)*450.,
            dt_s=9.,deposited_power_w=src.structural_power_w)
        self.assertAlmostEqual(r.total_stored_energy_change_j,
                               9.*l.nonthermal_material_power_w,
                               delta=5e-9)
        self.assertAlmostEqual(r.total_deposited_energy_j,
                               9.*l.nonthermal_material_power_w,delta=2e-12)
        self.assertNotAlmostEqual(
            r.total_deposited_energy_j,9.*l.material_total_w,delta=1e-4)

    def test_repeated_source_and_cooling_steps_close_global_first_law(self):
        m=make_model(varying=True,contact=100.,bath=5.)
        temp=np.ones(N_STRUCT)*470.
        aggregate=np.zeros(N_STRUCT)
        export=0.
        deposit=0.
        for step in range(18):
            q=np.arange(1,N_STRUCT+1,dtype=float)*(10+step)
            r=advance_structural_enthalpy(
                m,temp,dt_s=2.5,deposited_power_w=q,
                bath_temperature_k=340.)
            aggregate+=r.stored_energy_change_j
            export+=r.total_bath_export_j
            deposit+=r.total_deposited_energy_j
            temp=r.temperature_k
        self.assertAlmostEqual(float(aggregate.sum())+export,
                               deposit,delta=3e-7)

    def test_missing_bath_reservoir_rejected(self):
        m=make_model(bath=2.)
        with self.assertRaisesRegex(ValueError,"bath temperature required"):
            advance_structural_enthalpy(
                m,np.ones(N_STRUCT)*400.,dt_s=1.,
                deposited_power_w=np.zeros(N_STRUCT))

    def test_bad_structural_network_contacts_fail_closed(self):
        m=make_model()
        g=np.zeros((N_STRUCT,N_STRUCT))
        g[0,1]=2.
        cp={k:v for k,v in zip(m.names,m.cp_j_kg_k)}
        with self.assertRaisesRegex(ValueError,"symmetric"):
            build_structural_thermal_network(
                mass_kg=m.mass_kg,cp_j_kg_k=cp,intermaterial_w_k=g)
        g[1,0]=-2.
        with self.assertRaisesRegex(ValueError,"symmetric"):
            build_structural_thermal_network(
                mass_kg=m.mass_kg,cp_j_kg_k=cp,intermaterial_w_k=g)
        with self.assertRaisesRegex(ValueError,"structural masses"):
            build_structural_thermal_network(
                mass_kg=np.ones(N_STRUCT-1),cp_j_kg_k=cp,
                intermaterial_w_k=np.zeros((N_STRUCT,N_STRUCT)))

    def test_invalid_sources_timestep_and_temperature_rejected(self):
        m=make_model()
        t=np.ones(N_STRUCT)*400.
        for source in (np.full(N_STRUCT,-1.),np.full(N_STRUCT,float("nan")),
                       np.ones(6)):
            with self.assertRaisesRegex(ValueError,"deposited watts"):
                advance_structural_enthalpy(
                    m,t,dt_s=1.,deposited_power_w=source)
        with self.assertRaisesRegex(ValueError,"step size"):
            advance_structural_enthalpy(
                m,t,dt_s=0.,deposited_power_w=np.ones(N_STRUCT))
        with self.assertRaisesRegex(ValueError,"qualified table range"):
            advance_structural_enthalpy(
                m,np.full(N_STRUCT,200.),dt_s=1.,
                deposited_power_w=np.zeros(N_STRUCT))

    def test_nonlinear_source_outside_material_table_domain_fails_closed(self):
        m=make_model(varying=True)
        t=np.full(N_STRUCT,600.)
        q=np.zeros(N_STRUCT)
        q[0]=3e10
        with self.assertRaises((ValueError,ArithmeticError)):
            advance_structural_enthalpy(
                m,t,dt_s=300.,deposited_power_w=q)

    def test_zero_forcing_equilibrium_remains_at_rest(self):
        m=make_model(contact=700.)
        t=np.ones(N_STRUCT)*440.
        r=advance_structural_enthalpy(
            m,t,dt_s=50.,deposited_power_w=np.zeros(N_STRUCT))
        np.testing.assert_array_equal(t,r.temperature_k)
        self.assertEqual(r.total_deposited_energy_j,0.)
        self.assertLess(abs(r.total_first_law_residual_j),1e-11)


if __name__=="__main__":
    unittest.main(verbosity=2)
