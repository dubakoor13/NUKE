"""Independent mathematical tests for Stage 8 thermal/neutronic boundary.

Synthetic coefficients only. No NERVA thermodynamic/cross-section data.
"""
from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
from scipy.sparse import csc_matrix

from .fx2th_thermal_energy import (
    ThermalCells,ThermalState,ThermalDeposition,advance_thermal_cells,
)
from .fx2th_thermal_feedback import advance_coupled_boundary
from .validate_fx2th_family_spacetime import model, initial


def thermals(n=1):
    return ThermalCells(
        solid_heat_capacity_j_k=np.full(n,200.),
        hydrogen_heat_capacity_j_k=np.full(n,60.),
        wall_conductance_w_k=np.full(n,3.),
        hydrogen_mass_flow_kg_s=np.full(n,.1),
        hydrogen_cp_j_kg_k=np.full(n,500.),
        inlet_temperature_k=np.full(n,300.),
    )


def temperatures(n=1):
    return ThermalState(0.,np.full(n,350.),np.full(n,330.))


def deposition(n=1,qs=10.,qh=15.,qo=5.,synthetic=True):
    return ThermalDeposition('synthetic-external-heat',
        np.full(n,qs),np.full(n,qh),np.full(n,qo),
        float(n*(qs+qh+qo)),synthetic=synthetic)


class ThermalEnergyTests(unittest.TestCase):
    def test_no_coupling_closed_system_storage_equals_input(self):
        props=ThermalCells([2.],[3.],[0.],[0.],[2.],[300.])
        s=ThermalState(0.,np.array([300.]),np.array([300.]))
        p=ThermalDeposition('synthetic',[4.],[6.],[3.],13.)
        x=advance_thermal_cells(props,s,p,1.)
        self.assertAlmostEqual(x.state.solid_temperature_k[0],302.)
        self.assertAlmostEqual(x.state.hydrogen_temperature_k[0],302.)
        self.assertAlmostEqual(x.fuel_storage_change_j,4.)
        self.assertAlmostEqual(x.hydrogen_storage_change_j,6.)
        self.assertAlmostEqual(x.unmodeled_energy_j,3.)
        self.assertAlmostEqual(x.declared_deposited_energy_j,13.)
        self.assertLess(abs(x.thermal_energy_residual_j),1e-9)

    def test_lumped_enthalpy_export_analytic(self):
        props=ThermalCells([2.],[4.],[0.],[1.],[2.],[300.])
        s=ThermalState(0.,np.array([300.]),np.array([300.]))
        p=deposition(qs=0.,qh=10.,qo=0.)
        x=advance_thermal_cells(props,s,p,1.)
        self.assertAlmostEqual(x.state.hydrogen_temperature_k[0],300.+10./6.)
        self.assertAlmostEqual(x.hydrogen_net_export_j,10./3.)
        self.assertLess(x.relative_energy_residual,1e-13)

    def test_internal_transfer_does_not_make_energy(self):
        props=ThermalCells([10.],[10.],[5.],[0.],[1.],[300.])
        s=ThermalState(0.,np.array([400.]),np.array([300.]))
        p=deposition(qs=0.,qh=0.,qo=0.)
        x=advance_thermal_cells(props,s,p,1.)
        self.assertAlmostEqual(x.state.solid_temperature_k[0],375.)
        self.assertAlmostEqual(x.state.hydrogen_temperature_k[0],325.)
        self.assertAlmostEqual(x.solid_to_hydrogen_transfer_j,250.)
        self.assertAlmostEqual(x.fuel_storage_change_j+x.hydrogen_storage_change_j,0.,places=10)

    def test_multiple_cells_sources_and_total_power(self):
        x=advance_thermal_cells(thermals(3),temperatures(3),deposition(3,30.,20.,10.),.2)
        self.assertAlmostEqual(x.declared_deposited_energy_j,36.)
        self.assertAlmostEqual(x.unmodeled_energy_j,6.)
        self.assertLess(x.relative_energy_residual,1e-11)
        self.assertEqual(x.state.solid_temperature_k.shape,(3,))

    def test_reject_missing_or_doubled_power(self):
        props=thermals(); s=temperatures()
        with self.assertRaisesRegex(ValueError,"partition"):
            advance_thermal_cells(props,s,ThermalDeposition("synthetic",[10.],[15.],[5.],60.),.2)
        with self.assertRaisesRegex(ValueError,"source ID"):
            advance_thermal_cells(props,s,ThermalDeposition("",[10.],[15.],[5.],30.),.2)

    def test_reject_invalid_physics_parameters(self):
        with self.assertRaises(ValueError):
            replace(thermals(),solid_heat_capacity_j_k=np.array([-1.]))
        with self.assertRaises(ValueError):
            replace(thermals(),hydrogen_cp_j_kg_k=np.array([0.]))
        with self.assertRaises(ValueError):
            advance_thermal_cells(thermals(),temperatures(),deposition(),0.)
        with self.assertRaises(ValueError):
            advance_thermal_cells(thermals(),ThermalState(0.,[-300.],[330.]),deposition(),.2)

    def test_energy_balance_large_flux_case(self):
        x=advance_thermal_cells(
            thermals(),temperatures(),deposition(qs=2e6,qh=1e5,qo=4e5),.1)
        self.assertLess(x.relative_energy_residual,5e-13)
        self.assertAlmostEqual(x.declared_deposited_energy_j,2.5e5)

    def test_report_does_not_equate_synthetic_to_qualified(self):
        x=advance_thermal_cells(thermals(),temperatures(),deposition(),.1)
        self.assertTrue(x.synthetic)


def providers(feedback=True):
    base=model(cells=1,groups=1)
    seen=[]
    def material(state):
        seen.append(float(state.solid_temperature_k[0]))
        factor=.002*(state.solid_temperature_k[0]-350.) if feedback else 0.
        return replace(base,integrated_loss=csc_matrix([[.8+factor]]))
    def deposition_provider(state,data):
        power=100.*float(state.flux[0,0])
        return ThermalDeposition(
            'synthetic-flux-to-power-test-rule',
            [0.8*power],[0.15*power],[0.05*power],power,
            synthetic=True
        )
    return base,material,deposition_provider,seen


class CouplingContractTests(unittest.TestCase):
    def test_feedback_converges_with_energy_and_neutron_balances(self):
        base,material,power,seen=providers()
        result=advance_coupled_boundary(initial(base),temperatures(),thermals(),.1,
                                        material,power)
        self.assertLess(result.thermal.relative_energy_residual,1e-11)
        self.assertLess(result.kinetic.global_balance_residual,1e-11)
        self.assertLess(result.fixed_point_temperature_residual_k,1e-8)
        self.assertTrue(result.synthetic)
        self.assertGreater(result.iterations,1)
        self.assertEqual(len(seen),result.iterations+1)

    def test_iterative_feedback_advances_one_time_step_only(self):
        base,material,power,_=providers()
        old=initial(base);old_t=temperatures()
        result=advance_coupled_boundary(old,old_t,thermals(),.2,material,power)
        self.assertAlmostEqual(result.kinetic.state.time_s,.2)
        self.assertAlmostEqual(result.thermal.state.time_s,.2)
        self.assertAlmostEqual(old.time_s,0.)
        np.testing.assert_allclose(old_t.solid_temperature_k,[350.])

    def test_matched_constant_callbacks_converge_to_decoupled_result(self):
        base,material,power,_=providers(False)
        coupled=advance_coupled_boundary(initial(base),temperatures(),thermals(),.1,material,power)
        from .fx2th_family_spacetime import advance_spatial_families
        direct_n=advance_spatial_families(base,initial(base),.1)
        direct_t=advance_thermal_cells(thermals(),temperatures(),power(direct_n.state,base),.1)
        np.testing.assert_allclose(coupled.kinetic.state.flux,direct_n.state.flux,rtol=1e-11)
        np.testing.assert_allclose(coupled.thermal.state.solid_temperature_k,direct_t.state.solid_temperature_k,rtol=1e-11)

    def test_nonconvergence_fails_no_false_success(self):
        base,material,power,_=providers()
        with self.assertRaisesRegex(RuntimeError,"did not converge"):
            advance_coupled_boundary(initial(base),temperatures(),thermals(),.1,
                                     material,power,max_iterations=1,
                                     temperature_tolerance_k=1e-12)

    def test_reject_temperature_and_neutron_time_mismatch(self):
        base,material,power,_=providers()
        with self.assertRaisesRegex(ValueError,"times"):
            advance_coupled_boundary(initial(base),ThermalState(.1,[350.],[330.]),
                                     thermals(),.1,material,power)

    def test_reject_changed_mesh_and_family_inputs(self):
        base,material,power,_=providers()
        def bad(state):
            if abs(float(state.solid_temperature_k[0])-350.)<1e-10:
                return base
            return replace(base,volumes_m3=np.array([9.]))
        with self.assertRaisesRegex(ValueError,"changed mesh"):
            advance_coupled_boundary(initial(base),temperatures(),thermals(),.1,bad,power)

    def test_reject_missing_power_source(self):
        base,material,_,_=providers()
        with self.assertRaises(TypeError):
            advance_coupled_boundary(initial(base),temperatures(),thermals(),.1,
                                     material,lambda s,d: {'power':10.})

    def test_reject_bad_neutron_source_or_cell_count(self):
        base,material,power,_=providers()
        with self.assertRaisesRegex(ValueError,"cell meshes"):
            advance_coupled_boundary(initial(base),temperatures(),thermals(2),.1,material,power)
        with self.assertRaisesRegex(ValueError,"neutron source"):
            advance_coupled_boundary(initial(base),temperatures(),thermals(),.1,material,power,
                                     external_neutron_source_per_m3=np.array([[2.,1.]]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
