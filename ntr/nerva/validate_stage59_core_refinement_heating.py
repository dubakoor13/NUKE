"""Stage59 acceptance: axial OpenMC score energy and Stage58 refinements.

ALL source scores, rates and material properties used here are SYNTHETIC;
no hot neutron transport data are read, validated, or normalized from a
real OpenMC HDF5 statepoint by these tests.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from .config import NervaConfig
from .fuel_cutcell_nonlinear_enthalpy import _NonlinearSystem
from .hydrogen_transient_fvm import primitive
from .validate_stage58_fuel_hydrogen_coupling import (
    coupled,initial,solve,
)
from .openmc_fuel_axial_heat_handoff import (
    EV_TO_J,OpenMCAxialFuelHeating,build_axial_overlap_m,
    project_openmc_axial_fuel_heating,check_heating_for_coupled_core,
)

SYNTH_SHA="a"*64
NUCDATA_SHA="b"*64


def input_heating(*,edges=None,neutron=None,photon=None,rate=1e19,unresolved=3.):
    if edges is None:
        edges=np.array([0.,.33,.99,1.32])
    if neutron is None:
        neutron=np.array([1.,2.,4.])
    if photon is None:
        photon=np.array([1.,2.,4.])
    return OpenMCAxialFuelHeating(
        axial_edges_m=edges,
        fuel_neutron_ev_per_source=neutron,
        fuel_photon_ev_per_source=photon,
        source_rate_particles_per_s=rate,
        statepoint_sha256=SYNTH_SHA,nuclear_data_sha256=NUCDATA_SHA,
        source_rate_basis="SYNTHETIC arbitrary particle rate for numerical tests only",
        unresolved_nonfuel_ev_per_source=unresolved)


class ConservationAndSourceEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=coupled(film=0.,nz=4,nx=20,ny=24)

    def test_exact_nonuniform_axial_overlap_and_power(self):
        src=input_heating()
        result=project_openmc_axial_fuel_heating(self.m.solid,src)
        powers=np.array([2.,4.,8.])*src.source_rate_particles_per_s*EV_TO_J
        np.testing.assert_allclose(
            result.source_axial_bin_w,powers,rtol=1e-14)
        expected=np.array([powers[0],powers[1]/2,powers[1]/2,powers[2]])
        np.testing.assert_allclose(
            result.mapped_axial_bin_w,expected,atol=1e-12,rtol=0)
        np.testing.assert_allclose(
            result.fuel_deposition_w.sum(axis=1),expected,
            atol=1e-12,rtol=0)
        self.assertAlmostEqual(result.mapped_fuel_power_w,
                               float(np.sum(powers)),delta=1e-12)
        self.assertLess(abs(result.fuel_power_balance_residual_w),1e-12)
        self.assertAlmostEqual(
            result.unresolved_nonfuel_power_w,
            3*src.source_rate_particles_per_s*EV_TO_J,delta=1e-12)
        self.assertEqual(
            result.transverse_approximation,
            "uniform_per_actual_fuel_only_volume")

    def test_every_fuel_cell_power_proportional_to_exact_volume(self):
        r=project_openmc_axial_fuel_heating(self.m.solid,input_heating())
        nz=self.m.nzone
        volume=self.m.solid.solid_volume_m3
        volumetric=r.fuel_deposition_w/volume
        for z in range(nz):
            self.assertAlmostEqual(
                float(np.max(volumetric[z])-np.min(volumetric[z])),
                0.,delta=1e-8)
        self.assertAlmostEqual(
            float(np.sum(r.transverse_fuel_weights)),1.,delta=2e-15)
        self.assertTrue(np.all(r.transverse_fuel_weights>0.))

    def test_source_refinement_and_target_refinement_preserve_total(self):
        z_source=np.array([0.,.05,.4,.6,1.32])
        target=np.array([0.,.1,.2,.3,.4,.8,1.32])
        mat=build_axial_overlap_m(z_source,target)
        np.testing.assert_allclose(
            mat.sum(axis=0),np.diff(z_source),atol=3e-15,rtol=0.)
        np.testing.assert_allclose(
            mat.sum(axis=1),np.diff(target),atol=3e-15,rtol=0.)
        source_w=np.array([3.,7.,11.,13.])
        target_w=mat@(source_w/np.diff(z_source))
        self.assertAlmostEqual(float(target_w.sum()),
                               float(source_w.sum()),delta=2e-14)

    def test_neutron_and_photon_power_retained_separately(self):
        src=input_heating(neutron=np.array([5.,2.,1.]),
                          photon=np.array([0.,0.,4.]))
        r=project_openmc_axial_fuel_heating(self.m.solid,src)
        self.assertAlmostEqual(
            r.fuel_neutron_power_w,
            8*src.source_rate_particles_per_s*EV_TO_J,delta=1e-12)
        self.assertAlmostEqual(
            r.fuel_photon_power_w,
            4*src.source_rate_particles_per_s*EV_TO_J,delta=1e-12)
        self.assertAlmostEqual(r.mapped_fuel_power_w,
            r.fuel_neutron_power_w+r.fuel_photon_power_w,delta=1e-12)

    def test_wrong_origin_geometry_and_missing_evidence_fails_closed(self):
        with self.assertRaises(ValueError):
            build_axial_overlap_m([0.,.66,1.32],[0.,1.31])
        with self.assertRaises(ValueError):
            build_axial_overlap_m([0.,.66,.66,1.32],[0.,1.32])
        with self.assertRaises(ValueError):
            build_axial_overlap_m([.1,1.32],[0.,1.32])
        bad=input_heating(rate=0)
        with self.assertRaises(ValueError):
            project_openmc_axial_fuel_heating(self.m.solid,bad)
        bad=input_heating()
        from dataclasses import replace
        with self.assertRaises(ValueError):
            project_openmc_axial_fuel_heating(
                self.m.solid,replace(bad,statepoint_sha256="not-a-sha"))
        with self.assertRaises(ValueError):
            project_openmc_axial_fuel_heating(
                self.m.solid,replace(bad,source_rate_basis=""))
        with self.assertRaises(ValueError):
            project_openmc_axial_fuel_heating(
                self.m.solid,replace(bad,fuel_photon_ev_per_source=[2.,4.]))
        with self.assertRaises(ValueError):
            project_openmc_axial_fuel_heating(
                self.m.solid,replace(bad,fuel_neutron_ev_per_source=[-1.,2.,4.]))

    def test_projected_heat_is_direct_stage58_fuel_watt_input(self):
        m=self.m
        source=input_heating()
        mapping=project_openmc_axial_fuel_heating(m.solid,source)
        q=check_heating_for_coupled_core(mapping,m)
        old=initial(m,tf=400.,th=400.)
        # No wall-film exchange so prescribed volume-weighted deposition
        # is the only energy added anywhere in the coupled model.
        r=solve(m,old,dt=1e-5,q=q)
        self.assertAlmostEqual(
            r.fuel_deposition_j,
            mapping.mapped_fuel_power_w*1e-5,delta=1e-10)
        self.assertLess(abs(r.hydrogen_total_energy_change_j),1e-8)
        self.assertLess(r.relative_first_law_residual,1e-8)

    def test_wrong_coupled_mesh_does_not_reuse_incompatible_mapping(self):
        mapping=project_openmc_axial_fuel_heating(
            self.m.solid,input_heating())
        bad=coupled(film=0.,nz=2,nx=20,ny=24)
        with self.assertRaises(ValueError):
            check_heating_for_coupled_core(mapping,bad)

    def test_stale_same_shape_heating_geometry_is_rejected(self):
        from dataclasses import replace
        mapping=project_openmc_axial_fuel_heating(
            self.m.solid,input_heating())
        # The same [nz,nfuel] shape is insufficient to establish that
        # projected watts belong to this exact physical z/area mesh.
        other_edges=np.array([0.,.2,.5,.9,1.32])
        other=replace(
            self.m,solid=replace(self.m.solid,axial_edges_m=other_edges))
        with self.assertRaisesRegex(ValueError,"different fuel geometry"):
            check_heating_for_coupled_core(mapping,other)
        other_area=self.m.solid.transverse.fuel_areas_cm2.copy()
        other_area[0]*=1.0001
        other=replace(
            self.m,solid=replace(self.m.solid,
                transverse=replace(self.m.solid.transverse,
                    fuel_areas_cm2=other_area)))
        with self.assertRaisesRegex(ValueError,"different fuel geometry"):
            check_heating_for_coupled_core(mapping,other)


class IndependentCoupledRefinementTests(unittest.TestCase):
    def test_timestep_refinement_of_nontrivial_sealed_exchange(self):
        m=coupled(film=1400.,nz=1,nx=20,ny=24)
        old=initial(m,tf=630.,th=320.)
        total=.00012

        def propagate(parts):
            state=old
            for _ in range(parts):
                state=solve(m,state,dt=total/parts).state
            return state
        full=propagate(1)
        half=propagate(2)
        quarter=propagate(4)
        reference=propagate(8)
        def distance(candidate):
            a=float(np.mean(candidate.fuel_temperature_k))
            b=float(np.mean(primitive(m.fluid,candidate.hydrogen).temperature_k))
            ar=float(np.mean(reference.fuel_temperature_k))
            br=float(np.mean(primitive(m.fluid,reference.hydrogen).temperature_k))
            return max(abs(a-ar),abs(b-br))
        errors=[distance(s) for s in (full,half,quarter)]
        self.assertGreater(errors[0],errors[1])
        self.assertGreater(errors[1],errors[2])
        self.assertGreater(errors[2],1e-11)

    def test_thermal_interface_geometry_refinement_moves_toward_exact_wall_limit(self):
        # The Stage55 scheme uses half-grid fuel resistance at curved
        # walls. Exact Stage53 arc geometry supplies the independent
        # limiting wall resistance when the fuel-side half-cell vanishes.
        errors=[]
        for nx,ny in ((20,24),(28,34),(40,46)):
            m=coupled(film=1400.,nz=1,nx=nx,ny=ny)
            old=initial(m,tf=500.,th=400.)
            gas=primitive(m.fluid,old.hydrogen)
            sys=_NonlinearSystem(
                m.solid,m.config,m.properties,m.solid_density_kg_m3,
                gas.density_kg_m3*m.fluid.volume_m3,
                old.fuel_temperature_k,gas.temperature_k,
                gas.temperature_k[0],
                np.zeros_like(old.fuel_temperature_k),
                np.zeros_like(gas.temperature_k),1e-5)
            _,wall=sys._operators(old.fuel_temperature_k,gas.temperature_k)
            observed=float(wall.sum())*100.
            c=m.config
            rc=c.coated_channel_radius_cm*.01
            ri=c.channel_radius_cm*.01
            kz=float(m.properties.coating_k_w_m_k.at(450.))
            h=float(m.properties.hydrogen_film_w_m2_k.at(400.))
            contact=float(np.sum(m.solid.transverse.fuel_zrc_contact_m2))
            limit=contact*100./(
                rc*math.log(rc/ri)/kz+rc/(ri*h))
            errors.append(limit-observed)
            self.assertGreater(limit,observed)
        self.assertGreater(errors[0],errors[1])
        self.assertGreater(errors[1],errors[2])

    def test_very_large_substep_request_rejected_without_accepting_partial_result(self):
        m=coupled(film=0.,nz=2,nx=20,ny=24)
        old=initial(m)
        before=old.hydrogen.conserved.copy()
        with self.assertRaises(ValueError):
            solve(m,old,dt=.02,max_substeps=1)
        np.testing.assert_array_equal(old.hydrogen.conserved,before)

    def test_coupled_energy_budget_during_externally_heated_open_flow(self):
        m=coupled(film=1300.,nz=2,nx=20,ny=24)
        old=initial(m,tf=480.,th=395.,vel=12.)
        q=4e5*m.solid.solid_volume_m3
        r=solve(m,old,dt=1.6e-5,q=q,
                bc=__import__(
                    "ntr.nerva.hydrogen_transient_fvm",
                    fromlist=["HydrogenBoundary"]).HydrogenBoundary("transmissive"))
        self.assertGreater(r.fuel_deposition_j,0.)
        self.assertGreater(float(np.sum(r.wall_heat_to_hydrogen_j)),0.)
        self.assertLess(r.relative_first_law_residual,2e-8)
        self.assertTrue(np.all(np.isfinite(r.state.hydrogen.conserved)))


if __name__=="__main__":
    unittest.main(verbosity=2)
