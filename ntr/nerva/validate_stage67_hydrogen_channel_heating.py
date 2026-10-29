"""Stage67 physical-cell H2 heating remapping and thermal energy verification.

These test statepoints are SYNTHETIC HDF5 in native OpenMC layout and use
externally declared source rate/element identity. There is no actual
OpenMC transport execution, evaluated data qualification, nor evidence
that recycled lattice cell IDs identify a unique fuel element instance.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from .layout import coolant_channel_positions
from .hydrogen_transient_fvm import primitive
from .validate_stage58_fuel_hydrogen_coupling import coupled,initial,solve
from .validate_stage60_native_hdf5_heating import (
    synthetic_statepoint, DATA_SHA,
)
from .openmc_fuel_axial_heat_handoff import EV_TO_J
from .openmc_native_hydrogen_channel_heating import (
    read_native_19_channel_hydrogen_heating,
)
from .openmc_native_spatial_fuel_heating import project_native_3d_fuel_heating
from .validate_stage65_coupled_nuclear_heating import (
    coupled_native_fixture,load,
)


CELL_IDS=tuple(range(301,320))
RATE=1e18


def fixture(path: Path, model, *, photon=True, reverse=False,
            nxy=3,nz=4,channel=8,source_index=2,
            outside=False):
    """Write native-moment HDF5 with one heated channel and known location."""
    synthetic_statepoint(path,dimension=(nxy,nxy,nz),n=8,photon=photon)
    t=model.solid.transverse
    x=np.linspace(t.x_edges_cm[0],t.x_edges_cm[-1],nxy+1)
    y=np.linspace(t.y_edges_cm[0],t.y_edges_cm[-1],nxy+1)
    cpos=coolant_channel_positions(model.config)
    cx,cy=cpos[channel]
    ix=int(np.clip(np.searchsorted(x,cx,side="right")-1,0,nxy-1))
    iy=int(np.clip(np.searchsorted(y,cy,side="right")-1,0,nxy-1))
    if outside:
        ix=0 if cx>0 else nxy-1
        iy=0 if cy>0 else nxy-1
    with h5py.File(path,"a") as h:
        mesh=h["tallies/meshes/mesh 19"]
        mesh["lower_left"][...]=[x[0],y[0],-66.]
        mesh["upper_right"][...]=[x[-1],y[-1],66.]
        tally=h["tallies/tally 3"]
        # HDF5 fixed-width scalar strings cannot be overwritten with
        # a longer name without truncation; recreate the native dataset.
        del tally["name"]
        tally.create_dataset(
            "name",data=np.bytes_("nerva_3d_fuel_hydrogen_heating"))
        del tally["filters"]
        ids=([11,12,13] if photon else [11,12])
        tally.create_dataset("filters",data=np.array(ids,dtype=np.int32))
        del tally["score_bins"]
        score="heating" if photon else "heating-local"
        tally.create_dataset("score_bins",data=np.array([score,"absorption"],dtype="S"))
        cell=h["tallies/filters/filter 12"]
        del cell["type"]
        cell.create_dataset("type",data=np.bytes_("cell"))
        del cell["bins"]
        cell.create_dataset("bins",data=np.asarray(CELL_IDS,dtype=np.int32))
        cell["n_bins"][...]=19
        parts=(["photon","neutron"] if reverse else ["neutron","photon"])
        if photon:
            pg=h.create_group("tallies/filters/filter 13")
            pg.create_dataset("type",data=np.bytes_("particle"))
            pg.create_dataset("bins",data=np.array(parts,dtype="S"))
            pg.create_dataset("n_bins",data=2)
        npc=2 if photon else 1
        results=np.zeros((nxy*nxy*nz*19*npc,2,2))
        mesh_idx=ix+nxy*(iy+nxy*source_index)
        for pi in range(npc):
            name=parts[pi] if photon else "neutron"
            val=4.0 if name=="neutron" else 1.5
            row=(mesh_idx*19+channel)*npc+pi
            results[row,0,0]=8*val
            results[row,0,1]=8*(val**2+7*.025**2)
        del tally["results"]
        tally.create_dataset("results",data=results)
    return ix,iy


def parse(path,model,**kwargs):
    options=dict(
        expected_channel_cell_ids=CELL_IDS,
        source_rate_particles_per_s=RATE,
        source_rate_basis="SYNTHETIC HDF5 source intensity, not reactor power",
        nuclear_data_sha256=DATA_SHA,
        isolated_element_geometry_confirmed=True,
    )
    options.update(kwargs)
    return read_native_19_channel_hydrogen_heating(path,model,**options)


class Stage67HydrogenChannelConservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m=coupled(film=0.,nz=4,nx=20,ny=24)

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/"statepoint.h5"
        self.ix,self.iy=fixture(self.path,self.m)

    def tearDown(self):
        self.tmp.cleanup()

    def test_native_two_particle_scores_and_exact_channel_axial_assignment(self):
        r=parse(self.path,self.m)
        self.assertEqual(r.deposition_w.shape,(4,19))
        self.assertEqual(r.source_channel_ids,CELL_IDS)
        self.assertEqual(r.score_name,"heating")
        np.testing.assert_allclose(r.neutron_deposition_w[2,8],
                                   4*RATE*EV_TO_J,atol=1e-13,rtol=0)
        np.testing.assert_allclose(r.photon_deposition_w[2,8],
                                   1.5*RATE*EV_TO_J,atol=1e-13,rtol=0)
        expected=5.5*RATE*EV_TO_J
        self.assertAlmostEqual(r.total_deposition_w,expected,delta=1e-13)
        self.assertEqual(np.count_nonzero(r.deposition_w),1)
        self.assertAlmostEqual(r.deposition_w[2,8],expected,delta=1e-13)
        self.assertAlmostEqual(r.first_law_source_w,expected,delta=1e-13)
        self.assertGreater(r.total_uncertainty_upper_bound_w,0)

    def test_neutron_only_heating_local_is_supported_without_double_photon(self):
        fixture(self.path,self.m,photon=False)
        r=parse(self.path,self.m)
        self.assertEqual(r.score_name,"heating-local")
        self.assertAlmostEqual(r.total_deposition_w,4*RATE*EV_TO_J,delta=1e-13)
        self.assertEqual(float(np.sum(r.photon_deposition_w)),0.)
        self.assertEqual(np.count_nonzero(r.deposition_w),1)

    def test_swapped_particle_order_does_not_change_energy_partition(self):
        a=parse(self.path,self.m)
        fixture(self.path,self.m,reverse=True)
        b=parse(self.path,self.m)
        np.testing.assert_allclose(a.neutron_deposition_w,b.neutron_deposition_w,
                                   atol=1e-13,rtol=0)
        np.testing.assert_allclose(a.photon_deposition_w,b.photon_deposition_w,
                                   atol=1e-13,rtol=0)

    def test_axial_rebin_preserves_total_per_channel_energy(self):
        m=coupled(film=0.,nz=3,nx=20,ny=24)
        r=parse(self.path,m)
        self.assertEqual(r.deposition_w.shape,(3,19))
        self.assertEqual(np.count_nonzero(r.deposition_w[:,8]),2)
        self.assertAlmostEqual(r.total_deposition_w,5.5*RATE*EV_TO_J,
                               delta=1e-13)

    def test_outside_bore_nonzero_tally_rejected_despite_global_energy_match(self):
        fixture(self.path,self.m,nxy=3,outside=True)
        with self.assertRaisesRegex(ValueError,"outside physical channel bore"):
            parse(self.path,self.m)

    def test_cell_id_order_mismatch_rejected_even_if_all_19_same_cells(self):
        wrong=list(CELL_IDS)
        wrong[0],wrong[1]=wrong[1],wrong[0]
        with self.assertRaisesRegex(ValueError,"channel identity"):
            parse(self.path,self.m,expected_channel_cell_ids=tuple(wrong))

    def test_whole_core_mesh_rejected_before_h2_heat_can_enter_element(self):
        with h5py.File(self.path,"a") as h:
            mesh=h["tallies/meshes/mesh 19"]
            mesh["lower_left"][...]=[-20.,-20.,-66.]
            mesh["upper_right"][...]=[20.,20.,66.]
        with self.assertRaisesRegex(ValueError,"whole-core"):
            parse(self.path,self.m)

    def test_missing_isolated_element_proof_rejected(self):
        with self.assertRaisesRegex(ValueError,"isolated physical"):
            parse(self.path,self.m,isolated_element_geometry_confirmed=False)

    def test_incompatible_photon_score_fails_closed(self):
        with h5py.File(self.path,"a") as h:
            ds=h["tallies/tally 3/score_bins"]
            ds[0]=np.bytes_("heating-local")
        with self.assertRaisesRegex(ValueError,"deposited-energy score"):
            parse(self.path,self.m)

    def test_coupled_hydrogen_deposition_enters_stage66_once_without_wall(self):
        r=parse(self.path,self.m)
        old=initial(self.m,tf=400.,th=400.)
        result=solve(self.m,old,dt=1e-5,
                     hydrogen_deposition_w=r.deposition_w)
        self.assertAlmostEqual(
            result.hydrogen_deposition_j,r.total_deposition_w*1e-5,
            delta=1e-12)
        self.assertLess(result.relative_first_law_residual,3e-9)
        self.assertEqual(float(np.sum(result.wall_heat_to_hydrogen_j)),0.)
        self.assertGreater(
            primitive_from_result(self.m,result)[2,8],400.)

    def test_combined_fuel_and_channel_h2_sources_enter_distinct_ledgers(self):
        fuel_fixture=Path(self.tmp.name)/"fuel_statepoint.h5"
        coupled_native_fixture(fuel_fixture,model=self.m)
        nf=load(fuel_fixture)
        f=project_native_3d_fuel_heating(
            self.m.config,nf,self.m.solid,
            isolated_element_geometry_confirmed=True)
        h=parse(self.path,self.m)
        result=solve(self.m,initial(self.m,tf=400.,th=400.),
                     dt=5e-6,q=f.resolved.fuel_deposition_w,
                     hydrogen_deposition_w=h.deposition_w)
        self.assertAlmostEqual(
            result.fuel_deposition_j,
            f.resolved.mapped_fuel_power_w*5e-6,delta=2e-12)
        self.assertAlmostEqual(
            result.hydrogen_deposition_j,h.total_deposition_w*5e-6,
            delta=2e-12)
        self.assertLess(result.relative_first_law_residual,3e-9)


def primitive_from_result(model,result):
    return primitive(model.fluid,result.state.hydrogen).temperature_k


if __name__=="__main__":
    unittest.main(verbosity=2)
