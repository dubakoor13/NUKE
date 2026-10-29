"""Stage68 whole-core native HDF5 energy ledger regression; synthetic data only."""
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from .validate_stage60_native_hdf5_heating import synthetic_statepoint
from .openmc_full_core_energy_ledger import (
    MATERIAL_ORDER, PARTICLE_ORDER, read_openmc_full_core_energy_ledger)

IDS={name:501+i for i,name in enumerate(MATERIAL_ORDER)}
M=dict(zip(MATERIAL_ORDER,(2.4,1.1,.7,.5,.4,.35,.3,.1,.15)))
P={"neutron":3.6,"photon":2.1,"electron":.25,"positron":.05}
RATE=1e19
SHA="c"*64


def fixture(path, *, materials=MATERIAL_ORDER, particles=PARTICLE_ORDER,
            bias_fuel=0., std=.004):
    synthetic_statepoint(path,n=8,photon=True)
    n=8
    with h5py.File(path,"a") as h:
        tg=h["tallies"]
        tg.attrs["n_tallies"]=2
        tg.attrs["ids"]=np.array([3,4],dtype=int)
        mat=tg["tally 3"]
        del mat["name"]
        mat.create_dataset("name",data=np.bytes_("nerva_material_transport"))
        mat["filters"][...]=[12]
        scores=["flux","absorption","fission","nu-fission","heating","heating-local"]
        del mat["score_bins"]
        mat.create_dataset("score_bins",data=np.array(scores,dtype="S"))
        fil=tg["filters/filter 12"]
        del fil["bins"]
        fil.create_dataset("bins",data=np.array([IDS[k] for k in materials]))
        fil["n_bins"][...]=len(materials)
        mm=np.zeros((len(materials),len(scores),2))
        for i,k in enumerate(materials):
            v=M[k]+(bias_fuel if k=="fuel" else 0.)
            mm[i,4,0]=n*v
            mm[i,4,1]=n*(v*v+(n-1)*std*std)
        del mat["results"]
        mat.create_dataset("results",data=mm)
        part=tg.create_group("tally 4")
        part.create_dataset("name",data=np.bytes_("nerva_particle_heating"))
        part.create_dataset("filters",data=np.array([13]))
        part.create_dataset("score_bins",data=np.array(["heating"],dtype="S"))
        part.create_dataset("nuclides",data=np.array(["total"],dtype="S"))
        part.create_dataset("n_realizations",data=n)
        pp=np.zeros((len(particles),1,2))
        for i,k in enumerate(particles):
            v=P[k]
            pp[i,0,0]=n*v
            pp[i,0,1]=n*(v*v+(n-1)*std*std)
        part.create_dataset("results",data=pp)
        pf=tg.create_group("filters/filter 13")
        pf.create_dataset("type",data=np.bytes_("particle"))
        pf.create_dataset("bins",data=np.array(particles,dtype="S"))
        pf.create_dataset("n_bins",data=len(particles))


class FullCoreEnergyAccounting(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/"statepoint.10.h5"
        fixture(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self,**kwargs):
        arguments=dict(expected_material_ids=IDS,
                       source_rate_particles_per_s=RATE,
                       source_rate_basis="SYNTHETIC source, not real reactor",
                       nuclear_data_sha256=SHA)
        arguments.update(kwargs)
        return read_openmc_full_core_energy_ledger(self.path,**arguments)

    def test_whole_core_material_and_particle_closure(self):
        r=self.read()
        self.assertAlmostEqual(r.material_total_w,6*RATE*1.602176634e-19,delta=1e-11)
        self.assertAlmostEqual(r.particle_total_w,r.material_total_w,delta=1e-11)
        self.assertAlmostEqual(r.material_minus_particle_w,0.,delta=1e-11)
        self.assertEqual(r.n_realizations,8)
        self.assertEqual(r.active_batches_at_statepoint,8)

    def test_electron_positron_heat_must_not_be_lost(self):
        r=self.read()
        nplusp=sum(r.particle_power_w[k] for k in ("neutron","photon"))
        self.assertGreater(r.charged_secondary_power_w,0.)
        self.assertAlmostEqual(nplusp+r.charged_secondary_power_w,
                               r.particle_total_w,delta=1e-11)
        self.assertAlmostEqual(r.charged_secondary_power_w/r.particle_total_w,
                               .05,delta=1e-13)

    def test_nonfuel_nonhydrogen_structural_energy_is_separate(self):
        r=self.read()
        self.assertGreater(r.nonthermal_material_power_w,0.)
        self.assertAlmostEqual(r.fuel_power_w+r.hydrogen_power_w+
                               r.nonthermal_material_power_w,
                               r.material_total_w,delta=1e-11)
        self.assertIn("whole-core",r.material_coverage_status)

    def test_reordered_independent_material_and_particle_bins(self):
        original=self.read()
        fixture(self.path,materials=tuple(reversed(MATERIAL_ORDER)),
                particles=tuple(reversed(PARTICLE_ORDER)))
        revised=self.read()
        for k in MATERIAL_ORDER:
            self.assertAlmostEqual(original.material_power_w[k],
                                   revised.material_power_w[k],delta=1e-12)
        for k in PARTICLE_ORDER:
            self.assertAlmostEqual(original.particle_power_w[k],
                                   revised.particle_power_w[k],delta=1e-12)

    def test_missing_charged_particles_fail_closed(self):
        fixture(self.path,particles=("neutron","photon"))
        with self.assertRaisesRegex(ValueError,"electron/positron"):
            self.read()

    def test_missing_or_repeated_material_ids_fail_closed(self):
        fixture(self.path,materials=MATERIAL_ORDER[:-1])
        with self.assertRaisesRegex(ValueError,"do not cover"):
            self.read()
        fixture(self.path)
        ids=dict(IDS)
        ids["inconel"]=ids["zrc"]
        with self.assertRaisesRegex(ValueError,"material-ID manifest"):
            self.read(expected_material_ids=ids)

    def test_real_energy_imbalance_exceeds_statistical_bound(self):
        fixture(self.path,bias_fuel=.6)
        with self.assertRaisesRegex(ArithmeticError,"does not close"):
            self.read()

    def test_covariance_safe_stderr_bound(self):
        r=self.read()
        expected=(len(MATERIAL_ORDER)+len(PARTICLE_ORDER))*.004
        factor=RATE*1.602176634e-19
        self.assertAlmostEqual(r.conservative_difference_stderr_upper_bound_w,
                               expected*factor,delta=1e-10)
        self.assertAlmostEqual(r.statistical_acceptance_limit_w,
                               5*expected*factor,delta=1e-10)

    def test_neutron_only_transport_not_four_particle_closure(self):
        with h5py.File(self.path,"a") as h:
            h.attrs["photon_transport"]=0
        with self.assertRaisesRegex(ValueError,"requires photon transport"):
            self.read()

    def test_corrupted_score_and_nuclide_identity(self):
        with h5py.File(self.path,"a") as h:
            h["tallies/tally 4/score_bins"][0]=np.bytes_("flux")
        with self.assertRaisesRegex(ValueError,"heating score"):
            self.read()
        fixture(self.path)
        with h5py.File(self.path,"a") as h:
            h["tallies/tally 4/nuclides"][0]=np.bytes_("U235")
        with self.assertRaisesRegex(ValueError,"total nuclide"):
            self.read()

    def test_bad_realizations_or_variance(self):
        with h5py.File(self.path,"a") as h:
            h["tallies/tally 4/n_realizations"][...]=9
        with self.assertRaisesRegex(ValueError,"realization count"):
            self.read()
        fixture(self.path)
        with h5py.File(self.path,"a") as h:
            h["tallies/tally 3/results"][0,4,1]=0.
        with self.assertRaisesRegex(ValueError,"negative variance"):
            self.read()

    def test_source_rate_hash_and_provenance_guards(self):
        r=self.read()
        self.assertEqual(self.read(expected_statepoint_sha256=r.statepoint_sha256).
                         statepoint_sha256,r.statepoint_sha256)
        with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
            self.read(expected_statepoint_sha256="f"*64)
        with self.assertRaisesRegex(ValueError,"source particle rate"):
            self.read(source_rate_particles_per_s=0.)
        with self.assertRaisesRegex(ValueError,"nuclear data SHA256"):
            self.read(nuclear_data_sha256="bad")


if __name__=="__main__":
    unittest.main(verbosity=2)
