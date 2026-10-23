"""Adversarial neutron material tally energy-window conservation QA.

Synthetic arrays here exercise only mathematical accounting; the native
OpenMC C++ eigenvalue statepoints and actual isotope nuclear data are
independently checked by the dedicated four-seed workflow.
"""
from __future__ import annotations

import numpy as np

from .real_eigenvalue_transport_neutron_energy_support import (
    assess_native_material_energy_score_partition,
)


def reject(fn, why):
    try:
        fn()
    except (ValueError,TypeError) as exc:
        assert why in str(exc),(why,str(exc))
    else:
        raise AssertionError("nonphysical neutron tally partition accepted")


def main()->int:
    e=np.array([[1.,2.,3.],[2.,3.,4.],[0.,0.,0.]],dtype=np.float64)
    base=np.array([6.,9.,0.])
    for score in ("flux","absorption","fission","nu-fission"):
        yes=assess_native_material_energy_score_partition(base,e,score_name=score)
        assert yes["all_material_energy_windows_closed_at_stated_numerical_tolerance"]
        assert max(abs(x) for x in yes["signed_unassigned_energy_window_residual_per_source"])==0.
        assert not yes["tail_neutrons_below_or_above_window_individually_attributed"]
        assert not yes["observed_energy_window_fraction_is_statistical_source_convergence_test"]
    # Residual must be exposed, NOT assigned as confirmed under/overflow.
    fail=assess_native_material_energy_score_partition([7.,9.,0.],e,score_name="flux")
    assert not fail["all_material_energy_windows_closed_at_stated_numerical_tolerance"]
    assert fail["signed_unassigned_energy_window_residual_per_source"][0]==1.
    assert fail["max_abs_unassigned_score"]==1.
    near=assess_native_material_energy_score_partition(
        [6. + 1e-11,9.,0.],e,score_name="flux")
    assert near["all_material_energy_windows_closed_at_stated_numerical_tolerance"]
    assert near["signed_unassigned_energy_window_residual_per_source"][0]>0.
    reject(lambda:assess_native_material_energy_score_partition(
        base,[[1.,2.],[2.,3.],[1.,1.]],score_name="flux"),"nonphysical")
    reject(lambda:assess_native_material_energy_score_partition(
        base,np.array([[1.,2.,3.],[1.,2.,3.]]),score_name="flux"),"nonphysical")
    reject(lambda:assess_native_material_energy_score_partition(
        base,np.array([[1.,2.,np.nan],[2.,3.,4.],[0.,0.,0.]]),score_name="flux"),"nonphysical")
    reject(lambda:assess_native_material_energy_score_partition(
        base,np.array([[1.,-1.,3.],[2.,3.,4.],[0.,0.,0.]]),score_name="flux"),"nonphysical")
    reject(lambda:assess_native_material_energy_score_partition(
        [6.,-9.,0.],e,score_name="flux"),"nonphysical")
    reject(lambda:assess_native_material_energy_score_partition(
        [6.,9.],e,score_name="flux"),"nonphysical")
    reject(lambda:assess_native_material_energy_score_partition(
        base,e,score_name="beta-effective"),"unsupported")
    print("True OpenMC scored neutron energy window partition numerical QA: PASS")
    print("  real isotope HDF5 and genuine source transport must be tested in C++ CI")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
