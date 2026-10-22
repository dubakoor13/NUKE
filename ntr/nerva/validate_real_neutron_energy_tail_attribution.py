"""Adversarial numerical tail-reconciliation tests (synthetic fixtures only)."""
from __future__ import annotations

import numpy as np

from .real_neutron_energy_tail_attribution import (
    SCORES, classify_window_partition, TAIL_EDGES_EV,
)


def reject(fn, expected: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert expected in str(exc), (expected,str(exc))
    else:
        raise AssertionError("invalid neutron energy-window conservation accepted")


def main() -> int:
    assert TAIL_EDGES_EV == (0.,1e-5,2e7,1e10)
    old=np.array([[2.,.2,.07,.15,0.],
                  [1.,.1,.04,.07,0.]],dtype=float)
    bands=np.zeros((2,3,len(SCORES)))
    bands[:,1,:]=old
    # Distinct physical scores generated below 1e-5 eV.
    bands[0,0,0]=2.46e-8
    bands[0,0,1]=2.11e-9
    # Independent high-energy (>20MeV) collision term
    bands[1,2,0]=6.7e-7
    bands[1,2,1]=1e-7
    full=bands.sum(axis=1)
    got=classify_window_partition(full,old,bands)
    assert got["independent_window_partition_conserved"]
    assert got["max_abs_full_energy_partition_residual"] < 1e-14
    assert len(got["measurable_material_score_residuals"])==4
    assert all(p["physically_scored_tails_explain_residual_to_1e_minus_11"]
               for p in got["measurable_material_score_residuals"])
    assert abs(got["explicit_tail_score_per_source"]["below_lower_bound"]-2.46e-8)<1e-13
    corrupt_middle=bands.copy()
    corrupt_middle[0,1,0]+=.02
    reject(lambda:classify_window_partition(full,old,corrupt_middle),
           "middle neutron-energy band differs")
    incorrect_total=bands.copy()
    incorrect_total[1,2,0]+=.02
    reject(lambda:classify_window_partition(full,old,incorrect_total),
           "explicit neutron energy tail bins do not conserve")
    negative=bands.copy()
    negative[0,1,1]=-0.1
    negative_old=negative[:,1,:].copy()
    negative_full=negative.sum(axis=1)
    reject(lambda:classify_window_partition(negative_full,negative_old,negative),
           "negative neutron")
    # Heating-local can be signed, and must not be used to infer reactor power.
    signed=bands.copy()
    signed[0,1,4]=-2.0
    signed_full=signed.sum(axis=1)
    assert classify_window_partition(signed_full,signed[:,1,:],signed)["independent_window_partition_conserved"]
    print("Neutron energy-window low/high tail attribution: PASS (synthetic fixtures)")
    print("Actual cold-core physical qualification requires real OpenMC HDF5")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
