"""Adversarial mathematical screening tests (no synthetic physical claims)."""
from __future__ import annotations

import numpy as np

from .executed_neutron_reaction_xs_integrity import (
    SCATTER_CAPTURE_FISSION_MTS,
    assess_evaluated_neutron_xs_curve,
    inspect_native_reaction_temperature_keys,
)


def reject(fn, msg: str) -> None:
    try:
        fn()
    except (TypeError, ValueError) as e:
        assert msg in str(e), (msg, str(e))
    else:
        raise AssertionError("nonphysical nuclear-data reaction curve accepted: " + msg)


def main() -> int:
    assert SCATTER_CAPTURE_FISSION_MTS == (2, 18, 102)
    # OpenMC permits a separate zero-K *elastic only* auxiliary
    # evaluation for Doppler broadening, not an actual transport temperature.
    zero_k=inspect_native_reaction_temperature_keys(
        2,["294K","600K"],{"294K","600K","0K"}
    )
    assert zero_k["extra_0K_elastic_scattering_reaction_separately_recorded"]
    assert zero_k["all_evaluated_transport_temperature_slices_present"]
    no_extra=inspect_native_reaction_temperature_keys(
        18,["294K","600K"],{"294K","600K"}
    )
    assert not no_extra["extra_0K_elastic_scattering_reaction_separately_recorded"]
    reject(lambda:inspect_native_reaction_temperature_keys(
        2,["294K","600K"],{"294K","0K"}),"missing requested transport-temperature")
    reject(lambda:inspect_native_reaction_temperature_keys(
        18,["294K"],{"294K","0K"}),"unexpected native neutron reaction")
    reject(lambda:inspect_native_reaction_temperature_keys(
        102,["294K"],{"294K","1000K"}),"unexpected native neutron reaction")
    xs=assess_evaluated_neutron_xs_curve(
        "TestNuclide",18,"600K",
        np.array([1e6,1.5e6,2e6,10e6],dtype=float),
        np.array([0.,0.,.2,.31],dtype=float),
    )
    assert xs["tabulated_points"]==4
    assert xs["exact_zero_xs_points"]==2
    assert xs["nonzero_xs_points"]==2
    assert xs["min_microscopic_xs_barns"]==0.
    assert xs["max_microscopic_xs_barns"]==.31
    assert xs["first_tabulated_energy_eV"]==1e6
    assert xs["reaction_threshold_can_be_above_neutron_grid_minimum"]
    for mt in SCATTER_CAPTURE_FISSION_MTS:
        out=assess_evaluated_neutron_xs_curve(
            "X",mt,"294K",[1e-5,1e-3,1.],[0.,.0001,5.]
        )
        assert out["mt"]==mt
        assert out["all_native_xs_finite_nonnegative"]
    # A legitimately identically zero cross section is not invented.
    zero=assess_evaluated_neutron_xs_curve(
        "X",102,"294K",[1e-5,1e-3,1.],[0.,0.,0.]
    )
    assert zero["exact_zero_xs_points"]==3
    assert zero["nonzero_xs_points"]==0

    reject(lambda:assess_evaluated_neutron_xs_curve(
        "X",2,"294K",[1.,2.,3.],[1.,-.00001,2.]),"negative native")
    reject(lambda:assess_evaluated_neutron_xs_curve(
        "X",18,"294K",[1.,2.,3.],[1.,float("nan"),2.]),"nonfinite native")
    reject(lambda:assess_evaluated_neutron_xs_curve(
        "X",102,"294K",[1.,2.,3.],[1.,float("inf"),2.]),"nonfinite native")
    reject(lambda:assess_evaluated_neutron_xs_curve(
        "X",2,"294K",[1.,2.,2.],[1.,2.,3.]),"invalid native")
    reject(lambda:assess_evaluated_neutron_xs_curve(
        "X",2,"294K",[2.,1.,3.],[1.,2.,3.]),"invalid native")
    reject(lambda:assess_evaluated_neutron_xs_curve(
        "X",2,"294K",[-1.,1.],[1.,2.]),"invalid native")
    reject(lambda:assess_evaluated_neutron_xs_curve(
        "X",2,"294K",[1.,2.],[1.]),"invalid native")
    reject(lambda:assess_evaluated_neutron_xs_curve(
        "X",2,"294K",[1.],[1.]),"invalid native")
    reject(lambda:assess_evaluated_neutron_xs_curve(
        "X",2,"294K",[[1.,2.],[3.,4.]],[1.,2.]),"invalid native")
    reject(lambda:assess_evaluated_neutron_xs_curve(
        "X",999,"294K",[1.,2.],[1.,2.]),"invalid evaluated neutron reaction identity")
    reject(lambda:assess_evaluated_neutron_xs_curve(
        "X",18,"",[1.,2.],[1.,2.]),"invalid evaluated neutron reaction identity")
    print("Native neutron reaction XS numerical regressions: PASS")
    print("  Threshold-zero accepted, negative/nonfinite/invalid-energy rejected")
    print("  Genuine evaluated OpenMC nuclide HDF5 tested separately in CI")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
