"""Numerical/integrity regressions for evaluated local heating KERMA coverage.

Synthetic inventory rows only; a separate CI workflow reads genuine
evaluated OpenMC cross-section HDF5 files for the physical coverage result.
"""
from __future__ import annotations

from .audit_kerma_coverage import assess_mt901_inventory
from .mt901_numeric import inspect_mt901_numeric_curves
from types import SimpleNamespace
import numpy as np


def reject(action, message):
    try:
        action()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("invalid neutron reaction inventory accepted: " + message)


def main() -> int:
    rows = [
        {"nuclide": "U235", "local_kerma_mt901_present": True, "local_kerma_mt901_numerical_usable": True},
        {"nuclide": "Zr90", "local_kerma_mt901_present": True, "local_kerma_mt901_numerical_usable": True},
        {"nuclide": "C0", "local_kerma_mt901_present": True, "local_kerma_mt901_numerical_usable": True},
    ]
    good = assess_mt901_inventory(rows)
    assert good["local_heating_evaluated_kerma_inventory_complete"]
    assert good["nasa_nerva_core_heating_benchmarked"] is False
    # A nominally present MT901 with all-zero KERMA is NOT an admissible
    # physical neutron-local heat coefficient table.
    unusable = assess_mt901_inventory([
        rows[0],
        {**rows[1], "local_kerma_mt901_numerical_usable": False},
        rows[2],
    ])
    assert not unusable["local_heating_evaluated_kerma_inventory_complete"]
    assert unusable["nuclides_mt901_present_but_numerically_unusable"] == ["Zr90"]

    def data(energy, coefficients, *, keys=("293K",)):
        reaction = SimpleNamespace(xs={
            key: SimpleNamespace(
                x=np.asarray(energy, dtype=float),
                y=np.asarray(coefficients, dtype=float))
            for key in keys
        })
        return SimpleNamespace(temperatures=list(keys), reactions={901:reaction})

    physical = inspect_mt901_numeric_curves(data(
        [1e-5, 1e3, 2e7], [0., 1.5e5, -1.0e3],
        keys=("293K", "900K"),
    ))
    assert physical["local_kerma_mt901_numerical_usable"]
    assert physical["temperatures_checked"] == 2
    assert all(x["number_of_nonzero_coefficients"] == 2
               for x in physical["per_temperature"])
    assert all(x["number_of_negative_coefficients"] == 1
               for x in physical["per_temperature"])

    all_zero = inspect_mt901_numeric_curves(data(
        [1e-5, 1e3, 2e7], [0., 0., 0.]
    ))
    assert not all_zero["local_kerma_mt901_numerical_usable"]
    assert "zero" in all_zero["per_temperature"][0]["issue"]

    missing_temp = data([1e-5, 1e3], [2., 3.])
    missing_temp.temperatures.append("1200K")
    assert not inspect_mt901_numeric_curves(missing_temp)[
        "local_kerma_mt901_numerical_usable"
    ]
    nan_curve = data([1e-5,1e3], [1., float("nan")])
    assert not inspect_mt901_numeric_curves(nan_curve)[
        "local_kerma_mt901_numerical_usable"
    ]
    duplicate_energy = data([1.,1.], [2., 3.])
    assert not inspect_mt901_numeric_curves(duplicate_energy)[
        "local_kerma_mt901_numerical_usable"
    ]
    absent_mt = data([1.,2.], [2.,3.])
    absent_mt.reactions.clear()
    assert not inspect_mt901_numeric_curves(absent_mt)[
        "local_kerma_mt901_numerical_usable"
    ]

    missing = assess_mt901_inventory([
        *rows[:1], {**rows[1], "local_kerma_mt901_present": False}, rows[2]
    ])
    assert not missing["local_heating_evaluated_kerma_inventory_complete"]
    assert missing["nuclides_missing_or_unverified_mt901"] == ["Zr90"]
    absent = assess_mt901_inventory([
        {**rows[0], "local_kerma_mt901_present": None}, rows[1], rows[2]
    ])
    assert absent["nuclides_missing_or_unverified_mt901"] == ["U235"]
    reject(lambda: assess_mt901_inventory([]), "empty")
    reject(lambda: assess_mt901_inventory(rows + [rows[0]]), "duplicate")
    reject(lambda: assess_mt901_inventory([
        {"nuclide": "U235", "local_kerma_mt901_present": "yes"}
    ]), "boolean")
    reject(lambda: assess_mt901_inventory([
        {"nuclide": "U235", "local_kerma_mt901_present": True,
         "local_kerma_mt901_numerical_usable": "yes"}
    ]), "boolean")
    reject(lambda: assess_mt901_inventory([
        {"nuclide": None, "local_kerma_mt901_present": True}
    ]), "nuclide identity")
    print("Evaluated HDF5 MT901 numeric KERMA and inventory QA: PASS")
    print("  synthetic inventories only; evaluated-data inspection performed in real CI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
