"""Numerical/integrity regressions for evaluated local heating KERMA coverage.

Synthetic inventory rows only; a separate CI workflow reads genuine
evaluated OpenMC cross-section HDF5 files for the physical coverage result.
"""
from __future__ import annotations

from .audit_kerma_coverage import assess_mt901_inventory


def reject(action, message):
    try:
        action()
    except ValueError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("invalid neutron reaction inventory accepted: " + message)


def main() -> int:
    rows = [
        {"nuclide": "U235", "local_kerma_mt901_present": True},
        {"nuclide": "Zr90", "local_kerma_mt901_present": True},
        {"nuclide": "C0", "local_kerma_mt901_present": True},
    ]
    good = assess_mt901_inventory(rows)
    assert good["local_heating_evaluated_kerma_inventory_complete"]
    assert good["nasa_nerva_core_heating_benchmarked"] is False
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
        {"nuclide": None, "local_kerma_mt901_present": True}
    ]), "nuclide identity")
    print("Evaluated HDF5 local-neutron MT901 coverage QA arithmetic: PASS")
    print("  synthetic inventories only; evaluated-data inspection performed in real CI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
