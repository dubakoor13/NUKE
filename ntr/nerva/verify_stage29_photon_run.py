"""Check a report from ACTUAL photon HDF5 preflight without inventing data.

A report with missing actual photon elements is a valid *negative* physical
finding, not a successful hot-core model. Binary/source qualification remains
separate from report arithmetic and model readiness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify_photon_report(report: dict) -> dict:
    required = set(report["required_photon_elements"])
    verified = set(report["verified_photon_elements"])
    missing = set(report["missing_photon_elements"])
    errors = report["photon_binary_audit_errors"]
    photon_hashes = {symbol for symbol, _ in report["photon_binary_sha256"]}
    if (report.get("photon_transport_requested") is not True or not required
            or report.get("photon_evaluated_release_verified") is not False
            or not isinstance(errors, list)
            or not verified.issubset(required)
            or missing != required - set(report["photon_materials"])
            or verified != photon_hashes
            or missing.intersection(verified)):
        raise ValueError("inconsistent actual photon library/source-coverage report")
    should_be_ready = (
        not missing and not errors and verified == required
        and not report["missing_neutron_nuclides"]
        and not report["missing_thermal_scattering_tables"]
        and not report["referenced_files_missing"]
    )
    if report["ready"] is not should_be_ready:
        raise ValueError("coupled physical-data preflight promoted incomplete photon inventory")
    return {
        "genuine_hdf5_photon_element_coverage_complete": (
            not missing and not errors and verified == required),
        "actual_transport_index_preflight_ready": should_be_ready,
        "nuclear_evaluated_release_edition_physically_qualified": False,
        "hot_NERVA_neutron_photon_heating_physically_qualified": False,
        "required_photon_element_count": len(required),
        "verified_photon_element_count": len(verified),
        "missing_photon_elements": sorted(missing),
        "photon_data_errors": errors,
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report",type=Path,required=True)
    a=p.parse_args()
    r=json.loads(a.report.read_text(encoding="utf-8"))
    result=verify_photon_report(r)
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
