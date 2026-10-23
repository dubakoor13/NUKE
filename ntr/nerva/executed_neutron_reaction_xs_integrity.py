"""Read-only evaluated neutron reaction cross-section integrity audit.

Checks native OpenMC HDF5 energy-dependent microscopic elastic scattering
(MT=2), fission (MT=18) and radiative capture (MT=102) evaluations at
every available nuclear-data temperature. Uses the actual root-reachable
materials and nuclear-data digests of genuine four-seed OpenMC transport.

Cross-section presence is not assumed: for example, nonfissile nuclides
need not have MT=18. A legitimately thresholded evaluated reaction can
start above the minimum incident energy, and zero XS values are valid.
Reject physically impossible negative or nonfinite cross sections and
malformed or missing *native temperature slices* of present reactions.
Never interpolate missing data or alter nuclear fuel, geometry or controls.

Numerical library consistency alone cannot qualify operating-temperature
scattering, evaluated KERMA, source convergence or historical NERVA fidelity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import openmc.data

from .executed_neutron_temperature_energy_grid import (
    audit_executed_neutron_energy_grids,
)
from .nuclear_data import _library_entries


# Elastic scattering, neutron-induced fission, radiative capture.
SCATTER_CAPTURE_FISSION_MTS = (2, 18, 102)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assess_evaluated_neutron_xs_curve(
    nuclide: str, mt: int, temperature_label: str,
    energy_eV: object, microscopic_xs_barns: object,
) -> dict:
    """Validate *tabulated* values, not a reconstructed interpolation."""
    if (not isinstance(nuclide, str) or not nuclide
            or type(mt) is not int or mt not in SCATTER_CAPTURE_FISSION_MTS
            or not isinstance(temperature_label, str) or not temperature_label):
        raise ValueError("invalid evaluated neutron reaction identity")
    energies = np.asarray(energy_eV, dtype=np.float64)
    xs = np.asarray(microscopic_xs_barns, dtype=np.float64)
    if (energies.ndim != 1 or xs.ndim != 1
            or len(energies) != len(xs) or len(energies) < 2
            or not np.isfinite(energies).all()
            or energies[0] < 0 or energies[-1] <= energies[0]
            or not np.all(np.diff(energies) > 0)):
        raise ValueError("invalid native evaluated neutron reaction energy grid")
    if not np.isfinite(xs).all():
        raise ValueError("nonfinite native evaluated neutron microscopic cross section")
    if np.any(xs < 0):
        raise ValueError("negative native evaluated neutron microscopic cross section")
    return {
        "native_temperature_label": temperature_label,
        "mt": mt,
        "nuclide": nuclide,
        "tabulated_points": int(len(xs)),
        "first_tabulated_energy_eV": float(energies[0]),
        "last_tabulated_energy_eV": float(energies[-1]),
        "min_microscopic_xs_barns": float(np.min(xs)),
        "max_microscopic_xs_barns": float(np.max(xs)),
        "exact_zero_xs_points": int(np.count_nonzero(xs == 0)),
        "nonzero_xs_points": int(np.count_nonzero(xs > 0)),
        "reaction_threshold_can_be_above_neutron_grid_minimum": True,
        "all_native_xs_finite_nonnegative": True,
        "tabulated_energy_grid_strictly_increasing": True,
    }


def inspect_native_reaction_temperature_keys(
    mt: int, evaluated_labels: list[str], reaction_labels: set[str]
) -> dict:
    """Keep explicit 0K elastic scattering auxiliary data distinct.

    OpenMC may retain a dedicated 0K elastic reaction for Doppler broadening
    rejection correction (DBRC), outside the transported kT material
    evaluations. This is NOT a missing hot-temperature table nor evidence
    of 0K reactor physics. Never treat it as an ordinary energy-grid node.
    """
    if (type(mt) is not int or mt not in SCATTER_CAPTURE_FISSION_MTS
            or not evaluated_labels or len(evaluated_labels)!=len(set(evaluated_labels))
            or any(not isinstance(label,str) or not label for label in evaluated_labels)
            or any(not isinstance(label,str) or not label for label in reaction_labels)):
        raise ValueError("invalid native neutron cross-section temperature provenance")
    ordinary=set(evaluated_labels)
    missing=ordinary-reaction_labels
    allowed_extras={"0K"} if mt==2 else set()
    unrecognized_extras=(reaction_labels-ordinary)-allowed_extras
    if missing:
        raise ValueError(
            "present evaluated neutron MT="+str(mt)+
            " missing requested transport-temperature slices: "+repr(sorted(missing))
        )
    if unrecognized_extras:
        raise ValueError(
            "unexpected native neutron reaction temperature slice: "+
            repr(sorted(unrecognized_extras))
        )
    return {
        "all_evaluated_transport_temperature_slices_present":True,
        "extra_0K_elastic_scattering_reaction_separately_recorded":
            mt==2 and "0K" in (reaction_labels-ordinary),
        "auxiliary_0K_is_not_an_executed_material_temperature":True,
        "unmatched_native_temperature_slices":[],
    }


def audit_executed_neutron_reaction_xs(
    summary: Path, index: Path
) -> dict:
    """Independent nuclear-data evidence for the actually executed core."""
    summary, index = Path(summary), Path(index)
    base = audit_executed_neutron_energy_grids(summary, index)
    evidence = base["incident_neutron_temperature_energy_grid_evidence"]
    required = set(evidence)
    indexed = {}
    for entry in _library_entries(index):
        if entry.data_type != "neutron":
            continue
        for name in entry.materials:
            if name in required:
                if name in indexed:
                    raise ValueError("duplicate executed neutron evaluation: " + name)
                indexed[name] = Path(entry.resolved_path)
    if set(indexed) != required:
        raise ValueError("missing evaluated neutron HDF5 for actual core material")
    rows = []
    total_curves = 0
    found_mt = {str(mt): 0 for mt in SCATTER_CAPTURE_FISSION_MTS}
    for nuclide in sorted(required):
        path = indexed[nuclide]
        expected = evidence[nuclide]
        if _sha(path) != expected["evaluated_hdf5_sha256"]:
            raise ValueError("executed neutron HDF5 evidence digest changed")
        data = openmc.data.IncidentNeutron.from_hdf5(path)
        if data.name != nuclide:
            raise ValueError("actual evaluated neutron isotope HDF5 identity mismatch")
        labels = list(data.temperatures)
        recorded = [r["native_temperature_label"] for r in expected["temperatures"]]
        if labels != recorded or len(set(labels)) != len(labels):
            raise ValueError("evaluated reaction temperature labels do not match verified energy tables")
        by_mt = []
        for mt in SCATTER_CAPTURE_FISSION_MTS:
            reaction = data.reactions.get(mt)
            if reaction is None:
                by_mt.append({
                    "mt": mt, "reaction_present": False,
                    "presence_required_for_all_nuclides": False,
                    "evaluated_temperature_curves": [],
                })
                continue
            if not isinstance(reaction.xs, dict):
                raise ValueError("native evaluated reaction has no temperature-indexed XS")
            # A reaction existing without one of the validated temperatures
            # is incomplete input; do not fabricate that temperature.
            native_temperature_evidence=inspect_native_reaction_temperature_keys(
                mt,labels,set(reaction.xs)
            )
            curves = []
            for label in labels:
                curve = reaction.xs[label]
                if not hasattr(curve, "x") or not hasattr(curve, "y"):
                    raise ValueError("native evaluated XS is not a tabulated energy curve")
                tested = assess_evaluated_neutron_xs_curve(
                    nuclide, mt, label, curve.x, curve.y
                )
                # An energy grid can legitimately begin at a threshold;
                # it must nonetheless overlap the isotope's energy domain.
                neutron_grid = np.asarray(data.energy[label], dtype=float)
                lo = max(tested["first_tabulated_energy_eV"], float(neutron_grid[0]))
                hi = min(tested["last_tabulated_energy_eV"], float(neutron_grid[-1]))
                if hi <= lo:
                    raise ValueError("evaluated reaction curve outside neutron energy domain")
                tested["overlaps_native_incident_neutron_energy_domain"] = True
                curves.append(tested)
                found_mt[str(mt)] += 1
                total_curves += 1
            by_mt.append({
                "mt":mt,
                "reaction_present":True,
                "presence_required_for_all_nuclides":False,
                "native_temperature_key_qualification":native_temperature_evidence,
                "evaluated_temperature_curves":curves,
            })
        rows.append({
            "nuclide":nuclide,
            "evaluated_hdf5_sha256":expected["evaluated_hdf5_sha256"],
            "evaluated_temperature_node_count":len(labels),
            "reaction_tables":by_mt,
        })
    return {
        "audit":"GENUINE_OPENMC_EXECUTED_NEUTRON_REACTION_XS_NUMERICAL_INTEGRITY",
        "actual_four_seed_summary_sha256":_sha(summary),
        "evaluated_cross_sections_xml_sha256":_sha(index),
        "verified_executed_material_ids":base["executed_reachable_material_ids"],
        "verified_evaluated_neutron_nuclide_count":len(rows),
        "selected_mt_reactions":list(SCATTER_CAPTURE_FISSION_MTS),
        "native_reaction_temperature_curves_checked":total_curves,
        "curves_by_mt":found_mt,
        "per_executed_neutron_nuclide":rows,
        "all_present_native_reaction_xs_numerically_valid":True,
        "all_existing_reactions_have_full_native_temperature_slices":True,
        "native_hdf5_digests_provenance_verified":True,
        "fission_source_convergence_qualified":False,
        "qualified_hot_reactor_neutron_transport_from_this_screen":False,
        "neutron_local_heating_mt901_qualified_by_this_screen":False,
        "reactor_design_or_fuel_modified":False,
        "interpretation":(
            "Actual OpenMC isotope HDF5 numerical check for elastic scattering, "
            "fission and capture, over every native evaluation temperature. "
            "Reaction absence is identified, not filled; thresholded zero "
            "cross sections are permitted, but negative or invalid tables "
            "are rejected. This is nuclear-data integrity, not hot reactor "
            "neutron transport validation, covariance qualification, "
            "fission-source convergence or a NASA NERVA benchmark."
        ),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--cross-sections", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit_executed_neutron_reaction_xs(a.summary, a.cross_sections)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Physical neutron nuclides:", result["verified_evaluated_neutron_nuclide_count"])
    print("Real evaluated reaction/temperature curves:",
          result["native_reaction_temperature_curves_checked"])
    print("MT reaction counts:", result["curves_by_mt"])
    print("Native energy-dependent cross section numerical integrity: PASS")
    print("Operating NERVA reactor physics qualification: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
