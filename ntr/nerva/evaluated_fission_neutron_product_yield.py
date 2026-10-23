"""Read-only evaluated fission neutron yield function consistency audit.

This independently checks *microscopic nuclear-data product yields* in the
same evaluated HDF5 files used by four real OpenMC neutron transport runs.
It is not a neutron tally, point kinetics beta_eff, reactor optimization,
hot-core physics or an interpolation/extrapolation repair.

For each physically instantiated nuclide with MT=18, evaluate total, prompt
and each delayed-neutron product yield on reproducibly subsampled genuine
positive-fission cross-section incident energies. Restrict sampling to all
available tabulated product yield domains; evaluate genuine OpenMC Function1D
objects with their native interpolation. Preserve all group contributions.
If one mode is missing, explicitly withhold closure qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import openmc.data

from .executed_neutron_reaction_xs_integrity import (
    audit_executed_neutron_reaction_xs,
)
from .nuclear_data import _library_entries


SAMPLED_MAX_ENERGIES_PER_TEMPERATURE = 96
# Informational evaluated-function consistency screen, not a statistically
# justified nuclear data uncertainty and not an estimate of beta_eff.
MAX_EVALUATED_YIELD_RELATIVE_MISMATCH = 1e-3
MODES = ("total", "prompt", "delayed")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def assess_evaluated_neutron_product_yields(
    incident_energies_eV: object,
    product_yields_by_mode: dict[str, list[object]],
) -> dict:
    """Independent pure-numeric evaluation of properly grouped yields.

    A delayed fission product has several delayed-precursor groups: sum ALL
    delayed-neutron yields, not merely the first group. The total, prompt,
    delayed yields are not required to coincide at machine precision.
    """
    e = np.asarray(incident_energies_eV, dtype=np.float64)
    if (e.ndim != 1 or len(e) < 2 or not np.isfinite(e).all()
            or e[0] < 0 or np.any(np.diff(e) <= 0)
            or not isinstance(product_yields_by_mode, dict)
            or any(m not in MODES for m in product_yields_by_mode)):
        raise ValueError("invalid evaluated fission neutron product energy samples")
    outputs = {}
    groups = {}
    for mode in MODES:
        data = product_yields_by_mode.get(mode, [])
        if not isinstance(data, list):
            raise ValueError("evaluated product yield mode group must be a list")
        arrays = []
        for values in data:
            arr = np.asarray(values, dtype=np.float64)
            if arr.shape != e.shape or not np.isfinite(arr).all():
                raise ValueError("nonfinite or misaligned evaluated fission neutron product yield")
            if np.any(arr < -1e-12):
                raise ValueError("negative evaluated fission neutron product yield")
            arrays.append(np.maximum(arr, 0.0))
        groups[mode] = len(arrays)
        outputs[mode] = np.sum(arrays, axis=0) if arrays else None
    comparable = all(outputs[m] is not None for m in MODES)
    mismatch = None
    if comparable:
        total = outputs["total"]
        parts = outputs["prompt"] + outputs["delayed"]
        denom = np.maximum(1e-9, np.maximum(np.abs(total), np.abs(parts)))
        mismatch = float(np.max(np.abs(total-parts)/denom))
        if np.any(total < 0) or np.any(parts < 0):
            raise ValueError("nonphysical negative cumulative fission neutron yield")
        if mismatch > MAX_EVALUATED_YIELD_RELATIVE_MISMATCH:
            raise ValueError(
                "separate evaluated total/prompt/delayed fission product yields "
                "differ by more than nuclear-data consistency screen"
            )
    return {
        "evaluated_incident_energy_samples":int(e.size),
        "first_sample_eV":float(e[0]),
        "last_sample_eV":float(e[-1]),
        "native_neutron_product_group_counts":groups,
        "all_total_prompt_delayed_yield_modes_available":comparable,
        "maximum_relative_total_minus_prompt_delayed_yield_mismatch":mismatch,
        "evaluated_yield_relative_consistency_screen":MAX_EVALUATED_YIELD_RELATIVE_MISMATCH,
        "evaluated_yield_consistency_passed_if_all_modes_present":(
            comparable and mismatch is not None
            and mismatch <= MAX_EVALUATED_YIELD_RELATIVE_MISMATCH
        ),
        "missing_modes": [mode for mode in MODES if outputs[mode] is None],
        "beta_effective_calculated":False,
    }


def _sample_fission_reaction(
    neutron_data, nuclide: str, temperature: str
) -> dict:
    fission=neutron_data.reactions[18]
    if temperature not in fission.xs:
        raise ValueError("missing physical MT18 evaluated transport-temperature slice")
    xs=fission.xs[temperature]
    e=np.asarray(xs.x,dtype=np.float64)
    values=np.asarray(xs.y,dtype=np.float64)
    if (len(e)<2 or e.shape!=values.shape or not np.isfinite(e).all()
            or not np.isfinite(values).all() or np.any(values<0)
            or e[0]<0 or np.any(np.diff(e)<=0)):
        raise ValueError("invalid original evaluated fission reaction energy/XS")
    # OpenMC separates explicit emitted products from derived quantities.
    # The independent total neutron yield is often in the top-level HDF5
    # 'total_nu' record and is attached to Reaction.derived_products.
    # Ignoring this field falsely classified U235/U238 as uncomparable.
    # Do not CONSTRUCT a total curve by adding prompt and delayed yields:
    # that would make the numerical closure tautologically true.
    products = [
        p for p in list(fission.products)+list(fission.derived_products)
        if p.particle=="neutron" and p.emission_mode in MODES
    ]
    directly_evaluated_modes={
        mode:sum(1 for p in fission.products
                 if p.particle=="neutron" and p.emission_mode==mode)
        for mode in MODES
    }
    derived_modes={
        mode:sum(1 for p in fission.derived_products
                 if p.particle=="neutron" and p.emission_mode==mode)
        for mode in MODES
    }
    # OpenMC Product.yield_ can be a polynomial or a Tabulated1D. ENDF
    # defines Tabulated1D inside its tabulated domain. To avoid extrapolation
    # masquerading as evaluated physics, restrict to the common true domain.
    lower,upper=float(e[0]),float(e[-1])
    for p in products:
        fn=p.yield_
        if isinstance(fn,openmc.data.Tabulated1D):
            px=np.asarray(fn.x,dtype=np.float64)
            if (len(px)<2 or not np.isfinite(px).all()
                    or px[0]<0 or np.any(np.diff(px)<=0)):
                raise ValueError("invalid native fission neutron product yield domain")
            lower=max(lower,float(px[0]))
            upper=min(upper,float(px[-1]))
    if upper<=lower:
        return {
            "nuclide":nuclide,
            "native_temperature_label":temperature,
            "product_tabulated_domains_overlap_positive_fission":False,
            "no_valid_evaluated_yield_comparison":True,
            "reason":"no shared product yield energy domain",
            "original_evaluated_product_modes":directly_evaluated_modes,
            "native_derived_evaluated_product_modes":derived_modes,
            "mt18_redundant_reaction":bool(fission.redundant),
        }
    eligible=np.flatnonzero((e>=lower)&(e<=upper)&(values>0))
    if len(eligible)<2:
        return {
            "nuclide":nuclide,
            "native_temperature_label":temperature,
            "product_tabulated_domains_overlap_positive_fission":False,
            "no_valid_evaluated_yield_comparison":True,
            "reason":"fewer than two positive-fission energy nodes in native common domain",
            "original_evaluated_product_modes":directly_evaluated_modes,
            "native_derived_evaluated_product_modes":derived_modes,
            "mt18_redundant_reaction":bool(fission.redundant),
        }
    ids=eligible[np.unique(np.linspace(
        0,len(eligible)-1,min(SAMPLED_MAX_ENERGIES_PER_TEMPERATURE,len(eligible)),
        dtype=np.int64
    ))]
    samples=e[ids]
    by_mode={mode:[] for mode in MODES}
    for product in products:
        values=np.asarray(
            [float(product.yield_(float(energy))) for energy in samples],
            dtype=np.float64
        )
        by_mode[product.emission_mode].append(values)
    analysis=assess_evaluated_neutron_product_yields(samples,by_mode)
    return {
        "nuclide":nuclide,
        "native_temperature_label":temperature,
        "product_tabulated_domains_overlap_positive_fission":True,
        "no_valid_evaluated_yield_comparison":False,
        "original_evaluated_product_modes":directly_evaluated_modes,
        "native_derived_evaluated_product_modes":derived_modes,
        "mt18_redundant_reaction":bool(fission.redundant),
        "derived_total_is_independent_evaluated_record_not_a_sum":(
            derived_modes["total"]>0
        ),
        **analysis,
    }


def audit_executed_fission_product_yield(
    summary_path: Path, nuclear_index: Path
) -> dict:
    summary_path, nuclear_index=Path(summary_path),Path(nuclear_index)
    preceding=audit_executed_neutron_reaction_xs(summary_path,nuclear_index)
    if (preceding["all_present_native_reaction_xs_numerically_valid"] is not True
            or preceding["verified_evaluated_neutron_nuclide_count"]<1
            or preceding["native_hdf5_digests_provenance_verified"] is not True):
        raise ValueError("missing real neutron reaction XS numerical provenance")
    isotopes={
        x["nuclide"]:x for x in preceding["per_executed_neutron_nuclide"]
        if any(
            t["mt"]==18 and t["reaction_present"]
            for t in x["reaction_tables"]
        )
    }
    if not isotopes:
        raise ValueError("no evaluated fission neutron source in executed core")
    indexed={}
    for item in _library_entries(nuclear_index):
        if item.data_type=="neutron":
            for isotope in item.materials:
                if isotope in isotopes:
                    if isotope in indexed:
                        raise ValueError("duplicate physically used fission neutron isotope")
                    indexed[isotope]=Path(item.resolved_path)
    if set(indexed)!=set(isotopes):
        raise ValueError("fissionable material nuclear-data index inconsistent")
    rows=[]
    for isotope in sorted(isotopes):
        file=indexed[isotope]
        if _sha(file)!=isotopes[isotope]["evaluated_hdf5_sha256"]:
            raise ValueError("source fission yield evaluation HDF5 changed since XS audit")
        data=openmc.data.IncidentNeutron.from_hdf5(file)
        if data.name!=isotope:
            raise ValueError("neutron yield HDF5 isotope identity mismatch")
        row={
            "nuclide":isotope,
            "evaluated_neutron_hdf5_sha256":_sha(file),
            "per_evaluated_temperature":[
                _sample_fission_reaction(data,isotope,t)
                for t in data.temperatures
            ],
        }
        rows.append(row)
    samples=[q for r in rows for q in r["per_evaluated_temperature"]]
    comparable=[x for x in samples if x.get(
        "all_total_prompt_delayed_yield_modes_available",False
    )]
    max_mismatch=(
        max(x["maximum_relative_total_minus_prompt_delayed_yield_mismatch"]
            for x in comparable)
        if comparable else None
    )
    return {
        "audit":"REAL_EVALUATED_CORE_FISSION_NEUTRON_PRODUCT_YIELD_NUMERICAL_CONSISTENCY",
        "actual_four_seed_openmc_summary_sha256":_sha(summary_path),
        "nuclear_data_cross_sections_xml_sha256":_sha(nuclear_index),
        "actual_evaluated_fissionable_neutron_nuclide_count":len(rows),
        "actual_evaluated_fission_yield_temperature_records":len(samples),
        "fully_comparable_fission_yield_temperature_records":len(comparable),
        "all_fission_yield_records_numerically_comparable":
            len(samples)==len(comparable),
        "maximum_evaluated_function_relative_yield_mismatch":max_mismatch,
        "real_nuclear_product_function_checks_executed":True,
        "all_compared_evaluated_yields_under_consistency_screen":bool(comparable),
        "per_evaluated_fissionable_isotope":rows,
        "energy_domain_extrapolation_performed":False,
        "nuclear_yield_curves_modified":False,
        "neutron_fission_source_stationarity_qualified":False,
        "beta_eff_or_kinetics_inferred":False,
        "hot_operating_core_physics_qualified":False,
        "interpretation":(
            "Read-only incident-energy fission product yield function QA "
            "of actually used evaluated HDF5 nuclides; independent total, "
            "prompt, delayed groups can be compared only where all modes "
            "and genuine shared incident-energy support exist. This is not "
            "adjoint beta_eff, source-statistics uncertainty, a historical "
            "NERVA benchmark or reactor physics design."
        ),
    }


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--cross-sections",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    report=audit_executed_fission_product_yield(a.summary,a.cross_sections)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("Actual fissionable evaluated neutron nuclides:",
          report["actual_evaluated_fissionable_neutron_nuclide_count"])
    print("Native fission product yield temperature records:",
          report["actual_evaluated_fission_yield_temperature_records"])
    print("Records with comparable total/prompt/delayed yields:",
          report["fully_comparable_fission_yield_temperature_records"])
    print("Largest native evaluated yield consistency mismatch:",
          report["maximum_evaluated_function_relative_yield_mismatch"])
    print("Hot-core physical qualification and adjoint beta_eff: NOT CLAIMED")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
