"""Real OpenMC reactor-core neutron population-size physics validation.

The existing 3000-vs-6000 source-population study compares one 4x4x4
source-space binning level and final eigenvalue estimates. This reads the
actual OpenMC HDF5 source banks at nested 2x2x2, 4x4x4 and 8x8x8 spatial
resolutions, then checks native absorption and nu-fission TRACKLENGTH
material/incident-energy tallies for BOTH unchanged-core populations.

It never treats source-particle sites within a generation as independent
fission histories. Population stability is NOT proof of fission-source
stationarity, keff accuracy or historical NASA NERVA reactor fidelity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .source_population_sensitivity import assess_population_sensitivity
from .source_resolution_audit import audit_real_transport_summary
from .source_energy_audit import audit_actual_source_energy_summary
from .core_reaction_ensemble import audit_real_neutron_reactions, SCORES


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_population_physics(
    summaries: tuple[dict, dict],
    spatial: tuple[dict, dict],
    source_energy: tuple[dict, dict],
    reactions: tuple[dict, dict],
    *,
    summary_shas: tuple[str, str],
) -> dict:
    """Numerical population comparison after independent HDF5 audits.

    Caller is responsible for running all three true OpenMC HDF5 audits.
    Assertions here reconcile independently extracted physics observables.
    They are exact consistency identities, NOT an empirical convergence
    acceptance rule.
    """
    if (len(summaries) != 2 or len(spatial) != 2
            or len(source_energy) != 2 or len(reactions) != 2
            or len(summary_shas) != 2):
        raise ValueError("exactly two physical source-population levels required")
    nlo = summaries[0].get("particles_per_generation")
    nhi = summaries[1].get("particles_per_generation")
    if (not isinstance(nlo, int) or not isinstance(nhi, int)
            or nlo < 100 or nhi <= nlo):
        raise ValueError("source population must strictly increase")
    if (summaries[0].get("cross_sections_xml_sha256")
            != summaries[1].get("cross_sections_xml_sha256")):
        raise ValueError("different evaluated data invalidate population physics")
    if summaries[0].get("saved_source_bank_batches") != summaries[1].get(
            "saved_source_bank_batches"):
        raise ValueError("different source generations invalidate population physics")
    if summaries[0].get("inactive_batches") != summaries[1].get("inactive_batches"):
        raise ValueError("different inactive neutron generations")
    if summaries[0].get("batches") != summaries[1].get("batches"):
        raise ValueError("different active neutron generations")
    for key in ("spatial_lower_cm", "spatial_upper_cm"):
        if summaries[0]["independent_seed_source_bank_shape_diagnostics"].get(key) != (
                summaries[1]["independent_seed_source_bank_shape_diagnostics"].get(key)):
            raise ValueError("different physical source scoring envelope")
    if reactions[0].get("material_ids") != reactions[1].get("material_ids"):
        raise ValueError("material neutron tally identities differ")
    if reactions[0].get("incident_energy_bins_eV") != (
            reactions[1].get("incident_energy_bins_eV")):
        raise ValueError("incident-neutron energy reaction tally intervals differ")
    if source_energy[0].get("neutron_energy_bin_edges_eV") != (
            source_energy[1].get("neutron_energy_bin_edges_eV")):
        raise ValueError("fission emission neutron energy scoring bins differ")
    if any(not isinstance(k, str) or len(k) != 64 for k in summary_shas):
        raise ValueError("transport summary provenance hash is malformed")

    resolved = []
    for summary, positions, energies, reactions_arm, expected_sha in zip(
            summaries, spatial, source_energy, reactions, summary_shas
    ):
        if (positions.get("real_transport_summary_sha256") != expected_sha
                or energies.get("real_transport_summary_sha256") != expected_sha
                or reactions_arm.get("real_transport_summary_sha256") != expected_sha):
            raise ValueError("statepoint audits not derived from the same transport summary")
        if not positions.get("grid_resolution_audit_passed"):
            raise ValueError("genuine nested source-space conservation not demonstrated")
        levels = positions.get("resolutions", [])
        if [item.get("bins_per_axis") for item in levels] != [2,4,8]:
            raise ValueError("required real nested 2/4/8 source grids missing")
        reference = summary.get("multi_checkpoint_source_bank_trajectory", {})
        snapshots = reference.get("runs", [])
        final_files = [row.get("statepoint_sha256") for row in summary.get("runs",[])]
        sha_matrix = [[site.get("statepoint_sha256") for site in row.get("snapshots",[])]
                      for row in snapshots]
        if (not final_files or len(final_files) < 4
                or len(sha_matrix) != len(final_files)
                or [item[-1] for item in sha_matrix] != final_files):
            raise ValueError("incomplete physical source-bank and final statepoint provenance")
        if (levels[0]["all_checkpoint_statepoint_sha256"] != sha_matrix
                or any(level["all_checkpoint_statepoint_sha256"] != sha_matrix
                       for level in levels)):
            raise ValueError("spatial histograms do not use actual saved HDF5 source banks")
        if (energies.get("saved_source_batches") != summary.get("saved_source_bank_batches")
                or [[p.get("statepoint_sha256") for p in entry.get("snapshots",[])]
                    for entry in energies.get("runs",[])] != sha_matrix):
            raise ValueError("neutron emission spectra use different HDF5 source particles")
        if (reactions_arm.get("statepoint_sha256") != final_files
                or reactions_arm.get("nuclear_data_index_sha256")
                != summary.get("cross_sections_xml_sha256")):
            raise ValueError("reaction rates use different evaluated neutron data or HDF5")
        coarse = np.array(levels[0]["mean_cross_seed_tv_per_checkpoint"],float)
        mid = np.array(levels[1]["mean_cross_seed_tv_per_checkpoint"],float)
        fine = np.array(levels[2]["mean_cross_seed_tv_per_checkpoint"],float)
        legacy = np.array(reference["between_seed_mean_total_variation_by_batch"],float)
        emission = np.array(energies["mean_cross_seed_energy_spectrum_tv_by_checkpoint"],
                            float)
        if (not (len(coarse) == len(mid) == len(fine) == len(legacy) == len(emission)
                 == len(summary["saved_source_bank_batches"]))
                or not all(np.isfinite(x).all() for x in
                           (coarse,mid,fine,legacy,emission))
                or np.any(coarse < 0) or np.any(fine > 1+1e-12)
                or np.any(coarse > mid+3e-12) or np.any(mid > fine+3e-12)
                or not np.allclose(mid,legacy,rtol=0,atol=3e-12)):
            raise ValueError("actual OpenMC source TV inconsistent across spatial grids")
        n = summary["particles_per_generation"]
        last = [row["snapshots"][-1] for row in snapshots]
        frac = np.array([float(x["inverse_simpson_distinct_position_count"])/n
                         for x in last])
        if not np.isfinite(frac).all() or np.any(frac <= 0) or np.any(frac > 1):
            raise ValueError("invalid actual neutron source position diversity")
        if (reactions_arm.get("independent_fission_source_stationarity_demonstrated")
                is not False
                or positions.get("source_stationarity_validated") is not False
                or energies.get("fission_source_spectral_stationarity_validated")
                is not False):
            raise ValueError("reaction/source auditing must not certify reactor stationarity")
        summary_rates = {}
        for score in SCORES:
            reaction = reactions_arm["reactions"][score]
            if len(reaction["per_seed_global_rate_per_source"]) != len(final_files):
                raise ValueError("missing source-independent neutron reaction samples")
            v = float(reaction["mean_global_reaction_per_source"])
            cv = float(reaction["between_seed_coefficient_of_variation"])
            tv = float(reaction["mean_pairwise_incident_energy_spectrum_tv"])
            if not all(math.isfinite(x) for x in (v,cv,tv)) or v <= 0 or cv < 0 or not 0 <= tv <= 1:
                raise ValueError("unphysical neutron reaction score or energy spectrum")
            summary_rates[score] = {
                "mean_reaction_rate_per_source":v,
                "descriptive_between_seed_cv":cv,
                "incident_neutron_reaction_energy_tv":tv,
            }
        resolved.append({
            "particles_per_generation":n,
            "source_cross_seed_tv_per_checkpoint": {
                str(side):items.tolist()
                for side,items in ((2,coarse),(4,mid),(8,fine))
            },
            "final_cross_seed_tv_8cube":float(fine[-1]),
            "final_source_emission_energy_tv":float(emission[-1]),
            "mean_effective_distinct_position_fraction":float(frac.mean()),
            "minimum_effective_distinct_position_fraction_all_checkpoints":
                float(levels[2]["minimum_effective_distinct_position_fraction"]),
            "source_position_diversity_screen":positions["source_diversity_screen_status"],
            "reactions":summary_rates,
            "underlying_transport_summary_sha256":expected_sha,
            "source_final_statepoint_sha256":final_files,
        })

    lower,higher = resolved
    return {
        "analysis":"REAL_OPENMC_CORE_MULTIPOPULATION_NEUTRON_PHYSICS_INTEGRITY",
        "nuclear_data_index_sha256":summaries[0]["cross_sections_xml_sha256"],
        "model_unchanged":True,
        "population_levels":resolved,
        "change_in_final_spatial_8cube_source_TV":
            higher["final_cross_seed_tv_8cube"]-lower["final_cross_seed_tv_8cube"],
        "change_in_final_source_emission_energy_TV":
            higher["final_source_emission_energy_tv"]-lower["final_source_emission_energy_tv"],
        "change_in_final_effective_distinct_position_fraction":
            higher["mean_effective_distinct_position_fraction"]-
            lower["mean_effective_distinct_position_fraction"],
        "difference_in_mean_absorption_reaction_rate_per_source":
            higher["reactions"]["absorption"]["mean_reaction_rate_per_source"]-
            lower["reactions"]["absorption"]["mean_reaction_rate_per_source"],
        "difference_in_mean_nu_fission_reaction_rate_per_source":
            higher["reactions"]["nu-fission"]["mean_reaction_rate_per_source"]-
            lower["reactions"]["nu-fission"]["mean_reaction_rate_per_source"],
        "neutron_tally_and_source_histogram_identity_passed":True,
        "source_stationarity_demonstrated":False,
        "reactor_physical_nasa_benchmark_validated":False,
        "interpretation":(
            "Genuine OpenMC saved source-bank x/y/z and emission energies, and "
            "material- and incident-energy resolved absorption and nu-fission "
            "TRACKLENGTH scores, all reconciled to the exact same executed "
            "core and evaluated nuclear data. Different RNG seeds between "
            "population arms confound causality. Fission source descendants "
            "within each eigenvalue generation are genealogically correlated. "
            "No per-particle-independent p-value, source convergence, neutron "
            "local heating, operating-temperature reactor physics or historical "
            "NASA NERVA validation is inferred."
        ),
    }


def verify_real_population_physics(low_root: Path, high_root: Path) -> dict:
    baseline = assess_population_sensitivity(low_root,high_root)
    reports = []
    spatial = []
    energy = []
    reactions = []
    shas = []
    for root in (Path(low_root),Path(high_root)):
        file = root/"real_core_transport_summary.json"
        summary = json.loads(file.read_text(encoding="utf-8"))
        reports.append(summary)
        shas.append(_digest(file))
        spatial.append(audit_real_transport_summary(file))
        energy.append(audit_actual_source_energy_summary(file))
        reactions.append(audit_real_neutron_reactions(file))
    result = compare_population_physics(
        tuple(reports),tuple(spatial),tuple(energy),tuple(reactions),
        summary_shas=tuple(shas),
    )
    if baseline["evaluated_cross_sections_sha256"] != result["nuclear_data_index_sha256"]:
        raise ValueError("legacy and independent evaluated XS audits disagree")
    for n, prefix in enumerate(("low_population","high_population")):
        expected = baseline[prefix]
        observed = result["population_levels"][n]
        if (expected["particles_per_generation"] != observed["particles_per_generation"]
                or not math.isclose(
                    expected["final_between_seed_source_total_variation"],
                    observed["source_cross_seed_tv_per_checkpoint"]["4"][-1],
                    abs_tol=3e-12,rel_tol=0)
                or not math.isclose(
                    expected["final_effective_position_fraction_of_bank"],
                    observed["mean_effective_distinct_position_fraction"],
                    abs_tol=3e-12,rel_tol=0)):
            raise ValueError("legacy population results disagree with authentic HDF5 audit")
    result["prior_population_sensitivity_metrics"] = baseline
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--low",type=Path,required=True)
    p.add_argument("--high",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=verify_real_population_physics(args.low,args.high)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                           encoding="utf-8")
    for level in result["population_levels"]:
        print("REAL OpenMC neutron source population",level["particles_per_generation"],
              "final 8^3 spatial TV",level["final_cross_seed_tv_8cube"],
              "final emission spectrum TV",level["final_source_emission_energy_tv"],
              "effective source-site fraction",level["mean_effective_distinct_position_fraction"])
        for name,item in level["reactions"].items():
            print("  neutron",name,"rate/source",item["mean_reaction_rate_per_source"],
                  "cross-seed CV",item["descriptive_between_seed_cv"])
    print("Fission-source convergence and NASA NERVA physically verified: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
