"""Generate a traceable engineering report for a completed NERVA full run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Markdown and JSON reports for a completed NERVA run."
    )
    parser.add_argument(
        "full_run_root",
        type=Path,
        help="Root containing transport/ and analysis/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output; default is NERVA_RUN_REPORT.md under the run root.",
    )
    return parser.parse_args()


def _load(path: Path, required: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _f(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}g}"


def main() -> int:
    args = parse_args()
    root = args.full_run_root
    transport = root / "transport"
    analysis = root / "analysis"

    provenance = _load(transport / "transport_provenance.json")
    data_report = _load(transport / "nuclear_data_report.json")
    temp_audit = _load(
        transport / "nuclear_data_temperature_audit.json",
        required=False,
    )
    qa = _load(analysis / "analysis_validation.json")
    power = _load(analysis / "power" / "metadata.json")
    diagnostics = _load(
        analysis / "openmc_diagnostics" / "openmc_diagnostics.json"
    )
    element = _load(
        analysis / "element_channels" / "element_channel_summary.json"
    )
    uncertainty = _load(
        analysis / "hottest_channel_uncertainty.json",
        required=False,
    )
    fuel = _load(analysis / "fuel" / "thermal_summary.json")
    tie = _load(analysis / "tie" / "tie_thermal_summary.json")

    keff = diagnostics.get("keff", {})
    mesh_err = diagnostics["mesh_heating_relative_error"]
    energy = diagnostics.get("energy_deposition", {})
    material_fraction = diagnostics.get("material_power_fractions", {})

    material_lines = [
        f"- {name}: {_f(100.0 * value, 5)} %"
        for name, value in sorted(
            material_fraction.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    hot = element["hottest_channel"]

    lines = [
        "# NERVA OpenMC Engineering Run Report",
        "",
        "> Non-calibrated low-enrichment NERVA-derived demonstrator.",
        "> This is a transport/thermal software result, not a validated historical critical experiment.",
        "",
        "## Reproducibility",
        "",
        f"- Repository commit: {provenance.get('git_commit')}",
        f"- OpenMC version: {provenance.get('openmc_version')}",
        f"- Python: {provenance.get('python_version', '').splitlines()[0]}",
        f"- Assembly: {provenance.get('assembly')}",
        f"- Nuclear data: {provenance['nuclear_data']['cross_sections_xml']}",
        f"- Nuclear-data SHA-256: {provenance['nuclear_data']['cross_sections_sha256']}",
        f"- Statepoint SHA-256: {(provenance.get('statepoint') or {}).get('sha256')}",
        f"- QA status: {qa.get('status')}",
        "",
        "## OpenMC run controls",
        "",
        f"- Particles/batch: {_f(provenance['config']['particles'])}",
        f"- Batches: {_f(provenance['config']['batches'])}",
        f"- Inactive batches: {_f(provenance['config']['inactive'])}",
        f"- Photon transport: {provenance['config']['photon_transport']}",
        f"- Energy groups: {_f(provenance['config']['diagnostic_energy_groups'])}",
        f"- Axial diagnostic bins: {_f(provenance['config']['axial_mesh_bins'])}",
        f"- Tally rel-error trigger: {provenance['config'].get('tally_rel_err_trigger')}",
        "",
        "## Nuclear-data checks",
        "",
        f"- Library preflight ready: {data_report.get('ready')}",
        f"- Missing neutron nuclides: {len(data_report.get('missing_neutron_nuclides', []))}",
        f"- Missing referenced files: {len(data_report.get('referenced_files_missing', []))}",
        (
            "- Explicit material temperature range coverage: "
            + str(
                None
                if temp_audit is None
                else temp_audit.get("model_explicit_temperature_coverage_ok")
            )
        ),
        "",
        "## Monte Carlo diagnostics",
        "",
        f"- Reported k_eff (diagnostic only): {_f(keff.get('mean'), 8)} +/- {_f(keff.get('std_dev'), 4)}",
        f"- 3-D max/mean-active power peaking: {_f(diagnostics['mesh_peaking']['max_to_mean_active'], 6)}",
        f"- Axial power peaking: {_f(diagnostics['axial_power_peaking'], 6)}",
        f"- Median mesh-heating rel. error: {_f(mesh_err['median_relative_error'], 6)}",
        f"- 95th-percentile mesh-heating rel. error: {_f(mesh_err['p95_relative_error'], 6)}",
        f"- Maximum mesh-heating rel. error: {_f(mesh_err['max_relative_error'], 6)}",
        "",
        "## Power normalization and deposition",
        "",
        f"- Requested reactor power: {_f(power['requested_power_MW'])} MW",
        f"- Fuel-solid deposited power: {_f(power['fuel_power_W'] / 1.0e6)} MW",
        f"- Tie-solid deposited power: {_f(power['tie_power_W'] / 1.0e6)} MW",
        f"- Direct fuel-channel H2 nuclear heating: {_f(power.get('fuel_hydrogen_direct_power_W', 0.0) / 1.0e6)} MW",
        f"- Direct tie H2 nuclear heating: {_f(power.get('tie_hydrogen_direct_power_W', 0.0) / 1.0e6)} MW",
        f"- Mesh heating-local total: {_f(energy.get('mesh_heating_local_W', 0.0) / 1.0e6)} MW",
        f"- Mesh heating total: {_f(energy.get('mesh_heating_W', 0.0) / 1.0e6)} MW",
        "",
        "### Material local-heating fractions",
        "",
        *(material_lines or ["- unavailable"]),
        "",
        "## Element x 19-channel reconstruction",
        "",
        f"- Fuel-element instances: {_f(element['fuel_element_instances'])}",
        f"- Channels solved: {_f(element['total_channels'])}",
        f"- Integrated element power max/mean: {_f(element['fuel_element_max_to_mean_integrated_power'], 6)}",
        f"- Hottest element instance: {_f(hot['element_instance'])}",
        f"- Hottest channel: {_f(hot['channel'])}",
        f"- Maximum channel outlet H2 temperature: {_f(element['maximum_channel_outlet_temperature_K'])} K",
        f"- Maximum reconstructed peak fuel temperature: {_f(element['maximum_peak_fuel_temperature_K'])} K",
        "",
        "## Representative fuel-channel solve",
        "",
        f"- H2 mass flow: {_f(fuel['fuel_mass_flow_kg_s'])} kg/s",
        f"- Outlet temperature: {_f(fuel['outlet_temperature_K'])} K",
        f"- Outlet pressure: {_f(fuel['outlet_pressure_Pa'] / 1.0e6)} MPa",
        f"- Maximum peak-fuel screening temperature: {_f(fuel['maximum_peak_fuel_temperature_K'])} K",
        "",
        "## Tie-tube counterflow solve",
        "",
        f"- H2 mass flow: {_f(tie['tie_mass_flow_kg_s'])} kg/s",
        f"- Outlet temperature: {_f(tie['outlet_temperature_K'])} K",
        f"- Outlet pressure: {_f(tie['outlet_pressure_Pa'] / 1.0e6)} MPa",
        f"- Supply heat fraction: {_f(tie['supply_heat_fraction'], 6)}",
        "",
    ]

    if uncertainty is not None:
        lines.extend(
            [
                "## Hottest-channel Monte Carlo sensitivity envelope",
                "",
                (
                    "- Fuel-element integrated heating relative sigma: "
                    + _f(
                        uncertainty[
                            "fuel_element_integrated_heating_relative_sigma"
                        ],
                        6,
                    )
                ),
                (
                    "- Direct-H2 integrated heating relative sigma: "
                    + _f(
                        uncertainty[
                            "direct_hydrogen_integrated_heating_relative_sigma"
                        ],
                        6,
                    )
                ),
                (
                    "- Peak fuel T, nominal / -1sigma / +1sigma: "
                    + _f(
                        uncertainty["nominal"][
                            "maximum_peak_fuel_temperature_K"
                        ]
                    )
                    + " / "
                    + _f(
                        uncertainty["minus_1sigma_screening"][
                            "maximum_peak_fuel_temperature_K"
                        ]
                    )
                    + " / "
                    + _f(
                        uncertainty["plus_1sigma_screening"][
                            "maximum_peak_fuel_temperature_K"
                        ]
                    )
                    + " K"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Fidelity boundary",
            "",
            "- Fuel loading remains a deliberately non-calibrated low-enrichment surrogate.",
            "- Reported k_eff is diagnostic only; no enrichment or control-worth tuning is performed.",
            "- Fuel-element integrated powers are direct OpenMC repeated-cell tally outputs.",
            "- Element-specific axial reconstruction currently applies the global OpenMC axial shape separably.",
            "- Solid wall power is divided equally among the 19 coolant channels within one fuel element.",
            "- Direct H2 nuclear heating is tallied separately and added to fluid enthalpy, not wall heat flux.",
            "- The hottest-channel 1-sigma result is a screening sensitivity envelope, not covariance-aware propagation.",
            "- A real statepoint depends on the external nuclear-data library documented above.",
            "",
            "## Machine-readable QA",
            "",
            str(analysis / "analysis_validation.json"),
            "",
        ]
    )

    output = (
        args.output
        if args.output is not None
        else root / "NERVA_RUN_REPORT.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_output = output.with_suffix(".json")
    json_output.write_text(
        json.dumps(
            {
                "transport_provenance": provenance,
                "nuclear_data_report": data_report,
                "temperature_audit": temp_audit,
                "analysis_validation": qa,
                "power": power,
                "openmc_diagnostics": diagnostics,
                "element_channels": element,
                "uncertainty": uncertainty,
                "fuel_thermal": fuel,
                "tie_thermal": tie,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("NERVA engineering run report written:")
    print(f"  Markdown: {output}")
    print(f"  JSON: {json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
