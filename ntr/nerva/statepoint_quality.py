"""Independent statepoint/provenance/uncertainty audit for OpenMC feedback.

A successful analysis QA only establishes internal consistency of derived
arrays. Before using a new material-heating shape in Picard coupling,
cross-check its identity against the EXACT newly generated OpenMC
statepoint, its source cross_sections.xml, and the normalization metadata.

This audit is necessary but not sufficient for real NERVA validation:
binary HDF5 signature, neutron XS correctness, statistical independence,
and historical experimental benchmarking remain distinct concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class SnapshotQuality:
    statepoint_sha256: str
    nuclear_data_sha256: str
    normalized_power_w: float
    source_rate_per_s: float
    heating_p95_relative_error: float
    fuel_temperature_k: float
    hydrogen_temperature_k: float
    hydrogen_density_g_cm3: float
    analysis_qa_passed: bool


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def _number(value: object, label: str, *, positive: bool = True) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(out) or (out <= 0 if positive else out < 0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return out


def _same_file(first: str | Path, second: Path, label: str) -> None:
    if not first or Path(first).resolve() != second.resolve():
        raise RuntimeError(f"{label} does not reference current statepoint/analysis")


def audit_fresh_transport(
    *, statepoint: Path, cross_sections_xml: Path,
    transport_provenance: Path, analysis_root: Path,
    requested_power_w: float, fuel_temperature_k: float,
    hydrogen_temperature_k: float, hydrogen_density_g_cm3: float,
    max_heating_p95_relative_error: float = 0.5,
) -> SnapshotQuality:
    """Fail-closed audit for a new actual transport and thermal reconstruction.

    Uses existing transport_provenance.json, power/metadata.json and
    analysis_validation.json, NEVER invents or reuses absent results.
    """
    statepoint = Path(statepoint).resolve()
    cross_sections_xml = Path(cross_sections_xml).resolve()
    analysis_root = Path(analysis_root).resolve()
    max_rel = _number(max_heating_p95_relative_error,
                      "max p95 relative heating error", positive=False)
    if max_rel > 10:
        raise ValueError("statistical uncertainty ceiling above 1000% is invalid")
    watts = _number(requested_power_w, "requested power")
    fuel_temp = _number(fuel_temperature_k, "fuel temperature")
    gas_temp = _number(hydrogen_temperature_k, "hydrogen temperature")
    density = _number(hydrogen_density_g_cm3, "hydrogen density")
    if not statepoint.is_file() or statepoint.suffix.lower() != ".h5":
        raise FileNotFoundError("new OpenMC statepoint .h5 not found")
    if not cross_sections_xml.is_file():
        raise FileNotFoundError("nuclear cross_sections.xml not found")
    transport = json.loads(Path(transport_provenance).read_text(encoding="utf-8"))
    metadata = json.loads(
        (analysis_root / "power" / "metadata.json").read_text(encoding="utf-8")
    )
    qa = json.loads(
        (analysis_root / "analysis_validation.json").read_text(encoding="utf-8")
    )
    sp = transport.get("statepoint")
    if not isinstance(sp, dict):
        raise RuntimeError("transport provenance has no completed statepoint")
    _same_file(sp.get("path"), statepoint, "transport provenance")
    digest = sha256(statepoint)
    if sp.get("sha256") != digest:
        raise RuntimeError("statepoint hash does not match transport provenance")
    if sp.get("size_bytes") != statepoint.stat().st_size:
        raise RuntimeError("statepoint size does not match transport provenance")
    nuclear = transport.get("nuclear_data")
    if not isinstance(nuclear, dict):
        raise RuntimeError("transport nuclear-data provenance missing")
    nuclear_digest = sha256(cross_sections_xml)
    if nuclear.get("cross_sections_sha256") != nuclear_digest:
        raise RuntimeError("nuclear-data XML checksum mismatch")
    if Path(nuclear.get("cross_sections_xml", "")).resolve() != cross_sections_xml:
        raise RuntimeError("transport used different nuclear cross_sections.xml")
    if nuclear.get("explicit_temperature_coverage_ok") is not True:
        raise RuntimeError("temperature-dependent neutron cross sections not audited")
    cfg = transport.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError("transport configuration missing")
    for name, expected in (
        ("fuel_temperature_k", fuel_temp),
        ("hydrogen_temperature_k", gas_temp),
        ("hydrogen_density_g_cm3", density),
    ):
        actual = _number(cfg.get(name), "transport "+name)
        if not math.isclose(actual, expected, rel_tol=5e-11, abs_tol=1e-10):
            raise RuntimeError(f"transport input {name} differs from Picard request")

    # The analysis metadata is generated by the EXISTING postprocessor:
    # a PASS status alone does not establish that it belongs to this run.
    _same_file(metadata.get("statepoint"), statepoint, "power metadata")
    normalized = _number(metadata.get("normalized_power_W"),
                         "normalized power")
    if not math.isclose(normalized, watts, rel_tol=1e-8):
        raise RuntimeError("normalized heating does not match prescribed power")
    power_mw = _number(metadata.get("requested_power_MW"),
                       "metadata requested MW")
    if not math.isclose(power_mw * 1e6, watts, rel_tol=1e-8):
        raise RuntimeError("metadata requested power differs from prescribed power")
    source_rate = _number(metadata.get("source_rate_per_s"), "source rate")
    mesh_p95 = _number(metadata.get("mesh_heating_p95_relative_error"),
                       "mesh p95 relative error", positive=False)
    if qa.get("status") != "PASS":
        raise RuntimeError("OpenMC analysis QA does not pass")
    checks = qa.get("checks")
    if not isinstance(checks, dict) or not checks or any(x is not True for x in checks.values()):
        raise RuntimeError("analysis checks incomplete or rejected")
    if Path(qa.get("analysis_root", "")).resolve() != analysis_root:
        raise RuntimeError("QA package belongs to another analysis directory")
    metrics = qa.get("diagnostic_values")
    if not isinstance(metrics, dict):
        raise RuntimeError("analysis QA diagnostics absent")
    qa_error = _number(metrics.get("heating_p95_relative_error"),
                       "QA heating p95 relative error", positive=False)
    qa_power_mw = _number(metrics.get("requested_power_MW"), "QA requested power")
    if not math.isclose(qa_error, mesh_p95, rel_tol=1e-6, abs_tol=1e-10):
        raise RuntimeError("mesh and QA Monte Carlo uncertainty disagree")
    if not math.isclose(qa_power_mw * 1e6, watts, rel_tol=1e-8):
        raise RuntimeError("QA normalized reactor power differs")
    if mesh_p95 > max_rel:
        raise RuntimeError(
            f"OpenMC heating p95 relative error {mesh_p95:.4g} exceeds "
            f"configured limit {max_rel:.4g}; increase sampling or explicitly "
            "revise tolerance with documented justification"
        )
    return SnapshotQuality(
        statepoint_sha256=digest, nuclear_data_sha256=nuclear_digest,
        normalized_power_w=normalized, source_rate_per_s=source_rate,
        heating_p95_relative_error=mesh_p95,
        fuel_temperature_k=fuel_temp,
        hydrogen_temperature_k=gas_temp,
        hydrogen_density_g_cm3=density, analysis_qa_passed=True,
    )
