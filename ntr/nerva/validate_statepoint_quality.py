"""Unit tests for statepoint identity, XS provenance and uncertainty gates.

Uses deliberately FAKE byte payloads with correct JSON provenance shape.
No OpenMC executable, nuclear-data library or actual neutron transport is
invoked. PASS asserts *only* software metadata-validation correctness.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile

from .statepoint_quality import audit_fresh_transport


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def expect_rejected(run, *, keyword: str) -> None:
    try:
        run()
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        if keyword.lower() not in str(exc).lower():
            raise AssertionError(f"expected error containing {keyword!r}; got {exc!r}") from exc
    else:
        raise AssertionError(f"failed to reject incompatible input: {keyword}")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sp = root / "transport"/"statepoint.80.h5"
        sp.parent.mkdir()
        sp.write_bytes(b"TEST-FIXTURE-NOT-A-REAL-OPENMC-HDF5-STATEPOINT")
        cross = root/"library"/"cross_sections.xml"
        cross.parent.mkdir()
        cross.write_text("<cross_sections />", encoding="utf-8")
        transport_path = root/"transport"/"transport_provenance.json"
        analysis = root/"analysis"
        meta_path = analysis/"power"/"metadata.json"
        qa_path = analysis/"analysis_validation.json"

        transport = {
            "statepoint": {
                "path": str(sp.resolve()),
                "sha256": _sha(sp),
                "size_bytes": sp.stat().st_size,
            },
            "config": {
                "fuel_temperature_k": 600.,
                "hydrogen_temperature_k": 350.,
                "hydrogen_density_g_cm3": 0.001,
            },
            "nuclear_data": {
                "cross_sections_xml": str(cross.resolve()),
                "cross_sections_sha256": _sha(cross),
                "explicit_temperature_coverage_ok": True,
            },
        }
        meta = {
            "statepoint": str(sp.resolve()),
            "requested_power_MW": 1.0,
            "normalized_power_W": 1e6,
            "source_rate_per_s": 3.2e17,
            "mesh_heating_p95_relative_error": 0.11,
        }
        qa = {
            "analysis_root": str(analysis.resolve()),
            "status": "PASS",
            "checks": {"normalized_power_closure": True,
                       "source_rate_consistency": True,
                       "element_channel_power_closure": True},
            "diagnostic_values": {
                "requested_power_MW": 1.0,
                "heating_p95_relative_error": 0.11,
            },
        }
        def run():
            return audit_fresh_transport(
                statepoint=sp, cross_sections_xml=cross,
                transport_provenance=transport_path,
                analysis_root=analysis, requested_power_w=1e6,
                fuel_temperature_k=600.,
                hydrogen_temperature_k=350.,
                hydrogen_density_g_cm3=0.001,
                max_heating_p95_relative_error=0.2,
            )

        def reset():
            _write(transport_path, transport)
            _write(meta_path, meta)
            _write(qa_path, qa)

        reset()
        good = run()
        assert good.statepoint_sha256 == _sha(sp)
        assert good.nuclear_data_sha256 == _sha(cross)
        assert math.isclose(good.heating_p95_relative_error, 0.11)
        assert good.analysis_qa_passed
        assert good.normalized_power_w == 1e6

        # A different, validly named statepoint must not be accepted.
        bogus = root/"transport"/"other.h5"
        bogus.write_bytes(b"OTHER-FAKE-STATEPOINT")
        expect_rejected(lambda: audit_fresh_transport(
            statepoint=bogus, cross_sections_xml=cross,
            transport_provenance=transport_path,
            analysis_root=analysis, requested_power_w=1e6,
            fuel_temperature_k=600.,
            hydrogen_temperature_k=350.,
            hydrogen_density_g_cm3=0.001,
        ), keyword="provenance")
        reset()
        # Tampered OpenMC statepoint bytes.
        sp.write_bytes(b"CHANGED-BYTES")
        expect_rejected(run, keyword="statepoint hash")
        sp.write_bytes(b"TEST-FIXTURE-NOT-A-REAL-OPENMC-HDF5-STATEPOINT")
        reset()

        # Tampered source XML or swapped source XS: abort.
        cross.write_text("<changed/>", encoding="utf-8")
        expect_rejected(run, keyword="checksum")
        cross.write_text("<cross_sections />", encoding="utf-8")
        reset()

        for label, base, path, patch, keyword in [
            ("wrong heating source", meta, meta_path,
             {"statepoint": str(bogus)}, "power metadata"),
            ("wrong normalization", meta, meta_path,
             {"normalized_power_W": 2e6}, "normalized heating"),
            ("high MC noise", meta, meta_path,
             {"mesh_heating_p95_relative_error": 0.4},
             "uncertainty disagree"),
            ("zero MC source rate", meta, meta_path,
             {"source_rate_per_s": 0.}, "source rate"),
            ("fake QA analysis", qa, qa_path,
             {"analysis_root": str(root/"wrong")}, "another analysis"),
            ("failed QA status", qa, qa_path,
             {"status": "FAIL"}, "does not pass"),
            ("incomplete QA", qa, qa_path,
             {"checks": {"power": False}}, "incomplete"),
            ("wrong fuel material temperature", transport, transport_path,
             {"config": {**transport["config"],"fuel_temperature_k": 800.}},
             "fuel_temperature_k"),
            ("XS coverage failure", transport, transport_path,
             {"nuclear_data": {**transport["nuclear_data"],
                               "explicit_temperature_coverage_ok": False}},
             "temperature-dependent"),
        ]:
            reset()
            updated = deepcopy(base)
            updated.update(patch)
            _write(path, updated)
            expect_rejected(run, keyword=keyword)

        reset()
        # A quality limit below actual p95 must independently reject.
        expect_rejected(lambda: audit_fresh_transport(
            statepoint=sp, cross_sections_xml=cross,
            transport_provenance=transport_path,
            analysis_root=analysis, requested_power_w=1e6,
            fuel_temperature_k=600.,
            hydrogen_temperature_k=350.,
            hydrogen_density_g_cm3=0.001,
            max_heating_p95_relative_error=0.1,
        ), keyword="exceeds")
        reset()
        # Non-finite statistics must not evade thresholds.
        altered=deepcopy(meta)
        altered["mesh_heating_p95_relative_error"]=float("nan")
        _write(meta_path,altered)
        expect_rejected(run,keyword="finite")
    print("NERVA OpenMC statepoint lineage and Monte Carlo uncertainty QA: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
