"""Unit checks for NERVA OpenMC nuclear-data preflight parsing."""

from __future__ import annotations

from pathlib import Path
import tempfile

from .nuclear_data import _library_entries, required_neutron_nuclides


def main() -> int:
    required = required_neutron_nuclides()
    assert "U235" in required
    assert "U238" in required
    assert "H1" in required
    assert "C0" in required or "C12" in required
    assert len(required) > 8

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "H1.h5").write_bytes(b"x")
        (root / "U235.h5").write_bytes(b"x")
        xml = root / "cross_sections.xml"
        xml.write_text(
            """<?xml version="1.0"?>
<cross_sections>
  <library materials="H1" path="H1.h5" type="neutron" />
  <library materials="U235" path="U235.h5" type="neutron" />
  <library materials="H" path="photon/H.h5" type="photon" />
</cross_sections>
""",
            encoding="utf-8",
        )
        entries = _library_entries(xml)
        assert len(entries) == 3
        assert entries[0].data_type == "neutron"
        assert entries[0].materials == ("H1",)
        assert entries[0].file_exists is True
        assert entries[2].data_type == "photon"
        assert entries[2].file_exists is False

    # OpenMC cross_sections.xml permits an absolute <directory> outside the
    # XML's folder. A prior parser incorrectly used the XML parent path.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        xs = root / "xs"
        xs.mkdir()
        source = root / "manifests"
        source.mkdir()
        (xs / "H1.h5").write_bytes(b"not-real-hdf5")
        xml = source / "cross_sections.xml"
        xml.write_text(
            "<cross_sections><directory>" + str(xs.resolve()) +
            "</directory><library materials='H1' "
            "path='H1.h5' type='neutron' /></cross_sections>",
            encoding="utf-8",
        )
        entries = _library_entries(xml)
        assert len(entries) == 1
        assert entries[0].file_exists
        assert Path(entries[0].resolved_path) == (xs / "H1.h5").resolve()

        # Temperature coverage audit must resolve the identical HDF5 path.
        from .audit_nuclear_data_temperatures import audit_temperature_coverage
        report = audit_temperature_coverage(cross_sections=xml)
        row = next(item for item in report["nuclides"] if item["nuclide"] == "H1")
        assert row["hdf5_path"] == entries[0].resolved_path
        # Fake bytes can never pass real OpenMC incident-neutron loading.
        assert row["load_error"] is not None

        # The file is not under the XML directory; invalid/missing paths
        # cannot be silently treated as present.
        (xs / "H1.h5").unlink()
        assert not _library_entries(xml)[0].file_exists

        # Relative <directory> resolves against OpenMC's run CWD; disallow
        # claiming that the preflight checked a potentially different file.
        xml.write_text(
            "<cross_sections><directory>../xs</directory>"
            "<library materials='H1' path='H1.h5' type='neutron' />"
            "</cross_sections>",
            encoding="utf-8",
        )
        try:
            _library_entries(xml)
        except ValueError as exc:
            assert "absolute" in str(exc)
        else:
            raise AssertionError("ambiguous relative OpenMC directory accepted")
        xml.write_text(
            "<cross_sections><library materials='H1' type='neutron' />"
            "</cross_sections>",
            encoding="utf-8",
        )
        try:
            _library_entries(xml)
        except ValueError as exc:
            assert "missing path" in str(exc)
        else:
            raise AssertionError("missing nuclear data path accepted")

    print("NERVA nuclear-data preflight validation: PASS")
    print(f"  required model nuclides: {len(required)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
