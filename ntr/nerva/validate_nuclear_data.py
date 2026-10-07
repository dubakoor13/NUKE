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

    print("NERVA nuclear-data preflight validation: PASS")
    print(f"  required model nuclides: {len(required)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
