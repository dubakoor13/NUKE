"""Unit checks for NERVA OpenMC nuclear-data preflight parsing."""

from __future__ import annotations

from pathlib import Path
import tempfile
from unittest.mock import patch

import openmc.data

from .config import NervaConfig
from .audit_nuclear_data_temperatures import audit_temperature_coverage

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

    # Synthetic HDF5-reader fixtures below are only negative physical
    # preflight tests. Genuine evaluated HDF5 and transport remain mandatory.
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        nuclides = required_neutron_nuclides()
        for name in nuclides:
            (root / (name + ".h5")).write_bytes(b"synthetic-invalid-hdf5")
        xml = root / "cross_sections.xml"
        def make_index(duplicate=False):
            library = [
                f'<library materials="{name}" path="{name}.h5" type="neutron"/>'
                for name in nuclides
            ]
            if duplicate:
                library.append('<library materials="H1" path="H1.h5" type="neutron"/>')
            xml.write_text(
                "<cross_sections>"+"".join(library)+"</cross_sections>",
                encoding="utf-8"
            )

        reference = 293.6059129982851
        class FakeIncidentNeutron:
            def __init__(self, name, values=None):
                self.name = name
                self.kTs = [
                    float(t) * openmc.data.K_BOLTZMANN
                    for t in (values if values is not None else [reference])
                ]

        def reader(path):
            return FakeIncidentNeutron(Path(path).stem)

        make_index()
        with patch.object(openmc.data.IncidentNeutron,"from_hdf5",side_effect=reader):
            covered = audit_temperature_coverage(
                cross_sections=xml,config=NervaConfig(core_rings=3,hydrogen_temperature_k=reference)
            )
            assert covered["model_explicit_temperature_coverage_ok"]
            near_boundary = audit_temperature_coverage(
                cross_sections=xml,
                config=NervaConfig(
                    core_rings=3,
                    hydrogen_temperature_k=reference + 5e-10,
                ),
            )
            assert near_boundary["model_explicit_temperature_coverage_ok"]
            assert near_boundary["nonphysical_extrapolation_rounding_slack_K"] < 1e-7
            assert covered["all_indexed_required_nuclides_have_verified_hdf5_identity"]
            assert not covered["evaluated_temperature_extrapolation_permitted"]

            # A 1-K out-of-range temperature would formerly pass the 10-K
            # nearest selector. Evaluated coverage MUST now remain false.
            outside = audit_temperature_coverage(
                cross_sections=xml,
                config=NervaConfig(core_rings=3,hydrogen_temperature_k=reference+1.),
            )
            hydrogen = next(
                row for row in outside["nuclides"] if row["nuclide"]=="H1"
            )
            assert hydrogen["explicit_requested_temperatures_in_range"] is False
            assert hydrogen[
                "explicit_requested_temperatures_within_nearest_tolerance"
            ] is False
            assert not outside["model_explicit_temperature_coverage_ok"]

        def swapped(path):
            name = Path(path).stem
            return FakeIncidentNeutron("U235" if name == "H1" else name)
        with patch.object(openmc.data.IncidentNeutron,"from_hdf5",side_effect=swapped):
            report = audit_temperature_coverage(
                cross_sections=xml,config=NervaConfig(core_rings=3,hydrogen_temperature_k=reference)
            )
            h1 = next(row for row in report["nuclides"] if row["nuclide"]=="H1")
            assert "identity mismatch" in h1["load_error"]
            assert not report["all_indexed_required_nuclides_have_verified_hdf5_identity"]
            assert not report["model_explicit_temperature_coverage_ok"]

        def broken_grid(path):
            name = Path(path).stem
            return FakeIncidentNeutron(
                name, [reference,float("nan")] if name=="H1" else [reference]
            )
        with patch.object(openmc.data.IncidentNeutron,"from_hdf5",side_effect=broken_grid):
            report = audit_temperature_coverage(
                cross_sections=xml,config=NervaConfig(core_rings=3,hydrogen_temperature_k=reference)
            )
            h1 = next(row for row in report["nuclides"] if row["nuclide"]=="H1")
            assert "invalid evaluated neutron temperature grid" in h1["load_error"]
            assert not report["model_explicit_temperature_coverage_ok"]

        make_index(duplicate=True)
        with patch.object(openmc.data.IncidentNeutron,"from_hdf5",side_effect=reader):
            try:
                audit_temperature_coverage(
                    cross_sections=xml,config=NervaConfig(core_rings=3,hydrogen_temperature_k=reference)
                )
            except ValueError as exc:
                assert "ambiguous duplicate" in str(exc)
            else:
                raise AssertionError("duplicate neutron cross-section index accepted")

    # Even a deliberately matched synthetic set only authorizes trying
    # transport, never historical NERVA physical qualification.
    from .physical_scattering_pair_qualification import assess_scattering_pair
    sample = {"model_explicit_temperature_coverage_ok":True,"nuclides":[]}
    kernels = {
        "ready":True,
        "tables":[{
            "table":"c_Be",
            "evaluated_temperatures_K":[reference],
            "actual_hdf5_kernel_loaded":True,
            "strict_tabulated_temperature_covered":True,
        }]
    }
    compatible = assess_scattering_pair(sample,sample,kernels)
    assert compatible["real_bound_vs_free_transport_comparison_permitted"]
    assert not compatible["nasa_nerva_hot_core_physically_validated"]
    not_covered = {
        "model_explicit_temperature_coverage_ok":False,
        "nuclides":[{
            "nuclide":"Be9","load_error":None,
            "available_min_K":reference,
            "available_max_K":reference,
            "explicit_requested_temperatures_in_range":False,
            "explicit_requested_temperatures_within_nearest_tolerance":False,
            "requested_by_material":[{
                "material_name":"beryllium radial reflector",
                "requested_temperature_K":reference+2.4,
            }],
        }],
    }
    rejected_pair = assess_scattering_pair(sample,not_covered,kernels)
    assert not rejected_pair["real_bound_vs_free_transport_comparison_permitted"]
    assert rejected_pair["uncovered_executed_material_nuclides"][0]["nuclide"]=="Be9"
    assert any("bound-scattering case lacks" in r for r in rejected_pair["blockers"])
    missing_kernel = {"ready":True,"tables":[{
        **kernels["tables"][0],
        "strict_tabulated_temperature_covered":False,
    }]}
    rejected_kernel = assess_scattering_pair(sample,sample,missing_kernel)
    assert not rejected_kernel["real_bound_vs_free_transport_comparison_permitted"]

    print("NERVA nuclear-data preflight validation: PASS")
    print(f"  required model nuclides: {len(required)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
