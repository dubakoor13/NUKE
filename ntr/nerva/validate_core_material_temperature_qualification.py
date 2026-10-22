"""Fail-closed numerical tests for physical OpenMC material temperature coverage.

Only synthetic material/nuclear-temperature fixtures here; the real
transport workflow loads evaluated NNDC isotope HDF5 via installed OpenMC
and hashes executed model XMLs and statepoints independently.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

from .core_material_temperature_qualification import (
    _executed_material_inventory,
    _settings_temperature,
    evaluate_executed_neutron_temperatures,
    audit_actual_core_neutron_temperature,
)


def reject(fn, text):
    try:
        fn()
    except (ValueError,FileNotFoundError,KeyError) as e:
        assert text in str(e), (text,str(e))
    else:
        raise AssertionError("unqualified nuclear material accepted: "+text)


def fixture():
    mats=[
        ("conceptual (U,Zr)C-graphite surrogate",None,("C0","U235","U238"),()),
        ("hydrogen coolant",293.6059129982851,("H1","H2"),()),
        ("ZrC protective coating",None,("C0","Zr90"),()),
        ("beryllium radial reflector",None,("Be9",),()),
        ("tie-tube graphite filler",None,("C0",),()),
        ("conceptual zirconium-hydride moderator",None,("H1","Zr90"),()),
        ("simplified Inconel tube surrogate",None,("Ni58",),()),
        ("natural boron-carbide control absorber",None,("B10",),()),
        ("aluminum-alloy pressure-vessel surrogate",None,("Al27",),()),
    ]
    root=ET.Element("materials")
    for i,(name,temp,nuclides,tables) in enumerate(mats,start=19):
        attrs={"id":str(i),"name":name}
        if temp is not None:
            attrs["temperature"]=str(temp)
        m=ET.SubElement(root,"material",attrs)
        for nuclide in nuclides:
            ET.SubElement(m,"nuclide",{"name":nuclide,"ao":"1.0"})
        for table in tables:
            ET.SubElement(m,"sab",{"name":table,"fraction":"1.0"})
    vals={name:[293.6059129982851] for item in mats for name in item[2]}
    return root,vals


def main()->int:
    with TemporaryDirectory() as dir:
        root=Path(dir)
        xml,temp=fixture()
        material_path=root/"materials.xml"
        ET.ElementTree(xml).write(material_path,encoding="utf-8",xml_declaration=True)
        setting=root/"settings.xml"
        setting.write_text(
            "<settings><run_mode>eigenvalue</run_mode>"
            "<photon_transport>false</photon_transport></settings>")
        mats=_executed_material_inventory(material_path)
        settings=_settings_temperature(setting)
        assert settings["solver_temperature_default_is_implicit"]
        assert settings["explicit_openmc_settings_default_temperature_K"] is None
        result=evaluate_executed_neutron_temperatures(mats,temp,settings)
        assert result["number_of_physical_materials"]==9
        assert result["number_with_explicit_exported_temperature"]==1
        assert result["number_relying_on_implicit_temperature"]==8
        assert len(result["evaluated_neutron_temperature_values_K"])==1
        assert result["evaluated_neutron_temperature_grid_has_multiple_temperatures"] is False
        assert not result["all_material_temperatures_explicit"]
        assert not result["thermal_solid_bound_scattering_qualifications_present"]
        assert result["all_explicit_material_temperatures_inside_nearest_evaluated_tolerance"]
        assert not result["hot_operating_temperature_neutron_physics_qualified"]
        assert not result["nasa_nerva_historical_criticality_or_hot_core_benchmark_validated"]

        # Named S(alpha,beta) assignments are NOT evidence of a real,
        # temperature-qualified evaluated HDF5 scattering kernel.
        warm=deepcopy(mats)
        for row in warm:
            row["explicit_material_temperature_K"]=600.
        warm[0]["bound_thermal_scattering_tables"]=["c_Graphite"]
        warm[3]["bound_thermal_scattering_tables"]=["c_Be"]
        warm[4]["bound_thermal_scattering_tables"]=["c_Graphite"]
        warm[5]["bound_thermal_scattering_tables"]=["c_H_in_ZrH","c_Zr_in_ZrH"]
        multi={key:[600.,900.] for key in temp}
        physical_setting={
            "solver_temperature_default_is_implicit":False,
            "temperature_selection_method_from_xml":"nearest",
            "explicit_nearest_tolerance_K":1.,
        }
        named_only=evaluate_executed_neutron_temperatures(
            warm,multi,physical_setting
        )
        assert not named_only["hot_operating_temperature_neutron_physics_qualified"]
        assert not named_only["thermal_solid_bound_scattering_qualifications_present"]

        verified={
            name:{"temperatures_K":[600.,900.],"hdf5_sha256":"fixture"}
            for name in ("c_Graphite","c_Be","c_H_in_ZrH","c_Zr_in_ZrH")
        }
        covered=evaluate_executed_neutron_temperatures(
            warm,multi,physical_setting,evaluated_bound_kernels=verified
        )
        # Complete HDF5 temperatures do not certify mixed (U,Zr)C/graphite
        # fuel: graphite phase fractions within the solid are unknown.
        assert not covered["thermal_solid_bound_scattering_qualifications_present"]
        assert not covered["hot_operating_temperature_neutron_physics_qualified"]
        assert not covered["mixed_fuel_graphite_phase_fractions_verified"]
        assert any("heterogeneous fuel" in r for r in covered["qualifier_reasons"])
        assert covered["executed_nearest_temperature_selector_qualified"]
        assert covered["effective_nearest_temperature_tolerance_K"]==1.
        assert not covered["nasa_nerva_historical_criticality_or_hot_core_benchmark_validated"]
        assert all(
            t["executed_material_temperature_covered"]
            for mat in covered["physical_materials"]
            for t in mat["evaluated_bound_kernel_temperature_evidence"]
        )
        # Counterfactual synthetic homogeneous fuel checks that the phase
        # gate is not a global hardcoded rejection. It is NOT reactor data.
        homogeneous=deepcopy(warm)
        homogeneous[0]["material_name"]="synthetic homogeneous fuel fixture"
        homogeneous[0]["bound_thermal_scattering_tables"]=[]
        qualified=evaluate_executed_neutron_temperatures(
            homogeneous,multi,physical_setting,evaluated_bound_kernels=verified
        )
        assert qualified["hot_operating_temperature_neutron_physics_qualified"]
        assert qualified["mixed_fuel_graphite_phase_fractions_verified"]
        assert not qualified["nasa_nerva_historical_criticality_or_hot_core_benchmark_validated"]

        incomplete_phase=deepcopy(homogeneous)
        incomplete_phase[5]["bound_thermal_scattering_tables"]=["c_Zr_in_ZrH"]
        bad_phase=evaluate_executed_neutron_temperatures(
            incomplete_phase,multi,physical_setting,evaluated_bound_kernels=verified
        )
        assert not bad_phase["hot_operating_temperature_neutron_physics_qualified"]
        assert bad_phase["physical_materials"][5]["bound_phase_table_assignment_matches"] is False

        # A Python-side loose tolerance cannot override the actual
        # executed settings.xml <temperature><tolerance> value.
        near=deepcopy(multi)
        near["U235"]=[599.,900.]
        strict=deepcopy(physical_setting)
        strict["explicit_nearest_tolerance_K"]=0.25
        rejected=evaluate_executed_neutron_temperatures(
            homogeneous,near,strict,evaluated_bound_kernels=verified
        )
        assert not rejected["hot_operating_temperature_neutron_physics_qualified"]
        assert rejected["effective_nearest_temperature_tolerance_K"]==0.25
        assert not rejected["all_explicit_material_temperatures_inside_nearest_evaluated_tolerance"]
        unknown_selector=deepcopy(physical_setting)
        unknown_selector["temperature_selection_method_from_xml"]="interpolation"
        rejected=evaluate_executed_neutron_temperatures(
            homogeneous,multi,unknown_selector,evaluated_bound_kernels=verified
        )
        assert not rejected["executed_nearest_temperature_selector_qualified"]
        assert not rejected["hot_operating_temperature_neutron_physics_qualified"]

        missing_kernel=deepcopy(verified)
        missing_kernel.pop("c_Be")
        result=evaluate_executed_neutron_temperatures(
            warm,multi,physical_setting,evaluated_bound_kernels=missing_kernel
        )
        assert not result["hot_operating_temperature_neutron_physics_qualified"]
        assert not result["thermal_solid_bound_scattering_qualifications_present"]
        wrong_temperature=deepcopy(verified)
        wrong_temperature["c_Graphite"]["temperatures_K"]=[700.,900.]
        result=evaluate_executed_neutron_temperatures(
            warm,multi,physical_setting,evaluated_bound_kernels=wrong_temperature
        )
        assert not result["hot_operating_temperature_neutron_physics_qualified"]
        assert not result["thermal_solid_bound_scattering_qualifications_present"]

        # A temperature just outside the tabulated data domain remains
        # unqualified, even if it is closer than the nearest-table tolerance.
        not_bracketed=deepcopy(multi)
        not_bracketed["U235"]=[604.,900.]
        outside=evaluate_executed_neutron_temperatures(
            warm,not_bracketed,physical_setting,evaluated_bound_kernels=verified
        )
        assert not outside["hot_operating_temperature_neutron_physics_qualified"]
        assert not outside["all_explicit_material_temperatures_inside_nearest_evaluated_tolerance"]
        fuel_u235=[x for x in outside["physical_materials"][0][
            "evaluated_isotope_temperature_evidence"] if x["nuclide"]=="U235"][0]
        assert fuel_u235["explicit_temperature_within_evaluated_range"] is False
        assert fuel_u235["explicit_temperature_nearest_table_covered"] is False

        wrong=deepcopy(mats)
        wrong[0]["explicit_material_temperature_K"]=1250.0
        report=evaluate_executed_neutron_temperatures(wrong,temp,settings)
        assert not report["all_explicit_material_temperatures_inside_nearest_evaluated_tolerance"]
        assert not report["hot_operating_temperature_neutron_physics_qualified"]

        updated=deepcopy(temp)
        updated.pop("Be9")
        reject(lambda:evaluate_executed_neutron_temperatures(mats,updated,settings),
               "missing/invalid")

        updated=deepcopy(temp)
        updated["U235"]=[600.,293.6059129982851]
        reject(lambda:evaluate_executed_neutron_temperatures(mats,updated,settings),
               "missing/invalid")
        updated=deepcopy(temp)
        updated["U235"]=[293.6059129982851,float("nan")]
        reject(lambda:evaluate_executed_neutron_temperatures(mats,updated,settings),
               "missing/invalid")

        reject(lambda:evaluate_executed_neutron_temperatures(mats,temp,settings,
                        nearest_tolerance_K=float("nan")),"tolerance")
        wrong=deepcopy(mats)
        wrong[1]["explicit_material_temperature_K"]=-1.
        reject(lambda:evaluate_executed_neutron_temperatures(wrong,temp,settings),
               "invalid exported")

        xml2=deepcopy(xml)
        xml2.find("material").set("temperature","nan")
        ET.ElementTree(xml2).write(material_path)
        reject(lambda:_executed_material_inventory(material_path),"nonphysical")
        xml2=deepcopy(xml)
        duplicate=ET.SubElement(xml2,"material",{"id":"19","name":"duplicate"})
        ET.SubElement(duplicate,"nuclide",{"name":"U235","ao":"1.0"})
        ET.ElementTree(xml2).write(material_path)
        reject(lambda:_executed_material_inventory(material_path),"duplicate")

        setting.write_text(
            "<settings><run_mode>eigenvalue</run_mode>"
            "<temperature><method>nearest</method><tolerance>nan</tolerance>"
            "</temperature></settings>"
        )
        reject(lambda:_settings_temperature(setting),"nonphysical neutron transport temperature tolerance")
        setting.write_text("<settings><run_mode>fixed source</run_mode></settings>")
        reject(lambda:_settings_temperature(setting),"neutron-only eigenvalue")

        (root/"real_core_transport_summary.json").write_text(
            '{"execution":"SYNTHETIC","assembly":"core","runs":[]}')
        reject(lambda:audit_actual_core_neutron_temperature(
            root/"real_core_transport_summary.json"),"actual four-seed")

    print("NERVA actual-material vs evaluated HDF5 temperature physics QA: PASS")
    print("  synthetic negative tests; no nuclear-data extrapolation or hot transport")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
