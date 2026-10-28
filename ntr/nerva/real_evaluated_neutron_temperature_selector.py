"""Stage45 genuine evaluated-XS neutron transport temperature selector comparison.

Two physically identical original C++ OpenMC neutron eigenvalue runs use
the EXACT same evaluated nuclear data and explicit 293.6059 K material
temperatures, but choose nearest or interpolation transport selection.
This evaluates the actual C++ cross-section selector, not historical hot
NASA reactor physics or an unprovided high-temperature nuclear library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from .audit_nuclear_data_temperatures import audit_temperature_coverage
from .config import NervaConfig

TEMPERATURE_K=293.6059129982851


def _hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError("original evaluated transport evidence missing: "+str(path))
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):
            h.update(chunk)
    return h.hexdigest()


def audit_actual_selector_runs(root:Path|str, *, cross_sections:Path|str) -> dict:
    root=Path(root)
    xs=Path(cross_sections)
    from dataclasses import replace
    checks=[]
    for method in ("nearest","interpolation"):
        folder=root/method
        provenance=json.loads((folder/"transport_provenance.json").read_text(encoding="utf-8"))
        cfg=provenance["config"]
        if (cfg.get("neutron_temperature_method")!=method
                or cfg.get("fuel_temperature_k")!=TEMPERATURE_K
                or cfg.get("hydrogen_temperature_k")!=TEMPERATURE_K
                or cfg.get("batches")!=24 or cfg.get("inactive")!=8
                or cfg.get("particles")!=3000 or cfg.get("seed")!=13007):
            raise ValueError("real OpenMC neutron source or material temperature altered")
        settings=ET.parse(folder/"settings.xml").getroot()
        if settings.findtext("temperature_method")!=method:
            raise ValueError("actual C++ settings.xml temperature operator differs from preflight")
        if method=="nearest" and float(settings.findtext("temperature_tolerance","nan"))!=10.0:
            raise ValueError("actual C++ nearest evaluated XS temperature tolerance disagrees")
        if method=="interpolation" and settings.find("temperature_tolerance") is not None:
            raise ValueError("interpolation unexpectedly exported nearest tolerance")
        for name in ("geometry.xml","materials.xml","settings.xml"):
            if _hash(folder/name)!=provenance["xml_sha256"][name]:
                raise ValueError("original exported OpenMC XML provenance invalid")
        if provenance["nuclear_data"]["cross_sections_sha256"]!=_hash(xs):
            raise ValueError("actual HDF5 evaluated neutron index differs across runs")
        from dataclasses import replace
        config=replace(NervaConfig(),core_rings=3,fuel_temperature_k=TEMPERATURE_K,
                       hydrogen_temperature_k=TEMPERATURE_K,
                       neutron_temperature_method=method)
        audit=audit_temperature_coverage(cross_sections=xs,config=config)
        if (audit["neutron_temperature_method"]!=method
                or not audit["model_explicit_temperature_coverage_ok"]):
            raise ValueError("original C++ neutron material kT table coverage failed")
        sp=folder/"statepoint.24.h5"
        if _hash(sp)!=provenance["statepoint"]["sha256"]:
            raise ValueError("original C++ HDF5 statepoint provenance invalid")
        with h5py.File(sp,"r") as h:
            if (bytes(h.attrs.get("filetype",b"")).rstrip(b"\0")!=b"statepoint"
                    or bytes(h["run_mode"][()]).rstrip(b"\0")!=b"eigenvalue"
                    or int(h["n_particles"][()])!=3000
                    or int(h["n_inactive"][()])!=8
                    or int(h["current_batch"][()])!=24
                    or int(h["seed"][()])!=13007
                    or int(h.attrs.get("source_present",0))!=1):
                raise ValueError("native actual neutron eigenvalue source fields invalid")
            k=np.asarray(h["k_generation"][()],dtype=float)
            entropy=np.asarray(h["entropy"][()],dtype=float)
            bank=np.asarray(h["source_bank"][()])
        if (k.shape!=(24,) or entropy.shape!=(24,) or bank.shape!=(3000,)
                or not np.isfinite(k).all() or np.any(k<=0)
                or not np.isfinite(entropy).all()):
            raise ValueError("actual neutron eigenvalue k/entropy/fission source not measured")
        checks.append((method,provenance,audit,k,entropy,bank))
    a,b=checks
    if any(a[1]["xml_sha256"][key]!=b[1]["xml_sha256"][key]
           for key in ("geometry.xml","materials.xml")):
        raise ValueError("the two original temperature-operator runs altered reactor geometry or materials")
    return {
        "schema":"nuke-stage45-genuine-cplusplus-evaluated-neutron-temperature-v1",
        "original_openmc_CPP_neutron_transport_runs":2,
        "original_neutron_eigenvalue_hdf5_scored":2,
        "actual_evaluated_neutron_library_index_sha256":_hash(xs),
        "geometry_and_material_xml_identical":True,
        "explicit_material_temperature_K":TEMPERATURE_K,
        "actual_transport_selectors_exercised":["nearest","interpolation"],
        "per_selector":[{
            "temperature_method":method,
            "original_statepoint_sha256":provenance["statepoint"]["sha256"],
            "neutron_cross_section_temperature_audit_passed":
                audit["model_explicit_temperature_coverage_ok"],
            "active_generation_k_arithmetic_mean":float(k[8:].mean()),
            "active_generation_k_min":float(k[8:].min()),
            "active_generation_k_max":float(k[8:].max()),
            "entropy_at_final_generation":float(entropy[-1]),
            "native_fission_bank_sha256":hashlib.sha256(bank.tobytes()).hexdigest(),
        } for method,provenance,audit,k,entropy,bank in checks],
        "max_absolute_k_generation_difference_between_two_selectors":
            float(np.max(np.abs(a[3]-b[3]))),
        "full_native_source_bank_array_bit_identical":bool(np.array_equal(a[5],b[5])),
        "hot_temperature_scattering_interpolation_verified":False,
        "historical_NASA_NERVA_reactor_physics_qualified":False,
        "physical_reactor_power_watts_qualified":False,
        "interpretation":[
            "Only one exact cold evaluated table is exercised by these real runs.",
            "Interpolation between two hot nuclear tables requires a separately available evaluated library.",
            "Numerical selector validation cannot establish temperature feedback coefficient, criticality, or fuel power.",
        ],
    }


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--cross-sections",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    r=audit_actual_selector_runs(args.root,cross_sections=args.cross_sections)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(r,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Original C++ neutron source runs",r["original_openmc_CPP_neutron_transport_runs"])
    print("Native k trajectory maximum selector difference",
          r["max_absolute_k_generation_difference_between_two_selectors"])
    print("Actual fission source banks exactly identical",
          r["full_native_source_bank_array_bit_identical"])
    print("True hot neutron scattering/historical NERVA qualified: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
