"""Independent energy-domain compatibility of genuine OpenMC fission-source banks.

Cross-check actual neutron emission energies in 20 saved eigenvalue HDF5
source banks against genuine temperature-resolved incident-neutron nuclear
cross-section grid domains of all physically root-reachable materials.

Conservatively intersect *all* native evaluated temperature grids for every
active isotope. This is a source-spectrum support screen, NOT evidence
that the collision-energy distribution stays inside the evaluated domains,
that thermal scattering is physically correct, or that the reactor has
qualified hot cross sections or fission-source convergence. Never clip
energies or synthesize/extrapolate microscopic nuclear data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .source_energy_audit import read_source_energy
from .executed_neutron_temperature_energy_grid import (
    audit_executed_neutron_energy_grids,
)


def _digest(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def evaluate_source_energy_domain(
    energies_eV: object, weights: object, lower_eV: float, upper_eV: float
) -> dict:
    """Exact inclusive physical-support partition, with weighted closure."""
    e=np.asarray(energies_eV,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64)
    lo=float(lower_eV)
    hi=float(upper_eV)
    if (e.ndim!=1 or w.ndim!=1 or len(e)!=len(w) or len(e)<2
            or not np.isfinite(e).all() or not np.isfinite(w).all()
            or np.any(e<=0) or np.any(w<=0)
            or not math.isfinite(lo) or not math.isfinite(hi)
            or lo<0 or not hi>lo):
        raise ValueError("nonphysical evaluated nuclear support or neutron-source energies")
    below=e<lo
    above=e>hi
    within=~(below|above)
    mass=float(np.sum(w,dtype=np.float64))
    weights=[
        float(np.sum(w[mask],dtype=np.float64))
        for mask in (below,within,above)
    ]
    if (not math.isfinite(mass) or mass<=0 or
            not math.isclose(sum(weights),mass,rel_tol=5e-13,abs_tol=1e-8)):
        raise ValueError("fission source weight partition failed conservation")
    return {
        "evaluated_incident_energy_min_eV":lo,
        "evaluated_incident_energy_max_eV":hi,
        "source_neutron_sites":int(len(e)),
        "total_bank_neutron_weight":mass,
        "below_grid_neutron_sites":int(np.count_nonzero(below)),
        "inside_grid_neutron_sites":int(np.count_nonzero(within)),
        "above_grid_neutron_sites":int(np.count_nonzero(above)),
        "below_grid_source_weight":weights[0],
        "inside_grid_source_weight":weights[1],
        "above_grid_source_weight":weights[2],
        "below_grid_weight_fraction":weights[0]/mass,
        "inside_grid_weight_fraction":weights[1]/mass,
        "above_grid_weight_fraction":weights[2]/mass,
        "source_energies_within_common_evaluated_grid":
            not (bool(np.any(below)) or bool(np.any(above))),
        "weighted_energy_partition_conserved":True,
    }


def audit_executed_source_energy_support(
    summary_path:Path, nuclear_index:Path, source_audit_path:Path
) -> dict:
    summary_path=Path(summary_path)
    nuclear_index=Path(nuclear_index)
    source_audit_path=Path(source_audit_path)
    summary=json.loads(summary_path.read_text(encoding="utf-8"))
    spectral=json.loads(source_audit_path.read_text(encoding="utf-8"))
    xsa=audit_executed_neutron_energy_grids(summary_path,nuclear_index)
    if (summary.get("execution")!="ACTUAL_OPENMC_TRANSPORT"
            or summary.get("assembly")!="core"
            or spectral.get("real_transport_summary_sha256")!=_digest(summary_path)
            or spectral.get("evaluated_cross_sections_xml_sha256")!=_digest(nuclear_index)
            or xsa["executed_transport_summary_sha256"]!=_digest(summary_path)
            or xsa["evaluated_neutron_cross_sections_xml_sha256"]!=_digest(nuclear_index)
            or xsa["four_seed_provenance_verified"] is not True):
        raise ValueError("genuine four-seed source/XSD provenance mismatch")
    evidence=xsa["incident_neutron_temperature_energy_grid_evidence"]
    materials=xsa["physically_instantiated_material_temperature_evidence"]
    if not evidence or not materials:
        raise ValueError("executed reactor contains no qualified neutron data inventory")
    # Intersect all native temperature tables for all isotopes in the
    # material: deliberately conservative, not an interpolation algorithm.
    domains={}
    for nuclide,ev in evidence.items():
        a=float(ev["common_incident_energy_min_eV"])
        b=float(ev["common_incident_energy_max_eV"])
        if not 0<=a<b or not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("invalid genuine isotope neutron energy support")
        domains[nuclide]=(a,b)
    by_material={}
    for item in materials:
        names=item["evaluated_neutron_nuclides"]
        if not names or len(set(names))!=len(names) or any(k not in domains for k in names):
            raise ValueError("unqualified executed material neutron isotope inventory")
        lo=max(domains[k][0] for k in names)
        hi=min(domains[k][1] for k in names)
        if not hi>lo:
            raise ValueError("no common nuclear neutron energy domain for an active material")
        by_material[str(item["material_id"])]={
            "name":item["material_name"],
            "active_nuclides":names,
            "native_neutron_domain_min_eV":lo,
            "native_neutron_domain_max_eV":hi,
        }
    lo_all=max(v[0] for v in domains.values())
    hi_all=min(v[1] for v in domains.values())
    if not hi_all>lo_all:
        raise ValueError("no common incident neutron grid across executed core")
    runs=summary.get("runs")
    batches=summary.get("saved_source_bank_batches")
    audit_runs=spectral.get("runs")
    if (not isinstance(runs,list) or len(runs)!=4
            or not isinstance(batches,list) or len(batches)!=5
            or not isinstance(audit_runs,list) or len(audit_runs)!=4
            or batches[-1]!=summary.get("batches")
            or any(a>=b for a,b in zip(batches,batches[1:]))):
        raise ValueError("expected four independent seeds and five actual checkpoints")
    prior={r["seed"]:r for r in audit_runs}
    if len(prior)!=4:
        raise ValueError("duplicated actual source spectroscopy RNG seed")
    seen=set()
    points=[]
    for run in runs:
        seed=run["seed"]
        final=Path(run["statepoint_path"])
        if (seed in seen or seed not in prior or
                not final.is_file() or
                _digest(final)!=run["statepoint_sha256"]):
            raise ValueError("wrong physical OpenMC source-spectrum statepoint provenance")
        seen.add(seed)
        native_snapshots=prior[seed]["snapshots"]
        if len(native_snapshots)!=len(batches):
            raise ValueError("original source energy audit checkpoint count inconsistent")
        for checkpoint, old in zip(batches,native_snapshots):
            path=final.parent/f"statepoint.{checkpoint:0{len(str(summary['batches']))}d}.h5"
            _,meta=read_source_energy(path)
            if (meta["seed"]!=seed or meta["completed_batch"]!=checkpoint
                    or old["statepoint_sha256"]!=meta["statepoint_sha256"]
                    or old["source_site_count"]!=meta["source_site_count"]):
                raise ValueError("source energy bank differs from independent spectrum audit")
            if meta["statepoint_sha256"] in {x["statepoint_sha256"] for x in points}:
                raise ValueError("duplicate actual OpenMC source-site bank")
            with h5py.File(path,"r") as h:
                bank=h["source_bank"][()]
                e=np.asarray(bank["E"],dtype=np.float64)
                w=np.asarray(bank["wgt"],dtype=np.float64)
            core=evaluate_source_energy_domain(e,w,lo_all,hi_all)
            m_rows={
                mid:evaluate_source_energy_domain(e,w,info["native_neutron_domain_min_eV"],
                                                  info["native_neutron_domain_max_eV"])
                for mid,info in by_material.items()
            }
            points.append({
                "seed":seed,"saved_batch":checkpoint,
                "statepoint_sha256":meta["statepoint_sha256"],
                "source_mean_energy_eV":meta["weighted_mean_energy_eV"],
                "core_all_isotope_support":core,
                "per_root_reachable_material_source_support":m_rows,
            })
    if len(points)!=20 or len(seen)!=4:
        raise ValueError("missing independent OpenMC fission bank spectral checkpoints")
    return {
        "audit":"GENUINE_OPENMC_FISSION_SOURCE_EVALUATED_NEUTRON_ENERGY_DOMAIN_SUPPORT",
        "actual_four_seed_summary_sha256":_digest(summary_path),
        "independently_validated_source_energy_audit_sha256":_digest(source_audit_path),
        "actual_evaluated_cross_sections_xml_sha256":_digest(nuclear_index),
        "physically_reachable_material_ids":xsa["executed_reachable_material_ids"],
        "physically_used_neutron_nuclide_count":len(domains),
        "real_independent_source_seeds":len(seen),
        "genuine_source_hdf5_checkpoint_count":len(points),
        "all_native_energy_temperature_nodes_scanned":True,
        "common_all_material_incident_neutron_energy_min_eV":lo_all,
        "common_all_material_incident_neutron_energy_max_eV":hi_all,
        "all_emitted_source_energies_within_all_active_evaluated_grids":
            all(row["core_all_isotope_support"]["source_energies_within_common_evaluated_grid"]
                for row in points),
        "max_fraction_of_source_weight_outside_common_incident_domain":
            max(row["core_all_isotope_support"]["below_grid_weight_fraction"]+
                row["core_all_isotope_support"]["above_grid_weight_fraction"]
                for row in points),
        "per_material_evaluated_grid_domains":by_material,
        "twenty_verified_source_spectrum_records":points,
        "emitted_source_bank_energy_weight_conserved":True,
        "transport_collision_energy_coverage_proven":False,
        "evaluated_thermal_scattering_phase_qualified":False,
        "hot_operating_core_neutron_physics_qualified":False,
        "fission_source_stationarity_qualified":False,
        "interpretation":(
            "Exact physical emitted neutron fission-bank spectrum versus all "
            "executed isotope incident-energy tables, fully SHA256-linked "
            "to genuine OpenMC C++ four-seed neutron transport. Even a "
            "100-percent in-domain emitted source does NOT prove in-domain "
            "neutron collision energies after scattering, bound scattering, "
            "hot nuclear data, stationary source, or historical NASA NERVA."
        ),
    }


def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--cross-sections",type=Path,required=True)
    p.add_argument("--source-energy",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    result=audit_executed_source_energy_support(
        a.summary,a.cross_sections,a.source_energy
    )
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("Genuine HDF5 fission-source checkpoints checked:",
          result["genuine_source_hdf5_checkpoint_count"])
    print("Active evaluated neutron nuclides:",
          result["physically_used_neutron_nuclide_count"])
    print("Common evaluated incident energy range [eV]:",
          result["common_all_material_incident_neutron_energy_min_eV"],
          result["common_all_material_incident_neutron_energy_max_eV"])
    print("Max emitted source-weight fraction out of evaluated support:",
          result["max_fraction_of_source_weight_outside_common_incident_domain"])
    print("Transport collision energy / hot NERVA qualified: NO")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
