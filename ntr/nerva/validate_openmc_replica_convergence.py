"""Synthetic numerical verification of independent OpenMC heating replicas.

No real OpenMC transport or nuclear cross sections are used here. Separate
real runs must use different OpenMC --seed values and the production
provenance gate; an identical metadata file cannot prove independence.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile

import numpy as np

from .openmc_replica_convergence import (
    _check_transport_identity, compare_mesh_replicas,
    load_mesh_replica, thermal_replica_consequence
)
from .validate_resolved_transient import write_synthetic_reconstructions


def reject(operation, error=(ValueError, RuntimeError)):
    try:
        operation()
    except error:
        return
    raise AssertionError("expected input rejection")


def _write_replica(root: Path, *, axial_shift: float = 0.) -> tuple[Path,Path,Path]:
    mesh, fuel, tie = write_synthetic_reconstructions(root)
    factors=np.array([1.+axial_shift,1.,1.-axial_shift])
    with np.load(mesh,allow_pickle=False) as d:
        data={k:np.array(d[k]) for k in d.files}
    keys=(
        "power_density_w_cm3","fuel_power_density_w_cm3",
        "fuel_hydrogen_power_density_w_cm3","tie_power_density_w_cm3",
        "tie_supply_hydrogen_power_density_w_cm3",
        "tie_return_hydrogen_power_density_w_cm3"
    )
    for key in keys:
        data[key]*=factors[np.newaxis,np.newaxis,:]
    data["power_density_std_w_cm3"]=np.full((2,2,3),10.)
    np.savez_compressed(mesh,**data)
    # Match the physical z weighting in the reconstructed solid/H2
    # axial channels to avoid an inconsistent synthetic energy partition.
    for path,keys in (
        (fuel,("channel_wall_axial_power_w",
               "channel_direct_h2_axial_power_w")),
        (tie,("tie_solid_axial_power_w",
              "tie_supply_direct_h2_axial_power_w",
              "tie_return_direct_h2_axial_power_w")),
    ):
        with np.load(path,allow_pickle=False) as d:
            arrays={k:np.array(d[k]) for k in d.files}
        for key in keys:
            arrays[key]*=factors if arrays[key].ndim==2 else factors[None,None,:]
        np.savez_compressed(path,**arrays)
    (root/"metadata.json").write_text(json.dumps({
        "requested_power_MW":0.0012,
        "normalized_power_W":1200.,
        "statepoint":"DELIBERATE-SYNTHETIC-FIXTURE",
    }),encoding="utf-8")
    return mesh,fuel,tie


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        a,b=root/"a",root/"b"
        a.mkdir()
        b.mkdir()
        mesh_a,fuel_a,tie_a=_write_replica(a)
        mesh_b,fuel_b,tie_b=_write_replica(b,axial_shift=0.03)
        ra,_=load_mesh_replica(a,allow_synthetic=True)
        rb,_=load_mesh_replica(b,allow_synthetic=True)
        good=compare_mesh_replicas(
            ra,rb,max_total_variation=0.02,
            max_axial_fraction_difference=0.02,
            max_significant_fraction=0.10
        )
        assert good["pass"]
        assert good["active_mesh_voxels"]==12
        assert math.isclose(good["reactor_power_w"],1200.)
        assert math.isclose(good["total_variation_power_fraction"],0.01,
                            abs_tol=1e-12)
        assert math.isclose(good["max_axial_power_fraction_difference"],0.01,
                            abs_tol=1e-12)
        assert good["fraction_active_voxels_above_sigma"]==0.
        assert good["replica_1_p95_local_relative_std"]>0

        case={
            "fuel_solid_capacity_j_k":5000.,
            "fuel_h2_capacity_j_k":500.,
            "fuel_ua_w_k":100.,
            "tie_solid_capacity_j_k":3000.,
            "tie_supply_h2_capacity_j_k":600.,
            "tie_return_h2_capacity_j_k":700.,
            "tie_supply_ua_w_k":80.,
            "tie_return_ua_w_k":90.,
            "hydrogen_cp_reference_j_kg_k":14300.,
            "enthalpy_model":"nasa7-h2",
            "steps":2,
            "dt_s":0.1,
            "initial_solid_k":500.,
            "initial_gas_k":300.,
            "inlet_temperature_k":300.,
            "max_temperature_difference_k":1.,
        }
        thermo=thermal_replica_consequence(
            mesh_a,fuel_a,tie_a,mesh_b,fuel_b,tie_b,case
        )
        assert thermo["status"]=="PASS"
        assert 0<thermo["max_resolved_temperature_difference_k"]<1.
        assert thermo["max_fuel_outlet_difference_k"]>0
        assert thermo["max_tie_return_outlet_difference_k"]>0
        assert max(thermo["maximum_energy_residual_by_run_j"])<1e-5
        changed={**case,"max_temperature_difference_k":1e-12}
        assert thermal_replica_consequence(
            mesh_a,fuel_a,tie_a,mesh_b,fuel_b,tie_b,changed
        )["status"]=="FAIL"

        # Report must reject too-large changes under tighter tolerances.
        assert not compare_mesh_replicas(
            ra,rb,max_total_variation=0.001
        )["pass"]
        # Synthetic data cannot be treated as real OpenMC.
        reject(lambda:load_mesh_replica(a,allow_synthetic=False))
        reject(lambda:load_mesh_replica(a,allow_synthetic=True,
                                        provenance_path=root/"nonexistent"))
        # Geometry and power are not arbitrary: never compare disjoint grids.
        reject(lambda:compare_mesh_replicas(
            ra,replace(rb,dimension=(2,2,4))
        ))
        reject(lambda:compare_mesh_replicas(
            ra,replace(rb,upper_right_cm=rb.upper_right_cm+1.)
        ))
        reject(lambda:compare_mesh_replicas(
            ra,replace(rb,reactor_power_w=1500.)
        ))
        reject(lambda:compare_mesh_replicas(
            ra,rb,max_total_variation=float("nan")
        ))
        # Pair runs must have distinct RNG seeds but identical geometry
        # / materials / tally input XML and neutron data hashes.
        template={
            "assembly":"reactor",
            "config":{"seed":11,"batches":80,"particles":4000},
            "nuclear_data":{"cross_sections_sha256":"xs-checksum",
                            "explicit_temperature_coverage_ok":True},
            "xml_sha256":{
                "geometry.xml":"g","materials.xml":"m",
                "tallies.xml":"t"
            }
        }
        other=json.loads(json.dumps(template))
        other["config"]["seed"]=22
        _check_transport_identity(template,other)
        copied=json.loads(json.dumps(other))
        copied["config"]["seed"]=11
        reject(lambda:_check_transport_identity(template,copied))
        changed_model=json.loads(json.dumps(other))
        changed_model["config"]["batches"]=100
        reject(lambda:_check_transport_identity(template,changed_model))
        changed_material=json.loads(json.dumps(other))
        changed_material["xml_sha256"]["materials.xml"]="another-materials"
        reject(lambda:_check_transport_identity(template,changed_material))
        changed_xs=json.loads(json.dumps(other))
        changed_xs["nuclear_data"]["cross_sections_sha256"]="other-library"
        reject(lambda:_check_transport_identity(template,changed_xs))

        # Tampered mesh energy integral must fail before comparison.
        with np.load(mesh_b,allow_pickle=False) as d:
            arrays={k:np.array(d[k]) for k in d.files}
        arrays["power_density_w_cm3"]*=1.25
        np.savez_compressed(mesh_b,**arrays)
        reject(lambda:load_mesh_replica(b,allow_synthetic=True))
    print(
        "Independent synthetic replica heating and resolved thermal "
        "consequence regression: PASS"
    )
    print(
        "Synthetic shifted replica total variation: 0.010; "
        f"thermal maximum shift: {thermo['max_resolved_temperature_difference_k']:.7g} K"
    )
    return 0


if __name__=="__main__":
    raise SystemExit(main())
