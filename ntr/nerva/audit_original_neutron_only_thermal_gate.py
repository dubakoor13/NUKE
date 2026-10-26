"""Stage 28 audit: neutron-only OpenMC heating is NOT qualified total core heat.

Reopens four ORIGINAL 2026-10-23 OpenMC HDF5 statepoints through the Stage
21/24 SHA-verified evidence bundle. Specifically checks the dangerous case
of positive ordinary 'heating' with photon_transport disabled, and requires
the real coupled StatePoint heating reader to refuse this total-power route.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py

from .fx2th_openmc_statepoint_heating import require_total_heating_transport_mode
from .real_native_tally_physics_ledger import audit_native_neutron_heating
from .real_transport_evidence_gate import _Bundle, sha256_bytes


def audit_original_neutron_only_thermal_gate(artifact: str | Path) -> dict:
    physics = audit_native_neutron_heating(artifact)
    if (physics["physical_local_heating_qualified"] is not False or
        physics["hot_core_reactor_power_normalization_allowed"] is not False):
        raise ValueError("original archive physical energy qualification changed")
    source = _Bundle(Path(artifact))
    try:
        observations = []
        for row in physics["seed_reaction_scores"]:
            seed = row["seed"]
            data = source.read(f"seed_{seed}/statepoint.120.h5")
            if sha256_bytes(data) != row["native_statepoint_sha256"]:
                raise ValueError("changed original statepoint")
            with h5py.File(io.BytesIO(data), "r") as h5:
                photon = bool(int(h5.attrs["photon_transport"]))
            if (photon or row["mesh_heating_score_ev_per_source"]<=0 or
                not row["local_heating_all_first_and_second_moments_zero"] or
                row["material_separated_local_heating_usable_for_watts"] is not False):
                raise ValueError("original neutron-only heating counterexample changed")
            rejected = False
            try:
                require_total_heating_transport_mode(
                    photon_transport=photon, score="heating")
            except ValueError as exc:
                rejected = "neutron-only" in str(exc)
            if not rejected:
                raise ValueError("neutron-only ordinary heating unexpectedly passed")
            observations.append({
                "seed": seed,
                "native_hdf5_sha256": sha256_bytes(data),
                "photon_transport": photon,
                "ordinary_heating_ev_per_source": row["mesh_heating_score_ev_per_source"],
                "all_heating_local_raw_moments_zero": True,
                "total_core_thermal_power_bridge_rejected": True,
            })
        if len(observations)!=4:
            raise ValueError("not four genuine original OpenMC source seeds")
        return {
            "schema": "nuke-neutron-only-heating-mode-stage28-v1",
            "evidence": "FOUR_ORIGINAL_OPENMC_0153_NEUTRON_ONLY_HDF5_STATEPOINTS",
            "seeds_checked": len(observations),
            "real_coupled_neutron_photon_transport_performed": False,
            "positive_ordinary_heating_is_complete_deposited_power": False,
            "qualified_local_heating_available": False,
            "stage10_complete_thermal_power_gate_rejected_all_original_runs": True,
            "hot_core_watt_normalization_allowed": False,
            "historical_NERVA_qualified": False,
            "original_hdf5_observations": observations,
            "next_requirements": [
                "Coupled neutron-photon transport with evaluated photon production and interaction data",
                "Independent material/temperature and energy-deposition data provenance",
                "Source stationarity and physically justified normalization",
                "Material-separated energy accounting and thermal-channel mapping",
            ],
        }
    finally:
        source.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("artifact", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit_original_neutron_only_thermal_gate(a.artifact)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2,allow_nan=False)+"\n",
                        encoding="utf-8")
    print("Genuine original neutron-only OpenMC HDF5 seeds:",result["seeds_checked"])
    print("Promote partial neutron 'heating' to full thermal power: REJECTED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
