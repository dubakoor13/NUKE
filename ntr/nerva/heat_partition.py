"""Conservative spatial heat-budget audit for an OpenMC NERVA snapshot.

All entries must be power density on the SAME Cartesian mesh in W/cm^3,
using the SAME neutron-only or neutron/photon energy-deposition score and
source-rate normalization. The four top-level components are disjoint.
Tie supply and tie return are SUBSETS of total tie hydrogen, never added
again to the total heat budget. Remaining power is deposition in all
unrepresented components (e.g., reflector, coating, vessel).

No reactor criticality or engineering design decisions are made here.
"""
from __future__ import annotations

import math
import numpy as np


def audit_heat_partition(
    mesh_power_density_w_cm3: np.ndarray,
    fuel_solid_w_cm3: np.ndarray,
    tie_solid_w_cm3: np.ndarray,
    fuel_hydrogen_w_cm3: np.ndarray,
    tie_hydrogen_w_cm3: np.ndarray,
    tie_supply_hydrogen_w_cm3: np.ndarray,
    tie_return_hydrogen_w_cm3: np.ndarray,
    voxel_volume_cm3: float,
    *,
    relative_tolerance: float = 1e-6,
) -> dict:
    """Audit each voxel and the entire thermal-power ledger.

    This is a deterministic *estimator-partition* identity: independent
    physics validation still requires transport convergence and nuclear
    data. We do NOT clip negative mesh deposits or invent lost heat.
    """
    if not math.isfinite(voxel_volume_cm3) or voxel_volume_cm3 <= 0:
        raise ValueError("OpenMC voxel volume must be positive finite")
    if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance <= 1e-2:
        raise ValueError("invalid heat-budget relative tolerance")
    raw = {
        "mesh": mesh_power_density_w_cm3,
        "fuel_solid": fuel_solid_w_cm3,
        "tie_solid": tie_solid_w_cm3,
        "fuel_hydrogen": fuel_hydrogen_w_cm3,
        "tie_hydrogen": tie_hydrogen_w_cm3,
        "tie_supply_hydrogen": tie_supply_hydrogen_w_cm3,
        "tie_return_hydrogen": tie_return_hydrogen_w_cm3,
    }
    fields: dict[str, np.ndarray] = {}
    for name, data in raw.items():
        field = np.asarray(data, dtype=float)
        if field.shape != np.shape(raw["mesh"]) or field.size == 0:
            raise ValueError(f"OpenMC {name} has incompatible mesh dimensions")
        if not np.isfinite(field).all():
            raise ValueError(f"OpenMC {name} has nonfinite heat deposits")
        fields[name] = field
    mesh = fields["mesh"]
    total = float(np.sum(mesh, dtype=np.float64) * voxel_volume_cm3)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("OpenMC thermal power must be positive finite")
    # Tolerances apply to mean power per voxel, not total power in a
    # multi-thousand-cell grid; otherwise local overcounting can be hidden.
    scale = max(float(np.max(np.abs(mesh))), total / (mesh.size * voxel_volume_cm3))
    tol_density = relative_tolerance * scale
    for name, data in fields.items():
        if float(np.min(data)) < -tol_density:
            raise ValueError(f"OpenMC {name} has physically negative deposited heat")

    assigned_density = (
        fields["fuel_solid"] + fields["tie_solid"] +
        fields["fuel_hydrogen"] + fields["tie_hydrogen"]
    )
    remaining_density = mesh - assigned_density
    if np.min(remaining_density) < -tol_density:
        raise ValueError(
            "OpenMC energy partition overcounts reactor mesh heating "
            "(possible overlapping cell filters or inconsistent scores)"
        )

    tie_split_error = (
        fields["tie_hydrogen"] -
        fields["tie_supply_hydrogen"] -
        fields["tie_return_hydrogen"]
    )
    if np.max(np.abs(tie_split_error)) > tol_density:
        raise ValueError(
            "OpenMC tie hydrogen total differs from supply + return "
            "(same-source tally conservation failed)"
        )

    totals = {
        name + "_w": float(np.sum(data, dtype=np.float64)*voxel_volume_cm3)
        for name, data in fields.items()
    }
    remaining_w = float(np.sum(remaining_density, dtype=np.float64)*voxel_volume_cm3)
    return {
        "mesh_total_w": total,
        "disjoint_components_w": {
            key: totals[key + "_w"] for key in (
                "fuel_solid", "tie_solid", "fuel_hydrogen", "tie_hydrogen"
            )
        },
        "remaining_unassigned_material_power_w": remaining_w,
        "remaining_unassigned_fraction": remaining_w / total,
        "minimum_remaining_voxel_w_cm3": float(np.min(remaining_density)),
        "maximum_tie_supply_return_residual_w_cm3": float(
            np.max(np.abs(tie_split_error))
        ),
        "voxel_relative_tolerance": relative_tolerance,
        "passed": True,
        "interpretation": (
            "Conservative spatial power bookkeeping for ONE OpenMC "
            "deposition estimator; remaining power is deposition in "
            "unmodeled material partitions, not missing reactor energy. "
            "This does not establish Monte Carlo convergence."
        ),
    }
