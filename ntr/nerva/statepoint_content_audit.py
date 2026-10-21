"""Validate OpenMC eigenvalue statepoint PHYSICS contents, not just its hash.

SHA-256 provenance demonstrates file identity, NOT that an arbitrary .h5
file contains a completed eigenvalue calculation. This independent HDF5
structural/numerical gate checks run mode, batch statistics, eigenvalue,
photon transport and tally presence BEFORE thermal feedback is accepted.

Structural validity is necessary but NOT sufficient evidence of a physical
reactor model: benchmark agreement, source convergence and XS accuracy are
separate. Files written by the CI tests are explicitly artificial HDF5.
"""
from __future__ import annotations

from pathlib import Path
import math

import h5py
import numpy as np


def _integer_dataset(handle: h5py.File, key: str, *, minimum: int = 0) -> int:
    try:
        raw = np.asarray(handle[key][()])
        if raw.ndim != 0 or raw.dtype.kind not in "iu":
            raise ValueError("not an integer scalar")
        value = int(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"OpenMC statepoint {key} is missing or invalid") from exc
    if value < minimum:
        raise ValueError(f"OpenMC statepoint {key} is below allowed minimum")
    return value


def _byte_string(value: object, label: str) -> str:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value[()]
    if not isinstance(value, (bytes, np.bytes_)):
        raise ValueError(f"OpenMC {label} must be a bytestring")
    try:
        return bytes(value).decode("ascii").rstrip("\x00")
    except UnicodeDecodeError as exc:
        raise ValueError(f"OpenMC {label} is not ASCII") from exc


def audit_eigenvalue_statepoint_content(path: Path) -> dict:
    """Reject partial/corrupt/non-eigenvalue or non-tallied OpenMC HDF5 files.

    Reads HDF5 and a small number of scalar datasets; does not load large
    mesh tallies, source banks, or use a synthetic OpenMC model.
    """
    try:
        with h5py.File(path, "r") as state:
            filetype = _byte_string(
                state.attrs.get("filetype"), "statepoint filetype"
            )
            if filetype != "statepoint":
                raise ValueError("HDF5 file is not an OpenMC statepoint")
            version = np.asarray(state.attrs.get("version", []))
            if (version.shape != (2,) or version.dtype.kind not in "iu"
                    or version[0] < 17):
                raise ValueError("invalid OpenMC statepoint format version")
            run_mode = _byte_string(state["run_mode"][()], "run mode")
            if run_mode != "eigenvalue":
                raise ValueError("NERVA thermal feedback requires eigenvalue transport")
            total_batches = _integer_dataset(state,"n_batches", minimum=2)
            current_batch = _integer_dataset(state,"current_batch", minimum=2)
            inactive = _integer_dataset(state,"n_inactive", minimum=0)
            particles = _integer_dataset(state,"n_particles", minimum=1)
            realizations = _integer_dataset(state,"n_realizations",minimum=2)
            generations = _integer_dataset(state,"generations_per_batch",minimum=1)
            if (current_batch > total_batches or inactive >= current_batch
                    or realizations > (current_batch - inactive) * generations):
                raise ValueError("OpenMC statepoint has inconsistent completed batch counts")
            if "k_combined" not in state:
                raise ValueError("OpenMC eigenvalue combined k-effective is missing")
            k = np.asarray(state["k_combined"][()],dtype=float)
            if k.shape != (2,) or not np.isfinite(k).all() or k[0] <= 0 or k[1] < 0:
                raise ValueError("OpenMC combined k-effective/uncertainty is invalid")
            if "tallies_present" not in state.attrs or int(state.attrs["tallies_present"]) != 1:
                raise ValueError("OpenMC statepoint contains no recorded tallies")
            if "tallies" not in state:
                raise ValueError("OpenMC statepoint tally collection missing")
            tally_count = int(state["tallies"].attrs.get("n_tallies", 0))
            if tally_count < 1:
                raise ValueError("OpenMC statepoint has no user tallies")
            if "photon_transport" not in state.attrs:
                raise ValueError("OpenMC statepoint photon-transport flag missing")
            photon_raw = int(state.attrs["photon_transport"])
            if photon_raw not in (0, 1):
                raise ValueError("invalid OpenMC photon-transport flag")
            if "seed" not in state:
                raise ValueError("OpenMC statepoint random seed missing")
            seed = _integer_dataset(state,"seed",minimum=1)
    except (OSError, KeyError) as exc:
        raise ValueError("file is not a readable, complete OpenMC eigenvalue statepoint") from exc

    return {
        "statepoint_format_major": int(version[0]),
        "statepoint_format_minor": int(version[1]),
        "n_batches": total_batches,
        "current_batch": current_batch,
        "inactive_batches": inactive,
        "active_realizations": realizations,
        "particles_per_generation": particles,
        "generations_per_batch": generations,
        "keff": float(k[0]),
        "keff_standard_deviation": float(k[1]),
        "photon_transport": bool(photon_raw),
        "seed": seed,
        "user_tallies": tally_count,
        "note": (
            "OpenMC HDF5 statepoint and eigenvalue/tally integrity only; "
            "no assertion of source convergence or historical NERVA fidelity"
        ),
    }
