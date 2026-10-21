"""Evaluate actual, temperature-resolved MT=901 KERMA coefficient values.

This examines the OpenMC evaluated HDF5 data already selected for the
core. It never replaces or fabricates coefficients. Reaction presence by
itself is not sufficient: all-zero or nonfinite numerical tables cannot
support neutron-only heating-local transport.

No criticality/geometry/material design changes, interpolation of
unprovided data, or claim of historically benchmarked heating is made.
"""
from __future__ import annotations

import numpy as np


def inspect_mt901_numeric_curves(data) -> dict:
    """Inspect OpenMC IncidentNeutron's *actual* tabulated MT901 functions.

    Every neutron temperature in the input nuclide library must supply
    finite, strictly increasing-energy, nonidentically-zero values.
    Negative evaluated coefficients are recorded but never clipped:
    validity of their physical magnitude requires independent evaluation.
    """
    temperatures = sorted(set(str(t) for t in data.temperatures))
    if not temperatures:
        raise ValueError("evaluated neutron nuclide has no temperatures")
    present = 901 in data.reactions
    xs_by_temp = data.reactions[901].xs if present else {}
    rows = []
    for temperature in temperatures:
        entry = {
            "temperature": temperature,
            "finite_nonzero_coefficients": False,
            "number_of_tabulated_points": None,
            "number_of_nonzero_coefficients": None,
            "number_of_negative_coefficients": None,
            "energy_min_eV": None,
            "energy_max_eV": None,
            "kerma_min_eV_barn": None,
            "kerma_max_eV_barn": None,
            "issue": None,
        }
        curve = xs_by_temp.get(temperature)
        if curve is None:
            entry["issue"] = "no MT901 data at this neutron XS temperature"
        else:
            try:
                # OpenMC stores continuous-energy microscopic reaction
                # functions as Tabulated1D with x/y arrays. Avoid invented
                # interpolation/extrapolation in the qualification itself.
                energy = np.asarray(curve.x, dtype=np.float64)
                kerma = np.asarray(curve.y, dtype=np.float64)
                if (
                    energy.ndim != 1 or kerma.ndim != 1
                    or len(energy) < 2 or len(energy) != len(kerma)
                    or not np.isfinite(energy).all()
                    or np.any(energy < 0) or np.any(np.diff(energy) <= 0)
                    or not np.isfinite(kerma).all()
                ):
                    raise ValueError("nonfinite, empty or invalid MT901 energy/KERMA grid")
                nonzero = int(np.count_nonzero(kerma))
                entry.update({
                    "number_of_tabulated_points": int(len(kerma)),
                    "number_of_nonzero_coefficients": nonzero,
                    "number_of_negative_coefficients": int(np.count_nonzero(kerma < 0)),
                    "energy_min_eV": float(energy[0]),
                    "energy_max_eV": float(energy[-1]),
                    "kerma_min_eV_barn": float(np.min(kerma)),
                    "kerma_max_eV_barn": float(np.max(kerma)),
                    "finite_nonzero_coefficients": nonzero > 0,
                    "issue": None if nonzero else "all evaluated MT901 KERMA coefficients are zero",
                })
            except (AttributeError, TypeError, ValueError) as error:
                entry["issue"] = str(error)
        rows.append(entry)
    return {
        "mt901_reaction_present": present,
        "temperatures_checked": len(temperatures),
        "per_temperature": rows,
        "local_kerma_mt901_numerical_usable": all(
            row["finite_nonzero_coefficients"] for row in rows
        ),
        "nasa_core_energy_deposition_validated": False,
        "interpretation": (
            "Nonzero finite tabulated evaluated MT901 is a necessary library "
            "qualification only. It cannot establish source convergence, "
            "heating tally precision, reactor temperature adequacy, "
            "photon conservation, or NASA NERVA experimental agreement."
        ),
    }
