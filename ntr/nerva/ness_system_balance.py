"""NESS/ELES to NUKE control-volume boundary audit (SI units).

This is a conservative diagnostic for exchanging data between independent
engine-system, hydrogen-property, and reactor solvers. It does not implement
ENABLER, an ELES engine solver, a turbine characteristic map, or OpenMC.

Signed convention: ports carry nonnegative mass-flow magnitudes; flow
direction is identified by the ``inlets`` or ``outlets`` collection.
Positive ``heat_to_fluid_w`` and ``shaft_to_fluid_w`` enter the volume;
positive ``storage_rate_w`` increases stored energy. Steady volumes have
zero storage rate. Specific enthalpies at all ports MUST use the SAME
reference zero (e.g., a verified NASA7/CoolProp bridging reference).

Mass:   sum(inlet mdot) - sum(outlet mdot) = 0
Energy: sum(inlet mdot*h) - sum(outlet mdot*h)
        + Q + Wshaft - dU/dt = 0

The audit does not treat a small mass/energy residual as validation of
nuclear power deposition, physical geometry, EOS, or engine-cycle fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

LBM_TO_KG = 0.45359237
PSI_TO_PA = 6894.757293168
BTU_TO_J = 1055.05585262
HP_TO_W = 745.6998715822702


@dataclass(frozen=True)
class HydrogenPort:
    name: str
    mass_flow_kg_s: float
    specific_enthalpy_j_kg: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("port name is required")
        if not math.isfinite(self.mass_flow_kg_s) or self.mass_flow_kg_s < 0:
            raise ValueError("port mass-flow magnitude must be finite and nonnegative")
        if not math.isfinite(self.specific_enthalpy_j_kg):
            raise ValueError("port enthalpy must be finite")


@dataclass(frozen=True)
class BoundaryAudit:
    inlet_mass_flow_kg_s: float
    outlet_mass_flow_kg_s: float
    mass_residual_kg_s: float
    energy_residual_w: float
    reference_energy_scale_w: float
    mass_ok: bool
    energy_ok: bool

    @property
    def passed(self) -> bool:
        return self.mass_ok and self.energy_ok


def _finite(value: float, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def audit_control_volume(
    inlets: Iterable[HydrogenPort],
    outlets: Iterable[HydrogenPort],
    *,
    heat_to_fluid_w: float = 0.0,
    shaft_to_fluid_w: float = 0.0,
    storage_rate_w: float = 0.0,
    mass_atol_kg_s: float = 1.0e-9,
    mass_rtol: float = 1.0e-9,
    power_atol_w: float = 1.0e-3,
    power_rtol: float = 1.0e-9,
) -> BoundaryAudit:
    """Verify a supplied control volume; never fit or alter its inputs.

    Relative energy normalization uses the net *thermal span* of the
    enthalpy set, not absolute enthalpy values, so a common reference shift
    does not conceal a nonzero energy residual. Explicit tolerances and
    reference-coherent enthalpies remain caller responsibilities.
    """
    inlet = tuple(inlets)
    outlet = tuple(outlets)
    if not inlet or not outlet:
        raise ValueError("at least one inlet and outlet are required")
    if not all(isinstance(s, HydrogenPort) for s in inlet + outlet):
        raise TypeError("inlets and outlets must contain HydrogenPort values")
    q = _finite(heat_to_fluid_w, "heat_to_fluid_w")
    work = _finite(shaft_to_fluid_w, "shaft_to_fluid_w")
    storage = _finite(storage_rate_w, "storage_rate_w")
    tolerances = (mass_atol_kg_s, mass_rtol, power_atol_w, power_rtol)
    if any(not math.isfinite(float(t)) or float(t) < 0 for t in tolerances):
        raise ValueError("all tolerances must be finite and nonnegative")
    mi = sum(s.mass_flow_kg_s for s in inlet)
    mo = sum(s.mass_flow_kg_s for s in outlet)
    mass_residual = mi - mo
    hin = sum(s.mass_flow_kg_s * s.specific_enthalpy_j_kg for s in inlet)
    hout = sum(s.mass_flow_kg_s * s.specific_enthalpy_j_kg for s in outlet)
    energy_residual = hin - hout + q + work - storage
    span = max(s.specific_enthalpy_j_kg for s in inlet + outlet) - min(
        s.specific_enthalpy_j_kg for s in inlet + outlet
    )
    energy_scale = max(
        (mi + mo) * span,
        abs(q) + abs(work) + abs(storage),
        1.0,
    )
    mass_scale = max(mi, mo, 1.0e-12)
    return BoundaryAudit(
        inlet_mass_flow_kg_s=mi,
        outlet_mass_flow_kg_s=mo,
        mass_residual_kg_s=mass_residual,
        energy_residual_w=energy_residual,
        reference_energy_scale_w=energy_scale,
        mass_ok=abs(mass_residual) <= mass_atol_kg_s + mass_rtol * mass_scale,
        energy_ok=abs(energy_residual) <= power_atol_w + power_rtol * energy_scale,
    )


def lbm_s_to_kg_s(value: float) -> float:
    """ELES mass-flow unit conversion."""
    return _finite(value, "lbm/s") * LBM_TO_KG


def psi_to_pa(value: float) -> float:
    """ELES absolute/differential pressure conversion (same scale)."""
    return _finite(value, "psi") * PSI_TO_PA


def rankine_to_kelvin(value: float) -> float:
    """ELES absolute temperature conversion."""
    return _finite(value, "deg R") * (5.0 / 9.0)


def btu_lbm_to_j_kg(value: float) -> float:
    """ELES enthalpy increment conversion; does not reconcile EOS zeros."""
    return _finite(value, "Btu/lbm") * BTU_TO_J / LBM_TO_KG


def horsepower_to_w(value: float) -> float:
    """Mechanical horsepower from ELES pump/turbine modules."""
    return _finite(value, "hp") * HP_TO_W
