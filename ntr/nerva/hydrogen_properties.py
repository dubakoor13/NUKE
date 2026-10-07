"""Hydrogen property backends for NERVA thermal-hydraulic solvers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HydrogenState:
    """Local hydrogen thermophysical properties."""

    density_kg_m3: float
    cp_j_kg_k: float
    dynamic_viscosity_pa_s: float
    thermal_conductivity_w_m_k: float
    prandtl: float


class CoolPropHydrogenModel:
    """Real-fluid hydrogen properties from CoolProp HEOS.

    CoolProp is an optional dependency. Import occurs only when this backend is
    instantiated, so the base NERVA model remains usable without CoolProp.
    """

    def __init__(self, fluid: str = "Hydrogen") -> None:
        try:
            from CoolProp.CoolProp import PropsSI
        except ImportError as exc:
            raise RuntimeError(
                "CoolProp is required for the real-fluid hydrogen backend. "
                "Install it with: python -m pip install CoolProp"
            ) from exc

        self._props_si = PropsSI
        self.fluid = fluid

    def state(self, temperature_k: float, pressure_pa: float) -> HydrogenState:
        if temperature_k <= 0.0 or pressure_pa <= 0.0:
            raise ValueError("hydrogen temperature and pressure must be positive")

        props = self._props_si
        fluid = self.fluid

        density = float(
            props("Dmass", "T", temperature_k, "P", pressure_pa, fluid)
        )
        cp = float(
            props("Cpmass", "T", temperature_k, "P", pressure_pa, fluid)
        )
        viscosity = float(
            props("VISCOSITY", "T", temperature_k, "P", pressure_pa, fluid)
        )
        conductivity = float(
            props("CONDUCTIVITY", "T", temperature_k, "P", pressure_pa, fluid)
        )

        if density <= 0.0 or cp <= 0.0 or viscosity <= 0.0 or conductivity <= 0.0:
            raise ValueError("CoolProp returned a non-positive hydrogen property")

        prandtl = cp * viscosity / conductivity
        return HydrogenState(
            density_kg_m3=density,
            cp_j_kg_k=cp,
            dynamic_viscosity_pa_s=viscosity,
            thermal_conductivity_w_m_k=conductivity,
            prandtl=prandtl,
        )
