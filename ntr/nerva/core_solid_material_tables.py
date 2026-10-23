"""Tabulated reactor-core solid thermal properties with exact stored energy.

Use measured/supplied fuel and ZrC temperature-dependent conductivity and
specific heat without fabricating historical material data. Density is an
explicit bulk input. The table is a piecewise-linear k(T), cp(T) law and
enthalpy is the EXACT integral of the interpolated cp(T), not cp(T)*T.

This is a drop-in material-law interface for core_hex_conforming_fem.py,
which uses the methods: validate, conductivity, secant_cp, enthalpy_change
and the density_kg_m3 attribute. NO temperature extrapolation is allowed.
Positive conductivity and cp are enforced at all table nodes.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np


@dataclass(frozen=True)
class TabulatedCoreSolidLaw:
    """SI: T [K], conductivity [W/(m K)], cp [J/(kg K)], density [kg/m3]."""
    density_kg_m3: float
    temperature_k: np.ndarray
    conductivity_w_m_k: np.ndarray
    cp_j_kg_k: np.ndarray

    def validate(self) -> None:
        if not math.isfinite(self.density_kg_m3) or self.density_kg_m3<=0:
            raise ValueError("solid density must be positive and finite")
        t=np.asarray(self.temperature_k,dtype=float)
        k=np.asarray(self.conductivity_w_m_k,dtype=float)
        cp=np.asarray(self.cp_j_kg_k,dtype=float)
        if (t.ndim!=1 or t.size<2
                or k.shape!=t.shape or cp.shape!=t.shape
                or not all(np.isfinite(a).all() for a in (t,k,cp))
                or np.any(t<=0) or np.any(np.diff(t)<=0)
                or np.any(k<=0) or np.any(cp<=0)):
            raise ValueError(
                "tabulated k/cp require strictly increasing positive K, "
                "finite positive conductivity and heat capacity")

    def _require_temperature(self, temperature_k:float)->float:
        t=float(temperature_k)
        x=np.asarray(self.temperature_k,dtype=float)
        if not math.isfinite(t) or t<x[0] or t>x[-1]:
            raise ValueError(
                f"material {t} K outside tabulated range "
                f"[{x[0]}, {x[-1]}] K; extrapolation forbidden")
        return t

    def conductivity(self, temperature_k:float)->float:
        t=self._require_temperature(temperature_k)
        result=float(np.interp(t,self.temperature_k,self.conductivity_w_m_k))
        if not math.isfinite(result) or result<=0:
            raise ValueError("nonphysical tabulated conductivity")
        return result

    def cp(self,temperature_k:float)->float:
        t=self._require_temperature(temperature_k)
        result=float(np.interp(t,self.temperature_k,self.cp_j_kg_k))
        if not math.isfinite(result) or result<=0:
            raise ValueError("nonphysical tabulated specific heat")
        return result

    def internal_energy_j_kg(self,temperature_k:float)->float:
        """Integral from first tabulated temperature to T, exact P1 cp."""
        t=self._require_temperature(temperature_k)
        x=np.asarray(self.temperature_k,dtype=float)
        cp=np.asarray(self.cp_j_kg_k,dtype=float)
        if t==x[0]:
            return 0.
        index=min(int(np.searchsorted(x,t,side="right")-1),len(x)-2)
        intervals=np.diff(x[:index+1])
        full=float(np.sum(
            0.5*(cp[:index]+cp[1:index+1])*intervals,
            dtype=np.float64))
        d=t-x[index]
        cp_t=self.cp(t)
        return full+0.5*(cp[index]+cp_t)*d

    def enthalpy_change(self,a:float,b:float)->float:
        """Signed integral cp(T)dT [J/kg] between arbitrary in-range states."""
        return self.internal_energy_j_kg(b)-self.internal_energy_j_kg(a)

    def secant_cp(self,a:float,b:float)->float:
        """Exact effective heat capacity between previous and current T."""
        a=self._require_temperature(a)
        b=self._require_temperature(b)
        if a==b:
            return self.cp(a)
        cp_eff=self.enthalpy_change(a,b)/(b-a)
        if not math.isfinite(cp_eff) or cp_eff<=0:
            raise ValueError("nonphysical tabulated secant heat capacity")
        return cp_eff

    @classmethod
    def from_npz(cls,path:Path|str,*,density_kg_m3:float
                 )->"TabulatedCoreSolidLaw":
        """NPZ with temperature_k, conductivity_w_m_k, cp_j_kg_k."""
        with np.load(path,allow_pickle=False) as data:
            expected=("temperature_k","conductivity_w_m_k","cp_j_kg_k")
            if not set(expected).issubset(data.files):
                raise KeyError(
                    "material NPZ must contain temperature_k, "
                    "conductivity_w_m_k and cp_j_kg_k")
            law=cls(
                density_kg_m3,
                np.asarray(data["temperature_k"],dtype=float).copy(),
                np.asarray(data["conductivity_w_m_k"],dtype=float).copy(),
                np.asarray(data["cp_j_kg_k"],dtype=float).copy())
        law.validate()
        return law
