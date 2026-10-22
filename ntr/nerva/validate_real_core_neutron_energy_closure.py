"""Numerical adversarial neutron spectral partition identities (fixtures only)."""
from __future__ import annotations

import numpy as np
from .neutron_energy_groups import diagnostic_energy_edges, spectral_fractions
from .real_core_neutron_energy_closure import _equal


def reject(call, term):
    try:
        call()
    except ValueError as exc:
        assert term in str(exc), (term,str(exc))
    else:
        raise AssertionError("invalid neutron spectral conservation data accepted")


def main() -> int:
    bins=diagnostic_energy_edges(8)
    assert bins.size == 9
    assert np.any(bins == .625)
    assert np.any(bins == 1e5)
    groups=np.column_stack((bins[:-1],bins[1:]))
    intensity=np.arange(1.,9.)
    frac=spectral_fractions(groups,intensity)
    assert abs(sum(frac.values())-1)<1e-14
    assert 0<frac["thermal_below_0p625_eV"]<1
    assert 0<frac["fast_above_100keV"]<1
    # Summing independently scored incident-energy neutron tracks is an
    # exact partition of material-integrated paths, not a fitted spectrum.
    mat=np.array([[1.,2.,3.],[2.,4.,6.]],dtype=float)
    en=np.repeat(mat[:,None,:]/8.,8,axis=1)
    _equal(mat,en.sum(axis=1),"material energy")
    altered=en.copy()
    altered[0,0,1]+=.1
    reject(lambda:_equal(mat,altered.sum(axis=1),"material energy"),
           "material energy")
    altered=mat.copy()
    altered[0,0]=np.nan
    reject(lambda:_equal(mat,altered,"finite test"),"finite test")
    crossed=groups.copy()
    crossed[2,1]=.626
    crossed[3,0]=.626
    reject(lambda:spectral_fractions(crossed,intensity),"crosses")
    print("Neutron energy filter boundaries, material partition and truncation rejection: PASS")
    print("All energy/reaction fixtures SYNTHETIC; actual C++ OpenMC required separately")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
