"""Numerical adversarial source-energy versus nuclear-grid QA.

Only fabricated one-dimensional energies here. The genuine four-seed
eigenvalue HDF5 and authentic evaluated isotope tables are audited
separately in physical OpenMC C++ CI; no fake reactor physics conclusions.
"""
from __future__ import annotations

import math
import numpy as np

from .real_fission_source_neutron_energy_support import (
    evaluate_source_energy_domain,
)


def reject(action, match: str)->None:
    try:
        action()
    except (ValueError,TypeError) as err:
        assert match in str(err),(match,str(err))
    else:
        raise AssertionError("invalid actual neutron energy support accepted")


def main()->int:
    e=np.array([1e-4,1e-3,3e-2,1e6,1e7,3e7])
    w=np.array([2.,1.,3.,4.,5.,6.])
    output=evaluate_source_energy_domain(e,w,1e-3,1e7)
    assert output["source_neutron_sites"]==6
    assert output["below_grid_neutron_sites"]==1
    assert output["inside_grid_neutron_sites"]==4
    assert output["above_grid_neutron_sites"]==1
    assert output["below_grid_source_weight"]==2.
    assert output["inside_grid_source_weight"]==13.
    assert output["above_grid_source_weight"]==6.
    assert math.isclose(sum(output[k] for k in (
        "below_grid_weight_fraction","inside_grid_weight_fraction",
        "above_grid_weight_fraction")),1.,rel_tol=0,abs_tol=1e-15)
    assert not output["source_energies_within_common_evaluated_grid"]
    assert output["weighted_energy_partition_conserved"]
    # Exactly inclusive evaluated energy boundaries, NOT below/above.
    boundary=evaluate_source_energy_domain([.001,1e7],[1.,2.],.001,1e7)
    assert boundary["source_energies_within_common_evaluated_grid"]
    assert boundary["inside_grid_neutron_sites"]==2
    assert boundary["inside_grid_weight_fraction"]==1.
    # Values on either side of the support stay visible, not clipped.
    reject(lambda:evaluate_source_energy_domain([0.,1.],[1.,1.],0.,2.),
           "nonphysical evaluated")
    reject(lambda:evaluate_source_energy_domain([1.,2.],[1.,0.],.5,3.),
           "nonphysical evaluated")
    reject(lambda:evaluate_source_energy_domain([1.,2.],[1.,np.nan],.5,3.),
           "nonphysical evaluated")
    reject(lambda:evaluate_source_energy_domain([1.,np.inf],[1.,1.],.5,3.),
           "nonphysical evaluated")
    reject(lambda:evaluate_source_energy_domain([1.,2.],[1.],.5,3.),
           "nonphysical evaluated")
    reject(lambda:evaluate_source_energy_domain([1.,2.],[1.,1.],3.,1.),
           "nonphysical evaluated")
    reject(lambda:evaluate_source_energy_domain([1.,2.],[1.,1.],-1.,3.),
           "nonphysical evaluated")
    reject(lambda:evaluate_source_energy_domain([1.,2.],[1.,1.],1.,float("inf")),
           "nonphysical evaluated")
    print("Evaluated source-neutron support numerical regressions: PASS")
    print("  inside/below/above nuclear-data weights conserve actual bank totals")
    print("  actual neutron HDF5 and evaluated nuclear data are checked in CI")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
