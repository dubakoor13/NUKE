"""Conservation-law regression independent of any historical reactor design.

Analytic neutron histories on an explicitly normalized, single transported
source-neutron basis. A real OpenMC HDF5 integration run exercises the
actual neutron+leakage tally estimators independently of this math test.
"""
from __future__ import annotations

from math import isclose
from .core_neutron_balance import _close_balance


def reject(call,substring):
    try:
        call()
    except ValueError as e:
        assert substring in str(e),str(e)
    else:
        raise AssertionError("physically invalid neutron accounting accepted: "+substring)


def main()->int:
    # A generation with 0.37 captures+fissions, 0.65 vacuum losses and
    # 0.02 extra live neutrons due to (n,xn) neutron multiplication.
    base={
        "absorption":(0.37,0.003),
        "scatter":(1.20,0.006),
        "nu-scatter":(1.22,0.006),
        "nu-fission":(0.014,0.001),
    }
    result=_close_balance(base,(0.65,0.003),(0.014,0.001))
    assert isclose(result["balance_disappearance_per_source"],1.,abs_tol=1e-12)
    assert result["neutron_balance_screen_passed"]
    assert result["nu_fission_vs_native_tracklength_k"]["screen_passed"]
    assert isclose(result["multiplicative_scattering_net_neutrons_per_source"],.02,abs_tol=1e-12)
    assert result["conservative_residual_standard_error_upper_bound"] == (
        .003+.003+.006+.006
    )
    assert result["is_core_physically_validated_nasa_nerva"] is False
    assert result["fission_source_stationarity_proven"] is False

    # Erroneously omitting scattered-neutron production biases conservation.
    fail=_close_balance({**base,"nu-scatter":(1.20,.001)},(.65,.001),(.014,.001),
                        abs_tolerance=.01,sigma_multiplier=1.)
    assert not fail["neutron_balance_screen_passed"]

    # Histories can conserve neutrons without reproducing νfission estimator.
    bad_fission=_close_balance(
        {**base,"nu-fission":(.35,.00001)},(.65,.00001),(.014,.00001),
        abs_tolerance=.01,sigma_multiplier=2.
    )
    assert bad_fission["neutron_balance_screen_passed"]
    assert not bad_fission["nu_fission_vs_native_tracklength_k"]["screen_passed"]

    # A high-variance sample must not be advertised as a physical solution:
    # screen may be consistent but uncertainty is separately reported.
    noisy=_close_balance(
        {**base,"absorption":(.42,.12)},(.65,.11),(.014,.003)
    )
    assert noisy["neutron_balance_screen_passed"]
    assert noisy["conservative_residual_standard_error_upper_bound"] > .2

    reject(lambda:_close_balance({**base,"nu-scatter":(-1.,0)},(.65,.003),(.014,.001)),
           "nonfinite/negative")
    reject(lambda:_close_balance({**base,"absorption":(float("nan"),0)},(.65,.003),(.014,.001)),
           "nonfinite")
    reject(lambda:_close_balance(base,(-.5,0),(.014,.001)),"leakage")
    reject(lambda:_close_balance(base,(float("inf"),0),(.014,.001)),"leakage")
    reject(lambda:_close_balance(base,(.65,0),(.014,.001),abs_tolerance=0),"tolerance")
    reject(lambda:_close_balance(base,(.65,0),(.014,.001),sigma_multiplier=0),"sigma")
    print("Whole-core eigenvalue-generation neutron accounting mathematics: PASS")
    print("  physical real OpenMC statepoint tested here: NO (integration workflow required)")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
