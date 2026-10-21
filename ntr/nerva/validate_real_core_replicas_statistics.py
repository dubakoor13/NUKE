"""Pure aggregation tests (synthetic k values, NEVER transport evidence)."""
from __future__ import annotations
import math
from .run_real_core_replicas import _aggregate

def reject(fn, expected):
    try:
        fn()
    except ValueError as error:
        assert expected in str(error), str(error)
    else:
        raise AssertionError("expected invalid replica rejection: "+expected)

def main():
    values=[
        {"seed":13,"keff_mean":0.991,"keff_std_dev":0.003},
        {"seed":17,"keff_mean":1.001,"keff_std_dev":0.003},
        {"seed":23,"keff_mean":0.996,"keff_std_dev":0.003},
        {"seed":29,"keff_mean":0.998,"keff_std_dev":0.003},
    ]
    summary=_aggregate(values)
    assert summary["independent_seed_count"]==4
    assert math.isclose(summary["formal_mean_keff_standard_error"],.0015)
    assert math.isclose(summary["inverse_variance_mean_keff"],.9965)
    assert summary["between_seed_chi_square"]>0
    assert summary["between_seed_degrees_of_freedom"]==3
    assert summary["source_convergence_independently_demonstrated"] is False
    assert summary["historical_nasa_nerva_benchmark_validated"] is False
    reject(lambda:_aggregate(values[:1]),">=2")
    duplicate=[dict(row) for row in values]
    duplicate[1]["seed"]=duplicate[0]["seed"]
    reject(lambda:_aggregate(duplicate),"duplicated")
    bad=[dict(row) for row in values]
    bad[1]["keff_std_dev"]=float("nan")
    reject(lambda:_aggregate(bad),"statistical")
    bad=[dict(row) for row in values]
    bad[1]["keff_mean"]=float("inf")
    reject(lambda:_aggregate(bad),"nonfinite")
    print("NERVA actual OpenMC independent-seed aggregation mathematics: PASS")
    print("  Monte Carlo transport executed by these unit tests: NO")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
