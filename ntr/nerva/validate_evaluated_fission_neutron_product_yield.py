"""Adversarial evaluated fission product yield numerical regressions.

Synthetic arrays ONLY. Never a substituted fission source evaluation.
"""
from __future__ import annotations

import numpy as np
import openmc.data
from types import SimpleNamespace

from .evaluated_fission_neutron_product_yield import (
    MAX_EVALUATED_YIELD_RELATIVE_MISMATCH,
    assess_evaluated_neutron_product_yields,
    _sample_fission_reaction,
)


def reject(fn, text: str) -> None:
    try:
        fn()
    except (ValueError,TypeError) as exc:
        assert text in str(exc), (text,str(exc))
    else:
        raise AssertionError("nonphysical fission product neutron yield accepted")


def main()->int:
    e=np.array([1e-5,1e-3,1.,1e6])
    prompt=np.array([2.35,2.36,2.38,2.45])
    delayed_a=np.array([.010,.010,.012,.012])
    delayed_b=np.array([.008,.009,.009,.012])
    total=prompt+delayed_a+delayed_b
    good=assess_evaluated_neutron_product_yields(
        e,{"total":[total],"prompt":[prompt],
           "delayed":[delayed_a,delayed_b]}
    )
    assert good["all_total_prompt_delayed_yield_modes_available"]
    assert good["native_neutron_product_group_counts"]["delayed"]==2
    assert good["maximum_relative_total_minus_prompt_delayed_yield_mismatch"]<1e-14
    assert good["evaluated_yield_consistency_passed_if_all_modes_present"]
    assert not good["beta_effective_calculated"]
    assert good["evaluated_incident_energy_samples"]==4

    # Genuine OpenMC HDF5 formats attach an independent total-nu record
    # to Reaction.derived_products. A validator that only checks ordinary
    # Reaction.products misses the nuclear evaluation's true total yield.
    def neutron_product(mode, vals):
        product=openmc.data.Product("neutron")
        product.emission_mode=mode
        product.yield_=openmc.data.Tabulated1D(e,vals)
        return product

    native_fission=SimpleNamespace(
        xs={"294K":SimpleNamespace(x=e,y=np.ones(len(e)))},
        products=[
            neutron_product("prompt",prompt),
            neutron_product("delayed",delayed_a),
            neutron_product("delayed",delayed_b),
        ],
        derived_products=[neutron_product("total",total)],
        redundant=False,
    )
    native_fake=SimpleNamespace(reactions={18:native_fission})
    native_row=_sample_fission_reaction(native_fake,"synthetic isotope","294K")
    assert native_row["all_total_prompt_delayed_yield_modes_available"]
    assert native_row["derived_total_is_independent_evaluated_record_not_a_sum"]
    assert native_row["original_evaluated_product_modes"]["total"]==0
    assert native_row["native_derived_evaluated_product_modes"]["total"]==1
    assert native_row["maximum_relative_total_minus_prompt_delayed_yield_mismatch"]<1e-13
    native_fission.derived_products=[]
    uncomparable=_sample_fission_reaction(native_fake,"synthetic isotope","294K")
    assert not uncomparable["all_total_prompt_delayed_yield_modes_available"]
    assert uncomparable["missing_modes"]==["total"]

    tiny=assess_evaluated_neutron_product_yields(
        e,{"total":[total*(1+4e-6)],"prompt":[prompt],
           "delayed":[delayed_a,delayed_b]}
    )
    assert 1e-6<tiny["maximum_relative_total_minus_prompt_delayed_yield_mismatch"]<1e-5
    assert MAX_EVALUATED_YIELD_RELATIVE_MISMATCH == 1e-3

    absent=assess_evaluated_neutron_product_yields(
        e,{"prompt":[prompt],"delayed":[delayed_a,delayed_b]}
    )
    assert absent["missing_modes"]==["total"]
    assert absent["maximum_relative_total_minus_prompt_delayed_yield_mismatch"] is None
    assert not absent["evaluated_yield_consistency_passed_if_all_modes_present"]

    reject(lambda:assess_evaluated_neutron_product_yields(
        e,{"total":[total*1.03],"prompt":[prompt],
           "delayed":[delayed_a,delayed_b]}),"differ by more")
    reject(lambda:assess_evaluated_neutron_product_yields(
        e,{"prompt":[np.array([2.,2.,-1.,2.])]}),"negative evaluated")
    reject(lambda:assess_evaluated_neutron_product_yields(
        e,{"prompt":[np.array([2.,2.,np.nan,2.])]}),"nonfinite or misaligned")
    reject(lambda:assess_evaluated_neutron_product_yields(
        e,{"prompt":[np.array([2.,2.,2.])]}),"nonfinite or misaligned")
    reject(lambda:assess_evaluated_neutron_product_yields(
        [1e-5,1e-5,1.],{"total":[]}),"invalid evaluated")
    reject(lambda:assess_evaluated_neutron_product_yields(
        [1e-5,1.,float("inf")],{"total":[]}),"invalid evaluated")
    reject(lambda:assess_evaluated_neutron_product_yields(
        e,{"unsupported":[total]}),"invalid evaluated")
    reject(lambda:assess_evaluated_neutron_product_yields(
        e,{"delayed":tuple([delayed_a])}),"mode group must be a list")

    print("Fission product yield numerical QA: PASS (synthetic arrays)")
    print("  missing evaluated yield group cannot be called validated;")
    print("  fission yield closure is a nuclear-data screen, NOT beta_eff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
