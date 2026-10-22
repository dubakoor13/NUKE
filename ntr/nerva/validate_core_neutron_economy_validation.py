"""Numerical/adversarial checks for genuine-source neutron economy identities.

These fixtures are SYNTHETIC. Actual physical evidence is required from
four independent OpenMC C++ real statepoint transport executions.
"""
from __future__ import annotations

from copy import deepcopy

from .core_neutron_economy_validation import analyze_neutron_economy_window


def reject(fn, snippet):
    try:
        fn()
    except (ValueError,KeyError) as exc:
        assert snippet in str(exc),(snippet,str(exc))
    else:
        raise AssertionError("invalid neutron physics accepted: "+snippet)


def main() -> int:
    ids=list(range(19,28))
    active={19,20,21,22}
    analog={
        "first_active_batch":31,"last_active_batch":40,
        "active_realizations":10,
        "statepoint_sha256":"a"*64,
        "analog_absorption_per_source":.04,
        "native_vacuum_leakage_per_source":1.02,
        "net_multiplicative_scattering_per_source":.06,
        "analog_nu_fission_per_source":.0117,
        "conservation_balance_per_source":1.,
        "neutron_balance_strict_physical_screen_passed":True,
    }
    track={
        "first_active_batch":31,"last_active_batch":40,
        "number_of_active_generations":10,
        "statepoint_sha256":"a"*64,
        "global_reaction_rates_per_source":{
            "absorption":.04,"nu-fission":.0117
        },
        "material_reaction_rates_per_source":{
            "absorption":[.012,0.,.001,.027,0.,0.,0.,0.,0.],
            "nu-fission":[.0117,0.,0.,0.,0.,0.,0.,0.,0.],
        },
    }
    result=analyze_neutron_economy_window(analog,track,ids,active)
    assert result["native_neutron_balance_abs_residual"]<1e-14
    assert result["leakage_is_over_one_due_to_neutron_multiplication"]
    assert result["leakage_fraction_of_total_neutron_terminations"]>.9
    assert abs(sum(result["absorption_by_reachable_material"].values())-.04)<1e-12
    assert not (not result["fractions_are_not_independent_neutron_probabilities"])
    assert result["fuel_fission_source_yield_per_starting_neutron"]==.0117
    assert result["native_vacuum_leakage_per_starting_neutron"]==1.02

    invalid=deepcopy(analog)
    invalid["native_vacuum_leakage_per_source"]=.95
    reject(lambda:analyze_neutron_economy_window(invalid,track,ids,active),
           "leakage/absorption balance violated")
    invalid=deepcopy(analog)
    invalid["net_multiplicative_scattering_per_source"]=-.06
    reject(lambda:analyze_neutron_economy_window(invalid,track,ids,active),
           "invalid actual neutron multiplicative scattering")
    invalid=deepcopy(track)
    invalid["material_reaction_rates_per_source"]["absorption"][4]=.001
    reject(lambda:analyze_neutron_economy_window(analog,invalid,ids,active),
           "not conserved by physical material tally")
    invalid=deepcopy(track)
    invalid["material_reaction_rates_per_source"]["absorption"][3]-=.001
    invalid["material_reaction_rates_per_source"]["absorption"][4]=.001
    reject(lambda:analyze_neutron_economy_window(analog,invalid,ids,active),
           "in unreachable nuclear material")
    invalid=deepcopy(track)
    invalid["material_reaction_rates_per_source"]["nu-fission"][0]-=.002
    invalid["material_reaction_rates_per_source"]["nu-fission"][1]=.002
    reject(lambda:analyze_neutron_economy_window(analog,invalid,ids,active),
           "outside physical fuel")
    invalid=deepcopy(track)
    invalid["statepoint_sha256"]="0"*64
    reject(lambda:analyze_neutron_economy_window(analog,invalid,ids,active),
           "window identity mismatch")
    invalid=deepcopy(track)
    invalid["global_reaction_rates_per_source"]["nu-fission"]=0.1
    reject(lambda:analyze_neutron_economy_window(analog,invalid,ids,active),
           "not conserved")
    invalid=deepcopy(analog)
    invalid["conservation_balance_per_source"]=1.1
    reject(lambda:analyze_neutron_economy_window(invalid,track,ids,active),
           "independent native")
    invalid=deepcopy(track)
    invalid["global_reaction_rates_per_source"]["absorption"]=float("nan")
    reject(lambda:analyze_neutron_economy_window(analog,invalid,ids,active),
           "invalid actual neutron tracklength absorption")
    invalid=deepcopy(analog)
    invalid["active_realizations"]=9
    reject(lambda:analyze_neutron_economy_window(invalid,track,ids,active),
           "window identity mismatch")
    print("Measured native neutron fate, leakage/scatter closure and banked fission: PASS")
    print("  Negative synthetic tests only; real C++ OpenMC source validation separate")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
