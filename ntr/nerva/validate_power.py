"""Fast unit validation for NERVA tally power normalization."""

from __future__ import annotations

import numpy as np

from .power import EV_TO_J, heating_score_for_transport, normalize_regular_mesh


def main() -> int:
    heating = np.array([1.0, 2.0, 3.0, 4.0]) * 1.0e6
    fission = np.array([1.0, 1.0, 2.0, 2.0])
    flux = np.array([2.0, 4.0, 6.0, 8.0])
    requested_power = 10.0e6
    cell_volume = 5.0

    fields = normalize_regular_mesh(
        heating,
        fission,
        flux,
        cell_volume_cm3=cell_volume,
        reactor_power_w=requested_power,
    )

    assert np.isclose(fields.total_power_w, requested_power, rtol=1.0e-12)
    expected_source_rate = requested_power / (np.sum(heating) * EV_TO_J)
    assert np.isclose(fields.source_rate_s, expected_source_rate, rtol=1.0e-12)
    assert np.all(fields.power_density_w_cm3 >= 0.0)
    assert np.all(fields.fission_rate_cm3_s >= 0.0)
    assert np.all(fields.flux_cm2_s >= 0.0)

    recovered_power = float(np.sum(fields.power_density_w_cm3) * cell_volume)
    assert np.isclose(recovered_power, requested_power, rtol=1.0e-12)

    # Neutron-only locally deposited gamma energy is NOT the physically
    # transported gamma heating; normalization and spatial profiles differ.
    assert heating_score_for_transport(False) == "heating-local"
    assert heating_score_for_transport(True) == "heating"
    local = np.array([4.0e6, 1.0e6])
    transported = np.array([1.0e6, 2.5e6])
    for coupled in (False, True):
        selected = transported if coupled else local
        other = local if coupled else transported
        field = normalize_regular_mesh(
            heating_ev_per_source=selected,
            fission_per_source=np.ones(2),
            flux_tracklength_per_source=np.ones(2),
            cell_volume_cm3=2.,
            reactor_power_w=3e6,
        )
        assert np.isclose(field.source_rate_s, 3e6/(np.sum(selected)*EV_TO_J))
        assert np.isclose(np.sum(field.power_density_w_cm3)*2, 3e6)
        fraction = field.power_density_w_cm3[0]*2/3e6
        assert np.isclose(fraction,selected[0]/np.sum(selected))
        assert not np.isclose(fraction, other[0]/np.sum(other))

    print("NERVA tally normalization validation: PASS")
    print(f"  requested/recovered power: {requested_power:.6e} W")
    print(f"  source rate: {fields.source_rate_s:.6e} source/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
