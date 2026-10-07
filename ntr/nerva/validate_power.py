"""Fast unit validation for NERVA tally power normalization."""

from __future__ import annotations

import numpy as np

from .power import EV_TO_J, normalize_regular_mesh


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

    print("NERVA tally normalization validation: PASS")
    print(f"  requested/recovered power: {requested_power:.6e} W")
    print(f"  source rate: {fields.source_rate_s:.6e} source/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
