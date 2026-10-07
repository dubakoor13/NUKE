"""Validate the separable element/channel power reconstruction math."""

from __future__ import annotations

import numpy as np

from .solve_element_channels import (
    _normalized_shape,
    _rebin_shape_to_edges,
)


def main() -> int:
    shape = _normalized_shape(np.array([1.0, 2.0, 3.0]))
    assert np.isclose(np.sum(shape), 1.0)
    assert np.all(shape >= 0.0)

    zeros = _normalized_shape(np.zeros(3))
    assert np.all(zeros == 0.0)

    fuel_fraction = _normalized_shape(
        np.array([1.0, 2.0, 1.0])
    )
    axial_shape = _normalized_shape(
        np.array([1.0, 3.0])
    )
    total_power = 400.0

    element_axial = (
        fuel_fraction[:, None]
        * total_power
        * axial_shape[None, :]
    )
    channel_wall = element_axial[:, None, :] / 19.0

    assert np.isclose(np.sum(element_axial), total_power)
    assert np.isclose(np.sum(channel_wall) * 19.0, total_power)

    rebinned = _rebin_shape_to_edges(
        np.array([10.0, 20.0]),
        np.array([0.0, 1.0, 2.0]),
        np.array([0.0, 0.5, 1.5, 2.0]),
    )
    assert rebinned.shape == (3,)
    assert np.isclose(np.sum(rebinned), 30.0)
    assert np.all(rebinned >= 0.0)

    print("NERVA element/channel reconstruction validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
