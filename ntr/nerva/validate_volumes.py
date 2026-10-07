"""Validate construction of NERVA stochastic volume calculations."""

from __future__ import annotations

import openmc

from .config import NervaConfig
from .model import build_model
from .calculate_volumes import _INSTANCE_CELL_NAMES, _bounding_box


def main() -> int:
    config = NervaConfig(
        core_rings=3,
        particles=100,
        batches=12,
        inactive=4,
    )
    model = build_model(config, assembly="reactor")
    lower_left, upper_right = _bounding_box(config, "reactor")

    materials = list(model.materials)
    cells = [
        cell
        for cell in model.geometry.get_all_cells().values()
        if cell.name in _INSTANCE_CELL_NAMES
    ]

    material_calc = openmc.VolumeCalculation(
        materials,
        1000,
        lower_left=lower_left,
        upper_right=upper_right,
    )
    cell_calc = openmc.VolumeCalculation(
        cells,
        1000,
        lower_left=lower_left,
        upper_right=upper_right,
    )

    assert material_calc.domain_type == "material"
    assert cell_calc.domain_type == "cell"
    assert len(material_calc.ids) == len(materials)
    assert len(cell_calc.ids) == len(cells)
    assert len(cells) >= 20

    print("NERVA stochastic-volume setup validation: PASS")
    print(f"  materials: {len(materials)}")
    print(f"  repeated cell definitions: {len(cells)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
