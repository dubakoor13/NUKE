"""Fast geometry-only validation with no OpenMC runtime dependency."""

from __future__ import annotations

from .config import NervaConfig
from .geometry import cluster_summary
from .layout import (
    coolant_channel_positions,
    minimum_channel_ligament_cm,
    outer_channel_margin_cm,
)


def main() -> int:
    config = NervaConfig()
    config.validate()

    channels = coolant_channel_positions(config)
    assert len(channels) == 19, f"expected 19 channels, got {len(channels)}"

    ligament = minimum_channel_ligament_cm(config)
    margin = outer_channel_margin_cm(config)

    assert ligament > 0.0, f"channel coatings overlap: ligament={ligament:.6f} cm"
    assert margin > 0.0, f"outer channel crosses inner fuel hex: margin={margin:.6f} cm"

    tie_radii = (
        config.tie_inner_tube_inner_radius_cm,
        config.tie_inner_tube_outer_radius_cm,
        config.tie_zrh_inner_radius_cm,
        config.tie_zrh_outer_radius_cm,
        config.tie_outer_tube_inner_radius_cm,
        config.tie_outer_tube_outer_radius_cm,
        config.tie_zrc_inner_radius_cm,
        config.tie_zrc_outer_radius_cm,
        config.tie_graphite_inner_radius_cm,
    )
    assert all(a < b for a, b in zip(tie_radii, tie_radii[1:]))

    summary = cluster_summary(config)
    assert summary["fuel_elements"] == 6
    assert summary["tie_tubes"] == 1
    assert summary["fuel_coolant_channels"] == 114
    assert summary["tie_tube_hydrogen_passages"] == 2

    print("NERVA fuel-element + tie-tube layout validation: PASS")
    print(f"  fuel-element channel count: {len(channels)}")
    print(f"  bore diameter: {config.coolant_bore_diameter_cm:.5f} cm")
    print(f"  bore pitch: {config.coolant_bore_pitch_cm:.5f} cm")
    print(f"  minimum coated-channel ligament: {ligament:.5f} cm")
    print(f"  minimum channel-to-flat margin: {margin:.5f} cm")
    print("  tie-tube radial stack: ordered")
    print("  cluster: 6 fuel elements + 1 tie tube")
    print("  cluster fuel coolant channels: 114")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
