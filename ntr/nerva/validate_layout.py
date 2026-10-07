"""Fast geometry-only validation with no OpenMC runtime dependency."""

from __future__ import annotations

from .config import NervaConfig
from .layout import coolant_channel_positions, minimum_channel_ligament_cm, outer_channel_margin_cm


def main() -> int:
    config = NervaConfig()
    config.validate()

    channels = coolant_channel_positions(config)
    assert len(channels) == 19, f"expected 19 channels, got {len(channels)}"

    ligament = minimum_channel_ligament_cm(config)
    margin = outer_channel_margin_cm(config)

    assert ligament > 0.0, f"channel coatings overlap: ligament={ligament:.6f} cm"
    assert margin > 0.0, f"outer channel crosses inner fuel hex: margin={margin:.6f} cm"

    print("NERVA fuel-element layout validation: PASS")
    print(f"  channel count: {len(channels)}")
    print(f"  bore diameter: {config.coolant_bore_diameter_cm:.5f} cm")
    print(f"  bore pitch: {config.coolant_bore_pitch_cm:.5f} cm")
    print(f"  minimum coated-channel ligament: {ligament:.5f} cm")
    print(f"  minimum channel-to-flat margin: {margin:.5f} cm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
